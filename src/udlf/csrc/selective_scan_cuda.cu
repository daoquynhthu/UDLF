#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#include <vector>

namespace {
constexpr int kState = 16;
constexpr int kChunk = 64;
constexpr int kWarpsPerBlock = 8;

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_grad(float x) {
  const float s = 1.0f / (1.0f + expf(-x));
  return s * (1.0f + x * (1.0f - s));
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void scan_fwd_kernel(
    const float* u, const float* delta, const float* a,
    const float* b, const float* c, const float* d, const float* z,
    const float* delta_bias, float* y, float* checkpoints,
    int batch, int dim, int length, int chunks) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int channel = blockIdx.x * (blockDim.x >> 5) + warp;
  if (channel >= batch * dim) return;
  const int bi = channel / dim;
  const int di = channel - bi * dim;
  float state = 0.0f;
  const int ud_base = (bi * dim + di) * length;
  const int bc_base = bi * kState * length;
  const int cp_base = ((bi * dim + di) * (chunks + 1)) * kState;
  if (lane < kState) checkpoints[cp_base + lane] = 0.0f;
  for (int t = 0; t < length; ++t) {
    const float dt = log1pf(expf(delta[ud_base + t] + delta_bias[di]));
    const float x = u[ud_base + t];
    float contribution = 0.0f;
    if (lane < kState) {
      const int idx = bc_base + lane * length + t;
      const float alpha = expf(dt * a[di * kState + lane]);
      state = alpha * state + x * dt * b[idx];
      contribution = state * c[idx];
    }
    const float scan_out = warp_sum(contribution);
    if (lane == 0) y[ud_base + t] = (scan_out + d[di] * x) * silu(z[ud_base + t]);
    if ((t + 1) % kChunk == 0 || t + 1 == length) {
      const int ci = (t + 1 + kChunk - 1) / kChunk;
      if (lane < kState) checkpoints[cp_base + ci * kState + lane] = state;
    }
  }
}

__global__ void scan_bwd_kernel(
    const float* gy, const float* u, const float* delta, const float* a,
    const float* b, const float* c, const float* d, const float* z,
    const float* delta_bias, const float* checkpoints,
    float* gu, float* gdelta, float* ga, float* gb, float* gc,
    float* gd, float* gz, float* gbias,
    int batch, int dim, int length, int chunks) {
  __shared__ float shared_gb[kWarpsPerBlock][kState];
  __shared__ float shared_gc[kWarpsPerBlock][kState];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int channel = blockIdx.x * (blockDim.x >> 5) + warp;
  if (channel >= batch * dim) return;
  const int bi = channel / dim;
  const int di = channel - bi * dim;
  const int ud_base = channel * length;
  const int bc_base = bi * kState * length;
  const int cp_base = channel * (chunks + 1) * kState;
  float adj = 0.0f;
  float local_d = 0.0f, local_bias = 0.0f;

  for (int chunk = chunks - 1; chunk >= 0; --chunk) {
    const int start = chunk * kChunk;
    const int end = min(length, start + kChunk);
    const int count = end - start;
    float history[kChunk + 1];
    history[0] = lane < kState ? checkpoints[cp_base + chunk * kState + lane] : 0.0f;
    for (int j = 0; j < count; ++j) {
      const int t = start + j;
      const float dt = log1pf(expf(delta[ud_base + t] + delta_bias[di]));
      const float x = u[ud_base + t];
      if (lane < kState) {
        const int idx = bc_base + lane * length + t;
        const float alpha = expf(dt * a[di * kState + lane]);
        history[j + 1] = alpha * history[j] + x * dt * b[idx];
      } else history[j + 1] = 0.0f;
    }
    for (int j = count - 1; j >= 0; --j) {
      const int t = start + j;
      const int ui = ud_base + t;
      const float raw_dt = delta[ui] + delta_bias[di];
      const float dt_sigmoid = 1.0f / (1.0f + expf(-raw_dt));
      const float dt = log1pf(expf(raw_dt));
      const float x = u[ui];
      const float contribution = lane < kState ? history[j + 1] * c[bc_base + lane * length + t] : 0.0f;
      const float base = __shfl_sync(0xffffffff, warp_sum(contribution), 0) + d[di] * x;
      const float gout = gy[ui];
      const float gbase = gout * silu(z[ui]);
      if (lane == 0) {
        gz[ui] = gout * base * silu_grad(z[ui]);
        local_d += gbase * x;
      }
      float gx_part = 0.0f, gdt_part = 0.0f;
      float gb_part = 0.0f, gc_part = 0.0f;
      if (lane < kState) {
        const int idx = bc_base + lane * length + t;
        const float alpha = expf(dt * a[di * kState + lane]);
        const float gstate = adj + gbase * c[idx];
        gc_part = gbase * history[j + 1];
        gb_part = gstate * x * dt;
        atomicAdd(&ga[di * kState + lane], gstate * dt * alpha * history[j]);
        gx_part = gstate * dt * b[idx];
        gdt_part = gstate * (a[di * kState + lane] * alpha * history[j] + x * b[idx]);
        adj = gstate * alpha;
      }
      if (lane < kState) {
        shared_gb[warp][lane] = gb_part;
        shared_gc[warp][lane] = gc_part;
      }
      __syncthreads();
      if (warp == 0 && lane < kState) {
        float gb_sum = 0.0f, gc_sum = 0.0f;
        for (int w = 0; w < kWarpsPerBlock; ++w) {
          gb_sum += shared_gb[w][lane];
          gc_sum += shared_gc[w][lane];
        }
        const int reduced_idx = bc_base + lane * length + t;
        atomicAdd(&gb[reduced_idx], gb_sum);
        atomicAdd(&gc[reduced_idx], gc_sum);
      }
      __syncthreads();
      const float gx = warp_sum(gx_part);
      const float gdt = warp_sum(gdt_part);
      if (lane == 0) {
        gu[ui] = gx + gbase * d[di];
        gdelta[ui] = gdt * dt_sigmoid;
        local_bias += gdelta[ui];
      }
    }
  }
  if (lane == 0) {
    atomicAdd(&gd[di], local_d);
    atomicAdd(&gbias[di], local_bias);
  }
}
}  // namespace

std::vector<at::Tensor> selective_scan_forward_cuda(
    at::Tensor u, at::Tensor delta, at::Tensor a,
    at::Tensor b, at::Tensor c, at::Tensor d,
    at::Tensor z, at::Tensor delta_bias) {
  TORCH_CHECK(u.is_cuda() && u.scalar_type() == at::kFloat, "inputs must be CUDA float32");
  TORCH_CHECK(a.size(1) == kState, "UDLF kernel requires d_state=16");
  const int batch = u.size(0), dim = u.size(1), length = u.size(2);
  TORCH_CHECK(dim % kWarpsPerBlock == 0, "channel dimension must be divisible by 8");
  const int chunks = (length + kChunk - 1) / kChunk;
  auto y = at::empty_like(u);
  auto checkpoints = at::empty({batch, dim, chunks + 1, kState}, u.options());
  const int threads = 256, warps = threads / 32;
  scan_fwd_kernel<<<(batch * dim + warps - 1) / warps, threads>>>(
      u.data_ptr<float>(), delta.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(),
      c.data_ptr<float>(), d.data_ptr<float>(), z.data_ptr<float>(), delta_bias.data_ptr<float>(),
      y.data_ptr<float>(), checkpoints.data_ptr<float>(), batch, dim, length, chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, checkpoints};
}

std::vector<at::Tensor> selective_scan_backward_cuda(
    at::Tensor grad_y, at::Tensor u, at::Tensor delta,
    at::Tensor a, at::Tensor b, at::Tensor c, at::Tensor d,
    at::Tensor z, at::Tensor delta_bias, at::Tensor checkpoints) {
  auto gu = at::zeros_like(u), gdelta = at::zeros_like(delta), ga = at::zeros_like(a);
  auto gb = at::zeros_like(b), gc = at::zeros_like(c), gd = at::zeros_like(d);
  auto gz = at::zeros_like(z), gbias = at::zeros_like(delta_bias);
  const int batch = u.size(0), dim = u.size(1), length = u.size(2);
  TORCH_CHECK(dim % kWarpsPerBlock == 0, "channel dimension must be divisible by 8");
  const int chunks = (length + kChunk - 1) / kChunk;
  const int threads = 256, warps = threads / 32;
  scan_bwd_kernel<<<(batch * dim + warps - 1) / warps, threads>>>(
      grad_y.contiguous().data_ptr<float>(), u.data_ptr<float>(), delta.data_ptr<float>(),
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), d.data_ptr<float>(),
      z.data_ptr<float>(), delta_bias.data_ptr<float>(), checkpoints.data_ptr<float>(),
      gu.data_ptr<float>(), gdelta.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(),
      gc.data_ptr<float>(), gd.data_ptr<float>(), gz.data_ptr<float>(), gbias.data_ptr<float>(),
      batch, dim, length, chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gu, gdelta, ga, gb, gc, gd, gz, gbias};
}
