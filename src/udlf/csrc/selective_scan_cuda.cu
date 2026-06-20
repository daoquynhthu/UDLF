#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#include <vector>

namespace {
constexpr int kState = 16;
constexpr int kChunk = 64;

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_grad(float x) {
  const float s = 1.0f / (1.0f + expf(-x));
  return s * (1.0f + x * (1.0f - s));
}

__global__ void scan_fwd_kernel(
    const float* u, const float* delta, const float* a,
    const float* b, const float* c, const float* d, const float* z,
    const float* delta_bias, float* y, float* checkpoints,
    int batch, int dim, int length, int chunks) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= batch * dim) return;
  const int bi = channel / dim;
  const int di = channel - bi * dim;
  float state[kState] = {0.0f};
  const int ud_base = (bi * dim + di) * length;
  const int bc_base = bi * kState * length;
  const int cp_base = ((bi * dim + di) * (chunks + 1)) * kState;
  for (int n = 0; n < kState; ++n) checkpoints[cp_base + n] = 0.0f;
  for (int t = 0; t < length; ++t) {
    const float dt = log1pf(expf(delta[ud_base + t] + delta_bias[di]));
    const float x = u[ud_base + t];
    float out = d[di] * x;
    for (int n = 0; n < kState; ++n) {
      const int idx = bc_base + n * length + t;
      const float alpha = expf(dt * a[di * kState + n]);
      state[n] = alpha * state[n] + x * dt * b[idx];
      out += state[n] * c[idx];
    }
    y[ud_base + t] = out * silu(z[ud_base + t]);
    if ((t + 1) % kChunk == 0 || t + 1 == length) {
      const int ci = (t + 1 + kChunk - 1) / kChunk;
      for (int n = 0; n < kState; ++n) checkpoints[cp_base + ci * kState + n] = state[n];
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
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= batch * dim) return;
  const int bi = channel / dim;
  const int di = channel - bi * dim;
  const int ud_base = channel * length;
  const int bc_base = bi * kState * length;
  const int cp_base = channel * (chunks + 1) * kState;
  float adj[kState] = {0.0f};
  float local_d = 0.0f, local_bias = 0.0f;

  for (int chunk = chunks - 1; chunk >= 0; --chunk) {
    const int start = chunk * kChunk;
    const int end = min(length, start + kChunk);
    const int count = end - start;
    float history[kChunk + 1][kState];
    for (int n = 0; n < kState; ++n) history[0][n] = checkpoints[cp_base + chunk * kState + n];
    for (int j = 0; j < count; ++j) {
      const int t = start + j;
      const float dt = log1pf(expf(delta[ud_base + t] + delta_bias[di]));
      const float x = u[ud_base + t];
      for (int n = 0; n < kState; ++n) {
        const int idx = bc_base + n * length + t;
        const float alpha = expf(dt * a[di * kState + n]);
        history[j + 1][n] = alpha * history[j][n] + x * dt * b[idx];
      }
    }
    for (int j = count - 1; j >= 0; --j) {
      const int t = start + j;
      const int ui = ud_base + t;
      const float raw_dt = delta[ui] + delta_bias[di];
      const float dt_sigmoid = 1.0f / (1.0f + expf(-raw_dt));
      const float dt = log1pf(expf(raw_dt));
      const float x = u[ui];
      float base = d[di] * x;
      for (int n = 0; n < kState; ++n) base += history[j + 1][n] * c[bc_base + n * length + t];
      const float gout = gy[ui];
      const float gbase = gout * silu(z[ui]);
      gz[ui] = gout * base * silu_grad(z[ui]);
      local_d += gbase * x;
      float gx = gbase * d[di];
      float gdt = 0.0f;
      for (int n = 0; n < kState; ++n) {
        const int idx = bc_base + n * length + t;
        const float alpha = expf(dt * a[di * kState + n]);
        const float gstate = adj[n] + gbase * c[idx];
        atomicAdd(&gc[idx], gbase * history[j + 1][n]);
        atomicAdd(&gb[idx], gstate * x * dt);
        atomicAdd(&ga[di * kState + n], gstate * dt * alpha * history[j][n]);
        gx += gstate * dt * b[idx];
        gdt += gstate * (a[di * kState + n] * alpha * history[j][n] + x * b[idx]);
        adj[n] = gstate * alpha;
      }
      gu[ui] = gx;
      gdelta[ui] = gdt * dt_sigmoid;
      local_bias += gdelta[ui];
    }
  }
  atomicAdd(&gd[di], local_d);
  atomicAdd(&gbias[di], local_bias);
}
}  // namespace

std::vector<at::Tensor> selective_scan_forward_cuda(
    at::Tensor u, at::Tensor delta, at::Tensor a,
    at::Tensor b, at::Tensor c, at::Tensor d,
    at::Tensor z, at::Tensor delta_bias) {
  TORCH_CHECK(u.is_cuda() && u.scalar_type() == at::kFloat, "inputs must be CUDA float32");
  TORCH_CHECK(a.size(1) == kState, "UDLF kernel requires d_state=16");
  const int batch = u.size(0), dim = u.size(1), length = u.size(2);
  const int chunks = (length + kChunk - 1) / kChunk;
  auto y = at::empty_like(u);
  auto checkpoints = at::empty({batch, dim, chunks + 1, kState}, u.options());
  const int threads = 128;
  scan_fwd_kernel<<<(batch * dim + threads - 1) / threads, threads>>>(
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
  const int chunks = (length + kChunk - 1) / kChunk;
  const int threads = 128;
  scan_bwd_kernel<<<(batch * dim + threads - 1) / threads, threads>>>(
      grad_y.contiguous().data_ptr<float>(), u.data_ptr<float>(), delta.data_ptr<float>(),
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), d.data_ptr<float>(),
      z.data_ptr<float>(), delta_bias.data_ptr<float>(), checkpoints.data_ptr<float>(),
      gu.data_ptr<float>(), gdelta.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(),
      gc.data_ptr<float>(), gd.data_ptr<float>(), gz.data_ptr<float>(), gbias.data_ptr<float>(),
      batch, dim, length, chunks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gu, gdelta, ga, gb, gc, gd, gz, gbias};
}
