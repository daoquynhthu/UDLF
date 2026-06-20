#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> selective_scan_forward_cuda(
    torch::Tensor u, torch::Tensor delta, torch::Tensor a,
    torch::Tensor b, torch::Tensor c, torch::Tensor d,
    torch::Tensor z, torch::Tensor delta_bias);

std::vector<torch::Tensor> selective_scan_backward_cuda(
    torch::Tensor grad_y, torch::Tensor u, torch::Tensor delta,
    torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor d,
    torch::Tensor z, torch::Tensor delta_bias, torch::Tensor checkpoints);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &selective_scan_forward_cuda, "UDLF selective scan forward");
  m.def("backward", &selective_scan_backward_cuda, "UDLF selective scan backward");
}
