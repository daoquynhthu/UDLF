param([string[]]$Arguments)

$ErrorActionPreference = "Stop"
$python = "L:\NAIME_REMOTE\envs\.venv312\Scripts\python.exe"
$env:PYTHONPATH = "src"
$env:TORCH_CUDA_ARCH_LIST = "8.9"
$env:MAX_JOBS = "8"
$env:CL = "/Zm200 /bigobj"
$env:UDLF_CUDA_BUILD_STRICT = "1"
$env:UDLF_CUDA_BUILD_VERBOSE = "1"
$env:Path = "L:\NAIME_REMOTE\envs\.venv312\Scripts;$env:Path"
$env:TEMP = "L:\UDLF_REMOTE\workspace-service\tmp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

$vsDevCmd = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022" -Filter VsDevCmd.bat -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $vsDevCmd) { throw "Visual Studio VsDevCmd.bat was not found" }
cmd /s /c "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] }
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw "cl.exe is unavailable after VsDevCmd initialization" }

$code = @'
import torch
from udlf.selective_scan import selective_scan

torch.manual_seed(41)
device = "cuda"
shape = (2, 8, 70)
u = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
delta = (torch.randn(shape, device=device) * 0.1).requires_grad_()
a = (-torch.rand(8, 16, device=device) - 0.1).requires_grad_()
b = torch.randn(2, 16, 70, device=device, requires_grad=True)
c = torch.randn(2, 16, 70, device=device, requires_grad=True)
d = torch.randn(8, device=device, requires_grad=True)
z = torch.randn(shape, device=device, requires_grad=True)
bias = torch.randn(8, device=device, requires_grad=True)
inputs = (u, delta, a, b, c, d, z, bias)

y = selective_scan(u, delta, a, b, c, d, z=z, delta_bias=bias)
weight = torch.randn_like(y)
grads = torch.autograd.grad((y * weight).sum(), inputs)

refs = tuple(value.detach().clone().requires_grad_(True) for value in inputs)
ru, rdelta, ra, rb, rc, rd, rz, rbias = refs
dt = torch.nn.functional.softplus(rdelta + rbias.view(1, -1, 1))
state = torch.zeros(2, 8, 16, device=device)
ys = []
for t in range(70):
    state = state * torch.exp(dt[:, :, t, None] * ra[None]) + ru[:, :, t, None] * dt[:, :, t, None] * rb[:, None, :, t]
    base = (state * rc[:, None, :, t]).sum(-1) + rd * ru[:, :, t]
    ys.append(base * torch.nn.functional.silu(rz[:, :, t]))
ref_y = torch.stack(ys, dim=-1)
ref_grads = torch.autograd.grad((ref_y * weight).sum(), refs)

torch.testing.assert_close(y, ref_y, rtol=3e-4, atol=3e-4)
for actual, expected in zip(grads, ref_grads):
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
print("custom_scan_forward_backward=ok")
'@

$code | & $python -
if ($LASTEXITCODE -ne 0) { throw "custom selective-scan smoke failed with exit code $LASTEXITCODE" }
