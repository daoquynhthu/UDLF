param(
    [string]$UdlfRun = "udlf_fineweb_edu_64m_residency_fixed_3000",
    [string]$MambaRun = "mamba_fineweb_edu_64m_custom_fused_3000",
    [string]$OutputName = "gap_attribution.json"
)

$ErrorActionPreference = "Stop"
$python = "L:\NAIME_REMOTE\envs\.venv312\Scripts\python.exe"
$env:PYTHONPATH = "src"
$env:TORCH_CUDA_ARCH_LIST = "8.9"
$env:UDLF_CUDA_BUILD_STRICT = "1"
$env:TEMP = "L:\UDLF_REMOTE\workspace-service\tmp"
$env:TMP = $env:TEMP
$vsDevCmd = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022" -Filter VsDevCmd.bat -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
cmd /s /c "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] }
}
$env:Path = "L:\NAIME_REMOTE\envs\.venv312\Scripts;$env:Path"
& $python scripts\attribute_udlf_gap.py `
    --udlf "L:\UDLF_REMOTE\runs\$UdlfRun\latest.pt" `
    --mamba "L:\UDLF_REMOTE\runs\$MambaRun\latest.pt" `
    --data "L:\NAIME_REMOTE\datasets\fineweb_edu_1b_ctx1024" `
    --udlf-metrics "L:\UDLF_REMOTE\runs\$UdlfRun\metrics.jsonl" `
    --mamba-metrics "L:\UDLF_REMOTE\runs\$MambaRun\metrics.jsonl" `
    --output "L:\UDLF_REMOTE\runs\$UdlfRun\$OutputName"
if ($LASTEXITCODE -ne 0) { throw "UDLF gap attribution failed with exit code $LASTEXITCODE" }
