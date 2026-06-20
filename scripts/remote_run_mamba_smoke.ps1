param([string[]]$Arguments)

$ErrorActionPreference = "Stop"
$python = "L:\NAIME_REMOTE\envs\.venv312\Scripts\python.exe"
$env:PYTHONPATH = "src"
$env:TORCH_CUDA_ARCH_LIST = "8.9"
$env:UDLF_CUDA_BUILD_STRICT = "1"
$env:TEMP = "L:\UDLF_REMOTE\workspace-service\tmp"
$env:TMP = $env:TEMP
$vsDevCmd = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022" -Filter VsDevCmd.bat -Recurse -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $vsDevCmd) { throw "Visual Studio VsDevCmd.bat was not found" }
cmd /s /c "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && set" | ForEach-Object {
    if ($_ -match "^([^=]+)=(.*)$") { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] }
}
$env:Path = "L:\NAIME_REMOTE\envs\.venv312\Scripts;$env:Path"
$config = "L:\UDLF_REMOTE\runs\mamba_custom_scan_smoke\workspace_config.json"
$smokeConfig = "L:\UDLF_REMOTE\runs\mamba_custom_scan_smoke\manual_smoke_config.json"
$settings = Get-Content $config -Raw | ConvertFrom-Json
$settings.auto_batch = $false
$settings.batch_size = 2
$settings.grad_accum_steps = 1
$settings.max_steps = 2
$settings.eval_every = 0
$settings.save_every = 0
$settings.latest_every = 0
$settings.allow_run_overwrite = $true
[IO.File]::WriteAllText($smokeConfig, ($settings | ConvertTo-Json -Depth 20), [Text.UTF8Encoding]::new($false))
& $python -m udlf.training.train --config $smokeConfig
if ($LASTEXITCODE -ne 0) { throw "Mamba model smoke failed with exit code $LASTEXITCODE" }
"run_dir=$($settings.run_dir)"
Get-ChildItem $settings.run_dir -File -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
$metrics = Join-Path $settings.run_dir "metrics.jsonl"
if (Test-Path $metrics) { Get-Content $metrics -Tail 2 }
