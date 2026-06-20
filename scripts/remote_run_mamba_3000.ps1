param([string]$RunName = "mamba_fineweb_edu_64m_custom_fused_3000")

$ErrorActionPreference = "Stop"
$python = "L:\NAIME_REMOTE\envs\.venv312\Scripts\python.exe"
$repo = "L:\UDLF_REMOTE\UDLF"
$runDir = Join-Path "L:\UDLF_REMOTE\runs" $RunName
$env:PYTHONPATH = Join-Path $repo "src"
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

$template = Join-Path $repo "configs\training_templates\mamba_fineweb_edu_64m_3000_remote_4090.json"
$settings = Get-Content $template -Raw | ConvertFrom-Json
$settings.run_dir = $runDir
$settings.batch_size = 36
$settings.grad_accum_steps = 1
$settings.auto_batch = $false
$settings.max_steps = 3000
$settings | Add-Member -NotePropertyName eval_batch_size -NotePropertyValue 8 -Force
$settings | Add-Member -NotePropertyName allow_run_overwrite -NotePropertyValue $false -Force
$settings | Add-Member -NotePropertyName resume -NotePropertyValue "" -Force
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$config = Join-Path $runDir "formal_config.json"
[IO.File]::WriteAllText($config, ($settings | ConvertTo-Json -Depth 20), [Text.UTF8Encoding]::new($false))

Push-Location $repo
try {
    & $python -m udlf.training.train --config $config
    if ($LASTEXITCODE -ne 0) { throw "formal Mamba training failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
