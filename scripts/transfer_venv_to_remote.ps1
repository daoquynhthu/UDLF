param(
    [string]$RemoteUser = "",
    [string]$RemoteHost = "",
    [string]$RemoteRoot = "",
    [string]$VenvName = ".venv312",
    [string]$PythonHome = "",
    [string]$WorkspaceConfig = "",
    [switch]$SkipArchive,
    [switch]$SkipUpload
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-UdlfWorkspaceConfig
if (-not $RemoteUser) { $RemoteUser = Resolve-UdlfConfigValue $workspace "remote.user" "UDLF_REMOTE_USER" "" -AllowMissing }
if (-not $RemoteHost) { $RemoteHost = Resolve-UdlfConfigValue $workspace "remote.host" "UDLF_REMOTE_HOST" "" -AllowMissing }
if (-not $RemoteRoot) { $RemoteRoot = Resolve-UdlfConfigValue $workspace "remote.root" "UDLF_REMOTE_ROOT" "" -AllowMissing }
if (-not $PythonHome) { $PythonHome = Resolve-UdlfConfigValue $workspace "remote.python_home" "UDLF_REMOTE_PYTHON_HOME" "" -AllowMissing }
if (-not $RemoteUser) { throw "Missing remote user. Set remote.user or UDLF_REMOTE_USER." }
if (-not $RemoteHost) { throw "Missing remote host. Set remote.host or UDLF_REMOTE_HOST." }
if (-not $RemoteRoot) { throw "Missing remote root. Set remote.root or UDLF_REMOTE_ROOT." }
if (-not $PythonHome) { throw "Missing remote python home. Set remote.python_home or UDLF_REMOTE_PYTHON_HOME." }
$venvPath = Join-Path $repoRoot $VenvName
$archivePath = Join-Path $env:TEMP "$VenvName-udlf-remote.zip"
$remoteArchive = "$RemoteRoot\envs\$VenvName-udlf-remote.zip"
$remoteVenv = "$RemoteRoot\envs\$VenvName"
$remoteArchiveScp = $remoteArchive -replace "\\", "/"
$python = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPath)) {
    throw "Virtual environment not found: $venvPath"
}
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

Write-Host "Local venv : $venvPath"
Write-Host "Archive    : $archivePath"
Write-Host "Remote     : ${RemoteUser}@${RemoteHost}:$remoteVenv"

if (-not $SkipArchive) {
    if (Test-Path -LiteralPath $archivePath) {
        Remove-Item -LiteralPath $archivePath -Force
    }
    $env:UDLF_VENV_SRC = $venvPath
    $env:UDLF_VENV_ZIP = $archivePath
    @'
import os
import sys
import time
import zipfile
from pathlib import Path

src = Path(os.environ["UDLF_VENV_SRC"])
dst = Path(os.environ["UDLF_VENV_ZIP"])
files = [p for p in src.rglob("*") if p.is_file()]
total = sum(p.stat().st_size for p in files)
done = 0
last_print = 0.0
start = time.time()

print(f"Packing {len(files)} files, {total / (1024 ** 3):.2f} GiB")
with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
    for idx, path in enumerate(files, 1):
        rel = path.relative_to(src.parent)
        zf.write(path, rel.as_posix())
        done += path.stat().st_size
        now = time.time()
        if now - last_print >= 1.0 or idx == len(files):
            pct = 100.0 * done / max(1, total)
            speed = done / max(1e-6, now - start) / (1024 ** 2)
            sys.stdout.write(f"\rarchive {pct:6.2f}%  {done/(1024**3):.2f}/{total/(1024**3):.2f} GiB  {speed:.1f} MiB/s")
            sys.stdout.flush()
            last_print = now
print()
print(f"Archive complete: {dst} ({dst.stat().st_size / (1024 ** 3):.2f} GiB)")
'@ | & $python -
}

ssh -o BatchMode=yes "$RemoteUser@$RemoteHost" "if not exist $RemoteRoot mkdir $RemoteRoot && if not exist $RemoteRoot\envs mkdir $RemoteRoot\envs"

if (-not $SkipUpload) {
    Write-Host "Uploading archive with scp progress..."
    scp -o BatchMode=yes $archivePath "${RemoteUser}@${RemoteHost}:$remoteArchiveScp"
}

$remotePs = @"
`$ErrorActionPreference = 'Stop'
`$ProgressPreference = 'SilentlyContinue'
`$archive = '$remoteArchive'
`$venv = '$remoteVenv'
`$pythonHome = '$PythonHome'
if (Test-Path -LiteralPath `$venv) { Remove-Item -LiteralPath `$venv -Recurse -Force }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent `$venv) | Out-Null
Expand-Archive -LiteralPath `$archive -DestinationPath (Split-Path -Parent `$venv) -Force
`$cfg = Join-Path `$venv 'pyvenv.cfg'
if (Test-Path -LiteralPath `$cfg) {
    `$lines = Get-Content -LiteralPath `$cfg
    `$lines = `$lines | ForEach-Object {
        if (`$_ -match '^home\s*=') { "home = `$pythonHome" }
        elseif (`$_ -match '^executable\s*=') { "executable = `$pythonHome\python.exe" }
        elseif (`$_ -match '^command\s*=') { "command = `$pythonHome\python.exe -m venv `$venv" }
        else { `$_ }
    }
    Set-Content -LiteralPath `$cfg -Value `$lines -Encoding utf8
}
& (Join-Path `$venv 'Scripts\python.exe') -c "import sys; print(sys.executable); import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda); import datasets; print('datasets', datasets.__version__)"
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remotePs))
ssh -o BatchMode=yes "$RemoteUser@$RemoteHost" "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $encoded"

Write-Host "Remote venv ready: $remoteVenv"
