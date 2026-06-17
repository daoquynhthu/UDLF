param(
    [string]$RunName = "",
    [string]$RunsRoot = "",
    [string]$RemoteRepo = "",
    [string]$RemotePython = "",
    [string]$WorkspaceConfig = "",
    [int]$TailLog = 80
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $PSCommandPath

. "$ScriptDir\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-UdlfWorkspaceConfig -AllowMissing

if (-not $RunsRoot) { $RunsRoot = Resolve-UdlfConfigValue $workspace "remote.runs" "UDLF_REMOTE_RUNS" "" -AllowMissing }
if (-not $RemoteRepo) { $RemoteRepo = Resolve-UdlfConfigValue $workspace "remote.repo" "UDLF_REMOTE_REPO" "" -AllowMissing }
if (-not $RemotePython) { $RemotePython = Resolve-UdlfConfigValue $workspace "remote.python" "UDLF_REMOTE_PYTHON" "" -AllowMissing }
if (-not $RunsRoot) { $RunsRoot = "L:/UDLF_REMOTE/runs" }
if (-not $RemoteRepo) { $RemoteRepo = "L:/UDLF_REMOTE/UDLF" }
if (-not $RemotePython) { $RemotePython = "python" }

$runExpr = if ($RunName) {
    "`$run = Join-Path '$RunsRoot' '$RunName'"
} else {
    "`$run = Get-ChildItem '$RunsRoot' -Directory -ErrorAction SilentlyContinue | Where-Object { (Test-Path (Join-Path `$_.FullName 'train.log')) -or (Test-Path (Join-Path `$_.FullName 'metrics.jsonl')) } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName"
}

$cmd = @"
`$ErrorActionPreference = 'Continue'
Write-Output '========== GPU =========='
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
Write-Output ''
Write-Output '========== Processes =========='
Get-CimInstance Win32_Process |
  Where-Object { `$_.CommandLine -match 'udlf|UDLF|python.*train' } |
  Select-Object ProcessId,ParentProcessId,Name,CommandLine |
  Format-Table -AutoSize
Write-Output ''
Write-Output '========== Remote Repo =========='
Write-Output '$RemoteRepo'
if (Test-Path '$RemoteRepo') {
  Set-Location '$RemoteRepo'
  & '$RemotePython' -m compileall -q src
  Write-Output "compileall_exit=`$LASTEXITCODE"
} else {
  Write-Output 'missing remote repo'
}
Write-Output ''
Write-Output '========== Run =========='
$runExpr
if (`$run -and (Test-Path `$run)) {
  Write-Output `$run
  Get-ChildItem `$run | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
  Write-Output ''
  Write-Output '========== train.log =========='
  Get-Content (Join-Path `$run 'train.log') -Tail $TailLog -ErrorAction SilentlyContinue
  Write-Output ''
  Write-Output '========== launcher stderr =========='
  Get-Content (Join-Path `$run 'launcher.stderr.log') -Tail 40 -ErrorAction SilentlyContinue
  Write-Output ''
  Write-Output '========== latest metrics =========='
  Get-Content (Join-Path `$run 'metrics.jsonl') -Tail 1 -ErrorAction SilentlyContinue
} else {
  Write-Output 'no run directory found'
}
"@

& "$ScriptDir\remote.ps1" cmd $cmd
