param(
    [Parameter(Position = 0)]
    [ValidateSet("status", "start", "stop", "restart", "logs")]
    [string]$Command = "status",
    [string]$WorkspaceConfig = "",
    [int]$Tail = 80
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $PSCommandPath
. "$ScriptDir\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-UdlfWorkspaceConfig
$sshHost = Resolve-UdlfConfigValue $workspace "remote.ssh" "UDLF_REMOTE_SSH"
$remoteRoot = Resolve-UdlfConfigValue $workspace "remote.root" "UDLF_REMOTE_ROOT"
$taskName = "UDLF Workspace Agent"
$serviceRoot = "$remoteRoot/workspace-service"
$processPattern = "remote_workspace_agent.py|remote_workspace_supervisor.py"

$script = switch ($Command) {
    "status" {
        "Get-ScheduledTask -TaskName '$taskName' | Select-Object TaskName,State | Format-List; " +
        "Get-CimInstance Win32_Process | Where-Object { `$_.CommandLine -match '$processPattern' } | " +
        "Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | Format-List"
    }
    "start" { "Start-ScheduledTask -TaskName '$taskName'; Start-Sleep 2; Get-ScheduledTask -TaskName '$taskName' | Select-Object TaskName,State | Format-List" }
    "stop" {
        "Stop-ScheduledTask -TaskName '$taskName' -ErrorAction SilentlyContinue; " +
        "Get-CimInstance Win32_Process | Where-Object { `$_.CommandLine -match '$processPattern' } | " +
        "ForEach-Object { taskkill /PID `$_.ProcessId /T /F 2>`$null | Out-Null }; Write-Output 'workspace agent stopped'"
    }
    "restart" {
        "Stop-ScheduledTask -TaskName '$taskName' -ErrorAction SilentlyContinue; " +
        "Get-CimInstance Win32_Process | Where-Object { `$_.CommandLine -match '$processPattern' } | " +
        "ForEach-Object { taskkill /PID `$_.ProcessId /T /F 2>`$null | Out-Null }; Start-Sleep 2; " +
        "Start-ScheduledTask -TaskName '$taskName'; Start-Sleep 4; Get-ScheduledTask -TaskName '$taskName' | Select-Object TaskName,State | Format-List; " +
        "Get-CimInstance Win32_Process | Where-Object { `$_.CommandLine -match '$processPattern' } | Select-Object ProcessId,Name,CommandLine | Format-List"
    }
    "logs" {
        "Get-Content '$serviceRoot/agent.stdout.log' -Tail $Tail -ErrorAction SilentlyContinue; " +
        "Get-Content '$serviceRoot/agent.stderr.log' -Tail $Tail -ErrorAction SilentlyContinue"
    }
}

& "$ScriptDir\ssh_cmd.ps1" -RemoteHost $sshHost -ScriptBlock $script -Raw
exit $LASTEXITCODE
