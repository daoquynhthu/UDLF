param(
    [string]$RemoteUser = "",
    [string]$RemoteHost = "",
    [string]$RemoteRoot = "",
    [string]$WorkspaceConfig = ""
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$workspace = Get-UdlfWorkspaceConfig
if (-not $RemoteUser) { $RemoteUser = Resolve-UdlfConfigValue $workspace "remote.user" "UDLF_REMOTE_USER" "" -AllowMissing }
if (-not $RemoteHost) { $RemoteHost = Resolve-UdlfConfigValue $workspace "remote.host" "UDLF_REMOTE_HOST" "" -AllowMissing }
if (-not $RemoteRoot) { $RemoteRoot = Resolve-UdlfConfigValue $workspace "remote.root" "UDLF_REMOTE_ROOT" "" -AllowMissing }
if (-not $RemoteUser) { throw "Missing remote user. Set remote.user or UDLF_REMOTE_USER." }
if (-not $RemoteHost) { throw "Missing remote host. Set remote.host or UDLF_REMOTE_HOST." }
if (-not $RemoteRoot) { throw "Missing remote root. Set remote.root or UDLF_REMOTE_ROOT." }

$remoteScript = @"
`$ErrorActionPreference = 'Stop'
`$root = '$RemoteRoot'
if (-not (Test-Path -LiteralPath `$root)) {
    throw "Missing remote root: `$root"
}

`$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
`$paths = @(
    `$root,
    (Join-Path `$root 'UDLF'),
    (Join-Path `$root 'runs'),
    (Join-Path `$root 'datasets'),
    (Join-Path `$root 'envs')
)

Write-Host "Hardening UDLF ACL under `$root for `$currentUser"
foreach (`$path in `$paths) {
    if (-not (Test-Path -LiteralPath `$path)) {
        Write-Host "skip missing: `$path"
        continue
    }

    Write-Host "harden: `$path"
    icacls `$path /inheritance:r | Out-Host
    icacls `$path /grant:r "`${currentUser}:(OI)(CI)F" "*S-1-5-18:(OI)(CI)F" "*S-1-5-32-544:(OI)(CI)F" | Out-Host
    icacls `$path /remove:g "*S-1-5-11" "*S-1-5-32-545" "*S-1-1-0" | Out-Host
}

Write-Host "--- effective roots"
foreach (`$path in `$paths) {
    if (Test-Path -LiteralPath `$path) {
        Write-Host "--- `$path"
        icacls `$path | Out-Host
    }
}
"@

$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteScript))
ssh "$RemoteUser@$RemoteHost" "powershell -NoProfile -EncodedCommand $encoded"
