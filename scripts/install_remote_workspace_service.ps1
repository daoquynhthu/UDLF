param(
    [string]$WorkspaceConfig = "",
    [int]$Port = 9543,
    [string]$BindHost = "127.0.0.1",
    [switch]$SkipSync
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $PSCommandPath
$RepoRoot = Split-Path -Parent $ScriptDir
. "$ScriptDir\load_workspace_config.ps1" -ConfigPath $WorkspaceConfig
$resolvedWorkspaceConfig = $ConfigPath
$workspace = Get-UdlfWorkspaceConfig
$sshHost = Resolve-UdlfConfigValue $workspace "remote.ssh" "UDLF_REMOTE_SSH"
$remoteRoot = Resolve-UdlfConfigValue $workspace "remote.root" "UDLF_REMOTE_ROOT"
$remoteRepo = Resolve-UdlfConfigValue $workspace "remote.repo" "UDLF_REMOTE_REPO"
$remoteRuns = Resolve-UdlfConfigValue $workspace "remote.runs" "UDLF_REMOTE_RUNS"
$remotePython = Resolve-UdlfConfigValue $workspace "remote.python" "UDLF_REMOTE_PYTHON"

if (-not $SkipSync) {
    & "$ScriptDir\sync_to_remote.ps1" -WorkspaceConfig $WorkspaceConfig
    if ($LASTEXITCODE -ne 0) { throw "Remote sync failed" }
}

$setup = @"
`$ErrorActionPreference = 'Stop'
`$root = '$remoteRoot'
`$repo = '$remoteRepo'
`$runs = '$remoteRuns'
`$python = '$remotePython'
`$serviceRoot = Join-Path `$root 'workspace-service'
`$certDir = Join-Path `$serviceRoot 'tls'
`$jobs = Join-Path `$serviceRoot 'jobs'
`$staging = Join-Path `$serviceRoot 'staging'
New-Item -ItemType Directory -Force -Path `$serviceRoot,`$certDir,`$jobs,`$staging | Out-Null

`$openssl = (Get-Command openssl.exe -ErrorAction SilentlyContinue).Source
if (-not `$openssl) {
    `$candidates = @(
        'C:/Program Files/Git/usr/bin/openssl.exe',
        'C:/Program Files/Git/mingw64/bin/openssl.exe'
    )
    `$openssl = `$candidates | Where-Object { Test-Path -LiteralPath `$_ } | Select-Object -First 1
}
if (-not `$openssl) { throw 'OpenSSL is required once to generate the workspace TLS certificate' }
`$opensslRoot = Split-Path -Parent (Split-Path -Parent `$openssl)
`$opensslConfig = Join-Path `$opensslRoot 'ssl/openssl.cnf'
if (Test-Path -LiteralPath `$opensslConfig) {
    `$env:OPENSSL_CONF = `$opensslConfig
} else {
    `$condaConfig = Join-Path (Split-Path -Parent `$openssl) '../ssl/openssl.cnf'
    if (Test-Path -LiteralPath `$condaConfig) {
        `$env:OPENSSL_CONF = (Resolve-Path -LiteralPath `$condaConfig).Path
    }
}

`$cert = Join-Path `$certDir 'workspace.crt.pem'
`$key = Join-Path `$certDir 'workspace.key.pem'
if (-not (Test-Path `$cert) -or -not (Test-Path `$key)) {
    `$opensslStdout = Join-Path `$serviceRoot 'openssl.stdout.log'
    `$opensslStderr = Join-Path `$serviceRoot 'openssl.stderr.log'
    `$opensslProcess = Start-Process -FilePath `$openssl -ArgumentList @(
        'req','-x509','-newkey','rsa:3072','-sha256','-nodes','-days','825',
        '-subj','/CN=udlf-workspace','-keyout',`$key,'-out',`$cert
    ) -Wait -PassThru -NoNewWindow -RedirectStandardOutput `$opensslStdout -RedirectStandardError `$opensslStderr
    if (`$opensslProcess.ExitCode -ne 0) {
        `$opensslError = Get-Content -LiteralPath `$opensslStderr -Raw -ErrorAction SilentlyContinue
        throw "TLS certificate generation failed: `$opensslError"
    }
}

`$tokenPath = Join-Path `$serviceRoot 'token.txt'
if (-not (Test-Path `$tokenPath) -or (Get-Item `$tokenPath).Length -lt 32) {
    `$bytes = New-Object byte[] 32
    `$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    `$rng.GetBytes(`$bytes)
    `$rng.Dispose()
    [Convert]::ToBase64String(`$bytes) | Set-Content -LiteralPath `$tokenPath -NoNewline -Encoding ASCII
}
`$token = [string](Get-Content -LiteralPath `$tokenPath -Raw)
`$currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
& icacls `$tokenPath /inheritance:r /grant:r 'SYSTEM:F' 'Administrators:F' "`$currentIdentity`:R" | Out-Null
& icacls `$key /inheritance:r /grant:r 'SYSTEM:F' 'Administrators:F' "`$currentIdentity`:R" | Out-Null
`$taskName = 'UDLF Workspace Agent'
`$agent = Join-Path `$repo 'scripts/remote_workspace_agent.py'
`$supervisor = Join-Path `$repo 'scripts/remote_workspace_supervisor.py'
`$serviceConfig = Join-Path `$serviceRoot 'service.json'
@{
    python = `$python
    agent = `$agent
    repo = `$repo
    arguments = @(
        '--host', '$BindHost', '--port', '$Port',
        '--token-file', `$tokenPath, '--cert', `$cert, '--key', `$key,
        '--root', `$root, '--repo', `$repo, '--runs', `$runs, '--python', `$python,
        '--database', (Join-Path `$serviceRoot 'jobs.sqlite3'),
        '--jobs-root', `$jobs, '--staging-root', `$staging
    )
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath `$serviceConfig -Encoding UTF8

`$servicePython = Join-Path (Split-Path -Parent `$python) 'pythonw.exe'
if (-not (Test-Path -LiteralPath `$servicePython)) { `$servicePython = `$python }
`$wrapper = Join-Path `$serviceRoot 'start_udlf_workspace.cmd'
`$wrapperLines = @(
    '@echo off',
    ('cd /d "{0}"' -f `$repo),
    ('"{0}" "{1}" --config "{2}"' -f `$servicePython, `$supervisor, `$serviceConfig)
)
Set-Content -LiteralPath `$wrapper -Value `$wrapperLines -Encoding ASCII
`$taskRun = 'cmd.exe /c "' + `$wrapper + '"'
`$createOutput = schtasks /Create /TN "`$taskName" /SC ONCE /ST 23:59 /TR "`$taskRun" /F 2>&1
if (`$LASTEXITCODE -ne 0) {
    throw "Failed to create current-user scheduled task: `$createOutput"
}
if ('$BindHost' -eq '127.0.0.1' -or '$BindHost' -eq 'localhost') {
    try {
        Get-NetFirewallRule -DisplayName 'UDLF Workspace HTTPS' -ErrorAction SilentlyContinue | Remove-NetFirewallRule -ErrorAction SilentlyContinue
    } catch {
        Write-Output "Skipping firewall cleanup for loopback-only service: `$(`$_.Exception.Message)"
    }
} else {
    try {
        Get-NetFirewallRule -DisplayName 'UDLF Workspace HTTPS' -ErrorAction SilentlyContinue | Remove-NetFirewallRule -ErrorAction SilentlyContinue
        New-NetFirewallRule -DisplayName 'UDLF Workspace HTTPS' -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port -Profile Any -RemoteAddress '10.0.0.0/8','172.16.0.0/12','192.168.0.0/16' | Out-Null
    } catch {
        throw "Failed to configure firewall for non-loopback bind: `$(`$_.Exception.Message)"
    }
}
Start-ScheduledTask -TaskName `$taskName
Start-Sleep -Seconds 3

`$fingerprintLine = & `$openssl x509 -in `$cert -noout -fingerprint -sha256
`$fingerprint = (`$fingerprintLine -split '=', 2)[1] -replace ':', ''
[pscustomobject]@{
    ok = `$true
    host = '$BindHost'
    port = $Port
    token = `$token
    certificate_sha256 = `$fingerprint
    task_state = (Get-ScheduledTask -TaskName `$taskName).State.ToString()
    service_root = `$serviceRoot
} | ConvertTo-Json -Compress
exit 0
"@

Write-Output "Installing encrypted remote workspace service in the background..."
$remoteInstallerScp = ($remoteRoot.TrimEnd("/\") + "/workspace-service/install_udlf_workspace_service.ps1")
$remoteInstallerPs = ($remoteRoot.TrimEnd("/\") + "\workspace-service\install_udlf_workspace_service.ps1")
$localInstaller = Join-Path ([System.IO.Path]::GetTempPath()) ("install_udlf_workspace_service_{0}.ps1" -f ([Guid]::NewGuid().ToString("N")))
Set-Content -LiteralPath $localInstaller -Value $setup -Encoding UTF8
try {
    scp $localInstaller "${sshHost}:$remoteInstallerScp" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to upload generated remote installer" }
    $result = @(ssh $sshHost "powershell -NoProfile -ExecutionPolicy Bypass -File `"$remoteInstallerPs`"" 2>&1)
} finally {
    Remove-Item -LiteralPath $localInstaller -Force -ErrorAction SilentlyContinue
}
$remoteExitCode = $LASTEXITCODE
if ($remoteExitCode -ne 0) {
    $result | ForEach-Object { Write-Error $_ }
    throw "Remote workspace service installation failed"
}
$serviceJson = $result | Where-Object { $_ -match '^\{.*\}$' } | Select-Object -Last 1
if (-not $serviceJson) { throw "Remote workspace service installation returned no service metadata" }
$service = $serviceJson | ConvertFrom-Json
$localConfig = Get-Content -LiteralPath $resolvedWorkspaceConfig -Raw | ConvertFrom-Json
$serviceConfig = [pscustomobject]@{
    host = $service.host
    port = [int]$service.port
    token = $service.token
    certificate_sha256 = $service.certificate_sha256
}
if ($localConfig.remote.PSObject.Properties["workspace_service"]) {
    $localConfig.remote.workspace_service = $serviceConfig
} else {
    $localConfig.remote | Add-Member -NotePropertyName workspace_service -NotePropertyValue $serviceConfig
}
$localConfig | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $resolvedWorkspaceConfig -Encoding UTF8
[pscustomobject]@{
    ok = $service.ok
    host = $service.host
    port = $service.port
    certificate_sha256 = $service.certificate_sha256
    task_state = $service.task_state
    service_root = $service.service_root
    local_config_updated = $resolvedWorkspaceConfig
}
