param(
    [Parameter(Position = 0)]
    [ValidateSet("health", "status", "sync", "push", "pull", "shell", "compare-writes", "test", "train", "report", "audit", "cleanup", "logs", "stop")]
    [string]$Command = "health",
    [Parameter(Position = 1)]
    [string]$Arg1 = "",
    [Parameter(Position = 2)]
    [string]$Arg2 = "",
    [string]$WorkspaceConfig = "",
    [string]$ParamsJson = "",
    [int]$TimeoutSeconds = 3600,
    [switch]$NoFollow
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $WorkspaceConfig) {
    $WorkspaceConfig = Join-Path $RepoRoot "configs\workspace.local.json"
}
$python = Join-Path $RepoRoot ".venv312\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
$client = Join-Path $PSScriptRoot "remote_workspace_client.py"
$common = @($client, "--config", $WorkspaceConfig)

function Ensure-WorkspaceTunnel {
    param([string]$ConfigPath)

    $config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
    $service = $config.remote.workspace_service
    if (-not $service) { return }
    if ($service.host -ne "127.0.0.1" -and $service.host -ne "localhost") { return }

    $sshHost = $config.remote.ssh
    if (-not $sshHost) {
        if ($config.remote.user -and $config.remote.host) {
            $sshHost = "$($config.remote.user)@$($config.remote.host)"
        } else {
            throw "workspace_service uses local tunnel but remote.ssh is not configured"
        }
    }
    $port = [int]$service.port
    $needle = "127.0.0.1:${port}:127.0.0.1:${port}"
    $existing = Get-CimInstance Win32_Process |
        Where-Object { $_.Name -like "ssh*" -and $_.CommandLine -like "*$needle*" -and $_.CommandLine -like "*$sshHost*" } |
        Select-Object -First 1
    if ($existing) { return }

    Start-Process -FilePath "ssh.exe" -ArgumentList @("-N", "-L", $needle, $sshHost) -WindowStyle Hidden | Out-Null
    Start-Sleep -Seconds 2
}

Push-Location $RepoRoot
try {
    Ensure-WorkspaceTunnel -ConfigPath $WorkspaceConfig
    switch ($Command) {
        "health" { & $python @common health }
        "status" {
            $args = $common + @("status")
            if ($Arg1) { $args += $Arg1 }
            & $python @args
        }
        "sync" { & $python @common sync }
        "push" {
            if (-not $Arg1 -or -not $Arg2) { throw "push requires a local source path and workspace-relative destination path" }
            & $python @common push $Arg1 $Arg2
        }
        "pull" {
            if (-not $Arg1 -or -not $Arg2) { throw "pull requires a workspace-relative source path and local destination path" }
            & $python @common pull $Arg1 $Arg2
        }
        "shell" {
            if (-not $Arg1) { throw "shell requires a local .ps1 file path" }
            $script = Get-Content -LiteralPath $Arg1 -Raw
            $arguments = if ($ParamsJson) { @($ParamsJson | ConvertFrom-Json) } else { @() }
            $payload = @{
                script = $script
                cwd = $(if ($Arg2) { $Arg2 } else { "." })
                arguments = $arguments
            } | ConvertTo-Json -Compress
            $args = $common + @("job", "shell", "--payload", $payload, "--timeout", "$TimeoutSeconds")
            if ($NoFollow) { $args += "--no-follow" }
            & $python @args
        }
        "compare-writes" {
            if (-not $Arg1 -or -not $Arg2) {
                throw "compare-writes requires baseline and candidate run names"
            }
            $script = Get-Content -LiteralPath (Join-Path $PSScriptRoot "v8_compare_write_protocols.ps1") -Raw
            $payload = @{
                script = $script
                cwd = "."
                arguments = @("-BaselineRun", $Arg1, "-CandidateRun", $Arg2, "-Compact")
            } | ConvertTo-Json -Compress
            $args = $common + @("job", "shell", "--payload", $payload, "--timeout", "$TimeoutSeconds")
            if ($NoFollow) { $args += "--no-follow" }
            & $python @args
        }
        "test" {
            $args = $common + @("job", "diagnostics", "--timeout", "$TimeoutSeconds")
            if ($NoFollow) { $args += "--no-follow" }
            & $python @args
        }
        "train" {
            if (-not $Arg1) { throw "train requires a template name" }
            $params = if ($ParamsJson) { $ParamsJson | ConvertFrom-Json } else { @{} }
            $payload = @{ template = $Arg1; run_name = $Arg2; params = $params } | ConvertTo-Json -Compress -Depth 8
            $args = $common + @("job", "train", "--payload", $payload, "--timeout", "$TimeoutSeconds")
            if ($NoFollow) { $args += "--no-follow" }
            & $python @args
        }
        "report" {
            if (-not $Arg1) { throw "report requires a run name" }
            $payload = @{ run_name = $Arg1 } | ConvertTo-Json -Compress
            & $python @common job report --payload $payload --timeout "$TimeoutSeconds"
        }
        "audit" {
            if (-not $Arg1) { throw "audit requires a run name" }
            $params = if ($ParamsJson) { $ParamsJson | ConvertFrom-Json } else { @{} }
            $payloadObject = @{
                run_name = $Arg1
                checkpoint_glob = $(if ($params.CheckpointGlob) { $params.CheckpointGlob } else { "step_*.pt" })
                max_checkpoints = $(if ($null -ne $params.MaxCheckpoints) { $params.MaxCheckpoints } else { 0 })
                batch_count = $(if ($null -ne $params.BatchCount) { $params.BatchCount } else { 4 })
                batch_size = $(if ($null -ne $params.BatchSize) { $params.BatchSize } else { 2 })
                chunk_len = $(if ($null -ne $params.ChunkLen) { $params.ChunkLen } else { 256 })
                boundary_tokens = $(if ($null -ne $params.BoundaryTokens) { $params.BoundaryTokens } else { 64 })
            }
            $payload = $payloadObject | ConvertTo-Json -Compress
            $args = $common + @("job", "temporal_audit", "--payload", $payload, "--timeout", "$TimeoutSeconds")
            if ($NoFollow) { $args += "--no-follow" }
            & $python @args
        }
        "cleanup" {
            if (-not $Arg1) { throw "cleanup requires a run name" }
            $payload = @{ run_name = $Arg1 } | ConvertTo-Json -Compress
            & $python @common job cleanup_checkpoints --payload $payload --timeout "$TimeoutSeconds"
        }
        "logs" {
            if (-not $Arg1) { throw "logs requires a job id" }
            $args = $common + @("logs", $Arg1)
            if (-not $NoFollow) { $args += "--follow" }
            & $python @args
        }
        "stop" {
            if (-not $Arg1) { throw "stop requires a job id" }
            & $python @common stop $Arg1
        }
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
