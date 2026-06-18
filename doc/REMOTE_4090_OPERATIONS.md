# Remote 4090 Operations

This document defines the current UDLF remote 4090 workflow. It covers reusable
remote operations only: configuration, SSH execution, code sync, status
inspection, log polling, STOP-file shutdown, and detached process launch.

Current isolated remote layout:

```text
Remote root:        L:\UDLF_REMOTE
Remote repo:        L:\UDLF_REMOTE\UDLF
Remote runs:        L:\UDLF_REMOTE\runs
Workspace service:  L:\UDLF_REMOTE\workspace-service
Python venv:        L:\NAIME_REMOTE\envs\.venv312
Service port:       9543 through local SSH tunnel only
```

The UDLF workspace reuses the mature Python environment from NAIME_REMOTE, but
the repository, run outputs, service database, staging files, token, and TLS
certificate are under `L:\UDLF_REMOTE`. Do not point UDLF jobs at the remote
NAIME repository.

The current validated remote smoke candidate is fixed-diffusion K=4 real-token
query recall. It is a smoke recipe for validating remote sync, launch, logs,
metrics, checkpoints, and the core state gate; it is not a long-run recipe.

## Private Configuration

Create local machine configuration from the example:

```powershell
Copy-Item configs\workspace.example.json configs\workspace.local.json
```

Required remote keys:

```text
remote.user
remote.host
remote.ssh
remote.root
remote.repo
remote.runs
remote.datasets
remote.venv
remote.python
```

Preferred environment overrides:

```text
UDLF_REMOTE_USER
UDLF_REMOTE_HOST
UDLF_REMOTE_SSH
UDLF_REMOTE_ROOT
UDLF_REMOTE_REPO
UDLF_REMOTE_RUNS
UDLF_REMOTE_DATASETS
UDLF_REMOTE_PYTHON
UDLF_REMOTE_PYTHON_HOME
```

Only `UDLF_*` environment names are supported in this workspace.

## Shared Machine Rules

- Check GPU and process ownership before launch.
- Do not kill unknown processes.
- Prefer STOP-file shutdown over interrupts or force-kill.
- Do not leave visible PowerShell, CMD, or bash windows on the remote desktop.
- Launch long-running jobs through `scripts/launch_train_detached.py` or a later
  UDLF-specific wrapper.
- Keep private host names, credentials, absolute paths, datasets, checkpoints,
  and run outputs out of git.

## Preflight

GPU snapshot:

```powershell
.\scripts\ssh_cmd.ps1 -ScriptBlock {
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
}
```

Training-related processes:

```powershell
.\scripts\ssh_cmd.ps1 -ScriptBlock {
    Get-CimInstance Win32_Process |
      Where-Object { $_.CommandLine -match "udlf|UDLF|python.*train" } |
      Select-Object ProcessId,ParentProcessId,Name,CommandLine
}
```

Disk space:

```powershell
.\scripts\ssh_cmd.ps1 -ScriptBlock {
    Get-PSDrive -PSProvider FileSystem |
      Select-Object Name,Used,Free,Root
}
```

## Sync Code

Sync this workspace to `remote.repo`:

```powershell
.\scripts\sync_to_remote.ps1
```

The sync package includes:

- `src/`
- `scripts/`
- `configs/`
- `doc/`
- `tests/`
- `experiments/`
- root files: `README.md`, `plan.md`, `progress.md`, `issues.md`,
  `pyproject.toml`, `requirements.txt`, `.gitignore` when present

It excludes `.git`, virtual environments, run outputs, artifacts, datasets,
logs, checkpoints, and zip files.

## HTTPS Workspace Service

Install or refresh the isolated UDLF workspace service:

```powershell
.\scripts\install_remote_workspace_service.ps1
```

The installer:

- syncs code unless `-SkipSync` is provided;
- writes service state under `L:\UDLF_REMOTE\workspace-service`;
- starts a current-user scheduled task named `UDLF Workspace Agent`;
- binds the HTTPS agent to remote `127.0.0.1:9543`;
- updates ignored `configs\workspace.local.json` with the pinned TLS
  fingerprint and service token.

Because the service binds to remote loopback only, local access uses an SSH
local forward. `scripts\remote_workspace.ps1` starts the tunnel automatically
when `remote.workspace_service.host` is `127.0.0.1`.

Health check:

```powershell
.\scripts\remote_workspace.ps1 health
```

Expected root paths in the health response:

```text
root: L:\UDLF_REMOTE
repo: L:\UDLF_REMOTE\UDLF
runs: L:\UDLF_REMOTE\runs
```

Manage the scheduled task:

```powershell
.\scripts\manage_remote_workspace_service.ps1 status
.\scripts\manage_remote_workspace_service.ps1 restart
.\scripts\manage_remote_workspace_service.ps1 logs -Tail 80
```

## Inspect Remote State

General status:

```powershell
.\scripts\remote.ps1 status
```

Generic remote command:

```powershell
.\scripts\remote.ps1 cmd { Get-Location; nvidia-smi }
```

Inspect latest or named run:

```powershell
.\scripts\inspect_remote_training.ps1
.\scripts\inspect_remote_training.ps1 -RunName <RUN_NAME>
```

Poll logs without creating a persistent remote tail process:

```powershell
.\scripts\watch_remote.ps1 -RunName <RUN_NAME>
.\scripts\watch_remote.ps1 -RunName <RUN_NAME> -Follow:$false -TailLines 120
```

## STOP-File Shutdown

Create a STOP file for a named run:

```powershell
.\scripts\remote.ps1 stop <RUN_NAME>
```

For shutdown/logoff guard usage on the remote machine:

```powershell
.\scripts\shutdown_guard.ps1 -RunName <RUN_NAME>
```

The guard does not force-kill the trainer. It creates `STOP` and waits briefly
for clean exit.

## Detached Launch

The current launcher starts a Python module in a hidden detached process and
writes:

- `daemon.pid`
- `launch_cmd.txt`
- `module_args.txt`
- `launcher.stdout.log`
- `launcher.stderr.log`

Example:

```powershell
.\scripts\launch_train_detached.ps1 `
  -RunName udlf_smoke_001 `
  -Module udlf.training.train `
  -- --config configs\training_templates\udlf_smoke.json
```

## Fixed K=4 Real-Token Query-Recall Smoke

Create a private launch config from the tracked template. Do not commit the
generated `.local.json` file:

```powershell
python scripts\prepare_remote_smoke_config.py `
  --data-path "<REMOTE_SAVED_TOKEN_DATASET>"
```

Then sync code and launch the detached run:

```powershell
.\scripts\sync_to_remote.ps1
.\scripts\launch_train_detached.ps1 `
  -RunName udlf_remote_real_token_query_recall_smoke `
  -Module udlf.training.train `
  -- --config configs\training_templates\udlf_remote_real_token_query_recall_smoke.local.json
```

Inspect the run:

```powershell
.\scripts\inspect_remote_training.ps1 -RunName udlf_remote_real_token_query_recall_smoke
.\scripts\watch_remote.ps1 -RunName udlf_remote_real_token_query_recall_smoke -Follow:$false
```

The tracked example keeps `data_path` as a placeholder. Use only private config
or environment-specific generated config for real remote paths.
