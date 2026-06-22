param(
    [string]$RunName = "udlf_fineweb_edu_64m_residency_fixed_3000",
    [string]$Checkpoint = "latest.pt",
    [string]$OutputName = "local_channel_diagnosis.json",
    [int]$Sequences = 128,
    [int]$BatchSize = 2
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$python = "L:\NAIME_REMOTE\envs\.venv312\Scripts\python.exe"
& $python scripts\diagnose_local_channel.py `
    --checkpoint "L:\UDLF_REMOTE\runs\$RunName\$Checkpoint" `
    --data "L:\NAIME_REMOTE\datasets\fineweb_edu_1b_ctx1024" `
    --output "L:\UDLF_REMOTE\runs\$RunName\$OutputName" `
    --sequences $Sequences `
    --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw "local channel diagnosis failed with exit code $LASTEXITCODE" }
