param(
    [string]$RunName = "udlf_fineweb_edu_64m_residency_fixed_3000",
    [string]$Checkpoint = "latest.pt",
    [string]$OutputName = "gradient_horizon_diagnosis.json",
    [int]$BatchSize = 4,
    [int]$Sequences = 4
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$python = "L:\NAIME_REMOTE\envs\.venv312\Scripts\python.exe"
& $python scripts\diagnose_gradient_horizons.py `
    --checkpoint "L:\UDLF_REMOTE\runs\$RunName\$Checkpoint" `
    --data "L:\NAIME_REMOTE\datasets\fineweb_edu_1b_ctx1024" `
    --output "L:\UDLF_REMOTE\runs\$RunName\$OutputName" `
    --batch-size $BatchSize `
    --sequences $Sequences
if ($LASTEXITCODE -ne 0) { throw "gradient horizon diagnosis failed with exit code $LASTEXITCODE" }
