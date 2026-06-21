param(
    [string]$Label = "diagnose",
    [string]$RunName = "udlf_fineweb_edu_64m_3000_solver2_contended",
    [string]$Checkpoint = "latest.pt",
    [string]$OutputName = "failure_diagnosis.json",
    [int]$BatchSize = 8
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$python = "L:\NAIME_REMOTE\envs\.venv312\Scripts\python.exe"
& $python scripts\diagnose_udlf_failure.py `
    --checkpoint "L:\UDLF_REMOTE\runs\$RunName\$Checkpoint" `
    --data "L:\NAIME_REMOTE\datasets\fineweb_edu_1b_ctx1024" `
    --output "L:\UDLF_REMOTE\runs\$RunName\$OutputName" `
    --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw "UDLF failure diagnosis failed with exit code $LASTEXITCODE" }
