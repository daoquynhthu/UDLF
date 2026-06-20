param([string]$Label = "diagnose")

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$python = "L:\NAIME_REMOTE\envs\.venv312\Scripts\python.exe"
& $python scripts\diagnose_udlf_failure.py `
    --checkpoint "L:\UDLF_REMOTE\runs\udlf_fineweb_edu_64m_3000_solver2_contended\latest.pt" `
    --data "L:\NAIME_REMOTE\datasets\fineweb_edu_1b_ctx1024" `
    --output "L:\UDLF_REMOTE\runs\udlf_fineweb_edu_64m_3000_solver2_contended\failure_diagnosis.json"
if ($LASTEXITCODE -ne 0) { throw "UDLF failure diagnosis failed with exit code $LASTEXITCODE" }
