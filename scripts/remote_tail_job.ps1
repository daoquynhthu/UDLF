param([string[]]$Arguments)

if ($Arguments.Count -lt 1) { throw "job id is required" }
$root = Join-Path "L:\UDLF_REMOTE\workspace-service\jobs" $Arguments[0]
foreach ($name in @("stdout.log", "stderr.log")) {
    $path = Join-Path $root $name
    if (Test-Path $path) {
        "[$name]"
        Get-Content $path -Tail 80
    }
}
if ($Arguments.Count -ge 2) {
    $runRoot = Join-Path "L:\UDLF_REMOTE\runs" $Arguments[1]
    Get-ChildItem $runRoot -Recurse -File -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    $trainLog = Join-Path $runRoot "train.log"
    if (Test-Path $trainLog) {
        "[train.log]"
        Get-Content $trainLog -Tail 80
    }
    $metrics = Join-Path $runRoot "metrics.jsonl"
    if (Test-Path $metrics) {
        "[metrics.jsonl]"
        Get-Content $metrics -Tail 5
    }
}
