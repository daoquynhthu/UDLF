param([string]$Label = "snapshot")

nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
$cache = Join-Path $env:LOCALAPPDATA "torch_extensions"
Get-ChildItem $cache -Filter udlf_selective_scan_cuda.pyd -Recurse -ErrorAction SilentlyContinue |
    Select-Object FullName, Length, LastWriteTime
