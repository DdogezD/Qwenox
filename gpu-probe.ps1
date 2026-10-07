# GPU diagnostics probe
$samples = (Get-Counter '\GPU Process Memory(*)\Local Usage' -ErrorAction SilentlyContinue).CounterSamples |
    Where-Object { $_.CookedValue -gt 100MB } |
    Sort-Object CookedValue -Descending |
    Select-Object -First 8 InstanceName, @{n='LocalGB';e={[math]::Round($_.CookedValue/1GB,2)}}
$samples | Format-Table -AutoSize

$shared = (Get-Counter '\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples |
    Where-Object { $_.CookedValue -gt 100MB } |
    Sort-Object CookedValue -Descending |
    Select-Object -First 8 InstanceName, @{n='SharedGB';e={[math]::Round($_.CookedValue/1GB,2)}}
$shared | Format-Table -AutoSize

$util = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples |
    Where-Object { $_.CookedValue -gt 1 } |
    Sort-Object CookedValue -Descending |
    Select-Object -First 8 InstanceName, @{n='Util%';e={[math]::Round($_.CookedValue,1)}}
$util | Format-Table -AutoSize

$mem = Get-CimInstance Win32_OperatingSystem
"RAM: free {0:N1} GB / total {1:N1} GB" -f ($mem.FreePhysicalMemory/1MB), ($mem.TotalVisibleMemorySize/1MB)
