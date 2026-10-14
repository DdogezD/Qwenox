# vram_conc_sampler.ps1 (tools/vram_conc_win.sh) -Out <csv> -Stop <file> : 1 s GPU memory samples until <file> exists
param([string]$Out, [string]$Stop)
"t,adapter_ded_gb,adapter_sh_gb,proc_ded_gb,proc_sh_gb,ws_gb,priv_gb" | Out-File -Encoding ascii $Out
while (-not (Test-Path $Stop)) {
  $p = Get-Process qwenox-engine-win -ErrorAction SilentlyContinue | Select-Object -First 1
  $ad = 0; $as = 0; $pd = 0; $ps = 0
  try {
    $c = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage','\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples
    foreach ($s in $c) {
      $path = $s.Path
      if ($path -like '*gpu adapter memory*') {
        if ($path -like '*dedicated usage') { $ad += $s.CookedValue } else { $as += $s.CookedValue }
      } elseif ($p -and ($path -like ("*pid_" + $p.Id + "_*"))) {
        if ($path -like '*dedicated usage') { $pd += $s.CookedValue } else { $ps += $s.CookedValue }
      }
    }
  } catch {}
  $ws = 0; $pv = 0
  if ($p) { $p.Refresh(); $ws = $p.WorkingSet64; $pv = $p.PrivateMemorySize64 }
  $t = (Get-Date).ToString('HH:mm:ss')
  ('{0},{1:N3},{2:N3},{3:N3},{4:N3},{5:N3},{6:N3}' -f $t, ($ad/1GB), ($as/1GB), ($pd/1GB), ($ps/1GB), ($ws/1GB), ($pv/1GB)) | Out-File -Append -Encoding ascii $Out
  Start-Sleep -Milliseconds 700
}
