param([string]$Since)
# Prints the number of WHEA-Logger events since $Since, then one line per event.
$t = [datetime]::Parse($Since)
$ev = @(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'; StartTime=$t} -ErrorAction SilentlyContinue)
$ev.Count
foreach ($e in $ev) {
  $m = ($e.Message -replace '\s+', ' ')
  if ($m.Length -gt 220) { $m = $m.Substring(0, 220) }
  "{0} id={1} {2}" -f $e.TimeCreated.ToString('HH:mm:ss'), $e.Id, $m
}
