param(
    [Parameter(Mandatory=$true)][string]$Tag,
    [Parameter(Mandatory=$true)][string]$Prompt,
    [int]$MaxTokens = 800
)
$ErrorActionPreference = "Stop"
$Api = "http://127.0.0.1:8731/v1"
$Repo = "C:\Users\dogez\gfx1151-engine"

$deadline = (Get-Date).AddMinutes(20)
$up = $false
while ((Get-Date) -lt $deadline) {
    try { Invoke-RestMethod -Uri "$Api/models" -TimeoutSec 5 | Out-Null; $up = $true; break }
    catch { Start-Sleep -Seconds 5 }
}
if (!$up) { throw "API not up" }

$body = @{
    model = "gdec"
    messages = @(@{ role = "user"; content = $Prompt })
    temperature = 0.0
    max_tokens = $MaxTokens
    stream = $false
} | ConvertTo-Json -Depth 5
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$r = Invoke-RestMethod -Uri "$Api/chat/completions" -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 3600
$sw.Stop()

$log = Get-ChildItem (Join-Path $Repo "logs\engine-win-*.log") |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$chain = Get-Content $log.FullName | Select-String "chain: .*ngram=" | Select-Object -Last 1
$de = Get-Content $log.FullName | Select-String "decode eval" | Select-Object -Last 1
Write-Host "tag=$Tag ctok=$($r.usage.completion_tokens) wall=$([math]::Round($sw.Elapsed.TotalSeconds,1))s"
Write-Host $de
Write-Host ($chain -replace '^\[.*?\] ','')
