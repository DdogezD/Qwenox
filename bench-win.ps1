param(
    [Parameter(Mandatory=$true)][string]$Tag,
    [int]$PpChars = 48000,
    [int]$TgTokens = 256,
    [int]$TimeoutMin = 40
)

$ErrorActionPreference = "Stop"
$Api = "http://127.0.0.1:8731/v1"
$Repo = "C:\Users\dogez\gfx1151-engine"
$OutDir = Join-Path $Repo "logs\bench"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Wait for API to come up (engine cold load takes ~10 min)
$deadline = (Get-Date).AddMinutes($TimeoutMin)
$up = $false
while ((Get-Date) -lt $deadline) {
    try {
        Invoke-RestMethod -Uri "$Api/models" -TimeoutSec 5 | Out-Null
        $up = $true; break
    } catch { Start-Sleep -Seconds 10 }
}
if (!$up) { throw "API did not come up within $TimeoutMin min" }
Write-Host "[bench] API is up"

function Chat([string]$Prompt, [int]$MaxTokens) {
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
    return @{ resp = $r; wall_s = [math]::Round($sw.Elapsed.TotalSeconds, 2) }
}

# --- A: greedy correctness reference ---
$a = Chat "用中文写一段关于秋天的散文，大约200字。" 128
$aText = $a.resp.choices[0].message.content
Set-Content -Path (Join-Path $OutDir "$Tag-greedy.txt") -Value $aText -Encoding UTF8
Write-Host "[bench] greedy: $($a.resp.usage.prompt_tokens) ptok -> $($a.resp.usage.completion_tokens) ctok, wall $($a.wall_s)s"

# --- B: prefill throughput (long prompt, 1 token out) ---
$para = "秋天的风吹过金黄的田野，稻谷低垂着头，仿佛在向大地致谢。远处的山峦层林尽染，红的枫、黄的银杏、绿的松柏交织成一幅油画。村民们忙着收割，笑声在田埂间回荡。这是第{0}段，记录着不同的编号以避免内容完全重复，让每一个段落都有独特的标记。"
$sb = New-Object System.Text.StringBuilder
$i = 0
while ($sb.Length -lt $PpChars) { [void]$sb.AppendLine(($para -f $i)); $i++ }
$long = $sb.ToString()
$b = Chat $long 1
Write-Host "[bench] prefill: $($b.resp.usage.prompt_tokens) ptok, wall $($b.wall_s)s"

# --- C: decode throughput (short prompt, long generation) ---
$c = Chat "请详细讲述中国古代四大发明的历史、影响以及它们如何改变世界文明进程，尽量详尽。" $TgTokens
$cText = $c.resp.choices[0].message.content
Set-Content -Path (Join-Path $OutDir "$Tag-tg.txt") -Value $cText -Encoding UTF8
Write-Host "[bench] decode: $($c.resp.usage.completion_tokens) ctok, wall $($c.wall_s)s"

# --- parse newest engine log ---
$log = Get-ChildItem (Join-Path $Repo "logs\engine-win-*.log") |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$lines = Get-Content $log.FullName
$pe = ($lines | Select-String "prompt eval") |
    Sort-Object { [int]([regex]::Match($_.Line, '/ (\d+) tokens').Groups[1].Value) } |
    Select-Object -Last 1
$de = $lines | Select-String "decode eval" | Select-Object -Last 1
$da = $lines | Select-String "draft acceptance" | Select-Object -Last 1

$summary = @"
tag      : $Tag
log      : $($log.Name)
greedy   : ptok=$($a.resp.usage.prompt_tokens) ctok=$($a.resp.usage.completion_tokens) wall=$($a.wall_s)s
prefillB : ptok=$($b.resp.usage.prompt_tokens) wall=$($b.wall_s)s
decodeC  : ctok=$($c.resp.usage.completion_tokens) wall=$($c.wall_s)s
$pe
$de
$da
"@
Write-Host ""
Write-Host $summary
Set-Content -Path (Join-Path $OutDir "$Tag-summary.txt") -Value $summary -Encoding UTF8
