param(
    [int]$Match = 4, [int]$Min = 1, [int]$Max = 8,
    [int]$Gen = 200, [string]$Tag = "ng4", [string]$TokFile = "", [int]$Reps = 1
)
$ErrorActionPreference = "Stop"
Set-Location "C:\Users\dogez\gfx1151-engine"

$flags = @{
    GDEC_QSA_KV_BF16=1; GDEC_QSA_WMMA=1; GDEC_QSA_WMMA_BTV=1
    GDEC_MOE_LT=1; GDEC_MOE_LT_BF16=1; GDEC_GR_BF16=1
    GDEC_GDN_STREAM=1; GDEC_GDN_WAVE=1; GDEC_NOWARMUP=1
    GDEC_INDEX_FUSED2=1; GDEC_PP_MOE_OUT=1; GDEC_INDEX_STREAM_SELECT=1
    GDEC_GEMM_WMMA=1; GDEC_PREFILL_TAIL_SLACK=1024; GDEC_GDN_FUSED=1
    GDEC_KVSNAP=0; GDEC_RCKPT=0; GDEC_SPEC_GAMMA=2
    GDEC_NGRAM_MATCH=$Match; GDEC_NGRAM_MIN=$Min; GDEC_NGRAM_MAX=$Max
}
foreach ($k in $flags.Keys) { Set-Item "env:$k" $flags[$k] }

$ModelDir = "C:\Users\dogez\.models\Qwen3.8-Flash-Next-Abliterated-HGN"
$base = Join-Path $ModelDir "Qwen3.8-Flash-Next-Abliterated.hgn"
$ovl  = Join-Path $ModelDir "Qwen3.8-Flash-Next-Abliterated.overlay.hgn"
$mtp  = Join-Path $ModelDir "Qwen3.8-Flash-Next-Abliterated-mtp.hgn"

$log = "logs\bench\serve-probe-$Tag.log"
$proc = Start-Process -FilePath ".\build\gdec-win.exe" -NoNewWindow -PassThru `
    -ArgumentList @("`"$base`"", "`"$ovl`"", "`"$mtp`"", "--serve", "--port", "8730", "--maxctx", "262144") `
    -RedirectStandardError $log -RedirectStandardOutput "$log.stdout"

$deadline = (Get-Date).AddMinutes(20)
$up = $false
while ((Get-Date) -lt $deadline) {
    if (Test-Path $log) {
        if (Select-String -Path $log -Pattern "serve: listening" -Quiet) { $up = $true; break }
    }
    if ($proc.HasExited) { throw "engine exited early, see $log" }
    Start-Sleep -Seconds 3
}
if (!$up) { throw "engine did not start" }
Write-Host "engine up (pid $($proc.Id))"

if ($TokFile) {
    $ids = @((Get-Content $TokFile -Raw).Trim() -split ',' | ForEach-Object { [int]$_ })
} else {
    # period-13 id sequence x40 = 520 prompt ids
    $blk = @(99203,112757,118362,100695,111020,130020,132195,100375,95874,100330,95874,100357,95874)
    $ids = @()
    for ($i = 0; $i -lt 40; $i++) { $ids += $blk }
}
$reqline = "GEN 1 $Gen 1 151643 $($ids.Count) $($ids -join ' ') 4"

for ($rep = 1; $rep -le $Reps; $rep++) {
    $reqN = "GEN $rep $Gen 1 151643 $($ids.Count) $($ids -join ' ') 4"
    $cli = New-Object System.Net.Sockets.TcpClient("127.0.0.1", 8730)
    $st = $cli.GetStream()
    $wr = New-Object System.IO.StreamWriter($st)
    $wr.AutoFlush = $true
    $wr.NewLine = "`n"
    $wr.Write($reqN + "`n")
    $rd = New-Object System.IO.StreamReader($st)
    $toks = @()
    while ($true) {
        $line = $rd.ReadLine()
        if ($null -eq $line) { break }
        if ($line.StartsWith("T ")) { $toks += [int]($line -split ' ')[2] }
        elseif ($line.StartsWith("D ")) { Write-Host "rep${rep} D: $line"; break }
    }
    $cli.Close()
    if ($rep -eq 1) {
        Set-Content -Path "logs\bench\serve-probe-$Tag-tokens.txt" -Value ($toks -join ",") -Encoding ASCII
    }
    Write-Host "rep${rep}: got $($toks.Count) tokens"
}
Write-Host "--- chain lines ---"
Get-Content $log | Select-String "chain: .*ngram=" | ForEach-Object { Write-Host ($_ -replace '^\[.*?\] ','') }

Stop-Process -Id $proc.Id -Force
Write-Host "engine stopped"
