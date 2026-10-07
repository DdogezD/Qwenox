param(
    [Parameter(Mandatory=$true)][string]$Tag,
    [string]$ExtraEnv = "",      # e.g. "GDEC_PROF=1;GDEC_PHASE=1"
    [int]$NTokens = 32768,
    [int]$Gen = 1,
    [int]$Spec = 0,              # >0: --spec-gen N --gamma $Gamma (MTP)
    [int]$Gamma = 3,
    [switch]$Dump
)
$ErrorActionPreference = "Continue"
Set-Location "C:\Users\dogez\gfx1151-engine"

$flags = @{
    GDEC_QSA_KV_BF16=1; GDEC_QSA_WMMA=1; GDEC_QSA_WMMA_BTV=1
    GDEC_MOE_LT=1; GDEC_MOE_LT_BF16=1; GDEC_GR_BF16=1
    GDEC_GDN_STREAM=1; GDEC_GDN_WAVE=1; GDEC_NOWARMUP=1
    GDEC_INDEX_FUSED2=1; GDEC_PP_MOE_OUT=1; GDEC_INDEX_STREAM_SELECT=1
    GDEC_GEMM_WMMA=1; GDEC_PREFILL_TAIL_SLACK=1024; GDEC_GDN_FUSED=1
    GDEC_KVSNAP=0; GDEC_RCKPT=0; GDEC_SPEC_GAMMA=2
}
foreach ($k in $flags.Keys) { Set-Item "env:$k" $flags[$k] }
foreach ($kv in ($ExtraEnv -split ';' | Where-Object { $_ -match '=' })) {
    $k, $v = $kv -split '=', 2
    Set-Item "env:$k" $v
}

$ModelDir = "C:\Users\dogez\.models\Qwen3.8-Flash-Next-Abliterated-HGN"
$base = Join-Path $ModelDir "Qwen3.8-Flash-Next-Abliterated.hgn"
$ovl  = Join-Path $ModelDir "Qwen3.8-Flash-Next-Abliterated.overlay.hgn"
$mtp  = Join-Path $ModelDir "Qwen3.8-Flash-Next-Abliterated-mtp.hgn"

$tokFile = "logs\bench\tok$NTokens.txt"
if (!(Test-Path $tokFile)) {
    $rng = New-Object System.Random(42)
    $ids = 1..$NTokens | ForEach-Object { 1000 + $rng.Next(50000) }
    Set-Content -Path $tokFile -Value ($ids -join ",") -Encoding ASCII
}

$out = "logs\bench\cli-$Tag.log"
$args_ = @("`"$base`"", "`"$ovl`"", "`"$mtp`"", "--tokens-file", "`"$tokFile`"", "--gen", "$Gen", "--maxctx", "262144")
if ($Spec -gt 0) { $args_ = @("`"$base`"", "`"$ovl`"", "`"$mtp`"", "--tokens-file", "`"$tokFile`"", "--spec-gen", "$Spec", "--gamma", "$Gamma", "--maxctx", "262144") }
if ($Dump) { $args_ += "--dump" }
$proc = Start-Process -FilePath ".\build\gdec-win.exe" -NoNewWindow -Wait -PassThru `
    -ArgumentList $args_ `
    -RedirectStandardError $out -RedirectStandardOutput "$out.stdout"
Write-Host "exit=$($proc.ExitCode) log=$out"
