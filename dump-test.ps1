param(
    [Parameter(Mandatory=$true)][int]$Fused,
    [Parameter(Mandatory=$true)][string]$Out
)
$ErrorActionPreference = "Stop"
Set-Location "C:\Users\dogez\gfx1151-engine"

$flags = @{
    GDEC_QSA_KV_BF16=1; GDEC_QSA_WMMA=1; GDEC_QSA_WMMA_BTV=1
    GDEC_MOE_LT=1; GDEC_MOE_LT_BF16=1; GDEC_GR_BF16=1
    GDEC_GDN_STREAM=1; GDEC_GDN_WAVE=1; GDEC_NOWARMUP=1
    GDEC_INDEX_FUSED2=1; GDEC_PP_MOE_OUT=1; GDEC_INDEX_STREAM_SELECT=1
    GDEC_GEMM_WMMA=1; GDEC_PREFILL_TAIL_SLACK=1024
    GDEC_KVSNAP=0; GDEC_RCKPT=0
}
foreach ($k in $flags.Keys) { Set-Item "env:$k" $flags[$k] }
if ($Fused -eq 1) { $env:GDEC_GDN_FUSED = "1" } else { Remove-Item env:GDEC_GDN_FUSED -ErrorAction SilentlyContinue }

$ModelDir = "C:\Users\dogez\.models\Qwen3.8-Flash-Next-Abliterated-HGN"
$base = Join-Path $ModelDir "Qwen3.8-Flash-Next-Abliterated.hgn"
$ovl  = Join-Path $ModelDir "Qwen3.8-Flash-Next-Abliterated.overlay.hgn"
$mtp  = Join-Path $ModelDir "Qwen3.8-Flash-Next-Abliterated-mtp.hgn"

$toks = (1000..1511) -join ","
$ErrorActionPreference = "Continue"
$proc = Start-Process -FilePath ".\build\gdec-win.exe" -NoNewWindow -Wait -PassThru `
    -ArgumentList @("`"$base`"", "`"$ovl`"", "`"$mtp`"", "--tokens", $toks, "--gen", "8", "--dump") `
    -RedirectStandardError $Out -RedirectStandardOutput "$Out.stdout"
exit $proc.ExitCode
