param(
    [Parameter(Mandatory = $true)]
    [string]$CacheDir
)

$ErrorActionPreference = "Stop"
$revision = "c5630abae1d940eafe0697512a0325494b02ab42"

uvx --from huggingface-hub hf download PaddlePaddle/PaddleOCR-VL-1.6 `
    --revision $revision `
    --cache-dir $CacheDir

Write-Host "PaddleOCR-VL 1.6 $revision is ready under $CacheDir."
