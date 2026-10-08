param(
    [string]$Environment = ".venv-qwen",
    [Parameter(Mandatory = $true)]
    [string]$CacheDir
)

$ErrorActionPreference = "Stop"
$kernelRevision = "7cdb05d472d6c954c7d03182ed836ebfd4610df0"

uv venv $Environment --python 3.12
$python = Join-Path $Environment "Scripts\python.exe"

uv pip install --python $python torch==2.11.0 torchvision==0.26.0 pillow==12.3.0 `
    --index-url https://download.pytorch.org/whl/cu130
uv pip install --python $python `
    transformers==5.15.1 `
    accelerate==1.14.0 `
    safetensors==0.8.0 `
    kernels==0.16.0 `
    triton-windows==3.6.0.post26

& $python -c `
    "from huggingface_hub import snapshot_download; import sys; snapshot_download('kernels-community/finegrained-fp8', repo_type='kernel', revision=sys.argv[2], cache_dir=sys.argv[1])" `
    $CacheDir $kernelRevision

$kernelRoot = Join-Path $CacheDir "kernels--kernels-community--finegrained-fp8"
$refDirectory = Join-Path $kernelRoot "refs"
New-Item -ItemType Directory -Force -Path $refDirectory | Out-Null
Set-Content -LiteralPath (Join-Path $refDirectory "v4") -Value $kernelRevision -NoNewline

Write-Host "Qwen FP8 runtime ready at $python"
Write-Host "Set runtime_executables.transformers to that path in config.local.yaml."
