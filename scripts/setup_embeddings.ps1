param(
    [string]$Environment = ".venv-embeddings"
)

$ErrorActionPreference = "Stop"

$python = Join-Path $Environment "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    uv venv $Environment --python 3.12
    if ($LASTEXITCODE -ne 0) { throw "Failed to create embedding environment." }
}

uv pip install --python $python torch==2.11.0 torchvision==0.26.0 `
    --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "Failed to install CPU PyTorch." }
uv pip install --python $python `
    transformers==5.19.0 `
    accelerate==1.14.0 `
    safetensors==0.8.0 `
    pillow==12.0.0 `
    sentence-transformers==6.1.0
if ($LASTEXITCODE -ne 0) { throw "Failed to install embedding dependencies." }

Write-Host "CPU embedding runtime ready at $python"
Write-Host "Set runtime_executables.embeddings to that path in config.local.yaml."
