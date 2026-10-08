param(
    [string]$Environment = ".venv-decisions"
)

$ErrorActionPreference = "Stop"

$python = Join-Path $Environment "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    uv venv $Environment --python 3.12
    if ($LASTEXITCODE -ne 0) { throw "Failed to create decision environment." }
}

uv pip install --python $python torch==2.11.0 torchvision==0.26.0 `
    --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw "Failed to install CUDA PyTorch." }
uv pip install --python $python `
    transformers==5.15.1 `
    accelerate==1.14.0 `
    safetensors==0.8.0 `
    pillow==12.0.0
if ($LASTEXITCODE -ne 0) { throw "Failed to install decision dependencies." }

Write-Host "Decision runtime ready at $python"
Write-Host "Set runtime_executables.decisions to that path in config.local.yaml."
