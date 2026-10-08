param(
    [string]$Environment = ".venv-ocr"
)

$ErrorActionPreference = "Stop"

uv venv $Environment --python 3.12
$python = Join-Path $Environment "Scripts\python.exe"

uv pip install --python $python `
    torch==2.11.0 `
    torchvision==0.26.0 `
    pillow==12.3.0 `
    --index-url https://download.pytorch.org/whl/cu130
uv pip install --python $python `
    paddleocr==3.7.0 `
    transformers==5.15.1 `
    accelerate==1.14.0 `
    safetensors==0.8.0

Write-Host "OCR runtime ready at $python"
