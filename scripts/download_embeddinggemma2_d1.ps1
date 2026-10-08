param(
    [Parameter(Mandatory = $true)]
    [string]$CacheDir
)

$ErrorActionPreference = "Stop"

$resolvedCache = [System.IO.Path]::GetFullPath($CacheDir)
New-Item -ItemType Directory -Force -Path $resolvedCache | Out-Null

uvx --from huggingface-hub hf download google/embeddinggemma-2 `
    --revision 914f7f89142e33e77833254d9c9b90c3cef7303b `
    --cache-dir $resolvedCache
if ($LASTEXITCODE -ne 0) { throw "Failed to download EmbeddingGemma 2." }

uvx --from huggingface-hub hf download LiquidAI/d1-3B `
    --revision 051bcc464b01b9f92942b364d9586b0ef5912432 `
    --cache-dir $resolvedCache
if ($LASTEXITCODE -ne 0) { throw "Failed to download d1-3B." }

Write-Host "Pinned snapshots downloaded to $resolvedCache"
