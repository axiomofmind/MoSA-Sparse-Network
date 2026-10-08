<#
.SYNOPSIS
Start Sparse and open its local dashboard. Press Ctrl+C to stop.
.EXAMPLE
.\scripts\start_sparse.ps1
.EXAMPLE
.\scripts\start_sparse.ps1 -Mock
#>
[CmdletBinding()]
param(
    [switch]$Mock,
    [switch]$LoadFleet,
    [string]$Config,
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [switch]$NoBrowser,
    [switch]$NoClipboard
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$browserJob = $null
$trackingJob = $null

if ($Mock -and $Config) { throw "Choose either -Mock or -Config." }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install uv and run this launcher again."
}

Push-Location -LiteralPath $projectRoot
try {
    $configPath = if ($Mock) { "configs/mock.yaml" } elseif ($Config) { $Config } else { "config.local.yaml" }
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "Configuration not found: $configPath. Use -Mock to try the dashboard, or configure config.local.yaml."
    }
    $configPath = (Resolve-Path -LiteralPath $configPath).Path
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    try { $listener.Start() }
    catch { throw "Port $Port is unavailable. Stop the existing service or use -Port with another port." }
    finally { $listener.Stop() }

    if (-not $env:SPARSE_API_TOKEN) {
        $env:SPARSE_API_TOKEN = [guid]::NewGuid().ToString("N")
    }
    if ($env:SPARSE_API_TOKEN.Length -lt 16) {
        throw "SPARSE_API_TOKEN must contain at least 16 characters."
    }
    if (-not $NoClipboard) {
        try {
            Set-Clipboard -Value $env:SPARSE_API_TOKEN
            Write-Host "Connection token copied to clipboard. Paste it into the dashboard's Connect dialog."
        } catch {
            Write-Warning 'Could not copy the token. Retrieve it in this terminal with: $env:SPARSE_API_TOKEN'
        }
    } else {
        Write-Host 'Connection token is available in this terminal as $env:SPARSE_API_TOKEN.'
    }

    $dashboardUrl = "http://127.0.0.1:$Port/dashboard/"
    Write-Host "Starting Sparse with $configPath"
    Write-Host "Dashboard: $dashboardUrl"
    Write-Host "Keep this terminal open. Press Ctrl+C to stop. First startup may take longer while dependencies or models load."
    if ($LoadFleet) {
        Write-Host "Waiting for all models to load before serving the dashboard (-LoadFleet)."
    } elseif (-not $Mock) {
        Write-Host "Once connected, choose Models & Hardware > Start fleet to load your models."
    }

    if (-not $NoBrowser) {
        $browserJob = Start-Job -ArgumentList $dashboardUrl -ScriptBlock {
            param($url)
            $deadline = (Get-Date).AddMinutes(5)
            while ((Get-Date) -lt $deadline) {
                try {
                    $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
                    if ($response.StatusCode -eq 200) {
                        Start-Process $url
                        return
                    }
                } catch { }
                Start-Sleep -Milliseconds 500
            }
        }
    }

    $launchArguments = @("run", "sparse-network", "--config", $configPath, "api", "serve", "--host", "127.0.0.1", "--port", "$Port")
    if ($LoadFleet -or $Mock) { $launchArguments += "--load-fleet" }
    # Persist process identities so the stop script can safely identify orphans.
    $trackingDirectory = Join-Path $projectRoot ".sparse-data\launcher"
    New-Item -ItemType Directory -Path $trackingDirectory -Force | Out-Null
    $trackingPath = Join-Path $trackingDirectory ("session-" + [guid]::NewGuid().ToString("N") + ".json")
    $trackingJob = Start-Job -ArgumentList $PID, $trackingPath, $projectRoot -ScriptBlock {
        param($launcherId, $recordPath, $root)
        $known = @{}
        while ($true) {
            $processes = @(Get-CimInstance Win32_Process)
            $parents = @($processes | Where-Object {
                $_.ParentProcessId -eq $launcherId -and $_.Name -eq "uv.exe" -and
                $_.CommandLine -match 'sparse-network'
            })
            $found = @($parents)
            while ($parents.Count) {
                $parentIds = @($parents.ProcessId)
                $parents = @($processes | Where-Object { $_.ParentProcessId -in $parentIds })
                $found += $parents
            }
            foreach ($entry in $found) {
                $key = "$($entry.ProcessId):$($entry.CreationDate.ToUniversalTime().Ticks)"
                $known[$key] = @{
                    Id = $entry.ProcessId
                    Created = $entry.CreationDate.ToUniversalTime().Ticks.ToString()
                    Name = $entry.Name
                }
            }
            if ($known.Count) {
                $json = @{ Root = $root; Processes = @($known.Values) } | ConvertTo-Json -Depth 4
                $temporaryPath = "$recordPath.tmp"
                [System.IO.File]::WriteAllText($temporaryPath, $json)
                Move-Item -LiteralPath $temporaryPath -Destination $recordPath -Force
            }
            Start-Sleep -Milliseconds 500
        }
    }
    & uv @launchArguments
    if ($LASTEXITCODE -ne 0) { throw "Sparse exited with code $LASTEXITCODE." }
} finally {
    if ($null -ne $trackingJob) {
        Stop-Job -Job $trackingJob
        Remove-Job -Job $trackingJob -Force
    }
    if ($null -ne $browserJob) {
        Stop-Job -Job $browserJob
        Remove-Job -Job $browserJob -Force
    }
    Pop-Location
}
