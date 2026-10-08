<#
.SYNOPSIS
Stop this checkout's Sparse controllers and tracked model runtimes.
.DESCRIPTION
Stops active work immediately. Session records from start_sparse.ps1 allow safe
cleanup after a controller exits. PID creation times protect against PID reuse.
Untracked orphan model servers are left alone because ownership is unknown.
.EXAMPLE
.\scripts\stop_sparse.ps1 -WhatIf
.EXAMPLE
.\scripts\stop_sparse.ps1
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$processes = @(Get-CimInstance Win32_Process)
$targets = @{}

function Add-SparseTarget($entry) {
    if ($entry.ProcessId -ne $PID) { $targets[[int]$entry.ProcessId] = $entry }
}

# A checkout path plus the controller command identifies a live Sparse root.
foreach ($entry in $processes) {
    if ($entry.Name -match '^(uv|python|pythonw|sparse-network)\.exe$' -and
        $entry.CommandLine -and
        $entry.CommandLine.IndexOf($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
        $entry.CommandLine -match 'sparse[-_]network' -and
        $entry.CommandLine -match '\bapi\s+serve\b') {
        Add-SparseTarget $entry
    }
}

$trackingDirectory = Join-Path $projectRoot ".sparse-data\launcher"
if (Test-Path -LiteralPath $trackingDirectory) {
    foreach ($file in Get-ChildItem -LiteralPath $trackingDirectory -Filter 'session-*.json' -File) {
        try { $record = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json }
        catch { Write-Warning "Skipping unreadable session record: $($file.Name)"; continue }
        if ($record.Root -ne $projectRoot) { continue }
        foreach ($saved in $record.Processes) {
            $entry = $processes | Where-Object {
                $_.ProcessId -eq $saved.Id -and $_.Name -eq $saved.Name -and
                $_.CreationDate.ToUniversalTime().Ticks.ToString() -eq $saved.Created
            } | Select-Object -First 1
            if ($entry) { Add-SparseTarget $entry }
        }
    }
}

# Capture descendants before stopping their parents, including Python workers.
do {
    $previousCount = $targets.Count
    foreach ($entry in $processes) {
        if ($targets.ContainsKey([int]$entry.ParentProcessId)) {
            $parent = $targets[[int]$entry.ParentProcessId]
            if ($entry.CreationDate -ge $parent.CreationDate) { Add-SparseTarget $entry }
        }
    }
} while ($targets.Count -gt $previousCount)

if (-not $targets.Count) {
    Write-Host "No running Sparse processes identified for $projectRoot."
    Write-Host "Untracked orphan model servers are not stopped automatically."
    return
}

# Stop producers first, then captured children; recheck identities before acting.
foreach ($entry in @($targets.Values | Sort-Object CreationDate)) {
    $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($entry.ProcessId)"
    if (-not $current -or $current.CreationDate -ne $entry.CreationDate) { continue }
    if ($PSCmdlet.ShouldProcess("$($entry.Name) PID $($entry.ProcessId)", "Stop Sparse process (interrupts active tasks)")) {
        Stop-Process -Id $entry.ProcessId -Force -ErrorAction Stop
        Write-Host "Stopped $($entry.Name) PID $($entry.ProcessId)"
    }
}
