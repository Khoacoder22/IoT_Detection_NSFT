param(
    [Parameter(Mandatory = $true)]
    [string]$InputFile,

    [ValidateRange(1, 1000)]
    [int]$Top = 10
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $projectRoot

if (-not (Test-Path $InputFile)) {
    throw "Result file not found: $InputFile"
}

$rows = @(Import-Csv $InputFile)
if ($rows.Count -eq 0) {
    throw "The CSV contains no result rows: $InputFile"
}

Write-Host "Successful rows: $($rows.Count)" -ForegroundColor Cyan
$rows |
    Sort-Object { [double]$_.MCC } -Descending |
    Select-Object -First $Top `
        @{Name = "Dataset"; Expression = { $_.'Data Type' }},
        Kernel,
        SCALER,
        @{Name = "Q"; Expression = { $_.Model -replace 'SpectralNFST-Q', '' }},
        @{Name = "MCC"; Expression = { "{0:F6}" -f [double]$_.MCC }},
        @{Name = "F1 Macro"; Expression = { "{0:F2}" -f [double]$_.'F1 Macro' }},
        @{Name = "ACC"; Expression = { "{0:F2}" -f [double]$_.ACC }},
        @{Name = "Train(s)"; Expression = { "{0:F2}" -f [double]$_.'Training time' }} |
    Format-Table -AutoSize

Write-Host "MCC is the primary ranking metric; also inspect F1 Macro and the confusion matrix." -ForegroundColor Yellow

