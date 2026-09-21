[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $projectRoot

$resultsRoot = Join-Path $projectRoot "results/knfst_comparison"
$modelsRoot = Join-Path $resultsRoot "models"
$inputPath = Join-Path $resultsRoot "all_datasets_l05_exponential_kernel_full.csv"
if (-not (Test-Path -LiteralPath $modelsRoot)) {
    throw "Model result directory not found. Run tools/knfst/run_knfst.ps1 first."
}

$modelFiles = @(Get-ChildItem -LiteralPath $modelsRoot -Filter "*.csv" -File)
if ($modelFiles.Count -eq 0) {
    throw "No model CSV files found in $modelsRoot"
}
$rows = @($modelFiles | ForEach-Object { Import-Csv -LiteralPath $_.FullName })
if ($rows.Count -eq 0) {
    throw "Model CSV files contain no result rows."
}
if (@($rows | Where-Object { $_.Model -eq "KNFST" }).Count -eq 0) {
    throw "KNFST results are missing. Run tools/knfst/run_knfst.ps1 once before ranking."
}

$rows |
    Sort-Object @{ Expression = { $_.'Data Type' } }, @{ Expression = { $_.'Input Model' } }, @{ Expression = { [int]$_.Seed } } |
    Export-Csv -LiteralPath $inputPath -NoTypeInformation -Encoding utf8

$requiredColumns = @("Data Type", "Model", "MCC", "F1 Macro", "FPR", "Training time", "Test time")
$availableColumns = @($rows[0].PSObject.Properties.Name)
$missingColumns = @($requiredColumns | Where-Object { $_ -notin $availableColumns })
if ($missingColumns.Count -gt 0) {
    throw "Missing required CSV columns: $($missingColumns -join ', ')"
}

function Get-Mean {
    param([object[]]$Items, [string]$Property)
    $values = @($Items | ForEach-Object { [double]$_.PSObject.Properties[$Property].Value })
    return [double](($values | Measure-Object -Average).Average)
}

function Get-SampleStd {
    param([object[]]$Items, [string]$Property)
    $values = @($Items | ForEach-Object { [double]$_.PSObject.Properties[$Property].Value })
    if ($values.Count -lt 2) {
        return 0.0
    }
    $mean = [double](($values | Measure-Object -Average).Average)
    $sumSquared = 0.0
    foreach ($value in $values) {
        $sumSquared += [math]::Pow($value - $mean, 2)
    }
    return [math]::Sqrt($sumSquared / ($values.Count - 1))
}

$summary = foreach ($group in ($rows | Group-Object Model)) {
    $items = @($group.Group)
    [pscustomobject]@{
        Model = $group.Name
        Datasets = @($items.'Data Type' | Sort-Object -Unique).Count
        Runs = $items.Count
        MCC_Mean = Get-Mean $items "MCC"
        MCC_Std = Get-SampleStd $items "MCC"
        F1_Mean = Get-Mean $items "F1 Macro"
        F1_Std = Get-SampleStd $items "F1 Macro"
        FPR_Mean = Get-Mean $items "FPR"
        Train_s_Mean = Get-Mean $items "Training time"
        Test_s_Mean = Get-Mean $items "Test time"
    }
}

$sorted = @($summary | Sort-Object @{ Expression = { $_.MCC_Mean }; Descending = $true }, @{ Expression = { $_.F1_Mean }; Descending = $true })
$ranking = for ($index = 0; $index -lt $sorted.Count; $index++) {
    [pscustomobject]@{
        Rank = $index + 1
        Model = $sorted[$index].Model
        Datasets = $sorted[$index].Datasets
        Runs = $sorted[$index].Runs
        MCC_Mean = $sorted[$index].MCC_Mean
        MCC_Std = $sorted[$index].MCC_Std
        F1_Mean = $sorted[$index].F1_Mean
        F1_Std = $sorted[$index].F1_Std
        FPR_Mean = $sorted[$index].FPR_Mean
        Train_s_Mean = $sorted[$index].Train_s_Mean
        Test_s_Mean = $sorted[$index].Test_s_Mean
    }
}

$rankingFile = Join-Path $projectRoot "results/knfst_comparison/all_datasets_l05_exponential_kernel_full_ranking.csv"
$ranking | Export-Csv -LiteralPath $rankingFile -NoTypeInformation -Encoding utf8

Write-Host "Ranking across all datasets by mean MCC (F1 Macro breaks ties)" -ForegroundColor Cyan
$ranking |
    Select-Object `
        Rank,
        Model,
        Datasets,
        Runs,
        @{ Name = "MCC mean+/-std"; Expression = { "{0:F6} +/- {1:F6}" -f $_.MCC_Mean, $_.MCC_Std } },
        @{ Name = "F1 mean+/-std"; Expression = { "{0:F2} +/- {1:F2}" -f $_.F1_Mean, $_.F1_Std } },
        @{ Name = "FPR mean"; Expression = { "{0:F2}" -f $_.FPR_Mean } },
        @{ Name = "Train(s)"; Expression = { "{0:F2}" -f $_.Train_s_Mean } },
        @{ Name = "Test(s)"; Expression = { "{0:F4}" -f $_.Test_s_Mean } } |
    Format-Table -AutoSize

$incomplete = @($ranking | Where-Object { $_.Datasets -ne 8 -or $_.Runs -ne 24 })
if ($incomplete.Count -gt 0) {
    Write-Warning "Ranking is preliminary: every completed model must have 8 datasets and 24 runs (8 datasets x 3 seeds)."
}

$best = $ranking[0]
Write-Host ("BEST MODEL: {0} | mean MCC={1:F6} | mean F1={2:F2} | datasets={3} | runs={4}" -f $best.Model, $best.MCC_Mean, $best.F1_Mean, $best.Datasets, $best.Runs) -ForegroundColor Green
Write-Host "Ranking CSV: $rankingFile" -ForegroundColor Cyan
