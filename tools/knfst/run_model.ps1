[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet(
        "KNFST", "HHH", "HHHv2", "SpectralNFST",
        "KNN", "LDA", "NuSVC", "SGD", "GauNB", "NC", "RNC",
        "LGBM", "MLP", "TabNet", "FTTransformer", "SAINT"
    )]
    [string]$Model
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $projectRoot

$fixedKernel = "l05_exponential_kernel"
$fixedScaler = "QuantileTransformer"
$fixedPoly = -1
$fixedSeeds = @(42, 43, 44)
$datasets = [ordered]@{
    "BoT_IoT" = 1000
    "CIC_IoT2023" = 1000
    "ToN_IoT" = 1000
    "UNSW_NB15" = 1000
    "IoTID20" = 2000
    "N_BaIoT" = 1000
    "Edge_IIoTset" = 1000
    "5G_NIDD" = 1000
}

$resultsRoot = Join-Path $projectRoot "results/knfst_comparison"
$modelsRoot = Join-Path $resultsRoot "models"
$tempRoot = Join-Path $resultsRoot ".tmp"
$modelOutput = Join-Path $modelsRoot "${Model}.csv"
New-Item -ItemType Directory -Force $modelsRoot | Out-Null
New-Item -ItemType Directory -Force $tempRoot | Out-Null

$dependencyChecks = @{
    "LGBM" = @{
        Code = "import lightgbm"
        Install = "python -m pip install lightgbm"
    }
    "TabNet" = @{
        Code = "import torch; import pytorch_tabnet"
        Install = "python -m pip install torch pytorch-tabnet"
    }
    "FTTransformer" = @{
        Code = "import torch; import tab_transformer_pytorch"
        Install = "python -m pip install torch tab-transformer-pytorch"
    }
    "SAINT" = @{
        Code = "import torch"
        Install = "python -m pip install torch"
    }
}
if ($dependencyChecks.ContainsKey($Model)) {
    $check = $dependencyChecks[$Model]
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    & python -c $check.Code 2>$null
    $dependencyExitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedErrorActionPreference
    if ($dependencyExitCode -ne 0) {
        throw "Missing dependency for $Model. Install it first: $($check.Install)"
    }
}

Write-Host "Running model: $Model" -ForegroundColor Cyan
Write-Host "Datasets : $($datasets.Keys.Count) full registered datasets"
Write-Host "Kernel   : $fixedKernel (fixed)"
Write-Host "Seeds    : $($fixedSeeds -join ', ') (fixed)"
Write-Host "Output   : $modelOutput"

foreach ($dataset in $datasets.Keys) {
    $limit = [int]$datasets[$dataset]
    $dataType = "${dataset}_${limit}"
    Write-Host "`n=== $dataType ===" -ForegroundColor Cyan

    foreach ($seed in $fixedSeeds) {
        $existingRows = if (Test-Path -LiteralPath $modelOutput) {
            @(Import-Csv -LiteralPath $modelOutput)
        } else {
            @()
        }
        $completed = @(
            $existingRows | Where-Object {
                $_.'Data Type' -eq $dataType -and [int]$_.Seed -eq $seed
            }
        )
        if ($completed.Count -gt 1) {
            throw "Duplicate cached rows: model=$Model dataset=$dataType seed=$seed"
        }
        if ($completed.Count -eq 1) {
            Write-Host "[SKIP] model=$Model seed=$seed" -ForegroundColor DarkYellow
            continue
        }

        $tempFile = Join-Path $tempRoot "${Model}__${dataType}__seed${seed}.csv"
        if (Test-Path -LiteralPath $tempFile) {
            Remove-Item -LiteralPath $tempFile -Force
        }

        $arguments = @(
            "code/train_evaluate.py",
            "--dataset", $dataset,
            "--limit", $limit,
            "--model", $Model,
            "--kernel", $fixedKernel,
            "--scaler", $fixedScaler,
            "--poly", $fixedPoly,
            "--seed", $seed,
            "--output", $tempFile
        )

        Write-Host "[RUN ] model=$Model seed=$seed" -ForegroundColor Green
        & python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment failed: dataset=$dataset model=$Model seed=$seed (exit code $LASTEXITCODE)."
        }

        $newRows = @(Import-Csv -LiteralPath $tempFile)
        if ($newRows.Count -ne 1) {
            throw "Run did not create exactly one result row: $tempFile"
        }

        $ordered = [ordered]@{
            Seed = $seed
            "Input Model" = $Model
        }
        foreach ($property in $newRows[0].PSObject.Properties) {
            $ordered[$property.Name] = $property.Value
        }
        $resultRow = [pscustomobject]$ordered

        if (Test-Path -LiteralPath $modelOutput) {
            $resultRow | Export-Csv -LiteralPath $modelOutput -Append -NoTypeInformation -Encoding utf8
        } else {
            $resultRow | Export-Csv -LiteralPath $modelOutput -NoTypeInformation -Encoding utf8
        }
        Remove-Item -LiteralPath $tempFile -Force
    }
}

$finalRows = @(Import-Csv -LiteralPath $modelOutput)
$duplicates = @(
    $finalRows |
        Group-Object 'Data Type', Seed |
        Where-Object { $_.Count -gt 1 }
)
if ($duplicates.Count -gt 0) {
    throw "Duplicate dataset-seed rows found in $modelOutput"
}

$finalRows |
    Sort-Object @{ Expression = { $_.'Data Type' } }, @{ Expression = { [int]$_.Seed } } |
    Export-Csv -LiteralPath $modelOutput -NoTypeInformation -Encoding utf8

Write-Host "`nDONE: $Model has $($finalRows.Count) rows in one CSV." -ForegroundColor Green
Write-Host "Model CSV: $modelOutput" -ForegroundColor Cyan
