param(
    [ValidateSet("BoT_IoT", "CIC_IoT2023", "ToN_IoT", "UNSW_NB15", "IoTID20", "N_BaIoT", "Edge_IIoTset", "5G_NIDD")]
    [string]$Dataset = "BoT_IoT",

    [ValidateSet(1000, 2000)]
    [int]$Limit = 1000,

    # Comma-separated text is intentional: it works reliably with Windows
    # PowerShell's `powershell -File ...` command-line argument handling.
    [string]$Kernels = "rbf,laplacian,linear",
    [string]$Scalers = "QuantileTransformer,StandardScaler,RobustScaler",
    [string]$Components = "1,2,3",

    [ValidateSet(-1, 0, 2, 3)]
    [int]$Poly = -1,

    # Set to 0 to use all rows. Use 100 or 250 for the first parameter search.
    [ValidateRange(0, 1000000)]
    [int]$SamplesPerClass = 100,

    [int]$Seed = 42,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $projectRoot

$validKernels = @("linear", "poly", "rbf", "sigmoid", "abel", "laplacian", "sobolev", "rff", "chi2","l05_exponential_kernel")
$validScalers = @("QuantileTransformer", "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer")
$kernelList = @($Kernels -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$scalerList = @($Scalers -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$componentList = @($Components -split "," | ForEach-Object {
    $parsedQ = 0
    if (-not [int]::TryParse($_.Trim(), [ref]$parsedQ)) {
        throw "Invalid Q value: $_"
    }
    $parsedQ
})
foreach ($kernel in $kernelList) {
    if ($kernel -notin $validKernels) { throw "Invalid kernel: $kernel" }
}
foreach ($scaler in $scalerList) {
    if ($scaler -notin $validScalers) { throw "Invalid scaler: $scaler" }
}
foreach ($q in $componentList) {
    if ($q -lt 1) { throw "Every Q value must be at least 1." }
}
if (($Limit -eq 2000) -and ($Dataset -notin @("ToN_IoT", "IoTID20", "N_BaIoT"))) {
    throw "Limit 2000 is available only for ToN_IoT, IoTID20, and N_BaIoT."
}

if ([string]::IsNullOrWhiteSpace($Output)) {
    $sampleTag = if ($SamplesPerClass -eq 0) { "full" } else { "sample$SamplesPerClass" }
    $Output = "results/spectral_nfst/grid_${Dataset}_${Limit}_${sampleTag}_seed${Seed}.csv"
}

$outputPath = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Output))
$outputDirectory = Split-Path $outputPath -Parent
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

# Fail before expensive training if Excel or another program has locked the CSV.
if (Test-Path $outputPath) {
    try {
        $lockCheck = [System.IO.File]::Open(
            $outputPath,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
        $lockCheck.Close()
    }
    catch {
        throw "Cannot write to '$Output'. Close this CSV in Excel, then run this same command again."
    }
}

# Build a lookup from successful CSV rows. This makes the grid resumable: runs
# already present are skipped, while interrupted or permission-failed runs retry.
$completed = @{}
if (Test-Path $outputPath) {
    Import-Csv $outputPath | ForEach-Object {
        $key = "$($_.'Data Type')|$($_.Poly)|$($_.Kernel)|$($_.SCALER)|$($_.Model)"
        $completed[$key] = $true
    }
}

$total = $kernelList.Count * $scalerList.Count * $componentList.Count
$position = 0
Write-Host "Grid: $total configurations for ${Dataset}_${Limit}" -ForegroundColor Cyan
Write-Host "Output: $Output"
Write-Host "Existing successful rows will be skipped automatically."
Write-Host "Do not open this output CSV in Excel while the grid is active." -ForegroundColor Yellow

foreach ($kernel in $kernelList) {
    foreach ($scaler in $scalerList) {
        foreach ($q in $componentList) {
            $position += 1
            $modelName = "SpectralNFST-Q$q"
            $dataType = "${Dataset}_${Limit}"
            $key = "$dataType|$Poly|$kernel|$scaler|$modelName"

            if ($completed.ContainsKey($key)) {
                Write-Host "[$position/$total] SKIP: $kernel | $scaler | Q=$q" -ForegroundColor DarkGray
                continue
            }

            Write-Host "[$position/$total] RUN : $kernel | $scaler | Q=$q" -ForegroundColor Cyan
            $arguments = @(
                "code/run_spectral_nfst_classification.py",
                "--dataset", $Dataset,
                "--limit", $Limit,
                "--kernel", $kernel,
                "--scaler", $scaler,
                "--poly", $Poly,
                "--components-per-class", $q,
                "--seed", $Seed,
                "--output", $Output
            )
            if ($SamplesPerClass -gt 0) {
                $arguments += @("--samples-per-class", $SamplesPerClass)
            }

            & python @arguments
            if ($LASTEXITCODE -ne 0) {
                throw "Grid stopped at: $kernel | $scaler | Q=$q. Fix the error and run the same command again; completed rows will be skipped."
            }
            $completed[$key] = $true
        }
    }
}

Write-Host "DONE. All requested configurations are in: $Output" -ForegroundColor Green
Write-Host "View the top 10 with:"
Write-Host "powershell -ExecutionPolicy Bypass -File tools/spectral_nfst/show_top_results.ps1 -InputFile `"$Output`""
