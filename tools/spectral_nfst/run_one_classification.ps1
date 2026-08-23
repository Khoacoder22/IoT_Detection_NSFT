param(
    [ValidateSet("BoT_IoT", "CIC_IoT2023", "ToN_IoT", "UNSW_NB15", "IoTID20", "N_BaIoT", "Edge_IIoTset", "5G_NIDD")]
    [string]$Dataset = "BoT_IoT",

    [ValidateSet(1000, 2000)]
    [int]$Limit = 1000,

    [ValidateSet("linear", "poly", "rbf", "sigmoid", "abel", "laplacian", "sobolev")]
    [string]$Kernel = "rbf",

    [ValidateSet("QuantileTransformer", "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer")]
    [string]$Scaler = "QuantileTransformer",

    [ValidateSet(-1, 0, 2, 3)]
    [int]$Poly = -1,

    [ValidateRange(1, 1000)]
    [int]$Q = 2,

    # Set to 0 to use all rows in the selected dataset file.
    [ValidateRange(0, 1000000)]
    [int]$SamplesPerClass = 100,

    [int]$Seed = 42,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
Set-Location $projectRoot

if (($Limit -eq 2000) -and ($Dataset -notin @("ToN_IoT", "IoTID20"))) {
    throw "Limit 2000 is available only for ToN_IoT and IoTID20."
}

if ([string]::IsNullOrWhiteSpace($Output)) {
    $sampleTag = if ($SamplesPerClass -eq 0) { "full" } else { "sample$SamplesPerClass" }
    $Output = "results/spectral_nfst/${Dataset}_${Limit}_${Kernel}_${Scaler}_Q${Q}_${sampleTag}_seed${Seed}.csv"
}

$arguments = @(
    "code/run_spectral_nfst_classification.py",
    "--dataset", $Dataset,
    "--limit", $Limit,
    "--kernel", $Kernel,
    "--scaler", $Scaler,
    "--poly", $Poly,
    "--components-per-class", $Q,
    "--seed", $Seed,
    "--output", $Output
)
if ($SamplesPerClass -gt 0) {
    $arguments += @("--samples-per-class", $SamplesPerClass)
}

Write-Host "Dataset : ${Dataset}_${Limit}" -ForegroundColor Cyan
Write-Host "Config  : kernel=$Kernel, scaler=$Scaler, poly=$Poly, Q=$Q, samples/class=$SamplesPerClass, seed=$Seed"
Write-Host "Output  : $Output"
Write-Host "Do not open this output CSV in Excel while the run is active." -ForegroundColor Yellow

& python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "The experiment failed with exit code $LASTEXITCODE. Read the error printed above."
}

Write-Host "DONE. Result saved to: $Output" -ForegroundColor Green

