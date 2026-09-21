[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet(
        "HHH", "HHHv2", "SpectralNFST",
        "KNN", "LDA", "NuSVC", "SGD", "GauNB", "NC", "RNC",
        "LGBM", "MLP", "TabNet", "FTTransformer", "SAINT"
    )]
    [string]$Competitor
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run_model.ps1") $Competitor
