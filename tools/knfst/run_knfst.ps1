[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run_model.ps1") KNFST
