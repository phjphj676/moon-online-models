$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
  $elapsed = Measure-Command { $output = moon run --target native cmd/benchmark }
  $output
  Write-Output ("elapsed_ms=" + [math]::Round($elapsed.TotalMilliseconds, 3))
} finally {
  Pop-Location
}
