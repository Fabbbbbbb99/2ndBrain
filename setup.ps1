$ErrorActionPreference = "Stop"
$pyArgs = @("$PSScriptRoot\scripts\setup.py") + $args
if ((Get-Command py -ErrorAction SilentlyContinue) -and (& py -3.13 -c "import sys" 2>$null; $LASTEXITCODE -eq 0)) {
    & py -3.13 @pyArgs
    exit $LASTEXITCODE
}
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py -or $py.Source -like "*WindowsApps*") {
    Write-Error "Python 3.13+ is required. Install Python 3.13+ and re-run."
    exit 2
}
& $py.Source @pyArgs
exit $LASTEXITCODE
