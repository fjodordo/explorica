# =============================================================================
# ci_test_build.ps1
#
# Build the Explorica wheel and run tests against the installed package.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/ci_test_build.ps1
#
# Description:
#   This script builds the project wheel, installs it in a clean virtual
#   environment together with development dependencies, and runs the test
#   suite against the installed package.
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
#
# Dependencies:
#   - build
# =============================================================================

$ErrorActionPreference = "Stop"

function FailureMessage {
    Write-Host "❌ Tests failed! (╥﹏╥)"
}

trap {
    FailureMessage
    exit 1
}

Write-Host "Creating build..."

Remove-Item -Recurse -Force dist, build, venv_test -ErrorAction SilentlyContinue

python -m build --wheel --outdir dist

Write-Host "Creating clean venv..."
python -m venv venv_test

& "venv_test\Scripts\Activate.ps1"

python -m pip install --upgrade pip

Write-Host "Installing wheel in clean env..."

$package = "$($wheel.FullName)`[dev`]"
pip install $package

Write-Host "Running pytest..."

$env:PYTHONPATH = ""

pytest tests -v --maxfail=1

Write-Host "✅ Tests successfully passed! /ᐠ > ˕ <マ ₊˚⊹♡"