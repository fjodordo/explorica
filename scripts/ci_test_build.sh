#!/usr/bin/env bash
# =============================================================================
# ci_test_build.sh
#
# Build the Explorica wheel and run tests against the installed package.
#
# Usage:
#   bash scripts/ci_test_build.sh
#
# Description:
#   This script builds the project wheel, installs it in a clean virtual
#   environment together with development dependencies, and runs the test
#   suite against the installed package. This ensures that the built
#   distribution works correctly and includes all required files.
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
#
# Dependencies:
#   - build
# =============================================================================

set -e

failure_message() {
    echo "❌ Tests failed! (╥﹏╥)"
}

trap failure_message ERR


echo Creating build...

rm -rf dist build venv_test
python -m build --wheel --outdir dist

echo Creating clean venv...
python -m venv venv_test
source venv_test/bin/activate
python -m pip install --upgrade pip

echo Installing wheel in clean env
WHEEL=$(ls dist/*.whl)

pip install "${WHEEL}[dev]"

echo "Running pytest..."
PYTHONPATH="" pytest tests -v --maxfail=1

echo "✅ Tests successfully passed! /ᐠ > ˕ <マ ₊˚⊹♡"