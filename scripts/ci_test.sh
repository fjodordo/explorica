#!/usr/bin/env bash
# =============================================================================
# ci_test.sh
#
# Run unit and integration tests on the source code for the Explorica project.
#
# Usage:
#   bash scripts/ci_test.sh
#
# Description:
#   This script executes pytest on the Explorica source code to verify that all
#   unit and integration tests pass. It also generates a coverage report in XML
#   format. Intended primarily for local development and quick feedback during
#   feature implementation. The script stops on the first failing test due to
#   '--maxfail=1' and exits with code 1 if any test fails.
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
#
# Dependencies:
#   - pytest
#   - pytest-cov
# =============================================================================

set -e

failure_message() {
    echo "❌ Tests failed! (╥﹏╥)"
}

trap failure_message ERR


echo "Running pytest..."
pytest --cov=explorica --cov-report=xml --maxfail=1

echo "✅ Tests successfully passed! /ᐠ > ˕ <マ ₊˚⊹♡"