#!/usr/bin/env bash
# =============================================================================
# ci_lint.sh
#
# Runs static analysis and code style checks on the explorica source code.
#
# Usage:
#   bash scripts/ci_lint.sh
#
# Description:
#   Sequentially runs Pylint, Flake8, and Black against the explorica source
#   code. All tools are configured with a maximum line length of 88 characters,
#   consistent with Black's default formatting style.
#   Black is run in check-only mode and does not modify any files.
#
# Exit codes:
#   0 - All checks passed
#   1 - One or more checks failed
#
# Dependencies:
#   - pylint
#   - flake8
#   - black
# =============================================================================

set -e

failure_message() {
    echo "❌ Linters failed! (╥﹏╥)"
}

trap failure_message ERR


echo "Checking explorica with Pylint..."
PYTHONPATH=src pylint src/explorica --max-line-length=88

echo "Checking explorica with Flake8..."
flake8 src/explorica --max-line-length=88

echo "Checking explorica format with Black..."
black --check src/explorica -l 88


echo "✅ Linters sucessfully passed! /ᐠ > ˕ <マ ₊˚⊹♡"