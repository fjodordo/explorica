#!/usr/bin/env bash
# =============================================================================
# docs_doctests.sh
#
# Runs doctests embedded in the Sphinx documentation.
#
# Usage:
#   bash scripts/docs_doctests.sh
#
# Description:
#   Navigates to the /docs directory relative to the script location
#   and runs Sphinx doctest builder. Only doctests included in the
#   generated documentation are executed, which effectively covers
#   the public API only.
#
# Exit codes:
#   0 - All doctests passed
#   1 - One or more doctests failed
#
# Dependencies:
#   - make
#   - Sphinx (sphinx-build must be available in the current environment)
# =============================================================================

set -e

failure_message() {
    echo "❌ Doctests failed! (╥﹏╥)"
}

trap failure_message ERR

echo "Running doctests with Sphinx..."

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

cd "$SCRIPT_DIR/../docs"

make doctest SPHINXOPTS="-W"

echo "✅ Doctests successfully passed! /ᐠ > ˕ <マ ₊˚⊹♡"