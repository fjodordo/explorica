#!/usr/bin/env bash
# =============================================================================
# docs_build.sh
#
# Builds the Sphinx HTML documentation and enforces strict warning checks.
#
# Usage:
#   bash scripts/docs_build.sh
#
# Description:
#   Navigates to the /docs directory relative to the script location,
#   cleans any previously generated build artifacts, and runs a fresh
#   Sphinx HTML build. The -W flag treats all Sphinx warnings as errors,
#   ensuring documentation quality is enforced in CI.
#
# Exit codes:
#   0 - Build completed successfully
#   1 - Build failed (Sphinx error or warning treated as error)
#
# Dependencies:
#   - make
#   - Sphinx (sphinx-build must be available in the current environment)
# =============================================================================

set -e

failure_message() {
    echo "❌ Sphinx checks failed! (╥﹏╥)"
}

trap failure_message ERR

echo "Building documentation with Sphinx..."

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

cd "$SCRIPT_DIR/../docs"

make clean
make html SPHINXOPTS="-W"

echo "✅ Sphinx checks sucessfully passed! /ᐠ > ˕ <マ ₊˚⊹♡"
