#!/usr/bin/env bash
# =============================================================================
# docs_link_check.sh
#
# Validates all external links present in the Sphinx documentation.
#
# Usage:
#   bash scripts/docs_link_check.sh
#
# Description:
#   Navigates to the /docs directory relative to the script location
#   and runs the Sphinx linkcheck builder, which crawls all external
#   URLs found in the documentation and reports broken or redirected links.
#   The -q flag suppresses verbose output, and -W treats warnings as errors.
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
    echo "❌ Link checks failed! (╥﹏╥)"
}

trap failure_message ERR

echo "Running link checks with Sphinx..."

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

cd "$SCRIPT_DIR/../docs"

make linkcheck SPHINXOPTS="-qW"

echo "✅ Link checks successfully passed! /ᐠ > ˕ <マ ₊˚⊹♡"