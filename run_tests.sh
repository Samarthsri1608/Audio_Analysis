#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_tests.sh — One-command test runner for the AI Speaking Assessment Engine
#
# Usage:
#   ./run_tests.sh                     # Run all tests
#   ./run_tests.sh -k "reproducibility"# Run only reproducibility tests
#   ./run_tests.sh -k "not Reproduc"   # Skip slow reproducibility tests
#   ./run_tests.sh --base-url http://other-host:8000
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_ROOT/audio"
LOG="$PROJECT_ROOT/test_results.log"

echo "════════════════════════════════════════════════════════"
echo "  AI Speaking Assessment Engine — Test Runner"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════"

# Activate virtual environment
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
    echo "  ✅ Virtual environment activated: $VENV"
else
    echo "  ⚠️  No virtualenv found at $VENV — using system Python"
fi

# Install test dependencies if not present
python -c "import pytest, requests" 2>/dev/null || {
    echo "  📦 Installing test dependencies..."
    pip install pytest requests --quiet
}

# Check server is up
echo ""
echo "  🔍 Checking server at http://localhost:8000 ..."
if curl -s --connect-timeout 5 http://localhost:8000/ > /dev/null; then
    echo "  ✅ Server is up"
else
    echo ""
    echo "  ❌ Server is NOT running at http://localhost:8000"
    echo "     Please start it first with:"
    echo "     cd backend && uvicorn main:app --host 0.0.0.0 --port 8000"
    echo ""
    exit 1
fi

echo ""
echo "  🧪 Running test suite..."
echo "  Output will be saved to: $LOG"
echo ""

cd "$SCRIPT_DIR"

# Run pytest with verbose output and short tracebacks
python -m pytest tests/test_api.py \
    -v \
    --tb=short \
    --no-header \
    -p no:warnings \
    "$@" \
    2>&1 | tee "$LOG"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Test log saved to: $LOG"
echo "════════════════════════════════════════════════════════"
