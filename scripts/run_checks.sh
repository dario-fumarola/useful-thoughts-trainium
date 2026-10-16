#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p local/checks
OUT="$(mktemp -d local/checks/tests.XXXXXX)"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q 2>&1 | tee "$OUT/tests.txt"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/cpu_smoke.py --out "$OUT/toy_smoke.json" > "$OUT/toy_smoke.txt"
python -m compileall -q weekend_thinking scripts tests
printf '\nCPU checks complete. This does not qualify Neuron training or distributed resume.\nLogs: %s\n' "$OUT"
