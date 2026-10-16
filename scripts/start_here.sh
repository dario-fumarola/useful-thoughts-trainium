#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/verify_package.py
mkdir -p local/checks
OUT="$(mktemp -d local/checks/bootstrap.XXXXXX)"
printf 'Writing local checks to %s\n' "$OUT"
python -m weekend_thinking.hardware --out "$OUT/host.json"
if [[ ! -f configs/site.local.json ]]; then
  cp configs/site.example.json configs/site.local.json
fi
python - <<'PY'
import importlib.util,sys
required=('torch','numpy','pytest')
missing=[x for x in required if importlib.util.find_spec(x) is None]
if missing:
    sys.exit('Missing helper dependencies: '+', '.join(missing)+'. Activate the working Neuron environment. No packages have been installed.')
PY
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q 2>&1 | tee "$OUT/tests.txt"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/cpu_smoke.py --out "$OUT/toy_smoke.json" > "$OUT/toy_smoke.txt"
printf '\nLocal checks complete; no Neuron training has been launched.\n'
printf 'Resolve actual paths/cutoff in configs/site.local.json, then run:\n  bash scripts/start_codex.sh\n'
