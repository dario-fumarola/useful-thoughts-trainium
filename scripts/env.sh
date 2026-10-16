#!/usr/bin/env bash
# Source this file to reuse the installed stack without modifying it.
export PATH="/home/ubuntu/ema-trainium/.venv/bin:/opt/aws/neuron/bin:$PATH"
export PYTHONPATH="/home/ubuntu/dario:/home/ubuntu/dario/local/test-deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
