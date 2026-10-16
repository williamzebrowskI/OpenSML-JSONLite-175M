#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/pretraining${PYTHONPATH:+:$PYTHONPATH}"
exec .venv-jsonaudit119/bin/python -u pretraining/scripts/run_sft_128_no_prune.py
