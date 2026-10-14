#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="pretraining:pretraining/scripts:pretraining/data/json_sft_validator_20261010${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -u -m extract_068.train "$@"
