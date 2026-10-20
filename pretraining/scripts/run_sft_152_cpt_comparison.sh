#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:$PWD/pretraining${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
exec .venv-jsonaudit119/bin/python -u -m json_cpt_compare_152.run "$@"
