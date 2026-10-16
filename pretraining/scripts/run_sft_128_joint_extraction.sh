#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/pretraining${PYTHONPATH:+:$PYTHONPATH}"
exec .venv-jsonaudit119/bin/python -u -m native_json_128.train "$@"
