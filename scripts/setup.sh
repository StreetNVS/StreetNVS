#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$STREETNVS_REPO"
python3 -m venv "$STREETNVS_RUNTIME/bootstrap"
"$STREETNVS_RUNTIME/bootstrap/bin/python" -m pip install 'uv==0.13.7'
UV="$STREETNVS_RUNTIME/bootstrap/bin/uv"
"$UV" venv --python 3.11 .venv
"$UV" pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 'torch==2.9.1' 'torchvision==0.24.1'
"$UV" pip install --python .venv/bin/python -e .
printf 'Ready: source scripts/env.sh && source .venv/bin/activate\n'
