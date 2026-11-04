#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../scripts/env.sh"
cd "$STREETNVS_REPO"
UV="$STREETNVS_RUNTIME/bootstrap/bin/uv"
if [[ ! -x "$UV" || ! -x .venv/bin/python ]]; then
    echo 'Run bash scripts/setup.sh first.' >&2
    exit 1
fi
"$UV" venv --python 3.11 .venv-data
"$UV" pip install --python .venv-data/bin/python -r data_processing/requirements.txt
.venv-data/bin/python data_processing/setup_dependencies.py waymo
"$UV" pip install --python .venv/bin/python 'plyfile==1.1' 'imageio==2.37.0' \
    'imageio-ffmpeg==0.8.1' setuptools wheel ninja
.venv/bin/python data_processing/setup_dependencies.py rasterizer
"$UV" pip install --python .venv/bin/python --no-build-isolation --no-deps \
    "$STREETNVS_RUNTIME/sources/diff-point-rasterization"
printf 'Ready: .venv-data for raw extraction; .venv for rendering and inference.\n'
