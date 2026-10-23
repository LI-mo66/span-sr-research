#!/usr/bin/env bash
# Optional only. Existing working SG6 environment is preferred and is not modified.
set -euo pipefail
NEW_ENV="${NEW_ENV:-/home/featurize/work/sg6-stripadd-env}"
[[ ! -e "$NEW_ENV" ]] || { echo "Existing environment preserved: $NEW_ENV"; exit 2; }
command -v conda >/dev/null
conda create -y -p "$NEW_ENV" python=3.11.8
"$NEW_ENV/bin/python" -m pip install numpy==1.26.4
"$NEW_ENV/bin/python" -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
"$NEW_ENV/bin/python" -m pip install opencv-python-headless==4.10.0.84 scipy==1.13.1
printf 'New interpreter: %s/bin/python\nSet PYTHON_BIN to this path before launch.\n' "$NEW_ENV"
