#!/usr/bin/env bash
# Engramscope own-memory test on SmolLM2-1.7B, rented Linux GPU runner (instructions: LAPTOP.md, rules: PREREG-1.7B.md).
#   bash spikes/own_memory/rent.sh check   # about 10 minutes
#   bash spikes/own_memory/rent.sh         # everything else; rerun to resume
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
v=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | head -1)
read -r major minor <<<"$v"
cuda=$((major * 100 + minor))
if   [ "$cuda" -ge 1302 ]; then torch="torch==2.15.0+cu132"; index=https://download.pytorch.org/whl/cu132
elif [ "$cuda" -ge 1300 ]; then torch="torch==2.14.1+cu130"; index=https://download.pytorch.org/whl/cu130
elif [ "$cuda" -ge 1206 ]; then torch="torch==2.14.1+cu126"; index=https://download.pytorch.org/whl/cu126
else echo "driver supports CUDA $major.$minor; pick a machine with a newer driver" >&2; exit 1; fi
echo "Driver supports CUDA $major.$minor: using $torch"
export ENGRAMSCOPE_OWNMEM="${ENGRAMSCOPE_OWNMEM:-$HOME/engramscope-ownmem}"
mkdir -p "$ENGRAMSCOPE_OWNMEM/data"
uv run --no-project --python 3.12 --with "$torch" --with transformers==5.24.0 --with peft==0.22.0 --with pyarrow \
  --with tokenizers==0.23.3 --with sympy --with numpy --with requests --with psutil \
  --index-strategy unsafe-best-match --extra-index-url "$index" python -u scale.py "$@"
