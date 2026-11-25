#!/usr/bin/env bash
# Engramscope own-memory test on SmolLM2-1.7B, runner for a rented Linux GPU (instructions: LAPTOP.md, rules: PREREG-1.7B.md).
#   bash rent.sh start   runs everything in the background and shows progress. Closing the terminal doesn't stop it.
#                        Run it again after a stop or a pod restart: finished steps are skipped and training resumes.
#   bash rent.sh watch   shows progress again
#   bash rent.sh send    after the run: packs the results and prints a one-time code to paste into the chat
#   bash rent.sh check   (or any other scale.py step) runs that step in the foreground
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"

# Keep everything under /workspace when it exists: RunPod keeps it when a pod restarts but wipes the container disk.
ROOT="${RENT_ROOT:-/workspace}"
if [ ! -d "$ROOT" ] || [ ! -w "$ROOT" ]; then ROOT="$HOME"; fi
export ENGRAMSCOPE_OWNMEM="${ENGRAMSCOPE_OWNMEM:-$ROOT/engramscope-ownmem}"
export HF_HOME="${HF_HOME:-$ROOT/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROOT/uv-python}"
WORK="$ENGRAMSCOPE_OWNMEM"
LOG="$WORK/run.log"
LOCK="$WORK/run.lock"
PIDFILE="$WORK/run.pid"
RES="$WORK/scale"
if [ "${SCALE_SMOKE:-}" = 1 ]; then RES="$WORK/scale-smoke"; fi
RUNPODCTL_URL=https://github.com/runpod/runpodctl/releases/download/v3.1.1/runpodctl-linux-amd64
RUNPODCTL_SHA=ae0a2e5ff723c8fceb609e7120dfcaa4d681bf3015adc74141812a21f2bb9486
mkdir -p "$WORK/data"

pick_torch() {  # the torch build follows the driver's CUDA version (PREREG-1.7B.md, Environment)
  if [ -n "${RENT_TORCH:-}" ]; then  # code tests without a GPU only
    torch="$RENT_TORCH"
    index="${RENT_INDEX:?set RENT_INDEX together with RENT_TORCH}"
    return
  fi
  if ! command -v nvidia-smi >/dev/null; then
    echo "No NVIDIA driver found (nvidia-smi is missing). This needs a machine with an NVIDIA GPU." >&2
    exit 1
  fi
  local v major minor cuda
  v=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | head -1)
  if [ -z "$v" ]; then
    echo "Could not read the CUDA version from nvidia-smi." >&2
    exit 1
  fi
  read -r major minor <<<"$v"
  cuda=$((major * 100 + minor))
  if   [ "$cuda" -ge 1302 ]; then torch="torch==2.15.0+cu132"; index=https://download.pytorch.org/whl/cu132
  elif [ "$cuda" -ge 1300 ]; then torch="torch==2.14.1+cu130"; index=https://download.pytorch.org/whl/cu130
  elif [ "$cuda" -ge 1206 ]; then torch="torch==2.14.1+cu126"; index=https://download.pytorch.org/whl/cu126
  else
    echo "The driver supports CUDA $major.$minor, but this needs 12.6 or newer. Pick a machine with a newer driver." >&2
    exit 1
  fi
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1); CUDA $major.$minor, so $torch"
}

check_space() {  # about 20 GB in total: PyTorch, the 3.4 GB model, the data and the checkpoints
  local need_kb=$((25 * 1024 * 1024)) used_kb=0 free_kb d
  for d in "$WORK" "$HF_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"; do
    if [ -d "$d" ]; then used_kb=$((used_kb + $(du -sk "$d" | cut -f1))); fi
  done
  free_kb=$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')
  if [ $((free_kb + used_kb)) -lt "$need_kb" ]; then
    echo "Only $((free_kb / 1048576)) GB is free in $ROOT, and the test needs about 25 GB. Give the pod a 40 GB volume disk." >&2
    exit 1
  fi
}

run_scale() {
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  pick_torch
  echo "files under $ROOT"
  uv run --no-project --python 3.12 --with "$torch" --with transformers==5.24.0 --with peft==0.22.0 --with pyarrow \
    --with tokenizers==0.23.3 --with sympy --with numpy --with requests --with psutil \
    --index-strategy unsafe-best-match --extra-index-url "$index" python -u scale.py "$@"
}

watch_run() {
  local pid=""
  if [ -f "$PIDFILE" ]; then pid=$(cat "$PIDFILE"); fi
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "(Progress appears about once a minute. You can close this terminal at any time; the test keeps running.)"
    tail -n 30 --pid="$pid" -F "$LOG"
  elif [ -f "$LOG" ]; then
    tail -n 30 "$LOG"
  fi
  echo
  if [ -f "$RES/summary.json" ]; then
    echo "Finished. Next, run: bash $HERE/rent.sh send"
  elif [ ! -f "$LOG" ]; then
    echo "The test hasn't been started here yet. To start it: bash $HERE/rent.sh start"
  elif flock -n "$LOCK" true; then
    echo "The test isn't running. If it stopped early, the reason is in the lines above. To resume: bash $HERE/rent.sh start"
  fi
}

start() {
  pick_torch
  check_space
  if ! flock -n "$LOCK" true; then
    echo "The test is already running. Showing its progress."
    watch_run
    return
  fi
  echo "===== start $(date -u '+%Y-%m-%d %H:%M UTC') =====" >>"$LOG"
  # setsid puts the run in its own session, so closing the browser terminal doesn't end it; flock stops a second copy.
  setsid nohup bash -c 'echo $$ >"$1"; exec flock -n "$2" bash ./rent.sh all' _ "$PIDFILE" "$LOCK" >>"$LOG" 2>&1 </dev/null &
  sleep 2
  echo "Started in the background. The full log is $LOG"
  echo "To see progress later, open a terminal and run: bash $HERE/rent.sh watch"
  if [ -t 1 ]; then watch_run; fi
}

send_results() {
  if ! flock -n "$LOCK" true; then echo "Note: the test is still running, so this sends the results so far."; fi
  if [ ! -f "$RES/summary.json" ]; then echo "Note: there is no summary yet, so this sends the partial results."; fi
  local out="$WORK/ownmem-1.7B-results.tar.gz" bin="$ROOT/bin/runpodctl-3.1.1" items=() f
  for f in "$(basename "$RES")" run.log; do
    if [ -e "$WORK/$f" ]; then items+=("$f"); fi
  done
  if [ "${#items[@]}" -eq 0 ]; then
    echo "There are no results in $WORK yet." >&2
    exit 1
  fi
  tar -czf "$out" -C "$WORK" --exclude='*.pt' --exclude='*.tmp' "${items[@]}"
  if [ ! -x "$bin" ]; then  # the same pinned version the receiving side uses
    mkdir -p "$ROOT/bin"
    curl -fsSL -o "$bin.part" "$RUNPODCTL_URL"
    echo "$RUNPODCTL_SHA  $bin.part" | sha256sum -c --quiet -
    chmod +x "$bin.part"
    mv "$bin.part" "$bin"
  fi
  echo "Paste the code below into the chat, and keep this terminal open until I confirm I have the files."
  "$bin" send "$out"
}

case "${1:-all}" in
  start) start ;;
  watch) watch_run ;;
  send) send_results ;;
  *) run_scale "$@" ;;
esac
