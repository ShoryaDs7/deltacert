#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# validation/engine_swap/run_flagship.sh — FLAGSHIP CASE STUDY
#
# Company scenario: the vLLM-#36117 class — "a new vLLM is out; do our
# model's outputs survive the upgrade?" (documented: v0.11->v0.12 dropped a
# model from ~60% to ~13% accuracy; teams revert upgrades because they
# "can't take time to test more").
#
# This script replicates the company workflow EXACTLY: two virtualenvs on
# one GPU (the two engine versions), the model captured in each, compared by
# the official CLI. Then GSM8K accuracy is measured in BOTH engines via
# lm-eval — the downstream truth for the same change.
#
# Usage:
#   bash validation/engine_swap/run_flagship.sh \
#        meta-llama/Llama-3.1-8B-Instruct  0.8.5  0.9.0
#   (model, vllm_version_A, vllm_version_B)
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
VLLM_A="${2:-0.8.5}"
VLLM_B="${3:-0.9.0}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CANARIES="$REPO/validation/canaries_v1.txt"

# cd out of $REPO before any `python -m deltacert...` invocation: $REPO IS the
# deltacert package directory (contains deltacert.py, cli.py, collectors.py
# directly), so running with cwd=$REPO makes Python resolve the top-level
# "deltacert" name to the sibling file deltacert/deltacert.py (a single
# module) instead of the installed package, since cwd is searched first —
# "ModuleNotFoundError: No module named 'deltacert.collectors'; 'deltacert'
# is not a package". Same collision pod_setup.sh hit; same fix (cd to parent).
cd "$(dirname "$REPO")"

echo "=== engine_swap flagship: vLLM $VLLM_A vs $VLLM_B on $MODEL ==="

echo "--- ensuring domain-tagged canary file exists ---"
python3 -c "
import sys
sys.path.insert(0, '$REPO/validation')
from flagship_common import load_canaries_with_domains
load_canaries_with_domains('$CANARIES')
"

for TAG in A B; do
  VER_VAR="VLLM_$TAG"; VER="${!VER_VAR}"
  ENV="$HERE/env_$TAG"
  if [ ! -d "$ENV" ]; then
    echo "--- building venv $TAG (vllm==$VER) ---"
    python -m venv "$ENV"
    "$ENV/bin/pip" -q install --upgrade pip
    "$ENV/bin/pip" -q install "vllm==$VER" lm-eval
    # --no-deps: this venv only needs deltacert.cli/collectors importable —
    # deltacert's own pyproject pulls a newer `transformers`, which upgrades
    # past whatever version vllm==$VER already resolved as compatible,
    # breaking older vLLM's tokenizer code (e.g. vLLM 0.8.5 expects
    # `all_special_tokens_extended` on the tokenizer, absent from newer
    # transformers' TokenizersBackend). Keep vLLM's own dependency
    # resolution intact.
    "$ENV/bin/pip" -q install -e "$REPO" --no-deps
  fi
  echo "--- capture in env $TAG ---"
  N_PROMPTS=$(grep -c . "$CANARIES")
  "$ENV/bin/python" -m deltacert.cli capture \
      --model "$MODEL" \
      --prompts "$CANARIES" \
      --n-prompts "$N_PROMPTS" \
      --output "$HERE/capture_$TAG.npz" \
      --backend vllm
  echo "--- downstream: GSM8K (lm-eval, limit 200) in env $TAG ---"
  "$ENV/bin/python" -m lm_eval --model vllm \
      --model_args "pretrained=$MODEL" \
      --tasks gsm8k --limit 200 --batch_size auto \
      --output_path "$HERE/gsm8k_env$TAG" || \
      echo "[!] lm-eval failed in env $TAG — downstream number missing"
done

echo "--- compare captures (official CLI, current env) ---"
python -m deltacert.cli certify \
    --model "$MODEL" \
    --checks engine_swap \
    --baseline "$HERE/capture_A.npz" \
    --candidate "$HERE/capture_B.npz" \
    --output "$HERE/cert_engine_swap.json" || true
test -f "$HERE/cert_engine_swap.json" || { echo "cert missing"; exit 1; }

echo "--- assemble result.json (schema-gated) ---"
python "$HERE/assemble_result.py" \
    --model "$MODEL" --ver-a "$VLLM_A" --ver-b "$VLLM_B" --here "$HERE"

echo "=== engine_swap flagship complete ==="
