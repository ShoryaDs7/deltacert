#!/usr/bin/env bash
# validation/pod_setup.sh — one-shot environment setup for a fresh GPU pod.
#
# Bakes in every fix discovered across prior pod runs so we don't rediscover
# them live on paid GPU time:
#   1. bitsandbytes needs libnvJitLink.so.13 on LD_LIBRARY_PATH (torch's cu13
#      wheels ship it at a nonstandard path; bitsandbytes doesn't find it
#      automatically).
#   2. HF token must be copied into $HF_HOME's cache — `hf auth login`
#      defaults to ~/.cache/huggingface regardless of HF_HOME if HF_HOME is
#      set AFTER login; copy the token file over explicitly.
#   3. gptqmodel, not auto-gptq — auto-gptq 0.7.1 (latest release) imports
#      `no_init_weights` from transformers.modeling_utils, removed in
#      transformers 5.x. gptqmodel is the maintained alternative.
#   4. accelerate is a core dependency now (device_map="auto" needs it) —
#      already in pyproject.toml, but installed here explicitly as a
#      belt-and-suspenders check.
#
# Usage (from /workspace after extracting deltacert.tar.gz):
#   bash deltacert/validation/pod_setup.sh
set -euo pipefail

echo "=== [1/6] Installing deltacert + extras ==="
cd /workspace/deltacert
pip install -q -e ".[vllm,bnb,gptq,validation]"

echo "=== [2/6] Fixing bitsandbytes CUDA lib path ==="
NVJITLINK_DIR=$(dirname "$(find / -iname 'libnvJitLink.so.13' 2>/dev/null | head -1)")
if [ -z "$NVJITLINK_DIR" ] || [ "$NVJITLINK_DIR" = "." ]; then
    echo "  WARNING: libnvJitLink.so.13 not found — bitsandbytes may fail to load."
else
    echo "export LD_LIBRARY_PATH=${NVJITLINK_DIR}:\$LD_LIBRARY_PATH" >> ~/.bashrc
    export LD_LIBRARY_PATH="${NVJITLINK_DIR}:${LD_LIBRARY_PATH:-}"
    echo "  Set LD_LIBRARY_PATH -> ${NVJITLINK_DIR}"
fi
python3 -c "import bitsandbytes; print('  bitsandbytes OK', bitsandbytes.__version__)"

echo "=== [3/6] Verifying GPTQ backend (gptqmodel, not auto-gptq) ==="
python3 -c "import gptqmodel; print('  gptqmodel OK', gptqmodel.__version__)"
python3 -c "from transformers import GPTQConfig; print('  GPTQConfig OK')"
if python3 -c "import auto_gptq" 2>/dev/null; then
    echo "  WARNING: auto_gptq is also installed — uninstall it to avoid conflicts:"
    echo "    pip uninstall -y auto-gptq"
fi

echo "=== [4/6] HuggingFace auth ==="
mkdir -p /workspace/hf_cache
if [ -n "${HF_TOKEN:-}" ]; then
    hf auth login --token "$HF_TOKEN" 2>&1 | tail -3
else
    echo "  No HF_TOKEN env var set — run 'hf auth login --token <token>' manually."
fi
cp /root/.cache/huggingface/token /workspace/hf_cache/token 2>/dev/null || true
cp /root/.cache/huggingface/stored_tokens /workspace/hf_cache/stored_tokens 2>/dev/null || true
export HF_HOME=/workspace/hf_cache
echo "export HF_HOME=/workspace/hf_cache" >> ~/.bashrc
hf auth whoami || echo "  WARNING: HF auth not confirmed — gated models (Llama) will 401."

echo "=== [5/6] Core package sanity check ==="
cd /workspace  # avoid deltacert.py (the module) shadowing the deltacert package
python3 -c "
import deltacert as dc
import vllm
import lm_eval
import accelerate
import torch
print('deltacert', dc.__version__)
print('vllm', vllm.__version__)
print('accelerate', accelerate.__version__)
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
"

echo "=== [6/6] Done. Environment ready. ==="
echo "Remember to 'export HF_HOME=/workspace/hf_cache' and set LD_LIBRARY_PATH"
echo "in every NEW shell session (not just this one) before running tests —"
echo "source ~/.bashrc or start a fresh SSH session to pick both up."
