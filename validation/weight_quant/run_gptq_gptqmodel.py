"""
GPTQ sweep for Qwen using gptqmodel (fast, proper CUDA kernels) instead of
auto_gptq (slow CPU fallback, no compiled extension in this environment).
Mirrors run_flagship.py's GPTQ row logic exactly, just swaps the quantization
backend. Meant to run in an isolated venv with gptqmodel installed.

Run:
    /root/gptq_venv/bin/python3 -u validation/weight_quant/run_gptq_gptqmodel.py \
        --model Qwen/Qwen2.5-7B-Instruct --gsm8k-n 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flagship_common import (
    gsm8k_accuracy_hf, load_canaries_with_domains, load_gsm8k, build_cert,
)
from deltacert.collectors import capture_logits_hf, cos_sims_from_logit_matrices

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gsm8k-n", type=int, default=100)
    ap.add_argument("--gptq-bits", type=int, nargs="+", default=[4, 3, 2])
    args = ap.parse_args()

    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    canaries, domains = load_canaries_with_domains()
    problems = load_gsm8k(args.gsm8k_n)

    print("=== BASELINE: fp16 ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fp16_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda:0")
    fp16_logits = capture_logits_hf("", canaries, model=fp16_model, tokenizer=tokenizer)
    fp16_acc, _ = gsm8k_accuracy_hf(fp16_model, tokenizer, problems)
    print(f"  fp16 acc = {fp16_acc:.3f}")
    del fp16_model
    torch.cuda.empty_cache()

    calibration = [c for c in canaries[:16]]  # real canary prompts as GPTQ calibration data

    for bits in sorted(set(args.gptq_bits), reverse=True):
        row_name = f"gptq_int{bits}"
        print(f"\n=== ROW: {row_name} (gptqmodel) ===")

        quant_config = QuantizeConfig(bits=bits, group_size=128)
        model = GPTQModel.load(args.model, quant_config)
        model.quantize(calibration, batch_size=1)

        # Quantization-mode model isn't fully materialized for raw forward
        # passes (meta tensors) — save + reload in inference mode first.
        save_path = os.path.join("/root/gptq_saved", row_name)
        model.save(save_path)
        del model
        torch.cuda.empty_cache()
        model = GPTQModel.load(save_path)

        print(f"  [1/2] d_COMM (per domain, worst domain certifies) …")
        q_logits = capture_logits_hf("", canaries, model=model, tokenizer=tokenizer)
        cos_sims = cos_sims_from_logit_matrices(fp16_logits, q_logits)
        cert = build_cert("weight_quant", cos_sims, args.model,
                          domain_labels=domains, quant_method="gptq")
        cert_path = os.path.join(HERE, f"cert_{row_name}.json")
        with open(cert_path, "w") as f:
            json.dump(cert, f, indent=2)
        d = cert["layers"]["weight_quant"]["d_comm"]
        certified = cert["layers"]["weight_quant"]["certified"]
        print(f"      d = {d}  certified = {certified}")

        print(f"  [2/2] downstream: GSM8K on the quantized model …")
        q_acc, records = gsm8k_accuracy_hf(model, tokenizer, problems)
        print(f"      fp16 acc = {fp16_acc:.3f}  {row_name} acc = {q_acc:.3f} "
              f"({(q_acc - fp16_acc) * 100:+.1f} pts)")
        with open(os.path.join(HERE, f"gsm8k_{row_name}.json"), "w") as f:
            json.dump({"fp16_acc": fp16_acc, f"{row_name}_acc": q_acc, "records": records}, f, indent=2)

        del model
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
