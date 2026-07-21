"""
validation/flagship_common.py — shared utilities for the 6 flagship case studies.

Everything here is a REAL measurement helper — no mocks, no simulation:
  * GSM8K exact-match accuracy (dataset from HF hub, greedy decode, standard
    final-number extraction — same protocol lm-eval uses for gsm8k)
  * Perplexity on wikitext-2 (standard sliding-window protocol)
  * lm-eval subprocess wrapper for the short_eval number (industry-standard
    harness — flagships shell out to the same tool companies already trust)
  * deterministic seeding + environment stamping
  * ValidationResult assembly through the harness (schema-gated)

Used by: weight_quant, kv_cache_quant, spec_decoding, engine_swap,
trajectory, provider_drift flagship scripts.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
GSM8K_FEWSHOT = 5
SEED = 1234


# ─────────────────────────────────────────────────────────────────────────────
# Determinism + environment stamp
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_gpu_memory_free(min_free_gb: float = 70.0, timeout_s: float = 90.0,
                             poll_interval_s: float = 2.0) -> None:
    """Block until at least min_free_gb of GPU memory is actually free, or
    timeout. vLLM v1 spawns a separate EngineCore PROCESS per engine — a
    Python-side `del llm; gc.collect(); torch.cuda.empty_cache()` does not
    guarantee that child process has finished shutting down and released its
    CUDA memory yet. Starting a second engine too soon fails with
    'Free memory ... is less than desired GPU memory utilization'. Poll real
    nvidia-smi output instead of trusting Python-side cleanup timing."""
    import subprocess
    import time as _time
    deadline = _time.time() + timeout_s
    while _time.time() < deadline:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=10,
            )
            free_mb = float(out.stdout.strip().splitlines()[0])
            if free_mb / 1024.0 >= min_free_gb:
                return
        except Exception:
            pass
        _time.sleep(poll_interval_s)
    print(f"    [!] wait_for_gpu_memory_free: {min_free_gb}GB not free after "
          f"{timeout_s}s — proceeding anyway, next engine start may fail.")


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def environment_stamp() -> dict:
    stamp = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "seed": SEED,
    }
    for mod in ("torch", "transformers", "vllm", "bitsandbytes", "datasets"):
        try:
            m = __import__(mod)
            stamp[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            stamp["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return stamp


# ─────────────────────────────────────────────────────────────────────────────
# GSM8K exact-match — the downstream-truth workhorse
# Protocol matches lm-eval's gsm8k task: few-shot CoT, greedy, extract the
# final number after '####' (gold) / last number in generation (prediction).
# ─────────────────────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\$?[\d,]*\.?\d+")


def extract_final_number(text: str) -> Optional[str]:
    """Last number in the text, normalized (commas/$ stripped, trailing .0
    dropped). Same convention as standard GSM8K scoring."""
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    val = matches[-1].replace(",", "").replace("$", "")
    if val.endswith("."):
        val = val[:-1]
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return None


def gold_answer(gsm8k_answer_field: str) -> str:
    """GSM8K gold answers end in '#### <number>'."""
    ans = gsm8k_answer_field.split("####")[-1].strip()
    return extract_final_number(ans) or ans


def load_gsm8k(n_problems: int, split: str = "test") -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rng = random.Random(SEED)
    idx = rng.sample(range(len(ds)), min(n_problems, len(ds)))
    return [{"question": ds[i]["question"], "gold": gold_answer(ds[i]["answer"])}
            for i in sorted(idx)]


def build_gsm8k_prompt(question: str, fewshot: Optional[List[dict]] = None) -> str:
    parts = []
    for ex in (fewshot or []):
        parts.append(f"Question: {ex['question']}\nAnswer: {ex['cot']}\n")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n".join(parts)


def load_gsm8k_fewshot(k: int = GSM8K_FEWSHOT) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(SEED + 1)
    idx = rng.sample(range(len(ds)), k)
    return [{"question": ds[i]["question"], "cot": ds[i]["answer"].split("####")[0].strip()
             + f" The answer is {gold_answer(ds[i]['answer'])}."} for i in idx]


def gsm8k_accuracy_hf(
    model, tokenizer, problems: List[dict], device: str = "cuda",
    max_new_tokens: int = 256, batch_log_every: int = 10,
) -> Tuple[float, List[dict]]:
    """Greedy-decode each problem, score exact match. Returns (accuracy,
    per-problem records) so failures are auditable."""
    import torch
    fewshot = load_gsm8k_fewshot()
    records = []
    correct = 0
    model.eval()
    for i, prob in enumerate(problems):
        prompt = build_gsm8k_prompt(prob["question"], fewshot)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=3584).to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        gen = gen.split("Question:")[0]
        pred = extract_final_number(gen)
        ok = pred is not None and pred == prob["gold"]
        correct += int(ok)
        records.append({"question": prob["question"][:80], "gold": prob["gold"],
                        "pred": pred, "correct": ok})
        if (i + 1) % batch_log_every == 0:
            print(f"    gsm8k {i+1}/{len(problems)} acc so far "
                  f"{correct/(i+1):.3f}")
    return correct / len(problems), records


def gsm8k_accuracy_vllm(llm, problems: List[dict],
                        max_new_tokens: int = 256) -> Tuple[float, List[dict], float]:
    """Same protocol through a vLLM engine. Also returns measured tokens/sec
    (the business-gain number for spec_decoding)."""
    import vllm
    fewshot = load_gsm8k_fewshot()
    prompts = [build_gsm8k_prompt(p["question"], fewshot) for p in problems]
    params = vllm.SamplingParams(temperature=0.0, max_tokens=max_new_tokens,
                                 stop=["Question:"])
    t0 = time.time()
    outs = llm.generate(prompts, params)
    dt = time.time() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    toks_per_sec = total_tokens / dt if dt > 0 else 0.0
    records, correct = [], 0
    for prob, out in zip(problems, outs):
        pred = extract_final_number(out.outputs[0].text)
        ok = pred is not None and pred == prob["gold"]
        correct += int(ok)
        records.append({"gold": prob["gold"], "pred": pred, "correct": ok,
                        "text": out.outputs[0].text})
    return correct / len(problems), records, toks_per_sec


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity — wikitext-2, standard sliding-window protocol
# ─────────────────────────────────────────────────────────────────────────────

def perplexity_wikitext(model, tokenizer, device: str = "cuda",
                        n_tokens: int = 32768, stride: int = 2048,
                        max_length: int = 2048) -> float:
    import torch
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                    split="test")["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][:, :n_tokens]
    nlls, count = [], 0
    model.eval()
    for begin in range(0, input_ids.shape[1] - 1, stride):
        end = min(begin + max_length, input_ids.shape[1])
        ids = input_ids[:, begin:end].to(device)
        with torch.inference_mode():
            out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        nlls.append(out.loss.float() * n)
        count += n
        if end == input_ids.shape[1]:
            break
    import math as _m
    return float(_m.exp(sum(float(x) for x in nlls) / count))


# ─────────────────────────────────────────────────────────────────────────────
# lm-eval subprocess wrapper — the short_eval number
# ─────────────────────────────────────────────────────────────────────────────

def run_lm_eval(model_args: str, tasks: str, out_json: str,
                limit: Optional[int] = 200, extra: Sequence[str] = ()) -> Optional[dict]:
    """Shell out to lm-eval (pip install lm-eval). Returns the results dict,
    or None with a loud message if lm-eval is absent — the flagship then
    records no short_eval block rather than a fake one."""
    cmd = [sys.executable, "-m", "lm_eval", "--model", "hf",
           "--model_args", model_args, "--tasks", tasks,
           "--batch_size", "1", "--output_path", out_json]
    if limit:
        cmd += ["--limit", str(limit)]
    cmd += list(extra)
    print("    running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"    [!] lm-eval unavailable/failed ({e}). short_eval will be "
              "omitted from result.json — install lm-eval to claim result E.")
        return None
    candidates = [out_json] + [os.path.join(out_json, f)
                               for f in (os.listdir(out_json) if os.path.isdir(out_json) else [])]
    for c in candidates:
        if os.path.isfile(c) and c.endswith(".json"):
            with open(c, encoding="utf-8") as f:
                return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Cert building — wraps the real collectors.py functions
# Replaces the non-existent _make_certificate that the original scripts assumed.
# ─────────────────────────────────────────────────────────────────────────────

def build_cert(layer_name: str, cos_sims: list, model_id: str,
               domain_labels: Optional[list] = None,
               quant_method: Optional[str] = None) -> dict:
    """Build a canonical deltacert certificate dict from cos_sims.

    Without domain_labels: uses certify_from_layers + _make_layer_result
    from collectors.py (unchanged, blended d_comm), same as before.

    With domain_labels (index-aligned with cos_sims): routes through
    deltacert.certify_layer(), which certifies on the WORST domain rather
    than the blended average — see certify_layer()'s docstring.

    quant_method: optional, e.g. "bnb" or "gptq" — see certify_layer()'s
    _PROVISIONAL_METHOD_BUDGETS docstring for why a single universal tau
    produced a false negative on a real, harmless bnb int8 config.
    """
    from deltacert.collectors import certify_from_layers, _make_layer_result
    if domain_labels is None:
        layer = _make_layer_result(cos_sims)
    else:
        from deltacert.deltacert import certify_layer
        layer = certify_layer(cos_sims, domain_labels=domain_labels,
                              quant_method=quant_method)
        layer["n_prompts"] = layer["n_samples"]
        layer["cos_sim_min"] = float(min(cos_sims))
        layer["cos_sim_mean"] = float(sum(cos_sims) / len(cos_sims))
    return certify_from_layers(model_id, {layer_name: layer})


# ─────────────────────────────────────────────────────────────────────────────
# KV quant context manager — applies int8/int4 hooks during generation
# for the downstream measurement in kv_cache_quant flagship.
# Mirrors the hook logic in collect_kv_cache_quant without calling it.
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def kv_quant_context(model, mode: str = "int8"):
    """Attach KV cache quantization forward hooks during generation.
    Hooks k_proj and v_proj outputs — same projection points as the collector.
    Restores clean model state on exit regardless of exceptions."""
    import torch

    if mode == "int8":
        levels = 127
    elif mode == "int4":
        levels = 7
    else:
        raise ValueError(f"kv_quant_context: unknown mode '{mode}'. Use 'int8' or 'int4'.")

    def compress(t):
        orig_dtype = t.dtype
        scale = t.float().abs().max() / levels + 1e-8
        return (t.float() / scale).round().clamp(-levels, levels).to(torch.int8), scale, orig_dtype

    def decompress(packed):
        q, scale, dtype = packed
        return q.to(dtype) * scale.to(dtype)

    kv_modules = [mod for name, mod in model.named_modules()
                  if any(name.endswith(k) for k in ("k_proj", "v_proj"))]
    handles = []
    for mod in kv_modules:
        def _hook(m, inp, out, _c=compress, _d=decompress):
            return _d(_c(out))
        handles.append(mod.register_forward_hook(_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Cert reading + result assembly (through the schema-gated harness)
# ─────────────────────────────────────────────────────────────────────────────

def read_cert_d(cert_path: str, layer_name: str) -> float:
    import math
    with open(cert_path, encoding="utf-8") as f:
        cert = json.load(f)
    layer = cert["layers"][layer_name]
    d = layer["d_comm"]
    return math.inf if d == "inf" else float(d)


def assemble_and_save_result(**kwargs) -> str:
    """Thin wrapper so every flagship records through the same schema gate."""
    from deltacert.validation.harness import ValidationResult, save_result
    out_path = kwargs.pop("out_path")
    result = ValidationResult(**kwargs)
    save_result(result, out_path)
    print(f"    result.json written (schema-validated): {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Real, hardcoded natural-language prompts for the two domains with no
# convenient single-repo HF dataset (multilingual, chat). These are genuine
# real-language utterances, not synthetic filler — legitimate canary prompts
# same as any other domain, just authored instead of pulled from a named
# benchmark dataset.
# ─────────────────────────────────────────────────────────────────────────────

_MULTILINGUAL_PROMPTS = [
    "Quelle est la capitale de la France et pourquoi est-elle importante ?",
    "Explique-moi la théorie de la relativité en termes simples.",
    "¿Cuáles son las principales causas del cambio climático?",
    "Escribe un breve resumen sobre la revolución industrial.",
    "Wie funktioniert die Photosynthese bei Pflanzen?",
    "Was sind die wichtigsten Grundsätze der Demokratie?",
    "भारत की राजधानी क्या है और यह ऐतिहासिक रूप से क्यों महत्वपूर्ण है?",
    "मशीन लर्निंग क्या है, इसे सरल शब्दों में समझाइए।",
    "人工智能的主要应用领域有哪些？",
    "请解释一下什么是量子计算。",
    "気候変動の主な原因は何ですか？",
    "日本の伝統文化について簡単に説明してください。",
    "Qual è la differenza tra intelligenza artificiale e machine learning?",
    "Descrivi brevemente la storia dell'Impero Romano.",
    "Как работает блокчейн простыми словами?",
    "Расскажите о главных причинах Первой мировой войны.",
    "O que é a teoria da evolução de Darwin?",
    "Explique brevemente como funciona a internet.",
    "Wat zijn de belangrijkste oorzaken van inflatie?",
    "Leg uit hoe een computer processor werkt.",
    "Quels sont les effets de la déforestation sur la biodiversité ?",
    "Décris brièvement le fonctionnement du système immunitaire humain.",
    "¿Cómo afecta el envejecimiento de la población a la economía?",
    "Explica de manera sencilla qué es la inteligencia artificial.",
    "Welche Rolle spielen erneuerbare Energien für die Zukunft?",
    "Beschreibe kurz, wie Impfstoffe funktionieren.",
    "प्रकाश संश्लेषण की प्रक्रिया को सरल शब्दों में समझाएं।",
    "अंतरिक्ष अन्वेषण के मुख्य लाभ क्या हैं?",
    "全球变暖对海洋生态系统有什么影响？",
    "简要描述一下丝绸之路的历史意义。",
    "デジタル技術が教育に与える影響について説明してください。",
    "再生可能エネルギーの主な種類を教えてください。",
    "Quali sono i principali vantaggi dell'energia solare?",
    "Spiega brevemente come funziona il sistema immunitario.",
    "Каковы основные причины экономической инфляции?",
    "Опишите кратко, как работают вакцины.",
    "Quais são os principais desafios da urbanização moderna?",
    "Explique como funciona o sistema solar de forma simples.",
    "Hoe beïnvloedt kunstmatige intelligentie de arbeidsmarkt?",
    "Beschrijf in het kort de werking van het menselijk brein.",
    "Quelles sont les conséquences économiques du changement climatique ?",
    "Explique brièvement comment fonctionne un moteur électrique.",
    "¿Qué papel juega la educación en el desarrollo económico?",
    "Describe brevemente cómo funciona la memoria humana.",
    "Was verursacht Erdbeben und wie können sie vorhergesagt werden?",
    "Erkläre kurz, was eine Pandemie von einer Epidemie unterscheidet.",
    "योग के स्वास्थ्य पर क्या लाभ होते हैं?",
    "जलवायु परिवर्तन को रोकने के उपाय क्या हैं?",
    "什么是区块链技术，它有哪些实际应用？",
    "简述一下文艺复兴时期的主要成就。",
]

_CHAT_PROMPTS = [
    "Can you help me plan a relaxing weekend trip to the mountains?",
    "What's a good, simple recipe for a weeknight pasta dinner?",
    "I'm feeling stressed about a job interview tomorrow, any advice?",
    "What are some fun icebreaker questions for a team meeting?",
    "Can you suggest a few books similar to Dune for a sci-fi fan?",
    "How should I start a polite email to a professor I've never met?",
    "What's a thoughtful gift for a friend who just moved into a new apartment?",
    "Can you help me write a birthday message for my sister?",
    "What are some easy exercises I can do at home without equipment?",
    "I want to start journaling — how do I get into the habit?",
    "What's a good way to organize a small book club with friends?",
    "Can you recommend a beginner-friendly hobby I could pick up this year?",
    "How do I politely decline a social invitation without being rude?",
    "What questions should I ask when adopting a rescue dog?",
    "Can you help me come up with a fun theme for a birthday party?",
    "What's a good way to start a conversation with someone new at a party?",
    "How can I make my apartment feel cozier on a small budget?",
    "What are some tips for staying motivated to exercise regularly?",
    "Can you suggest a relaxing playlist mood for a rainy afternoon?",
    "How do I write a thank-you note after a job interview?",
    "What's a good way to break in new running shoes without blisters?",
    "Can you suggest a few low-maintenance houseplants for a beginner?",
    "How do I start learning to cook without feeling overwhelmed?",
    "What's a polite way to ask a coworker to keep their music down?",
    "Can you help me plan a surprise anniversary dinner at home?",
    "What are some good conversation starters for a first date?",
    "How do I ask my landlord about fixing a leaky faucet politely?",
    "What's a fun, low-cost weekend activity for a family with kids?",
    "Can you suggest ways to make a long car trip more enjoyable?",
    "How do I write a friendly reminder email without sounding pushy?",
    "What are some good stretches to do before going for a run?",
    "Can you help me think of a creative name for a new podcast?",
    "What's a thoughtful way to check in on a friend going through a hard time?",
    "How do I organize a small potluck dinner with friends?",
    "What are some tips for packing efficiently for a week-long trip?",
    "Can you suggest a comforting meal for a cold winter evening?",
    "How do I politely ask a group chat to stop notifications at night?",
    "What's a good way to introduce myself on the first day of a new job?",
    "Can you help me brainstorm ideas for a small home office setup?",
    "What are some easy ways to reduce food waste at home?",
    "How do I write a heartfelt congratulations message for a graduate?",
    "What's a good way to unwind after a long, stressful day at work?",
    "Can you suggest a beginner-friendly board game for family game night?",
    "How do I ask a friend to help me move without it feeling like a big favor?",
    "What's a nice way to thank a mentor who helped me a lot this year?",
    "Can you help me plan a simple, budget-friendly picnic in the park?",
    "What are some good questions to ask on a house tour before renting?",
    "How do I start a small vegetable garden on an apartment balcony?",
    "What's a kind way to tell a friend I need some space right now?",
    "Can you suggest a relaxing evening routine to help me sleep better?",
]

# Domain -> how many canary prompts to draw from it when building fresh
_DOMAIN_TARGET_N = {
    "math": 50,
    "code": 50,
    "multilingual": 50,
    "chat": 50,
    "general": 50,
}


def load_canaries_with_domains(
    path: str = "validation/canaries_v1.txt",
) -> Tuple[List[str], List[str]]:
    """The shared domain-tagged canary set: (prompts, domains), index-aligned.

    5 domains — math, code, multilingual, chat, general — so certification
    can be stratified per-domain (worst-domain d_comm, not a blended average
    that can hide a domain-specific regression). If the file doesn't exist
    yet, build it ONCE from real sources and save it as `domain<TAB>prompt`
    lines — after that it is frozen and shipped in the repo.
    """
    if os.path.exists(path):
        prompts, domains = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                dom, _, text = line.rstrip("\n").partition("\t")
                prompts.append(text)
                domains.append(dom)
        return prompts, domains

    print(f"    canary file absent — building {path} from real sources (one-time).")
    from datasets import load_dataset
    rng = random.Random(SEED)

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    math_prompts = [gsm[i]["question"]
                     for i in rng.sample(range(len(gsm)), _DOMAIN_TARGET_N["math"])]

    humaneval = load_dataset("openai/openai_humaneval", split="test")
    code_prompts = [humaneval[i]["prompt"]
                     for i in rng.sample(range(len(humaneval)), _DOMAIN_TARGET_N["code"])]

    wiki = [t.strip() for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                                            split="test")["text"]
            if 120 < len(t.strip()) < 400]
    general_prompts = rng.sample(wiki, _DOMAIN_TARGET_N["general"])

    multilingual_prompts = rng.sample(
        _MULTILINGUAL_PROMPTS, min(_DOMAIN_TARGET_N["multilingual"], len(_MULTILINGUAL_PROMPTS)))
    chat_prompts = rng.sample(
        _CHAT_PROMPTS, min(_DOMAIN_TARGET_N["chat"], len(_CHAT_PROMPTS)))

    rows: List[Tuple[str, str]] = (
        [("math", p) for p in math_prompts]
        + [("code", p) for p in code_prompts]
        + [("multilingual", p) for p in multilingual_prompts]
        + [("chat", p) for p in chat_prompts]
        + [("general", p) for p in general_prompts]
    )
    rng.shuffle(rows)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{dom}\t{text.replace(chr(10), ' ').replace(chr(9), ' ')}"
                          for dom, text in rows))

    prompts = [text for _, text in rows]
    domains = [dom for dom, _ in rows]
    return prompts, domains


def load_canaries(path: str = "validation/canaries_v1.txt") -> List[str]:
    """Backward-compatible: prompts only, no domain labels. Prefer
    load_canaries_with_domains() for new code so certification can be
    stratified per domain."""
    prompts, _ = load_canaries_with_domains(path)
    return prompts
