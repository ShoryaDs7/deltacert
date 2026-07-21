# Preliminary local run — Qwen2.5-3B-Instruct, fp8 KV-cache

Single-position certification passes (`certified: true`, `d_comm = 5.8273`) while
paired GSM8K drops 0.69 → 0.56 (−13 points). This is the same signature as the
feedback-driven false-safe documented on Qwen2.5-7B-Instruct in the paper (§5.5):
single-position certification passes on an fp8 KV-cache change while downstream
accuracy collapses.

**Not part of the paper's validated flagship set.** This is a single, unsigned,
local run on a smaller model, with no trajectory or free-running pass behind it —
none of the corroborating evidence the paper requires before calling something a
confirmed instance of the feedback-driven class. It is retained here, undeleted,
as a candidate data point consistent with the paper's own disclosed limitation
that the feedback-driven class may have further members "at other context
lengths, formats, or models" (§6) — not swept out because the number looks bad.
