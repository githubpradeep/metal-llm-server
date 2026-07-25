# Curriculum: Understanding `llama-sinks`

A study textbook for this repo. Goal: you can explain **every major subsystem**
from Metal kernels up through the HTTP scheduler — with diagrams, call graphs,
and check-yourself questions — even if much of the code was AI-assisted.

Binary name: **`llama-sinks`**. Primary target: **Gemma4 E2B / E4B** on Apple
Silicon via **GGUF + Metal**.

---

## How to use this

1. Read chapters **in order** the first time (Part I → VI).
2. For each chapter: skim the diagram → read the prose → open the cited files →
   answer the checklist at the end **without looking**.
3. **Canonical path only:** [`textbook/`](textbook/). Do not use
   `docs/archive/deprecated/` (stale numbers, CLI, and line refs).
4. Living performance diary: root [`AGENTS.md`](../AGENTS.md).

Full chapter index: [`textbook/README.md`](textbook/README.md).

---

## Syllabus (≈ 15–25 hours of serious study)

| Part | Chapters | What you should be able to do |
|------|----------|-------------------------------|
| **I — Map** | 00–01 | Draw the full stack; name every file’s job |
| **II — Model** | 02–03, 17 | Explain Gemma4 quirks; load a GGUF; know shapes |
| **III — GPU** | 04–08 | Trace one token through Metal; draw Q4_0 + KV |
| **IV — Paths** | 09–10 | Decode vs prefill; fused vs ggml; hybrid auto |
| **V — Serve** | 11–13 | Request → queue → slot → schedule → SSE |
| **VI — Advanced** | 14–16 | MTP; perf lab method; glossary + drills |

---

## Suggested weekly plan

| Day | Focus |
|-----|--------|
| 1 | Ch 00–02 (system + Gemma4 arch) |
| 2 | Ch 03 + 17 + 04–05 (weights, shapes, Metal, quant) |
| 3 | Ch 06–08 (KV, attention, MLP/PLE) |
| 4 | Ch 09–10 (decode + prefill walkthroughs) |
| 5 | Ch 11–13 (server, scheduler, pool) |
| 6 | Ch 14–15 (MTP + optimization lab) |
| 7 | Ch 16 drills + rebuild mental model from memory |

---

## Honesty note (for study)

This curriculum teaches **what the system does and why**, so you can reason about
it in interviews or debugging. It does not claim you hand-wrote every kernel.
Treat “I can redraw the diagram and explain the hybrid KV-append bug” as the bar.
