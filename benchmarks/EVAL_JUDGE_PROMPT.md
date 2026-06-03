# How to Use: LLM-as-a-Judge Evaluation

## Quick Steps

1. Start your server: `cargo run --release -- --gpu --serve <model_dir>`
2. Run the eval: `python3 benchmarks/eval_run.py`
3. Copy the contents of `benchmarks/eval_outputs.json`
4. Paste into Kiro chat with this message:

---

## Paste this into Kiro chat along with the eval_outputs.json:

```
Judge the following Gemma4 E4B model outputs. For each response:

1. Score 1-10 on these dimensions:
   - **Correctness**: Is the answer factually/logically correct?
   - **Completeness**: Does it address all parts of the prompt?
   - **Clarity**: Is it well-organized and easy to understand?
   - **Instruction Following**: Does it respect all constraints (format, length, tone)?
   - **Helpfulness**: Would a real user find this useful?

2. Give a brief 1-2 sentence justification per response.

3. At the end, provide:
   - Per-category average scores
   - Overall average score
   - Top 3 strengths of this model
   - Top 3 weaknesses / failure modes
   - Verdict: is this model usable for production chat at this size?

Use the rubric provided for each prompt to calibrate your scoring.
Be strict but fair — this is a 4B parameter model with Q4 quantization.

Here are the outputs:
[paste eval_outputs.json contents here]
```

---

## Scoring Guide

| Score | Meaning |
|-------|---------|
| 9-10  | Excellent — correct, complete, well-written, follows all instructions |
| 7-8   | Good — mostly correct with minor issues or omissions |
| 5-6   | Acceptable — gets the main point but has notable gaps or errors |
| 3-4   | Poor — significant errors, incomplete, or misses the point |
| 1-2   | Failure — wrong answer, incoherent, or refuses inappropriately |

## What This Measures

- **Writing**: Can it produce well-structured, tone-appropriate text?
- **Reasoning**: Can it solve logic puzzles and avoid common traps?
- **Math**: Can it do step-by-step computation correctly?
- **Coding**: Can it write correct, idiomatic code and spot bugs?
- **Extraction**: Can it structure information from unstructured text?
- **STEM**: Does it have accurate technical knowledge?
- **Humanities**: Does it have accurate historical/philosophical knowledge?
- **Instruction Following**: Can it follow complex formatting constraints?
- **Safety**: Does it handle sensitive requests appropriately?
- **Conversation**: Can it maintain coherence across multiple turns?
