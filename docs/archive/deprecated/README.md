# Deprecated docs (do not study)

These files were moved out of the student path on 2026-07-24.

They contain **stale defaults, wrong throughput numbers, outdated CLI, and
drifted line numbers**. They are kept only as archaeological notes.

**Canonical docs:** [`../../CURRICULUM.md`](../../CURRICULUM.md) and
[`../../textbook/`](../../textbook/).

**Living perf diary:** [`../../../AGENTS.md`](../../../AGENTS.md).

| File | Why it was wrong for study |
|------|----------------------------|
| `01_engine_overview.md` | Old tok/s, CLI (`serve --model`), pool defaults |
| `02_decode_path.md` | Useful call-graph shape; line numbers stale |
| `03_tensor_shapes.md` | Shapes mostly OK; ctx cap 4096 / Q4-only story outdated |
| `04_kv_cache.md` | Layout math OK; some throughput tables ancient |
| `05_quantization.md` | Q4_0-centric; GGUF K-quant is primary now |
| `06_metal_execution.md` | Dispatch counts/times from old fusion config |
| `08_bottleneck_analysis.md` | Pre–hybrid-auto / pre–PLE-f16 numbers |
| `ARCHITECTURE.md` | Llama 3.2 16-layer ASCII |
| `BLOG.md` / `BLOG_GEMMA4.md` | Narrative snapshots from early product stage |

Do not link these from README or textbook.
