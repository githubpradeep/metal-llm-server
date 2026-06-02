"""
Compare outputs from llama.cpp reference vs local server.

Usage:
    python3 benchmarks/compare_outputs.py [reference_outputs.json] [local_outputs.json]

This compares greedy (temperature=0) outputs to verify implementation correctness.
"""

import json
import sys
import os
from difflib import SequenceMatcher

script_dir = os.path.dirname(os.path.abspath(__file__))

ref_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "reference_outputs.json")
local_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(script_dir, "local_outputs.json")

if not os.path.exists(ref_path):
    print(f"ERROR: Reference file not found: {ref_path}")
    print("Run the Colab benchmark first and download reference_outputs.json")
    sys.exit(1)

if not os.path.exists(local_path):
    print(f"ERROR: Local file not found: {local_path}")
    print("Run: python3 benchmarks/benchmark_local.py")
    sys.exit(1)

with open(ref_path) as f:
    ref_data = json.load(f)
with open(local_path) as f:
    local_data = json.load(f)

ref_results = {r["id"]: r for r in ref_data["results"]}
local_results = {r["id"]: r for r in local_data["results"]}

print("=" * 70)
print("GEMMA4 E4B INFERENCE QUALITY COMPARISON")
print("=" * 70)
print(f"Reference: {ref_data['engine']} ({ref_data['throughput_tok_s']} tok/s)")
print(f"Local:     {local_data['engine']} ({local_data['throughput_tok_s']} tok/s)")
print(f"Settings:  temperature=0 (greedy), max_tokens=200")
print("=" * 70)

total_similarity = 0
num_prompts = 0
results_table = []

for prompt_id in ref_results:
    if prompt_id not in local_results:
        print(f"  SKIP: {prompt_id} (not in local results)")
        continue

    ref_text = ref_results[prompt_id]["content"]
    local_text = local_results[prompt_id]["content"]

    # Compute similarity ratio
    similarity = SequenceMatcher(None, ref_text, local_text).ratio()
    total_similarity += similarity
    num_prompts += 1

    # Check first N characters match
    prefix_match = 0
    for i in range(min(len(ref_text), len(local_text))):
        if ref_text[i] == local_text[i]:
            prefix_match += 1
        else:
            break

    status = "✅ MATCH" if similarity > 0.8 else "⚠️  DIFF" if similarity > 0.5 else "❌ FAIL"

    results_table.append({
        "id": prompt_id,
        "similarity": similarity,
        "prefix_match": prefix_match,
        "ref_len": len(ref_text),
        "local_len": len(local_text),
        "status": status,
    })

print(f"\n{'Prompt':<20} {'Status':<12} {'Similarity':>10} {'Prefix Match':>13} {'Ref Len':>8} {'Local Len':>9}")
print("-" * 75)

for r in results_table:
    print(f"{r['id']:<20} {r['status']:<12} {r['similarity']:>9.1%} {r['prefix_match']:>10} ch {r['ref_len']:>8} {r['local_len']:>9}")

avg_similarity = total_similarity / num_prompts if num_prompts > 0 else 0

print("-" * 75)
print(f"{'AVERAGE':<20} {'':12} {avg_similarity:>9.1%}")
print()

# Interpretation
if avg_similarity > 0.85:
    print("🎉 EXCELLENT: Outputs are highly similar. Implementation is likely correct.")
    print("   Minor differences are expected from different Q4 quantization methods.")
elif avg_similarity > 0.6:
    print("⚠️  ACCEPTABLE: Outputs are somewhat similar but have notable differences.")
    print("   This could be from different quantization granularity or sampling edge cases.")
    print("   Check individual outputs that show low similarity.")
elif avg_similarity > 0.3:
    print("⚠️  CONCERNING: Significant differences detected.")
    print("   Review the specific prompts with low similarity scores.")
    print("   Could indicate a numerical issue in the forward pass.")
else:
    print("❌ PROBLEM: Outputs are very different from reference.")
    print("   Likely a bug in the implementation or wrong model weights.")

# Show detailed diffs for low-similarity prompts
print("\n" + "=" * 70)
print("DETAILED COMPARISON (prompts with <80% similarity)")
print("=" * 70)

for r in results_table:
    if r["similarity"] < 0.8:
        prompt_id = r["id"]
        print(f"\n--- {prompt_id} (similarity: {r['similarity']:.1%}) ---")
        print(f"REFERENCE (first 300 chars):")
        print(f"  {ref_results[prompt_id]['content'][:300]}")
        print(f"\nLOCAL (first 300 chars):")
        print(f"  {local_results[prompt_id]['content'][:300]}")
        print()

# Speed comparison
print("\n" + "=" * 70)
print("SPEED COMPARISON")
print("=" * 70)
print(f"Reference ({ref_data['engine']}): {ref_data['throughput_tok_s']} tok/s")
print(f"Local ({local_data['engine']}):     {local_data['throughput_tok_s']} tok/s")
if ref_data['throughput_tok_s'] > 0:
    ratio = local_data['throughput_tok_s'] / ref_data['throughput_tok_s']
    print(f"Speed ratio: {ratio:.2f}x {'(faster)' if ratio > 1 else '(slower)'}")
