"""
Gemma4 E4B Comprehensive Evaluation Runner

Runs all prompts from eval_prompts.json against the local server,
collects outputs into eval_outputs.json for LLM-as-a-Judge scoring.

Usage:
    python3 benchmarks/eval_run.py [--port 8080] [--max-tokens 400]

Output:
    benchmarks/eval_outputs.json — ready to paste into Kiro for judging
"""

import json
import time
import sys
import os
import urllib.request
from datetime import datetime

# --- Config ---
PORT = 8080
MAX_TOKENS_OVERRIDE = None  # None = use per-prompt max_tokens from eval_prompts.json

if "--port" in sys.argv:
    PORT = int(sys.argv[sys.argv.index("--port") + 1])
if "--max-tokens" in sys.argv:
    MAX_TOKENS_OVERRIDE = int(sys.argv[sys.argv.index("--max-tokens") + 1])

BASE_URL = f"http://localhost:{PORT}"
script_dir = os.path.dirname(os.path.abspath(__file__))


def chat_completion(messages, max_tokens=400, temperature=0.0):
    """Call the local OpenAI-compatible server."""
    payload = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def run_single_turn(prompt_data):
    """Run a single-turn prompt."""
    messages = prompt_data["messages"]
    max_tokens = MAX_TOKENS_OVERRIDE or prompt_data.get("max_tokens", 400)

    start = time.time()
    response = chat_completion(messages, max_tokens=max_tokens, temperature=0.0)
    elapsed = time.time() - start

    content = response["choices"][0]["message"]["content"]
    usage = response.get("usage", {})

    return {
        "content": content,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "elapsed_seconds": round(elapsed, 3),
    }


def run_multi_turn(prompt_data):
    """Run a multi-turn prompt (collect response for first turn, then send follow-up)."""
    messages = prompt_data["messages"]
    max_tokens = MAX_TOKENS_OVERRIDE or prompt_data.get("max_tokens", 400)

    # Find the split point: first empty assistant message
    turn1_messages = []
    turn2_messages = []
    split_found = False

    for msg in messages:
        if msg["role"] == "assistant" and msg["content"] == "" and not split_found:
            split_found = True
            continue
        if not split_found:
            turn1_messages.append(msg)
        else:
            turn2_messages.append(msg)

    # Turn 1
    start = time.time()
    response1 = chat_completion(turn1_messages, max_tokens=max_tokens, temperature=0.0)
    elapsed1 = time.time() - start
    turn1_content = response1["choices"][0]["message"]["content"]

    # Build full context for turn 2
    full_messages = turn1_messages + [{"role": "assistant", "content": turn1_content}] + turn2_messages

    # Turn 2
    start = time.time()
    response2 = chat_completion(full_messages, max_tokens=max_tokens, temperature=0.0)
    elapsed2 = time.time() - start
    turn2_content = response2["choices"][0]["message"]["content"]

    usage1 = response1.get("usage", {})
    usage2 = response2.get("usage", {})

    return {
        "turn1_content": turn1_content,
        "turn2_content": turn2_content,
        "content": f"[Turn 1]\n{turn1_content}\n\n[Turn 2]\n{turn2_content}",
        "prompt_tokens": usage1.get("prompt_tokens", 0) + usage2.get("prompt_tokens", 0),
        "completion_tokens": usage1.get("completion_tokens", 0) + usage2.get("completion_tokens", 0),
        "elapsed_seconds": round(elapsed1 + elapsed2, 3),
    }


def main():
    # Check server
    try:
        urllib.request.urlopen(f"{BASE_URL}/health", timeout=5)
    except Exception:
        print(f"ERROR: Cannot reach server at {BASE_URL}/health")
        print("Start the server first: cargo run --release -- --gpu --serve <model_dir>")
        sys.exit(1)

    print(f"✓ Server reachable at {BASE_URL}")

    # Load prompts
    with open(os.path.join(script_dir, "eval_prompts.json")) as f:
        eval_data = json.load(f)

    prompts = eval_data["prompts"]
    categories = {c["id"]: c["name"] for c in eval_data["categories"]}

    print(f"✓ Loaded {len(prompts)} evaluation prompts across {len(categories)} categories")
    print(f"{'='*60}")

    results = []
    total_tokens = 0
    total_time = 0
    errors = []

    for i, prompt_data in enumerate(prompts, 1):
        prompt_id = prompt_data["id"]
        category = prompt_data["category"]
        turns = prompt_data.get("turns", 1)

        print(f"[{i}/{len(prompts)}] {prompt_id} ({categories[category]})...", end=" ", flush=True)

        try:
            if turns > 1:
                result = run_multi_turn(prompt_data)
            else:
                result = run_single_turn(prompt_data)

            results.append({
                "id": prompt_id,
                "category": category,
                "category_name": categories[category],
                "turns": turns,
                "prompt": prompt_data["messages"],
                "rubric": prompt_data.get("rubric", ""),
                "response": result["content"],
                "completion_tokens": result["completion_tokens"],
                "elapsed_seconds": result["elapsed_seconds"],
            })

            total_tokens += result["completion_tokens"]
            total_time += result["elapsed_seconds"]
            tok_s = result["completion_tokens"] / result["elapsed_seconds"] if result["elapsed_seconds"] > 0 else 0
            print(f"✓ {result['completion_tokens']} tok, {result['elapsed_seconds']:.1f}s ({tok_s:.1f} tok/s)")

        except Exception as e:
            print(f"✗ ERROR: {e}")
            errors.append({"id": prompt_id, "error": str(e)})
            results.append({
                "id": prompt_id,
                "category": category,
                "category_name": categories[category],
                "turns": turns,
                "prompt": prompt_data["messages"],
                "rubric": prompt_data.get("rubric", ""),
                "response": f"[ERROR: {e}]",
                "completion_tokens": 0,
                "elapsed_seconds": 0,
            })

    # Summary
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Prompts run: {len(results)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg throughput: {total_tokens/total_time:.1f} tok/s" if total_time > 0 else "")

    # Per-category stats
    print(f"\n  Per-category breakdown:")
    cat_stats = {}
    for r in results:
        cat = r["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"tokens": 0, "time": 0, "count": 0}
        cat_stats[cat]["tokens"] += r["completion_tokens"]
        cat_stats[cat]["time"] += r["elapsed_seconds"]
        cat_stats[cat]["count"] += 1

    for cat, stats in sorted(cat_stats.items()):
        tok_s = stats["tokens"] / stats["time"] if stats["time"] > 0 else 0
        print(f"    {categories[cat]:<25} {stats['count']} prompts, {stats['tokens']:>4} tok, {tok_s:.1f} tok/s")

    # Save output
    output = {
        "metadata": {
            "engine": "gemma4-rust-metal",
            "model": "gemma-4-e4b-it (Q4 custom quantization)",
            "eval_version": eval_data["version"],
            "timestamp": datetime.now().isoformat(),
            "settings": {
                "temperature": 0.0,
                "max_tokens": MAX_TOKENS_OVERRIDE or "per-prompt (see eval_prompts.json)",
            },
            "server_url": BASE_URL,
        },
        "summary": {
            "total_prompts": len(results),
            "total_tokens": total_tokens,
            "total_time_seconds": round(total_time, 3),
            "avg_throughput_tok_s": round(total_tokens / total_time, 2) if total_time > 0 else 0,
            "errors": len(errors),
        },
        "results": results,
    }

    output_path = os.path.join(script_dir, "eval_outputs.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {output_path}")
    print(f"  → Paste this file into Kiro chat for LLM-as-a-Judge evaluation")

    if errors:
        print(f"\n⚠️  Errors:")
        for e in errors:
            print(f"    {e['id']}: {e['error']}")


if __name__ == "__main__":
    main()
