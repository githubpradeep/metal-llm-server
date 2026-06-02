"""
Gemma4 E4B Quality Benchmark - Local Server
Run this against your running server (cargo run --release -- --gpu --serve ...)

Usage:
    python3 benchmarks/benchmark_local.py [--port 8080]
"""

import json
import time
import sys
import urllib.request

PORT = 8080
if "--port" in sys.argv:
    PORT = int(sys.argv[sys.argv.index("--port") + 1])

BASE_URL = f"http://localhost:{PORT}"

def chat_completion(messages, max_tokens=200, temperature=0.0):
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

    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


# Check server is running
try:
    urllib.request.urlopen(f"{BASE_URL}/health", timeout=5)
except Exception as e:
    print(f"ERROR: Cannot reach server at {BASE_URL}/health")
    print(f"Start the server first: cargo run --release -- --gpu --serve <model_dir>")
    sys.exit(1)

print(f"Server reachable at {BASE_URL}")

# Load prompts
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "prompts.json")) as f:
    prompts = json.load(f)

results = []
total_tokens = 0
total_time = 0

for prompt_data in prompts:
    prompt_id = prompt_data["id"]
    messages = prompt_data["messages"]

    print(f"Running: {prompt_id}...", end=" ", flush=True)
    start = time.time()

    response = chat_completion(messages, max_tokens=200, temperature=0.0)

    elapsed = time.time() - start
    content = response["choices"][0]["message"]["content"]
    usage = response["usage"]

    results.append({
        "id": prompt_id,
        "content": content,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "elapsed_seconds": round(elapsed, 3),
    })

    total_tokens += usage["completion_tokens"]
    total_time += elapsed
    print(f"{usage['completion_tokens']} tokens in {elapsed:.2f}s")

# Save results
output = {
    "engine": "gemma4-rust-metal",
    "model": "gemma-4-e4b-q4-custom",
    "settings": {"temperature": 0.0, "max_tokens": 200},
    "total_tokens": total_tokens,
    "total_time_seconds": round(total_time, 3),
    "throughput_tok_s": round(total_tokens / total_time, 2) if total_time > 0 else 0,
    "results": results,
}

output_path = os.path.join(script_dir, "local_outputs.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print(f"DONE: {total_tokens} tokens in {total_time:.2f}s ({total_tokens/total_time:.1f} tok/s)")
print(f"Results saved to {output_path}")
print(f"{'='*60}")

# Print outputs for quick inspection
for r in results:
    print(f"\n--- {r['id']} ({r['completion_tokens']} tokens, {r['elapsed_seconds']}s) ---")
    print(r["content"][:300])
