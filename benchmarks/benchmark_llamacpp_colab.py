"""
Gemma4 E4B Quality Benchmark - llama.cpp Reference (Run on Google Colab with GPU)

Instructions:
1. Upload this file to Colab
2. Run all cells
3. Download the generated `reference_outputs.json`

This runs Gemma4 E4B at Q4_0 quantization with greedy decoding (temperature=0)
to produce deterministic reference outputs for comparison.
"""

# %% Cell 1: Install dependencies
# !pip install llama-cpp-python huggingface_hub

# %% Cell 2: Download model
from huggingface_hub import hf_hub_download
import json
import time

# Download Q4_0 GGUF of Gemma4 E4B
# Note: If this exact quant isn't available, use the closest Q4 variant
print("Downloading Gemma4 E4B GGUF...")
model_path = hf_hub_download(
    repo_id="unsloth/gemma-4-4b-it-GGUF",  # or bartowski's quant
    filename="gemma-4-4b-it-Q4_0.gguf",     # adjust filename as needed
)
print(f"Model downloaded to: {model_path}")

# %% Cell 3: Load model
from llama_cpp import Llama

print("Loading model...")
llm = Llama(
    model_path=model_path,
    n_ctx=1024,
    n_gpu_layers=-1,  # offload all to GPU
    verbose=False,
)
print("Model loaded!")

# %% Cell 4: Run benchmark
with open("prompts.json") as f:
    prompts = json.load(f)

results = []
total_tokens = 0
total_time = 0

for prompt_data in prompts:
    prompt_id = prompt_data["id"]
    messages = prompt_data["messages"]

    print(f"Running: {prompt_id}...")
    start = time.time()

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=200,
        temperature=0.0,  # GREEDY - deterministic
        top_p=1.0,
        top_k=-1,  # disabled
    )

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
    print(f"  → {usage['completion_tokens']} tokens in {elapsed:.2f}s")

# %% Cell 5: Save results
output = {
    "engine": "llama.cpp",
    "model": "gemma-4-4b-it-Q4_0",
    "settings": {"temperature": 0.0, "max_tokens": 200},
    "total_tokens": total_tokens,
    "total_time_seconds": round(total_time, 3),
    "throughput_tok_s": round(total_tokens / total_time, 2) if total_time > 0 else 0,
    "results": results,
}

with open("reference_outputs.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print(f"DONE: {total_tokens} tokens in {total_time:.2f}s ({total_tokens/total_time:.1f} tok/s)")
print(f"Results saved to reference_outputs.json")
print(f"{'='*60}")

# Print outputs for quick inspection
for r in results:
    print(f"\n--- {r['id']} ---")
    print(r["content"][:200])
