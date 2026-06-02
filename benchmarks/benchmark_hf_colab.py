"""
Gemma4 E4B Quality Benchmark - HuggingFace Transformers Reference (Run on Colab with GPU)

This runs the FULL PRECISION (bf16) model as the gold standard reference.
Your Q4 output should be semantically equivalent but may differ in exact wording.

Instructions:
1. Open Google Colab, select T4 or A100 GPU runtime
2. Upload this file and prompts.json
3. Run all cells
4. Download reference_outputs.json
"""

# %% Cell 1: Install
# !pip install transformers torch accelerate

# %% Cell 2: Load model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 1. Define your 4-bit configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",  # or bfloat16, depending on your GPU
    bnb_4bit_quant_type="nf4",         # Recommended 4-bit type
    bnb_4bit_use_double_quant=True
)
import time

model_id = "google/gemma-4-E4B-it"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
print(f"Model loaded on: {model.device}")

# %% Cell 3: Run benchmark
with open("prompts.json") as f:
    prompts = json.load(f)

results = []
total_tokens = 0
total_time = 0

for prompt_data in prompts:
    prompt_id = prompt_data["id"]
    messages = prompt_data["messages"]

    print(f"Running: {prompt_id}...", end=" ", flush=True)

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]

    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,  # GREEDY - deterministic
            temperature=None,
            top_p=None,
        )

    elapsed = time.time() - start

    # Decode only the generated tokens (skip prompt)
    generated_ids = outputs[0][prompt_tokens:]
    content = tokenizer.decode(generated_ids, skip_special_tokens=True)
    completion_tokens = len(generated_ids)

    results.append({
        "id": prompt_id,
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_seconds": round(elapsed, 3),
    })

    total_tokens += completion_tokens
    total_time += elapsed
    print(f"{completion_tokens} tokens in {elapsed:.2f}s")

# %% Cell 4: Save results
output = {
    "engine": "huggingface-transformers",
    "model": "google/gemma-4-4b-it (bf16)",
    "settings": {"temperature": 0.0, "max_tokens": 200, "do_sample": False},
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
    print(f"\n--- {r['id']} ({r['completion_tokens']} tokens) ---")
    print(r["content"][:300])
