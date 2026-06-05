"""
Measure prefill throughput against the local OpenAI-compatible server.

Start the server first, then run:
    python3 benchmarks/prefill_benchmark.py --port 8080
"""

import argparse
import json
import time
import urllib.error
import urllib.request


MODEL = "gemma-4-e4b-q4"


def parse_metrics(text):
    metrics = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, value = line.split()[:2]
        metrics[name] = float(value)
    return metrics


def get_metrics(base_url):
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as resp:
        return parse_metrics(resp.read().decode("utf-8"))


def request_json(url, payload, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def request_json_or_error(url, payload, timeout):
    try:
        return request_json(url, payload, timeout), None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"error": {"message": body, "code": str(exc.code)}}
        return None, {
            "status": exc.code,
            "code": parsed.get("error", {}).get("code", str(exc.code)),
            "message": parsed.get("error", {}).get("message", body),
        }


def make_prompt(target_words):
    words = [f"marker{i}" for i in range(target_words)]
    return (
        "Read the following tokens and reply with one short word. "
        + " ".join(words)
    )


def run_case(base_url, target_words, max_tokens, timeout):
    before = get_metrics(base_url)
    started = time.time()
    body, error = request_json_or_error(
        f"{base_url}/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": make_prompt(target_words)}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stop": ["<end_of_turn>"],
        },
        timeout,
    )
    elapsed_ms = (time.time() - started) * 1000.0
    after = get_metrics(base_url)

    prefill_tokens = after.get("llama_prefill_tokens_total", 0.0) - before.get(
        "llama_prefill_tokens_total", 0.0
    )
    prefill_ms = after.get("llama_prefill_latency_ms_total", 0.0) - before.get(
        "llama_prefill_latency_ms_total", 0.0
    )
    prefill_chunks = after.get("llama_prefill_chunks_total", 0.0) - before.get(
        "llama_prefill_chunks_total", 0.0
    )
    tok_s = (prefill_tokens / (prefill_ms / 1000.0)) if prefill_ms > 0 else 0.0

    if error:
        return {
            "target_words": target_words,
            "error": error,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "prefill_tokens": int(prefill_tokens),
            "prefill_chunks": int(prefill_chunks),
            "prefill_ms": prefill_ms,
            "prefill_tok_s": tok_s,
            "request_ms": elapsed_ms,
        }

    usage = body["usage"]
    return {
        "target_words": target_words,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "prefill_tokens": int(prefill_tokens),
        "prefill_chunks": int(prefill_chunks),
        "prefill_ms": prefill_ms,
        "prefill_tok_s": tok_s,
        "request_ms": elapsed_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sizes", default="32,64,128,256,512")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
        assert resp.status == 200

    sizes = [int(size.strip()) for size in args.sizes.split(",") if size.strip()]
    print("words,prompt_tokens,prefill_chunks,prefill_ms,prefill_tok_s,request_ms,status")
    for size in sizes:
        results = [
            run_case(base_url, size, args.max_tokens, args.timeout)
            for _ in range(args.repeats)
        ]
        errors = [row["error"] for row in results if row.get("error")]
        if errors:
            first = errors[0]
            print(
                f"{size},0.0,0.0,0.0,0.00,0.0,"
                f"error:{first['status']}:{first['code']}:{first['message']}"
            )
            continue

        avg_prompt = sum(row["prompt_tokens"] for row in results) / len(results)
        avg_chunks = sum(row["prefill_chunks"] for row in results) / len(results)
        avg_prefill_ms = sum(row["prefill_ms"] for row in results) / len(results)
        avg_tok_s = sum(row["prefill_tok_s"] for row in results) / len(results)
        avg_request_ms = sum(row["request_ms"] for row in results) / len(results)
        print(
            f"{size},{avg_prompt:.1f},{avg_chunks:.1f},"
            f"{avg_prefill_ms:.1f},{avg_tok_s:.2f},{avg_request_ms:.1f},ok"
        )


if __name__ == "__main__":
    main()
