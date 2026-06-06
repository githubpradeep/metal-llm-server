"""
Correctness smoke for chunked and multi-request prefill through the local server.

Start the server first, then run:
    python3 benchmarks/prefill_correctness.py --port 8080

The check uses greedy decoding and compares sequential baseline outputs against
concurrent requests that should exercise the batched prefill path. It also
checks /metrics to verify that at least one prefill batch contained multiple
requests during the concurrent phase.
"""

import argparse
import json
import threading
import time
import urllib.error
import urllib.request


MODEL = "gemma-4-e4b-q4"


def marker_prompt(prefix, markers, suffix):
    return f"{prefix} " + " ".join(f"marker{i}" for i in range(markers)) + f" {suffix}"


DEFAULT_CASES = [
    (
        "short",
        "Reply with exactly one concise sentence about deterministic inference.",
    ),
    (
        "medium",
        marker_prompt(
            "Read these markers, then reply with exactly one concise sentence about batching.",
            70,
            "Keep the answer stable.",
        ),
    ),
    (
        "long",
        marker_prompt(
            "Read these markers, then reply with exactly one concise sentence about KV cache reuse.",
            170,
            "Keep the answer stable.",
        ),
    ),
]


def chat_payload(content, max_tokens):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "min_p": 0.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "stream": False,
        "stop": ["<end_of_turn>"],
    }


def request_json(method, url, payload=None, timeout=120.0):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AssertionError(f"HTTP {exc.code} from {url}: {body}") from exc


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


def run_chat(base_url, name, prompt, max_tokens, timeout):
    started_at = time.time()
    _, body = request_json(
        "POST",
        f"{base_url}/v1/chat/completions",
        chat_payload(prompt, max_tokens),
        timeout=timeout,
    )
    choice = body["choices"][0]
    return {
        "name": name,
        "content": choice["message"]["content"],
        "finish_reason": choice["finish_reason"],
        "usage": body["usage"],
        "elapsed": time.time() - started_at,
    }


def run_concurrent(base_url, cases, max_tokens, timeout):
    barrier = threading.Barrier(len(cases))
    results = [None] * len(cases)
    errors = [None] * len(cases)
    threads = []

    def worker(index, case):
        name, prompt = case
        try:
            barrier.wait(timeout=timeout)
            results[index] = run_chat(base_url, name, prompt, max_tokens, timeout)
        except Exception as exc:
            errors[index] = exc

    for index, case in enumerate(cases):
        thread = threading.Thread(target=worker, args=(index, case))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise AssertionError("concurrent correctness request did not finish before timeout")

    for error in errors:
        if error is not None:
            raise AssertionError(f"concurrent correctness request failed: {error}") from error

    return results


def metric_delta(after, before, name):
    return after.get(name, 0.0) - before.get(name, 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument(
        "--require-batched-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    urllib.request.urlopen(f"{base_url}/health", timeout=5).read()

    print("running sequential greedy baseline")
    baseline = [
        run_chat(base_url, name, prompt, args.max_tokens, args.timeout)
        for name, prompt in DEFAULT_CASES
    ]
    for result in baseline:
        print(
            f"baseline name={result['name']} "
            f"prompt_tokens={result['usage']['prompt_tokens']} "
            f"completion_tokens={result['usage']['completion_tokens']} "
            f"elapsed={result['elapsed']:.2f}s"
        )

    before = get_metrics(base_url)
    print("running concurrent greedy comparison")
    concurrent = run_concurrent(base_url, DEFAULT_CASES, args.max_tokens, args.timeout)
    after = get_metrics(base_url)

    for expected, actual in zip(baseline, concurrent):
        if actual["content"] != expected["content"]:
            raise AssertionError(
                f"content mismatch for {expected['name']}\n"
                f"baseline:   {expected['content']!r}\n"
                f"concurrent: {actual['content']!r}"
            )
        if actual["usage"]["prompt_tokens"] != expected["usage"]["prompt_tokens"]:
            raise AssertionError(
                f"prompt token mismatch for {expected['name']}: "
                f"baseline={expected['usage']['prompt_tokens']} "
                f"concurrent={actual['usage']['prompt_tokens']}"
            )
        print(
            f"ok match name={actual['name']} "
            f"completion_tokens={actual['usage']['completion_tokens']} "
            f"elapsed={actual['elapsed']:.2f}s"
        )

    prefill_batches = metric_delta(after, before, "llama_prefill_batches_total")
    prefill_items = metric_delta(after, before, "llama_prefill_batch_items_total")
    decode_batches = metric_delta(after, before, "llama_decode_batches_total")
    decode_items = metric_delta(after, before, "llama_decode_batch_items_total")

    print(f"prefill_batches_delta={prefill_batches:.0f}")
    print(f"prefill_batch_items_delta={prefill_items:.0f}")
    print(f"decode_batches_delta={decode_batches:.0f}")
    print(f"decode_batch_items_delta={decode_items:.0f}")

    if args.require_batched_prefill and prefill_items <= prefill_batches:
        raise AssertionError(
            "concurrent run did not prove multi-request prefill batching: "
            f"items_delta={prefill_items:.0f}, batches_delta={prefill_batches:.0f}"
        )

    print("ok prefill correctness")


if __name__ == "__main__":
    main()
