"""
Mixed prefill/decode fairness smoke test for the local OpenAI-compatible server.

Start the server first, then run:
    python3 benchmarks/mixed_batching_fairness.py --port 8080

Useful scheduler tuning while testing:
    LLAMA_PREFILL_TOKENS_PER_TICK=32 cargo run --release -- --gpu --serve <model_dir> --port 8080
"""

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request


MODEL = "gemma-4-e4b-q4"


def chat_payload(content, max_tokens, stream=False):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "stop": ["<end_of_turn>"],
    }


def parse_sse_events(resp):
    for raw_line in resp:
        line = raw_line.decode("utf-8").strip()
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        if data == "[DONE]":
            yield "[DONE]"
        else:
            yield json.loads(data)


def request_json(base_url, payload, timeout):
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AssertionError(f"HTTP {exc.code} from /v1/chat/completions: {body}") from exc


def stream_decode(base_url, max_tokens, stream_started, token_times, errors, timeout):
    payload = chat_payload(
        "Write a long comma-separated list of simple counting words. "
        "Do not stop early; continue until the token budget is exhausted.",
        max_tokens=max_tokens,
        stream=True,
    )
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for event in parse_sse_events(resp):
                if event == "[DONE]":
                    break
                stream_started.set()
                choice = event["choices"][0]
                content = choice["delta"].get("content", "")
                if content:
                    token_times.append(time.time())
    except Exception as exc:
        errors.append(exc)
        stream_started.set()


def long_prefill(base_url, words, max_tokens, timeout):
    prompt = (
        "Read the following markers and then reply with one short sentence. "
        + " ".join(f"marker{i}" for i in range(words))
    )
    started_at = time.time()
    body = request_json(
        base_url,
        chat_payload(prompt, max_tokens=max_tokens, stream=False),
        timeout=timeout,
    )
    return time.time() - started_at, body["usage"]


def summarize_gaps(token_times):
    if len(token_times) < 2:
        return {
            "count": len(token_times),
            "max_gap": 0.0,
            "mean_gap": 0.0,
            "p95_gap": 0.0,
        }

    gaps = [b - a for a, b in zip(token_times, token_times[1:])]
    sorted_gaps = sorted(gaps)
    p95_index = min(len(sorted_gaps) - 1, int(len(sorted_gaps) * 0.95))
    return {
        "count": len(token_times),
        "max_gap": max(gaps),
        "mean_gap": statistics.mean(gaps),
        "p95_gap": sorted_gaps[p95_index],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--stream-tokens", type=int, default=64)
    parser.add_argument("--prefill-words", type=int, default=180)
    parser.add_argument("--prefill-max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-stream-gap", type=float, default=5.0)
    parser.add_argument("--min-stream-chunks", type=int, default=4)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    urllib.request.urlopen(f"{base_url}/health", timeout=5).read()

    stream_started = threading.Event()
    token_times = []
    errors = []
    stream_thread = threading.Thread(
        target=stream_decode,
        args=(
            base_url,
            args.stream_tokens,
            stream_started,
            token_times,
            errors,
            args.timeout,
        ),
    )

    started_at = time.time()
    stream_thread.start()
    if not stream_started.wait(timeout=args.timeout):
        raise AssertionError("stream did not start before timeout")
    if errors:
        raise AssertionError(f"stream request failed before fairness load: {errors[0]}")

    prefill_elapsed, prefill_usage = long_prefill(
        base_url,
        args.prefill_words,
        args.prefill_max_tokens,
        args.timeout,
    )
    stream_thread.join(timeout=args.timeout)
    if stream_thread.is_alive():
        raise AssertionError("stream request did not finish before timeout")
    if errors:
        raise AssertionError(f"stream request failed: {errors[0]}")

    summary = summarize_gaps(token_times)
    total_elapsed = time.time() - started_at

    print(f"stream_chunks={summary['count']}")
    print(f"stream_max_gap_s={summary['max_gap']:.3f}")
    print(f"stream_p95_gap_s={summary['p95_gap']:.3f}")
    print(f"stream_mean_gap_s={summary['mean_gap']:.3f}")
    print(f"prefill_elapsed_s={prefill_elapsed:.3f}")
    print(f"prefill_prompt_tokens={prefill_usage['prompt_tokens']}")
    print(f"prefill_completion_tokens={prefill_usage['completion_tokens']}")
    print(f"total_elapsed_s={total_elapsed:.3f}")

    if summary["count"] < args.min_stream_chunks:
        raise AssertionError(
            f"stream produced only {summary['count']} chunks; "
            f"expected at least {args.min_stream_chunks}"
        )
    if summary["max_gap"] > args.max_stream_gap:
        raise AssertionError(
            f"stream max token gap {summary['max_gap']:.3f}s exceeded "
            f"{args.max_stream_gap:.3f}s while long prefill was active"
        )

    print("ok mixed prefill/decode fairness")


if __name__ == "__main__":
    main()
