"""
Scheduler stress smoke for the local OpenAI-compatible server.

Start the server first, then run:
    python3 benchmarks/stress_scheduler.py --port 8080

This exercises mixed prompt lengths, staggered arrivals, stream cancellation,
client-side request timeouts, batching metrics, and final idle gauges.
"""

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request


MODEL = "gemma-4-e4b-q4"
PHASE_GAUGES = (
    "llama_queued_requests",
    "llama_active_requests",
    "llama_prefilling_requests",
    "llama_decoding_requests",
)


def marker_prompt(idx, markers):
    if markers == 0:
        return f"Reply with exactly one short sentence. Stress request {idx}."
    return (
        f"Read these markers and reply with exactly one short sentence. Stress request {idx}. "
        + " ".join(f"marker{i}" for i in range(markers))
    )


def chat_payload(content, max_tokens, stream=False):
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "min_p": 0.0,
        "top_k": 0,
        "stream": stream,
        "stop": ["<end_of_turn>"],
    }


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
        raise AssertionError(f"HTTP {exc.code}: {body}") from exc


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


def metric_delta(after, before, name):
    return after.get(name, 0.0) - before.get(name, 0.0)


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def run_mixed_wave(base_url, requests, max_tokens, timeout, stagger_sec, marker_counts):
    results = [None] * requests
    errors = [None] * requests
    threads = []

    def worker(index):
        try:
            if stagger_sec > 0:
                time.sleep(index * stagger_sec)
            markers = marker_counts[index % len(marker_counts)]
            started_at = time.time()
            body = request_json(
                base_url,
                chat_payload(marker_prompt(index, markers), max_tokens=max_tokens),
                timeout=timeout,
            )
            results[index] = {
                "idx": index,
                "markers": markers,
                "elapsed": time.time() - started_at,
                "prompt_tokens": body["usage"]["prompt_tokens"],
                "completion_tokens": body["usage"]["completion_tokens"],
                "content": body["choices"][0]["message"]["content"].replace("\n", "\\n")[:120],
            }
        except Exception as exc:
            errors[index] = exc

    started_at = time.time()
    for index in range(requests):
        thread = threading.Thread(target=worker, args=(index,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise AssertionError("mixed wave request did not finish before timeout")

    for index, error in enumerate(errors):
        if error is not None:
            raise AssertionError(f"mixed wave request {index} failed: {error}") from error

    elapsed = time.time() - started_at
    latencies = [result["elapsed"] for result in results]
    tokens = sum(result["completion_tokens"] for result in results)
    return results, {
        "elapsed": elapsed,
        "p50": statistics.median(latencies),
        "p95": percentile(latencies, 0.95),
        "max": max(latencies),
        "completion_tokens": tokens,
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


def cancel_stream(base_url, index, cancel_after_chunks, timeout):
    payload = chat_payload(
        f"Write twenty numbered short sentences about cancellation handling. Stream {index}.",
        max_tokens=128,
        stream=True,
    )
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for event in parse_sse_events(resp):
            if event == "[DONE]":
                break
            content = event["choices"][0]["delta"].get("content", "")
            if content:
                chunks += 1
            if chunks >= cancel_after_chunks:
                break
    return chunks


def run_cancellation_probe(base_url, streams, cancel_after_chunks, timeout):
    results = [0] * streams
    errors = [None] * streams
    threads = []

    def worker(index):
        try:
            results[index] = cancel_stream(base_url, index, cancel_after_chunks, timeout)
        except Exception as exc:
            errors[index] = exc

    for index in range(streams):
        thread = threading.Thread(target=worker, args=(index,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise AssertionError("stream cancellation probe did not finish before timeout")

    for index, error in enumerate(errors):
        if error is not None:
            raise AssertionError(f"stream cancellation probe {index} failed: {error}") from error

    return results


def run_client_timeout_probe(base_url, probes, client_timeout):
    timeouts = 0
    for index in range(probes):
        try:
            request_json(
                base_url,
                chat_payload(marker_prompt(index, 180), max_tokens=64),
                timeout=client_timeout,
            )
        except TimeoutError:
            timeouts += 1
        except OSError as exc:
            if "timed out" in str(exc):
                timeouts += 1
            else:
                raise
    return timeouts


def wait_for_idle(base_url, timeout):
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = get_metrics(base_url)
        if all(last.get(gauge, 0.0) == 0.0 for gauge in PHASE_GAUGES):
            return last
        time.sleep(1)
    raise AssertionError(f"scheduler did not return to idle: {last}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--stagger-sec", type=float, default=0.03)
    parser.add_argument("--marker-counts", default="0,40,120,180")
    parser.add_argument("--cancel-streams", type=int, default=3)
    parser.add_argument("--cancel-after-chunks", type=int, default=2)
    parser.add_argument("--client-timeout-probes", type=int, default=2)
    parser.add_argument("--client-timeout-secs", type=float, default=0.5)
    args = parser.parse_args()

    marker_counts = [int(value) for value in args.marker_counts.split(",") if value.strip()]
    base_url = f"http://127.0.0.1:{args.port}"
    urllib.request.urlopen(f"{base_url}/health", timeout=5).read()

    before = get_metrics(base_url)
    wave_results, wave = run_mixed_wave(
        base_url,
        args.requests,
        args.max_tokens,
        args.timeout,
        args.stagger_sec,
        marker_counts,
    )
    after_wave = get_metrics(base_url)

    print(
        f"mixed_wave completed={len(wave_results)}/{args.requests} "
        f"elapsed={wave['elapsed']:.2f}s p50={wave['p50']:.2f}s "
        f"p95={wave['p95']:.2f}s max={wave['max']:.2f}s "
        f"completion_tokens={wave['completion_tokens']}"
    )
    print(
        "mixed_wave batching "
        f"prefill_batches={metric_delta(after_wave, before, 'llama_prefill_batches_total'):.0f} "
        f"prefill_items={metric_delta(after_wave, before, 'llama_prefill_batch_items_total'):.0f} "
        f"decode_batches={metric_delta(after_wave, before, 'llama_decode_batches_total'):.0f} "
        f"decode_items={metric_delta(after_wave, before, 'llama_decode_batch_items_total'):.0f}"
    )

    cancel_counts = run_cancellation_probe(
        base_url, args.cancel_streams, args.cancel_after_chunks, args.timeout
    )
    print(
        f"stream_cancellation streams={args.cancel_streams} "
        f"content_chunks_before_close={cancel_counts}"
    )

    client_timeouts = run_client_timeout_probe(
        base_url, args.client_timeout_probes, args.client_timeout_secs
    )
    print(
        f"client_timeout_probes timeouts={client_timeouts}/{args.client_timeout_probes} "
        f"client_timeout_secs={args.client_timeout_secs}"
    )

    final_metrics = wait_for_idle(base_url, args.timeout)
    for gauge in PHASE_GAUGES:
        assert final_metrics.get(gauge) == 0.0, f"{gauge} did not return to idle"

    print(
        "final_batch_metrics "
        f"prefill_avg={final_metrics.get('llama_prefill_batch_items_avg', 0.0):.3f} "
        f"prefill_max={final_metrics.get('llama_prefill_batch_items_max', 0.0):.0f} "
        f"decode_avg={final_metrics.get('llama_decode_batch_items_avg', 0.0):.3f} "
        f"decode_max={final_metrics.get('llama_decode_batch_items_max', 0.0):.0f}"
    )
    print("ok scheduler stress")


if __name__ == "__main__":
    main()
