"""
Regression smoke test for the local OpenAI-compatible server.

Start the server first, then run:
    python3 benchmarks/server_regression.py --port 8080
"""

import argparse
import json
import threading
import time
import urllib.error
import urllib.request

import mixed_batching_fairness
import prefill_correctness
import stress_scheduler


MODEL = "gemma-4-e4b-q4"
PHASE_GAUGES = (
    "llama_queued_requests",
    "llama_active_requests",
    "llama_prefilling_requests",
    "llama_decoding_requests",
)


def request_json(method, url, payload=None, timeout=30.0):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body) if body else None


def expect_http_error(method, url, payload, expected_status, expected_code):
    try:
        request_json(method, url, payload)
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8"))
        code = body["error"]["code"]
        assert exc.code == expected_status, f"expected {expected_status}, got {exc.code}"
        assert code == expected_code, f"expected error code {expected_code}, got {code}"
        return

    raise AssertionError(f"expected HTTP {expected_status} {expected_code}")


def chat_payload(content, max_tokens=32, stream=False, **overrides):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": stream,
        "stop": ["<end_of_turn>"],
    }
    payload.update(overrides)
    return payload


def check_health_and_models(base_url):
    with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
        assert resp.status == 200
        assert resp.read().decode("utf-8") == "ok"

    status, body = request_json("GET", f"{base_url}/v1/models")
    assert status == 200
    model_ids = {item["id"] for item in body["data"]}
    assert MODEL in model_ids
    print("ok health/models")


def check_structured_errors(base_url):
    expect_http_error(
        "POST",
        f"{base_url}/v1/chat/completions",
        {"model": MODEL, "messages": [], "max_tokens": 32},
        400,
        "empty_messages",
    )
    expect_http_error(
        "POST",
        f"{base_url}/v1/chat/completions",
        chat_payload("hello", max_tokens=0),
        400,
        "invalid_max_tokens",
    )
    expect_http_error(
        "POST",
        f"{base_url}/v1/chat/completions",
        chat_payload("hello", temperature=-0.1),
        400,
        "invalid_temperature",
    )
    print("ok structured errors")


def check_sync_chat(base_url):
    status, body = request_json(
        "POST",
        f"{base_url}/v1/chat/completions",
        chat_payload("Write a short greeting.", max_tokens=32, stream=False),
        timeout=120,
    )
    assert status == 200
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert "<end_of_turn>" not in choice["message"]["content"]
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] > 0
    print("ok sync chat")


def check_chunked_prefill(base_url):
    long_prompt = " ".join(
        [
            "This is a chunked prefill regression prompt.",
            "Keep the reply short.",
            "The following words intentionally make the prompt long enough",
            "to cross the default prompt chunk boundary while still being simple.",
        ]
        + [f"marker{i}" for i in range(90)]
    )
    status, body = request_json(
        "POST",
        f"{base_url}/v1/chat/completions",
        chat_payload(long_prompt, max_tokens=8, stream=False, temperature=0.0),
        timeout=180,
    )
    assert status == 200
    choice = body["choices"][0]
    assert choice["finish_reason"] in {"stop", "length"}
    assert "<end_of_turn>" not in choice["message"]["content"]
    assert body["usage"]["prompt_tokens"] > 64
    assert body["usage"]["completion_tokens"] > 0
    print("ok chunked prefill")


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


def check_stream_chat(base_url):
    payload = chat_payload("Count from one to five.", max_tokens=32, stream=True)
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    chunks = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        assert resp.status == 200
        for event in parse_sse_events(resp):
            chunks.append(event)
            if event == "[DONE]":
                break

    assert chunks, "stream returned no chunks"
    assert chunks[-1] == "[DONE]"
    role_chunks = [
        chunk for chunk in chunks if chunk != "[DONE]" and chunk["choices"][0]["delta"].get("role")
    ]
    assert role_chunks, "stream did not include assistant role delta"

    text = "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks
        if chunk != "[DONE]"
    )
    assert "<end_of_turn>" not in text
    final = chunks[-2]
    assert final["choices"][0]["finish_reason"] == "stop"
    print("ok stream chat")


def post_chat(base_url, idx, max_tokens, timeout):
    started_at = time.time()
    try:
        status, body = request_json(
            "POST",
            f"{base_url}/v1/chat/completions",
            chat_payload(
                f"Reply with exactly one short sentence. Request number {idx}.",
                max_tokens=max_tokens,
                stream=False,
                temperature=0.0,
            ),
            timeout=timeout,
        )
        content = body["choices"][0]["message"]["content"]
        return {
            "ok": True,
            "idx": idx,
            "status": status,
            "elapsed": time.time() - started_at,
            "tokens": body["usage"]["completion_tokens"],
            "content": content.replace("\n", "\\n")[:120],
        }
    except Exception as exc:
        return {
            "ok": False,
            "idx": idx,
            "status": "exception",
            "elapsed": time.time() - started_at,
            "error": str(exc),
        }


def check_concurrency(base_url, requests, max_tokens, timeout):
    results = [None] * requests
    threads = []
    started_at = time.time()

    def worker(i):
        results[i] = post_chat(base_url, i, max_tokens, timeout)

    for i in range(requests):
        thread = threading.Thread(target=worker, args=(i,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    ok_count = sum(1 for result in results if result and result["ok"])
    for result in results:
        status = "OK" if result["ok"] else "ERR"
        print(
            f"{status} req={result['idx']} status={result['status']} "
            f"elapsed={result['elapsed']:.2f}s tokens={result.get('tokens', 0)}"
        )
        if result["ok"]:
            assert "<end_of_turn>" not in result["content"]
            print(f"  {result['content']}")
        else:
            print(f"  {result.get('error', '')}")

    assert ok_count == requests, f"completed={ok_count}/{requests}"
    print(f"ok concurrency completed={ok_count}/{requests} elapsed={time.time() - started_at:.2f}s")


def parse_metrics(text):
    metrics = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, value = line.split()[:2]
        metrics[name] = float(value)
    return metrics


def check_idle_metrics(base_url):
    time.sleep(5)
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as resp:
        assert resp.status == 200
        metrics = parse_metrics(resp.read().decode("utf-8"))

    for gauge in (
        "llama_config_queue_depth",
        "llama_config_kv_pool_slots",
        "llama_config_request_timeout_secs",
        "llama_config_prefill_tokens_per_tick",
    ):
        assert gauge in metrics, f"{gauge} missing from /metrics"

    for gauge in PHASE_GAUGES:
        assert metrics.get(gauge) == 0.0, f"{gauge} did not return to idle: {metrics.get(gauge)}"
    for counter in (
        "llama_prefill_batches_total",
        "llama_prefill_batch_items_total",
        "llama_decode_batches_total",
        "llama_decode_batch_items_total",
    ):
        assert counter in metrics, f"{counter} missing from /metrics"
    for gauge in (
        "llama_prefill_batch_items_avg",
        "llama_prefill_batch_items_max",
        "llama_decode_batch_items_avg",
        "llama_decode_batch_items_max",
        "llama_request_latency_ms_avg",
        "llama_request_latency_ms_max",
        "llama_prefill_latency_ms_avg",
        "llama_prefill_latency_ms_max",
        "llama_decode_latency_ms_avg",
        "llama_decode_latency_ms_max",
        "llama_decode_compute_latency_ms_avg",
        "llama_decode_compute_latency_ms_max",
    ):
        assert gauge in metrics, f"{gauge} missing from /metrics"
    print("ok idle metrics")


def check_mixed_fairness(base_url, args):
    stream_started = threading.Event()
    token_times = []
    stream_stats = {
        "events": 0,
        "role_seen": False,
        "content_chunks": 0,
        "content_chars": 0,
        "finish_reason": None,
        "done_seen": False,
        "text_sample": "",
    }
    errors = []
    stream_thread = threading.Thread(
        target=mixed_batching_fairness.stream_decode,
        args=(
            base_url,
            args.mixed_stream_tokens,
            stream_started,
            token_times,
            stream_stats,
            errors,
            args.timeout,
            args.mixed_stream_prompt,
        ),
    )

    stream_thread.start()
    if not stream_started.wait(timeout=args.timeout):
        raise AssertionError("stream did not start before timeout")
    if errors:
        raise AssertionError(f"stream request failed before fairness load: {errors[0]}")

    prefill_elapsed, prefill_usage = mixed_batching_fairness.long_prefill(
        base_url,
        args.mixed_prefill_words,
        args.mixed_prefill_max_tokens,
        args.timeout,
    )
    stream_thread.join(timeout=args.timeout)
    if stream_thread.is_alive():
        raise AssertionError("stream request did not finish before timeout")
    if errors:
        raise AssertionError(f"stream request failed: {errors[0]}")

    summary = mixed_batching_fairness.summarize_gaps(token_times)
    print(
        "mixed fairness "
        f"stream_chunks={summary['count']} "
        f"stream_events={stream_stats['events']} "
        f"finish_reason={stream_stats['finish_reason']} "
        f"max_gap={summary['max_gap']:.3f}s "
        f"p95_gap={summary['p95_gap']:.3f}s "
        f"prefill_elapsed={prefill_elapsed:.3f}s "
        f"prefill_prompt_tokens={prefill_usage['prompt_tokens']}"
    )

    assert summary["count"] >= args.mixed_min_stream_chunks, (
        f"stream produced only {summary['count']} chunks; "
        f"expected at least {args.mixed_min_stream_chunks}"
    )
    assert summary["max_gap"] <= args.mixed_max_stream_gap, (
        f"stream max token gap {summary['max_gap']:.3f}s exceeded "
        f"{args.mixed_max_stream_gap:.3f}s while long prefill was active"
    )
    print("ok mixed fairness")


def check_prefill_correctness(base_url, args):
    baseline = [
        prefill_correctness.run_chat(
            base_url,
            name,
            prompt,
            args.prefill_correctness_max_tokens,
            args.timeout,
        )
        for name, prompt in prefill_correctness.DEFAULT_CASES
    ]

    before = prefill_correctness.get_metrics(base_url)
    concurrent = prefill_correctness.run_concurrent(
        base_url,
        prefill_correctness.DEFAULT_CASES,
        args.prefill_correctness_max_tokens,
        args.timeout,
    )
    prefill_correctness.compare_outputs("normal", baseline, concurrent)

    if args.prefill_correctness_reversed_order:
        reversed_concurrent = prefill_correctness.run_concurrent(
            base_url,
            list(reversed(prefill_correctness.DEFAULT_CASES)),
            args.prefill_correctness_max_tokens,
            args.timeout,
        )
        prefill_correctness.compare_outputs("reversed", baseline, reversed_concurrent)

    after = prefill_correctness.get_metrics(base_url)

    prefill_batches = prefill_correctness.metric_delta(
        after, before, "llama_prefill_batches_total"
    )
    prefill_items = prefill_correctness.metric_delta(
        after, before, "llama_prefill_batch_items_total"
    )
    assert prefill_items > prefill_batches, (
        "prefill correctness run did not prove multi-request prefill batching: "
        f"items_delta={prefill_items:.0f}, batches_delta={prefill_batches:.0f}"
    )
    print(
        "ok prefill correctness "
        f"prefill_batches_delta={prefill_batches:.0f} "
        f"prefill_items_delta={prefill_items:.0f}"
    )


def check_scheduler_stress(base_url, args):
    marker_counts = [
        int(value) for value in args.stress_marker_counts.split(",") if value.strip()
    ]
    before = stress_scheduler.get_metrics(base_url)
    _, wave = stress_scheduler.run_mixed_wave(
        base_url,
        args.stress_requests,
        args.stress_max_tokens,
        args.timeout,
        args.stress_stagger_sec,
        marker_counts,
    )
    after_wave = stress_scheduler.get_metrics(base_url)

    cancel_counts = stress_scheduler.run_cancellation_probe(
        base_url,
        args.stress_cancel_streams,
        args.stress_cancel_after_chunks,
        args.timeout,
    )
    client_timeouts = stress_scheduler.run_client_timeout_probe(
        base_url,
        args.stress_client_timeout_probes,
        args.stress_client_timeout_secs,
    )
    final_metrics = stress_scheduler.wait_for_idle(base_url, args.timeout)

    for gauge in PHASE_GAUGES:
        assert final_metrics.get(gauge) == 0.0, f"{gauge} did not return to idle"

    print(
        "ok scheduler stress "
        f"requests={args.stress_requests} "
        f"elapsed={wave['elapsed']:.2f}s "
        f"p95={wave['p95']:.2f}s "
        f"prefill_items_delta={stress_scheduler.metric_delta(after_wave, before, 'llama_prefill_batch_items_total'):.0f} "
        f"decode_items_delta={stress_scheduler.metric_delta(after_wave, before, 'llama_decode_batch_items_total'):.0f} "
        f"cancel_chunks={cancel_counts} "
        f"client_timeouts={client_timeouts}/{args.stress_client_timeout_probes}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--prefill-correctness", action="store_true")
    parser.add_argument("--prefill-correctness-max-tokens", type=int, default=24)
    parser.add_argument(
        "--prefill-correctness-reversed-order",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mixed-fairness", action="store_true")
    parser.add_argument("--mixed-stream-tokens", type=int, default=64)
    parser.add_argument(
        "--mixed-stream-prompt",
        default=mixed_batching_fairness.DEFAULT_STREAM_PROMPT,
    )
    parser.add_argument("--mixed-prefill-words", type=int, default=180)
    parser.add_argument("--mixed-prefill-max-tokens", type=int, default=1)
    parser.add_argument("--mixed-max-stream-gap", type=float, default=5.0)
    parser.add_argument("--mixed-min-stream-chunks", type=int, default=4)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--stress-requests", type=int, default=16)
    parser.add_argument("--stress-max-tokens", type=int, default=20)
    parser.add_argument("--stress-stagger-sec", type=float, default=0.03)
    parser.add_argument("--stress-marker-counts", default="0,40,120,180")
    parser.add_argument("--stress-cancel-streams", type=int, default=2)
    parser.add_argument("--stress-cancel-after-chunks", type=int, default=2)
    parser.add_argument("--stress-client-timeout-probes", type=int, default=1)
    parser.add_argument("--stress-client-timeout-secs", type=float, default=0.5)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    check_health_and_models(base_url)
    check_structured_errors(base_url)
    check_sync_chat(base_url)
    check_chunked_prefill(base_url)
    check_stream_chat(base_url)
    check_concurrency(base_url, args.requests, args.max_tokens, args.timeout)
    if args.prefill_correctness:
        check_prefill_correctness(base_url, args)
    if args.mixed_fairness:
        check_mixed_fairness(base_url, args)
    if args.stress:
        check_scheduler_stress(base_url, args)
    check_idle_metrics(base_url)
    print("all regression checks passed")


if __name__ == "__main__":
    main()
