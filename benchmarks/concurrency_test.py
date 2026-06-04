"""
Concurrency smoke test for the local OpenAI-compatible server.

Usage:
    python3 benchmarks/concurrency_test.py --requests 10 --port 8080
"""

import argparse
import json
import threading
import time
import urllib.error
import urllib.request


def post_chat(base_url, idx, max_tokens, timeout):
    payload = json.dumps({
        "model": "gemma-4-e4b-q4",
        "messages": [{
            "role": "user",
            "content": f"Reply with exactly one short sentence. Request number {idx}.",
        }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
            elapsed = time.time() - start
            usage = body.get("usage", {})
            content = body["choices"][0]["message"]["content"].replace("\n", "\\n")
            return {
                "idx": idx,
                "ok": True,
                "status": resp.status,
                "elapsed": elapsed,
                "tokens": usage.get("completion_tokens", 0),
                "content": content[:120],
            }
    except urllib.error.HTTPError as exc:
        return {
            "idx": idx,
            "ok": False,
            "status": exc.code,
            "elapsed": time.time() - start,
            "error": exc.read().decode("utf-8", errors="replace")[:200],
        }
    except Exception as exc:
        return {
            "idx": idx,
            "ok": False,
            "status": "exception",
            "elapsed": time.time() - start,
            "error": str(exc),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    urllib.request.urlopen(f"{base_url}/health", timeout=5).read()

    results = [None] * args.requests
    threads = []
    start = time.time()

    def worker(i):
        results[i] = post_chat(base_url, i, args.max_tokens, args.timeout)

    for i in range(args.requests):
        thread = threading.Thread(target=worker, args=(i,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    total_elapsed = time.time() - start
    ok_count = sum(1 for result in results if result and result["ok"])
    total_tokens = sum(result.get("tokens", 0) for result in results if result)

    for result in results:
        status = "OK" if result["ok"] else "ERR"
        print(
            f"{status} req={result['idx']} status={result['status']} "
            f"elapsed={result['elapsed']:.2f}s tokens={result.get('tokens', 0)}"
        )
        if result["ok"]:
            print(f"  {result['content']}")
        else:
            print(f"  {result.get('error', '')}")

    print()
    print(f"completed={ok_count}/{args.requests}")
    print(f"total_elapsed={total_elapsed:.2f}s")
    print(f"total_completion_tokens={total_tokens}")
    if total_elapsed > 0:
        print(f"aggregate_completion_tok_s={total_tokens / total_elapsed:.2f}")


if __name__ == "__main__":
    main()
