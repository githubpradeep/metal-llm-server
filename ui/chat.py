#!/usr/bin/env python3
"""Simple chat UI for a local OpenAI-compatible server."""

import os
import sys
import json
import argparse
import urllib.parse
from pathlib import Path

import requests
import urllib3
from bs4 import BeautifulSoup
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.panel import Panel
from rich.live import Live

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

console = Console()

_REQUEST_KW = {"timeout": 15, "verify": True}


def _http_get(url, **kwargs):
    merged = {**_REQUEST_KW, **kwargs}
    try:
        return requests.get(url, **merged)
    except requests.exceptions.SSLError:
        merged["verify"] = False
        return requests.get(url, **merged)


def _http_post(url, **kwargs):
    merged = {**_REQUEST_KW, **kwargs}
    try:
        return requests.post(url, **merged)
    except requests.exceptions.SSLError:
        merged["verify"] = False
        return requests.post(url, **merged)


def load_dotenv():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_dotenv()

DEFAULT_SYSTEM = (
    "You are a helpful assistant with optional web search. "
    "Call web_search only when the user needs current or factual info from the internet "
    "(news, flights, weather, places, prices, sports scores, etc.). "
    "For greetings, chit-chat, opinions, or questions you can answer from knowledge, "
    "reply normally and do NOT call any tool."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chat with a local OpenAI-compatible server"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("OPENAI_HOST", "localhost"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("OPENAI_PORT", "8080")),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="System prompt",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming responses",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    return parser.parse_args()


def _ddg_unwrap(href: str) -> str:
    if "uddg=" in href:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        if qs.get("uddg"):
            return qs["uddg"][0]
    return href


def duckduckgo_search(query: str, count: int = 5) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = _http_get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers=headers,
    )
    if resp.status_code != 200:
        return f"Search failed (HTTP {resp.status_code})"
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for result in soup.select(".result")[:count]:
        a = result.select_one("a.result__a")
        if not a:
            continue
        title = a.get_text(strip=True)
        link = _ddg_unwrap(a.get("href", ""))
        snippet_el = result.select_one(".result__snippet")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        results.append(f"- [{title}]({link}): {snippet}")
    return "\n".join(results) if results else "No search results found."


def firecrawl_search(query: str, count: int = 5) -> str | None:
    key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    if not key:
        return None
    resp = _http_post(
        "https://api.firecrawl.dev/v1/search",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={"query": query, "limit": count},
    )
    if resp.status_code != 200:
        console.print(f"[dim]Firecrawl HTTP {resp.status_code}; falling back[/]")
        return None
    data = resp.json()
    items = data.get("data") or data.get("web") or []
    if not items:
        return None
    results = []
    for item in items[:count]:
        title = item.get("title") or item.get("url") or "result"
        link = item.get("url") or ""
        snippet = item.get("description") or item.get("markdown") or ""
        if isinstance(snippet, str) and len(snippet) > 280:
            snippet = snippet[:277] + "..."
        results.append(f"- [{title}]({link}): {snippet}")
    return "\n".join(results) if results else None


def web_search(query: str, count: int = 5) -> str:
    try:
        fc = firecrawl_search(query, count=count)
        if fc:
            return fc
        return duckduckgo_search(query, count=count)
    except Exception as e:
        return f"Search failed: {e}"


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current or up-to-date information. "
            "Do not use for greetings or simple conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}


def handle_tool_calls(messages, tool_calls) -> bool:
    if not tool_calls:
        return False

    tool_calls_data = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            tool_calls_data.append(tc)
        else:
            tool_calls_data.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )

    messages.append({"role": "assistant", "tool_calls": tool_calls_data, "content": None})

    for tc in tool_calls_data:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
            if name in ("web_search", "bing_search"):
                query = args["query"]
                console.print(f"[dim]Web search: {query}[/]")
                result = web_search(query)
                preview = result.replace("\n", " | ")
                if len(preview) > 160:
                    preview = preview[:157] + "..."
                console.print(f"[dim]Results: {preview}[/]")
            else:
                result = f"Unknown tool: {name}"
                console.print(f"[red]{result}[/]")
        except Exception as e:
            console.print(f"[red]Tool call error: {e}[/]")
            result = f"Error: {e}"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": result,
            }
        )
    return True


def _merge_tool_call_delta(acc: dict, delta_tc) -> None:
    idx = delta_tc.index
    if idx not in acc:
        acc[idx] = {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        }
    entry = acc[idx]
    if delta_tc.id:
        entry["id"] = delta_tc.id
    if delta_tc.function:
        if delta_tc.function.name:
            entry["function"]["name"] += delta_tc.function.name
        if delta_tc.function.arguments:
            entry["function"]["arguments"] += delta_tc.function.arguments


def chat_once(client, args, messages, stream: bool):
    """One model turn → ("tool", calls) or ("content", text). Streams content live."""
    kwargs = dict(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        tools=[SEARCH_TOOL],
        tool_choice="auto",
        stream=stream,
    )

    if not stream:
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        if msg.tool_calls:
            return "tool", list(msg.tool_calls)
        return "content", (msg.content or "").strip()

    stream_resp = client.chat.completions.create(**kwargs)
    tool_acc: dict = {}
    content = ""
    live = None
    try:
        for chunk in stream_resp:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    _merge_tool_call_delta(tool_acc, tc)
                continue
            piece = delta.content or ""
            if not piece or tool_acc:
                continue
            content += piece
            if live is None:
                live = Live(console=console, refresh_per_second=15)
                live.start()
            live.update(
                Panel(Markdown(content), title="[bold green]Assistant[/bold green]")
            )
    finally:
        if live is not None:
            live.stop()
            console.print()

    if tool_acc:
        return "tool", [tool_acc[i] for i in sorted(tool_acc)]
    return "content", content.strip()


def main():
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}/v1"

    client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "not-needed"))
    messages = [{"role": "system", "content": args.system}]
    use_stream = not args.no_stream

    console.print(
        Panel.fit(
            f"[bold cyan]Chat UI[/bold cyan]\n"
            f"Server: [green]{base_url}[/green]\n"
            f"Model:  [yellow]{args.model}[/yellow]\n"
            f"Stream: [{'green' if use_stream else 'red'}]{'on' if use_stream else 'off'}[/]\n"
            f"Type [bold]exit[/] or [bold]Ctrl+C[/] to quit.\n"
            f"Type [bold]/clear[/] to clear history."
        )
    )

    while True:
        try:
            prompt = Prompt.ask("[bold blue]You[/bold blue]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye![/]")
            break

        if not prompt:
            continue
        if prompt.strip().lower() in ("exit", "quit"):
            console.print("[yellow]Goodbye![/]")
            break
        if prompt.strip() == "/clear":
            messages = [messages[0]]
            console.print("[dim]History cleared.[/]")
            continue

        messages.append({"role": "user", "content": prompt})

        while True:
            kind, payload = chat_once(client, args, messages, stream=use_stream)
            if kind == "tool":
                handle_tool_calls(messages, payload)
                continue

            content = payload
            if not content:
                console.print("[red]Assistant returned an empty response.[/]")
                messages.pop()
                break

            if not use_stream:
                console.print("[bold green]Assistant[/bold green]")
                console.print(Markdown(content))

            messages.append({"role": "assistant", "content": content})
            break


if __name__ == "__main__":
    try:
        main()
    except Exception:
        console.print_exception()
        sys.exit(1)
