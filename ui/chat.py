#!/usr/bin/env python3
"""Simple chat UI for a local OpenAI-compatible server."""

import os
import sys
import traceback
import argparse
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

console = Console()


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
        default="You are a helpful assistant.",
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


def main():
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}/v1"

    client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "not-needed"))
    messages = [{"role": "system", "content": args.system}]

    console.print(
        Panel.fit(
            f"[bold cyan]Chat UI[/bold cyan]\n"
            f"Server: [green]{base_url}[/green]\n"
            f"Model:  [yellow]{args.model}[/yellow]\n"
            f"Stream: [{'red' if args.no_stream else 'green'}]{'off' if args.no_stream else 'on'}[/]\n"
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

        if args.no_stream:
            console.print("[bold green]Assistant[/bold green]")
            with console.status("[dim]thinking...[/]"):
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    stream=False,
                )
            content = resp.choices[0].message.content or ""
            console.print(Markdown(content))
        else:
            content = ""
            with Live(console=console, refresh_per_second=15) as live:
                live.update(Text("[bold green]Assistant[/bold green]\n"))
                stream = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    stream=True,
                )
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta.content or ""
                    content += delta
                    live.update(
                        Panel(Markdown(content), title="[bold green]Assistant[/bold green]")
                    )
            console.print()

        messages.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        console.print_exception()
        sys.exit(1)
