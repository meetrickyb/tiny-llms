"""A terminal chat app, written the way apps for hosted LLMs are written.

It uses the official OpenAI client library and knows nothing about the tiny
model: it sends the conversation to whatever server `--base-url` points at
and prints the reply as it streams back. Pointed at `wikipedia-llm/serve.py`
it talks to the model trained in this repo; pointed at a hosted provider it
talks to a frontier model. The code is the same either way.

    python chat.py
"""

from __future__ import annotations

import argparse
import os

from openai import APIConnectionError, OpenAI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="where the LLM is served")
    parser.add_argument("--model", default=None, help="model name; defaults to the first one the server lists")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    # A hosted provider needs a real key; the local server ignores it.
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "not-needed"))
    try:
        available = [served.id for served in client.models.list().data]
    except APIConnectionError:
        raise SystemExit(f"No LLM server is answering at {args.base_url}. Start one first.")
    model = args.model or available[0]
    if model not in available:
        raise SystemExit(f"The server has no model named {model!r}. It serves: {', '.join(available)}")

    print(f"Chatting with {model} at {args.base_url}. Empty line to quit.")
    messages: list[dict[str, str]] = []
    while True:
        try:
            text = input("\nyou> ").strip()
        except EOFError:
            break
        if not text:
            break
        messages.append({"role": "user", "content": text})

        print(f"{model}> ", end="", flush=True)
        reply = ""
        chunks = client.chat.completions.create(
            model=model, messages=messages, temperature=args.temperature, stream=True
        )
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                reply += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
        # The whole conversation is sent back each turn; that is the only memory an LLM has.
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
