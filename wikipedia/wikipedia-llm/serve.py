"""Serve the trained model over HTTP, speaking the same API as hosted LLMs.

Most apps reach a language model through the OpenAI chat-completions API:
they POST a list of messages and read the reply back. This server implements
just enough of that API for such an app to use the tiny model instead, by
changing nothing but the URL it points at.

A model trained only by train.py has never seen a conversation, so the server
treats the last user message as the title of a Wikipedia article and replies
with the opening paragraph the model writes for it. A model fine-tuned on the
pairs from prepare_qa.py answers questions; name it after --qa to prompt it
that way.

Several checkpoints can be served at once. Each is a model named after its
file, and the app picks one by name, as it would with a hosted provider.

    python wikipedia-llm/serve.py --checkpoint out/stage1.pt out/gpt-0.5.pt --qa gpt-0.5
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from generate import load_model, pick_device, stream
from model import GPT
from prepare_qa import format_prompt
from tokenizer import Tokenizer


def last_user_message(messages: list[dict[str, Any]]) -> str:
    """Pull the text of the most recent user message out of a chat request."""
    for message in reversed(messages):
        if message["role"] == "user":
            content = message["content"]
            # Content is either a plain string or a list of typed parts.
            if isinstance(content, list):
                content = " ".join(part["text"] for part in content if part.get("type") == "text")
            return content.strip()
    raise KeyError("no user message")


def reply(
    model: GPT, tokenizer: Tokenizer, prompt: str, max_tokens: int, temperature: float
) -> Iterator[str]:
    """Yield the first paragraph the model writes after `prompt`."""
    pending = ""
    started = False
    for piece in stream(model, tokenizer, prompt, max_tokens, temperature, new_article=True):
        pending += piece
        if not started:
            pending = pending.lstrip()
        end = pending.find("\n\n")
        if end != -1:
            if end:
                yield pending[:end]
            return
        # A paragraph break can arrive split across two tokens, so trailing
        # newlines are held back until the next piece shows what follows.
        ready = pending.rstrip("\n")
        if ready:
            started = True
            yield ready
            pending = pending[len(ready) :]


class Handler(BaseHTTPRequestHandler):
    """Answers the two endpoints a chat client needs: list models, complete a chat."""

    # Model name -> (model, tokenizer, whether it was fine-tuned on question-answer pairs)
    models: dict[str, tuple[GPT, Tokenizer, bool]]
    max_tokens: int
    # One request is handled at a time.
    lock = threading.Lock()

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            models = [{"id": name, "object": "model", "created": 0, "owned_by": "tiny-llms"} for name in self.models]
            self.send_json(200, {"object": "list", "data": models})
        else:
            self.send_error_json(404, f"unknown path {self.path}")

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error_json(404, f"unknown path {self.path}")
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            message = last_user_message(body["messages"])
            model_name = body.get("model") or next(iter(self.models))
            max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or self.max_tokens)
            # A temperature of exactly 0 would divide by zero when sampling.
            temperature = max(float(body.get("temperature", 0.8)), 0.05)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            self.send_error_json(400, f"malformed request: {error!r}")
            return
        if model_name not in self.models:
            self.send_error_json(404, f"no model named {model_name!r}; available: {', '.join(self.models)}")
            return
        model, tokenizer, qa = self.models[model_name]

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        base = {"id": completion_id, "created": int(time.time()), "model": model_name}
        with self.lock:
            prompt = format_prompt(message) if qa else f"{message}\n\n"
            pieces = reply(model, tokenizer, prompt, max_tokens, temperature)
            if body.get("stream"):
                self.stream_chunks(base, pieces)
                return
            text = "".join(pieces)
        prompt_tokens = len(tokenizer.encode(message))
        completion_tokens = len(tokenizer.encode(text))
        self.send_json(
            200,
            {
                **base,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        )

    def stream_chunks(self, base: dict[str, Any], pieces: Iterator[str]) -> None:
        """Send the reply as server-sent events, one per piece, as hosted APIs do."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(delta: dict[str, str], finish_reason: str | None = None) -> None:
            choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
            chunk = {**base, "object": "chat.completion.chunk", "choices": [choice]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        try:
            send({"role": "assistant", "content": ""})
            for piece in pieces:
                send({"content": piece})
            send({}, "stop")
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client stopped listening

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json(status, {"error": {"message": message, "type": "invalid_request_error"}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=[Path("out/stage1.pt")])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=120, help="reply length when the app does not set one")
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda or auto")
    parser.add_argument(
        "--qa",
        nargs="*",
        default=None,
        metavar="NAME",
        help="models fine-tuned on question-answer pairs; with no names, all of them",
    )
    args = parser.parse_args()

    names = [path.stem for path in args.checkpoint]
    qa_names = set(names if args.qa == [] else args.qa or [])
    if unknown := qa_names - set(names):
        parser.error(f"--qa names models that are not being served: {', '.join(sorted(unknown))}")

    device = pick_device(args.device)
    Handler.models = {}
    Handler.max_tokens = args.max_tokens
    for name, path in zip(names, args.checkpoint):
        model, tokenizer = load_model(path, device)
        Handler.models[name] = (model, tokenizer, name in qa_names)
        style = "answers questions" if name in qa_names else "continues text"
        print(f"  {name}: {model.num_parameters():,} parameters, {style}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"OpenAI-compatible API at http://{args.host}:{args.port}/v1  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
