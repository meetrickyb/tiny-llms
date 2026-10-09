# Chat app

A terminal chat app that uses the model trained in [wikipedia](../wikipedia/)
as its LLM, in place of a hosted frontier model.

The point is what the app does *not* contain. [chat.py](chat.py) is written
the way apps for hosted LLMs are written: it uses the official OpenAI client
library, sends the conversation to a URL, and prints the reply as it streams
back. It has no idea how small the model on the other end is. Swapping a
frontier model for the tiny one is a change of URL, not a change of code.

```
chat.py  --HTTP, OpenAI chat API-->  serve.py  -->  the tiny GPT
```

## Run it

With the virtual environment from the [repo README](../README.md) activated:

```bash
pip install -r chat-app/requirements.txt
```

Start the model server in one terminal. It needs a checkpoint, either trained
or downloaded as described in the [wikipedia README](../wikipedia/README.md):

```bash
cd wikipedia
python wikipedia-llm/serve.py --checkpoint out/gpt-0.5.pt --qa
```

Chat with it from another, naming the model as you would with a hosted one:

```bash
python chat-app/chat.py --model gpt-0.5
```

The server names each model after its checkpoint file, and can serve several
at once (`--checkpoint out/gpt-0.5.pt out/stage1.pt --qa gpt-0.5`). Without
`--model`, the app uses the first one the server lists.

## Point it at a different LLM

Any server that speaks the OpenAI chat API works. For example, a model served
locally by [Ollama](https://ollama.com):

```bash
python chat-app/chat.py --base-url http://localhost:11434/v1 --model llama3.2
```

For a hosted provider, pass its base URL and model name, and put your key in
the `OPENAI_API_KEY` environment variable.

Asking both the same question is the quickest way to see what a few million
parameters and a few megabytes of text buy, next to billions of parameters
and a large share of the internet.

## What to expect

The tiny model answers in the shape of an answer: a confident sentence or two
of encyclopedia prose about roughly the right subject. For subjects it was
fine-tuned on it often recalls real fragments; the rest is invented. It also
reads only your latest message, since its training pairs were single
questions, so it does not follow a conversation.
