# tiny-llms

Small language models built in plain Python and trained on a laptop (CPU, or
the GPU in an Apple Silicon Mac), to show how LLMs are made. Each folder is a self-contained project.

| Folder | What it is |
|---|---|
| [wikipedia](wikipedia/) | A tiny GPT trained from scratch on Wikipedia's articles about the history of Canada |
| [chat-app](chat-app/) | A chat app that uses that model as its LLM, through the same API as a hosted one |

## Setup

One virtual environment at the repo root is shared by all projects.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r wikipedia/requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead.
