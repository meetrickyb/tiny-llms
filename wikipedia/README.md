# Tiny LLM trained on Wikipedia

A GPT-style language model, built and trained from scratch on a laptop CPU,
using Wikipedia's articles about the history of Canada as its only source of
knowledge about the world.

## What to expect

The model learns to write text that *looks like* a Wikipedia article about
Canadian history: the right vocabulary, plausible dates and place names,
mostly grammatical sentences. The facts are invented. A few million parameters
and a few megabytes of text are enough to learn the shape of the language, not
to store knowledge reliably. That gap is the reason real LLMs are pretrained
on a large share of the internet, and the reason "chat with your documents"
products use retrieval rather than training.

## Run it

Run everything from this folder, with the virtual environment from the
[repo README](../README.md) activated.

```bash
# 1. Download the articles (about 3,000, resumable)
python download_wiki.py --category "History of Canada"

# 2. Train the tokenizer and encode the text
python stage1_from_scratch/prepare.py

# 3. Train the model
python stage1_from_scratch/train.py

# 4. Talk to it
python stage1_from_scratch/generate.py
```

Any category works: pass a different `--category` to step 1.

Training speed depends heavily on the CPU. For a quick first run on a slow
machine, shrink the job:

```bash
python stage1_from_scratch/train.py --max-iters 2000 --batch-size 16 --n-embd 128
```

## Reading order

| File | What it shows |
|---|---|
| [download_wiki.py](download_wiki.py) | Where training data comes from |
| [stage1_from_scratch/tokenizer.py](stage1_from_scratch/tokenizer.py) | How text becomes numbers (byte-pair encoding) |
| [stage1_from_scratch/prepare.py](stage1_from_scratch/prepare.py) | Building the training and validation sets |
| [stage1_from_scratch/model.py](stage1_from_scratch/model.py) | The transformer: embeddings, attention, feed-forward layers |
| [stage1_from_scratch/train.py](stage1_from_scratch/train.py) | The training loop: predict, measure the error, adjust |
| [stage1_from_scratch/generate.py](stage1_from_scratch/generate.py) | Sampling text one token at a time |

## Things to try

- `--temperature 0.2` versus `--temperature 1.5` in `generate.py`.
- A larger model: `train.py --n-layer 6 --n-embd 256 --block-size 256`.
- Watch the gap between train loss and validation loss. When training loss
  keeps falling but validation loss rises, the model has started memorising.

## Data licence

The training text is from [Wikipedia](https://en.wikipedia.org/) and is
licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The articles and trained checkpoints are not committed to this repo.
