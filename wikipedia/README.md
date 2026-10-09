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
python wikipedia-llm/prepare.py

# 3. Train the model
python wikipedia-llm/train.py

# 4. Talk to it
python wikipedia-llm/generate.py
```

Any category works: pass a different `--category` to step 1.

On an Apple Silicon Mac, `train.py` uses the built-in GPU (MPS) automatically,
which is about twice as fast as the CPU for this model: roughly 0.1 s per step
on an M1, so the default 5,000 steps take under ten minutes. Pass
`--device cpu` to force the CPU.

Elsewhere, training speed depends heavily on the CPU. For a quick first run on
a slow machine, shrink the job:

```bash
python wikipedia-llm/train.py --max-iters 2000 --batch-size 16 --n-embd 128
```

## Teach it to answer questions

The model from step 3 continues text. Asked "What is the Hudson's Bay
Company?", it carries on as if that were the first line of an article,
because it has never seen a question being answered. Fine-tuning changes
that: a short second round of training, starting from the trained weights, on
text shaped like the behaviour we want.

Fine-tuning is where the validation loss earns its keep. It bottoms out
within a few hundred steps and then climbs while the training loss keeps
falling: the model has stopped learning how to answer and started reciting
the answers it was shown. `train.py` saves the checkpoint from the lowest
point, not the last step.

```bash
# 5. Build question-and-answer pairs from the articles
python wikipedia-llm/prepare_qa.py

# 6. Fine-tune the trained model on them
python wikipedia-llm/train.py --init-from out/stage1.pt --data-dir data/qa \
    --out out/gpt-0.5.pt --max-iters 3000 --lr 3e-4 \
    --sample-prompt "Question: What is the Hudson's Bay Company? Answer:"

# 7. Serve it over the same API that hosted LLMs use
python wikipedia-llm/serve.py --checkpoint out/gpt-0.5.pt --qa
```

With the server running, the [chat app](../chat-app/) talks to it exactly as
it would talk to a frontier model.

## Skip the training

Trained checkpoints are attached to the
[releases](https://github.com/meetrickyb/tiny-llms/releases). A checkpoint is
one file holding the weights, the model size and the tokenizer, so steps 1 to 3
are not needed to use it:

```bash
mkdir -p out
curl -L -o out/gpt-0.5.pt \
  https://github.com/meetrickyb/tiny-llms/releases/download/wikipedia-v1/gpt-0.5.pt
python wikipedia-llm/serve.py --checkpoint out/gpt-0.5.pt --qa
```

| File | Model | Trained with |
|---|---|---|
| `gpt-0.5.pt` | 5.9M parameters, answers questions | `wikipedia-gpt-large.pt` fine-tuned as in step 6 |
| `wikipedia-gpt-large.pt` | 5.9M parameters, continues text | `train.py --n-layer 6 --n-embd 256 --block-size 256` |
| `wikipedia-gpt-small.pt` | 2.6M parameters, continues text | the default `train.py` settings |

The two that continue text work with `generate.py --checkpoint`.

## Reading order

| File | What it shows |
|---|---|
| [download_wiki.py](download_wiki.py) | Where training data comes from |
| [wikipedia-llm/tokenizer.py](wikipedia-llm/tokenizer.py) | How text becomes numbers (byte-pair encoding) |
| [wikipedia-llm/prepare.py](wikipedia-llm/prepare.py) | Building the training and validation sets |
| [wikipedia-llm/model.py](wikipedia-llm/model.py) | The transformer: embeddings, attention, feed-forward layers |
| [wikipedia-llm/train.py](wikipedia-llm/train.py) | The training loop: predict, measure the error, adjust |
| [wikipedia-llm/generate.py](wikipedia-llm/generate.py) | Sampling text one token at a time |
| [wikipedia-llm/prepare_qa.py](wikipedia-llm/prepare_qa.py) | Making fine-tuning data: question-and-answer pairs |
| [wikipedia-llm/serve.py](wikipedia-llm/serve.py) | Putting the model behind the API that apps expect |

## Things to try

- `--temperature 0.2` versus `--temperature 1.5` in `generate.py`.
- A larger model: `train.py --n-layer 6 --n-embd 256 --block-size 256`.
- Watch the gap between train loss and validation loss. When training loss
  keeps falling but validation loss rises, the model has started memorising.

## Data licence

The training text is from [Wikipedia](https://en.wikipedia.org/) and is
licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The articles and trained checkpoints are not committed to this repo.
