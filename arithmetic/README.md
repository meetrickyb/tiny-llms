# Arithmetic

A GPT with 400,000 parameters that adds two numbers of up to three digits.

The trained model is named `gpt-0.5-math`.

## Usage

With the virtual environment from the [repo README](../README.md) activated,
from this folder:

```bash
# Train (under three minutes on a laptop CPU), or download the trained model instead
python train.py
mkdir -p out && curl -L -o out/gpt-0.5-math.pt \
  https://github.com/meetrickyb/tiny-llms/releases/download/arithmetic-v1/gpt-0.5-math.pt

# Ask it interactively
python calculate.py --model gpt-0.5-math

# One sum
python calculate.py --model gpt-0.5-math --prompt "347 + 589"

# Score it on 10,000 problems held out of training
python calculate.py --test 10000
```

## Options

| Command | Effect |
|---|---|
| `train.py --digits 4` | Train on numbers of up to four digits |
| `train.py --n-layer 1 --n-embd 32` | Train a smaller model |
| `train.py --device mps` | Train on the Apple Silicon GPU instead of the CPU |
| `train.py --out out/NAME.pt` | Save the model under another name |
| `calculate.py --model NAME` | Use `out/NAME.pt`; the default is `gpt-0.5-math` |

## Files

| File | Contents |
|---|---|
| [data.py](data.py) | Sums as tokens, and the held-out split |
| [model.py](model.py) | The transformer |
| [train.py](train.py) | The training loop |
| [calculate.py](calculate.py) | Getting an answer from the trained model |
