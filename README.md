# LLM from Scratch

A small decoder-only transformer trained for character-level next-token
prediction on Tiny Shakespeare. The architecture is based on Andrej Karpathy's
[Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture and its
[official code](https://github.com/karpathy/ng-video-lecture), then organized
into a reusable training and inference program.

This is an educational GPT implementation, not yet an instruction-tuned
chatbot. `bigram.py` is the introductory baseline and `gpt.py` is the complete
transformer starting point.

## Architecture

- Character-level tokenizer
- Learned token and positional embeddings
- Masked multi-head self-attention
- Pre-layer-normalized residual transformer blocks
- Position-wise feed-forward networks
- Autoregressive sampling with temperature and optional top-k filtering
- Training checkpoints and resume support

The default configuration follows the lecture: a 256-token context window,
384-dimensional embeddings, 6 attention heads, and 6 transformer blocks.

## Setup

Python 3.10-3.12 is recommended. PyTorch wheels are not always available for
the newest Python release on every platform. For example, this repository's
original Intel macOS environment uses Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Quick smoke test

The full lecture configuration can take a long time on CPU. Start with a small
model to verify the entire training and checkpoint pipeline:

```bash
python gpt.py \
  --max-iters 20 \
  --eval-interval 10 \
  --eval-iters 5 \
  --batch-size 16 \
  --block-size 64 \
  --n-embd 96 \
  --n-head 4 \
  --n-layer 2 \
  --max-new-tokens 100 \
  --checkpoint checkpoints/smoke.pt
```

Poor output is expected after only 20 iterations; this command verifies that
training, evaluation, generation, and checkpoint saving all work.

## Train the full model

```bash
python gpt.py
```

The checkpoint is saved to `checkpoints/gpt.pt`. To continue training it for
another 5,000 iterations:

```bash
python gpt.py --resume
```

The architecture flags are read from the checkpoint when resuming. Training
arguments such as learning rate, batch size, and iteration count can still be
changed.

## Generate text from a checkpoint

```bash
python gpt.py \
  --mode generate \
  --checkpoint checkpoints/gpt.pt \
  --prompt "ROMEO:" \
  --max-new-tokens 500 \
  --temperature 0.8 \
  --top-k 20
```

The prompt can only contain characters that appeared in the training corpus.

## Tests

```bash
python -m unittest discover -s tests
```

The tests verify tokenizer round trips, model dimensions, autoregressive
generation, and the causal-attention guarantee that future tokens cannot alter
earlier logits.

## Roadmap

- Replace character tokenization with a custom BPE tokenizer
- Add training metrics and tokenizer comparisons
- Implement LoRA adapters
- Fine-tune on conversational data
- Build and publish a Hugging Face chatbot demo

## Acknowledgment

The transformer architecture and teaching progression are adapted from Andrej
Karpathy's MIT-licensed `ng-video-lecture` repository. This project adds its own
module boundaries, validation, command-line interface, checkpoints, tests, and
sampling controls.
