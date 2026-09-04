# LLM from Scratch

A small decoder-only transformer trained for next-token prediction on Tiny
Shakespeare. It supports both character-level and custom byte-level BPE
tokenization.

## Architecture

- Character-level baseline and custom byte-level BPE tokenizer
- Learned token and positional embeddings
- Masked multi-head self-attention
- Pre-layer-normalized residual transformer blocks
- Position-wise feed-forward networks
- Autoregressive sampling with temperature and optional top-k filtering
- Training checkpoints and resume support

The default configuration uses a 256-token context window, 384-dimensional
embeddings, 6 attention heads, and 6 transformer blocks.

## Tokenizer benchmark

The custom BPE implementation starts with the 256 possible byte values and
learns frequent adjacent-token merges directly from the training corpus. On the
checked-in Tiny Shakespeare dataset:

| Tokenizer | Vocabulary | Encoded tokens | Characters/token | Sequence reduction |
| --- | ---: | ---: | ---: | ---: |
| Character | 65 | 1,115,394 | 1.000 | baseline |
| Byte BPE | 320 | 743,756 | 1.500 | 33.3% |

Run `python benchmark_tokenizers.py --bpe-vocab-size 320` to reproduce the
comparison. The measured improvement is a 33.3% sequence-length reduction, not
a vocabulary-size reduction.

## Roadmap

- Add training metrics and tokenizer comparisons
- Implement LoRA adapters
- Fine-tune on conversational data
- Build and publish a Hugging Face chatbot demo

## Acknowledgment

The transformer architecture and teaching progression are adapted from Andrej
Karpathy's MIT-licensed
[`ng-video-lecture`](https://github.com/karpathy/ng-video-lecture) repository.
