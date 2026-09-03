# LLM from Scratch

A small decoder-only transformer trained for character-level next-token
prediction on Tiny Shakespeare.

## Architecture

- Character-level tokenizer
- Learned token and positional embeddings
- Masked multi-head self-attention
- Pre-layer-normalized residual transformer blocks
- Position-wise feed-forward networks
- Autoregressive sampling with temperature and optional top-k filtering
- Training checkpoints and resume support

The default configuration: a 256-token context window,
384-dimensional embeddings, 6 attention heads, and 6 transformer blocks.
