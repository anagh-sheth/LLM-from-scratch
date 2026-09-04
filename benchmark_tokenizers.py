"""Compare character and byte-pair tokenization on a text corpus."""

import argparse
from pathlib import Path

from tokenizer import BytePairTokenizer, CharTokenizer


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_dir / "input.txt")
    parser.add_argument("--bpe-vocab-size", type=int, default=320)
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    character_tokenizer = CharTokenizer.from_text(text)
    bpe_tokenizer = BytePairTokenizer.train(text, args.bpe_vocab_size)
    character_tokens = len(character_tokenizer.encode(text))
    bpe_tokens = len(bpe_tokenizer.encode(text))
    reduction = 100 * (1 - bpe_tokens / character_tokens)

    print("tokenizer  vocabulary  tokens     chars/token")
    print(
        f"character  {character_tokenizer.vocab_size:>10,}  "
        f"{character_tokens:>10,}  {len(text) / character_tokens:>11.3f}"
    )
    print(
        f"BPE        {bpe_tokenizer.vocab_size:>10,}  "
        f"{bpe_tokens:>10,}  {len(text) / bpe_tokens:>11.3f}"
    )
    print(f"BPE sequence-length reduction: {reduction:.1f}%")


if __name__ == "__main__":
    main()
