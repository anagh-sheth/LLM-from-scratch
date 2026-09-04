"""Tokenizers used by the language model.

The byte-pair tokenizer is intentionally implemented without a tokenization
library so the merge-training and encoding algorithms remain visible.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol


class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...

    def to_dict(self) -> dict[str, object]: ...


class CharTokenizer:
    """A reversible character-level tokenizer learned from a text corpus."""

    def __init__(self, chars: list[str]) -> None:
        if not chars:
            raise ValueError("Tokenizer vocabulary cannot be empty")
        self.chars = sorted(set(chars))
        self.stoi = {char: index for index, char in enumerate(self.chars)}
        self.itos = {index: char for char, index in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(list(text))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[char] for char in text]
        except KeyError as exc:
            raise ValueError(
                f"Character {exc.args[0]!r} is not in the tokenizer vocabulary"
            ) from exc

    def decode(self, token_ids: list[int]) -> str:
        try:
            return "".join(self.itos[token_id] for token_id in token_ids)
        except KeyError as exc:
            raise ValueError(f"Unknown token id: {exc.args[0]}") from exc

    def to_dict(self) -> dict[str, object]:
        return {"type": "char", "chars": self.chars}


def _pair_counts(token_ids: list[int]) -> Counter[tuple[int, int]]:
    return Counter(zip(token_ids, token_ids[1:]))


def _merge_pair(
    token_ids: list[int], pair: tuple[int, int], replacement: int
) -> list[int]:
    merged: list[int] = []
    index = 0
    while index < len(token_ids):
        if (
            index < len(token_ids) - 1
            and token_ids[index] == pair[0]
            and token_ids[index + 1] == pair[1]
        ):
            merged.append(replacement)
            index += 2
        else:
            merged.append(token_ids[index])
            index += 1
    return merged


class BytePairTokenizer:
    """A minimal byte-level BPE tokenizer with a learned merge table.

    Raw UTF-8 bytes occupy token IDs 0-255, so any input can be represented
    without an unknown token. Learned byte sequences are assigned IDs from 256
    upward in the order their pairs are merged.
    """

    BASE_VOCAB_SIZE = 256

    def __init__(self, merges: dict[tuple[int, int], int] | None = None) -> None:
        self.merges = merges or {}
        self.vocabulary: dict[int, bytes] = {
            token_id: bytes([token_id]) for token_id in range(self.BASE_VOCAB_SIZE)
        }
        for pair, token_id in sorted(self.merges.items(), key=lambda item: item[1]):
            if pair[0] not in self.vocabulary or pair[1] not in self.vocabulary:
                raise ValueError(f"Invalid BPE merge dependency: {pair}")
            self.vocabulary[token_id] = (
                self.vocabulary[pair[0]] + self.vocabulary[pair[1]]
            )

    @classmethod
    def train(cls, text: str, vocab_size: int = 320) -> "BytePairTokenizer":
        if not text:
            raise ValueError("Cannot train a tokenizer on empty text")
        if vocab_size < cls.BASE_VOCAB_SIZE:
            raise ValueError(
                f"BPE vocabulary size must be at least {cls.BASE_VOCAB_SIZE}"
            )

        token_ids = list(text.encode("utf-8"))
        merges: dict[tuple[int, int], int] = {}
        for token_id in range(cls.BASE_VOCAB_SIZE, vocab_size):
            counts = _pair_counts(token_ids)
            if not counts:
                break
            pair = max(counts, key=lambda candidate: (counts[candidate], candidate))
            if counts[pair] < 2:
                break
            merges[pair] = token_id
            token_ids = _merge_pair(token_ids, pair, token_id)
        return cls(merges)

    @property
    def vocab_size(self) -> int:
        return self.BASE_VOCAB_SIZE + len(self.merges)

    def encode(self, text: str) -> list[int]:
        token_ids = list(text.encode("utf-8"))
        while len(token_ids) >= 2:
            counts = _pair_counts(token_ids)
            pair = min(
                counts,
                key=lambda candidate: self.merges.get(candidate, float("inf")),
            )
            if pair not in self.merges:
                break
            token_ids = _merge_pair(token_ids, pair, self.merges[pair])
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        try:
            encoded = b"".join(self.vocabulary[token_id] for token_id in token_ids)
        except KeyError as exc:
            raise ValueError(f"Unknown token id: {exc.args[0]}") from exc
        return encoded.decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, object]:
        serialized_merges = [
            [left, right, token_id]
            for (left, right), token_id in sorted(
                self.merges.items(), key=lambda item: item[1]
            )
        ]
        return {"type": "bpe", "merges": serialized_merges}


def tokenizer_from_dict(state: dict[str, object]) -> CharTokenizer | BytePairTokenizer:
    tokenizer_type = state.get("type")
    if tokenizer_type == "char":
        chars = state.get("chars")
        if not isinstance(chars, list) or not all(isinstance(char, str) for char in chars):
            raise ValueError("Invalid character tokenizer state")
        return CharTokenizer(chars)
    if tokenizer_type == "bpe":
        serialized_merges = state.get("merges")
        if not isinstance(serialized_merges, list):
            raise ValueError("Invalid BPE tokenizer state")
        merges: dict[tuple[int, int], int] = {}
        for merge in serialized_merges:
            if (
                not isinstance(merge, list)
                or len(merge) != 3
                or not all(isinstance(value, int) for value in merge)
            ):
                raise ValueError("Invalid BPE merge entry")
            left, right, token_id = merge
            merges[(left, right)] = token_id
        return BytePairTokenizer(merges)
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type!r}")
