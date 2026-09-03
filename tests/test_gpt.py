import unittest

import torch

from gpt import CharTokenizer, GPTConfig, GPTLanguageModel


class CharTokenizerTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        tokenizer = CharTokenizer.from_text("cab cab")
        text = "a cab"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_unknown_character_has_clear_error(self) -> None:
        tokenizer = CharTokenizer.from_text("abc")
        with self.assertRaisesRegex(ValueError, "not in the tokenizer vocabulary"):
            tokenizer.encode("z")


class GPTLanguageModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1337)
        self.config = GPTConfig(
            vocab_size=10,
            block_size=8,
            n_embd=16,
            n_head=4,
            n_layer=2,
            dropout=0.0,
        )
        self.model = GPTLanguageModel(self.config)
        self.model.eval()

    def test_forward_shape_and_loss(self) -> None:
        tokens = torch.randint(self.config.vocab_size, (2, 6))
        logits, loss = self.model(tokens, tokens)
        self.assertEqual(logits.shape, (2, 6, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertEqual(loss.ndim, 0)

    def test_future_tokens_do_not_change_past_logits(self) -> None:
        first = torch.tensor([[1, 2, 3, 4, 5]])
        second = torch.tensor([[1, 2, 3, 8, 9]])
        first_logits, _ = self.model(first)
        second_logits, _ = self.model(second)
        torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3])

    def test_generation_can_exceed_context_window(self) -> None:
        prompt = torch.tensor([[1, 2, 3]])
        generated = self.model.generate(prompt, max_new_tokens=10)
        self.assertEqual(generated.shape, (1, 13))

    def test_rejects_sequences_larger_than_context_window(self) -> None:
        tokens = torch.randint(self.config.vocab_size, (1, 9))
        with self.assertRaisesRegex(ValueError, "exceeds block size"):
            self.model(tokens)


if __name__ == "__main__":
    unittest.main()
