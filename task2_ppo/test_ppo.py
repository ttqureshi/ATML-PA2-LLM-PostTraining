"""Regression checks for the clipped PPO surrogate and cache diagnostics."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from common.generation import batch_generate, response_token_logprobs
from task2_ppo.analyze_clipping import affected_token_fraction
from task2_ppo.ppo import ppo_policy_loss


class PPOClippingTests(unittest.TestCase):
    def test_surrogate_uses_minimum_for_both_advantage_signs(self):
        ratio = torch.tensor([[1.5, 0.5]])
        advantage = torch.tensor([[1.0, -1.0]])
        mask = torch.ones_like(ratio)

        loss, observed_ratio, clip_fraction = ppo_policy_loss(
            ratio.log(), torch.zeros_like(ratio), advantage, mask, eps=0.2
        )

        # min(1.5, 1.2) = 1.2; min(-0.5, -0.8) = -0.8.
        self.assertAlmostEqual(loss.item(), -0.2, places=6)
        self.assertTrue(torch.allclose(observed_ratio, ratio))
        self.assertAlmostEqual(clip_fraction.item(), 1.0)

    def test_affected_tokens_are_subset_of_outside_range_tokens(self):
        ratio = torch.tensor([[1.5, 0.5, 1.5, 0.5, 1.5]])
        advantage = torch.tensor([[1.0, -1.0, -1.0, 1.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]])

        _, observed_ratio, clip_fraction = ppo_policy_loss(
            ratio.log(), torch.zeros_like(ratio), advantage, mask, eps=0.2
        )
        affected = affected_token_fraction(observed_ratio, advantage, mask, eps=0.2)

        self.assertAlmostEqual(clip_fraction.item(), 1.0)
        self.assertAlmostEqual(affected.item(), 0.5)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 3

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "prompt"

    def __call__(self, rendered, return_tensors, padding, truncation, max_length):
        ids = torch.tensor([[1, 2]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, token_ids, skip_special_tokens=True):
        return "response"


class TinyGeneratingLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Embedding(6, 6)

    def generate(self, input_ids, **kwargs):
        response = torch.tensor([[3, 4]], device=input_ids.device)
        return torch.cat((input_ids, response), dim=1)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.logits(input_ids))


class PPOGenerationTests(unittest.TestCase):
    def test_generated_tokens_can_be_used_in_gradient_tracked_policy_update(self):
        model = TinyGeneratingLM()
        generated = batch_generate(
            model, TinyTokenizer(), [[{"role": "user", "content": "x"}]],
            max_prompt_length=8, max_new_tokens=2, do_sample=False,
        )

        self.assertFalse(torch.is_inference(generated["sequences"]))
        self.assertFalse(torch.is_inference(generated["response_ids"]))
        logprobs, _ = response_token_logprobs(
            model, generated["sequences"], generated["attention_mask"],
            generated["prompt_width"], generated["response_ids"],
        )
        logprobs.sum().backward()
        self.assertTrue(torch.isfinite(model.logits.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
