"""Regression checks for the clipped PPO surrogate and cache diagnostics."""
from __future__ import annotations

import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
