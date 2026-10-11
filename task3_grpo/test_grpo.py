from __future__ import annotations

import unittest
import math

import torch

from task3_grpo.analyze_group_size import analyze
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences
from task3_grpo.engine import categorical_entropy


class GrpoObjectiveTests(unittest.TestCase):
    def test_advantages_are_normalized_within_each_prompt(self):
        rewards = torch.tensor([0., 2., 10., 10.])
        groups = torch.tensor([0, 0, 1, 1])
        actual = group_relative_advantages(rewards, groups)
        self.assertTrue(torch.allclose(actual, torch.tensor([-1., 1., 0., 0.]), atol=2e-6))

    def test_dr_grpo_uses_fixed_length_and_masked_sequences_have_no_signal(self):
        new = torch.zeros((2, 3), requires_grad=True)
        old = torch.zeros_like(new)
        ref = torch.zeros_like(new)
        advantages = torch.tensor([1., 0.])
        mask = mask_truncated_sequences(torch.tensor([[1., 1., 0.], [1., 1., 1.]]), [False, True])
        canonical, _ = grpo_policy_loss(new, old, advantages, mask, ref, .2, 0)
        dr, _ = grpo_policy_loss(new, old, advantages, mask, ref, .2, 0,
                                 "dr_grpo", max_completion_length=3)
        self.assertAlmostEqual(float(canonical.detach()), -.5)
        self.assertAlmostEqual(float(dr.detach()), -1 / 3)
        canonical.backward()
        self.assertTrue(torch.all(new.grad[1] == 0))

    def test_group_sizes_use_same_generations(self):
        cache = {str(pid): [{"source_index": pid, "generation_index": j, "completion": f"response-{pid}-{j}",
                             "reward": float((pid + j) % 3 == 0)} for j in range(8)]
                 for pid in range(4)}
        report = analyze(cache)
        self.assertEqual([report["conditions"][str(k)]["generation_count"]
                          for k in (2, 4, 8)], [32, 32, 32])
        self.assertEqual([report["conditions"][str(k)]["overall"]["groups"]
                          for k in (2, 4, 8)], [16, 8, 4])

    def test_full_categorical_entropy_respects_response_mask(self):
        logits = torch.tensor([[[0., 0.], [0., math.log(3.)]]])
        mask = torch.tensor([[1., 0.]])
        self.assertAlmostEqual(float(categorical_entropy(logits, mask)), math.log(2.), places=6)
        mask = torch.tensor([[1., 1.]])
        expected = (math.log(2.) - .25 * math.log(.25) - .75 * math.log(.75)) / 2
        self.assertAlmostEqual(float(categorical_entropy(logits, mask)), expected, places=6)

    def test_logged_sampled_kl_is_log_policy_minus_log_reference(self):
        new = torch.tensor([[math.log(.6), math.log(.4)]])
        ref = torch.tensor([[math.log(.3), math.log(.5)]])
        mask = torch.tensor([[1., 0.]])
        _, detail = grpo_policy_loss(new, new, torch.tensor([1.]), mask, ref, .2, .1)
        self.assertAlmostEqual(float(detail["sampled_kl"]), math.log(2.), places=6)
        self.assertGreaterEqual(float(detail["kl_penalty"]), 0.)


if __name__ == "__main__":
    unittest.main()
