from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from task1_dpo.dpo import dpo_loss


class DPOObjectiveTests(unittest.TestCase):
    def test_matches_assignment_equation_and_margin_accuracy(self):
        pc = torch.tensor([-3.0, -6.0], requires_grad=True)
        pr = torch.tensor([-4.0, -3.0])
        rc = torch.tensor([-5.0, -4.0])
        rr = torch.tensor([-5.0, -5.0])
        beta = 0.3
        margin = (pc - rc) - (pr - rr)
        expected = -F.logsigmoid(beta * margin).mean()
        actual, diagnostics = dpo_loss(pc, pr, rc, rr, beta)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            diagnostics["preference_accuracy"], (margin > 0).float().mean()
        )
        actual.backward()
        self.assertTrue(torch.isfinite(pc.grad).all())

    def test_identical_policy_and_reference_has_zero_margin(self):
        chosen = torch.tensor([-2.0, -1.0])
        rejected = torch.tensor([-3.0, -4.0])
        loss, diagnostics = dpo_loss(
            chosen, rejected, chosen, rejected, beta=0.1
        )
        torch.testing.assert_close(loss, torch.tensor(float("0.6931471805599453")))
        self.assertEqual(float(diagnostics["preference_accuracy"]), 0.0)


if __name__ == "__main__":
    unittest.main()
