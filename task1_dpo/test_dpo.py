from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.generation import response_sequence_logprobs
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


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.logit_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.logit_table(input_ids))


class ResponseLogprobTests(unittest.TestCase):
    def test_response_only_computation_matches_full_sequence_values_and_gradients(self):
        torch.manual_seed(31)
        model = TinyLM(vocab_size=11)
        ids = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "response_mask": torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]], dtype=torch.float32),
        }

        optimized_score, optimized_tokens, mask = response_sequence_logprobs(model, batch)
        optimized_loss = optimized_score.sum()
        optimized_grad, = torch.autograd.grad(optimized_loss, (model.logit_table.weight,), retain_graph=True)

        logits = model(ids).logits[:, :-1, :]
        labels = ids[:, 1:]
        full_token_logps = F.log_softmax(logits.float(), dim=-1).gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1)
        reference_score = (full_token_logps * batch["response_mask"][:, 1:]).sum(-1)
        reference_grad, = torch.autograd.grad(reference_score.sum(), (model.logit_table.weight,))

        torch.testing.assert_close(optimized_score, reference_score)
        torch.testing.assert_close(optimized_tokens * mask, full_token_logps * mask)
        torch.testing.assert_close(optimized_grad, reference_grad)


if __name__ == "__main__":
    unittest.main()
