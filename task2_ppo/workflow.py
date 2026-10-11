"""Shared, fixed PPO rollout and evaluation calculations."""
from __future__ import annotations

import statistics

import torch

from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.metrics import masked_mean, sampled_kl
from common.models import reference_mode, token_values


def prompt_schedule(n_rows: int, seed: int, count: int) -> list[int]:
    """Deterministic prompt IDs; every fork sees the same order."""
    import random

    if n_rows <= 0:
        raise ValueError("PPO prompt pool is empty")
    rng = random.Random(seed)
    result = []
    while len(result) < count:
        cycle = list(range(n_rows))
        rng.shuffle(cycle)
        result.extend(cycle)
    return result[:count]


def token_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits.float(), dim=-1)
    return masked_mean(-(logp.exp() * logp).sum(-1), mask)


def rollout(policy, tokenizer, reward_model, reward_tokenizer, rows, indices, cfg):
    from common.data import prompt_messages

    prompts = [prompt_messages(rows[i]) for i in indices]
    gen = batch_generate(
        policy, tokenizer, prompts,
        int(cfg["max_prompt_length"]), int(cfg["max_response_length"]),
        **cfg["generation"],
    )
    with torch.no_grad():
        old_logp, old_logits = response_token_logprobs(
            policy, gen["sequences"], gen["attention_mask"],
            gen["prompt_width"], gen["response_ids"],
        )
        entropy = token_entropy(old_logits, gen["response_mask"])
        with reference_mode(policy):
            ref_logp, _ = response_token_logprobs(
                policy, gen["sequences"], gen["attention_mask"],
                gen["prompt_width"], gen["response_ids"],
            )
        reward = score_reward_pairs(
            reward_model, reward_tokenizer, prompts, gen["responses"],
            max_length=int(cfg["reward_max_length"]),
        )
    if any(gen["truncated"]):
        reward = reward - torch.tensor(
            gen["truncated"], device=reward.device, dtype=reward.dtype
        ) * float(cfg["missing_eos_penalty"])
    gen.update(
        prompt_indices=indices, prompts=prompts, old_logp=old_logp.detach(),
        ref_logp=ref_logp.detach(), reward=reward.detach(),
        entropy=float(entropy.item()),
        kl=float(sampled_kl(old_logp, ref_logp, gen["response_mask"]).item()),
    )
    return gen


def evaluate_policy(policy, tokenizer, reward_model, reward_tokenizer, rows, cfg):
    from common.data import prompt_messages

    records = []
    for index, row in enumerate(rows):
        prompts = [prompt_messages(row)]
        gen = batch_generate(
            policy, tokenizer, prompts,
            int(cfg["max_prompt_length"]), int(cfg["eval_max_response_length"]),
            **cfg["generation"],
        )
        with torch.no_grad():
            logp, logits = response_token_logprobs(
                policy, gen["sequences"], gen["attention_mask"],
                gen["prompt_width"], gen["response_ids"],
            )
            entropy = token_entropy(logits, gen["response_mask"])
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy, gen["sequences"], gen["attention_mask"],
                    gen["prompt_width"], gen["response_ids"],
                )
            score = score_reward_pairs(
                reward_model, reward_tokenizer, prompts, gen["responses"],
                max_length=int(cfg["reward_max_length"]),
            )
        mask = gen["response_mask"]
        records.append({
            "index": index,
            "source_index": row.get("source_index", row.get("id", index)),
            "prompt": prompts[0],
            "response": gen["responses"][0],
            "reward": float(score[0].item()),
            "kl_per_token": float(sampled_kl(logp, ref_logp, mask).item()),
            "entropy": float(entropy.item()),
            "response_length": int(gen["response_lengths"][0]),
            "terminated_with_eos": bool(gen["terminated_with_eos"][0]),
        })
    if not records:
        raise ValueError("Held-out PPO prompt set is empty")
    def mean(key):
        return statistics.fmean(float(r[key]) for r in records)
    lengths = [r["response_length"] for r in records]
    summary = {
        "n": len(records), "reward_mean": mean("reward"),
        "kl_per_token_mean": mean("kl_per_token"),
        "entropy_mean": mean("entropy"),
        "response_length_mean": mean("response_length"),
        "response_length_std": statistics.pstdev(lengths),
        "eos_rate": statistics.fmean(float(r["terminated_with_eos"]) for r in records),
    }
    return summary, records


def stability_summary(logs: list[dict]) -> dict:
    """Defined variability of update diagnostics across the matched fork budget."""
    def std(key):
        return statistics.pstdev(float(x[key]) for x in logs)
    return {
        "policy_loss_std_across_updates": std("policy_loss"),
        "gradient_norm_std_across_updates": std("policy_gradient_norm"),
        "clip_fraction_mean_across_updates": statistics.fmean(float(x["clip_fraction"]) for x in logs),
    }
