"""On-policy GRPO continuation from the supplied midpoint adapter."""
from __future__ import annotations

import json
import time
import torch

from common.data import prompt_messages, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import set_seed
from common.models import reference_mode, trainable_parameters
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def categorical_entropy(logits: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Mean full-vocabulary policy entropy at unmasked response positions."""
    with torch.no_grad():
        log_probs = torch.log_softmax(logits.detach().float(), dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        return (entropy * token_mask).sum() / token_mask.sum().clamp_min(1)


def schedule_indices(n, updates, per_update, seed):
    if n < 1 or min(updates, per_update) < 1:
        raise ValueError("Need nonempty prompt pool, updates, and prompts per update")
    gen = torch.Generator().manual_seed(seed)
    order = []
    while len(order) < updates * per_update:
        order.extend(torch.randperm(n, generator=gen).tolist())
    return [order[i:i + per_update] for i in range(0, updates * per_update, per_update)]


def run(bundle, output, updates, loss_type, run_name):
    cfg = bundle["cfg"]
    if loss_type not in {"grpo", "dr_grpo"}:
        raise ValueError(loss_type)
    policy, tok, optimizer = bundle["policy"], bundle["tokenizer"], bundle["optimizer"]
    rows = bundle["prompt_rows"]
    k, p = int(cfg["num_generations"]), int(cfg["prompts_per_update"])
    epochs, cap = int(cfg["policy_epochs"]), int(cfg["max_completion_length"])
    if min(k, p, epochs, cap) < 1:
        raise ValueError("GRPO configuration counts must be positive")
    out = repo_path(output)
    out.mkdir(parents=True, exist_ok=True)
    results = repo_path(cfg["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    trajectory = results / f"{run_name}_trajectory.jsonl"
    trajectory.write_text("", encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generated_total = 0
    schedule = schedule_indices(len(rows), updates, p, int(cfg["seed"]))
    for step, indices in enumerate(schedule, 1):
        set_seed(int(cfg["seed"]) + step)
        prompts = [prompt_messages(rows[i]) for i in indices]
        repeated = [prompt for prompt in prompts for _ in range(k)]
        batch = batch_generate(policy, tok, repeated, int(cfg["max_prompt_length"]), cap,
                               **cfg["generation"])
        lengths = batch["response_lengths"]
        generated_total += sum(lengths)
        n = len(repeated)
        mask = batch["response_mask"]
        if cfg["mask_truncated_completions"]:
            mask = mask_truncated_sequences(mask, batch["truncated"])
        group_ids = torch.arange(p, device=mask.device).repeat_interleave(k)
        rewards = score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"],
                                     repeated, batch["responses"]).to(mask.device)
        advantages = group_relative_advantages(rewards, group_ids)
        active_tokens = max(float(mask.sum()), 1.0)
        old, ref = [], []
        was_training = policy.training
        policy.eval()
        with torch.no_grad():
            for i in range(n):
                args = (batch["sequences"][i:i + 1], batch["attention_mask"][i:i + 1],
                        batch["prompt_width"], batch["response_ids"][i:i + 1])
                old.append(response_token_logprobs(policy, *args)[0].detach())
                with reference_mode(policy):
                    ref.append(response_token_logprobs(policy, *args)[0].detach())
        if was_training:
            policy.train()
        stats = []
        norms = []
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            for i in range(n):
                args = (batch["sequences"][i:i + 1], batch["attention_mask"][i:i + 1],
                        batch["prompt_width"], batch["response_ids"][i:i + 1])
                new, logits = response_token_logprobs(policy, *args)
                entropy = categorical_entropy(logits, mask[i:i + 1])
                # Sequence-weighted surrogate and token-weighted KL over the full rollout.
                loss, detail = grpo_policy_loss(new, old[i], advantages[i:i + 1],
                    mask[i:i + 1], ref[i], float(cfg["clip_epsilon"]), 0.0,
                    loss_type, max_completion_length=cap)
                log_ref_over_policy = ref[i] - new
                kl = (torch.exp(log_ref_over_policy) - log_ref_over_policy - 1.0)
                total = loss / n + float(cfg["kl_beta"]) * (kl * mask[i:i + 1]).sum() / active_tokens
                total.backward()
                stats.append({key: float(value) for key, value in detail.items()} |
                             {"tokens": float(mask[i].sum()), "sample_entropy": float(entropy)})
                del new, logits, total
            norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
            norms.append(float(norm))
            optimizer.step()
        group_std = rewards.reshape(p, k).std(dim=1, unbiased=False)
        last = stats[-n:]
        median = sorted(lengths)[n // 2]
        conditional = {}
        for label, selected in (("short", [i for i in range(n) if lengths[i] <= median]),
                                ("long", [i for i in range(n) if lengths[i] > median])):
            selected = [i for i in selected if float(mask[i].sum()) > 0]
            coefficients = [abs(float(advantages[i])) /
                            (float(mask[i].sum()) if loss_type == "grpo" else cap) for i in selected]
            conditional[label] = {
                "count": len(selected),
                "mean_abs_policy_coefficient_per_token": sum(coefficients) / len(selected) if selected else None,
                "mean_abs_policy_weight_per_completion":
                    sum(c * float(mask[i].sum()) for c, i in zip(coefficients, selected)) / len(selected)
                    if selected else None,
            }
        weighted = lambda key: sum(s[key] * s["tokens"] for s in last) / active_tokens
        entry = {
            "update": step, "prompt_indices": indices,
            "prompt_ids": [rows[i].get("source_index", i) for i in indices],
            "mean_reward": float(rewards.mean()),
            "sampled_kl": weighted("sampled_kl"),
            "kl_penalty": weighted("kl_penalty"),
            "group_reward_std": float(group_std.mean()),
            "uninformative_group_fraction": float((group_std <= 1e-6).float().mean()),
            "policy_loss": sum(s["policy_term"] for s in last) / n,
            "gradient_norm": sum(norms) / len(norms),
            "sample_entropy": weighted("sample_entropy"),
            "sampled_token_surprisal": weighted("sampled_token_surprisal"),
            "clip_fraction": weighted("clip_fraction"),
            "response_length": sum(lengths) / n,
            "generated_tokens": sum(lengths),
            "generated_tokens_cumulative": generated_total,
            "active_completion_fraction": float((mask.sum(-1) > 0).float().mean()),
            "length_conditioned": conditional,
        }
        with trajectory.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(json.dumps({key: entry[key] for key in ("update", "mean_reward", "sampled_kl", "group_reward_std", "uninformative_group_fraction")}))
    policy.save_pretrained(out)
    summary = {
        "run_name": run_name, "adapter": str(out),
        "source_midpoint": cfg["paths"]["grpo_midpoint_policy"],
        "loss_type": loss_type, "updates": updates, "group_size": k,
        "prompts_per_update": p, "policy_epochs": epochs,
        "max_completion_length": cap, "seed": int(cfg["seed"]),
        "generated_tokens": generated_total,
        "maximum_generated_token_budget": updates * p * k * cap,
        "wall_clock_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "trajectory": str(trajectory),
    }
    (results / f"{run_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
