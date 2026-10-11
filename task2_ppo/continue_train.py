from __future__ import annotations

import argparse
import time

import torch

from torch.optim import AdamW

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    trainable_parameters,
    value_parameter_groups,
    token_values,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss
from task2_ppo.workflow import prompt_schedule, rollout


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    results = repo_path(cfg["results_dir"]) / run_name
    results.mkdir(parents=True, exist_ok=True)
    log_path = results / "updates.jsonl"
    log_path.write_text("", encoding="utf-8")
    policy, critic = bundle["policy"], bundle["value_model"]
    p_opt, v_opt = bundle["policy_optimizer"], bundle["value_optimizer"]
    count = int(cfg["updates"])
    per_update = int(cfg["prompts_per_update"])
    indices = prompt_schedule(len(bundle["prompt_rows"]), int(cfg["seed"]), count * per_update)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    logs = []
    for update in range(count):
        selected = indices[update * per_update:(update + 1) * per_update]
        policy.eval()
        batch = rollout(
            policy, bundle["tokenizer"], bundle["reward_model"],
            bundle["reward_tokenizer"], bundle["prompt_rows"], selected, cfg,
        )
        seq, attn, width = batch["sequences"], batch["attention_mask"], batch["prompt_width"]
        rmask = batch["response_mask"]
        critic.eval()
        with torch.no_grad():
            old_values = token_values(critic, seq, attn)[:, width - 1:-1].float()
            rewards = shaped_rewards(
                batch["reward"], batch["old_logp"], batch["ref_logp"],
                rmask, cfg["kl_beta"],
            )
            advantages, returns = compute_gae(
                rewards, old_values, rmask,
                gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]),
            )
            advantages = normalize_advantages(advantages, rmask)
        policy.train()
        critic.train()
        epoch_logs = []
        for _ in range(int(cfg["ppo_epochs"])):
            p_opt.zero_grad(set_to_none=True)
            new_logp, _ = response_token_logprobs(
                policy, seq, attn, width, batch["response_ids"]
            )
            policy_loss, _, clipped = ppo_policy_loss(
                new_logp, batch["old_logp"], advantages, rmask,
                eps=float(cfg["clip_epsilon"]),
            )
            policy_loss.backward()
            policy_grad = torch.nn.utils.clip_grad_norm_(
                trainable_parameters(policy), float(cfg["max_grad_norm"])
            )
            p_opt.step()
            v_opt.zero_grad(set_to_none=True)
            predicted = token_values(critic, seq, attn)[:, width - 1:-1].float()
            value_loss = value_mse_loss(predicted, returns, rmask) * float(cfg["value_coef"])
            value_loss.backward()
            value_grad = torch.nn.utils.clip_grad_norm_(
                trainable_parameters(critic), float(cfg["max_grad_norm"])
            )
            v_opt.step()
            epoch_logs.append((float(policy_loss.item()), float(value_loss.item()),
                               float(clipped.item()), float(policy_grad.item()),
                               float(value_grad.item())))
        record = {
            "update": update + 1, "prompt_indices": selected,
            "generated_tokens": int(rmask.sum().item()),
            "reward": float(batch["reward"].mean().item()),
            "kl_per_token": batch["kl"],
            "policy_loss": sum(x[0] for x in epoch_logs) / len(epoch_logs),
            "value_loss": sum(x[1] for x in epoch_logs) / len(epoch_logs),
            "clip_fraction": sum(x[2] for x in epoch_logs) / len(epoch_logs),
            "policy_gradient_norm": sum(x[3] for x in epoch_logs) / len(epoch_logs),
            "value_gradient_norm": sum(x[4] for x in epoch_logs) / len(epoch_logs),
            "entropy": batch["entropy"],
            "response_length": sum(batch["response_lengths"]) / len(batch["response_lengths"]),
            "elapsed_seconds": time.perf_counter() - start,
        }
        append_jsonl(log_path, record)
        logs.append(record)
        print(f"PPO {run_name}: update {update + 1}/{count}, reward={record['reward']:.4f}, KL={record['kl_per_token']:.4f}", flush=True)
    policy.save_pretrained(out)
    critic.save_pretrained(out / "value")
    summary = {
        "run_name": run_name, "updates": count, "seed": int(cfg["seed"]),
        "clip_epsilon": float(cfg["clip_epsilon"]), "kl_beta": float(cfg["kl_beta"]),
        "total_generated_tokens": sum(r["generated_tokens"] for r in logs),
        "wall_clock_seconds": time.perf_counter() - start,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "output_adapter": str(out),
    }
    save_json(results / "summary.json", summary)
    return summary, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
