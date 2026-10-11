from __future__ import annotations

import argparse
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import save_json, set_seed
from common.models import load_policy, load_tokenizer
from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_adapter
from task2_ppo.workflow import stability_summary


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def affected_token_fraction(ratio, advantage, mask, eps: float) -> torch.Tensor:
    """Fraction where clipping changes the minimum surrogate, not merely ratio range."""
    affected = (((ratio > 1 + eps) & (advantage > 0)) |
                ((ratio < 1 - eps) & (advantage < 0))).float()
    return (affected * mask).sum() / mask.sum().clamp_min(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    cached = analyze_cache(cfg, rows)
    save_json(repo_path(cfg["results_dir"]) / "cached_clipping.json", cached)
    comparisons = []
    for eps in cfg["clip_values"]:
        name = f"clip_{float(eps):.2f}"
        summary, logs = run_ppo(
            args.config, output=str(repo_path(cfg["output"]).parent / name),
            updates=int(cfg["fork_updates"]), clip_epsilon=float(eps), run_name=name,
        )
        heldout, _ = evaluate_adapter(args.config, summary["output_adapter"], name)
        comparisons.append({"epsilon": float(eps), "heldout": heldout,
                            "training": summary, "stability": stability_summary(logs)})
    save_json(repo_path(cfg["results_dir"]) / "clipping_comparison.json", comparisons)


def analyze_cache(cfg: dict, rows: list[dict]) -> list[dict]:
    """Evaluate fixed cached responses with the supplied midpoint and identical masks."""
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    by_source = {str(r["source_index"]): r for r in prompts if "source_index" in r}
    totals = {float(e): {"surrogate_sum": 0.0, "outside": 0.0,
                         "affected": 0.0, "tokens": 0} for e in cfg["clip_values"]}
    trajectories = []
    for row in rows:
        source = str(row["source_index"])
        if source in by_source:
            prompt_row = by_source[source]
            if "prompt_id" in row and row["prompt_id"] != prompt_row.get("prompt_id"):
                raise ValueError(f"Cached prompt ID mismatch for source {source}")
            prompt = prompt_messages(prompt_row)
        elif source.isdigit() and int(source) < len(prompts):
            prompt = prompt_messages(prompts[int(source)])
        else:
            raise ValueError(f"Cached source_index {source!r} is absent from the training prompt pool")
        pids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        rids = row.get("response_ids")
        if rids is None:
            rids = tokenizer(str(row["response"]), add_special_tokens=False)["input_ids"]
        if isinstance(rids, torch.Tensor):
            rids = rids.tolist()
        old = torch.as_tensor(row["old_logprobs"], dtype=torch.float32).flatten()
        ref = torch.as_tensor(row["ref_logprobs"], dtype=torch.float32).flatten()
        if len(rids) + 1 == len(old) and row.get("terminated_with_eos"):
            rids = rids + [tokenizer.eos_token_id]
        if len(rids) != len(old) or len(rids) != len(ref):
            raise ValueError(
                f"Cached response {source}: retokenization produced {len(rids)} tokens "
                f"for {len(old)} cached log-probabilities. Exact cached token IDs are unavailable; "
                "cannot evaluate the surrogate without changing the fixed token alignment."
            )
        ids = torch.tensor([pids + rids], device=next(policy.parameters()).device)
        attn = torch.ones_like(ids)
        response_ids = torch.tensor([rids], device=ids.device)
        mask = torch.as_tensor(row.get("response_mask", [1] * len(rids)),
                               device=ids.device, dtype=torch.float32).reshape(1, -1)
        if mask.shape != response_ids.shape:
            raise ValueError("Cached response mask and token IDs have different lengths")
        with torch.no_grad():
            new, _ = response_token_logprobs(policy, ids, attn, len(pids), response_ids)
            values = torch.as_tensor(row["values"], device=ids.device, dtype=torch.float32).reshape(1, -1)
            if values.shape != mask.shape:
                raise ValueError(f"Cached value estimates do not align with response {source}")
            task_reward = torch.tensor(
                [float(row["effective_terminal_reward"])], device=ids.device
            )
            old, ref = old.to(ids.device)[None, :], ref.to(ids.device)[None, :]
            shaped = shaped_rewards(task_reward, old, ref, mask, cfg["kl_beta"])
            advantage, _ = compute_gae(shaped, values, mask,
                                       gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]))
            trajectories.append(tuple(x.detach().cpu() for x in (new, old, advantage, mask)))
    all_adv = torch.cat([x[2][x[3].bool()] for x in trajectories])
    adv_mean = all_adv.mean()
    adv_std = all_adv.std(unbiased=False).clamp_min(1e-6)
    for new, old, advantage, mask in trajectories:
        advantage = (advantage - adv_mean) / adv_std
        for eps, total in totals.items():
            loss, ratio, frac = ppo_policy_loss(new, old, advantage, mask, eps=eps)
            n = int(mask.sum().item())
            total["surrogate_sum"] += -float(loss.item()) * n
            total["outside"] += float(frac.item()) * n
            total["affected"] += float(affected_token_fraction(ratio, advantage, mask, eps).item()) * n
            total["tokens"] += n
    return [{"epsilon": eps, "clipped_surrogate": t["surrogate_sum"] / t["tokens"],
             "clip_fraction": t["outside"] / t["tokens"],
             "affected_token_fraction": t["affected"] / t["tokens"], "valid_tokens": t["tokens"]}
            for eps, t in totals.items()]


if __name__ == "__main__":
    main()
