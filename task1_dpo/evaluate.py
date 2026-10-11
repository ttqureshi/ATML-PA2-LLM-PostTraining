from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from common.data import (
    load_yaml, prompt_messages, prompt_messages_from_preference,
    read_jsonl, repo_path, write_jsonl,
)
from common.generation import (
    batch_generate, response_sequence_logprobs, response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json, set_seed
from common.metrics import sampled_kl, word_count, word_limit_compliance
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.train import make_collate


def _device_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _stats(values):
    if not values:
        return {"mean": None, "std": None, "count": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "count": len(values),
    }


def _stratum(row):
    for key in ("stratum", "length_stratum", "length_group", "length_category", "stratum_label", "group"):
        if key in row and row[key] is not None:
            return str(row[key])
    return None


def _evaluate_pairs(policy, tokenizer, rows, cfg, beta):
    loader = DataLoader(
        rows, batch_size=int(cfg["batch_size"]), shuffle=False,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    device = next(policy.parameters()).device
    records = []
    for offset, (chosen, rejected) in enumerate(loader):
        chosen = _device_batch(chosen, device)
        rejected = _device_batch(rejected, device)
        with torch.inference_mode():
            with reference_mode(policy):
                ref_c, _, _ = response_sequence_logprobs(policy, chosen)
                ref_r, _, _ = response_sequence_logprobs(policy, rejected)
            pol_c, _, _ = response_sequence_logprobs(policy, chosen)
            pol_r, _, _ = response_sequence_logprobs(policy, rejected)
            # Per-pair losses retain exact weighting for global and stratum means.
            logits = beta * ((pol_c - ref_c) - (pol_r - ref_r))
            losses = -torch.nn.functional.logsigmoid(logits)
            margin = (pol_c - ref_c) - (pol_r - ref_r)
        start = offset * int(cfg["batch_size"])
        for j in range(len(losses)):
            row = rows[start + j]
            records.append({
                "index": start + j,
                "prompt_id": row.get("prompt_id", row.get("id", start + j)),
                "stratum": _stratum(row),
                "dpo_loss": float(losses[j].item()),
                "margin": float(margin[j].item()),
                "preference_correct": bool(margin[j].item() > 0),
                "policy_chosen_logp": float(pol_c[j].item()),
                "policy_rejected_logp": float(pol_r[j].item()),
                "ref_chosen_logp": float(ref_c[j].item()),
                "ref_rejected_logp": float(ref_r[j].item()),
            })
    return records


def _generate(policy, tokenizer, rows, cfg, include_reference=True):
    generation_cfg = cfg["generation"]
    batch_size = int(cfg["batch_size"])
    records = []
    kl_numerator = 0.0
    kl_denominator = 0.0
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset:offset + batch_size]
        prompts = [prompt_messages_from_preference(r) for r in batch_rows]
        seed = int(cfg["seed"]) + offset
        set_seed(seed)
        generated = batch_generate(
            policy, tokenizer, prompts, int(cfg["max_sequence_length"]),
            int(cfg["max_generation_tokens"]), **generation_cfg,
        )
        with torch.inference_mode():
            pol_logp, _ = response_token_logprobs(
                policy, generated["sequences"], generated["attention_mask"],
                generated["prompt_width"], generated["response_ids"],
            )
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy, generated["sequences"], generated["attention_mask"],
                    generated["prompt_width"], generated["response_ids"],
                )
            mask = generated["response_mask"]
            kl_denominator += float(mask.sum().item())
            # Keep the course helper as the metric definition.
            batch_kl = float(sampled_kl(pol_logp, ref_logp, mask).item())
            kl_numerator += batch_kl * float(mask.sum().item())
            per_example_kl = (((pol_logp - ref_logp) * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)).tolist()
            if include_reference:
                set_seed(seed)
                with reference_mode(policy):
                    reference_generated = batch_generate(
                        policy, tokenizer, prompts, int(cfg["max_sequence_length"]),
                        int(cfg["max_generation_tokens"]), **generation_cfg,
                    )
        for j, row in enumerate(batch_rows):
            record = {
                "index": offset + j,
                "prompt_id": row.get("prompt_id", row.get("id", offset + j)),
                "prompt": prompts[j],
                "response": generated["responses"][j],
                "response_length_tokens": generated["response_lengths"][j],
                "response_word_count": word_count(generated["responses"][j]),
                "terminated_with_eos": generated["terminated_with_eos"][j],
                "truncated": generated["truncated"][j],
                "sampled_kl_per_response_token": float(per_example_kl[j]),
            }
            if include_reference:
                record["reference_response"] = reference_generated["responses"][j]
                record["reference_response_length_tokens"] = reference_generated["response_lengths"][j]
            records.append(record)
    return records, kl_numerator / max(kl_denominator, 1.0)


def _add_rewards(records, cfg):
    reward_model, reward_tokenizer = load_reward_model(cfg)
    batch_size = int(cfg["batch_size"])
    for offset in range(0, len(records), batch_size):
        chunk = records[offset:offset + batch_size]
        prompts = [record["prompt"] for record in chunk]
        responses = [record["response"] for record in chunk]
        scores = score_reward_pairs(reward_model, reward_tokenizer, prompts, responses)
        ref_scores = None
        if "reference_response" in chunk[0]:
            ref_scores = score_reward_pairs(
                reward_model, reward_tokenizer, prompts,
                [record["reference_response"] for record in chunk],
            )
        for j, record in enumerate(chunk):
            record["reward_score"] = float(scores[j].item())
            if ref_scores is not None:
                record["reference_reward_score"] = float(ref_scores[j].item())
                record["reward_delta_vs_reference"] = (
                    record["reward_score"] - record["reference_reward_score"]
                )
    del reward_model
    return records


def evaluate_adapter(
    config_path: str, adapter: str, name: str,
    dataset_path: str | None = None, generate: bool = True,
    reward: bool = True, include_reference: bool = True,
    beta: float | None = None,
):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    rows = read_jsonl(dataset_path or cfg["paths"]["dpo_standard_eval"])
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    policy.eval()
    pairs = _evaluate_pairs(policy, tokenizer, rows, cfg, float(cfg["beta"] if beta is None else beta))
    result_dir = repo_path(cfg["results_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(result_dir / f"{name}_pairs.jsonl", pairs)
    summary = {
        "name": name, "adapter": adapter,
        "dataset": dataset_path or cfg["paths"]["dpo_standard_eval"],
        "pair_count": len(pairs),
        "heldout_dpo_loss": statistics.mean(p["dpo_loss"] for p in pairs),
        "heldout_preference_accuracy": statistics.mean(p["preference_correct"] for p in pairs),
        "beta_for_loss": float(cfg["beta"] if beta is None else beta),
    }
    by_stratum = defaultdict(list)
    for pair in pairs:
        if pair["stratum"] is not None:
            by_stratum[pair["stratum"]].append(pair)
    if by_stratum:
        summary["strata"] = {
            key: {
                "count": len(group),
                "heldout_preference_accuracy": statistics.mean(p["preference_correct"] for p in group),
                "heldout_dpo_loss": statistics.mean(p["dpo_loss"] for p in group),
            }
            for key, group in sorted(by_stratum.items())
        }
    if generate:
        generations, kl = _generate(policy, tokenizer, rows, cfg, include_reference)
        summary["sampled_kl_per_response_token"] = kl
        summary["response_length_tokens"] = _stats([g["response_length_tokens"] for g in generations])
        if include_reference:
            summary["reference_response_length_tokens"] = _stats(
                [g["reference_response_length_tokens"] for g in generations]
            )
        if reward:
            _add_rewards(generations, cfg)
            summary["mean_reward_score"] = statistics.mean(g["reward_score"] for g in generations)
            if include_reference:
                summary["mean_reference_reward_score"] = statistics.mean(
                    g["reference_reward_score"] for g in generations
                )
        write_jsonl(result_dir / f"{name}_generations.jsonl", generations)
        if reward and include_reference:
            candidates = sorted(
                generations, key=lambda r: r["reward_delta_vs_reference"], reverse=True
            )[:20]
            write_jsonl(result_dir / f"{name}_qualitative_candidates.jsonl", candidates)
    save_json(result_dir / f"{name}_summary.json", summary)
    print(summary)
    return summary


def evaluate_word_limits(config_path: str, adapter: str, name: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    if not rows:
        raise ValueError("Word-limit prompt set is empty")
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    policy.eval()
    records = []
    batch_size = int(cfg["batch_size"])
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset:offset + batch_size]
        prompts = [prompt_messages(r) for r in batch_rows]
        set_seed(int(cfg["seed"]) + offset)
        generated = batch_generate(
            policy, tokenizer, prompts, int(cfg["max_sequence_length"]),
            int(cfg["max_generation_tokens"]), **cfg["generation"],
        )
        for j, row in enumerate(batch_rows):
            prompt_text = " ".join(str(m.get("content", "")) for m in prompts[j] if m.get("role") == "user")
            compliance = word_limit_compliance(prompt_text, generated["responses"][j])
            if compliance is None:
                raise ValueError(f"Could not parse word limit in prompt {row.get('prompt_id', offset + j)}")
            records.append({
                "prompt_id": row.get("prompt_id", offset + j),
                "prompt": prompt_text,
                "response": generated["responses"][j],
                "response_length_tokens": generated["response_lengths"][j],
                "word_count": word_count(generated["responses"][j]),
                "compliant": bool(compliance),
            })
    result_dir = repo_path(cfg["results_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(result_dir / f"{name}_word_limits.jsonl", records)
    summary = {
        "name": name, "count": len(records),
        "word_limit_compliance": statistics.mean(r["compliant"] for r in records),
        "response_length_tokens": _stats([r["response_length_tokens"] for r in records]),
    }
    save_json(result_dir / f"{name}_word_limits_summary.json", summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--beta", type=float)
    args = ap.parse_args()
    evaluate_adapter(args.config, args.adapter, args.name, args.dataset, beta=args.beta)


if __name__ == "__main__":
    main()
