from __future__ import annotations

import argparse
import gc
import statistics

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--force", action="store_true", help="replace existing saved responses and results")
    args = ap.parse_args()
    evaluate_dataset(args.config, args.dataset, force=args.force)


def _policy_summary(records: list[dict]) -> dict:
    lengths = [r["response_length"] for r in records]
    return {
        "n": len(records),
        "exact_accuracy": statistics.fmean(r["exact_correct"] for r in records),
        "format_compliance": statistics.fmean(r["format_compliant"] for r in records),
        "response_length_mean": statistics.fmean(lengths),
        "response_length_std": statistics.pstdev(lengths),
    }


def summarize_pairs(records: list[dict]) -> dict:
    """Judge A means aligned policy preferred; verifier A means only it is correct."""
    strict = [r for r in records if r["verifier_preference"] != "TIE"]
    return {
        "n": len(records),
        "ai_pairwise_win_rate_vs_sft": statistics.fmean(
            1.0 if r["judge_preference"] == "A" else 0.5 if r["judge_preference"] == "TIE" else 0.0
            for r in records
        ),
        "judge_ties": sum(r["judge_preference"] == "TIE" for r in records),
        "verifier_ties": sum(r["verifier_preference"] == "TIE" for r in records),
        "verifier_judge_agreement_all": statistics.fmean(
            r["judge_preference"] == r["verifier_preference"] for r in records
        ),
        "verifier_judge_agreement_strict": (
            statistics.fmean(r["judge_preference"] == r["verifier_preference"] for r in strict)
            if strict else None
        ),
        "n_strict_verifier_pairs": len(strict),
    }


def evaluate_dataset(config_path: str, dataset: str, force: bool = False):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    if not rows:
        raise ValueError(f"Empty {dataset} evaluation set")
    out = repo_path(cfg["results_dir"]) / "task5_feedback" / dataset
    output_files = [out / f"{name}_responses.jsonl" for name in policy_specs(cfg)]
    output_files.extend(out / f"{name}_vs_sft_pairs.jsonl" for name in ("rlvr", "rlaif"))
    output_files.append(out / "summary.json")
    existing = [path for path in output_files if path.exists()]
    if existing and not force:
        raise FileExistsError(
            f"Task 5 {dataset} results already exist ({existing[0]}). "
            "Use --force to replace saved responses and results."
        )
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(cfg["base_model"])
    records_by_policy = {}
    summaries = {}
    for name in policy_specs(cfg):
        set_seed(int(cfg["seed"]))
        policy = load_frozen_policy(cfg, name)
        records = []
        for index, row in enumerate(rows):
            prompt = prompt_messages(row)
            generated = batch_generate(
                policy, tokenizer, [prompt],
                max_prompt_length=int(cfg.get("max_prompt_length", 256)),
                max_new_tokens=int(cfg["math_max_new_tokens"]),
                **cfg["generation"],
            )
            response = generated["responses"][0]
            prediction = extract_designated_final(response)
            records.append({
                "index": index,
                "source_index": row["source_index"],
                "question": row["question"],
                "gold_final": str(row["gold_final"]),
                "policy": name,
                "response": response,
                "predicted_final": prediction,
                "exact_correct": exact_reward(response, str(row["gold_final"])),
                "format_compliant": float(prediction is not None),
                "response_length": int(generated["response_lengths"][0]),
                "terminated_with_eos": bool(generated["terminated_with_eos"][0]),
            })
        records_by_policy[name] = records
        summaries[name] = _policy_summary(records)
        write_jsonl(out / f"{name}_responses.jsonl", records)
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    judge = PairwiseAIJudge(cfg, out / "judge_cache.json")
    pairs_by_policy = {}
    for name in ("rlvr", "rlaif"):
        pairs = []
        for candidate, sft in zip(records_by_policy[name], records_by_policy["sft"]):
            if candidate["source_index"] != sft["source_index"]:
                raise ValueError("Policy response sets are not paired on the same fixed IDs")
            decision = judge.compare(candidate["question"], candidate["response"], sft["response"])
            v_candidate, v_sft = candidate["exact_correct"], sft["exact_correct"]
            verifier = "A" if v_candidate > v_sft else "B" if v_sft > v_candidate else "TIE"
            pairs.append({
                "source_index": candidate["source_index"], "question": candidate["question"],
                "policy": name, "candidate_response": candidate["response"],
                "sft_response": sft["response"], "candidate_correct": v_candidate,
                "sft_correct": v_sft, "judge_preference": decision,
                "verifier_preference": verifier,
                "verifier_judge_agree": decision == verifier,
            })
        pairs_by_policy[name] = summarize_pairs(pairs)
        write_jsonl(out / f"{name}_vs_sft_pairs.jsonl", pairs)
    summary = {
        "dataset": dataset, "n": len(rows), "seed": int(cfg["seed"]),
        "evaluation_protocol": {
            "base_model": cfg["base_model"],
            "ai_judge_model": cfg["ai_judge_model"],
            "policy_adapters": policy_specs(cfg),
            "dataset_path": dataset_path(cfg, dataset),
            "source_indices": [row["source_index"] for row in rows],
            "generation": cfg["generation"],
            "math_max_new_tokens": int(cfg["math_max_new_tokens"]),
            "max_prompt_length": int(cfg.get("max_prompt_length", 256)),
        },
        "policies": summaries, "pairwise": pairs_by_policy,
    }
    save_json(out / "summary.json", summary)
    print(summary)
    return summary


if __name__ == "__main__":
    main()
