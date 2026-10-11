from __future__ import annotations

import argparse
import json
from common.data import load_yaml, read_jsonl, repo_path
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate


def compare(config_path: str):
    cfg = load_yaml(config_path)
    results = {}
    for loss_type, name in (("grpo", "normalization_canonical"),
                            ("dr_grpo", "normalization_dr_grpo")):
        output = f"outputs/task3_grpo/{name}"
        train = run_grpo(config_path, output=output, updates=int(cfg["fork_updates"]),
                         loss_type=loss_type, run_name=name)
        heldout = evaluate(config_path, output, name=name)
        trajectory = read_jsonl(train["trajectory"])
        results[loss_type] = {
            "train": train, "heldout": heldout,
            "mean_short_policy_coefficient_per_token": _mean_stat(trajectory, "short"),
            "mean_long_policy_coefficient_per_token": _mean_stat(trajectory, "long"),
        }
    # Both forks use the same seed, prompt schedule, K, decoding settings, update
    # count, and maximum generation-token budget. Actual output lengths are logged.
    a, b = results["grpo"]["train"], results["dr_grpo"]["train"]
    for key in ("updates", "group_size", "prompts_per_update", "policy_epochs",
                "max_completion_length", "seed", "maximum_generated_token_budget"):
        if a[key] != b[key]:
            raise AssertionError(f"Normalization forks differ on {key}")
    canonical = read_jsonl(repo_path(cfg["results_dir"]) / "normalization_canonical_heldout_examples.jsonl")
    dr = read_jsonl(repo_path(cfg["results_dir"]) / "normalization_dr_grpo_heldout_examples.jsonl")
    if len(canonical) != len(dr) or any(x["source_index"] != y["source_index"] for x, y in zip(canonical, dr)):
        raise AssertionError("Normalization forks have unmatched held-out prompts")
    paired = [{"source_index": x["source_index"], "prompt": x["prompt"],
               "canonical_response": x["response"], "dr_response": y["response"],
               "canonical_reward": x["reward"], "dr_reward": y["reward"],
               "reward_delta_dr_minus_canonical": y["reward"] - x["reward"],
               "canonical_length": x["response_length"], "dr_length": y["response_length"]}
              for x, y in zip(canonical, dr)]
    paired_path = repo_path(cfg["results_dir"]) / "normalization_paired_examples.jsonl"
    with paired_path.open("w", encoding="utf-8") as f:
        for row in paired:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    results["paired_examples"] = str(paired_path)
    path = repo_path(cfg["results_dir"]) / "normalization_comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _mean_stat(trajectory, length_bin):
    values = [row["length_conditioned"][length_bin]["mean_abs_policy_coefficient_per_token"]
              for row in trajectory]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    print(json.dumps(compare(args.config), indent=2))


if __name__ == "__main__":
    main()
