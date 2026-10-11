from __future__ import annotations

import argparse
import statistics
from collections import Counter

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import load_tokenizer
from task1_dpo.evaluate import _stratum, evaluate_adapter, evaluate_word_limits
from task1_dpo.train import run_training


def _training_length_diagnostics(rows, tokenizer):
    differences = []
    groups = Counter()
    for row in rows:
        chosen, rejected = preference_responses(row)
        chosen_length = len(tokenizer(chosen, add_special_tokens=False)["input_ids"])
        rejected_length = len(tokenizer(rejected, add_special_tokens=False)["input_ids"])
        differences.append(chosen_length - rejected_length)
        label = _stratum(row)
        if label is not None:
            groups[label] += 1
    return {
        "pair_count": len(rows),
        "chosen_minus_rejected_tokens_mean": statistics.mean(differences),
        "chosen_minus_rejected_tokens_std": statistics.pstdev(differences),
        "preferred_longer_fraction": statistics.mean(d > 0 for d in differences),
        "matched_exact_length_fraction": statistics.mean(d == 0 for d in differences),
        "rejected_longer_fraction": statistics.mean(d < 0 for d in differences),
        "released_stratum_counts": dict(groups),
    }


def run_length_study(config_path: str, standard_adapter: str | None = None):
    cfg = load_yaml(config_path)
    standard_adapter = standard_adapter or cfg["standard_output"]
    standard_path = repo_path(standard_adapter)
    if not (standard_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"Standard one-epoch adapter is missing at {standard_path}. Run task1_dpo.train first."
        )
    balanced_train = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified_eval = read_jsonl(cfg["paths"]["dpo_length_eval"])
    if not balanced_train or not stratified_eval:
        raise ValueError("Length-control datasets must be nonempty")
    labels = {_stratum(row) for row in stratified_eval}
    if None in labels or len(labels) != 3:
        raise ValueError(
            "Expected three explicit course length strata in the held-out data; "
            f"found {sorted(str(label) for label in labels)}"
        )

    tokenizer = load_tokenizer(cfg["base_model"])
    training_data = _training_length_diagnostics(
        read_jsonl(cfg["paths"]["dpo_standard_train"]), tokenizer
    )
    balanced_data = _training_length_diagnostics(balanced_train, tokenizer)
    del tokenizer

    training = run_training(
        config_path, "length_balanced",
        dataset_path=cfg["paths"]["dpo_length_train"],
        output_path=cfg["length_output"],
    )
    standard_eval = evaluate_adapter(
        config_path, standard_adapter, "standard_stratified",
        dataset_path=cfg["paths"]["dpo_length_eval"], generate=False,
    )
    balanced_eval = evaluate_adapter(
        config_path, cfg["length_output"], "length_balanced_stratified",
        dataset_path=cfg["paths"]["dpo_length_eval"], generate=False,
    )
    standard_words = evaluate_word_limits(config_path, standard_adapter, "standard")
    balanced_words = evaluate_word_limits(config_path, cfg["length_output"], "length_balanced")
    result = {
        "seed": int(cfg["seed"]),
        "stratified_dataset": cfg["paths"]["dpo_length_eval"],
        "word_limit_dataset": cfg["paths"]["word_limit_prompts"],
        "standard_training_data": training_data,
        "balanced_training_data": balanced_data,
        "balanced_training": training,
        "standard_stratified": standard_eval,
        "balanced_stratified": balanced_eval,
        "standard_word_limits": standard_words,
        "balanced_word_limits": balanced_words,
    }
    save_json(repo_path(cfg["results_dir"]) / "length_study.json", result)
    print(result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--standard-adapter")
    args = ap.parse_args()
    run_length_study(args.config, args.standard_adapter)


if __name__ == "__main__":
    main()
