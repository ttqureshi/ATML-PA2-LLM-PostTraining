from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task1_dpo.evaluate import evaluate_adapter
from task1_dpo.train import run_training


def run_beta_study(config_path: str):
    cfg = load_yaml(config_path)
    betas = [float(b) for b in cfg["betas"]]
    if len(betas) != len(set(betas)):
        raise ValueError("Beta conditions must be unique")
    n = int(cfg["short_ablation_examples"])
    if n <= 0:
        raise ValueError("short_ablation_examples must be positive")
    summaries = []
    for beta in betas:
        label = f"beta_{beta:.2f}"
        output = f"outputs/task1_dpo/{label}"
        training = run_training(
            config_path, label,
            dataset_path=cfg["paths"]["dpo_standard_train"],
            output_path=output, beta=beta, max_examples=n,
        )
        evaluation = evaluate_adapter(config_path, output, label, beta=beta)
        summaries.append({"beta": beta, "budget": "short", "training": training, "evaluation": evaluation})
    path = repo_path(cfg["results_dir"]) / "beta_study.json"
    save_json(path, {
        "dataset": cfg["paths"]["dpo_standard_train"],
        "examples_per_condition": n,
        "heldout_dataset": cfg["paths"]["dpo_standard_eval"],
        "seed": int(cfg["seed"]),
        "conditions": summaries,
        "note": "Short beta forks use a different training budget from standard one-epoch DPO.",
    })
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    run_beta_study(args.config)


if __name__ == "__main__":
    main()
