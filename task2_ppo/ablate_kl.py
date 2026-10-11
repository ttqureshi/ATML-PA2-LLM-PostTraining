from __future__ import annotations

import argparse
from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    comparisons = []
    for beta in cfg["kl_values"]:
        name = f"kl_{float(beta):.2f}"
        train, logs = run_ppo(
            args.config, output=str(repo_path(cfg["output"]).parent / name),
            updates=int(cfg["fork_updates"]), kl_beta=float(beta), run_name=name,
        )
        heldout, _ = evaluate_adapter(args.config, train["output_adapter"], name)
        comparisons.append({"beta_kl": float(beta), "training": train, "heldout": heldout,
                            "trajectory": logs})
    save_json(repo_path(cfg["results_dir"]) / "kl_comparison.json", comparisons)


if __name__ == "__main__":
    main()
