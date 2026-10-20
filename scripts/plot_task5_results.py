from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from common.data import load_yaml, repo_path
from common.logging_utils import load_json


def export_task5(config_path: str = "configs/feedback.yaml") -> None:
    cfg = load_yaml(config_path)
    root = repo_path(cfg["results_dir"]) / "task5_feedback"
    comparison = load_json(root / "comparison.json")
    expected_policies = {"sft", "rlvr", "rlaif"}
    if set(comparison["policies"]) != expected_policies:
        raise ValueError(f"Expected policies {expected_policies}; got {set(comparison['policies'])}")
    for dataset, expected_n in (("gsm", 300), ("transfer", 100)):
        summary = load_json(root / dataset / "summary.json")
        if summary["n"] != expected_n:
            raise ValueError(f"Expected {expected_n} fixed {dataset} rows; got {summary['n']}")
        for policy in expected_policies:
            if summary["policies"][policy]["n"] != expected_n:
                raise ValueError(f"{dataset}/{policy} has the wrong response count")
            if len((root / dataset / f"{policy}_responses.jsonl").read_text(encoding="utf-8").splitlines()) != expected_n:
                raise ValueError(f"{dataset}/{policy} response file has the wrong row count")
        for policy in ("rlvr", "rlaif"):
            if summary["pairwise"][policy]["n"] != expected_n:
                raise ValueError(f"{dataset}/{policy} pairwise comparison count is incomplete")
            if len((root / dataset / f"{policy}_vs_sft_pairs.jsonl").read_text(encoding="utf-8").splitlines()) != expected_n:
                raise ValueError(f"{dataset}/{policy} pairwise file has the wrong row count")
    diagnostic = comparison["diagnostics"]
    if diagnostic["n_problems"] != 20 or diagnostic["n_controlled_pairs"] != 80:
        raise ValueError("Task 5 controlled diagnostic coverage is incomplete")

    metric_rows = []
    for policy, conditions in comparison["policies"].items():
        for dataset in ("gsm", "transfer"):
            summary = conditions[dataset]
            metric_rows.append({
                "policy": policy,
                "dataset": dataset,
                "n": summary["n"],
                "exact_accuracy": summary["exact_accuracy"],
                "format_compliance": summary["format_compliance"],
                "response_length_mean": summary["response_length_mean"],
                "response_length_std": summary["response_length_std"],
                "accuracy_drop_gsm_minus_transfer": conditions["accuracy_drop_gsm_minus_transfer"],
            })
    with (root / "feedback_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    pairwise_rows = []
    for policy, datasets in comparison["pairwise_vs_sft"].items():
        for dataset in ("gsm", "transfer"):
            pairwise_rows.append({"policy": policy, "dataset": dataset, **datasets[dataset]})
    with (root / "feedback_pairwise.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairwise_rows[0]))
        writer.writeheader()
        writer.writerows(pairwise_rows)

    diagnostic_rows = []
    types = ["corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
             "persuasive_filler_correct", "gold_distractor_wrong_final"]
    for mechanism in ("rlvr", "rlaif"):
        for variant in types:
            scores = diagnostic[mechanism][variant]
            diagnostic_rows.append({"mechanism": mechanism, "variant": variant, **scores})
    with (root / "feedback_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(diagnostic_rows[0]))
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    policies = list(comparison["policies"])
    labels = ["SFT", "RLVR", "RLAIF"]
    x = list(range(len(policies)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, dataset in enumerate(("gsm", "transfer")):
        vals = [comparison["policies"][p][dataset]["exact_accuracy"] for p in policies]
        ax.bar([v + (i - 0.5) * width for v in x], vals, width, label=dataset.upper())
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Exact-answer accuracy")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "task5_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for mechanism in ("rlvr", "rlaif"):
        vals = [comparison["pairwise_vs_sft"][mechanism][dataset]["ai_pairwise_win_rate_vs_sft"]
                for dataset in ("gsm", "transfer")]
        ax.plot(["GSM8K", "SVAMP"], vals, marker="o", linewidth=2, label=mechanism.upper())
    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AI pairwise win rate vs SFT")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "task5_pairwise_win_rates.png", dpi=180)
    plt.close(fig)

    categories = ["corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
                  "persuasive_filler_correct", "gold_distractor_wrong_final"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, mechanism in zip(axes, ("rlvr", "rlaif")):
        block = diagnostic[mechanism]
        ax.bar(range(len(categories)), [block[c]["better_response_rate"] for c in categories], label="Better")
        ax.bar(range(len(categories)), [block[c]["tie_rate"] for c in categories],
               bottom=[block[c]["better_response_rate"] for c in categories], label="Tie")
        bottoms = [block[c]["better_response_rate"] + block[c]["tie_rate"] for c in categories]
        ax.bar(range(len(categories)), [block[c]["wrong_preference_rate"] for c in categories],
               bottom=bottoms, label="Wrong")
        ax.set_title(mechanism.upper())
        ax.set_xticks(range(len(categories)), ["Reasoning\ncorrupt", "Final\nwrong", "Filler", "Gold\ndistractor"], rotation=0)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Pairwise outcome rate")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures / "task5_diagnostic_preferences.png", dpi=180)
    plt.close(fig)

    manifest = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"):
        manifest[str(path.relative_to(repo_path("."))).replace("\\", "/")] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (root / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    export_task5()
