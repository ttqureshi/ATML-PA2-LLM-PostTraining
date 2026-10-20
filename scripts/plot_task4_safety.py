from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from common.data import load_yaml, repo_path
from common.logging_utils import load_json


def export_safety(config_path: str = "configs/feedback.yaml") -> None:
    cfg = load_yaml(config_path)
    root = repo_path(cfg["results_dir"]) / "task4_safety"
    summary = load_json(root / "safety_summary.json")
    policies = ("sft", "dpo", "ppo", "grpo")
    metrics = summary["policies"]
    if set(metrics) != set(policies) or summary.get("prompt_count") != 450:
        raise ValueError("Task 4 summary is incomplete or has unexpected policies/prompt count")

    fields = ("safe_answer_rate", "safe_over_refusal_rate", "unsafe_compliance_rate",
              "unsafe_justified_refusal_rate", "ambiguous_judge_rate", "mean_response_tokens")
    rows = [{"policy": policy, **{field: metrics[policy][field] for field in fields}}
            for policy in policies]
    with (root / "safety_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    category_rows = []
    for policy in policies:
        for key, values in metrics[policy]["categories"].items():
            benchmark_class, category = key.split("/", 1)
            category_rows.append({"policy": policy, "benchmark_class": benchmark_class,
                                  "category": category, "n": values["n"],
                                  **{f"rate_{label.lower()}": values["rates"][label]
                                     for label in sorted(values["rates"])}})
    with (root / "safety_category_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(category_rows[0]))
        writer.writeheader()
        writer.writerows(category_rows)

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    display_names = ["SFT", "DPO", "PPO", "GRPO"]
    rate_fields = ("safe_answer_rate", "safe_over_refusal_rate",
                   "unsafe_compliance_rate", "unsafe_justified_refusal_rate")
    labels = ("Safe answer", "Safe over-refusal", "Unsafe compliance", "Justified refusal")
    x = list(range(len(policies)))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, (field, label) in enumerate(zip(rate_fields, labels)):
        vals = [metrics[p][field] for p in policies]
        ax.bar([v + (i - 1.5) * width for v in x], vals, width, label=label)
    ax.set_xticks(x, display_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "task4_safety_calibration.png", dpi=180)
    plt.close(fig)

    for benchmark_class in ("SAFE", "UNSAFE"):
        cats = sorted({row["category"] for row in category_rows
                       if row["benchmark_class"] == benchmark_class})
        labels_set = sorted(metrics["sft"]["categories"][
            next(k for k in metrics["sft"]["categories"] if k.startswith(benchmark_class + "/"))
        ]["rates"])
        fig, axes = plt.subplots(len(policies), 1, figsize=(max(8, len(cats) * 0.9), 9), sharex=True, sharey=True)
        for ax, policy in zip(axes, policies):
            row_index = {row["category"]: row for row in category_rows
                         if row["policy"] == policy and row["benchmark_class"] == benchmark_class}
            for j, label in enumerate(labels_set):
                values = [row_index[c][f"rate_{label.lower()}"] for c in cats]
                ax.plot(range(len(cats)), values, marker="o", label=label)
            ax.set_ylim(0, 1)
            ax.set_ylabel(policy.upper())
            ax.spines[["top", "right"]].set_visible(False)
        axes[-1].set_xticks(range(len(cats)), cats, rotation=35, ha="right")
        axes[0].legend(frameon=False, ncol=min(3, len(labels_set)))
        fig.suptitle(f"{benchmark_class.title()} XSTest category behavior")
        fig.tight_layout()
        fig.savefig(figures / f"task4_{benchmark_class.lower()}_category_behavior.png", dpi=180)
        plt.close(fig)

    audit = summary.get("ai_assisted_audit")
    if audit:
        labels = sorted(audit["confusion_audit_labels_rows_judge_labels_columns"])
        confusion = audit["confusion_audit_labels_rows_judge_labels_columns"]
        with (root / "ai_assisted_audit_confusion.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ai_assisted_label", *labels])
            for label in labels:
                writer.writerow([label, *[confusion[label][other] for other in labels]])

        matrix = [[confusion[label][other] for other in labels] for label in labels]
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels)), [x.replace("_", "\n") for x in labels], rotation=0)
        ax.set_yticks(range(len(labels)), [x.replace("_", "\n") for x in labels])
        ax.set_xlabel("Qwen safety judge label")
        ax.set_ylabel("AI-assisted audit label")
        ax.set_title(f"Cross-model agreement (n={audit['n']}, {audit['cross_model_agreement']:.1%})")
        threshold = max(max(row) for row in matrix) / 2
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center",
                        color="white" if value > threshold else "black")
        fig.colorbar(image, ax=ax, label="Audit rows")
        fig.tight_layout()
        fig.savefig(figures / "task4_ai_assisted_audit_confusion.png", dpi=180)
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
    export_safety()
