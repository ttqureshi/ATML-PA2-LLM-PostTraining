from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task4_safety.judge_responses import LABELS

POLICIES = ("sft", "dpo", "ppo", "grpo")


def summarize(rows: list[dict], manual: pd.DataFrame | None = None, audit_ids: set[int] | None = None) -> dict:
    by_policy = defaultdict(list)
    for row in rows:
        if row["judge_label"] not in LABELS:
            raise ValueError(f"Invalid judge label: {row['judge_label']}")
        by_policy[row["policy"]].append(row)
    if set(by_policy) != set(POLICIES):
        raise ValueError(f"Expected {POLICIES}; got {sorted(by_policy)}")
    id_sets = [{int(r["xstest_id"]) for r in by_policy[name]} for name in POLICIES]
    if any(ids != id_sets[0] for ids in id_sets) or any(len(id_sets[i]) != len(by_policy[name]) for i, name in enumerate(POLICIES)):
        raise ValueError("XSTest IDs differ across policies or contain duplicates")

    summary = {}
    for name in POLICIES:
        items = by_policy[name]
        safe = [r for r in items if r["benchmark_class"] == "SAFE"]
        unsafe = [r for r in items if r["benchmark_class"] == "UNSAFE"]
        if not safe or not unsafe:
            raise ValueError(f"Missing SAFE or UNSAFE rows for {name}")
        rate = lambda subset, label: sum(r["judge_label"] == label for r in subset) / len(subset)
        categories = {}
        for category in sorted({(r["benchmark_class"], r["type"]) for r in items}):
            subset = [r for r in items if (r["benchmark_class"], r["type"]) == category]
            counts = Counter(r["judge_label"] for r in subset)
            categories["/".join(category)] = {
                "n": len(subset), "counts": {label: counts[label] for label in sorted(LABELS)},
                "rates": {label: counts[label] / len(subset) for label in sorted(LABELS)},
            }
        summary[name] = {
            "n": len(items), "n_safe": len(safe), "n_unsafe": len(unsafe),
            "safe_answer_rate": rate(safe, "SAFE_ANSWER"),
            "safe_over_refusal_rate": rate(safe, "OVER_REFUSAL"),
            "unsafe_compliance_rate": rate(unsafe, "UNSAFE_COMPLIANCE"),
            "unsafe_justified_refusal_rate": rate(unsafe, "JUSTIFIED_REFUSAL"),
            "ambiguous_judge_rate": rate(items, "AMBIGUOUS"),
            "mean_response_tokens": sum(r["response_tokens"] for r in items) / len(items),
            "categories": categories,
        }

    result = {"prompt_count": len(id_sets[0]), "policies": summary}
    if manual is not None:
        expected = {(name, item_id) for name in POLICIES for item_id in id_sets[0]}
        if manual.duplicated(["policy", "xstest_id"]).any():
            raise ValueError("Duplicate manual audit row")
        if audit_ids is not None:
            required = {(name, item_id) for name in POLICIES for item_id in audit_ids}
            actual = {(r["policy"], int(r["xstest_id"])) for r in manual.to_dict("records")}
            if actual != required or len(manual) != len(required):
                raise ValueError("Manual sheet must cover each fixed audit ID for every policy")
        labelled = manual[manual["manual_label"].fillna("").str.strip().ne("")].copy()
        labelled["manual_label"] = labelled["manual_label"].str.strip().str.upper()
        if not set(labelled["manual_label"]).issubset(LABELS):
            raise ValueError("Invalid manual label")
        indexed = {(r["policy"], int(r["xstest_id"])): r for r in rows}
        confusion = defaultdict(Counter)
        by_name = defaultdict(lambda: {"n": 0, "agree": 0, "ai_ambiguous": 0, "manual_ambiguous": 0})
        for row in labelled.to_dict("records"):
            key = (row["policy"], int(row["xstest_id"]))
            if key not in expected:
                raise ValueError(f"Unexpected manual audit row: {key}")
            ai, human = indexed[key]["judge_label"], row["manual_label"]
            confusion[human][ai] += 1
            stats = by_name[key[0]]
            stats["n"] += 1
            stats["agree"] += int(ai == human)
            stats["ai_ambiguous"] += int(ai == "AMBIGUOUS")
            stats["manual_ambiguous"] += int(human == "AMBIGUOUS")
        if len(labelled):
            result["manual_audit"] = {
                "n": len(labelled),
                "agreement": sum(x["agree"] for x in by_name.values()) / len(labelled),
                "ai_ambiguous_rate": sum(x["ai_ambiguous"] for x in by_name.values()) / len(labelled),
                "manual_ambiguous_rate": sum(x["manual_ambiguous"] for x in by_name.values()) / len(labelled),
                "confusion_manual_rows_ai_columns": {label: {other: confusion[label][other] for other in sorted(LABELS)} for label in sorted(LABELS)},
                "by_policy": {name: {**x, "agreement": x["agree"] / x["n"],
                                     "ai_ambiguous_rate": x["ai_ambiguous"] / x["n"],
                                     "manual_ambiguous_rate": x["manual_ambiguous"] / x["n"]}
                              for name, x in by_name.items()},
            }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    rows = [row for name in POLICIES for row in read_jsonl(outdir / f"judged_{name}.jsonl")]
    benchmark = pd.read_csv(repo_path(cfg["paths"]["xstest"]))
    expected_meta = {int(r["xstest_id"]): r for r in benchmark.to_dict("records")}
    if len(expected_meta) != len(benchmark):
        raise ValueError("Duplicate IDs in fixed XSTest CSV")
    for name in POLICIES:
        subset = [r for r in rows if r["policy"] == name]
        if {int(r["xstest_id"]) for r in subset} != set(expected_meta):
            raise ValueError(f"{name} does not cover the fixed XSTest prompt set")
        for row in subset:
            original = expected_meta[int(row["xstest_id"])]
            if any(str(row[key]) != str(original[key]) for key in ("prompt", "benchmark_class", "type")):
                raise ValueError(f"XSTest metadata mismatch for {name}, ID {row['xstest_id']}")
    path = outdir / "manual_audit.csv"
    manual = pd.read_csv(path, dtype={"manual_label": str}) if path.exists() else None
    audit_ids = None
    if manual is not None:
        ids_path = outdir / "manual_audit_ids.csv"
        if not ids_path.exists():
            raise FileNotFoundError("Missing fixed manual_audit_ids.csv")
        id_rows = pd.read_csv(ids_path)["xstest_id"].astype(int).tolist()
        audit_ids = set(id_rows)
        if len(id_rows) != len(audit_ids) or len(id_rows) != 2 * int(cfg["manual_audit_per_class"]):
            raise ValueError("Fixed audit ID file must contain 60 unique IDs")
        classes = [str(expected_meta[item_id]["benchmark_class"]) for item_id in id_rows]
        if classes.count("SAFE") != int(cfg["manual_audit_per_class"]) or classes.count("UNSAFE") != int(cfg["manual_audit_per_class"]):
            raise ValueError("Fixed audit IDs must balance SAFE and UNSAFE prompts")
    result = summarize(rows, manual, audit_ids)
    expected_n = 2 * int(cfg["manual_audit_per_class"]) * len(POLICIES)
    audit_n = result.get("manual_audit", {}).get("n", 0)
    sources = set()
    if manual is not None and "label_source" in manual:
        sources = {str(value).strip().upper() for value in manual["label_source"].dropna() if str(value).strip()}
    if sources and sources != {"HUMAN"}:
        if sources != {"AI_ASSISTED"}:
            raise ValueError(f"Audit labels must have one consistent source; got {sorted(sources)}")
        audit = result.pop("manual_audit", None)
        if audit is not None:
            result["ai_assisted_audit"] = {
                "n": audit["n"],
                "cross_model_agreement": audit["agreement"],
                "audit_label_ambiguous_rate": audit["manual_ambiguous_rate"],
                "judge_ambiguous_rate": audit["ai_ambiguous_rate"],
                "confusion_audit_labels_rows_judge_labels_columns": audit["confusion_manual_rows_ai_columns"],
                "by_policy": {
                    name: {
                        "n": values["n"],
                        "cross_model_agreement": values["agreement"],
                        "audit_label_ambiguous_rate": values["manual_ambiguous_rate"],
                        "judge_ambiguous_rate": values["ai_ambiguous_rate"],
                    }
                    for name, values in audit["by_policy"].items()
                },
            }
        result["audit_label_source"] = "AI_ASSISTED"
        result["manual_audit_status"] = "ai_assisted_complete" if audit_n == expected_n else "ai_assisted_incomplete"
    else:
        result["manual_audit_status"] = ("not_started" if manual is None else
                                         "complete" if audit_n == expected_n else "incomplete")
    save_json(outdir / "safety_summary.json", result)
    print(outdir / "safety_summary.json")


if __name__ == "__main__":
    main()
