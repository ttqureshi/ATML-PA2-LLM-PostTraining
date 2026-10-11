from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


POLICIES = ("sft", "dpo", "ppo", "grpo")


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ("SAFE", "UNSAFE"):
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    rows = {name: read_jsonl(outdir / f"generated_{name}.jsonl") for name in POLICIES}
    ids = fixed_audit_ids(rows["sft"], int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    sheet = outdir / "manual_audit.csv"
    ids_path = outdir / "manual_audit_ids.csv"
    if sheet.exists() and not args.force:
        raise FileExistsError(f"{sheet} exists; pass --force to replace it")
    if ids_path.exists() and not args.force:
        existing = pd.read_csv(ids_path)["xstest_id"].astype(int).tolist()
        if existing != ids:
            raise ValueError("Existing fixed audit IDs differ; pass --force to replace them")
    records = []
    for name in POLICIES:
        indexed = {int(r["xstest_id"]): r for r in rows[name]}
        if len(indexed) != len(rows[name]) or not set(ids).issubset(indexed):
            raise ValueError(f"Missing or duplicate fixed audit IDs for {name}")
        for item_id in ids:
            row = indexed[item_id]
            records.append({
                "xstest_id": item_id, "policy": name,
                "benchmark_class": row["benchmark_class"], "type": row["type"],
                "prompt": row["prompt"], "response": row["response"],
                "manual_label": "", "notes": "",
            })
    pd.DataFrame({"xstest_id": ids}).to_csv(ids_path, index=False)
    pd.DataFrame(records).to_csv(sheet, index=False)
    print(sheet, len(records))
    print("Label responses manually without viewing judged_*.jsonl first.")


if __name__ == "__main__":
    main()
