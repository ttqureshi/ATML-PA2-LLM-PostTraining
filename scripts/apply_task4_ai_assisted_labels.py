from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common.data import repo_path

LABELS = {
    "SAFE_ANSWER",
    "JUSTIFIED_REFUSAL",
    "UNSAFE_COMPLIANCE",
    "OVER_REFUSAL",
    "AMBIGUOUS",
}
KEY = ("policy", "xstest_id")
MATCH_FIELDS = ("benchmark_class", "type", "prompt", "response")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply explicitly AI-assisted Task 4 audit labels.")
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()

    outdir = repo_path("results") / "task4_safety"
    manual_path = outdir / "manual_audit.csv"
    ai_path = outdir / "manual_audit_ai_labeled.csv"
    manual_fields, manual_rows = read_csv(manual_path)
    ai_fields, ai_rows = read_csv(ai_path)
    if len(manual_rows) != 240 or len(ai_rows) != 240:
        raise ValueError("Expected 240 rows in both the fixed audit sheet and AI label file")

    def index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
        result = {(r["policy"], str(int(r["xstest_id"]))): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate policy / XSTest ID in audit rows")
        return result

    base = index(manual_rows)
    proposed = index(ai_rows)
    if base.keys() != proposed.keys():
        raise ValueError("AI labels do not cover exactly the fixed audit rows")
    for key, row in base.items():
        other = proposed[key]
        if any(row[field] != other[field] for field in MATCH_FIELDS):
            raise ValueError(f"Audit content mismatch for {key}")
        if row.get("manual_label", "").strip():
            raise ValueError(f"Refusing to overwrite an existing label for {key}")
        label = other.get("ai_label", "").strip().upper()
        if label not in LABELS:
            raise ValueError(f"Invalid AI label for {key}: {label!r}")
        row["xstest_id"] = str(int(row["xstest_id"]))
        row["manual_label"] = label
        row["label_source"] = "AI_ASSISTED"
        row["notes"] = "AI-generated rubric label; interpret as AI-assisted audit, not an independent human judgment."

    if "label_source" not in manual_fields:
        manual_fields.insert(manual_fields.index("notes"), "label_source")
    write_csv(manual_path, manual_fields, manual_rows)

    audit_fields = ["xstest_id", "policy", "benchmark_class", "type", "prompt", "response",
                    "audit_label", "label_source", "rationale"]
    audit_rows = []
    for row in ai_rows:
        audit_rows.append({
            "xstest_id": str(int(row["xstest_id"])),
            "policy": row["policy"],
            "benchmark_class": row["benchmark_class"],
            "type": row["type"],
            "prompt": row["prompt"],
            "response": row["response"],
            "audit_label": row["ai_label"].strip().upper(),
            "label_source": "AI_ASSISTED",
            "rationale": row.get("ai_rationale", ""),
        })
    write_csv(outdir / "ai_assisted_audit.csv", audit_fields, audit_rows)
    print(f"Saved {len(audit_rows)} AI-assisted labels; source field is AI_ASSISTED")


if __name__ == "__main__":
    main()
