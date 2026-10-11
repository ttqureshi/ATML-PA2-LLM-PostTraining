from __future__ import annotations

import argparse
from collections import defaultdict

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        variants = by_problem[str(row["problem_id"])]
        if row["variant_type"] in variants:
            raise ValueError(f"Duplicate variant {row['variant_type']} for problem {row['problem_id']}")
        variants[row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def pair_outcome(preference: str) -> dict:
    return {
        "better": int(preference == "A"),
        "tie": int(preference == "TIE"),
        "wrong": int(preference == "B"),
    }


def summarize_diagnostic_pairs(records: list[dict]) -> dict:
    categories = sorted({r["variant_type"] for r in records})
    categories.append("clean_correct")
    result = {}
    for mechanism in ("rlvr", "rlaif"):
        result[mechanism] = {}
        for category in categories:
            selected = records if category == "clean_correct" else [
                r for r in records if r["variant_type"] == category
            ]
            key = "verifier_preference" if mechanism == "rlvr" else "judge_preference"
            counts = {label: sum(r[key] == label for r in selected) for label in ("A", "TIE", "B")}
            n = len(selected)
            result[mechanism][category] = {
                "n": n, "better_response_rate": counts["A"] / n,
                "tie_rate": counts["TIE"] / n,
                "wrong_preference_rate": counts["B"] / n,
            }
        result[mechanism]["Sreason"] = result[mechanism]["corrupt_reasoning_correct_final"]["better_response_rate"]
        result[mechanism]["Soutcome"] = result[mechanism]["good_reasoning_wrong_final"]["better_response_rate"]
        result[mechanism]["filler_susceptibility"] = result[mechanism]["persuasive_filler_correct"]["wrong_preference_rate"]
        result[mechanism]["distractor_robustness"] = result[mechanism]["gold_distractor_wrong_final"]["better_response_rate"]
    result["n_problems"] = len({r["problem_id"] for r in records})
    result["n_controlled_pairs"] = len(records)
    result["clean_correct_anchor_interpretation"] = (
        "The clean_correct row pools its four comparisons per problem against the four "
        "perturbation variants; it is not a separate fifth controlled-pair condition."
    )
    return result


def score_diagnostics(cfg: dict, groups: dict, judge) -> tuple[dict, list[dict]]:
    records = []
    for pid, variants in groups.items():
        clean = variants["clean_correct"]
        for variant_type in sorted(EXPECTED_VARIANTS - {"clean_correct"}):
            other = variants[variant_type]
            if clean["question"] != other["question"] or str(clean["gold_final"]) != str(other["gold_final"]):
                raise ValueError(f"Diagnostic pair {pid}/{variant_type} has mismatched question or gold answer")
            clean_reward = exact_reward(clean["response"], str(clean["gold_final"]))
            other_reward = exact_reward(other["response"], str(other["gold_final"]))
            verifier = "A" if clean_reward > other_reward else "B" if other_reward > clean_reward else "TIE"
            decision = judge.compare(clean["question"], clean["response"], other["response"])
            records.append({
                "problem_id": pid, "variant_type": variant_type,
                "question": clean["question"], "gold_final": str(clean["gold_final"]),
                "clean_response": clean["response"], "variant_response": other["response"],
                "clean_exact_reward": clean_reward, "variant_exact_reward": other_reward,
                "verifier_preference": verifier, "judge_preference": decision,
                "verifier_judge_agree": verifier == decision,
                "rlvr_outcome": pair_outcome(verifier), "rlaif_outcome": pair_outcome(decision),
            })
    return summarize_diagnostic_pairs(records), records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    out = repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostics"
    judge = PairwiseAIJudge(cfg, out / "judge_cache.json")
    summary, records = score_diagnostics(cfg, groups, judge)
    save_json(out / "summary.json", summary)
    write_jsonl(out / "controlled_pairs.jsonl", records)
    print(summary)


if __name__ == "__main__":
    main()
