from __future__ import annotations

import argparse
from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json


def _failure_types(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "correct": sum(bool(r["exact_correct"]) for r in rows),
        "wrong_designated_final": sum(bool(r["format_compliant"]) and not bool(r["exact_correct"]) for r in rows),
        "missing_designated_final": sum(not bool(r["format_compliant"]) for r in rows),
        "n": n,
    }


def combine_results(cfg: dict) -> dict:
    base = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm = load_json(base / "gsm" / "summary.json")
    transfer = load_json(base / "transfer" / "summary.json")
    diagnostic = load_json(base / "diagnostics" / "summary.json")
    by_policy = {}
    for name in ("sft", "rlvr", "rlaif"):
        in_domain = gsm["policies"][name]
        out_domain = transfer["policies"][name]
        by_policy[name] = {
            "gsm": in_domain,
            "transfer": out_domain,
            "accuracy_drop_gsm_minus_transfer": in_domain["exact_accuracy"] - out_domain["exact_accuracy"],
            "format_compliance_drop_gsm_minus_transfer": in_domain["format_compliance"] - out_domain["format_compliance"],
            "response_length_change_transfer_minus_gsm": out_domain["response_length_mean"] - in_domain["response_length_mean"],
            "gsm_failure_types": _failure_types(read_jsonl(base / "gsm" / f"{name}_responses.jsonl")),
            "transfer_failure_types": _failure_types(read_jsonl(base / "transfer" / f"{name}_responses.jsonl")),
        }
    pairwise = {}
    for name in ("rlvr", "rlaif"):
        g, t = gsm["pairwise"][name], transfer["pairwise"][name]
        pairwise[name] = {
            "gsm": g, "transfer": t,
            "ai_win_rate_drop_gsm_minus_transfer": g["ai_pairwise_win_rate_vs_sft"] - t["ai_pairwise_win_rate_vs_sft"],
            "agreement_change_transfer_minus_gsm": t["verifier_judge_agreement_all"] - g["verifier_judge_agreement_all"],
        }
    result = {"policies": by_policy, "pairwise_vs_sft": pairwise, "diagnostics": diagnostic}
    save_json(base / "comparison.json", result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    result = combine_results(load_yaml(args.config))
    print(result)


if __name__ == "__main__":
    main()
