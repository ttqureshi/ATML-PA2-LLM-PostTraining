from __future__ import annotations

import argparse
from collections import defaultdict
import json
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Students should decide and document exactly how prompts/completions are partitioned for the
    requested equal-generation comparison.
    """
    if k not in {2, 4, 8}:
        raise ValueError("Expected K in {2,4,8}")
    groups = []
    for prompt_id, rows in sorted(by_prompt.items()):
        if len(rows) < 8:
            raise ValueError(f"Prompt {prompt_id} has fewer than eight completions")
        # Use exactly the same first eight generations per prompt in every
        # condition, in consecutive non-overlapping blocks of K.
        selected = sorted(rows, key=lambda row: int(row["generation_index"]))[:8]
        if len({int(row["generation_index"]) for row in selected}) != 8:
            raise ValueError(f"Prompt {prompt_id} has duplicate generation indices")
        groups.extend(selected[i:i + k] for i in range(0, 8, k))
    return groups


def _summarize(groups):
    if not groups:
        return {"groups": 0, "informative_group_rate": None,
                "mean_group_reward_std": None, "relative_signal_variance": None}
    deviations = []
    stds = []
    for group in groups:
        rewards = np.asarray([float(row["reward"]) for row in group], dtype=float)
        mean, std = rewards.mean(), rewards.std()
        stds.append(std)
        deviations.extend(((rewards - mean) / (std + 1e-6)).tolist())
    return {"groups": len(groups),
            "informative_group_rate": float(np.mean(np.asarray(stds) > 1e-6)),
            "mean_group_reward_std": float(np.mean(stds)),
            "relative_signal_variance": float(np.var(deviations))}


def analyze(by_prompt, sizes=(2, 4, 8)):
    # Difficulty is the mean reward of all eight cached generations for a
    # prompt. Split once at the across-prompt median; ties go to the easy bin.
    difficulty = {pid: float(np.mean([float(r["reward"]) for r in sorted(rows, key=lambda r: int(r["generation_index"]))[:8]]))
                  for pid, rows in by_prompt.items()}
    median = float(np.median(list(difficulty.values())))
    ranked = sorted(difficulty, key=lambda pid: (difficulty[pid], pid))
    hard = set(ranked[:len(ranked) // 2])
    report = {"binning_rule": "Rank prompts by mean reward of eight fixed completions; lower half hard and upper half easy, breaking ties by source_index.",
              "difficulty_median": median, "conditions": {}}
    examples = []
    for label, candidates in (("hard", ranked[:len(ranked) // 2]),
                              ("easy", ranked[len(ranked) // 2:])):
        for pid in candidates[:2]:
            rows = sorted(by_prompt[pid], key=lambda row: int(row["generation_index"]))[:8]
            examples.append({"difficulty_bin": label, "source_index": pid,
                             "mean_reward": difficulty[pid],
                             "rewards": [float(row["reward"]) for row in rows],
                             "completions": [row["completion"] for row in rows]})
    report["illustrative_cached_examples"] = examples
    for k in sizes:
        groups = regroup_equal_generation_budget(by_prompt, int(k))
        generation_count = sum(len(group) for group in groups)
        bins = {label: _summarize([group for group in groups
                                   if (str(group[0]["source_index"]) in hard) == (label == "hard")])
                for label in ("hard", "easy")}
        report["conditions"][str(k)] = {
            "generation_count": generation_count, "overall": _summarize(groups), "difficulty_bins": bins}
    if len({v["generation_count"] for v in report["conditions"].values()}) != 1:
        raise AssertionError("Group-size conditions used unequal generation counts")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    report = analyze(by_prompt, cfg["group_sizes"])
    path = repo_path(cfg["results_dir"]) / "group_size.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved equal-generation group-size analysis to {path}")
    print(json.dumps(report["conditions"], indent=2))


if __name__ == "__main__":
    main()
