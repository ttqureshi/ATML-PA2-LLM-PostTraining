from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.models import load_policy, load_tokenizer


def policy_specs(cfg):
    expected = {
        "sft": None,
        "dpo": "outputs/task1_dpo/standard",
        "ppo": "outputs/task2_ppo/standard",
        "grpo": "outputs/task3_grpo/standard",
    }
    for name, adapter in expected.items():
        if cfg["policies"].get(name) != adapter:
            raise ValueError(
                f"Task 4 requires the fixed {name} policy {adapter!r}; "
                f"config specifies {cfg['policies'].get(name)!r}"
            )
    return expected


def load_xstest(cfg):
    df = pd.read_csv(repo_path(cfg["paths"]["xstest"]))
    if df["xstest_id"].duplicated().any():
        raise ValueError("Duplicate XSTest IDs")
    return df


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    if adapter and not repo_path(adapter).exists():
        raise FileNotFoundError(f"Required standard adapter missing: {adapter}")
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model, tokenizer, prompts, max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0, top_p=1.0, do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]), "policy": policy_name,
                "prompt": str(row["prompt"]), "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]), "response": response,
                "response_tokens": int(n_tok),
            })
    del model
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policy", choices=["all", "sft", "dpo", "ppo", "grpo"], default="all")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    names = list(policy_specs(cfg)) if args.policy == "all" else [args.policy]
    for name in names:
        adapter = policy_specs(cfg)[name]
        if adapter and not repo_path(adapter).exists():
            raise FileNotFoundError(f"Required standard adapter missing: {adapter}")
        path = outdir / f"generated_{name}.jsonl"
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} exists; pass --force to regenerate")
    for name in names:
        path = outdir / f"generated_{name}.jsonl"
        set_seed(int(cfg["seed"]))
        rows = generate_for_policy(cfg, name, args.batch_size)
        write_jsonl(path, rows)
        save_json(outdir / f"generated_{name}_meta.json", {
            "policy": name, "adapter": policy_specs(cfg)[name], "seed": int(cfg["seed"]),
            "prompt_ids": [r["xstest_id"] for r in rows],
            "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0,
                         "max_new_tokens": int(cfg["safety_max_new_tokens"])},
        })
        print(path, len(rows))


if __name__ == "__main__":
    main()
