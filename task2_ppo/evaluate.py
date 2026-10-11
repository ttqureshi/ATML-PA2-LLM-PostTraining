from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json, set_seed
from common.models import load_policy, load_reward_model, load_tokenizer
from task2_ppo.workflow import evaluate_policy


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    evaluate_adapter(args.config, args.adapter, args.name)


def evaluate_adapter(config_path: str, adapter: str, name: str):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    set_seed(int(cfg["seed"]))
    reward_model, reward_tokenizer = bundle["reward"]
    summary, records = evaluate_policy(
        bundle["policy"], bundle["tokenizer"], reward_model,
        reward_tokenizer, bundle["rows"], cfg,
    )
    target = repo_path(cfg["results_dir"]) / name
    save_json(target / "heldout_summary.json", summary)
    write_jsonl(target / "heldout_examples.jsonl", records)
    print(summary)
    return summary, records


if __name__ == "__main__":
    main()
