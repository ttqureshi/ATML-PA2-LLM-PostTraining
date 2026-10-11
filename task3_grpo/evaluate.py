from __future__ import annotations

import argparse
import json

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate(config_path: str, adapter: str, name: str = "standard"):
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg, policy, tokenizer = bundle["cfg"], bundle["policy"], bundle["tokenizer"]
    reward_model, reward_tokenizer = bundle["reward"]
    examples = []
    cap = int(cfg["max_completion_length"])
    # One deterministic response per held-out prompt, same decoding for forks.
    for index, row in enumerate(bundle["rows"]):
        prompt = prompt_messages(row)
        batch = batch_generate(policy, tokenizer, [prompt], int(cfg["max_prompt_length"]),
                               cap, do_sample=False)
        response = batch["responses"][0]
        reward = float(score_reward_pairs(reward_model, reward_tokenizer,
                                          [prompt], [response])[0])
        with torch.no_grad():
            args = (batch["sequences"], batch["attention_mask"],
                    batch["prompt_width"], batch["response_ids"])
            new = response_token_logprobs(policy, *args)[0]
            with reference_mode(policy):
                ref = response_token_logprobs(policy, *args)[0]
        mask = batch["response_mask"]
        kl = float(((new - ref) * mask).sum() / mask.sum().clamp_min(1))
        examples.append({"source_index": row.get("source_index", index),
                         "prompt": prompt, "response": response,
                         "reward": reward, "sampled_kl": kl,
                         "response_length": batch["response_lengths"][0],
                         "truncated": batch["truncated"][0]})
    if not examples:
        raise ValueError("The held-out GRPO prompt pool is empty")
    metrics = {
        "name": name, "adapter": str(repo_path(adapter)), "examples": len(examples),
        "mean_reward": sum(x["reward"] for x in examples) / len(examples),
        "mean_sampled_kl": sum(x["sampled_kl"] for x in examples) / len(examples),
        "mean_response_length": sum(x["response_length"] for x in examples) / len(examples),
        "truncated_fraction": sum(x["truncated"] for x in examples) / len(examples),
        "decoding": {"do_sample": False, "max_new_tokens": cap},
    }
    results = repo_path(cfg["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    with (results / f"{name}_heldout_examples.jsonl").open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    (results / f"{name}_heldout_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    print(json.dumps(evaluate(args.config, args.adapter, args.name), indent=2))


if __name__ == "__main__":
    main()
