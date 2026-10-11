from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        if max_examples <= 0:
            raise ValueError("max_examples must be positive")
        rows = rows[:int(max_examples)]
    if not rows:
        raise ValueError(f"No training rows in {path}")

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dataset_path": path,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def _to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _pair_logps(model, chosen, rejected):
    chosen_logp, _, _ = response_sequence_logprobs(model, chosen)
    rejected_logp, _, _ = response_sequence_logprobs(model, rejected)
    return chosen_logp, rejected_logp


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.mkdir(parents=True, exist_ok=True)
    results = repo_path(cfg["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    log_path = results / f"{run_name}_train.jsonl"
    if log_path.exists():
        log_path.unlink()

    model = bundle["model"]
    optimizer = bundle["optimizer"]
    accumulation = int(cfg["grad_accum_steps"])
    if accumulation < 1:
        raise ValueError("grad_accum_steps must be positive")
    if bundle["beta"] <= 0:
        raise ValueError("beta must be positive")
    epochs = int(cfg["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be positive")
    device = next(model.parameters()).device
    timer = wall_timer()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_correct = 0.0
    total_pairs = 0
    optimizer_steps = 0
    micro_steps = 0

    for epoch in range(epochs):
        loader = bundle["loader"]
        n_batches = len(loader)
        for batch_index, (chosen, rejected) in enumerate(loader):
            chosen = _to_device(chosen, device)
            rejected = _to_device(rejected, device)
            with torch.no_grad(), reference_mode(model):
                ref_chosen, ref_rejected = _pair_logps(model, chosen, rejected)
            policy_chosen, policy_rejected = _pair_logps(model, chosen, rejected)
            loss, diagnostics = dpo_loss(
                policy_chosen, policy_rejected,
                ref_chosen, ref_rejected, bundle["beta"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite DPO loss at epoch {epoch + 1}, batch {batch_index + 1}")

            # A partial final accumulation window is divided by its actual size.
            window_start = (batch_index // accumulation) * accumulation
            window_size = min(accumulation, n_batches - window_start)
            (loss / window_size).backward()
            batch_size = chosen["input_ids"].shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += float(diagnostics["preference_accuracy"].item()) * batch_size
            total_pairs += batch_size
            micro_steps += 1
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == n_batches:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters(model), float(cfg["max_grad_norm"])
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                append_jsonl(log_path, {
                    "run_name": run_name, "epoch": epoch + 1,
                    "optimizer_step": optimizer_steps, "micro_step": micro_steps,
                    "batch_loss": float(loss.detach().item()),
                    "batch_preference_accuracy": float(diagnostics["preference_accuracy"].item()),
                    "grad_norm": float(grad_norm.item()),
                    "examples_seen": total_pairs, "elapsed_seconds": timer(),
                })

    model.save_pretrained(output)
    bundle["tokenizer"].save_pretrained(output)
    summary = {
        "run_name": run_name,
        "config": str(config_path),
        "dataset": str(bundle["dataset_path"]),
        "examples": len(bundle["rows"]),
        "epochs": epochs,
        "beta": bundle["beta"],
        "seed": int(cfg["seed"]),
        "batch_size": int(cfg["batch_size"]),
        "grad_accum_steps": accumulation,
        "optimizer_steps": optimizer_steps,
        "mean_training_dpo_loss": total_loss / total_pairs,
        "mean_training_preference_accuracy": total_correct / total_pairs,
        "elapsed_seconds": timer(),
        "adapter": str(output),
    }
    save_json(results / f"{run_name}_train_summary.json", summary)
    print(summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
