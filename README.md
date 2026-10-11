# ATML PA2: LLM post-training experiments

This repository extends the [course starter](https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining) for the Fall 2026 assignment. The assignment PDF is the authority for experiment definitions and reporting requirements.

## Setup

Run from the repository root in a Python environment with a CUDA GPU and enough disk space for the course assets plus the public base, reward, and judge models.

```bash
python -m pip install -r requirements.txt
python -m scripts.check_environment
python -m scripts.download_assets
python -m scripts.validate_assets
```

The downloader pins the course asset revision `0b350481fb03f5525a35bcdec4131bd4fe487f98` from [the course Hugging Face dataset](https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets). Do not edit files under `data/`, `cached/`, or `checkpoints/`. Public Hugging Face models load on demand. The released configuration files fix seed, prompt indices, model IDs, LoRA settings, budgets, and decoding.

## Reproduce the experiments

Run the command blocks in order. Each module is a fresh process and saves its own JSON/JSONL results under `results/`. Task 4 depends on the **standard** adapters from Tasks 1–3. Task 5 uses supplied frozen adapters and can run independently after setup.

### Task 1: DPO

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

The standard run trains for one epoch. The beta sweep uses three separate 600-example forks from the base policy. The length study trains a separate adapter on the supplied balanced set and evaluates the three fixed held-out strata and common word-limit prompts.

### Task 2: PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

The standard policy and critic continue from the supplied midpoint for 20 updates. Both eight-update ablation families restart from that same midpoint. The clipping command first analyzes the fixed cached rollout batch, then runs the clipping forks.

### Task 3: GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

The standard continuation runs 20 updates from the supplied midpoint. The K=2/4/8 comparison regroups the same cached completions at equal generation count. Canonical and Dr. GRPO normalization use separate short forks from the same midpoint.

### Task 4: Safety calibration

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
# Manually label all rows in results/task4_safety/manual_audit.csv without viewing AI labels.
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

The audit sheet fixes 30 safe and 30 unsafe XSTest IDs and includes each of the four policy responses for those IDs. Enter one of `SAFE_ANSWER`, `JUSTIFIED_REFUSAL`, `UNSAFE_COMPLIANCE`, `OVER_REFUSAL`, or `AMBIGUOUS` in every `manual_label` cell. Keep `notes` for disagreements. The judge and aggregate scripts do not fill manual labels. If the audit is incomplete, `safety_summary.json` marks it incomplete.

### Task 5: RLVR versus RLAIF

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

The GSM8K and SVAMP runs use the fixed course sets. The controlled study uses the 100 supplied responses, and each AI preference decision is cached. The final command combines saved results; it does not retrain policies.

## Results and provenance

`results/` holds small JSON/JSONL/CSV logs and generated-response records for reported values. `outputs/` holds student adapters and is ignored by Git. Downloaded data, caches, checkpoints, and model weights are ignored. Record the exact Git commit used for the final results.

The course starter supplies `common/`, configurations, data/model loaders, fixed verifiers and judges, and experiment scaffolds; those parts are attributed to the [course repository](https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining). No additional external code has been copied into this repository.

The assignment permits LLM coding assistance but requires the PDF report's wording, interpretation, and analysis to be entirely the student's own. This repository does not generate report prose.
