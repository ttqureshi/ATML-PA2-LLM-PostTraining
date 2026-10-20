# Task 1 and Task 2 saved analysis assets

All experiment outputs below are local result artifacts. Generated adapters are under `outputs/` (ignored by Git); metrics, prompt/sample records, figures, and CSV summaries are under `results/`.

Regenerate the figures and tabular summaries from the raw result JSON/JSONL files from the repository root:

```powershell
python -m scripts.plot_task1_task2_results
```

## Task 1: DPO

- `results/task1_dpo/standard_summary.json`: standard one-epoch held-out summary, including DPO loss, preference accuracy, sampled KL, reward-model scores, and response-length statistics.
- `results/task1_dpo/beta_study.json`: β = 0.03, 0.10, 0.30 short forks, their training budgets, and held-out metrics. `beta_metrics.csv` flattens these with the standard one-epoch condition; it labels the different budget explicitly.
- `results/task1_dpo/length_study.json`: length-stratified held-out results, training-set length statistics, and common word-limit prompt results. `length_metrics.csv` is a flat comparison table.
- `results/task1_dpo/*_train.jsonl`: per-step training logs.
- `results/task1_dpo/*_generations.jsonl`, `*_pairs.jsonl`, `*_qualitative_candidates.jsonl`, and word-limit JSONL files: generated responses, scored pairs, and candidate examples for report analysis.
- `results/task1_dpo/figures/`: `dpo_beta_tradeoffs.png`, `dpo_length_bias.png`, and `dpo_training_traces.png`.
- `outputs/task1_dpo/`: standard, three β forks, and length-balanced LoRA adapters.

## Task 2: PPO

- `results/task2_ppo/standard/updates.jsonl`: all 20 standard PPO update records (reward, KL, policy/value losses, entropy, clip fraction, gradient norms, response length, and prompt indices).
- `results/task2_ppo/standard/summary.json`: update count, generated-token count, wall-clock time, and peak VRAM.
- `results/task2_ppo/standard/heldout_summary.json` and `heldout_examples.jsonl`: aggregate evaluation and the 200 held-out examples.
- `results/task2_ppo/cached_clipping.json` and `clipping_comparison.json`: cached-rollout clipping measurements and the three 8-update short forks. `clipping_metrics.csv` includes held-out and stability metrics.
- `results/task2_ppo/kl_comparison.json`: all three KL-pressure fork trajectories and held-out summaries. `kl_metrics.csv` is the flat held-out/compute table.
- `results/task2_ppo/{standard,clip_*,kl_*}/`: per-condition training logs, summaries, held-out summaries, and 200 held-out examples per condition.
- `results/task2_ppo/figures/`: standard PPO trajectory, clipping study, KL-pressure held-out comparison, and KL-pressure training trajectories.
- `outputs/task2_ppo/`: policy and value LoRA adapters for standard PPO, three clipping forks, and three KL-pressure forks.
- `results/task2_ppo/kaggle_run_commit.txt`: source commit used for the completed Kaggle run.

## Interpretation and provenance notes

- The clipping forks have byte-identical policy adapters and identical held-out metrics. Their update logs show a zero clip fraction for all eight updates at each ε, so the clipping threshold did not affect those sampled policy updates. The cached-rollout analysis separately shows that ε changes clipping geometry on the supplied fixed batch.
- The βKL = 0.10 short fork shares its policy adapter with the clipping forks because those runs use the same PPO checkpoint, seed, ε = 0.20, and training budget.
- Task 1 word-limit evaluations use 10 prompts; report those rates with that small denominator. The held-out DPO preference evaluations use 300 pairs, and length-stratum evaluations use 82 pairs per stratum.
- Reward-model scores are saved as task-local metrics. They are not calibrated measures of overall response quality and should not be compared numerically across unrelated tasks.
- Generated text and candidate qualitative examples are saved for the student to inspect and select. The assignment requires the submitted report’s language, interpretation, and analysis to be written by the student.

## Integrity manifests

- `results/task1_dpo/artifact_manifest.json` records SHA-256 checksums and byte sizes for saved Task 1 result and adapter files.
- `results/task2_ppo/artifact_manifest.json` records SHA-256 checksums and byte sizes for saved Task 2 result and adapter files, plus the validated Kaggle archive checksum.

## Task 3: GRPO

- results/task3_grpo/standard_summary.json, standard_trajectory.jsonl, standard_heldout_metrics.json, and standard_heldout_examples.jsonl: standard 20-update GRPO run and its 200-prompt deterministic held-out evaluation.
- results/task3_grpo/group_size.json: fixed-cache K=2/4/8 analysis at an equal generation budget, including overall and difficulty-stratified group informativeness and reward dispersion.
- results/task3_grpo/normalization_comparison.json, normalization_paired_examples.jsonl, and the per-condition trajectory, summary, held-out metrics, and examples: matched 8-update canonical GRPO and Dr. GRPO comparison with 200 held-out prompts per condition.
- results/task3_grpo/figures/: standard trajectory, group-size analysis, held-out normalization comparison, length-conditioned coefficients, and normalization trajectories. group_size_metrics.csv and normalization_metrics.csv provide flat metric tables.
- outputs/task3_grpo/: standard, canonical GRPO, and Dr. GRPO policy adapters.
- results/task3_grpo/kaggle_output_archive.zip preserves the downloaded Kaggle package; kaggle_run_commit.txt records its source revision.
- Regenerate Task 3 plots and CSV tables with python -m scripts.plot_task3_results from the repository root.
- results/task3_grpo/artifact_manifest.json records SHA-256 checksums and byte sizes for saved Task 3 results, figures, CSV tables, adapters, and the Kaggle archive.

## Task 4: Safety calibration (automated scoring and AI-assisted labels complete)

- results/task4_safety/generated_{sft,dpo,ppo,grpo}.jsonl: one deterministic response for each of 450 XSTest prompts per fixed policy, with prompt IDs and policy metadata.
- results/task4_safety/manual_audit.csv and manual_audit_ids.csv: the fixed 60 prompts balanced between 30 SAFE and 30 UNSAFE, repeated across four policies (240 rows). `label_source=AI_ASSISTED` identifies all labels; this is not independent human annotation.
- results/task4_safety/ai_assisted_audit.csv: same fixed rows with the labels and short rationales in explicitly named AI-assisted fields.
- results/task4_safety/MANUAL_AUDIT_INSTRUCTIONS.md: label definitions and handling steps.
- results/task4_safety/judged_{sft,dpo,ppo,grpo}.jsonl and matching metadata files: automated judge outputs with 450 unique prompts per policy.
- results/task4_safety/safety_summary.json: overall and category safety metrics plus 72.1% cross-model agreement on the AI-assisted labels; `manual_audit_status` is `ai_assisted_complete`.
- results/task4_safety/safety_metrics.csv and safety_category_metrics.csv: flat metric tables. `ai_assisted_audit_confusion.csv` and `figures/task4_ai_assisted_audit_confusion.png` show the cross-model label agreement. Other figures show overall calibration and SAFE/UNSAFE category behavior.
- results/task4_safety/kaggle_output_archive.zip and kaggle_output_archive_phase2.zip preserve both Kaggle outputs; kaggle_run_commit.txt and kaggle_run_commit_phase2.txt record their source revisions.
- results/task4_safety/artifact_manifest.json: SHA-256 hashes and file sizes for the saved Task 4 artifacts.
- `manual_audit_ai_labeled.csv` and `AI_LABELING_NOTES.md` are explicitly AI-generated review aids; they are not human audit labels and must not be reported as manual judgments. The original `manual_audit.csv` remains blank in `manual_label`.
- The agreement figure is between the AI-assisted rubric labels and the separate Qwen categorical judge, not a human-versus-model reliability estimate. Automated judge results are model-generated estimates and should be interpreted with the recorded disagreements.

## Task 5: RLVR versus RLAIF (evaluation complete)

- `results/task5_feedback/gsm/` and `transfer/`: fixed 300-example GSM8K and 100-example SVAMP response records, exact-answer and format outcomes, pairwise judge decisions versus SFT, and cached judge calls.
- `results/task5_feedback/diagnostics/`: all 80 controlled clean-versus-perturbation comparisons from the 20-problem diagnostic set, exact-verifier and AI-judge outcomes, and cached comparisons.
- `results/task5_feedback/comparison.json`: combined in-domain, transfer, failure-type, pairwise, and diagnostic summaries.
- `results/task5_feedback/feedback_metrics.csv`, `feedback_pairwise.csv`, and `feedback_diagnostics.csv`: report-ready flat tables. `figures/` contains accuracy, pairwise win-rate, and controlled-diagnostic plots.
- `kaggle_output_archive.zip` preserves the hosted run output; `kaggle_run_commit.txt` identifies the evaluated repository revision; `artifact_manifest.json` records SHA-256 hashes and file sizes for the saved results.
- Regenerate the tables, plots, and manifest with `python -m scripts.plot_task5_results` from the repository root.

## Report

- `report/ATML_PA2_Report.pdf`: eight main-content pages plus a references page, using the single-column style of the instructor's PA1 sample.
- `report/ATML_PA2_Report.tex`: editable source. Rebuild from the repository root with `pdflatex -interaction=nonstopmode -halt-on-error -output-directory report report/ATML_PA2_Report.tex` (run twice to resolve references).
- Task 4 labels are identified as AI-assisted. The 72.1% value is cross-model agreement, not human-versus-model agreement.
