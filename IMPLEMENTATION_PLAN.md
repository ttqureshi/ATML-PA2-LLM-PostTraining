# Implementation plan and dependency map

This plan tracks the coding and experiment work required by `ATML-PA2.pdf`. It is an engineering checklist, not report text.

| Workstream | Dependencies | Independent implementation | Required run/evidence |
| --- | --- | --- | --- |
| Setup | Starter clone | Pin packages; download revision-locked assets; validate all schemas | `check_environment` and `validate_assets` pass |
| Task 1 DPO | Setup | Correct objective; one-epoch baseline; beta forks; length-control fork; held-out evaluator | DPO loss/accuracy, sampled KL, reward, length, three strata, word limits, examples |
| Task 2 PPO | Setup | Correct clipped objective; 20-update continuation; cached clipping; epsilon/KL forks | Reward/KL/loss/entropy/clip/gradient/length trajectories; VRAM/time; held-out comparisons |
| Task 3 GRPO | Setup | Correct per-prompt group advantage; 20-update continuation; equal-generation K cache; normalization forks | Reward/KL/group informativeness/loss/gradient/entropy/length; VRAM/time; difficulty bins and length-conditioned statistics |
| Task 5 feedback | Setup | Fixed SFT/RLVR/RLAIF GSM and SVAMP evaluation; controlled perturbation scoring | Exact accuracy, AI win rate, format, length, verifier-judge agreement, ties, transfer drop, Sreason/Soutcome |
| Task 4 safety | Standard Task 1/2/3 adapters | Fixed four-policy XSTest generation; blind audit sheet; categorical judge; aggregate | Safe answer/over-refusal, unsafe compliance/justified refusal, ambiguity, category breakdown, manual agreement |
| Task 6 synthesis | Completed results of Tasks 1–5 | Human-authored analysis and report | Cross-task discussion in eight-page NeurIPS PDF; no AI-generated report text |

## Run order

1. Validate the pinned environment and course assets.
2. Implement and verify Tasks 1, 2, 3, and 5 independently. Keep each task in its own folder and use `common/` only for genuinely shared code.
3. Run the Task 1 standard DPO training, Task 2 standard PPO continuation, and Task 3 standard GRPO continuation. Keep their final adapters for Task 4.
4. Run the ablations and diagnostic analyses with fixed seeds, prompt IDs, decoding, and budgets. Retain all generated responses and machine-readable logs.
5. Run Task 4 on the four specified frozen policies only. Prepare and complete the 60-prompt blind manual audit before consulting AI labels.
6. Inspect saved examples and metrics. The student writes the PDF report, checks every table against saved files, commits the final code state, and publishes a public GitHub repository.

## Integrity checks

- Do not modify fixed datasets, cached rollouts, or supplied checkpoints.
- Do not use Task 4 or Task 5 evaluation data to tune Tasks 1–3.
- Distinguish the standard DPO one-epoch budget from the short beta forks.
- Reinitialize every PPO/GRPO fork from the same supplied midpoint state.
- Match prompt IDs and decoding within each controlled comparison.
- Keep sampled KL, reward scores, binary verifier scores, and AI preference rates as separate metrics.
- Report failed runs and missing results explicitly; never fabricate metrics.

## Current local limits

The local interpreter has pinned packages but CPU-only PyTorch and no CUDA GPU. Public model downloads and GPU training/evaluation remain necessary before empirical results can be claimed. The fixed course asset download completed and all non-self-referential pinned hashes were verified. Task 4's human audit and the PDF report require the student.
