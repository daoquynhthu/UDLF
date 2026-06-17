# Issues

This file tracks only repeated problems, active blockers, or major risks that
materially affect execution. Fast, one-off fixes should not be recorded here.

## Active

### Stage A state-causality gate is not passing yet

Status: open.

The Stage A training harness now has real-data loading, CUDA execution,
segmented carry, checkpoint/resume, metrics, and intervention evaluation. The
training pipeline itself is usable, but the current exact-memory synthetic probe
does not yet prove that the model relies on the persistent latent state.

Evidence:

- A 30-step FineWeb CUDA probe made zero-state much worse than correct state,
  but swapped state was effectively tied with correct state.
- A 200-step repeating-pattern CUDA probe stayed near the random-token baseline;
  zero-state was worse, shifted state was unchanged, and swapped state was only
  marginally worse.
- After adding suffix-only loss masking and correcting shifted-state evaluation,
  a 600-step CUDA probe showed correct state beating zero, swapped, and
  time-shifted state in one run. This is encouraging but still not a closed
  gate because small perturbations were effectively tied and cross-seed
  reproducibility is not measured.
- `scripts/check_state_probe.py` makes this explicit: the current suffix probe
  fails the default gate because `intervention_perturbed_delta` is slightly
  negative.
- A second seed reproduced the zero/swapped/time-shifted result, but increasing
  perturbation strength to `0.2` made the perturbation metric worse. Random
  perturbation should be treated as a separate state-manifold robustness problem
  until it is averaged across trials or replaced with a more structured
  intervention.
- A two-seed matrix with randomized segment lengths at 600 steps produced strong
  zero and swapped deltas, and inverted-state damage was very large. One seed
  still missed the shifted-state threshold slightly, while perturbation and
  attenuation remained near zero even when averaged across trials.
- Query recall gives a cleaner core-causality signal than repeat. Seeds `700`
  and `701` passed the core gate at 400 steps, but robustness still did not
  consistently pass because perturbation can remain neutral or slightly helpful.

Impact:

- The local trainer can be used for controlled experiments.
- Remote scale-up should still wait; the current failure is conceptual/training
  objective related, not an infrastructure blocker.

Resolution direction:

- Add a clearer controlled state-carry task or mask the impossible prefix loss
  so the memory-dependent portion is directly optimized.
- Keep core causality and robustness as separate gates.
- Use query recall as the ordered temporal-state gate and run a wider seed
  matrix before closing Phase 3.
- Only promote Phase 3 to complete once correct state reliably beats zero,
  swapped, and time-shifted state across seeds; treat perturbation/attenuation
  as a robustness gate unless the architecture is changed to make them
  semantically meaningful destructive interventions.

### Experiment checkpoints can be overwritten by ad-hoc run config mistakes

Status: open.

An ad-hoc query-recall re-evaluation intended to resume seed `703` omitted the
resume field in the generated temporary config and overwrote that ignored run
directory's `latest.pt` and `best.pt` with a short fresh run.

Impact:

- The repository is not affected because `runs/` is ignored.
- The seed `703` run directory should not be used for checkpoint continuation
  without rerunning it.
- Manual temp-config mutation is too error-prone for checkpointed experiments.

Resolution direction:

- Use `scripts/run_state_probe_matrix.py --resume-existing` for resumed matrix
  runs.
- Keep eval/save/log interval overrides in the runner instead of hand-editing
  temporary configs.
- Add stricter trainer-side resume guards before longer experiments.


## Resolved

### Stage A training harness missing checkpoint and intervention infrastructure

Resolved on 2026-06-17.

The harness now includes config/runtime/logging/checkpoint modules, run config
snapshots, async metrics, CSV export, latest/best checkpoints, resume support,
failed-run checkpoints, segmented state carry, and intervention evaluation.
