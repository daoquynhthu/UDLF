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

Impact:

- The local trainer can be used for controlled experiments.
- Remote scale-up should still wait; the current failure is conceptual/training
  objective related, not an infrastructure blocker.

Resolution direction:

- Add a clearer controlled state-carry task or mask the impossible prefix loss
  so the memory-dependent portion is directly optimized.
- Add randomized truncation boundaries and stricter intervention pass/fail
  metrics.
- Only promote Phase 3 to complete once correct state reliably beats
  zero/swapped/shifted/perturbed state across seeds.


## Resolved

### Stage A training harness missing checkpoint and intervention infrastructure

Resolved on 2026-06-17.

The harness now includes config/runtime/logging/checkpoint modules, run config
snapshots, async metrics, CSV export, latest/best checkpoints, resume support,
failed-run checkpoints, segmented state carry, and intervention evaluation.
