# Issues

This file tracks only repeated problems, active blockers, or major risks that
materially affect execution. Fast, one-off fixes should not be recorded here.

## Active

### Stage A robustness gate is not passing yet

Status: open.

The Stage A training harness now has real-data loading, CUDA execution,
segmented carry, checkpoint/resume, metrics, and intervention evaluation. The
training pipeline itself is usable, and the ordered query-recall task now passes
the core state-causality gate across four CUDA seeds. Robustness remains open:
random perturbation and attenuation are still near-neutral for some seeds.

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
- A clean 4-seed query-recall matrix with 6-token shifted-state intervention
  passed the core gate for seeds `700`, `701`, `702`, and `703`. Perturbation
  robustness is still inconsistent.
- A 4-seed diffusion ablation on query recall showed positive perturbation
  deltas for fixed diffusion and state-dependent diffusion across all seeds.
  ODE also had positive perturbation deltas, but they were much smaller.
  Attenuation remained inconsistent across modes.

Impact:

- The local trainer can be used for controlled experiments.
- Remote scale-up should still wait for Phase 4 ablations, but the blocker is
  no longer core state causality.

Resolution direction:

- Keep robustness separate from core causality during Phase 4.
- Run the query-recall diffusion matrix across at least four seeds for ODE,
  fixed diffusion, and state-dependent diffusion using
  `scripts/run_state_probe_matrix.py --set diffusion_mode=...`.
- Compare perturbation, attenuation, and inverted-state deltas by diffusion
  mode. If fixed or state-dependent diffusion consistently improves
  perturbation robustness over ODE, promote that mode to the next ablation.
- Carry fixed and state-dependent diffusion into solver-step ablations as
  robustness candidates.
- Do not use attenuation as a blocking robustness gate until a structured
  attenuation probe is defined; current attenuation deltas are too close to
  zero and inconsistent.


## Resolved

### Stage A training harness missing checkpoint and intervention infrastructure

Resolved on 2026-06-17.

The harness now includes config/runtime/logging/checkpoint modules, run config
snapshots, async metrics, CSV export, latest/best checkpoints, resume support,
failed-run checkpoints, segmented state carry, and intervention evaluation.

### Experiment checkpoints can be overwritten by ad-hoc run config mistakes

Resolved on 2026-06-18.

An ad-hoc query-recall re-evaluation intended to resume seed `703` omitted the
resume field in a generated temporary config and overwrote that ignored run
directory's checkpoints. The trainer now refuses to start a fresh run in a run
directory containing `latest.pt`, `metrics.jsonl`, or `model_latest.pt` unless
`resume` is set or `allow_run_overwrite=true` is explicit.
