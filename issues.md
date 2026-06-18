# Issues

This file tracks only repeated problems, active blockers, or major risks that
materially affect execution. Fast, one-off fixes should not be recorded here.

Active issues are not passive notes. Every active issue must have:

- the concrete blocker or risk;
- why it blocks or constrains the plan;
- a resolution plan with ordered next actions;
- exit criteria that make the issue closable;
- a current owner context, which is normally this workspace unless stated
  otherwise.

If an issue does not need planned resolution, it does not belong in this file.

## Active

### Stage A robustness gate is not passing yet

Status: open.

Blocker classification:

- This is not blocking fixed K=4 remote smoke, because core state causality now
  passes on both synthetic query recall and real-token query recall.
- It is blocking any claim that Stage A has solved robust latent-state use under
  off-manifold perturbations.
- It is blocking promotion from smoke/diagnostic runs to longer scale-up runs
  that are meant to validate stochastic robustness rather than infrastructure.

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
- Remote fixed K=4 smoke is allowed only as an infrastructure and core-gate
  validation run. It must not be described as resolving robustness.

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
- Compare K=1/2/4/8 across more seeds before deciding whether deeper solver
  integration is worth the runtime cost. The first state-dependent seed shows
  K=1 already works, while K=4/K=8 are stronger but slower.
- State-dependent K=4 improved robustness deltas over K=1 across seeds
  `721-723`, but roughly halves throughput. The next decision is whether fixed
  diffusion shows the same pattern before picking a default.
- Fixed diffusion shows the same K=4-over-K=1 pattern across seeds `721-723`:
  K=4 is slower but gives stronger intervention margins and more consistent
  attenuation deltas.
- Phase 4 summary tables make fixed K=4 the current pragmatic default
  candidate. This resolves the immediate default-selection ambiguity, but not
  the broader robustness-gate design problem.
- The short real-token fixed K=4 run verified stability but not causal state use
  on language data: zero-state was worse, while swapped/shifted/perturbed states
  were near-neutral or slightly helpful.
- The real-token query-recall diagnostic now passes the core gate across seeds
  `900`, `901`, and `902`, so fixed K=4 is acceptable for remote smoke. The
  remaining blocker is robustness interpretation, especially attenuation.
- Do not use attenuation as a blocking robustness gate until a structured
  attenuation probe is defined; current attenuation deltas are too close to
  zero and inconsistent.
- Keep robustness open, but do not block remote smoke on it; remote smoke should
  validate infrastructure and fixed K=4 stability, not claim robustness.

Resolution plan:

1. Freeze fixed K=4 as the current smoke/default candidate and stop changing
   the default unless a later ablation clearly beats it on both core gate and
   robustness metrics.
2. Run the fixed K=4 remote real-token query-recall smoke once private remote
   config exists. Treat the result as infrastructure validation plus core-gate
   confirmation only.
3. Define a structured robustness suite before any longer remote scale-up:
   attenuation must be replaced or supplemented with interventions that remain
   on or near the learned state manifold.
4. Add a small local robustness experiment that compares current random
   perturbation, attenuation, inverted state, and at least one structured
   perturbation on the same fixed K=4 checkpoints.
5. Only after the structured suite is implemented, decide whether the blocker
   is a model weakness, an evaluation artifact, or a scale-dependent effect.

Immediate next actions:

- Do not expand this issue with more raw observations unless they change the
  decision.
- Add a planned Phase 4 subtask for structured robustness diagnostics.
- Keep Phase 5 remote smoke scoped to fixed K=4 real-token query recall.

Exit criteria:

- A documented robustness gate exists with thresholds, target interventions,
  and rationale.
- Fixed K=4 or a replacement default passes that gate on at least three seeds,
  or the project explicitly downgrades robustness from a Stage A acceptance
  requirement.
- `plan.md` no longer depends on unresolved robustness before any long-running
  scale-up that claims stochastic latent robustness.


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
