# Progress

This file records concise action summaries only. Detailed planning belongs in
`plan.md`; recurring or blocking issues belong in `issues.md`.

## 2026-06-17

- Created UDLF workspace structure under `E:\UDLF`.
- Moved the UDLF mathematical design into `doc/`.
- Added remote 4090 operation documentation and scripts for the UDLF workspace.
- Added `README.md`, `.gitignore`, and a minimal `src/udlf` package placeholder.
- Created workflow tracking files: `plan.md`, `progress.md`, and `issues.md`.
- Initialized the workspace as a git repository on branch `main`.
- Started Phase 1 remote workflow adaptation.
- Removed unrelated legacy remote scripts, templates, docs, and compatibility
  fallbacks so the UDLF workspace is isolated.
- Rewrote the active remote 4090 operation document for UDLF.
- Updated core remote scripts for UDLF-first config names, generic sync,
  generic run inspection, STOP-file handling, and module-based detached launch.
- Added a minimal `udlf.training.train` smoke entrypoint and
  `configs/training_templates/udlf_remote_smoke.json` for validating workflow
  plumbing before real model training exists.
- Verified the smoke entrypoint with `pytest tests\test_smoke_training.py -q`
  and a direct `python -m udlf.training.train` local run.
- Started Phase 2 minimal UDLF implementation.
- Added `UDLFStageAModel` with observation injection, latent interaction,
  prior drift/diffusion, Euler-Maruyama stepping, readout, and single-path
  stage A loss.
- Added tests for forward shapes, deterministic ODE reproducibility, causal
  prefix independence, and state-dependent diffusion forward.
- Added `scripts/smoke_stage_a_forward.py`.
- Verified with `pytest tests\test_stage_a_model.py tests\test_smoke_training.py -q`
  and smoke forward runs for `ode` and `state_dependent` diffusion modes.
- Completed Phase 2 by adding fixed-diffusion reproducibility and explicit
  state-carry equivalence tests.
- Committed Phase 0 through Phase 2 as
  `5f526c4 Initialize UDLF workspace and stage A skeleton`.
- Started Phase 3 by adding a repeating-pattern synthetic dataset and minimal
  CPU Stage A optimizer loop.
- Added `configs/training_templates/udlf_stage_a_local_smoke.json` and a
  training-loop test.
- Verified with `pytest tests -q` and a local
  `python -m udlf.training.train --config configs\training_templates\udlf_stage_a_local_smoke.json`
  run.
- Confirmed the local machine has CUDA-enabled PyTorch on the RTX 5060 and ran
  a 20-step real-data FineWeb probe from a private temp config.
- Added disk-backed dataset loading for saved token datasets and Stage A
  template configs for local smoke, GPU probe, and local saved-data training.
- Refactored Stage A training into config/runtime/logging/checkpoint modules
  with run config snapshots, async metrics, CSV export, atomic latest/best
  checkpoints, resume, failed-run checkpoints, gradient accumulation, scheduler
  support, segmented carry, and intervention metrics.
- Verified the refactored training pipeline with `pytest tests -q`, static
  compile checks, an isolation scan for unrelated project names, a 30-step CUDA
  FineWeb run, and a 200-step CUDA synthetic state probe.
- The 30-step FineWeb probe completed with eval loss `10.4699`, latest/best
  checkpoints, `metrics.jsonl`, `metrics.csv`, and `config.json`.
- The 200-step synthetic state probe completed, but did not pass the state
  causality gate: zero-state was worse, while shifted state was unchanged and
  swapped state was only marginally worse.
- Added suffix-only loss masking for the repeating-pattern probe and fixed the
  shifted-state intervention to use a shorter-context state instead of rolling
  latent slots.
- Resumed the suffix-only CUDA probe from step 500 to 600. At step 600, correct
  state beat zero (`+0.6321` loss delta), swapped (`+0.0587`), and
  time-shifted (`+0.0408`), while small perturbation remained tied
  (`-0.0021`).
- Added `scripts/check_state_probe.py` to turn the latest eval intervention
  metrics into a pass/fail gate. The current suffix probe fails only the
  perturbation threshold under the default criteria.
- Ran a second suffix-only CUDA probe with seed `556`; correct state again beat
  zero, swapped, and time-shifted state, but random perturbation still improved
  loss slightly.
- Made intervention perturbation strength configurable. Re-evaluating seed
  `556` with perturb std `0.2` made the perturbation failure larger, so random
  perturbation is not yet a reliable destructive intervention for this model.
- Added randomized segment lengths for segmented state-carry training, with the
  suffix probe template using a 6-12 token range.
- Upgraded intervention evaluation to average multiple perturbation trials and
  log attenuated and inverted-state probes.
- Added `scripts/run_state_probe_matrix.py` with resume support and
  core/robustness check profiles.
- Ran a CUDA matrix for seeds `557` and `558`, first to 300 steps and then
  resumed to 600. At 600, both seeds strongly failed zero state and passed
  swapped state; seed `557` had shifted delta `+0.0171`, just under the current
  `+0.02` core threshold. Inverted state was strongly destructive in both runs,
  while perturbation and attenuation stayed near zero.
- Added `QueryRecallDataset`, where prefix values must be recalled through
  later query tokens, plus a `udlf_stage_a_query_recall_probe.json` template.
- Updated intervention evaluation to respect dataset-specific intervention
  splits and loss masks, so query-recall metrics only score answer targets.
- Ran query-recall CUDA probes for seeds `700` and `701` to 400 steps. Both
  passed the core gate; seed `700` also passed robustness, while seed `701`
  still failed robustness because perturbation delta was slightly negative.
- Ran query-recall seeds `702` and `703`. Seed `702` passed the core gate at
  400 steps. Seed `703` had strong zero/swapped signals but weak shifted delta
  at 400 and 800 steps under the one-token shift metric.
- Added configurable `intervention_shift_tokens`; query recall now uses a
  stronger 6-token shift because one-token truncation only removes one memory
  slot and is too weak for random-position queries.
- Added matrix runner overrides for eval/save/log intervals so checkpointed
  runs can be resumed and re-evaluated without hand-written temporary config
  mutation.
- An ad-hoc PowerShell resume/eval attempt omitted the intended resume field and
  overwrote the ignored seed `703` query-recall checkpoint with a short new run.
  The repository is unaffected, but that run directory is no longer reliable
  for checkpoint continuation.
- Ran a clean query-recall CUDA matrix with run prefix
  `runs/udlf_query_recall_shift6_matrix` for seeds `700`, `701`, `702`, and
  `703` at 400 steps. All four seeds passed `check_state_probe.py --profile
  core` with 6-token shifted-state intervention.
- Phase 3 is complete for the core state-causality gate. Robustness remains a
  separate unresolved gate because perturbation deltas are still near zero or
  slightly negative for some seeds.
- Started Phase 4 diffusion ablation on query recall with seed `710`. ODE,
  fixed diffusion, and state-dependent diffusion all passed the core gate at
  400 steps. Fixed and state-dependent diffusion produced stronger perturbation
  deltas than ODE in this single-seed run.
- Added `--set key=value` overrides to `scripts/run_state_probe_matrix.py` for
  controlled ablation runs without one-off templates.
- Added trainer-side run overwrite protection. Fresh training now refuses to
  start in a run directory with existing artifacts unless `resume` is set or
  `allow_run_overwrite=true` is explicit.
- Completed a 4-seed query-recall diffusion ablation for seeds `710`, `711`,
  `712`, and `713` at 400 steps across ODE, fixed diffusion, and
  state-dependent diffusion.
- All three diffusion modes passed the core gate for all four seeds.
- Fixed diffusion and state-dependent diffusion produced consistently positive
  perturbation deltas across all four seeds; ODE perturbation deltas were also
  positive but much smaller. Attenuation remained inconsistent and should not be
  used alone as a robustness gate.
- Ran a first solver-step scan for state-dependent diffusion on query recall
  with seed `720`: `K in {1, 2, 4, 8}` all passed the core gate. Higher `K`
  improved eval loss and some intervention deltas, but throughput dropped
  sharply; K=1 was already effective and much faster.
- Extended state-dependent K ablation for K=1 and K=4 to seeds `721`, `722`,
  and `723`. Both passed the core gate across all seeds. K=4 produced stronger
  shifted-state and robustness deltas, including consistently positive
  attenuation, at roughly half the throughput of K=1.
- Ran fixed-diffusion K=1 vs K=4 query-recall ablations for seeds `721`, `722`,
  and `723`. Both passed the core gate across all seeds. Fixed K=4 gave much
  stronger zero/swapped/shifted deltas and consistently positive attenuation
  deltas, while fixed K=1 was substantially faster but weaker.
- Ran a real saved-token confirmation on the local saved FineWeb subset using
  fixed diffusion K=4 for 100 CUDA steps. Training was stable: loss fell from
  `18.83` at step 10 to `9.70` at step 100, eval loss was `9.634`, peak CUDA
  allocation was about `2095 MB`, and checkpoints/metrics were written.
- The real-token intervention result should not be overclaimed: zero-state was
  worse by `+0.558` loss, but swapped/shifted/perturbed states were near-neutral
  or slightly better. The ordered query-recall gate remains the evidence for
  state causality; real-token runs currently verify training stability only.
- Added `scripts/summarize_phase4_ablation.py` and generated tracked Phase 4
  summaries under `doc/phase4_ablation_summary.md`,
  `doc/phase4_ablation_summary.csv`, and `doc/phase4_ablation_runs.csv`.
- The summary covers 28 Phase 4 query-recall runs across 8 mode/K groups.
  Fixed K=4 is the current pragmatic default candidate: it is simpler than
  state-dependent diffusion, has strong synthetic margins, and has already
  been confirmed stable on real saved-token training.
- Added a `real_query_recall` data task that builds query-recall sequences from
  saved token rows, giving real-token experiments a known state target instead
  of relying on plain next-token intervention signals.
- Ran fixed K=4 real-token query recall with seed `900`. At 200 steps, training
  was stable but core intervention did not pass. After resuming to 600 steps,
  eval loss was `7.736` and the core gate passed: zero `+4.726`, swapped
  `+0.139`, shifted `+0.068`, inverted `+0.673`. Robustness still failed:
  perturbation and attenuation deltas were slightly negative.
- Extended fixed K=4 real-token query recall to seeds `901` and `902` at 600
  steps. Both passed the core gate. Across seeds `900`, `901`, and `902`, the
  real-token core gate is now 3/3 pass; robustness still fails on attenuation
  for at least one seed.
- Added `configs/training_templates/udlf_remote_real_token_query_recall_smoke.example.json`
  as the remote smoke template for fixed K=4 real-token query recall. It keeps
  the remote saved-token dataset path as a private placeholder.
- Added `scripts/prepare_remote_smoke_config.py` to materialize a private
  `.local.json` remote smoke config from the tracked template and refuse
  unresolved placeholders.
- Updated `.gitignore` so generated config files under `configs/**` ending in
  `.local.json` stay out of git.
- Reworked `issues.md` so active issues are planned blockers with resolution
  actions, immediate next steps, and explicit exit criteria instead of passive
  records.
- Added `intervention_mixed_delta`, a structured mixed-state intervention that
  lightly mixes each state with another real state from the same batch.
- Added `scripts/evaluate_state_interventions.py` for read-only intervention
  evaluation of existing checkpoints without resuming or overwriting a run.
- Extended state-probe checking, matrix summaries, and Phase 4 summaries to
  carry the structured mixed-state metric.
- Re-evaluated the existing fixed K=4 real-token query-recall checkpoints for
  seeds `900`, `901`, and `902` with the mixed-state probe. The structured
  profile passed for all three with mixed deltas `+0.0126`, `+0.0070`, and
  `+0.0060`. Random perturbation and attenuation remain inconsistent, so the
  robustness blocker is not closed.
- Ran the mixed-alpha sweep for `alpha in {0.05, 0.1, 0.2, 0.4}` on the same
  three checkpoints. Batch-mix deltas grew with alpha, but seed `901` was
  slightly negative at alpha `0.05`.
- Added `doc/structured_robustness_probe.md` with the structured probe
  definition, commands, per-seed results, aggregate table, and current read.
- Added temporal-mix as a second structured intervention family. It exposed a
  real weakness: seed `901` had negative temporal-mix deltas across all tested
  alphas, so structured robustness is not ready to close.
- Split structured checking into `structured-batch` and `structured-temporal`
  profiles while keeping `structured` as the strict combined profile.
