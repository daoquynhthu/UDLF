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
