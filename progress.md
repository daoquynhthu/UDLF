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
