# UDLF Project Plan

This plan is the authoritative task boundary for the UDLF workspace. It tracks
future work at the project level; short action summaries belong in
`progress.md`, and recurring or blocking problems belong in `issues.md`.

## Scope

The current workspace goal is to turn the mathematical UDLF-LM design into a
research-grade implementation and experiment harness that can be run locally and
on an isolated UDLF remote 4090 workflow.

In scope:

- Preserve the core UDLF design as the source of architectural truth.
- Build a minimal, testable implementation of the stage A prior-path model.
- Keep remote 4090 operations reusable, auditable, and private-config driven.
- Add experiments that test causal state use, stochastic dynamics, and long
  horizon behavior rather than only aggregate perplexity.
- Maintain clear verification gates before scale-up.

Out of scope until explicitly promoted:

- Stage B controlled posterior training.
- Filtering-posterior state propagation.
- Competitive sparse dynamics.
- Large-scale language-model training before the small synthetic and diagnostic
  gates pass.
- Claiming continuous SDE behavior before step-size consistency tests support
  it.

## Operating Rules

- Keep `plan.md`, `progress.md`, and `issues.md` separate.
- Update `plan.md` when task boundaries, sequencing, or acceptance criteria
  change.
- Update `progress.md` after concrete actions with short factual summaries.
- Update `issues.md` only for repeated problems, current blockers, or risks that
  materially affect execution.
- Do not commit private remote configuration, host names, credentials, run
  outputs, checkpoints, or datasets.
- Keep the UDLF workspace fully isolated from unrelated remote workflows.

## Phase 0: Workspace Foundation

Status: complete.

Tasks:

- Establish stable project directories: `doc`, `scripts`, `configs`, `src`,
  `tests`, `experiments`, `runs`, and `artifacts`.
- Move the UDLF mathematical design into `doc`.
- Add remote 4090 operation documentation and reusable scripts for UDLF.
- Add repository hygiene: `.gitignore`, private config exclusion, and a project
  README.
- Initialize git for the UDLF workspace.

Acceptance criteria:

- The workspace has a clear top-level structure.
- `doc/` contains all current project documents.
- `scripts/` contains the imported remote operation helpers.
- `configs/workspace.example.json` exists and does not contain private values.
- `configs/workspace.local.json` is ignored.
- `git status` works from `E:\UDLF`.

## Phase 1: Remote Workflow Adaptation

Status: complete.

Purpose:

Make the remote 4090 workflow safe for UDLF without carrying assumptions from
other projects into future runs.

Tasks:

- Audit scripts for unrelated project names, paths, model identifiers, and
  environment variables.
- Adapt `sync_to_remote.ps1` so it packages this workspace correctly and does
  not rely on unrelated project exclusions or archive names.
- Adapt `remote.ps1`, `ssh_cmd.ps1`, `watch_remote.ps1`, and
  `inspect_remote_training.ps1` for UDLF run naming and log locations.
- Keep only UDLF training templates under `configs/training_templates`.
- Add a minimal UDLF remote smoke template only after a local training entrypoint
  exists.

Completed:

- Rewrote the main remote operations document for UDLF.
- Updated sync packaging to include `doc/` and exclude generated outputs.
- Reworked detached launch to target a configurable Python module instead of
  project-specific training wrappers.
- Reworked remote inspection around generic GPU/process/log/metric checks.
- Added a minimal UDLF smoke training entrypoint for validating launch/logging
  infrastructure.
- Removed unrelated legacy remote scripts, templates, docs, and compatibility
  environment-variable fallbacks from the active workspace.

Remaining:

- Replace the smoke-only `udlf.training.train` with, or extend it into, the real
  stage A training entrypoint once model code exists.
- Test SSH/status/sync against the remote 4090 once private config exists.

Acceptance criteria:

- Remote status, command execution, code sync, and log inspection work without
  requiring unrelated repository paths.
- The scripts still read private values from `configs/workspace.local.json` or
  environment overrides.
- No script defaults to another project's remote repo path.
- No unrelated environment-variable namespace is accepted as fallback.

Verification:

- Static scan for unrelated project names and old remote paths.
- Dry-run or no-op remote command through `ssh_cmd.ps1` when credentials are
  configured.
- Code sync to a disposable remote path before any training launch.

## Phase 2: Minimal UDLF Implementation Skeleton

Status: complete.

Purpose:

Create the smallest implementation surface that can express the stage A
single-sample prior-path model from the design document.

Tasks:

- Add package modules under `src/udlf`.
- Define configuration dataclasses for model, solver, training, and experiment
  settings.
- Implement token embedding, latent field state, observation injection,
  nonlinear latent interaction core, drift, diffusion, Euler-Maruyama stepping,
  and readout.
- Implement deterministic ODE mode as a separate baseline, not by driving
  training diffusion to zero.
- Implement single-path stage A loss with `S=1`.
- Keep multi-path `S>1` and posterior training out of the first implementation
  unless the state continuation rule is explicitly specified.

Completed:

- Added UDLF model configuration dataclass.
- Implemented RMSNorm, observation injection, latent interaction core,
  state-dependent drift/diffusion, Euler-Maruyama stepping, readout, and
  single-path stage A loss.
- Added deterministic ODE and stochastic state-dependent diffusion modes.
- Added CPU tests for shapes, deterministic reproducibility, future-token
  independence for earlier prior logits, and state-dependent diffusion forward.
- Added `scripts/smoke_stage_a_forward.py` for quick local/remote forward checks.
- Added focused tests for fixed diffusion mode and explicit state carry
  equivalence.
- Decided to keep the smoke training entrypoint as infrastructure validation;
  Phase 3 will extend it or add a real training path without changing that
  meaning.

Acceptance criteria:

- A small batch forward pass runs on CPU.
- Shapes and dtype behavior are covered by tests.
- Random seed control gives reproducible deterministic-mode output.
- No future token or posterior-only information can enter the prior path.

Verification:

- Unit tests for module shapes and causal state flow.
- A smoke forward script over synthetic token IDs.
- Static check that stage A code has no dependency on target token except in the
  loss calculation.

## Phase 3: Stage A Training Harness

Status: in progress.

Purpose:

Train the prior-path model on controlled tasks before attempting expensive
remote runs.

Tasks:

- Add synthetic datasets for delayed ambiguity, state correction, semantic
  compression, and simple exact-memory probes.
- Add training loop with truncated state carry, random truncation boundaries,
  gradient clipping, metric logging, and checkpointing.
- Log state norms, drift norms, diffusion ranges, jump magnitudes, gradient
  norms, and output loss.
- Add ODE, fixed diffusion, and state-dependent diffusion switches.
- Add state intervention evaluation: correct state, zero state, swapped state,
  time-shifted state, and perturbed state.

Completed:

- Added a deterministic repeating-pattern synthetic dataset.
- Added a minimal CPU Stage A optimizer loop with metrics and `train.log`.
- Added a local Stage A smoke config.
- Added test coverage that verifies the training loop writes metrics.
- Added disk-backed token dataset sampling for locally saved Hugging Face
  datasets.
- Refactored the Stage A trainer into explicit config, runtime, logging, and
  checkpoint modules.
- Added run-local `config.json`, async JSONL metrics, CSV conversion, atomic
  latest/best checkpoints, resume support, failed-run checkpoint capture,
  gradient accumulation, scheduler support, segmented state carry, and state
  intervention evaluation.
- Verified the refactored trainer on local CUDA with the saved FineWeb subset
  for 30 steps.
- Added suffix-only loss masking for the repeating-pattern probe so the
  impossible random prefix does not dominate the memory-dependent objective.
- Corrected the shifted-state intervention to use a shorter-context state
  rather than a latent-slot roll.
- Added randomized segment length support for Stage A state-carry training.
- Replaced single-sample perturbation evaluation with multi-trial perturbation
  averages plus attenuated and inverted-state probes.
- Added a multi-seed suffix-probe matrix runner with resume support.
- Added a query-recall synthetic task that requires binding prior positions to
  values and answering query tokens from persistent state.

Remaining:

- Broaden the query-recall matrix beyond the first two CUDA seeds and define the
  minimum seed count for Phase 3 completion.
- Strengthen the repeat-task core state-causality gate if it remains part of
  the acceptance suite: two additional CUDA matrix seeds at 600 steps passed
  zero and swapped-state checks, but one shifted-state delta remained just below
  the current threshold.
- Define a robustness gate separately from core causality. Inverted state is
  strongly destructive, while random perturbation and attenuation remain near
  zero and should not be treated as solved.

Acceptance criteria:

- Local training runs complete on at least one small synthetic task.
- Correct latent state beats zero/swapped/misaligned state on a task that
  actually requires memory.
- At least two seeds pass the core state-causality gate on an ordered recall
  task.
- Diffusion parameters do not immediately saturate at bounds without being
  reported.
- Failure modes are visible in logs rather than silent.

Verification:

- Local smoke training.
- Synthetic task metric report.
- Reproducibility check across at least two seeds for the smallest task using
  `scripts/run_state_probe_matrix.py`.

## Phase 4: Numerical and Architectural Ablations

Status: pending.

Purpose:

Separate useful model behavior from numerical artifacts and expensive redundant
structure.

Tasks:

- Compare `K in {2, 4, 8, 16}` for output distribution stability and state
  trajectory behavior.
- Compare ODE, fixed diffusion, and state-dependent diffusion.
- Compare full latent interaction core against simpler linear or MLP-only
  baselines.
- Compare readout variants: mean pooling, single query, and conditional
  multi-query.
- Measure whether increased `K` changes conclusions or only deepens the
  recurrence.

Acceptance criteria:

- Claims about continuous dynamics are withheld unless `K` consistency is
  credible.
- Any stochastic-dynamics benefit survives at least one deterministic ensemble
  or fixed-diffusion comparison.
- Expensive components have measurable behavioral contribution.

Verification:

- Ablation tables for task metrics, runtime, and stability metrics.
- Output KL or equivalent distribution comparison across `K`.

## Phase 5: Remote 4090 Smoke and Scale-Up

Status: pending.

Purpose:

Use the remote 4090 only after local implementation and small-task diagnostics
are strong enough to justify the cost.

Tasks:

- Configure `configs/workspace.local.json` locally, outside git.
- Sync code to a disposable remote UDLF path.
- Run remote compile/import checks.
- Run a short remote smoke training.
- Confirm logs, metrics, checkpoints, and STOP-file shutdown.
- Only then define longer remote runs.

Acceptance criteria:

- Remote launch is detached and does not leave visible windows.
- STOP-file shutdown works.
- Remote logs and metrics can be inspected locally.
- Checkpoints are written sparsely and in expected locations.
- No long run starts before smoke run verification.

Verification:

- Remote status preflight.
- Remote compile/import check.
- Short remote smoke run with clean shutdown.
- Local pull or inspection of smoke artifacts.

## Phase 6: Stage B Decision Gate

Status: pending.

Purpose:

Decide whether controlled posterior training is justified.

Entry conditions:

- Stage A prior-path training is stable.
- Pure-prior rollout has measurable task value.
- State-dependent diffusion shows a reproducible reason to exist, or the
  project explicitly pivots to deterministic/ODE variants.
- State intervention tests prove the persistent field is causally used.

Tasks:

- Design the controlled posterior implementation without allowing posterior
  state propagation.
- Define posterior dropout schedule and gradient-weight normalization.
- Define posterior-prior gap diagnostics before implementation.

Acceptance criteria:

- Stage B is not used to rescue a prior model that does not work.
- The double-track state protocol is testable.
- Any filtering-posterior experiment is labeled as a different training regime.

## Current Priority

Push Phase 3 past the state-causality gate locally: improve the controlled task,
training target, or model path until correct latent state reliably beats
zero/swapped/misaligned state before starting remote scale-up.
