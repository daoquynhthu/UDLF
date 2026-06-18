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
- Treat every active `issues.md` entry as a planned blocker: it must include
  impact, ordered resolution actions, immediate next actions, and exit
  criteria. If an entry does not need planned resolution, remove or resolve it.
- Do not commit private remote configuration, host names, credentials, run
  outputs, checkpoints, or datasets.
- Keep the UDLF workspace fully isolated from unrelated remote workflows.
- Keep console output compact during experiments. Full metrics and evaluator
  JSON must go to files under ignored artifacts or run directories; terminal
  output should be progress summaries only unless full JSON is explicitly
  requested for debugging.

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

Status: complete.

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
- Added direct dynamics instrumentation for drift RMS, diffusion sigma range,
  sigma RMS, and jump RMS, so diffusion saturation and jump scale are visible
  in metrics rather than inferred from loss.
- Added a query-recall synthetic task that requires binding prior positions to
  values and answering query tokens from persistent state.
- Ran a clean 4-seed query-recall CUDA matrix with 6-token shifted-state
  intervention. Seeds `700`, `701`, `702`, and `703` all passed the core
  state-causality gate at 400 steps.

Remaining:

- Strengthen the repeat-task core state-causality gate if it remains part of
  the acceptance suite: two additional CUDA matrix seeds at 600 steps passed
  zero and swapped-state checks, but one shifted-state delta remained just below
  the current threshold.
- Define a robustness gate separately from core causality. Inverted state is
  strongly destructive, while random perturbation and attenuation remain near
  zero and should not be treated as solved.
- Harden experiment artifact handling so ad-hoc eval or resume attempts cannot
  accidentally overwrite useful checkpoints without an explicit resume path.

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
- 4-seed query-recall core gate run on local CUDA.

## Phase 4: Numerical and Architectural Ablations

Status: in progress.

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

Completed:

- Started single-seed query-recall diffusion ablation for ODE, fixed diffusion,
  and state-dependent diffusion at 400 steps.
- Added generic `--set key=value` config overrides to the matrix runner for
  controlled ablations.
- Extended the query-recall diffusion ablation to four matched seeds for ODE,
  fixed diffusion, and state-dependent diffusion.
- Ran a first state-dependent solver-step scan for `K in {1, 2, 4, 8}` on one
  query-recall seed.
- Extended the state-dependent K=1 vs K=4 comparison to three additional
  query-recall seeds.
- Extended the fixed-diffusion K=1 vs K=4 comparison to three query-recall
  seeds.
- Ran a short real saved-token fixed K=4 confirmation on the local FineWeb
  subset. It verified training stability, checkpointing, and metrics on real
  token data, but did not establish language-data state causality.
- Added a tracked Phase 4 ablation summary table generated from local run
  metrics.
- Added a real-token query-recall data task and ran fixed K=4 confirmations
  that passed the core gate after 600 steps on three seeds.
- Added a structured mixed-state intervention metric and a read-only checkpoint
  evaluator so existing fixed K=4 checkpoints can be tested without resuming or
  overwriting training runs.
- Fixed intervention evaluation to use common random numbers for suffix
  rollouts and to report paired mean, standard error, and confidence intervals.
  Small robustness deltas from pre-CRN runs are historical only.
- Re-evaluated the matched ODE, fixed-diffusion, and state-dependent diffusion
  query-recall checkpoints for seeds `710-713` with CRN paired intervention
  metrics. State-dependent diffusion is now the stronger robustness candidate
  in this matched set; fixed diffusion remains the simpler smoke/default
  candidate.

Next:

- Treat fixed K=4 as the default candidate for local and eventual remote smoke
  runs, but do not treat it as the best robustness candidate.
- Continue local 5060 experiments before remote scale-up: run a medium
  state-dependent K=4 CRN confirmation and compare it against fixed K=4 on
  real-token query recall.
- The first three real-token state-dependent K=4 seeds are mixed: seeds
  `903-905` pass the core gate, but seed `904` fails the batch-mix robustness
  read and seed `905` has perturbation too close to zero. Next, add more
  real-token seeds or inspect why some runs are strongly sensitive to core
  interventions but not to structured/random robustness probes.
- Prepare a remote smoke config for fixed K=4 real-token query recall, using
  private data path configuration only.
- Run remote sync and fixed K=4 real-token query-recall smoke once private
  remote dataset path configuration is available.
- Use `scripts/prepare_remote_smoke_config.py` to generate the private remote
  smoke config; never commit the generated `.local.json`.
- Run `scripts/evaluate_state_interventions.py` on the existing fixed K=4
  real-token query-recall checkpoints and check
  `scripts/check_state_probe.py --profile structured` on new metrics-bearing
  runs. This is the first concrete structured robustness diagnostic; it does
  not replace the broader robustness gate yet.
- The first structured sweep over `alpha in {0.05, 0.1, 0.2, 0.4}` is complete
  for real-token fixed K=4 seeds `900`, `901`, and `902`. Batch-mix mostly
  behaves as expected, but temporal-mix fails on seed `901`; next, define
  per-probe thresholds and inspect whether temporal-mix is a valid destructive
  intervention for query recall. All future small-delta claims must be based on
  CRN paired statistics.
- Keep plain next-token language intervention metrics out of the core
  state-causality gate unless a meaningful target is defined.

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

Use the local RTX 5060 for medium-scale validation before remote scale-up. The
immediate priority is to extend state-dependent K=4 real-token query-recall
confirmation beyond seed `903`, with CRN paired intervention metrics and
dynamics instrumentation. Remote 4090 work should wait until the local framework
and robustness gates are cleaner.
