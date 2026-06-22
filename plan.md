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

Status: in progress.

Purpose:

Use the remote 4090 only after local implementation and small-task diagnostics
are strong enough to justify the cost.

Tasks:

- Configure `configs/workspace.local.json` locally, outside git.
- Sync code to a disposable remote UDLF path.
- Install an isolated HTTPS workspace service under `L:\UDLF_REMOTE`, reusing
  only the NAIME `.venv312` Python environment.
- Access the HTTPS service through an SSH local tunnel; keep the service bound
  to remote loopback instead of exposing it on the LAN.
- Run remote compile/import checks.
- Run a short remote smoke training.
- Confirm logs, metrics, checkpoints, and STOP-file shutdown.
- Only then define longer remote runs.

Completed:

- Created the isolated remote UDLF layout:
  `L:\UDLF_REMOTE\UDLF`, `L:\UDLF_REMOTE\runs`, and
  `L:\UDLF_REMOTE\workspace-service`.
- Synced the local UDLF repository to `L:\UDLF_REMOTE\UDLF`.
- Rebuilt the mature HTTPS workspace-service workflow for UDLF with
  `remote_workspace_agent.py`, client, supervisor, train-job wrapper,
  installer, manager, and local command wrapper.
- Installed the service as `UDLF Workspace Agent`, bound to remote
  `127.0.0.1:9543`, accessed locally through an SSH tunnel.
- Verified `remote_workspace.ps1 health` against the isolated root/repo/runs.
- Ran a minimal remote shell job through the HTTPS agent and confirmed the RTX
  4090 is visible.

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

Status: implementation scaffold complete; experiment gate pending.

Purpose:

Decide whether controlled posterior training is justified.

Entry conditions:

- Stage A prior-path training is stable.
- Pure-prior rollout has measurable task value.
- State-dependent diffusion shows a reproducible reason to exist, or the
  project explicitly pivots to deterministic/ODE variants.
- State intervention tests prove the persistent field is causally used.

Tasks:

- Implement the controlled posterior without allowing posterior state
  propagation. Status: implemented behind `mode="stage-b"`.
- Define posterior dropout schedule and gradient-weight normalization. Status:
  implemented as `posterior_dropout` with inverse retention normalization.
- Define posterior-prior gap diagnostics before implementation. Status:
  implemented as `posterior_prior_state_gap`, `loss_prior`,
  `loss_posterior`, `posterior_kl`, `posterior_used`, and
  `posterior_weight`.
- Run only smoke tests until Stage A has a reliable scale baseline.

Acceptance criteria:

- Stage B is not used to rescue a prior model that does not work.
- The double-track state protocol is testable.
- Any filtering-posterior experiment is labeled as a different training regime.
- Stage B large-scale experiments remain blocked until Stage A 64M throughput
  and baseline comparisons are clean.

## Phase 7: Full Architecture Faithfulness

Status: in progress.

Purpose:

Make the code path match the v0.6 design surface before drawing stronger
architecture conclusions.

Completed:

- Stage A single-path prior training.
- Stage A multi-sample prior objective with per-token `logsumexp`.
- Explicit multi-sample state continuation policy through
  `prior_state_selection`.
- Stage B controlled posterior with shared prior drift/diffusion, per-microstep
  posterior control, discrete control-energy KL, posterior dropout, and
  double-track prior-state propagation.
- Local smoke config for Stage B.
- Injection diagnostics for actual state jump, relative jump, allocation
  entropy, write-gate saturation, and injection-state cosine.
- Opt-in finite-difference stability diagnostics for injection gain, prior
  drift gain, and an FTLE proxy.

Remaining:

- Validate the new diagnostic fields on the isolated remote CUDA path.
- Explicit filtering-posterior ablation as a separately labeled regime.
- Large-scale Stage B is intentionally not scheduled yet.

## Current Priority

Stop spending more effort on narrow state-intervention diagnostics as the main
decision vehicle. They were useful for finding implementation bugs and obvious
fragility, but they are too small to settle the architecture question. The next
phase is to run the already implemented 64M-parameter UDLF versus standard
Mamba FineWeb-Edu ablation on the isolated remote RTX 4090 workflow.

Immediate implementation target:

- Full UDLF LLM model path suitable for causal language modeling. Status:
  implemented through `architecture="udlf"`.
- Standard Mamba-style sequence model baseline with matched tokenizer/data
  pipeline and roughly 64M trainable parameters. Status: implemented as a pure
  PyTorch Mamba/S6 baseline through `architecture="mamba"`.
- FineWeb-Edu data loader/configs using the local dataset location. Status:
  implemented using `E:/NAIME_DATA/datasets/fineweb_edu_1b_ctx1024`.
- 3000-step ablation configs with compact console logs and file-backed metrics.
  Status: implemented; local one-step CUDA sanity checks passed; remote paths
  and launch wrappers are verified. The first remote sanity exposed an invalid
  low-throughput schedule. Local UDLF throughput now passes the reference
  target with `solver_steps=2`; formal launch is still blocked on remote 4090
  auto-batch validation and a short throughput sanity run.
- Isolated remote service under `L:\UDLF_REMOTE`, reusing
  `L:\NAIME_REMOTE\envs\.venv312` only. Status: installed and health/job smoke
  verified through SSH tunnel.
- Clear comparison table: loss/perplexity, throughput, memory, stability, and
  checkpoint paths.

Immediate performance gate before any 3000-step remote ablation:

- Use NAIME-style real forward/backward probing on the 4090 to select the
  largest safe micro-batch under `vram_fraction`.
- Preserve the intended effective micro-batch count by automatically reducing
  `grad_accum_steps` when the selected micro-batch grows.
- Keep `dynamics_diagnostics=false` for LLM scale training unless a diagnostic
  run explicitly enables it.
- Keep `stability_diagnostics=false` for normal LLM scale training; enable it
  only for diagnostic short runs because it performs extra finite-difference
  model passes.
- Treat CPU as a test-only escape hatch. Real training configs must use CUDA
  and the trainer rejects non-CUDA devices unless `allow_cpu_training=true` is
  set explicitly.
- Use the solver-2 UDLF 64M template as the current performance configuration.
  Treat it as a changed integration granularity and validate quality/stability
  against the earlier solver-4 behavior.
- Replace the hand-written pure PyTorch Mamba baseline with an official or
  otherwise kernel-accelerated implementation before using it as the comparison
  baseline.
- Until official kernels are available, keep the hand-written Mamba baseline
  semantically aligned with Mamba1: official dt initialization, A/D
  parameterization, Add->Norm->Mixer residual structure, residual fp32 option,
  and padded internal vocabulary. Treat its throughput as a lower bound for a
  non-fused implementation, not as official Mamba performance.
- Confirm short-run UDLF and Mamba throughput from `metrics.jsonl`, not from a
  one-step eval/checkpoint-heavy smoke run.
- Eval must have an independent resource budget before any long run is
  trusted. For UDLF, eval uses `eval_batch_size` and the configured
  `segment_len` instead of inheriting the auto-selected training batch into a
  non-segmented full-sequence path. Long runs should set `eval_batch_size`
  explicitly when auto-batch is enabled.
- Do not restart the 3000-step ablation until this gate passes.

### Custom Mamba CUDA backend

- Own a narrow Mamba1 selective-scan CUDA backend for `sm_89`, `d_state=16`, and BF16/FP32 training.
- Store recurrent checkpoints every 64 tokens and recompute chunks in backward.
- Fail closed when a forced fused backend cannot build or load.
- Require gradient parity and sustained model throughput before a 3000-step run.

Console policy for future local/remote experiments:

- Training configs should use `console_log_mode="quiet"` for long runs unless
  interactive debugging needs live step logs.
- Evaluators and checkers should keep their default compact one-line output.
  Full JSON should be written to files and printed only with `--print-json`.

## UDLF Failure Remediation

Implementation status: complete locally; remote persistence validation pending.

1. Preserve slot identity throughout observation injection, latent interaction,
   and readout using one shared trainable identity tensor normalized to latent
   RMS scale.
2. Initialize the physical latent state at its normal RMS scale and initialize
   tied token embeddings at `std=0.02`; reject any gate with non-finite values,
   anomalous random-baseline loss, or first-step amplification.
3. Tie input/output vocabulary weights and set `latent_dim=792`, yielding
   `64,025,937` trainable parameters while moving capacity into the dynamics
   core.
4. Train with random 64-256-token truncation and a full-sequence BPTT step every
   32 optimizer steps. Keep truncation sampling on a generator independent of
   Brownian noise. Auto-batch truncated steps independently, and use a separately
   validated full-BPTT micro-batch with compensating accumulation so the rare
   full step does not throttle every ordinary step. Scale random-horizon batch
   inversely with horizon (for example 64x64, 32x128, 16x256) and compensate
   with accumulation so long random horizons cannot exceed the probed budget.
5. Run a 300-500 step remote 4090 gate first. Record fixed-sample validation
   loss, slot rank/cosine, grad norm distributions, full-BPTT step behavior,
   throughput, and reserved VRAM.
6. Launch a replacement 3000-step run only if the medium gate retains slot rank
   >= 8, pair cosine < 0.8, finite gradients, and a credible loss trajectory.
   The failed historical run remains the control and must not be overwritten.

## CUDA Residency Repair

The first repaired 3000-step launch was cancelled at step 41 after Windows
WDDM paged an overcommitted CUDA allocator into 64 GB system RAM. System RAM is
not a substitute for the RTX 4090's 24 GB VRAM; the process remained at 100%
reported utilization but only about 125 W and made no step progress for more
than 11 hours.

1. Replace arbitrary integer horizons with a small configured bucket set so
   CUDA sees stable activation shapes.
2. Cap the PyTorch process allocator to 95% of VRAM available at process start,
   expressed as an absolute-free-to-total fraction. Fail with CUDA OOM instead
   of allowing WDDM system-memory paging.
3. Release cached blocks when the `(horizon, micro-batch)` shape changes and
   reset peak statistics per optimizer step.
4. Record current allocated/reserved memory separately from per-step peaks,
   plus step duration, step throughput, and an on-disk heartbeat.
5. Validate 64, 128, 256, and full-512 paths sequentially in one long-lived
   remote process. Require bounded VRAM, finite gradients, and no progressive
   throughput collapse before another formal launch.
6. Sample the stable buckets with weights 0.6/0.3/0.1 for 64/128/256 and retain
   periodic full-512 steps. This preserves long-credit coverage without making
   the most expensive truncated horizon one third of all optimizer steps.
7. Use memory-model prediction to jump auto-batch probes toward the safe cap,
   then binary search only the failed bracket.

## Post-Repair Attribution

The repaired 64M run is complete. It resolved slot collapse and improved the
same-sample loss from `5.1396` to `4.8582`, but Mamba remains better at
`4.3899`. The next phase is attribution, not another blind full run.

1. Re-run component and state-carry ablations on the repaired checkpoint.
2. Report loss by token position and configured horizon bucket.
3. Audit full512 gradient clipping separately from truncated steps.
4. Profile solver/readout execution and identify semantics-preserving fusion or
   vectorization opportunities.
5. Require fixed-sample improvement before scheduling another 3000-step run.

Measured attribution is recorded in `doc/udlf_systemic_gap_attribution.md`.
The leading quality hypotheses are horizon-dependent gradient distortion and
insufficient independently parameterized transformation depth. The leading
performance cause is the serial Python/solver execution path. Matched profiling
measured a `39.6x` operator-call ratio and `56,962` UDLF CUDA launches in one
batch-2, length-128 training step.

### Causal diagnosis before the next architecture revision

Status: gradient diagnosis complete; architecture repair in progress.

1. Recompute gradients from the repaired checkpoint on identical validation
   sequences under 64, 128, 256, and full-512 credit horizons.
2. Report total and per-module gradient norms, the actual global-clip scale,
   and cross-horizon gradient cosine. Global clipping is a scalar rescale and
   must not be described as direction distortion without separate evidence.
3. Use deterministic ODE rollout for this diagnosis so Brownian path variance
   cannot contaminate the horizon comparison.
4. If long-horizon gradients are directionally consistent but only rescaled,
   repair the effective learning-rate protocol before changing architecture.
5. If horizon gradients disagree substantially while the quality gap remains
   flat by position, prioritize matched-parameter independently parameterized
   latent depth.
6. Do not launch another 3000-step run until the chosen repair passes unit
   tests, a remote CUDA smoke, and a fixed-sample quality gate.

Decision:

- gradient direction remains strongly aligned across horizons, so clipping is
  not the primary repair target;
- implement a matched-parameter independently parameterized latent hierarchy;
- train the candidate in ODE mode because final-checkpoint diffusion showed no
  measurable quality benefit;
- retain configurable clipping and record the applied clip scale, but do not
  claim it explains the existing quality gap.

Architecture candidate:

- four independently parameterized latent interaction blocks;
- residual delta accumulation scaled by `1/depth` so initial vector-field
  magnitude does not grow with hierarchy depth;
- each deep block output projection initialized with an additional
  `1/sqrt(depth)` scale to control full-horizon Jacobians while remaining
  learnable;
- `latent_dim=488`, `solver_steps=1`, and `diffusion_mode=ode`;
- `64,523,673` trainable parameters;
- legacy `prior_depth=1` checkpoints retain their original parameter surface.

Validation status:

- local RTX 5060 full-512 forward/backward: loss `10.856`, grad norm `3.85`,
  finite, peak reserved `2.18GB`;
- remote RTX 4090 smoke covered 64/128/256/full-512 with finite loss and slot
  rank `13.7-14.9`; full gradient norm was `7.08-7.25`;
- shared-card auto-batch now reads system-wide free VRAM and selected batch 24
  under an `11.09GB` process cap.

Next gate:

- run 1000 matched-data steps with fixed-sample evaluations at 250-step
  intervals;
- require a better fixed-sample trajectory than the repaired depth-1 model at
  matched steps before extending to 3000;
- use `full_bptt_batch_size=4` while unrelated CUDA jobs leave only about
  `11.7GB` actually available.

Active run:

- workspace job: `fef9dd1a32cf472680eefb6dd1412755`;
- run directory: `L:\UDLF_REMOTE\runs\udlf_hierarchical_64m_depth4_1000_gate`;
- selected schedule: batch 24, accumulation 3, effective batch 72;
- allocator cap: `11.09GB` from `11.67GB` system-wide available VRAM;
- first decision checkpoint: step 250 fixed validation.

Gate result:

- stopped cleanly at step 315 after external Jupyter CUDA workloads filled the
  card and pushed full-step duration as high as `1469s`;
- depth-4 step-250 eval loss was `6.6594` versus depth-1 `6.5678`;
- mean training loss remained worse by `0.06-0.23` in every 50-step window
  from step 100 through 299 despite a larger effective batch;
- depth-4 is rejected for both quality and execution cost.

Final repair candidate under test:

- preserve the validated `latent_dim=792`, shared prior core, and two half-step
  integration path;
- add one independent rank-64 residual adapter per solver substep;
- zero-initialize adapter outputs so the initial function exactly matches the
  stable repaired depth-1 model, then allow substeps to diverge through
  position-specific gradients;
- `64,330,065` parameters;
- 300-step gate must beat the depth-1 step-250 evaluation and loss windows
  before any 1000-step continuation;
- no 3000-step experiment may start until this candidate passes the complete
  300/1000-step gate and fixed-sample comparison.

Active adapter gate:

- workspace job: `ac4141d6ea054a41bc13e37cde3f924f`;
- run: `L:\UDLF_REMOTE\runs\udlf_solver_adapter_64m_300_gate`;
- system-wide available VRAM at launch: `6.80GB`; allocator cap: `6.46GB`;
- selected batch 15, accumulation 5, effective batch 75;
- first hard quality decision remains step-250 eval versus depth-one `6.5678`.

Adapter gate result:

- cancelled at step 64; no 3000-step continuation;
- step-63 loss `7.7210` versus depth-one `7.6423`;
- slot pair cosine `0.979` versus `0.184`, and centered RMS `0.148` versus
  `0.855`, proving early common-state collapse;
- same-device local throughput `328 tok/s` versus base `421 tok/s`, an
  intrinsic 22 percent regression independent of remote contention.

Current final-repair boundary:

1. Remove rejected hierarchy/adapter variants from the launch path; retain
   their reports as negative evidence.
2. Preserve the validated width-792, depth-one, solver-2 ODE quality baseline.
3. Add an independent full-BPTT batch probe to eliminate avoidable
   accumulation overhead under variable available VRAM.
4. Measure remote Python-3.12 `torch.compile` steady state only when external
   CUDA workloads release the card.
5. If compile does not materially improve launch count and throughput, fuse
   the complete recurrent token/solver cell with forward and gradient parity.
6. Reopen quality architecture work around local token representation without
   shrinking latent width or weakening slot separation.
7. Start 3000 steps only after both quality and performance gates pass.
