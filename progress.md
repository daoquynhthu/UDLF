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
- Fixed intervention evaluation to use common random numbers for suffix
  rollouts. Each candidate state now reuses the same paired Brownian suffix
  paths; intervention metrics report paired mean, standard error, and 95%
  confidence interval.
- Updated the read-only checkpoint evaluator to use the same CRN paired
  intervention path and expose `--pair-trials`.
- Re-ran CRN paired evaluation for fixed K=4 real-token query-recall seeds
  `900`, `901`, and `902` at alpha `0.2`. Batch-mix remained positive with
  tight intervals; perturbed deltas were negative across all three seeds, and
  temporal-mix remained inconclusive or negative.
- Added direct dynamics instrumentation for drift RMS, diffusion sigma min/max,
  sigma RMS, and jump RMS, closing the prior Phase 3 metrics gap around
  diffusion saturation visibility.
- Re-evaluated the matched ODE, fixed-diffusion, and state-dependent diffusion
  query-recall checkpoints for seeds `710-713` with CRN paired metrics.
  State-dependent diffusion had the strongest mean perturbation delta
  (`+0.0363`) and stayed positive on every seed; fixed diffusion averaged
  `+0.0311` but had one near-zero seed; ODE stayed near zero (`+0.0010`).
- Added `scripts/summarize_crn_interventions.py` and tracked CRN summary files
  under `doc/phase4_crn_intervention_*`.
- Reduced console output for long-running experiments: training now supports
  `console_log_mode` (`progress` or `quiet`), and checkpoint intervention
  evaluation prints only a compact one-line summary unless `--print-json` is
  explicitly requested.
- `scripts/check_state_probe.py` now also defaults to a compact one-line
  result; full JSON output requires `--print-json`.
- Ran a medium local 5060 real-token query-recall state-dependent K=4 training
  run for seed `903` to 600 steps. CRN eval at alpha `0.2` passed the core
  gate and produced perturb `+0.0309`, batch-mix `+0.0181`, and temporal-mix
  `+0.0048`.
- Ran a second local 5060 state-dependent K=4 real-token seed `904` to 600
  steps with quiet console logging. Core gate passed, but robustness did not:
  perturb was weak with CI crossing zero (`+0.0030`, CI `[-0.0018,+0.0078]`)
  and batch-mix was significantly negative (`-0.0077`, CI
  `[-0.0106,-0.0047]`).
- Added `doc/real_token_state_dependent_crn.md` to separate real-token
  state-dependent CRN evidence from the synthetic diffusion ablation.
- Ran local 5060 state-dependent K=4 real-token seed `905` to 600 steps with
  quiet console logging. Core gate passed. Batch-mix and temporal-mix were
  positive with non-crossing CIs, but perturbation remained near zero with CI
  crossing zero (`+0.0011`, CI `[-0.0017,+0.0040]`).
- Ran local 5060 state-dependent K=4 real-token seed `906` to 600 steps with
  quiet console logging. CRN eval completed, but the core profile failed:
  shifted-state delta was only `+0.0038` against the `+0.0200` threshold.
  Perturb and batch-mix were positive with non-crossing CIs, while temporal-mix
  crossed zero. This downgrades state-dependent real-token K=4 from "core
  stable, robustness mixed" to "not yet a replacement default".
- Ran a targeted seed `906` sigma-control with state-dependent K=4 and
  `sigma_max=0.01`. The same seed passed the core gate: shifted-state delta
  improved from `+0.0038` to `+0.1047`, perturb stayed positive
  (`+0.0097`, CI `[+0.0096,+0.0099]`), and batch-mix stayed positive
  (`+0.0335`, CI `[+0.0315,+0.0356]`). Added
  `doc/state_dependent_sigma_control.md`.
- Completed the fragile-seed state-dependent sigma matrix for seeds `904` and
  `906` across `sigma_max` values `0.005`, `0.010`, `0.015`, and `0.020`.
  Seed `906` passed the core gate at `0.005` and `0.010`, then failed at
  `0.015` and `0.020`. Seed `904` passed the core gate at all four amplitudes,
  but robustness remained inconsistent. Added
  `doc/state_dependent_sigma_matrix.csv`.
- Ran a seed `906` horizon check by continuing `sigma_max=0.010` and
  `sigma_max=0.020` state-dependent K=4 runs from 600 to 1200 steps.
  `sigma_max=0.010` lost its 600-step core pass at 1200 steps
  (`shifted=-0.0139`), and `sigma_max=0.020` also failed
  (`shifted=-0.0344`). Added `doc/state_dependent_horizon_check.csv`.
- Changed `scripts/run_state_probe_matrix.py` to default to compact one-line
  console output; full JSON printing now requires `--print-json`.
- Ran a fixed K=4 seed `906` 1200-step same-seed horizon control. It passed the
  core gate (`shifted=+0.0577`) while perturb and temporal-mix remained
  negative. This closes the small-diagnostic loop: the next phase should stop
  optimizing these probes and move to the planned 64M FineWeb-Edu ablation with
  a full LLM UDLF implementation and a standard Mamba baseline.
- Added the first full FineWeb-Edu 64M ablation framework. The trainer now
  supports `architecture="udlf"` and `architecture="mamba"`, records parameter
  count, skips state interventions for non-UDLF models, and keeps compact logs.
  Added a pure PyTorch standard Mamba/S6 baseline because `mamba_ssm`,
  `causal_conv1d`, and `triton` are not installed locally.
- Added 3000-step FineWeb-Edu templates:
  `configs/training_templates/udlf_fineweb_edu_64m_3000.json` (~68.1M params)
  and `configs/training_templates/mamba_fineweb_edu_64m_3000.json` (~63.7M
  params). Dataset target is `E:/NAIME_DATA/datasets/fineweb_edu_1b_ctx1024`.
- Added `doc/fineweb_edu_64m_ablation.md` and tests for Mamba forward/training
  integration.
- Fixed segmented UDLF training so segment losses backprop immediately when
  `detach_state_between_segments=true`; the previous implementation detached
  state but retained all segment graphs until the final backward, so it did not
  actually reduce activation memory.
- Ran one-step CUDA sanity checks for both 64M templates on the local RTX 5060.
  UDLF passed at 68.1M params with eval loss `10.8556`, `52.814` tok/s, and
  `1818.312` MB CUDA memory. Mamba passed at 63.7M params with eval loss
  `10.9912`, `89.969` tok/s, and `2573.117` MB CUDA memory.
- Created the isolated remote UDLF layout on the RTX 4090 host under
  `L:\UDLF_REMOTE`, separate from the remote NAIME repository.
- Reused only the mature remote Python environment
  `L:\NAIME_REMOTE\envs\.venv312`; UDLF repo, runs, workspace-service state,
  jobs, staging, token, and TLS files live under `L:\UDLF_REMOTE`.
- Ported the mature HTTPS workspace service scripts into UDLF and adapted them
  for UDLF paths, UDLF environment names, port `9543`, compact operations, and
  native `udlf.training.train` launch.
- Fixed the workspace agent TLS accept path so malformed handshakes do not
  terminate the server loop.
- Changed the UDLF workspace service to bind remote `127.0.0.1:9543` and use a
  local SSH tunnel instead of exposing the service on the LAN.
- Installed the service through `scripts/install_remote_workspace_service.ps1`
  as scheduled task `UDLF Workspace Agent`, using the current user and the
  remote `.venv312` Python.
- Verified `scripts/remote_workspace.ps1 health`: root
  `L:\UDLF_REMOTE`, repo `L:\UDLF_REMOTE\UDLF`, runs
  `L:\UDLF_REMOTE\runs`.
- Ran a minimal HTTPS shell job through the service and confirmed the remote
  RTX 4090 is visible with `24564` MiB total memory.
- Fixed service restart handling so `manage_remote_workspace_service.ps1`
  clears both supervisor and agent processes before starting a fresh scheduled
  task.
- Fixed current-user service ACL handling so the scheduled task can read the
  workspace token and TLS key after the service stopped running as SYSTEM.
- Verified remote `.venv312` import preflight from `L:\UDLF_REMOTE\UDLF`:
  `torch 2.11.0+cu128`, CUDA available, GPU `NVIDIA GeForce RTX 4090`.
- Ran the full test suite on the remote 4090 workspace with
  `PYTHONPATH=src` and a UDLF-owned temp directory; `21 passed`.
- Fixed the workspace train job argument handoff so `-Template` is parsed
  correctly by `remote_workspace_train_job.py`.
- Ran remote one-step 64M FineWeb-Edu sanity checks through the isolated HTTPS
  workspace service. UDLF passed at 68.1M params with eval loss `10.8625`,
  `42.691` tok/s, and `1818.312` MB CUDA memory. Mamba passed at 63.7M params
  with eval loss `10.9560`, `61.787` tok/s, and `2573.117` MB CUDA memory.
- Launched the formal remote UDLF 64M FineWeb-Edu 3000-step run as workspace
  job `fc497e2628a045cbb244e4010d107391`, run directory
  `L:\UDLF_REMOTE\runs\udlf_fineweb_edu_64m_3000_remote`.
- Cancelled the formal UDLF remote run before treating it as an experiment.
  The measured throughput was not a meaningful 4090 run: the remote template
  inherited a small-GPU micro-batch policy and the UDLF readout still executed
  the vocabulary projection once per token.
- Added NAIME-style automatic CUDA batch probing to the trainer. The probe runs
  real forward/backward/optimizer steps on candidate micro-batches, respects a
  configurable VRAM fraction, and then adjusts gradient accumulation to
  preserve the target effective micro-batch count.
- Vectorized UDLF latent readout across sequence positions and added a
  `dynamics_diagnostics` switch so large LLM runs do not collect per-step
  dynamics tensors unless explicitly requested.
- Changed remote 4090 templates to enable auto-batch probing instead of
  hard-coding the actual micro-batch.
- First UDLF auto-batch probe on the remote 4090 successfully fit batches `4`,
  `8`, `16`, and `32`; batch `32` used `7.52` GiB peak. A naive next-candidate
  probe at `64` was too aggressive. The probe is being corrected to use a small
  number of measured anchors, estimate a safe VRAM-derived upper bound, then
  binary-search the candidate interval instead of selecting only powers of two.
- Further remote probe debugging showed that UDLF memory is not smooth near
  the high-batch boundary: batch `36` was stable, but nearby higher candidates
  could push `nvidia-smi` used memory to almost the full card. The probe now
  records CUDA reserved memory as well as allocated memory and uses the larger
  value for selection. Remote testing is paused while non-UDLF Python processes
  are using roughly half of the 4090 memory.
- Implemented the main missing v0.6 architecture mechanisms: Stage A
  multi-sample prior training with per-token `logsumexp`, explicit
  multi-sample prior state selection, and Stage B controlled posterior training
  with double-track prior-state propagation, posterior dropout normalization,
  discrete control-energy KL, and posterior/prior gap metrics.
- Added a local Stage B smoke config and tests covering posterior prefix
  shapes, multi-sample prior metrics, and Stage B posterior metrics.
- Made CUDA the default training device and disabled CPU training unless
  `allow_cpu_training=true` is explicitly set for tests or tiny debugging.
- Disabled dynamics diagnostics by default for normal LLM-scale training.
- Added opt-in injection and finite-difference stability diagnostics covering
  state jump, write gates, allocation entropy, injection-state cosine,
  injection gain, prior drift gain, and an FTLE proxy.
- Reduced Stage A optimizer overhead by constructing posterior-control
  parameters only when `mode="stage-b"` enables the posterior path.
- Optimized gradient-norm accounting to avoid repeated scalar CPU transfers.
- Changed 4090 auto-batch accounting to use available CUDA memory times
  `vram_fraction`; remote templates now set that fraction to `0.95` without an
  additional hidden probe discount.
- Synced the updated code to the isolated remote UDLF repo and ran Stage B CUDA
  diagnostic short job `0a276dc10d5f45dcbd8231fe3d073597` in
  `L:\UDLF_REMOTE\runs\udlf_stage_b_diag_short_cuda`. It succeeded with
  `tokens_per_second=34.614`, `cuda_memory_reserved_mb=30.0`, and the expected
  injection/stability diagnostic fields present in `metrics.jsonl`. The remote
  4090 still had unrelated non-UDLF Python processes occupying most of the
  used memory, so this run is only a code-path/metrics smoke, not throughput
  evidence.
- Checked the remote 4090 before throughput launch; it was not idle. Two
  non-UDLF Anaconda Python processes were still occupying most of the used GPU
  memory, so formal remote throughput testing was deferred.
- Ran local RTX 5060 64M FineWeb-Edu throughput probes. UDLF auto-batch before
  the probe fix selected batch `10`, accumulation `2`, and averaged about
  `777` tokens/s. Manual UDLF probes found batch `28` reached about `1024`
  tokens/s, while batch `32` overfilled memory behavior and regressed. The
  pure PyTorch Mamba baseline selected batch `2`, accumulation `8`, and only
  reached about `175` tokens/s before being stopped, so it is not a valid
  high-performance Mamba baseline yet.
- Fixed auto-batch so the memory predictor no longer skips candidate probes.
  The selector now relies on actual forward/backward probes bounded by
  `auto_batch_max_probe_increment`. A repeat local UDLF auto run selected batch
  `24`, accumulation `1`, and averaged about `842` tokens/s under
  `vram_fraction=0.90`; metrics now include `batch_size`,
  `grad_accum_steps`, and `effective_batch_size`.
- Profiled a real UDLF 64M train step. The hot spot is launch overhead and
  small-op count, not a single slow matmul: even a shorter seq-128 profile
  produced about `64k` CUDA launches, with CPU launch overhead comparable to
  total CUDA compute time.
- Tested UDLF `solver_steps` as the dominant dispatch multiplier. On the local
  RTX 5060, batch `28` improved from about `1024` tokens/s at solver `4` to
  about `3291` tokens/s at solver `2`; solver `1` did not improve further.
  An auto-batch solver-2 probe selected batch `34` and reached `2425.7`
  tokens/s by step 3 before final checkpoint writing failed because E: was
  nearly full.
- Updated the UDLF 64M FineWeb-Edu template to use `solver_steps=2`. This is
  the current performance configuration; it still needs quality/stability
  validation before being treated as methodologically equivalent to solver 4.
- Checked remote compile viability. Remote `.venv312` has Python `3.12.10`,
  Torch `2.11.0+cu128`, CUDA, `torch.compile`, and Triton installed. A small
  compile smoke succeeded, but first compile took roughly 227 seconds and its
  run metrics were dominated by compile time. Compile remains a possible long
  run optimization, not a fast local iteration path.
- Ran local UDLF 64M solver-step quality/stability gate: solver `4` versus
  solver `2`, same seed/data/batch, 20 steps, eval at steps `10` and `20`, and
  stability diagnostics every `10` steps. The two settings matched closely on
  train loss, eval loss, grad norm, state RMS, injection finite-difference
  gain, drift finite-difference gain, and FTLE proxy. No NaN or Inf values were
  found. The result supports using solver `2` for the next remote short
  throughput/quality gate, but does not yet prove full long-run equivalence.
- Downloaded the official `state-spaces/mamba` repository to
  `artifacts/vendor/mamba` at commit `0048fbf` and attempted local Windows
  installation. Current main failed on the `tilelang==0.1.8` dependency chain.
  Stable `mamba-ssm==2.2.6.post3` plus `triton-windows` was tested in both the
  current Python 3.14 environment and a new `.venv_mamba_official` Python 3.12
  CUDA Torch environment. The only successful install path used
  `MAMBA_SKIP_CUDA_BUILD=TRUE`, which leaves `selective_scan_cuda` absent, so
  official `Mamba` cannot run locally yet.
- Launched the contended remote UDLF 64M FineWeb-Edu 3000-step run through the
  isolated workspace service as job `e315edf9ea04431c9920ea16e7f27302`, run
  directory `L:\UDLF_REMOTE\runs\udlf_fineweb_edu_64m_3000_solver2_contended`.
  Auto-batch selected batch `64`, grad accumulation `1`, effective batch `64`.
  At step `10`, metrics reported loss `10.7655`, `4491.5` tokens/s,
  `10533.896` MB CUDA allocated, and `11246.0` MB CUDA reserved.
- Stopped job `e315edf9ea04431c9920ea16e7f27302` after it stalled at the
  step-500 eval boundary: `train.log` and `metrics.jsonl` stopped at step
  `490`, stderr was empty, GPU utilization was `100%`, and only `444` MB
  VRAM was free. The immediate root cause was that UDLF eval used the training
  batch size while forcing `segment_len=0`, so the auto-selected batch `64`
  entered a non-segmented full-sequence eval path. Added explicit
  `eval_batch_size`, made UDLF eval reuse the configured training
  `segment_len`, and recorded `eval_batch_size` / `eval_segment_len` in
  metrics. Local `tests\test_stage_a_training.py` passed with
  `PYTHONPATH=src`.
- Synced commit `01b883b` to the isolated remote workspace and relaunched the
  same run from `latest.pt` as job `4dc7082c813d471180f5045d09c8e1cc`, with
  `eval_batch_size=8` and `eval_interventions=false`. The previous orphaned
  process had reached step `500`; the resumed job loaded `resume_step=500`,
  reselected batch `64`, and wrote step `510` normally at about `4599.7`
  tokens/s with empty stderr. The eval-resource fix still needs direct
  confirmation at the next eval boundary, step `1000`.
- Remote job `4dc7082c813d471180f5045d09c8e1cc` completed successfully at
  step `3000`. The run contains `3000` unique metric steps, no NaN or Inf
  numeric metrics, final train loss `5.0348`, final eval loss `5.0028`, final
  eval perplexity `148.84`, and last-100 mean throughput about `4943.3`
  tokens/s. Eval rows at steps `1000`, `1500`, `2000`, `2500`, and `3000`
  all record `eval_batch_size=8` and `eval_segment_len=64`, confirming the
  eval-resource fix. The older step-500 eval row remains pre-fix and lacks
  those fields.
- Reworked the hand-written Mamba baseline toward official Mamba1 semantics:
  separated the pure PyTorch selective-scan mixer from the official-style
  Add->Norm->Mixer block, adopted the official `dt_min/dt_max` log-uniform
  inverse-softplus initialization, preserved `A_log` and `D` no-weight-decay
  markers, added residual-in-fp32 handling, optional vocab padding, and
  GPT-style residual out-projection scaling. Training config and both Mamba
  64M templates now expose the official alignment parameters explicitly. The
  64M templates build a `63,742,080` parameter model with padded vocab `50264`
  while returning logits over the real vocab `50257`. Relevant tests passed:
  `tests\test_stage_a_model.py`, `tests\test_stage_a_training.py`, and
  `compileall`.
- Implemented and remotely compiled a repository-owned CUDA selective-scan
  forward/backward backend with 64-token checkpoint recomputation.
- Matched forward plus all eight gradient groups against PyTorch across a chunk
  boundary and completed a finite 64M fused training smoke.
- Replaced the serial scan with warp-per-channel state updates and block-level
  shared reductions for `B/C` gradients. Gradient parity still passes.
- Remote 64M throughput reached `8,490` token/s at batch 8, `14,360` at batch
  16, `20,237` at batch 32, and `21,303` at batch 36. Batch 36 reserves
  `20.73` GiB and stays below 95% of currently available VRAM.
- Found and fixed workspace-agent boolean overrides silently dropping `false`.
- Completed the fused Mamba 64M 3000-step run without NaN/Inf metrics. Final
  eval loss/ppl are `4.3347`/`76.30`, with last-100 throughput `44,718` tok/s.
- UDLF finished at eval loss/ppl `5.0028`/`148.84` and `4,943` tok/s. Mamba is
  better despite fewer parameters and about 44% fewer training tokens; strict
  confirmation still needs matched tokens and a larger fixed validation set.
- Diagnosed the UDLF checkpoint on the same 128 validation sequences as Mamba:
  loss `5.1396` versus `4.3899`. Final latent slots collapse to participation
  rank `1.82/16` and pairwise cosine `0.941`. Diffusion removal changes loss by
  only `-0.0006`; reset/shuffled carry worsens loss by `+0.133/+0.205`, proving
  the state is useful but trained through an overly short credit horizon.
- Implemented the first full UDLF failure remediation. A shared normalized,
  trainable slot identity now enters injection, dynamics, and readout; initial
  latent slots are non-collapsed; training reports slot cosine, centered RMS,
  and participation rank every step.
- Reallocated the 64M budget by tying the 50,257-token input/output matrix and
  increasing latent width from 512 to 792. Both UDLF templates now build
  `64,025,937` parameters.
- Added independent random 64-256 token credit horizons and periodic full BPTT
  every 32 steps. Auto-batch probes the full path when periodic full BPTT is
  enabled, preventing a truncated-path batch from later OOMing.
- Fixed tied-embedding initialization after the first CUDA gate exposed loss
  `76.5` and grad norm `666`; embedding `std=0.02` restored random-baseline
  scale. Raising initial latent RMS from 0.02 to 1 reduced first-injection
  finite-difference gain from about 50 to `1.015`.
- Final local RTX 5060 gate used the complete 64.03M model, real FineWeb-Edu,
  and full 128-token BPTT. It produced finite loss `10.875`, grad norm `7.71`,
  slot rank `14.93/16`, pair cosine `0.004`, fixed sigma `0.01`, and no OOM.
  Focused model/training tests passed: `34 passed`.
- Restricted pytest discovery to `tests/`. An unrestricted discovery run had
  incorrectly collected vendored Mamba tests and the unrelated untracked
  `pony_remote` workspace, producing external dependency and data-path errors.
- Synced the repair to the isolated remote workspace and measured full 512-token
  BPTT memory on the RTX 4090. Batch 12 peaked at `20.67 GiB` and passed the
  current-free-VRAM 95% budget; batch 13 peaked at `22.07 GiB` and failed it.
- Stopped the first 40-step remote gate after it exposed a scheduling defect:
  constraining all truncated steps to the full-BPTT-safe batch 12 yielded only
  `829 tok/s`. Added dual batch scheduling so normal steps auto-batch on the
  truncated graph while full steps use batch 12 and compensating accumulation.
  Throughput now uses cumulative actual tokens. The full-step micro-batch path
  has a dedicated regression test; the suite passes `36` tests.
- The second remote gate selected batch 64 at the 64-token horizon and reached
  `4383 tok/s` on step 1, but a subsequent longer random horizon saturated
  24 GiB because batch did not scale with truncation length. Stopped the run
  and added inverse horizon/batch scheduling with compensating accumulation:
  64x64, 32x128, 16x256, and measured 12x512. The suite passes `37` tests.
- The horizon-aware gate reached `4383 tok/s` on a 64-token step and kept loss
  finite through step 7, but the batch-12 full step encountered 24.1 GiB
  allocator occupancy after variable-horizon fragmentation. Reduced the formal
  full-BPTT micro-batch to 8; its clean measured peak was `13.38 GiB`, leaving
  enough margin for a long-lived process. Per-step accumulation rises to 8 so
  the optimizer still sees 64 sequences.
- Launched the formal repaired UDLF 64M FineWeb-Edu 3000-step run as job
  `5833830ea8854f4a9df8012dd224344a` in isolated run directory
  `udlf_fineweb_edu_64m_slot_repair_3000`. Auto-batch selected normal batch 64
  at a 14.63 GiB peak; periodic full-BPTT steps use batch 8 with accumulation
  8. The historical failed run remains untouched.
- Cancelled job `5833830ea8854f4a9df8012dd224344a` at step 41 after it made no
  progress for more than 11 hours. Through step 40 the architecture remained
  finite and non-collapsed (loss `8.149`, grad norm `1.904`, slot rank
  `14.88/16`), but arbitrary horizon shapes drove the WDDM allocator to a
  `30,436 MiB` historical reserved peak and system-memory paging.
- Implemented CUDA residency controls: fixed horizon buckets, a hard allocator
  cap based on 95% of free VRAM, cache release on shape transitions, per-step
  current/peak memory metrics, step timing, and heartbeat state. A local RTX
  5060 long-lived gate switched 32/64/full shapes under a 6.34 GiB cap while
  reserved memory stayed between 1.34 and 1.65 GiB. Test suite: `38 passed`.
- Completed remote job `06b4c0679add4ef8bd6a76c14663244b` across 128, 64,
  64, full512, 128, 256, 256, and full512 horizons. The allocator cap was
  `21.84 GiB`; current/peak reserved memory remained `14.18-15.47 GiB`, both
  full steps completed in about 60 seconds, and no shape showed progressive
  slowdown. Added 0.6/0.3/0.1 bucket weights and prediction-first auto-batch
  search. Test suite: `39 passed`.
- Verified prediction-first auto-batch remotely: probes dropped from roughly
  17 candidates to batch 4, 8, and 64, selecting 64 at a `14.63 GiB` peak
  under a `21.33 GiB` cap. Launched the new formal 3000-step run as job
  `c9beaf40b31a4b909b05c353743b45c9` in
  `udlf_fineweb_edu_64m_residency_fixed_3000`. At step 4 it reported finite
  loss `10.754`, grad norm `3.103`, slot rank `14.94`, and current reserved
  memory `15.32 GiB`; heartbeat showed step 5 running.
- Completed job `c9beaf40b31a4b909b05c353743b45c9` at step 3000 with no
  NaN/Inf or stderr. Final eval loss/PPL were `4.8266`/`124.79`, final slot
  rank was `11.34/16`, and peak reserved memory stayed `15.47 GiB` below the
  `21.84 GiB` cap. All 93 full512 steps completed; their mean duration was
  `60.89` seconds and maximum grad norm was `2.64`.
- On the same 128 validation sequences, repaired UDLF reached loss `4.8582`
  (PPL `128.79`) versus old UDLF `5.1396` and Mamba `4.3899`. The repair
  recovered `0.2814` loss and reduced the UDLF-Mamba gap from `0.7497` to
  `0.4683`, but did not close it. Sustained cumulative throughput was
  `2301.5 tok/s`; Mamba remains about 19x faster.
- Completed systemic-gap attribution. The UDLF-Mamba loss gap is nearly flat
  (`0.42-0.49`) across all 64-token position bins, rejecting primarily
  long-context decay. Mean-slot, centered-slot, shuffled-slot, and no-identity
  readout interventions worsen loss by `2.03`, `1.55`, `3.68`, and `1.19`,
  proving the repaired slot structure is functional. Long-horizon gradients
  exceed the clip threshold on `84-97%` of steps versus `1.9%` for Mamba.
- Corrected tied-embedding parameter accounting: repaired UDLF has `38.29M`
  non-vocabulary parameters, not `12.56M`; core undercapacity by raw parameter
  count is therefore rejected. Added reusable attribution tooling and tests.
- Profiled matched batch-2, length-128 forward/backward steps. UDLF issued
  `417,059` operator calls and `56,962` CUDA launches versus Mamba `10,538`
  operator calls; profiler wall time was `8.874s` versus `0.194s`. The
  performance deficit is graph fragmentation across the whole recurrent
  token/solver cell, not one slow matrix multiplication. Full suite passes
  `43` tests.
- Recomputed repaired-checkpoint gradients on the same four validation
  sequences under 64/128/256/512 credit horizons with ODE rollout. Total
  gradient cosines were `0.94-0.99`; module minima remained `0.91-1.00`.
  Clipping changes the effective scalar multiplier from `0.233` to `0.197`
  rather than creating a direction conflict. Architecture depth remains the
  primary quality repair target.
- Implemented a matched-parameter hierarchical UDLF candidate with four
  independent latent blocks, width 488, one ODE solver step, and `64,523,673`
  parameters. The first deep formulation normalized drift features to unit RMS
  and caused full-512 gradient norms of `49-1319`; it was rejected and replaced
  by `1/sqrt(depth)` residual delta accumulation. The corrected candidate
  passed local and remote full-BPTT stability checks.
- Fixed shared-WDDM VRAM accounting and auto-batch termination. On the remote
  4090, PyTorch incorrectly reported `22.46GB` free while system-wide
  `nvidia-smi` showed `11.67GB`. The trainer now takes the smaller value,
  applied an `11.09GB` allocator cap, and safely selected batch 24 via bounded
  probes. Full test suite passes `51` tests.
- Launched the hierarchical 64M 1000-step quality gate as workspace job
  `fef9dd1a32cf472680eefb6dd1412755` in
  `udlf_hierarchical_64m_depth4_1000_gate`. Under current shared-card pressure,
  auto-batch safely selected 24 with accumulation 3 for an effective batch of
  72 and began training under an `11.09GB` allocator cap.
- Stopped the depth-4 gate cleanly at step 315 when unrelated Jupyter CUDA
  jobs saturated the shared 4090. Depth-4 was also consistently worse than the
  repaired depth-1 model by `0.06-0.23` mean loss across step windows 100-299,
  so it is rejected rather than resumed. Added a depth-2 width-640 candidate
  with `64,585,681` parameters and the same two core evaluations per token as
  the old solver-2 path.
- Rejected the width-640 route before remote training: even a shared-core
  width-640 control raised matched full-512 initial gradient norm from `3.99`
  to `68.8`. Replaced it with rank-64 solver-step adapters on the validated
  width-792 trunk. The adapter candidate is functionally identical at
  initialization, passed remote 64/128/256/full smoke, and its two adapters
  diverged to cosine `0.368` after six steps.
- Launched the rank-64 solver-adapter 300-step gate as job
  `ac4141d6ea054a41bc13e37cde3f924f`. Bounded auto-batch used the true
  `6.80GB` system-wide free VRAM and selected batch 15 with accumulation 5;
  the run is isolated from any 3000-step launch decision.
- Cancelled the solver-adapter gate at step 64. It was already worse than the
  depth-one control at matched steps, drove slot cosine to `0.979` with
  centered RMS `0.148`, and reduced same-device local throughput by 22 percent.
  Current remote throughput is additionally contaminated by four Jupyter CUDA
  kernels and only `6.8GB` free VRAM, but the adapter regression is intrinsic
  and independently measured. The 3000-step launch remains blocked.
- Completed local-token and readout diagnosis on the repaired checkpoint.
  Current-token state writing and readout conditioning contribute `+1.421`
  and `+0.928` loss when removed, while injection changes state by only 7.3
  percent RMS. Found that implementation shared a readout key projection even
  though v0.6 specifies one per head; trained eight-head output rank was only
  `2.24`. Implemented an exactly parameter-matched head-specific-key candidate
  by reallocating one FF expansion unit. Local full-512 and remote all-horizon
  smoke passed; tests pass `56` cases.
- Launched the head-specific-readout 300-step quality gate as job
  `392e94f41cae43cbbb76d8736fbb163b`. Auto-batch selected 16 with accumulation
  4, exactly matching the control effective batch of 64. No performance work
  or 3000-step launch will proceed until this architecture gate is resolved.
