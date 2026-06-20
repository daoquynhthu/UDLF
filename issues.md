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

### Workspace agent can load a stale custom-scan binary

The optimized CUDA implementation reaches `21,303` token/s at batch 36 through
the explicit build environment, but the default train-job path measured only
`1,308` token/s with identical model settings, consistent with a stale serial
extension binary. This does not block a supervised shell launch but does block
using the generic train endpoint for the formal Mamba run.

Resolution plan:

1. Launch the formal run through the explicit VS/CUDA build environment.
2. Give custom extension variants content-addressed names or build directories.
3. Restart the workspace agent after deploying corrected boolean serialization.
4. Revalidate the generic train endpoint before using it for later runs.

Exit criteria: generic and explicit launch paths load the same extension and
produce comparable sustained throughput.

### 64M FineWeb-Edu ablation has not run yet

Status: open.

Blocker classification:

- This supersedes the small state-intervention diagnostics as the main
  architecture decision gate.
- It blocks claims about UDLF versus Mamba on language modeling quality,
  throughput, and stability.

Evidence:

- The local dataset exists at `E:/NAIME_DATA/datasets/fineweb_edu_1b_ctx1024`
  with `train`, `validation`, and tokenized `input_ids` rows of length `1025`.
- The first framework pass supports `architecture="udlf"` and
  `architecture="mamba"`.
- Parameter calibration gives roughly matched 64M models after refitting UDLF
  to use `latent_dim=512` plus an untied output head: UDLF ~68.1M and Mamba
  ~63.7M.
- One-step CUDA sanity passes for both templates locally and on the remote
  RTX 4090. Remote sanity uses
  `L:/NAIME_REMOTE/datasets/fineweb_edu_1b_ctx1024` and writes outputs under
  `L:\UDLF_REMOTE\runs`.
- The isolated remote 4090 workspace service is now installed and verified
  under `L:\UDLF_REMOTE`; the remaining risk is remote data-path/config
  correctness for the actual 3000-step launch.
- Local RTX 5060 throughput testing shows the current stack is still below the
  known NAIME 64M reference target of roughly `2000` tokens/s. UDLF reached
  about `1024` tokens/s at manual batch `28`, while the auto-batch path
  originally selected only batch `10`. The pure PyTorch Mamba baseline is much
  worse because it uses a Python per-token selective scan and is not a valid
  high-performance Mamba comparison yet.
- Profiling showed the main UDLF performance limiter is recurrent dispatch
  count. Reducing `solver_steps` from `4` to `2` crosses the local throughput
  target on the RTX 5060, reaching more than `2400` tokens/s in a short
  auto-batch run. This is a real engineering speedup but changes integration
  granularity, so quality and stability must be checked explicitly.
- A 20-step local solver `2` versus solver `4` quality/stability gate matched
  early train/eval loss and sampled stability diagnostics. This reduces the
  immediate risk of the solver-step change, but the remote short run and longer
  quality trajectory are still required before the 3000-step ablation.

Resolution plan:

1. Run a one-step CUDA sanity check for each 64M config to catch OOM or config
   errors. Done.
2. Fix the auto-batch selector so predicted memory estimates cannot skip
   real candidates that fit. Done locally; remote validation pending.
3. Replace or accelerate the pure PyTorch Mamba baseline before treating it as
   a performance comparison.
4. Profile and reduce UDLF per-token/per-solver dispatch overhead until local
   64M throughput is close enough to the NAIME reference to justify remote
   3000-step runs. Local throughput gate passes with `solver_steps=2`; remote
   validation pending.
5. Validate solver `2` on a remote short run once the 4090 is free, including
   throughput, eval loss, and stability diagnostics.
6. Finalize the exact run names for the remote 3000-step pair.
7. Launch the 3000-step UDLF and Mamba jobs with quiet console logging through
   the HTTPS workspace service.
8. Track step, train/eval loss, perplexity, throughput, CUDA memory, checkpoint
   status, and failures.
9. Summarize the first complete or failed ablation in
   `doc/fineweb_edu_64m_ablation.md`.

Exit criteria:

- Both runs either complete 3000 steps or fail with recorded failure
  checkpoints/logs.
- The comparison table is updated from actual metrics, not config intent.
- Any OOM or runtime bottleneck has a concrete follow-up plan.
- Short local and remote throughput sanity runs record selected batch,
  accumulation, effective batch, reserved memory, and tokens/s in metrics.

### Official Mamba baseline is not locally runnable on Windows yet

Status: open.

Blocker classification:

- This blocks replacing the current hand-written PyTorch Mamba baseline with a
  valid official high-performance comparison on the local Windows machine.
- It does not block UDLF-only local work, but it blocks any Mamba throughput or
  quality comparison from being treated as mature evidence.

Evidence:

- Official `state-spaces/mamba` was downloaded under
  `artifacts/vendor/mamba` at commit `0048fbf`.
- Current official main requires the newer dependency chain including
  `tilelang==0.1.8`; installing from main failed on local Windows/Python 3.14
  because the dependency metadata was inconsistent on the package index.
- Stable official `mamba-ssm==2.2.6.post3` avoids the `tilelang` chain, but
  PyPI has no Windows binary wheel for the local environment.
- `triton-windows==3.5.1.post24` installs and provides `import triton`, but
  pip metadata still does not satisfy the official dependency name `triton`.
- A local Python 3.12 venv at `.venv_mamba_official` can install CUDA Torch and
  `triton-windows`; `mamba-ssm` only installs with
  `MAMBA_SKIP_CUDA_BUILD=TRUE`, which leaves `selective_scan_cuda` missing.
  Importing `from mamba_ssm import Mamba` then fails with
  `ModuleNotFoundError: No module named 'selective_scan_cuda'`.

Resolution plan:

1. Do not use the hand-written PyTorch Mamba implementation as the final
   baseline.
2. Prefer a Linux/CUDA environment or a known-good prebuilt `mamba-ssm` wheel
   for the official baseline.
3. If staying on Windows, install a complete MSVC/CUDA extension build chain
   and verify `selective_scan_cuda` builds before integrating it.
4. Only after `from mamba_ssm import Mamba` and a CUDA forward smoke pass should
   the training harness add the official Mamba architecture path.

Exit criteria:

- `mamba_ssm`, `selective_scan_cuda`, and optional `causal_conv1d` import in
  the chosen environment.
- A CUDA forward smoke using official `Mamba` succeeds.
- The 64M baseline config uses the official module and records its package
  version/environment.

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
- A 4-seed pre-CRN diffusion ablation on query recall showed positive
  perturbation deltas for fixed diffusion and state-dependent diffusion across
  all seeds. This evidence is now downgraded: those intervention candidates did
  not share suffix Brownian paths, so small robustness deltas are not clean
  evidence of diffusion advantage. Large core deltas remain directionally
  useful, but small perturbation/attenuation/mixed deltas must be re-measured
  with CRN paired statistics.

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

- Treat the mixed-alpha sweep as evidence that structured perturbation is more
  informative than raw random noise, but not as a complete robustness solution.
  After the CRN fix, batch-mix at alpha `0.2` is positive across seeds
  `900-902` with tight paired intervals. Perturbed state remains negative on
  those seeds, and temporal-mix is still not a clean pass.
- Define per-probe thresholds and decide whether temporal-mix failure is an
  expected property of the task, an evaluation artifact, or evidence of brittle
  temporal state geometry.
- Re-run any diffusion-mode robustness comparison that depends on small deltas
  with common random numbers before using it to justify fixed or
  state-dependent diffusion.
- The first CRN re-evaluation of matched query-recall checkpoints changes the
  robustness read: state-dependent diffusion has the strongest mean
  perturbation delta and stays positive across seeds `710-713`. Fixed diffusion
  remains useful and simpler, but it is no longer the strongest robustness
  candidate from the current evidence.
- Real-token state-dependent K=4 does not yet transfer cleanly. Seeds
  `903-905` pass the core gate but have inconsistent robustness, while seed
  `906` fails the core shifted-state threshold (`+0.0038 < +0.0200`) despite
  positive perturb and batch-mix CIs. Do not promote state-dependent diffusion
  to default from synthetic CRN evidence alone.
- Do not expand this issue with more raw observations unless they change the
  decision or close one of the resolution-plan steps.
- Keep Phase 5 remote smoke scoped to fixed K=4 real-token query recall.
- Before running more identical state-dependent real-token confirmations,
  inspect whether the state-dependent parameterization or regularization is
  underconstrained. A small controlled variant is more informative than simply
  adding seed `907`.
- A targeted seed `906` control with `sigma_max=0.01` passed the core gate
  where `sigma_max=0.02` failed. This does not close the issue, but it turns
  the next resolution step into a sigma-range matrix rather than an immediate
  architecture rewrite.
- The fragile-seed sigma matrix is complete for seeds `904` and `906`.
  `sigma_max=0.01` is the best current state-dependent candidate, but it is not
  a default: seed `904` still has weak perturbation, and temporal-mix remains
  near zero.
- The sigma read is potentially confounded with training horizon and model
  scale. A 600-step medium local run can identify a sensitive control variable,
  but cannot establish a scale-independent default.
- The first horizon check confirms that concern: seed `906` with
  `sigma_max=0.010` passes the core gate at 600 steps but fails after
  continuation to 1200 steps. The state-dependent branch now has a
  horizon-instability issue, not just a sigma-selection issue.

Exit criteria:

- A documented robustness gate exists with thresholds, target interventions,
  and rationale.
- Fixed K=4 or a replacement default passes that gate on at least three seeds,
  or the project explicitly downgrades robustness from a Stage A acceptance
  requirement.
- Any state-dependent replacement candidate passes both the core gate and the
  CRN robustness gate on real-token seeds, not only on synthetic query recall.
- The selected state-dependent sigma range is confirmed on at least three
  real-token seeds, not only the fragile-seed matrix.
- The selected range survives at least one longer-horizon check and one
  architecture-scale check before being treated as a default.
- Shifted-state intervention remains destructive after longer training, or the
  core gate is replaced by a documented temporal-geometry probe that better
  matches the learned state protocol.
- `plan.md` no longer depends on unresolved robustness before any long-running
  scale-up that claims stochastic latent robustness.

### Remote 4090 64M throughput was invalidly low

Status: active.

The first remote 64M sanity run proved that the isolated service, data path,
checkpoint path, and CUDA environment work, but it did not prove the training
schedule is acceptable. The UDLF run used a small-GPU micro-batch policy on a
4090 and the model still performed latent readout vocabulary projection one
token at a time. That makes the reported `42.691` tok/s a pipeline defect, not
an architectural result.

Resolution plan:

1. Add NAIME-style automatic CUDA micro-batch probing using real
   forward/backward/optimizer steps.
2. Select the largest safe batch under a VRAM budget and automatically adjust
   gradient accumulation to preserve the configured effective micro-batch
   target.
3. Vectorize UDLF readout across sequence positions so the output projection is
   not dispatched once per token.
4. Disable dynamics instrumentation for normal LLM-scale runs.
5. Re-run short remote 4090 sanity for both UDLF and Mamba and record selected
   batch, accumulation, memory, and tokens/second before relaunching the
   3000-step ablation.
6. Use measured VRAM anchors to estimate a safe candidate upper bound, then
   binary-search the interval instead of relying on powers-of-two growth.
   Apply a safety multiplier before running each candidate.
7. Add a separate larger-candidate profile for UDLF batch `48/64` if we still
   want to characterize the high end of the 4090 envelope.

Exit criteria:

- Remote `config.json` records auto-selected `batch_size` and adjusted
  `grad_accum_steps`.
- `train.log` records successful probe candidates and the selected batch.
- A short remote sanity run reports throughput from normal train steps without
  step-1 eval/checkpoint distortion.
- The formal 3000-step jobs are relaunched only after this sanity gate passes.
- Batch `64` is either skipped by prediction in the normal auto-batch path or
  characterized separately under an explicit profiling run.
- Probe accounting uses CUDA reserved memory, not only allocated memory, so
  allocator reservation and fragmentation cannot be hidden behind a small
  `max_memory_allocated` value.
- Remote GPU is free of unrelated high-memory Python jobs before any formal
  ablation run starts; otherwise throughput and selected batch are not valid
  UDLF evidence.

## Resolved

### UDLF eval inherited the auto-selected training batch

Resolved on 2026-06-19.

The contended 4090 run initially reached step `490` normally and then stalled
at the step-500 eval boundary because `_evaluate_loss` used the auto-selected
training batch `64` while hard-coding UDLF `segment_len=0`. The trainer now has
an explicit `eval_batch_size`, UDLF eval reuses the configured `segment_len`,
and metrics record both values. The resumed remote 3000-step run completed
successfully; eval rows at steps `1000`, `1500`, `2000`, `2500`, and `3000`
all used `eval_batch_size=8` and `eval_segment_len=64`.

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

### Intervention evaluation used unpaired suffix Brownian paths

Resolved on 2026-06-18.

`_evaluate_interventions` previously advanced the same generator across
candidate states, so correct, zero, swapped, shifted, mixed, temporal-mixed,
attenuated, inverted, and perturbed suffix rollouts could use different
Brownian paths. This did not affect ODE and is unlikely to reverse large
zero/swapped/inverted effects, but it contaminated small stochastic deltas. The
evaluator now uses common random numbers for suffix rollouts, multiple paired
suffix seeds, and reports paired mean, standard error, and 95 percent
confidence intervals.

### Isolated remote workspace service was not established

Resolved on 2026-06-18.

UDLF now has a separate remote workspace under `L:\UDLF_REMOTE`, with repo,
runs, workspace-service state, staging, job database, token, and TLS files kept
outside the remote NAIME repository. The service reuses only
`L:\NAIME_REMOTE\envs\.venv312` and is installed as `UDLF Workspace Agent`.
Direct LAN binding was dropped because SSH-launched child processes were not a
stable service boundary and exposed 9543 unnecessarily. The final service binds
remote `127.0.0.1:9543`, and `scripts/remote_workspace.ps1` accesses it through
an SSH local tunnel. Health and a minimal GPU shell job passed.

### Full v0.6 diagnostics needed CUDA validation

Resolved on 2026-06-19.

The model/training surface now includes Stage A multi-sample prior training,
Stage B controlled posterior training, injection diagnostics, and opt-in
finite-difference stability diagnostics. A remote CUDA Stage B diagnostic
short run under the isolated UDLF workspace emitted injection jump, relative
jump, allocation entropy, write-gate saturation, injection-state cosine,
`stability_injection_fd_gain`, `stability_drift_fd_gain`, and
`stability_ftle_proxy`. Normal LLM-scale templates keep both diagnostics
disabled by default.
