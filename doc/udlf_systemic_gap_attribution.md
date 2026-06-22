# UDLF Systemic Gap Attribution

## Scope

This report attributes the remaining gap after the repaired 64M UDLF run. It
separates measured facts from architectural inference. The comparison uses the
same 128 FineWeb-Edu validation sequences and deterministic ODE evaluation for
UDLF.

## Confirmed Results

### The repair worked but did not close the gap

- old UDLF fixed-sample loss: `5.1396`;
- repaired UDLF fixed-sample loss: `4.8581`;
- Mamba fixed-sample loss: `4.3898`;
- remaining repaired UDLF minus Mamba gap: `0.4683`;
- repaired UDLF training tokens: approximately `98.1M`;
- Mamba training tokens: approximately `55.2M`.

The remaining deficit is not explained by fewer training tokens.

### The gap is not concentrated at long positions

| Token positions | UDLF | Mamba | Gap |
|---|---:|---:|---:|
| 0-64 | 5.0064 | 4.5905 | 0.4160 |
| 64-128 | 4.7662 | 4.2968 | 0.4694 |
| 128-192 | 4.8160 | 4.3400 | 0.4761 |
| 192-256 | 4.8418 | 4.3698 | 0.4720 |
| 256-320 | 4.8880 | 4.4229 | 0.4651 |
| 320-384 | 4.8867 | 4.4133 | 0.4735 |
| 384-448 | 4.8537 | 4.3601 | 0.4936 |
| 448-512 | 4.8056 | 4.3251 | 0.4805 |

The first 64 tokens already show a `0.416` deficit, and the gap remains nearly
flat afterward. This rejects a diagnosis based primarily on late-context state
decay. UDLF is weaker at local sequence modeling across the whole context.

### The latent state and slot specialization are functional

- normal carry loss: `4.9016` on the component-diagnostic batch;
- reset state every 64 tokens: `5.0384` (`+0.1367`);
- shuffle state across documents every 64 tokens: `5.1596` (`+0.2580`);
- stateless token mode: `6.1907`;
- injection only: `11.6007`;
- prior dynamics only: `8.9417`.

Readout-state interventions on 128 sequences:

- replace all slots with their mean: `+2.0298` loss;
- remove the slot mean and retain centered components: `+1.5497`;
- shuffle slot order against persistent identities: `+3.6805`;
- remove slot identities only at readout: `+1.1938`.

The repaired slot rank is not decorative. Both common state and slot-specific
state are used by readout, and persistent identities carry semantic alignment.

### Diffusion is not buying quality

- stochastic loss mean: `4.9029`;
- ODE loss: `4.9016`;
- ODE minus stochastic: `-0.00128`.

At this checkpoint the stochastic path adds compute and variance without a
measurable language-modeling gain.

### Core capacity is not smaller than Mamba

The tied output matrix must not be counted twice. Correct repaired UDLF
accounting is:

- total parameters: `64.03M`;
- shared token embedding/output matrix: `25.73M`;
- non-vocabulary parameters: `38.29M`;
- prior dynamics: `23.00M`;
- readout: `12.17M`;
- observation injection: `3.10M`.

The model has sufficient raw non-vocabulary parameter count. The issue is how
those parameters are organized and trained.

### Horizon gradients are almost always clipped

| Horizon | Steps | Mean grad norm | Fraction above clip=1 |
|---|---:|---:|---:|
| 64 | 1730 | 1.080 | 51.4% |
| 128 | 896 | 1.197 | 84.2% |
| 256 | 281 | 1.413 | 94.7% |
| full 512 | 93 | 1.474 | 96.8% |
| Mamba | 3000 | 0.571 | 1.9% |

This proves that the current optimizer sees horizon-dependent raw gradients but
maps nearly every long-horizon update onto the same clipping radius. It does
not by itself prove that clipping causes the quality gap, but it is the
strongest measured training-protocol mismatch.

Checkpoint-level gradients on the same four 512-token validation sequences
make the causal interpretation narrower:

| Credit horizon | Gradient norm | Clip scale at 1.0 |
|---|---:|---:|
| 64 | 4.292 | 0.233 |
| 128 | 4.561 | 0.219 |
| 256 | 4.853 | 0.206 |
| 512 | 5.087 | 0.197 |

Total-gradient cosine is `0.940` for 64 versus 512 and `0.993` for 256 versus
512. Minimum cross-horizon module cosines are approximately `0.91` for
injection, `0.92` for prior dynamics, `0.93` for initial state, `0.97` for
slot identity, and `1.00` for readout. These measurements use identical data
and deterministic ODE rollout.

Global norm clipping is a scalar rescale, and the measured gradients remain
strongly directionally aligned. The evidence therefore supports a modest
horizon-dependent effective-step-size mismatch, not a primary direction
conflict capable of explaining the full quality gap. Gradient clipping is
demoted from the leading root cause.

## Architectural Attribution

### Strong inference: insufficient hierarchical transformation depth

UDLF concentrates about `38.3M` non-vocabulary parameters into one recurrent
latent field. The same prior core is reused over time and solver substeps.
Mamba distributes its core across 12 independently parameterized residual
layers. The flat position-wise loss gap, including the first 64 tokens, is more
consistent with missing feature hierarchy than with insufficient persistent
memory.

This remains an inference until tested by adding independently parameterized
latent depth at matched parameter count.

### Confirmed engineering disadvantage: serial recurrent execution

UDLF executes a Python token loop. Each token performs observation injection,
two solver evaluations, latent attention/MLP operations, and readout. Mamba's
selective scan uses a fused CUDA path. Sustained throughput is about
`2.3k tok/s` versus `44.7k tok/s`, a roughly 19x gap. This cannot be closed by
batch tuning alone; the recurrent cell needs compilation/fusion or a different
parallel formulation.

Matched batch-2, length-128 forward/backward profiling measured:

| Metric | UDLF | Mamba |
|---|---:|---:|
| profiler wall time | 8.874 s | 0.194 s |
| profiler tokens/s | 28.85 | 1319.16 |
| operator calls | 417,059 | 10,538 |
| peak reserved | 1538 MB | 1002 MB |

The profiler wall-time ratio is `45.7x` and operator-call ratio is `39.6x`.
Profiler overhead exaggerates the production throughput ratio, but the launch
structure is unambiguous. UDLF issued `56,962` `cudaLaunchKernel` calls in one
profiled step. Major operation counts include `11,407` multiplies, `11,219`
copies, `8,914` in-place adds, `5,519` matrix multiplications, and `1,926`
batch matrix multiplications. Mamba used 12 fused selective-scan forward and
backward calls for its 12 layers.

There is no single dominant slow GEMM to optimize. The primary performance
unit must be the complete recurrent token/solver cell; optimizing individual
elementwise operations will not remove the launch-order bottleneck.

### Rejected primary explanations

- slot collapse: repaired rank remains about `11.3/16` and interventions prove
  specialization is used;
- insufficient token count: UDLF saw about 1.78x as many tokens;
- diffusion robustness benefit: ODE is marginally better;
- purely long-context failure: the gap is already large in positions 0-64;
- too few core parameters: corrected non-vocabulary capacity is `38.29M`.

## Next Falsifiable Experiments

1. Gradient protocol ablation at matched tokens: clip 1, clip 4, and an
   horizon-aware rule. Report clipping rate and fixed-sample loss.
2. Matched-parameter latent-depth ablation: split the prior core into multiple
   independent residual latent blocks without increasing total parameters.
3. ODE training ablation: remove diffusion sampling and its random-number path
   entirely, not only at evaluation.
4. CUDA profiler attribution followed by fusion of the recurrent cell. Require
   numerical parity before measuring throughput.

Another blind 3000-step run is not justified until at least one of these
experiments moves the fixed-sample loss in the predicted direction.

## Implemented Architecture Candidate

The first repair candidate now directly tests the leading attribution:

- `prior_depth=4` independent latent interaction blocks;
- width reduced from 792 to 488 to preserve the 64M scale;
- one solver update with a four-block deep vector field;
- ODE training, removing the diffusion path that showed no checkpoint benefit;
- block deltas accumulated with `1/depth` residual averaging.
- deep block output projections initialized at `1/sqrt(depth)` scale.

The resulting model has `64,523,673` parameters. The legacy depth-one path and
checkpoint key surface are unchanged.

An initial implementation applied RMS normalization to the final deep feature
before constructing drift. It passed short horizons but produced full-512
gradient norms from `49` to `1319` and was rejected. Direct scaled-delta
accumulation reduced local full-512 gradient norm to `3.85`; remote smoke then
measured finite full-512 norms of `7.08-7.25`, slot rank `13.7-14.9`, and peak
reserved memory `4.43GB`.

This proves implementation stability, not quality superiority. The next
decision point is a 1000-step matched-data quality gate with fixed-sample
evaluation every 250 steps.

The four-block width-488 candidate was subsequently rejected: it remained
`0.06-0.23` loss worse than depth one across every 50-step window from steps
100-299 and imposed substantially more launch overhead. A two-core width-640
candidate was also rejected before remote training because reducing latent
width alone increased matched full-512 initial gradient norm from `3.99` to
`68.8`.

The final candidate therefore preserves the validated width-792 shared core
and adds a separate zero-output-initialized rank-64 residual adapter to each of
the two solver half-steps. It has `64,330,065` parameters and is functionally
identical to the repaired depth-one model at initialization. After six remote
smoke steps, adapter output-weight cosine was `0.368` with difference norm
`0.0194`, proving that the half-step corrections receive distinct updates.

That adapter candidate was later rejected at step 64 because it induced early
common-state collapse and worsened matched-step loss. A subsequent fidelity
audit found a more direct defect in the original readout: v0.6 specifies a
separate key projection `W_{r,h}` for every readout head, but the implementation
shared one `d -> d` key projection across all eight heads. On the repaired
depth-one checkpoint, readout attention pair cosine was `0.428` but the eight
head outputs had effective rank only `2.24/8`, so the nominally large readout
used few independent value mixtures.

The head-specific-key candidate reallocates exactly `7d^2` weights from prior
FF width (`4 -> 3`) into seven additional readout key projections. It has
`64,024,353` parameters versus `64,025,937` for the control. Remote smoke
covered 64/128/256/full-512 paths without slot collapse; six-step head output
rank was `4.47/8`. This is implementation and geometry evidence only. A
300-step matched training gate must still establish quality.

## Ranked Root Causes

1. **Architectural depth and parameter organization.** Strong inference. A
   single shared recurrent field plus a `12.17M` readout is being asked to
   replace a 12-layer hierarchy. The uniform early-to-late loss gap supports a
   local representation deficit.
2. **Fragmented serial execution.** Confirmed performance cause. The complete
   recurrent cell generates roughly 40x as many operator calls as Mamba in the
   matched profiler.
3. **Unproductive diffusion.** Confirmed at the final checkpoint. It adds
   stochastic execution but no measurable evaluation gain.
4. **Horizon-dependent clipping.** Confirmed but secondary. The 64-512
   gradients remain strongly aligned; clipping mainly changes effective step
   size by about 16 percent across those horizons.

The evidence does not support another width increase, more training tokens, or
more aggressive GPU scheduling as primary remedies.
