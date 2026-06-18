# Real-Token State-Dependent CRN Checks

This document tracks local RTX 5060 medium runs for state-dependent K=4
real-token query recall.

These runs are intended to test whether the stronger synthetic CRN
state-dependent result transfers to real-token query recall. They do not
replace the fixed K=4 smoke candidate yet.

## Settings

- Data task: `real_query_recall`
- Solver steps: `K=4`
- Diffusion mode: `state_dependent`
- Max steps: `600`
- CRN eval: `--pair-trials 4 --mix-alpha 0.2`
- Console mode for new training runs: `quiet`

## Results

| seed | eval | zero | swapped | shifted | inverted | perturb | perturb 95% CI | batch mix | batch mix 95% CI | temporal mix | temporal 95% CI | core |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 903 | 8.0441 | +2.0290 | +0.0959 | +0.0599 | +0.4432 | +0.0309 | [+0.0279, +0.0338] | +0.0181 | [+0.0173, +0.0190] | +0.0048 | [+0.0020, +0.0076] | pass |
| 904 | 7.7770 | +3.5729 | +0.2056 | +0.0531 | +0.7786 | +0.0030 | [-0.0018, +0.0078] | -0.0077 | [-0.0106, -0.0047] | +0.0011 | [-0.0018, +0.0039] | pass |

## Current Read

Both state-dependent real-token runs pass the core gate. Robustness is mixed:
seed `903` supports the synthetic CRN result, while seed `904` does not. The
batch-mix failure on seed `904` is significant under the paired interval, so
state-dependent diffusion cannot yet replace fixed K=4 as the default. It
remains a robustness candidate that needs more seeds and possibly a better
structured gate.
