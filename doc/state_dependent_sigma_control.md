# State-Dependent Sigma Control

This document tracks a targeted local control after real-token
state-dependent K=4 seed `906` failed the core shifted-state gate with the
default `sigma_max=0.02`.

## Question

Does the seed `906` core failure look like a general state-dependent
parameterization failure, or is it sensitive to the diffusion amplitude range?

## Settings

- Data task: `real_query_recall`
- Solver steps: `K=4`
- Diffusion mode: `state_dependent`
- Seed: `906`
- Max steps: `600`
- CRN eval: `--pair-trials 4 --mix-alpha 0.2`
- Console mode: `quiet`

## Results

| setting | eval | zero | swapped | shifted | inverted | perturb | perturb 95% CI | batch mix | batch mix 95% CI | temporal mix | temporal 95% CI | core |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sigma_max=0.02` | 7.8460 | +0.9429 | +0.1886 | +0.0038 | +0.7039 | +0.0072 | [+0.0028, +0.0115] | +0.0375 | [+0.0328, +0.0421] | -0.0013 | [-0.0047, +0.0021] | fail |
| `sigma_max=0.01` | 7.8928 | +2.4377 | +0.1458 | +0.1047 | +0.5989 | +0.0097 | [+0.0096, +0.0099] | +0.0335 | [+0.0315, +0.0356] | +0.0019 | [-0.0004, +0.0041] | pass |

## Dynamics

| setting | train eval | sigma min | sigma max | sigma RMS | drift RMS | jump RMS | grad norm | state RMS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sigma_max=0.02` | 7.6768 | 0.0001001 | 0.0200195 | 0.0136530 | 0.1587440 | 0.0402881 | 9.1226 | 1.1289 |
| `sigma_max=0.01` | 7.7672 | 0.0001001 | 0.0100098 | 0.0070366 | 0.1564265 | 0.0392726 | 12.2228 | 1.1290 |

## Current Read

This is a useful control, not a closed decision. The same seed that failed
with `sigma_max=0.02` passes the core gate when the state-dependent diffusion
range is capped at `0.01`. Perturbation and batch-mix stay positive under CRN,
while temporal-mix remains too close to zero.

The immediate next experiment should be a small sigma-range matrix, not a full
architecture rewrite. Candidate settings:

- `sigma_max=0.005`
- `sigma_max=0.01`
- `sigma_max=0.015`
- `sigma_max=0.02`

Use at least seeds `904` and `906`, because those are currently the clearest
negative/fragile real-token cases. A replacement state-dependent default should
not be considered until this matrix passes both the core gate and CRN
robustness checks on the fragile seeds.
