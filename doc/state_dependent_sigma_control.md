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
- Seeds: `904`, `906`
- Max steps: `600`
- CRN eval: `--pair-trials 4 --mix-alpha 0.2`
- Console mode: `quiet`

## Results

| setting | eval | zero | swapped | shifted | inverted | perturb | perturb 95% CI | batch mix | batch mix 95% CI | temporal mix | temporal 95% CI | core |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sigma_max=0.02` | 7.8460 | +0.9429 | +0.1886 | +0.0038 | +0.7039 | +0.0072 | [+0.0028, +0.0115] | +0.0375 | [+0.0328, +0.0421] | -0.0013 | [-0.0047, +0.0021] | fail |
| `sigma_max=0.01` | 7.8928 | +2.4377 | +0.1458 | +0.1047 | +0.5989 | +0.0097 | [+0.0096, +0.0099] | +0.0335 | [+0.0315, +0.0356] | +0.0019 | [-0.0004, +0.0041] | pass |

Full matrix CSV: `doc/state_dependent_sigma_matrix.csv`.

| seed | sigma max | eval | shifted | perturb | batch mix | temporal mix | core |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 904 | 0.005 | 7.7663 | +0.0890 | +0.0003 | +0.0098 | +0.0060 | pass |
| 904 | 0.010 | 7.7677 | +0.0300 | +0.0031 | +0.0252 | +0.0003 | pass |
| 904 | 0.015 | 7.8463 | +0.0289 | +0.0321 | +0.0069 | -0.0010 | pass |
| 904 | 0.020 | 7.7770 | +0.0531 | +0.0030 | -0.0077 | +0.0011 | pass |
| 906 | 0.005 | 7.8996 | +0.0501 | -0.0000 | +0.0272 | +0.0018 | pass |
| 906 | 0.010 | 7.8928 | +0.1047 | +0.0097 | +0.0335 | +0.0019 | pass |
| 906 | 0.015 | 7.7751 | -0.0114 | -0.0293 | +0.0167 | -0.0153 | fail |
| 906 | 0.020 | 7.8460 | +0.0038 | +0.0072 | +0.0375 | -0.0013 | fail |

## Dynamics

| setting | train eval | sigma min | sigma max | sigma RMS | drift RMS | jump RMS | grad norm | state RMS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sigma_max=0.02` | 7.6768 | 0.0001001 | 0.0200195 | 0.0136530 | 0.1587440 | 0.0402881 | 9.1226 | 1.1289 |
| `sigma_max=0.01` | 7.7672 | 0.0001001 | 0.0100098 | 0.0070366 | 0.1564265 | 0.0392726 | 12.2228 | 1.1290 |

## Current Read

The matrix argues against immediately rewriting the state-dependent path, but
it is not a reliable global conclusion. It only shows that, at the current
training horizon and model scale, amplitude is a real control variable: seed
`906` passes the core gate at `sigma_max` `0.005` and `0.010`, then fails at
`0.015` and `0.020`.

It does not close the robustness issue. Seed `904` passes the core gate at all
tested amplitudes, but batch-mix and temporal-mix remain inconsistent. Seed
`906` is cleanest at `0.010`: core passes, perturb is positive with a tight
paired interval, and batch-mix is positive, but temporal-mix is still near
zero.

The result may interact with:

- training horizon: a higher `sigma_max` might need more steps to stabilize,
  or it might destabilize more clearly later;
- architecture scale: latent width, slot count, and readout capacity may change
  whether stochasticity is useful or destructive;
- optimizer schedule: the current 600-step setting is still a short diagnostic,
  not a converged training regime;
- task/data mix: real-token query recall is a diagnostic task, not a full
  language-modeling claim.

The next state-dependent candidate can use `sigma_max=0.01` only as a local
probe setting. It should not replace fixed K=4 until it passes cross-seed,
cross-horizon, and at least one scale check under the CRN core and robustness
checks.
