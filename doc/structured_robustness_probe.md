# Structured Robustness Probe

This document tracks the first structured robustness diagnostic for Stage A
fixed K=4 real-token query recall.

The goal is narrow: test whether structured moves in latent-state space damage
recall more consistently than raw Gaussian state noise or attenuation. This
does not close the broader robustness blocker by itself.

## Probe

For a batch of latent states `state`, the batch mixed-state intervention
evaluates:

```text
mixed_state = (1 - alpha) * state + alpha * state.flip(0)
```

The temporal mixed-state intervention evaluates:

```text
temporal_mixed_state = (1 - alpha) * state + alpha * shifted_state
```

where `shifted_state` is produced from a shorter context using the configured
`intervention_shift_tokens`.

This is distinct from:

- `swapped`: fully replaces each state with another sample's state.
- `perturbed`: adds isotropic random noise.
- `attenuated`: scales the state by `0.5`.

The diagnostic is implemented in:

- `src/udlf/training/train.py` as `intervention_mixed_delta` and
  `intervention_temporal_mixed_delta`.
- `scripts/evaluate_state_interventions.py` for read-only checkpoint eval.
- `scripts/check_state_probe.py --profile structured`, with split profiles
  `structured-batch` and `structured-temporal` for per-probe checks.

## Fixed K=4 Real-Token Sweep

Source checkpoints:

- `runs/udlf_real_token_query_recall_fixed_k4_seed900/latest.pt`
- `runs/udlf_real_token_query_recall_fixed_k4_seed901/latest.pt`
- `runs/udlf_real_token_query_recall_fixed_k4_seed902/latest.pt`

Command shape:

```powershell
$env:PYTHONPATH='src'
python scripts\evaluate_state_interventions.py `
  --run-dir runs/udlf_real_token_query_recall_fixed_k4_seed900 `
  --eval-batches 2 `
  --mix-alpha 0.2 `
  --output artifacts\structured_interventions\real_token_fixed_k4_seed900_alpha0p2.json
python scripts\check_state_probe.py `
  artifacts\structured_interventions\real_token_fixed_k4_seed900_alpha0p2.json `
  --profile structured
```

All outputs are under ignored `artifacts/structured_interventions/`.

The current evaluator uses common random numbers for suffix rollouts: each
candidate state is evaluated against the same paired Brownian suffix paths and
reports paired mean, standard error, and 95 percent confidence interval.

Earlier non-CRN mixed-alpha results should be treated as historical only. They
used different Brownian suffix paths for different candidate states, so small
deltas were not clean.

## CRN Results at Alpha 0.2

These runs used `--pair-trials 4 --eval-batches 1 --mix-alpha 0.2`.

| seed | batch mixed delta | 95% CI | temporal mixed delta | 95% CI | perturbed delta | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| 900 | +0.031961 | [+0.031461, +0.032462] | +0.005914 | [+0.005136, +0.006693] | -0.009939 | [-0.010938, -0.008940] |
| 901 | +0.022900 | [+0.020536, +0.025264] | -0.002878 | [-0.005239, -0.000518] | -0.027785 | [-0.031089, -0.024481] |
| 902 | +0.009607 | [+0.008504, +0.010709] | +0.000177 | [-0.001709, +0.002062] | -0.007523 | [-0.008165, -0.006881] |

## Legacy Non-CRN Sweep

| seed | alpha | batch mixed delta | temporal mixed delta | perturbed delta | attenuated delta |
| --- | --- | --- | --- | --- | --- |
| 900 | 0.05 | +0.003135 | -0.000238 | -0.001383 | -0.001946 |
| 900 | 0.10 | +0.011077 | +0.001477 | -0.001383 | -0.001946 |
| 900 | 0.20 | +0.046580 | +0.004907 | -0.001383 | -0.001946 |
| 900 | 0.40 | +1.020831 | +0.017347 | -0.001383 | -0.001946 |
| 901 | 0.05 | -0.002856 | -0.005522 | -0.019245 | -0.005871 |
| 901 | 0.10 | +0.001940 | -0.010490 | -0.019245 | -0.005871 |
| 901 | 0.20 | +0.021885 | -0.011231 | -0.019245 | -0.005871 |
| 901 | 0.40 | +0.388951 | -0.011348 | -0.019245 | -0.005871 |
| 902 | 0.05 | +0.001835 | +0.002648 | -0.005330 | +0.004276 |
| 902 | 0.10 | +0.003709 | +0.001874 | -0.005330 | +0.004276 |
| 902 | 0.20 | +0.014127 | +0.002345 | -0.005330 | +0.004276 |
| 902 | 0.40 | +0.135034 | +0.008442 | -0.005330 | +0.004276 |

Aggregate by alpha:

| alpha | mean batch mixed | min batch mixed | mean temporal mixed | min temporal mixed |
| --- | --- | --- | --- | --- |
| 0.05 | +0.000705 | -0.002856 | -0.001038 | -0.005522 |
| 0.10 | +0.005575 | +0.001940 | -0.002380 | -0.010490 |
| 0.20 | +0.027531 | +0.014127 | -0.001326 | -0.011231 |
| 0.40 | +0.514939 | +0.135034 | +0.004813 | -0.011348 |

## Current Read

The CRN alpha `0.2` results are now the clean reference. Batch-mix remains
positive across seeds with tight paired confidence intervals. Perturbed state
is negative across all three seeds under the same common suffix noise, so it
should not be used as evidence for robustness. Temporal-mix is not stable:
seed `901` is significantly negative and seed `902` is indistinguishable from
zero at this sample size.

The batch mixed-state probe becomes consistently positive from alpha `0.10`
upward and grows strongly with alpha. At alpha `0.05`, seed `901` is slightly
negative, so even the batch-mix family should not use a threshold that treats
near-zero deltas as decisive.

The next gate should not be a blanket nonnegative check over every structured
probe; it needs per-probe thresholds, CRN paired statistics, and a rationale for
which structured families are expected to be destructive.
