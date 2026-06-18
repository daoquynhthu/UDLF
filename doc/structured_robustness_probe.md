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

## Results

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

The batch mixed-state probe becomes consistently positive from alpha `0.10`
upward and grows strongly with alpha. At alpha `0.05`, seed `901` is slightly
negative, so even the batch-mix family should not use a threshold that treats
near-zero deltas as decisive.

The temporal mixed-state probe is not stable: seed `901` is negative for every
tested alpha. This is stronger evidence that the current robustness blocker is
real, not merely an artifact of raw Gaussian perturbation. The next gate should
not be a blanket nonnegative check over every structured probe; it needs
per-probe thresholds and a rationale for which structured families are expected
to be destructive.
