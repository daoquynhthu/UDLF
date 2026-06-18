# Structured Robustness Probe

This document tracks the first structured robustness diagnostic for Stage A
fixed K=4 real-token query recall.

The goal is narrow: test whether a small move toward another real latent state
damages recall more consistently than raw Gaussian state noise or attenuation.
This does not close the broader robustness blocker by itself.

## Probe

For a batch of latent states `state`, the mixed-state intervention evaluates:

```text
mixed_state = (1 - alpha) * state + alpha * state.flip(0)
```

This is distinct from:

- `swapped`: fully replaces each state with another sample's state.
- `perturbed`: adds isotropic random noise.
- `attenuated`: scales the state by `0.5`.

The diagnostic is implemented in:

- `src/udlf/training/train.py` as `intervention_mixed_delta`.
- `scripts/evaluate_state_interventions.py` for read-only checkpoint eval.
- `scripts/check_state_probe.py --profile structured`.

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

| seed | alpha | mixed delta | perturbed delta | attenuated delta |
| --- | --- | --- | --- | --- |
| 900 | 0.05 | +0.004780 | +0.000129 | -0.002078 |
| 900 | 0.10 | +0.012585 | +0.000129 | -0.002078 |
| 900 | 0.20 | +0.048568 | +0.000129 | -0.002078 |
| 900 | 0.40 | +1.023282 | +0.000129 | -0.002078 |
| 901 | 0.05 | +0.004267 | -0.016161 | -0.004255 |
| 901 | 0.10 | +0.006961 | -0.016161 | -0.004255 |
| 901 | 0.20 | +0.024500 | -0.016161 | -0.004255 |
| 901 | 0.40 | +0.387425 | -0.016161 | -0.004255 |
| 902 | 0.05 | +0.004185 | -0.005173 | +0.004883 |
| 902 | 0.10 | +0.005953 | -0.005173 | +0.004883 |
| 902 | 0.20 | +0.015532 | -0.005173 | +0.004883 |
| 902 | 0.40 | +0.135695 | -0.005173 | +0.004883 |

Aggregate by alpha:

| alpha | mean mixed delta | min mixed delta | max mixed delta |
| --- | --- | --- | --- |
| 0.05 | +0.004411 | +0.004185 | +0.004780 |
| 0.10 | +0.008500 | +0.005953 | +0.012585 |
| 0.20 | +0.029534 | +0.015532 | +0.048568 |
| 0.40 | +0.515467 | +0.135695 | +1.023282 |

## Current Read

The mixed-state probe is consistently positive across all three real-token
fixed K=4 seeds and all tested alphas. The effect grows with alpha, which is
the expected direction if the model is using the persistent state.

This is stronger evidence than the raw random perturbation and attenuation
metrics, which remain inconsistent on the same checkpoints. The robustness
blocker should stay open until the structured suite defines thresholds and
adds at least one more structured intervention family.
