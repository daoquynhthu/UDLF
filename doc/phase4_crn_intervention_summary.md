# CRN Diffusion Intervention Summary

This summary is generated from read-only checkpoint intervention evaluation with common random numbers.

Evaluation settings:

- Checkpoints: query-recall diffusion ablation seeds `710-713`.
- Pair trials: `4` suffix Brownian paths per run.
- Perturb trials: `16` state-noise samples per suffix path.
- Mixed alpha: `0.2`.

## Group Means

| mode | runs | seeds | eval | zero | swap | shift | mix | temporal | perturb | atten | invert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 4 | 710,711,712,713 | 3.970 | 1.829 | 1.439 | 0.592 | 0.022 | 0.037 | 0.031 | 0.001 | 8.385 |
| ode | 4 | 710,711,712,713 | 4.040 | 1.798 | 1.161 | 0.557 | 0.018 | 0.037 | 0.001 | 0.001 | 8.365 |
| state_dependent | 4 | 710,711,712,713 | 3.899 | 1.750 | 1.592 | 0.621 | 0.022 | 0.043 | 0.036 | 0.003 | 8.828 |

## Current Read

- Under CRN, state-dependent diffusion has the strongest mean perturbation delta in this matched set and stays positive on every seed.
- Fixed diffusion is also positive on average, but one seed is near zero and its paired interval can overlap zero.
- ODE perturbation deltas are near zero, as expected because suffix rollouts are deterministic and only the perturbation noise changes state.
- Fixed diffusion remains the simpler smoke/default candidate, but it is no longer the strongest robustness candidate after CRN re-evaluation.
- This does not close the broader robustness blocker; it only repairs the diffusion comparison that previously used unpaired suffix paths.

Detail runs summarized: 12
