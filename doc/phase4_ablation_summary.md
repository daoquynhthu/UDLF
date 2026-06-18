# Phase 4 Query Recall Ablation Summary

This summary is generated from local run metrics. It excludes checkpoints and run artifacts.

## Group Means

| mode | K | runs | seeds | eval | tok/s | zero | swap | shift | mix | temporal mix | perturb | atten | invert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 1 | 3 | 721,722,723 | 4.025 | 11961 | 1.230 | 1.186 | 0.224 | n/a | n/a | 0.026 | 0.004 | 6.910 |
| fixed | 2 | 4 | 710,711,712,713 | 3.927 | 8143 | 1.845 | 1.361 | 0.568 | n/a | n/a | 0.039 | 0.001 | 8.536 |
| fixed | 4 | 3 | 721,722,723 | 3.858 | 5294 | 2.533 | 1.743 | 0.680 | n/a | n/a | 0.038 | 0.002 | 9.825 |
| ode | 2 | 4 | 710,711,712,713 | 4.012 | 8217 | 1.791 | 1.098 | 0.471 | n/a | n/a | 0.002 | 0.001 | 8.069 |
| state_dependent | 1 | 4 | 720,721,722,723 | 3.954 | 10861 | 1.666 | 1.469 | 0.341 | n/a | n/a | 0.024 | -0.001 | 7.198 |
| state_dependent | 2 | 5 | 710,711,712,713,720 | 3.876 | 7948 | 1.810 | 1.551 | 0.599 | n/a | n/a | 0.035 | 0.005 | 8.929 |
| state_dependent | 4 | 4 | 720,721,722,723 | 3.840 | 4685 | 2.159 | 1.703 | 0.707 | n/a | n/a | 0.033 | 0.007 | 10.320 |
| state_dependent | 8 | 1 | 720 | 3.817 | 2690 | 2.652 | 1.933 | 0.720 | n/a | n/a | 0.040 | 0.005 | 12.675 |

## Current Read

- All summarized groups pass the query-recall core state-causality gate.
- Fixed K=4 and state-dependent K=4 produce stronger intervention margins than K=1 variants.
- K=1 variants are much faster and still usable for cheap screening.
- Attenuation remains too small and inconsistent to use as a blocking robustness gate.
- Structured batch-mix and temporal-mix perturbations are tracked for new runs; older summarized runs may not contain them.
- Fixed K=4 is the current pragmatic default candidate for real-token confirmation because it is simpler than state-dependent diffusion and has strong synthetic margins.

Detail runs summarized: 28
