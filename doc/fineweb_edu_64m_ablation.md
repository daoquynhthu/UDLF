# FineWeb-Edu 64M Ablation

This phase replaces the small state-intervention diagnostics as the main
architecture decision path.

## Dataset

- Local path: `E:/NAIME_DATA/datasets/fineweb_edu_1b_ctx1024`
- Format: Hugging Face `load_from_disk`
- Columns: `input_ids`, `attention_mask`, `labels`
- Train rows: `975610`
- Validation rows: `8908`
- Token row length: `1025`

The 3000-step ablation uses `input_ids` with `seq_len=512`.

## Models

| model | config | target params | measured params |
| --- | --- | --- | --- |
| UDLF LLM | `configs/training_templates/udlf_fineweb_edu_64m_3000.json` | 64M | ~68.1M |
| Mamba baseline | `configs/training_templates/mamba_fineweb_edu_64m_3000.json` | 64M | ~63.7M |

The Mamba baseline is a pure PyTorch selective-state-space implementation. It
uses the standard Mamba projection structure but not fused `mamba_ssm` kernels,
because `mamba_ssm`, `causal_conv1d`, and `triton` are not installed locally.

## Run Policy

- Console output is quiet by default.
- Metrics, logs, checkpoints, and configs are written under ignored `runs/`.
- Local RTX 5060 sanity uses micro-batch `1` with gradient accumulation `16`.
- Remote RTX 4090 runs use automatic micro-batch probing. The configured
  `batch_size * grad_accum_steps` is treated as the effective target, and the
  real micro-batch is selected at launch from measured CUDA memory use.
- UDLF uses `latent_dim=512` with an untied output head. This keeps total
  parameters near the 64M target without pushing the latent core width to the
  point where it cannot fit the local 8GB GPU.
- Evaluation runs every `250` steps.
- Latest checkpoints are written every `100` steps; full save cadence is `500`.
- Intervention probes are disabled for this phase.

## Commands

```powershell
$env:PYTHONPATH='src'
python -m udlf.training.train --config configs\training_templates\udlf_fineweb_edu_64m_3000.json
python -m udlf.training.train --config configs\training_templates\mamba_fineweb_edu_64m_3000.json
```

## Acceptance

The first comparison table should include:

- train loss and validation loss at steps `250..3000`;
- perplexity;
- tokens/second;
- CUDA memory;
- parameter count;
- checkpoint path;
- any NaN/overflow/failed-run checkpoint.

## Local Sanity

| model | params | step | train loss | eval loss | tok/s | CUDA memory MB | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UDLF | 68.1M | 1 | 10.8562 | 10.8556 | 52.814 | 1818.312 | pass |
| Mamba | 63.7M | 1 | 10.9584 | 10.9912 | 89.969 | 2573.117 | pass |

The UDLF sanity requires true segmented backward. Earlier segmented forward
still retained all segment graphs until the end of the sequence and OOMed on
the local 8GB GPU. The trainer now backprops each UDLF segment immediately
when `detach_state_between_segments=true`.

At these measured throughputs, 3000 local steps are multi-day runs. Launching
them is valid, but results should be monitored from `runs/*/metrics.jsonl`
rather than expected in the same interactive turn.

## Remote 4090 Sanity

Remote data path:

```text
L:/NAIME_REMOTE/datasets/fineweb_edu_1b_ctx1024
```

The remote run uses the isolated UDLF workspace service. Checkpoints and
metrics are under `L:\UDLF_REMOTE\runs`, not under the remote NAIME repository.

| model | remote run | params | step | train loss | eval loss | tok/s | CUDA memory MB | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UDLF | `udlf_fineweb_edu_64m_remote_sanity` | 68.1M | 1 | 10.8565 | 10.8625 | 42.691 | 1818.312 | link pass only |
| Mamba | `mamba_fineweb_edu_64m_remote_sanity` | 63.7M | 1 | 10.9584 | 10.9560 | 61.787 | 2573.117 | link pass only |

These runs only validate remote execution. They are not acceptable performance
baselines because they used small-GPU scheduling and included step-1
eval/checkpoint overhead. The next remote gate is a short auto-batch sanity run
that records selected batch, accumulation, memory, and normal training
tokens/second before any 3000-step launch.

## Local Solver-Step Quality Gate

After profiling showed recurrent solver dispatch as the dominant UDLF
throughput limiter, the UDLF 64M template was changed from `solver_steps=4` to
`solver_steps=2`. This changes integration granularity, so the first local
quality gate compares the two settings with the same data path, seed,
micro-batch, and evaluation cadence.

Configuration:

- Dataset: `E:/NAIME_DATA/datasets/fineweb_edu_1b_ctx1024`
- Model: UDLF 64M config, `seq_len=512`, fixed diffusion
- Batch: `24`, grad accumulation `1`
- Steps: `20`
- Eval: every `10` steps, `2` batches
- Diagnostics: stability diagnostics every `10` steps; dynamics diagnostics off
- Checkpoint cadence: latest/save disabled; best checkpoint written only on eval

| solver | step | train loss | eval loss | eval ppl | grad norm | state RMS | inj fd gain | drift fd gain | FTLE proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 10 | 10.79 | 10.77 | 47588.94 | 0.88 | 0.85 | 204.30 | 0.26 | -0.25 |
| 4 | 20 | 10.40 | 10.36 | 31625.71 | 1.66 | 0.86 | 200.92 | 0.26 | -0.25 |
| 2 | 10 | 10.78 | 10.77 | 47432.44 | 0.90 | 0.85 | 204.30 | 0.26 | -0.26 |
| 2 | 20 | 10.40 | 10.37 | 31833.89 | 1.66 | 0.86 | 201.07 | 0.26 | -0.26 |

No NaN or Inf values were found in either metrics file. On this short local
gate, solver `2` matches solver `4` on early loss trajectory and the sampled
stability diagnostics while preserving the throughput gain. This is not yet a
full equivalence claim; it is enough to keep solver `2` as the current
performance configuration for the next remote short-run gate.
