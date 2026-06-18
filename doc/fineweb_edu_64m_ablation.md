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
