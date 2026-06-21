from __future__ import annotations

from typing import Any

import torch


def udlf_parameter_breakdown(model) -> dict[str, Any]:
    embedding = sum(parameter.numel() for parameter in model.embedding.parameters())
    output_shared = model.output_weight is model.embedding.weight
    output_additional = 0 if output_shared else model.output_weight.numel()
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "total": total,
        "embedding": embedding,
        "output_shared_with_embedding": output_shared,
        "output_additional": output_additional,
        "initial_state": model.initial_state.numel(),
        "slot_identity": model.slot_identity.numel(),
        "inject": sum(parameter.numel() for parameter in model.inject.parameters()),
        "prior": sum(parameter.numel() for parameter in model.prior.parameters()),
        "readout": sum(parameter.numel() for parameter in model.readout.parameters()),
        "core_without_vocab_matrices": total - embedding - output_additional,
    }


def transform_states(states: torch.Tensor, mode: str) -> torch.Tensor:
    mean = states.mean(dim=-2, keepdim=True)
    if mode == "actual" or mode == "no_identity_readout":
        return states
    if mode == "mean_slots":
        return mean.expand_as(states)
    if mode == "centered_slots":
        return states - mean
    if mode == "shuffled_slots":
        return states.roll(1, dims=-2)
    raise ValueError(f"unknown state mode: {mode}")


def position_bins(udlf: torch.Tensor, mamba: torch.Tensor, width: int) -> list[dict[str, float | int]]:
    bins = []
    for start in range(0, udlf.shape[1], width):
        end = min(start + width, udlf.shape[1])
        udlf_loss = float(udlf[:, start:end].mean())
        mamba_loss = float(mamba[:, start:end].mean())
        bins.append(
            {
                "start": start,
                "end": end,
                "udlf_loss": udlf_loss,
                "mamba_loss": mamba_loss,
                "udlf_minus_mamba": udlf_loss - mamba_loss,
            }
        )
    return bins


def training_summary(rows: list[dict[str, Any]], seq_len: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    total_sequences = 0.0
    for row in rows:
        horizon = str(int(row.get("train_segment_len", -1)))
        groups.setdefault(horizon, []).append(row)
        total_sequences += float(row.get("train_step_effective_batch_size", row.get("effective_batch_size", 0)))
    by_horizon = {}
    for horizon, values in groups.items():
        grads = [float(row["grad_norm"]) for row in values]
        by_horizon[horizon] = {
            "steps": len(values),
            "mean_grad_norm": sum(grads) / len(grads),
            "clip_fraction": sum(value > 1.0 for value in grads) / len(grads),
            "mean_step_seconds": sum(float(row.get("step_seconds", 0)) for row in values) / len(values),
            "mean_step_tokens_per_second": sum(float(row.get("step_tokens_per_second", 0)) for row in values) / len(values),
        }
    return {
        "steps": len(rows),
        "estimated_training_tokens": int(total_sequences * (seq_len - 1)),
        "by_horizon": by_horizon,
    }
