from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from udlf.llm import MambaLMModel
from udlf.model import UDLFStageAModel
from udlf.attribution import position_bins, training_summary, transform_states
from udlf.training.checkpoint import normalize_state_dict_for_model
from udlf.training.config import train_config_from_dict


def load_checkpoint(path: Path, *, ode: bool = False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = train_config_from_dict(checkpoint["config"])
    if config.architecture == "udlf":
        model_config = replace(config.model_config(), diffusion_mode="ode") if ode else config.model_config()
        model = UDLFStageAModel(model_config)
    else:
        model = MambaLMModel(config.mamba_config())
    state_dict = normalize_state_dict_for_model(model, checkpoint["model"])
    if config.architecture == "udlf" and "slot_identity" not in state_dict:
        state_dict["slot_identity"] = torch.zeros_like(model.slot_identity)
    model.load_state_dict(state_dict)
    return model.cuda().eval(), config


def token_losses(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")


@torch.no_grad()
def rollout_states(model: UDLFStageAModel, prefix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    state = model.init_state(prefix.shape[0], device=prefix.device, dtype=model.embedding.weight.dtype)
    identity = model.slot_identity_features()
    states = []
    embeds = []
    for position in range(prefix.shape[1]):
        token = model.embedding(prefix[:, position])
        state = model.inject(state, token, identity)
        state = model.prior.euler_maruyama(state, token, slot_identity=identity)
        states.append(state)
        embeds.append(token)
    return torch.stack(states, dim=1), torch.stack(embeds, dim=1)


@torch.no_grad()
def evaluate_udlf_modes(
    model: UDLFStageAModel,
    rows: list[list[int]],
    batch_size: int,
) -> tuple[dict[str, float], torch.Tensor]:
    modes = ["actual", "mean_slots", "centered_slots", "shuffled_slots", "no_identity_readout"]
    totals = {mode: 0.0 for mode in modes}
    count = 0
    actual_position_losses = []
    for start in range(0, len(rows), batch_size):
        batch = torch.tensor(rows[start : start + batch_size], dtype=torch.long, device="cuda")
        prefix, targets = batch[:, :-1], batch[:, 1:]
        states, embeds = rollout_states(model, prefix)
        for mode in modes:
            identity = torch.zeros_like(model.slot_identity_features()) if mode == "no_identity_readout" else model.slot_identity_features()
            logits = model.readout(transform_states(states, mode), embeds, model.output_weight, identity)
            losses = token_losses(logits, targets)
            totals[mode] += float(losses.sum().cpu())
            if mode == "actual":
                actual_position_losses.append(losses.detach().cpu())
            del logits, losses
        count += targets.numel()
        del batch, prefix, targets, states, embeds
        torch.cuda.empty_cache()
    means = {mode: value / count for mode, value in totals.items()}
    return means, torch.cat(actual_position_losses, dim=0)


@torch.no_grad()
def evaluate_mamba_positions(model: MambaLMModel, rows: list[list[int]], batch_size: int) -> torch.Tensor:
    position_losses = []
    for start in range(0, len(rows), batch_size):
        batch = torch.tensor(rows[start : start + batch_size], dtype=torch.long, device="cuda")
        output = model(batch)
        position_losses.append(token_losses(output.logits, batch[:, 1:]).detach().cpu())
        del batch, output
        torch.cuda.empty_cache()
    return torch.cat(position_losses, dim=0)


def load_metrics(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--udlf", type=Path, required=True)
    parser.add_argument("--mamba", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--udlf-metrics", type=Path, required=True)
    parser.add_argument("--mamba-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--position-bin", type=int, default=64)
    args = parser.parse_args()

    udlf, udlf_config = load_checkpoint(args.udlf, ode=True)
    mamba, mamba_config = load_checkpoint(args.mamba)
    dataset = load_from_disk(str(args.data))[udlf_config.validation_split]
    rows = [dataset[index][udlf_config.data_column][: udlf_config.seq_len + 1] for index in range(args.sequences)]

    mode_losses, udlf_positions = evaluate_udlf_modes(udlf, rows, args.batch_size)
    del udlf
    torch.cuda.empty_cache()
    mamba_positions = evaluate_mamba_positions(mamba, rows, args.batch_size)
    report = {
        "sequences": len(rows),
        "seq_len": udlf_config.seq_len,
        "udlf_readout_state_modes": mode_losses,
        "udlf_mode_deltas": {mode: loss - mode_losses["actual"] for mode, loss in mode_losses.items()},
        "position_bins": position_bins(udlf_positions, mamba_positions, args.position_bin),
        "overall": {
            "udlf_loss": float(udlf_positions.mean()),
            "mamba_loss": float(mamba_positions.mean()),
            "udlf_minus_mamba": float(udlf_positions.mean() - mamba_positions.mean()),
        },
        "udlf_training": training_summary(load_metrics(args.udlf_metrics), udlf_config.seq_len),
        "mamba_training": training_summary(load_metrics(args.mamba_metrics), udlf_config.seq_len),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
