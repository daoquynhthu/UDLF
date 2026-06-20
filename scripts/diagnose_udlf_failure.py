from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from udlf.model import UDLFStageAModel
from udlf.training.checkpoint import normalize_state_dict_for_model
from udlf.training.config import train_config_from_dict


def sequence_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.no_grad()
def segmented_loss(model, batch: torch.Tensor, mode: str, segment_len: int = 64) -> float:
    state = None
    losses = []
    for start in range(0, batch.shape[1] - 1, segment_len):
        end = min(start + segment_len, batch.shape[1] - 1)
        if mode == "reset":
            state = None
        elif mode == "shuffle" and state is not None:
            state = state.roll(1, dims=0)
        logits, state = model.forward_prefix(batch[:, start:end], state=state)
        losses.append(sequence_loss(logits, batch[:, start + 1 : end + 1]))
    return float(torch.stack(losses).mean().cpu())


@torch.no_grad()
def component_loss(model, batch: torch.Tensor, mode: str) -> float:
    state = model.init_state(batch.shape[0], device=batch.device, dtype=model.embedding.weight.dtype)
    states = []
    embeds = []
    for t in range(batch.shape[1] - 1):
        if mode == "stateless":
            state = model.init_state(batch.shape[0], device=batch.device, dtype=model.embedding.weight.dtype)
        token = model.embedding(batch[:, t])
        if mode != "prior_only":
            state = model.inject(state, token)
        if mode != "inject_only":
            state = model.prior.euler_maruyama(state, token)
        states.append(state)
        embeds.append(token)
    logits = model.readout(torch.stack(states, dim=1), torch.stack(embeds, dim=1), model.output_weight)
    return float(sequence_loss(logits, batch[:, 1:]).cpu())


def slot_statistics(state: torch.Tensor) -> dict[str, float]:
    normalized = F.normalize(state.float(), dim=-1)
    cosine = normalized @ normalized.transpose(-1, -2)
    slots = state.shape[1]
    off_diagonal = (cosine.sum(dim=(-1, -2)) - slots) / (slots * (slots - 1))
    centered = state.float() - state.float().mean(dim=1, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    participation = energy.sum(dim=-1).square() / energy.square().sum(dim=-1).clamp_min(1e-12)
    return {
        "slot_pair_cosine": float(off_diagonal.mean().detach().cpu()),
        "slot_centered_rms": float(centered.square().mean().sqrt().detach().cpu()),
        "slot_participation_rank": float(participation.mean().detach().cpu()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = train_config_from_dict(checkpoint["config"])
    model = UDLFStageAModel(config.model_config()).cuda().eval()
    model.load_state_dict(normalize_state_dict_for_model(model, checkpoint["model"]))

    dataset = load_from_disk(str(args.data))[config.validation_split]
    rows = [dataset[index][config.data_column][: config.seq_len + 1] for index in range(args.batch_size)]
    batch = torch.tensor(rows, dtype=torch.long, device="cuda")

    groups = {
        "embedding": sum(p.numel() for p in model.embedding.parameters()),
        "output": model.output_weight.numel(),
        "initial_state": model.initial_state.numel(),
        "inject": sum(p.numel() for p in model.inject.parameters()),
        "prior": sum(p.numel() for p in model.prior.parameters()),
        "readout": sum(p.numel() for p in model.readout.parameters()),
    }
    total = sum(p.numel() for p in model.parameters())

    stochastic_losses = []
    stochastic_state = None
    with torch.no_grad():
        for seed in range(8):
            generator = torch.Generator(device="cuda").manual_seed(9000 + seed)
            logits, stochastic_state = model.forward_prefix(batch[:, :-1], generator=generator)
            stochastic_losses.append(float(sequence_loss(logits, batch[:, 1:]).cpu()))

    ode_model = type(model)(replace(model.config, diffusion_mode="ode")).cuda().eval()
    ode_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        ode_logits, ode_state = ode_model.forward_prefix(batch[:, :-1])
        ode_loss = float(sequence_loss(ode_logits, batch[:, 1:]).cpu())

    stochastic_mean = sum(stochastic_losses) / len(stochastic_losses)
    stochastic_se = torch.tensor(stochastic_losses).std(unbiased=True).item() / math.sqrt(len(stochastic_losses))
    report = {
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "parameter_total": total,
        "parameter_groups": groups,
        "core_without_vocab_matrices": total - groups["embedding"] - groups["output"],
        "stochastic_loss_mean": stochastic_mean,
        "stochastic_loss_se": stochastic_se,
        "ode_loss": ode_loss,
        "ode_minus_stochastic": ode_loss - stochastic_mean,
        "carry_loss": segmented_loss(ode_model, batch, "carry"),
        "reset_every_64_loss": segmented_loss(ode_model, batch, "reset"),
        "shuffle_every_64_loss": segmented_loss(ode_model, batch, "shuffle"),
        "inject_only_loss": component_loss(ode_model, batch, "inject_only"),
        "prior_only_loss": component_loss(ode_model, batch, "prior_only"),
        "stateless_token_loss": component_loss(ode_model, batch, "stateless"),
        "learned_initial_slots": slot_statistics(model.initial_state.unsqueeze(0)),
        "ode_final_slots": slot_statistics(ode_state),
        "stochastic_final_slots": slot_statistics(stochastic_state),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
