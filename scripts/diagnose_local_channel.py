from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from udlf.model import UDLFStageAModel
from udlf.training.checkpoint import normalize_state_dict_for_model
from udlf.training.config import train_config_from_dict


def token_losses(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")


def binned_mean(values: torch.Tensor, width: int) -> list[dict[str, float | int]]:
    return [
        {
            "start": start,
            "end": min(start + width, values.shape[1]),
            "mean": float(values[:, start : start + width].mean()),
        }
        for start in range(0, values.shape[1], width)
    ]


def readout_geometry(
    model: UDLFStageAModel,
    states: torch.Tensor,
    token_embeds: torch.Tensor,
    identity: torch.Tensor,
) -> dict[str, float]:
    readout = model.readout
    flat_state = states.reshape(-1, states.shape[-2], states.shape[-1])
    flat_token = token_embeds.reshape(-1, token_embeds.shape[-1])
    expanded_identity = identity.expand(flat_state.shape[0], -1, -1)
    mean_state = flat_state.mean(dim=1)
    condition = readout.condition(torch.cat([flat_token, mean_state], dim=-1))
    query_delta = readout.query_delta(condition).view(
        -1,
        readout.config.readout_heads,
        readout.config.latent_dim,
    )
    queries = readout.base_queries.unsqueeze(0) + query_delta
    keys = readout.key(flat_state + expanded_identity)
    if readout.config.readout_head_keys:
        keys = keys.view(
            flat_state.shape[0],
            flat_state.shape[1],
            readout.config.readout_heads,
            readout.config.latent_dim,
        )
        scores = torch.einsum("bhd,bmhd->bhm", queries, keys)
    else:
        scores = torch.einsum("bhd,bmd->bhm", queries, keys)
    scores = scores / states.shape[-1] ** 0.5
    weights = torch.softmax(scores, dim=-1)
    normalized_weights = F.normalize(weights.float(), dim=-1)
    cosine = normalized_weights @ normalized_weights.transpose(-1, -2)
    heads = weights.shape[1]
    pair_cosine = ((cosine.sum(dim=(-2, -1)) - heads) / (heads * (heads - 1))).mean()
    entropy = (-(weights.float() * weights.float().clamp_min(1e-12).log()).sum(dim=-1) / torch.log(
        torch.tensor(float(weights.shape[-1]), device=weights.device)
    )).mean()
    head_values = torch.einsum("bhm,bmd->bhd", weights, flat_state).float()
    centered = head_values - head_values.mean(dim=1, keepdim=True)
    gram = centered @ centered.transpose(-1, -2)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    rank = eigenvalues.sum(dim=-1).square() / eigenvalues.square().sum(dim=-1).clamp_min(1e-12)
    return {
        "readout_attention_entropy": float(entropy.cpu()),
        "readout_attention_pair_cosine": float(pair_cosine.cpu()),
        "readout_attention_max": float(weights.max(dim=-1).values.mean().cpu()),
        "readout_head_output_rank": float(rank.mean().cpu()),
        "readout_query_delta_rms": float(query_delta.float().pow(2).mean().sqrt().cpu()),
        "readout_base_query_rms": float(readout.base_queries.float().pow(2).mean().sqrt().cpu()),
    }


def load_model(path: Path) -> tuple[UDLFStageAModel, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = train_config_from_dict(checkpoint["config"])
    model = UDLFStageAModel(replace(config.model_config(), diffusion_mode="ode"))
    state_dict = normalize_state_dict_for_model(model, checkpoint["model"])
    model.load_state_dict(state_dict)
    return model.cuda().eval(), config


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--position-bin", type=int, default=64)
    args = parser.parse_args()

    model, config = load_model(args.checkpoint)
    dataset = load_from_disk(str(args.data))[config.validation_split]
    rows = [
        dataset[index][config.data_column][: config.seq_len + 1]
        for index in range(args.sequences)
    ]
    modes = (
        "normal",
        "pre_injection_state",
        "zero_readout_condition",
        "shuffled_readout_condition",
        "direct_token_tied_head",
    )
    losses = {mode: [] for mode in modes}
    diagnostic_sums: dict[str, float] = {}
    diagnostic_counts: dict[str, int] = {}
    geometry_sums: dict[str, float] = {}
    geometry_batches = 0

    for start in range(0, len(rows), args.batch_size):
        batch = torch.tensor(rows[start : start + args.batch_size], dtype=torch.long, device="cuda")
        prefix = batch[:, :-1]
        targets = batch[:, 1:]
        state = model.init_state(batch.shape[0], device=batch.device, dtype=model.embedding.weight.dtype)
        identity = model.slot_identity_features()
        pre_states = []
        post_states = []
        embeds = []
        diagnostics: dict[str, list[torch.Tensor]] = {}
        for position in range(prefix.shape[1]):
            token_embed = model.embedding(prefix[:, position])
            pre_states.append(state)
            state = model.inject(state, token_embed, identity, diagnostics=diagnostics)
            state = model.prior.euler_maruyama(
                state,
                token_embed,
                slot_identity=identity,
                diagnostics=diagnostics,
            )
            post_states.append(state)
            embeds.append(token_embed)

        pre = torch.stack(pre_states, dim=1)
        post = torch.stack(post_states, dim=1)
        token_embeds = torch.stack(embeds, dim=1)
        zero_embeds = torch.zeros_like(token_embeds)
        shuffled_embeds = token_embeds.roll(1, dims=0)
        mode_logits = {
            "normal": model.readout(post, token_embeds, model.output_weight, identity),
            "pre_injection_state": model.readout(pre, token_embeds, model.output_weight, identity),
            "zero_readout_condition": model.readout(post, zero_embeds, model.output_weight, identity),
            "shuffled_readout_condition": model.readout(post, shuffled_embeds, model.output_weight, identity),
            "direct_token_tied_head": token_embeds @ model.output_weight.T + model.readout.bias,
        }
        for name, value in readout_geometry(model, post, token_embeds, identity).items():
            geometry_sums[name] = geometry_sums.get(name, 0.0) + value
        geometry_batches += 1
        for mode, logits in mode_logits.items():
            losses[mode].append(token_losses(logits, targets).cpu())
        for name, values in diagnostics.items():
            for value in values:
                diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + float(value.float().cpu())
                diagnostic_counts[name] = diagnostic_counts.get(name, 0) + 1
        del batch, prefix, targets, pre, post, token_embeds, mode_logits
        torch.cuda.empty_cache()

    joined = {mode: torch.cat(values, dim=0) for mode, values in losses.items()}
    means = {mode: float(values.mean()) for mode, values in joined.items()}
    normal = means["normal"]
    report = {
        "checkpoint": str(args.checkpoint),
        "sequences": len(rows),
        "seq_len": config.seq_len,
        "losses": means,
        "deltas_from_normal": {mode: value - normal for mode, value in means.items()},
        "position_bins": {
            mode: binned_mean(values - joined["normal"], args.position_bin)
            for mode, values in joined.items()
            if mode != "normal"
        },
        "dynamics": {
            name: diagnostic_sums[name] / diagnostic_counts[name]
            for name in sorted(diagnostic_sums)
        },
        "readout_geometry": {
            name: geometry_sums[name] / geometry_batches
            for name in sorted(geometry_sums)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "losses": means, "deltas": report["deltas_from_normal"]}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
