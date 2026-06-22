from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk

from udlf.gradient_diagnostics import (
    collect_gradient_vector,
    pairwise_cosines,
    summarize_gradient,
)
from udlf.model import UDLFStageAModel
from udlf.training.checkpoint import normalize_state_dict_for_model
from udlf.training.config import train_config_from_dict


def segmented_backward(
    model: UDLFStageAModel,
    batch: torch.Tensor,
    horizon: int,
    *,
    use_amp: bool,
) -> float:
    prefix_length = batch.shape[1] - 1
    segment_length = prefix_length if horizon <= 0 or horizon >= prefix_length else horizon
    state = None
    total_loss = 0.0
    for start in range(0, prefix_length, segment_length):
        end = min(start + segment_length, prefix_length)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits, final_state = model.forward_prefix(batch[:, start:end], state=state)
            loss = F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                batch[:, start + 1 : end + 1].reshape(-1),
            )
        weight = end - start
        (loss * (weight / prefix_length)).backward()
        total_loss += float(loss.detach().cpu()) * weight / prefix_length
        state = final_state.detach()
    return total_loss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--horizons", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--clip-threshold", type=float, default=1.0)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_config = train_config_from_dict(checkpoint["config"])
    model_config = replace(train_config.model_config(), diffusion_mode="ode")
    model = UDLFStageAModel(model_config).cuda().train()
    state_dict = normalize_state_dict_for_model(model, checkpoint["model"])
    model.load_state_dict(state_dict)

    dataset = load_from_disk(str(args.data))[train_config.validation_split]
    rows = [
        dataset[index][train_config.data_column][: train_config.seq_len + 1]
        for index in range(args.sequences)
    ]
    if any(len(row) != train_config.seq_len + 1 for row in rows):
        raise ValueError("diagnostic rows are shorter than seq_len + 1")
    batch = torch.tensor(rows, dtype=torch.long, device="cuda")

    vectors: dict[int, torch.Tensor] = {}
    grouped_vectors: dict[int, dict[str, torch.Tensor]] = {}
    summaries: dict[str, object] = {}
    for horizon in args.horizons:
        model.zero_grad(set_to_none=True)
        loss = segmented_backward(model, batch, horizon, use_amp=train_config.amp)
        vector, groups = collect_gradient_vector(model)
        vectors[horizon] = vector
        grouped_vectors[horizon] = groups
        summary = summarize_gradient(vector, groups, clip_threshold=args.clip_threshold)
        summary["loss"] = loss
        summaries[str(horizon)] = summary
        torch.cuda.empty_cache()

    group_cosines: dict[str, dict[str, float]] = {}
    for group in next(iter(grouped_vectors.values())):
        group_cosines[group] = pairwise_cosines(
            {horizon: groups[group] for horizon, groups in grouped_vectors.items()}
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "sequences": args.sequences,
        "seq_len": train_config.seq_len,
        "diffusion_mode": "ode",
        "clip_threshold": args.clip_threshold,
        "horizons": summaries,
        "total_gradient_cosines": pairwise_cosines(vectors),
        "group_gradient_cosines": group_cosines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = {
        "output": str(args.output),
        "norms": {key: round(value["total_norm"], 4) for key, value in summaries.items()},
        "clip_scales": {key: round(value["clip_scale"], 4) for key, value in summaries.items()},
        "cosines": {key: round(value, 4) for key, value in report["total_gradient_cosines"].items()},
    }
    print(json.dumps(compact, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
