from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch
from datasets import load_from_disk

from udlf.llm import MambaLMModel
from udlf.model import UDLFStageAModel
from udlf.training.checkpoint import normalize_state_dict_for_model
from udlf.training.config import train_config_from_dict


def load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = train_config_from_dict(checkpoint["config"])
    if config.architecture == "udlf":
        model = UDLFStageAModel(replace(config.model_config(), diffusion_mode="ode"))
    else:
        model = MambaLMModel(config.mamba_config())
    model.load_state_dict(normalize_state_dict_for_model(model, checkpoint["model"]))
    return model.cuda().eval(), config


@torch.no_grad()
def evaluate(model, rows: list[list[int]], batch_size: int) -> dict[str, float]:
    losses = []
    for start in range(0, len(rows), batch_size):
        batch = torch.tensor(rows[start : start + batch_size], dtype=torch.long, device="cuda")
        loss = model(batch).loss
        losses.append(float(loss.cpu()))
    values = torch.tensor(losses, dtype=torch.float64)
    mean = float(values.mean())
    return {
        "loss": mean,
        "perplexity": math.exp(mean),
        "batch_loss_se": float(values.std(unbiased=True) / math.sqrt(len(values))),
        "batches": len(values),
        "sequences": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--udlf", type=Path, required=True)
    parser.add_argument("--mamba", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    udlf, udlf_config = load_model(args.udlf)
    mamba, mamba_config = load_model(args.mamba)
    dataset = load_from_disk(str(args.data))[udlf_config.validation_split]
    rows = [dataset[index][udlf_config.data_column][: udlf_config.seq_len + 1] for index in range(args.sequences)]
    report = {
        "udlf": evaluate(udlf, rows, args.batch_size),
        "mamba": evaluate(mamba, rows, args.batch_size),
        "seq_len": udlf_config.seq_len,
        "same_validation_rows": True,
    }
    report["loss_gap_udlf_minus_mamba"] = report["udlf"]["loss"] - report["mamba"]["loss"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
