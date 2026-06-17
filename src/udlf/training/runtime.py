from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch

from udlf.data import RepeatingPatternDataset, TokenDatasetFromDisk
from udlf.training.config import UDLFTrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def make_noise_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def build_datasets(config: UDLFTrainConfig):
    if config.data_path:
        train_dataset = TokenDatasetFromDisk(
            config.data_path,
            split=config.train_split,
            seq_len=config.seq_len,
            seed=config.seed + 1,
            column=config.data_column,
        )
        eval_dataset = TokenDatasetFromDisk(
            config.data_path,
            split=config.validation_split,
            seq_len=config.seq_len,
            seed=config.seed + 10_000,
            column=config.data_column,
        )
        return train_dataset, eval_dataset
    model_config = config.model_config()
    return (
        RepeatingPatternDataset(
            model_config.vocab_size,
            config.seq_len,
            seed=config.seed + 1,
            suffix_loss_only=config.synthetic_suffix_loss_only,
        ),
        RepeatingPatternDataset(
            model_config.vocab_size,
            config.seq_len,
            seed=config.seed + 10_000,
            suffix_loss_only=config.synthetic_suffix_loss_only,
        ),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, *, max_steps: int, warmup_steps: int, min_lr_ratio: float):
    if warmup_steps <= 0 and min_lr_ratio >= 1.0:
        return None

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        denom = max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
