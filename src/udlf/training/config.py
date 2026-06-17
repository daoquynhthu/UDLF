from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from udlf.config import UDLFModelConfig


@dataclass
class UDLFTrainConfig:
    mode: str = "stage-a"
    run_name: str = "stage_a"
    output_dir: str = "runs"
    run_dir: str = ""

    seed: int = 123
    noise_seed: int | None = None
    device: str = "auto"
    amp: bool = True
    compile_model: bool = False

    data_path: str = ""
    train_split: str = "train"
    validation_split: str = "validation"
    data_column: str = "input_ids"
    seq_len: int = 16
    synthetic_suffix_loss_only: bool = False

    batch_size: int = 8
    max_steps: int = 20
    grad_accum_steps: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 0
    min_lr_ratio: float = 1.0

    segment_len: int = 0
    detach_state_between_segments: bool = True

    log_every: int = 1
    eval_every: int = 0
    eval_batches: int = 4
    intervention_perturb_std: float = 0.05
    save_every: int = 0
    latest_every: int = 0
    async_checkpoint: bool = True
    async_checkpoint_queue: int = 2
    metrics_flush_every: int = 50
    metrics_fsync_every: int = 1000
    stop_file: str = ""
    stop_check_every: int = 1
    resume: str = ""
    strict_resume: bool = True

    vocab_size: int = 64
    latent_slots: int = 8
    latent_dim: int = 64
    embed_dim: int = 64
    ff_multiplier: int = 2
    latent_heads: int = 4
    readout_heads: int = 2
    solver_steps: int = 2
    beta_max: float = 0.2
    lambda_max: float = 0.5
    sigma_min: float = 1e-4
    sigma_max: float = 0.02
    diffusion_mode: str = "ode"
    fixed_sigma: float = 0.01

    sleep_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")
        if self.stop_check_every < 1:
            raise ValueError("stop_check_every must be >= 1")

    def model_config(self) -> UDLFModelConfig:
        return UDLFModelConfig(
            vocab_size=self.vocab_size,
            latent_slots=self.latent_slots,
            latent_dim=self.latent_dim,
            embed_dim=self.embed_dim,
            ff_multiplier=self.ff_multiplier,
            latent_heads=self.latent_heads,
            readout_heads=self.readout_heads,
            solver_steps=self.solver_steps,
            beta_max=self.beta_max,
            lambda_max=self.lambda_max,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            diffusion_mode=self.diffusion_mode,  # type: ignore[arg-type]
            fixed_sigma=self.fixed_sigma,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_run_dir(self) -> Path:
        if self.run_dir:
            return Path(self.run_dir).resolve()
        return (Path(self.output_dir) / self.run_name).resolve()


def load_raw_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_config_dict(raw: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    params = raw.get("parameters")
    if isinstance(params, dict):
        merged.update(params)
    merged.update({key: value for key, value in raw.items() if key != "parameters"})
    if "steps" in merged and "max_steps" not in merged:
        merged["max_steps"] = merged["steps"]
    return merged


def train_config_from_dict(raw: dict[str, Any]) -> UDLFTrainConfig:
    normalized = normalize_config_dict(raw)
    allowed = {field.name for field in fields(UDLFTrainConfig)}
    filtered = {key: value for key, value in normalized.items() if key in allowed}
    return UDLFTrainConfig(**filtered)
