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
    device: str = "cuda"
    allow_cpu_training: bool = False
    amp: bool = True
    compile_model: bool = False

    data_path: str = ""
    data_task: str = "next_token"
    train_split: str = "train"
    validation_split: str = "validation"
    data_column: str = "input_ids"
    seq_len: int = 16
    synthetic_task: str = "repeat"
    synthetic_suffix_loss_only: bool = False

    batch_size: int = 8
    max_steps: int = 20
    grad_accum_steps: int = 1
    auto_batch: bool = False
    auto_batch_max: int = 128
    vram_fraction: float = 0.90
    auto_batch_predict_safety: float = 1.35
    auto_batch_probe_budget_fraction: float = 1.0
    auto_batch_max_probe_increment: int = 8
    auto_adjust_grad_accum: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 0
    min_lr_ratio: float = 1.0

    segment_len: int = 0
    segment_len_min: int = 0
    segment_len_max: int = 0
    detach_state_between_segments: bool = True
    prior_path_samples: int = 1
    prior_state_selection: str = "first"
    lambda_prior: float = 1.0
    lambda_posterior: float = 1.0
    lambda_kl: float = 1.0
    posterior_dropout: float = 0.0
    posterior_dropout_max: float = 0.95

    log_every: int = 1
    console_log_mode: str = "progress"
    eval_every: int = 0
    eval_batches: int = 4
    eval_batch_size: int = 0
    dynamics_diagnostics: bool = False
    stability_diagnostics: bool = False
    stability_diagnostic_every: int = 0
    stability_diagnostic_eps: float = 1e-4
    intervention_shift_tokens: int = 1
    intervention_pair_trials: int = 4
    intervention_perturb_std: float = 0.05
    intervention_perturb_trials: int = 8
    intervention_mix_alpha: float = 0.1
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
    allow_run_overwrite: bool = False
    architecture: str = "udlf"
    eval_interventions: bool = True

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
    llm_dim: int = 512
    llm_layers: int = 12
    mamba_d_state: int = 16
    mamba_expand: int = 2
    mamba_conv_kernel: int = 4
    mamba_dt_rank: int = 0
    tie_embeddings: bool = True

    sleep_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.auto_batch_max < 1:
            raise ValueError("auto_batch_max must be >= 1")
        if not 0.1 <= self.vram_fraction <= 0.98:
            raise ValueError("vram_fraction must be between 0.1 and 0.98")
        if self.auto_batch_predict_safety < 1.0:
            raise ValueError("auto_batch_predict_safety must be >= 1.0")
        if not 0.5 <= self.auto_batch_probe_budget_fraction <= 1.0:
            raise ValueError("auto_batch_probe_budget_fraction must be between 0.5 and 1.0")
        if self.auto_batch_max_probe_increment < 1:
            raise ValueError("auto_batch_max_probe_increment must be >= 1")
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")
        if self.eval_batches < 1:
            raise ValueError("eval_batches must be >= 1")
        if self.eval_batch_size < 0:
            raise ValueError("eval_batch_size must be >= 0")
        if self.stability_diagnostic_every < 0:
            raise ValueError("stability_diagnostic_every must be >= 0")
        if self.stability_diagnostic_eps <= 0:
            raise ValueError("stability_diagnostic_eps must be positive")
        if self.console_log_mode not in {"progress", "quiet"}:
            raise ValueError("console_log_mode must be 'progress' or 'quiet'")
        if self.mode not in {"stage-a", "stage-b"}:
            raise ValueError("mode must be 'stage-a' or 'stage-b'")
        if self.architecture not in {"udlf", "mamba"}:
            raise ValueError("architecture must be 'udlf' or 'mamba'")
        if self.mode == "stage-b" and self.architecture != "udlf":
            raise ValueError("stage-b mode requires architecture='udlf'")
        if self.stop_check_every < 1:
            raise ValueError("stop_check_every must be >= 1")
        if self.prior_path_samples < 1:
            raise ValueError("prior_path_samples must be >= 1")
        if self.prior_state_selection not in {"first", "mean"}:
            raise ValueError("prior_state_selection must be 'first' or 'mean'")
        if self.prior_path_samples > 1 and self.segment_len > 0:
            raise ValueError("prior_path_samples > 1 currently requires segment_len=0")
        if self.mode == "stage-b" and self.segment_len > 0:
            raise ValueError("stage-b mode currently requires segment_len=0")
        if self.lambda_prior < 0 or self.lambda_posterior < 0 or self.lambda_kl < 0:
            raise ValueError("lambda_prior, lambda_posterior, and lambda_kl must be non-negative")
        if not 0.0 <= self.posterior_dropout < 1.0:
            raise ValueError("posterior_dropout must satisfy 0 <= p < 1")
        if not 0.0 <= self.posterior_dropout_max < 1.0:
            raise ValueError("posterior_dropout_max must satisfy 0 <= p < 1")
        if self.posterior_dropout > self.posterior_dropout_max:
            raise ValueError("posterior_dropout must be <= posterior_dropout_max")
        if self.segment_len_min < 0 or self.segment_len_max < 0:
            raise ValueError("segment_len_min and segment_len_max must be >= 0")
        if self.segment_len_min and self.segment_len_max and self.segment_len_min > self.segment_len_max:
            raise ValueError("segment_len_min must be <= segment_len_max")
        if self.intervention_perturb_trials < 1:
            raise ValueError("intervention_perturb_trials must be >= 1")
        if self.intervention_pair_trials < 1:
            raise ValueError("intervention_pair_trials must be >= 1")
        if self.intervention_shift_tokens < 1:
            raise ValueError("intervention_shift_tokens must be >= 1")
        if not 0.0 <= self.intervention_mix_alpha <= 1.0:
            raise ValueError("intervention_mix_alpha must be between 0 and 1")
        if self.llm_dim <= 0 or self.llm_layers <= 0:
            raise ValueError("llm_dim and llm_layers must be positive")
        if self.mamba_d_state <= 0 or self.mamba_expand <= 0 or self.mamba_conv_kernel <= 0:
            raise ValueError("mamba_d_state, mamba_expand, and mamba_conv_kernel must be positive")
        if self.mamba_dt_rank < 0:
            raise ValueError("mamba_dt_rank must be >= 0")

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
            tie_embeddings=self.tie_embeddings,
        )

    def mamba_config(self):
        from udlf.llm import MambaLMConfig

        return MambaLMConfig(
            vocab_size=self.vocab_size,
            d_model=self.llm_dim,
            n_layers=self.llm_layers,
            d_state=self.mamba_d_state,
            expand=self.mamba_expand,
            conv_kernel=self.mamba_conv_kernel,
            dt_rank=self.mamba_dt_rank,
            tie_embeddings=self.tie_embeddings,
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
