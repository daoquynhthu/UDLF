from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DiffusionMode = Literal["state_dependent", "fixed", "ode"]


@dataclass(frozen=True)
class UDLFModelConfig:
    vocab_size: int
    latent_slots: int = 16
    latent_dim: int = 256
    embed_dim: int = 256
    ff_multiplier: int = 4
    latent_heads: int = 4
    readout_heads: int = 4
    prior_depth: int = 1
    solver_adapter_rank: int = 0
    solver_steps: int = 4
    beta_max: float = 0.2
    lambda_max: float = 0.5
    sigma_min: float = 1e-4
    sigma_max: float = 0.05
    diffusion_mode: DiffusionMode = "state_dependent"
    fixed_sigma: float = 0.01
    rms_eps: float = 1e-6
    tie_embeddings: bool = True
    initial_slot_std: float = 1.0
    slot_identity_std: float = 0.02

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be > 1")
        if self.latent_slots <= 0:
            raise ValueError("latent_slots must be positive")
        if self.latent_dim <= 0 or self.embed_dim <= 0:
            raise ValueError("latent_dim and embed_dim must be positive")
        if self.latent_dim % self.latent_heads != 0:
            raise ValueError("latent_dim must be divisible by latent_heads")
        if self.prior_depth <= 0:
            raise ValueError("prior_depth must be positive")
        if self.solver_adapter_rank < 0:
            raise ValueError("solver_adapter_rank must be non-negative")
        if self.solver_steps <= 0:
            raise ValueError("solver_steps must be positive")
        if not (0.0 < self.sigma_min < self.sigma_max):
            raise ValueError("require 0 < sigma_min < sigma_max")
        if self.fixed_sigma < 0:
            raise ValueError("fixed_sigma must be non-negative")
        if self.initial_slot_std <= 0 or self.slot_identity_std <= 0:
            raise ValueError("initial_slot_std and slot_identity_std must be positive")
