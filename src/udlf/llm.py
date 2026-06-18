from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .modules import RMSNorm


@dataclass(frozen=True)
class MambaLMConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 12
    d_state: int = 16
    expand: int = 2
    conv_kernel: int = 4
    dt_rank: int = 0
    rms_eps: float = 1e-6
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be > 1")
        if self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError("d_model and n_layers must be positive")
        if self.d_state <= 0 or self.expand <= 0:
            raise ValueError("d_state and expand must be positive")
        if self.conv_kernel <= 0:
            raise ValueError("conv_kernel must be positive")
        if self.dt_rank < 0:
            raise ValueError("dt_rank must be >= 0")


@dataclass
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None
    final_state: Tensor | None = None


class MambaBlock(nn.Module):
    """Pure PyTorch Mamba/S6 block.

    This follows the standard selective-state-space block structure but avoids
    fused CUDA kernels so the baseline is runnable in the local environment.
    """

    def __init__(self, config: MambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.inner_dim = config.expand * config.d_model
        self.dt_rank = config.dt_rank if config.dt_rank > 0 else math.ceil(config.d_model / 16)
        self.norm = RMSNorm(config.d_model, config.rms_eps)
        self.in_proj = nn.Linear(config.d_model, 2 * self.inner_dim, bias=False)
        self.conv1d = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=config.conv_kernel,
            groups=self.inner_dim,
            bias=True,
        )
        self.x_proj = nn.Linear(self.inner_dim, self.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, config.d_state + 1, dtype=torch.float32)).repeat(self.inner_dim, 1))
        self.d = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        xz = self.in_proj(self.norm(x))
        x_inner, gate = xz.chunk(2, dim=-1)
        conv_input = x_inner.transpose(1, 2)
        conv_input = F.pad(conv_input, (self.config.conv_kernel - 1, 0))
        x_inner = self.conv1d(conv_input).transpose(1, 2)
        x_inner = F.silu(x_inner)

        batch, time, inner = x_inner.shape
        d_state = self.config.d_state
        x_params = self.x_proj(x_inner)
        dt_input, b, c = torch.split(x_params, [self.dt_rank, d_state, d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_input))
        a = -torch.exp(self.a_log.float()).to(dtype=x_inner.dtype, device=x_inner.device)

        state = torch.zeros(batch, inner, d_state, dtype=x_inner.dtype, device=x_inner.device)
        outputs: list[Tensor] = []
        for t in range(time):
            dt_t = dt[:, t].unsqueeze(-1)
            x_t = x_inner[:, t].unsqueeze(-1)
            decay = torch.exp(dt_t * a.unsqueeze(0))
            state = state * decay + dt_t * b[:, t].unsqueeze(1) * x_t
            y_t = (state * c[:, t].unsqueeze(1)).sum(dim=-1) + self.d.to(dtype=x_inner.dtype, device=x_inner.device) * x_inner[:, t]
            outputs.append(y_t * F.silu(gate[:, t]))
        y = torch.stack(outputs, dim=1)
        return residual + self.out_proj(y)


class MambaLMModel(nn.Module):
    def __init__(self, config: MambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, config.rms_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor, *, targets: Tensor | None = None) -> CausalLMOutput:
        if targets is None:
            if input_ids.shape[1] < 2:
                raise ValueError("input_ids must have at least two tokens when targets are omitted")
            prefix = input_ids[:, :-1]
            targets = input_ids[:, 1:]
        else:
            prefix = input_ids
            if targets.shape != prefix.shape:
                raise ValueError("targets must match input_ids shape when provided")

        hidden = self.embedding(prefix)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.norm(hidden))
        loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return CausalLMOutput(logits=logits, loss=loss)
