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
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    rms_eps: float = 1e-6
    residual_in_fp32: bool = True
    pad_vocab_size_multiple: int = 1
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
        if self.dt_min <= 0 or self.dt_max <= 0 or self.dt_min >= self.dt_max:
            raise ValueError("dt_min and dt_max must be positive with dt_min < dt_max")
        if self.dt_init not in {"constant", "random"}:
            raise ValueError("dt_init must be 'constant' or 'random'")
        if self.dt_scale <= 0:
            raise ValueError("dt_scale must be positive")
        if self.dt_init_floor <= 0:
            raise ValueError("dt_init_floor must be positive")
        if self.pad_vocab_size_multiple < 1:
            raise ValueError("pad_vocab_size_multiple must be >= 1")

    @property
    def padded_vocab_size(self) -> int:
        multiple = self.pad_vocab_size_multiple
        if self.vocab_size % multiple == 0:
            return self.vocab_size
        return self.vocab_size + multiple - (self.vocab_size % multiple)


@dataclass
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None
    final_state: Tensor | None = None


class MambaMixer(nn.Module):
    """Pure PyTorch Mamba1/S6 mixer.

    The tensor program mirrors the official Mamba1 slow path, but keeps the
    selective scan as a Python loop so it runs without custom CUDA extensions.
    """

    def __init__(self, config: MambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.conv_kernel
        self.expand = config.expand
        self.inner_dim = int(config.expand * config.d_model)
        self.dt_rank = config.dt_rank if config.dt_rank > 0 else math.ceil(config.d_model / 16)
        self.in_proj = nn.Linear(config.d_model, 2 * self.inner_dim, bias=config.bias)
        self.conv1d = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=config.conv_kernel,
            groups=self.inner_dim,
            padding=config.conv_kernel - 1,
            bias=config.conv_bias,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.inner_dim, self.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, config.d_state + 1, dtype=torch.float32)).repeat(self.inner_dim, 1))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]
        self.out_proj = nn.Linear(self.inner_dim, config.d_model, bias=config.bias)
        self._init_dt(config)

    def _init_dt(self, config: MambaLMConfig) -> None:
        dt_init_std = self.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.inner_dim) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True  # type: ignore[attr-defined]

    def forward(self, x: Tensor) -> Tensor:
        batch, time, _ = x.shape
        xz = self.in_proj(x).transpose(1, 2)
        x_inner, gate = xz.chunk(2, dim=1)
        x_inner = self.act(self.conv1d(x_inner)[..., :time])

        d_state = self.config.d_state
        x_params = self.x_proj(x_inner.transpose(1, 2).reshape(batch * time, self.inner_dim))
        dt_input, b, c = torch.split(x_params, [self.dt_rank, d_state, d_state], dim=-1)
        dt = F.linear(dt_input, self.dt_proj.weight).reshape(batch, time, self.inner_dim).transpose(1, 2)
        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype).view(1, -1, 1))
        b = b.reshape(batch, time, d_state).transpose(1, 2).contiguous()
        c = c.reshape(batch, time, d_state).transpose(1, 2).contiguous()
        a = -torch.exp(self.A_log.float()).to(dtype=x_inner.dtype, device=x_inner.device)

        state = torch.zeros(batch, self.inner_dim, d_state, dtype=x_inner.dtype, device=x_inner.device)
        outputs: list[Tensor] = []
        for t in range(time):
            dt_t = dt[:, :, t].unsqueeze(-1)
            x_t = x_inner[:, :, t].unsqueeze(-1)
            state = state * torch.exp(dt_t * a.unsqueeze(0)) + x_t * dt_t * b[:, :, t].unsqueeze(1)
            y_t = (state * c[:, :, t].unsqueeze(1)).sum(dim=-1)
            y_t = y_t + self.D.to(dtype=x_inner.dtype, device=x_inner.device) * x_inner[:, :, t]
            y_t = y_t * self.act(gate[:, :, t])
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Official-style Add -> Norm -> Mamba mixer block."""

    def __init__(self, config: MambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.norm = RMSNorm(config.d_model, config.rms_eps)
        self.mixer = MambaMixer(config)

    def forward(self, hidden_states: Tensor, residual: Tensor | None = None) -> tuple[Tensor, Tensor]:
        residual = hidden_states + residual if residual is not None else hidden_states
        normalized = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.config.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(normalized)
        return hidden_states, residual


class MambaLMModel(nn.Module):
    def __init__(self, config: MambaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.padded_vocab_size, config.d_model)
        self.blocks = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, config.rms_eps)
        self.lm_head = nn.Linear(config.d_model, config.padded_vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        if isinstance(module, MambaMixer):
            nn.init.kaiming_uniform_(module.out_proj.weight, a=math.sqrt(5))
            with torch.no_grad():
                module.out_proj.weight /= math.sqrt(self.config.n_layers)

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
        residual = None
        for block in self.blocks:
            hidden, residual = block(hidden, residual)
        residual = hidden + residual if residual is not None else hidden
        hidden = self.norm(residual.to(dtype=self.norm.weight.dtype))
        logits = self.lm_head(hidden)[..., : self.config.vocab_size]
        loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return CausalLMOutput(logits=logits, loss=loss)
