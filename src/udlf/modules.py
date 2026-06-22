from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import UDLFModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class ObservationInjection(nn.Module):
    def __init__(self, config: UDLFModelConfig) -> None:
        super().__init__()
        d = config.latent_dim
        self.norm = RMSNorm(d, config.rms_eps)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(config.embed_dim, d, bias=False)
        self.v = nn.Linear(config.embed_dim, d)
        self.z = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d + config.embed_dim, d)
        self.out_norm = RMSNorm(d, config.rms_eps)

    def forward(
        self,
        state: Tensor,
        token_embed: Tensor,
        slot_identity: Tensor | None = None,
        diagnostics: dict[str, list[Tensor]] | None = None,
    ) -> Tensor:
        # state: [B, M, d], token_embed: [B, de]
        normalized = self.norm(state)
        if slot_identity is not None:
            normalized = normalized + slot_identity
        q = self.q(normalized)
        k = self.k(token_embed).unsqueeze(-1)
        alpha = torch.softmax(torch.matmul(q, k).squeeze(-1) / math.sqrt(q.shape[-1]), dim=-1)
        candidate = torch.tanh(self.v(token_embed).unsqueeze(1) + self.z(state))
        token_broadcast = token_embed.unsqueeze(1).expand(-1, state.shape[1], -1)
        eta = torch.sigmoid(self.gate(torch.cat([normalized, token_broadcast], dim=-1)))
        updated = state + alpha.unsqueeze(-1) * eta * candidate
        output = self.out_norm(updated)
        if diagnostics is not None:
            jump = output - state
            entropy_scale = math.log(alpha.shape[-1]) if alpha.shape[-1] > 1 else 1.0
            state_rms = state.detach().pow(2).mean().sqrt()
            diagnostics.setdefault("injection_jump_rms", []).append(jump.detach().pow(2).mean().sqrt())
            diagnostics.setdefault("injection_relative_jump", []).append(jump.detach().pow(2).mean().sqrt() / (state_rms + 1e-8))
            diagnostics.setdefault("injection_alpha_entropy", []).append(
                (-(alpha.detach() * alpha.detach().clamp_min(1e-12).log()).sum(dim=-1) / entropy_scale).mean()
            )
            diagnostics.setdefault("injection_gate_mean", []).append(eta.detach().mean())
            diagnostics.setdefault("injection_gate_low_saturation", []).append((eta.detach() < 0.05).float().mean())
            diagnostics.setdefault("injection_gate_high_saturation", []).append((eta.detach() > 0.95).float().mean())
            diagnostics.setdefault("injection_state_cosine", []).append(
                F.cosine_similarity(state.detach().flatten(start_dim=1), output.detach().flatten(start_dim=1), dim=-1).mean()
            )
        return output


class LatentInteractionCore(nn.Module):
    def __init__(self, config: UDLFModelConfig) -> None:
        super().__init__()
        d = config.latent_dim
        hidden = config.ff_multiplier * d
        self.norm = RMSNorm(d, config.rms_eps)
        self.input_condition = nn.Linear(config.embed_dim, d)
        self.attn = nn.MultiheadAttention(d, config.latent_heads, batch_first=True)
        self.u = nn.Linear(3 * d, hidden)
        self.v = nn.Linear(3 * d, hidden)
        self.out = nn.Linear(hidden, d)

    def forward(
        self,
        state: Tensor,
        token_embed: Tensor,
        slot_identity: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        r = self.norm(state)
        if slot_identity is not None:
            r = r + slot_identity
        context, _ = self.attn(r, r, r, need_weights=False)
        condition = self.input_condition(token_embed).unsqueeze(1).expand_as(r)
        joined = torch.cat([r, context, condition], dim=-1)
        hidden = F.silu(self.u(joined)) * torch.sigmoid(self.v(joined))
        y = self.out(hidden)
        return y, r


class PriorDynamics(nn.Module):
    def __init__(self, config: UDLFModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.latent_dim
        self.core = LatentInteractionCore(config)
        self.additional_cores = nn.ModuleList(
            LatentInteractionCore(config) for _ in range(config.prior_depth - 1)
        )
        if config.prior_depth > 1:
            output_scale = 1.0 / math.sqrt(config.prior_depth)
            with torch.no_grad():
                for core in (self.core, *self.additional_cores):
                    core.out.weight.mul_(output_scale)
                    if core.out.bias is not None:
                        core.out.bias.mul_(output_scale)
        self.dissipation = nn.Linear(2 * d, d)
        self.diffusion = nn.Linear(2 * d, d)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.solver_adapters = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(2 * d, config.solver_adapter_rank, bias=False),
                    nn.SiLU(),
                    nn.Linear(config.solver_adapter_rank, d, bias=False),
                )
                for _ in range(config.solver_steps)
            ) if config.solver_adapter_rank > 0 else nn.ModuleList()
        for adapter in self.solver_adapters:
            nn.init.zeros_(adapter[-1].weight)

    def drift_and_sigma(
        self,
        state: Tensor,
        token_embed: Tensor,
        slot_identity: Tensor | None = None,
        solver_index: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not self.additional_cores:
            y, r = self.core(state, token_embed, slot_identity)
        else:
            features = state
            residual_scale = 1.0 / self.config.prior_depth
            y = torch.zeros_like(state)
            r = state
            for core in (self.core, *self.additional_cores):
                delta, r = core(features, token_embed, slot_identity)
                scaled_delta = residual_scale * delta
                features = features + scaled_delta
                y = y + scaled_delta
        joined = torch.cat([r, y], dim=-1)
        if self.solver_adapters:
            if solver_index is None or not 0 <= solver_index < len(self.solver_adapters):
                raise ValueError("solver_index is required for configured solver adapters")
            y = y + self.solver_adapters[solver_index](joined)
            joined = torch.cat([r, y], dim=-1)
        lam = self.config.lambda_max * torch.sigmoid(self.dissipation(joined))
        drift = self.config.beta_max * torch.tanh(y) - lam * state

        if self.config.diffusion_mode == "ode":
            sigma = torch.zeros_like(state)
        elif self.config.diffusion_mode == "fixed":
            sigma = torch.full_like(state, self.config.fixed_sigma)
        else:
            sigma = self.config.sigma_min + (self.config.sigma_max - self.config.sigma_min) * torch.sigmoid(
                self.diffusion(joined)
            )
        return drift, sigma

    def euler_maruyama(
        self,
        state: Tensor,
        token_embed: Tensor,
        generator: torch.Generator | None = None,
        slot_identity: Tensor | None = None,
        diagnostics: dict[str, list[Tensor]] | None = None,
    ) -> Tensor:
        ds = 1.0 / self.config.solver_steps
        sqrt_ds = math.sqrt(ds)
        z = state
        for solver_index in range(self.config.solver_steps):
            drift, sigma = self.drift_and_sigma(z, token_embed, slot_identity, solver_index)
            if self.config.diffusion_mode == "ode":
                noise = torch.zeros_like(z)
            else:
                noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator)
            jump = drift * ds + sigma * sqrt_ds * noise
            if diagnostics is not None:
                diagnostics.setdefault("drift_rms", []).append(drift.detach().pow(2).mean().sqrt())
                diagnostics.setdefault("sigma_min", []).append(sigma.detach().amin())
                diagnostics.setdefault("sigma_max", []).append(sigma.detach().amax())
                diagnostics.setdefault("sigma_rms", []).append(sigma.detach().pow(2).mean().sqrt())
                diagnostics.setdefault("jump_rms", []).append(jump.detach().pow(2).mean().sqrt())
            z = z + jump
        return z


class PosteriorControl(nn.Module):
    def __init__(self, config: UDLFModelConfig) -> None:
        super().__init__()
        d = config.latent_dim
        self.norm = RMSNorm(d, config.rms_eps)
        self.target = nn.Linear(config.embed_dim, d)
        self.token = nn.Linear(config.embed_dim, d)
        self.control = nn.Sequential(
            nn.Linear(3 * d, config.ff_multiplier * d),
            nn.SiLU(),
            nn.Linear(config.ff_multiplier * d, d),
        )

    def forward(self, state: Tensor, token_embed: Tensor, target_embed: Tensor) -> Tensor:
        normalized = self.norm(state)
        token = self.token(token_embed).unsqueeze(1).expand_as(normalized)
        target = self.target(target_embed).unsqueeze(1).expand_as(normalized)
        return self.control(torch.cat([normalized, token, target], dim=-1))


class LatentReadout(nn.Module):
    def __init__(self, config: UDLFModelConfig) -> None:
        super().__init__()
        d = config.latent_dim
        self.config = config
        self.condition = nn.Linear(config.embed_dim + d, d)
        self.base_queries = nn.Parameter(torch.randn(config.readout_heads, d) / math.sqrt(d))
        self.query_delta = nn.Linear(d, config.readout_heads * d)
        key_output_dim = config.readout_heads * d if config.readout_head_keys else d
        self.key = nn.Linear(d, key_output_dim, bias=False)
        self.merge = nn.Linear(config.readout_heads * d, d)
        self.norm = RMSNorm(d, config.rms_eps)
        self.to_embed = nn.Linear(d, config.embed_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(
        self,
        state: Tensor,
        token_embed: Tensor,
        embedding_weight: Tensor,
        slot_identity: Tensor | None = None,
    ) -> Tensor:
        original_shape = state.shape[:-2]
        if state.ndim < 3:
            raise ValueError("state must have shape [..., slots, dim]")
        state = state.reshape(-1, state.shape[-2], state.shape[-1])
        token_embed = token_embed.reshape(-1, token_embed.shape[-1])
        if slot_identity is not None:
            slot_identity = slot_identity.expand(state.shape[0], -1, -1)
        mean_state = state.mean(dim=1)
        cond = self.condition(torch.cat([token_embed, mean_state], dim=-1))
        query_delta = self.query_delta(cond).view(-1, self.config.readout_heads, self.config.latent_dim)
        queries = self.base_queries.unsqueeze(0) + query_delta
        keys = self.key(state if slot_identity is None else state + slot_identity)
        if self.config.readout_head_keys:
            keys = keys.view(
                state.shape[0],
                state.shape[1],
                self.config.readout_heads,
                self.config.latent_dim,
            )
            scores = torch.einsum("bhd,bmhd->bhm", queries, keys)
        else:
            scores = torch.einsum("bhd,bmd->bhm", queries, keys)
        scores = scores / math.sqrt(self.config.latent_dim)
        weights = torch.softmax(scores, dim=-1)
        heads = torch.einsum("bhm,bmd->bhd", weights, state)
        merged = self.merge(heads.flatten(start_dim=1))
        output_embed = self.to_embed(self.norm(merged))
        logits = output_embed @ embedding_weight.T + self.bias
        return logits.reshape(*original_shape, self.config.vocab_size)
