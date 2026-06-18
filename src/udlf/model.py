from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import math

from .config import UDLFModelConfig
from .modules import ObservationInjection, PriorDynamics, LatentReadout, PosteriorControl


@dataclass
class StageAOutput:
    logits: Tensor
    loss: Tensor | None
    final_state: Tensor


@dataclass
class PosteriorPrefixOutput:
    prior_logits: Tensor
    posterior_logits: Tensor
    prior_final_state: Tensor
    posterior_final_state: Tensor
    posterior_kl: Tensor


class UDLFStageAModel(nn.Module):
    """Single-path prior UDLF model.

    This implements the first-stage protocol: inject current token, run one
    prior path, read out the next-token distribution, and carry only the prior
    terminal state forward.
    """

    def __init__(self, config: UDLFModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.output_weight = self.embedding.weight if config.tie_embeddings else nn.Parameter(
            torch.empty(config.vocab_size, config.embed_dim)
        )
        if not config.tie_embeddings:
            nn.init.normal_(self.output_weight, mean=0.0, std=0.02)
        self.initial_state = nn.Parameter(torch.zeros(config.latent_slots, config.latent_dim))
        self.inject = ObservationInjection(config)
        self.prior = PriorDynamics(config)
        self.posterior_control = PosteriorControl(config)
        self.readout = LatentReadout(config)

    def init_state(self, batch_size: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        state = self.initial_state
        if device is not None or dtype is not None:
            state = state.to(device=device or state.device, dtype=dtype or state.dtype)
        return state.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def forward_prefix(
        self,
        input_ids: Tensor,
        *,
        state: Tensor | None = None,
        generator: torch.Generator | None = None,
        diagnostics: dict[str, list[Tensor]] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        batch, steps = input_ids.shape
        if steps < 1:
            raise ValueError("input_ids must contain at least one token")
        if state is None:
            state = self.init_state(batch, device=input_ids.device, dtype=self.embedding.weight.dtype)

        states: list[Tensor] = []
        token_embeds: list[Tensor] = []
        for t in range(steps):
            token_embed = self.embedding(input_ids[:, t])
            state = self.inject(state, token_embed)
            state = self.prior.euler_maruyama(state, token_embed, generator=generator, diagnostics=diagnostics)
            states.append(state)
            token_embeds.append(token_embed)
        logits = self.readout(torch.stack(states, dim=1), torch.stack(token_embeds, dim=1), self.output_weight)
        return logits, state

    def forward_posterior_prefix(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        *,
        state: Tensor | None = None,
        prior_generator: torch.Generator | None = None,
        posterior_generator: torch.Generator | None = None,
        diagnostics: dict[str, list[Tensor]] | None = None,
    ) -> PosteriorPrefixOutput:
        if input_ids.ndim != 2 or target_ids.ndim != 2:
            raise ValueError("input_ids and target_ids must have shape [batch, time]")
        if input_ids.shape != target_ids.shape:
            raise ValueError("input_ids and target_ids must have matching shapes")
        batch, steps = input_ids.shape
        if steps < 1:
            raise ValueError("input_ids must contain at least one token")
        if state is None:
            state = self.init_state(batch, device=input_ids.device, dtype=self.embedding.weight.dtype)

        prior_state = state
        posterior_state = state
        prior_states: list[Tensor] = []
        posterior_states: list[Tensor] = []
        token_embeds: list[Tensor] = []
        kl_terms: list[Tensor] = []
        for t in range(steps):
            token_embed = self.embedding(input_ids[:, t])
            target_embed = self.embedding(target_ids[:, t])

            prior_state = self.inject(prior_state, token_embed)
            prior_state = self.prior.euler_maruyama(
                prior_state,
                token_embed,
                generator=prior_generator,
                diagnostics=diagnostics,
            )

            posterior_state = self.inject(posterior_state, token_embed)
            ds = 1.0 / self.config.solver_steps
            sqrt_ds = math.sqrt(ds)
            kl = torch.zeros((), device=posterior_state.device, dtype=posterior_state.dtype)
            for _ in range(self.config.solver_steps):
                drift, sigma = self.prior.drift_and_sigma(posterior_state, token_embed)
                control = self.posterior_control(posterior_state, token_embed, target_embed)
                if self.config.diffusion_mode == "ode":
                    noise = torch.zeros_like(posterior_state)
                else:
                    noise = torch.randn(
                        posterior_state.shape,
                        device=posterior_state.device,
                        dtype=posterior_state.dtype,
                        generator=posterior_generator,
                    )
                posterior_state = posterior_state + (drift + sigma * control) * ds + sigma * sqrt_ds * noise
                kl = kl + 0.5 * control.pow(2).mean() * ds
            prior_states.append(prior_state)
            posterior_states.append(posterior_state)
            token_embeds.append(token_embed)
            kl_terms.append(kl)

        stacked_embeds = torch.stack(token_embeds, dim=1)
        prior_logits = self.readout(torch.stack(prior_states, dim=1), stacked_embeds, self.output_weight)
        posterior_logits = self.readout(torch.stack(posterior_states, dim=1), stacked_embeds, self.output_weight)
        return PosteriorPrefixOutput(
            prior_logits=prior_logits,
            posterior_logits=posterior_logits,
            prior_final_state=prior_state,
            posterior_final_state=posterior_state,
            posterior_kl=torch.stack(kl_terms).mean(),
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        targets: Tensor | None = None,
        state: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> StageAOutput:
        if targets is None:
            if input_ids.shape[1] < 2:
                raise ValueError("input_ids must have at least two tokens when targets are omitted")
            prefix = input_ids[:, :-1]
            targets = input_ids[:, 1:]
        else:
            prefix = input_ids
            if targets.shape != prefix.shape:
                raise ValueError("targets must match input_ids shape when provided")

        logits, final_state = self.forward_prefix(prefix, state=state, generator=generator)
        loss = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return StageAOutput(logits=logits, loss=loss, final_state=final_state)
