from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import UDLFModelConfig
from .modules import ObservationInjection, PriorDynamics, LatentReadout


@dataclass
class StageAOutput:
    logits: Tensor
    loss: Tensor | None
    final_state: Tensor


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
        self.initial_state = nn.Parameter(torch.zeros(config.latent_slots, config.latent_dim))
        self.inject = ObservationInjection(config)
        self.prior = PriorDynamics(config)
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
    ) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        batch, steps = input_ids.shape
        if steps < 1:
            raise ValueError("input_ids must contain at least one token")
        if state is None:
            state = self.init_state(batch, device=input_ids.device, dtype=self.embedding.weight.dtype)

        logits: list[Tensor] = []
        for t in range(steps):
            token_embed = self.embedding(input_ids[:, t])
            state = self.inject(state, token_embed)
            state = self.prior.euler_maruyama(state, token_embed, generator=generator)
            logits.append(self.readout(state, token_embed, self.embedding.weight))
        return torch.stack(logits, dim=1), state

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
