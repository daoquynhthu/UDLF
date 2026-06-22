from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn


GRADIENT_GROUPS = (
    "embedding",
    "initial_state",
    "slot_identity",
    "inject",
    "prior",
    "readout",
    "other",
)


def parameter_group(name: str) -> str:
    root = name.split(".", 1)[0]
    if root == "output_weight":
        return "embedding"
    return root if root in GRADIENT_GROUPS[:-1] else "other"


def collect_gradient_vector(model: nn.Module) -> tuple[Tensor, dict[str, Tensor]]:
    vectors: list[Tensor] = []
    grouped: dict[str, list[Tensor]] = {name: [] for name in GRADIENT_GROUPS}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        gradient = parameter.grad
        flat = (
            torch.zeros(parameter.numel(), dtype=torch.float32)
            if gradient is None
            else gradient.detach().float().cpu().reshape(-1)
        )
        vectors.append(flat)
        grouped[parameter_group(name)].append(flat)
    vector = torch.cat(vectors)
    grouped_vectors = {
        name: torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32)
        for name, parts in grouped.items()
    }
    return vector, grouped_vectors


def vector_norm(vector: Tensor) -> float:
    return float(torch.linalg.vector_norm(vector.double()))


def cosine_similarity(left: Tensor, right: Tensor) -> float:
    denominator = torch.linalg.vector_norm(left.double()) * torch.linalg.vector_norm(right.double())
    if float(denominator) == 0.0:
        return math.nan
    return float(torch.dot(left.double(), right.double()) / denominator)


def summarize_gradient(
    vector: Tensor,
    grouped_vectors: Mapping[str, Tensor],
    *,
    clip_threshold: float,
) -> dict[str, object]:
    norm = vector_norm(vector)
    clip_scale = min(1.0, clip_threshold / norm) if clip_threshold > 0.0 and norm > 0.0 else 1.0
    group_norms = {name: vector_norm(group) for name, group in grouped_vectors.items()}
    squared = norm * norm
    return {
        "total_norm": norm,
        "clip_scale": clip_scale,
        "clipped_norm": norm * clip_scale,
        "group_norms": group_norms,
        "group_squared_norm_fraction": {
            name: (value * value / squared if squared > 0.0 else 0.0)
            for name, value in group_norms.items()
        },
    }


def pairwise_cosines(vectors: Mapping[int, Tensor]) -> dict[str, float]:
    keys = sorted(vectors)
    return {
        f"{left}_vs_{right}": cosine_similarity(vectors[left], vectors[right])
        for index, left in enumerate(keys)
        for right in keys[index + 1 :]
    }
