import math

import torch
from torch import nn

from udlf.gradient_diagnostics import (
    collect_gradient_vector,
    cosine_similarity,
    parameter_group,
    pairwise_cosines,
    summarize_gradient,
)


def test_gradient_summary_groups_parameters_and_computes_clip_scale():
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    model(torch.ones(1, 2)).sum().backward()
    vector, groups = collect_gradient_vector(model)
    summary = summarize_gradient(vector, groups, clip_threshold=0.5)
    assert summary["total_norm"] > 0.5
    assert math.isclose(summary["clipped_norm"], 0.5, rel_tol=1e-6)
    assert math.isclose(sum(summary["group_squared_norm_fraction"].values()), 1.0, rel_tol=1e-6)


def test_gradient_cosines_report_direction_agreement():
    left = torch.tensor([1.0, 2.0])
    same = 2.0 * left
    opposite = -left
    assert math.isclose(cosine_similarity(left, same), 1.0, rel_tol=1e-6)
    assert math.isclose(cosine_similarity(left, opposite), -1.0, rel_tol=1e-6)
    assert math.isclose(pairwise_cosines({64: left, 128: same})["64_vs_128"], 1.0, rel_tol=1e-6)


def test_tied_output_weight_is_accounted_as_embedding_capacity():
    assert parameter_group("output_weight") == "embedding"
