from __future__ import annotations

import torch

from udlf.attribution import position_bins, training_summary, transform_states, udlf_parameter_breakdown
from udlf.config import UDLFModelConfig
from udlf.model import UDLFStageAModel


def test_state_transforms_preserve_expected_components():
    states = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

    mean_slots = transform_states(states, "mean_slots")
    centered = transform_states(states, "centered_slots")
    shuffled = transform_states(states, "shuffled_slots")

    assert torch.allclose(mean_slots, states.mean(dim=-2, keepdim=True).expand_as(states))
    assert torch.allclose(centered.mean(dim=-2), torch.zeros_like(centered.mean(dim=-2)))
    assert torch.equal(shuffled, states.roll(1, dims=-2))


def test_position_bins_reports_udlf_minus_mamba():
    udlf = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
    mamba = torch.tensor([[1.0, 3.0, 3.0, 5.0]])

    bins = position_bins(udlf, mamba, 2)

    assert bins[0]["udlf_loss"] == 3.0
    assert bins[0]["udlf_minus_mamba"] == 1.0
    assert bins[1]["udlf_minus_mamba"] == 3.0


def test_training_summary_separates_horizons_and_counts_tokens():
    rows = [
        {
            "train_segment_len": 64,
            "train_step_effective_batch_size": 4,
            "grad_norm": 0.5,
            "step_seconds": 2.0,
            "step_tokens_per_second": 100.0,
        },
        {
            "train_segment_len": 64,
            "train_step_effective_batch_size": 4,
            "grad_norm": 1.5,
            "step_seconds": 4.0,
            "step_tokens_per_second": 50.0,
        },
    ]

    summary = training_summary(rows, seq_len=8)

    assert summary["estimated_training_tokens"] == 56
    assert summary["by_horizon"]["64"]["clip_fraction"] == 0.5
    assert summary["by_horizon"]["64"]["mean_step_seconds"] == 3.0


def test_parameter_breakdown_does_not_double_count_tied_output():
    tied = UDLFStageAModel(
        UDLFModelConfig(vocab_size=32, latent_slots=4, latent_dim=16, embed_dim=16, tie_embeddings=True)
    )
    untied = UDLFStageAModel(
        UDLFModelConfig(vocab_size=32, latent_slots=4, latent_dim=16, embed_dim=16, tie_embeddings=False)
    )

    tied_groups = udlf_parameter_breakdown(tied)
    untied_groups = udlf_parameter_breakdown(untied)

    assert tied_groups["output_shared_with_embedding"] is True
    assert tied_groups["output_additional"] == 0
    assert untied_groups["output_shared_with_embedding"] is False
    assert untied_groups["output_additional"] == 32 * 16
