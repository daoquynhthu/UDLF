from __future__ import annotations

import torch

from udlf.config import UDLFModelConfig
from udlf.llm import MambaLMConfig, MambaLMModel
from udlf.model import UDLFStageAModel


def small_config(**overrides):
    values = {
        "vocab_size": 31,
        "latent_slots": 4,
        "latent_dim": 16,
        "embed_dim": 16,
        "ff_multiplier": 2,
        "latent_heads": 4,
        "readout_heads": 2,
        "solver_steps": 2,
        "diffusion_mode": "ode",
    }
    values.update(overrides)
    return UDLFModelConfig(**values)


def test_stage_a_forward_shapes_and_loss():
    torch.manual_seed(1)
    model = UDLFStageAModel(small_config())
    input_ids = torch.randint(0, model.config.vocab_size, (3, 5))

    output = model(input_ids)

    assert output.logits.shape == (3, 4, model.config.vocab_size)
    assert output.final_state.shape == (3, model.config.latent_slots, model.config.latent_dim)
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)


def test_ode_mode_is_reproducible_without_noise():
    torch.manual_seed(2)
    model = UDLFStageAModel(small_config(diffusion_mode="ode"))
    input_ids = torch.tensor([[1, 2, 3, 4]])

    first = model(input_ids).logits
    second = model(input_ids).logits

    assert torch.allclose(first, second)


def test_future_tokens_do_not_affect_earlier_prior_logits_in_ode_mode():
    torch.manual_seed(3)
    model = UDLFStageAModel(small_config(diffusion_mode="ode"))
    a = torch.tensor([[5, 6, 7, 8]])
    b = torch.tensor([[5, 20, 21, 22]])

    logits_a, _ = model.forward_prefix(a[:, :1])
    logits_b, _ = model.forward_prefix(b[:, :1])

    assert torch.allclose(logits_a[:, 0], logits_b[:, 0])


def test_state_dependent_diffusion_forward_accepts_seeded_generator():
    torch.manual_seed(4)
    model = UDLFStageAModel(small_config(diffusion_mode="state_dependent"))
    input_ids = torch.tensor([[1, 2, 3]])
    generator = torch.Generator().manual_seed(123)

    output = model(input_ids, generator=generator)

    assert output.logits.shape == (1, 2, model.config.vocab_size)
    assert torch.isfinite(output.loss)


def test_fixed_diffusion_is_reproducible_with_same_noise_seed():
    torch.manual_seed(5)
    model = UDLFStageAModel(small_config(diffusion_mode="fixed", fixed_sigma=0.02))
    input_ids = torch.tensor([[2, 4, 6, 8]])

    first = model(input_ids, generator=torch.Generator().manual_seed(77)).logits
    second = model(input_ids, generator=torch.Generator().manual_seed(77)).logits

    assert torch.allclose(first, second)


def test_explicit_state_carry_matches_single_prefix_in_ode_mode():
    torch.manual_seed(6)
    model = UDLFStageAModel(small_config(diffusion_mode="ode"))
    input_ids = torch.tensor([[3, 1, 4, 1, 5]])

    full_logits, _ = model.forward_prefix(input_ids)
    first_logits, carried = model.forward_prefix(input_ids[:, :2])
    second_logits, _ = model.forward_prefix(input_ids[:, 2:], state=carried)
    segmented_logits = torch.cat([first_logits, second_logits], dim=1)

    assert torch.allclose(full_logits, segmented_logits, atol=1e-5, rtol=1e-5)


def test_posterior_prefix_keeps_prior_and_posterior_states_separate():
    torch.manual_seed(7)
    model = UDLFStageAModel(small_config(diffusion_mode="ode"), enable_posterior=True)
    input_ids = torch.tensor([[3, 1, 4, 1]])
    targets = torch.tensor([[1, 4, 1, 5]])

    output = model.forward_posterior_prefix(input_ids, targets)

    assert output.prior_logits.shape == (1, 4, model.config.vocab_size)
    assert output.posterior_logits.shape == (1, 4, model.config.vocab_size)
    assert output.prior_final_state.shape == output.posterior_final_state.shape
    assert output.posterior_kl.ndim == 0
    assert torch.isfinite(output.posterior_kl)


def test_mamba_lm_forward_shapes_and_loss():
    torch.manual_seed(8)
    model = MambaLMModel(MambaLMConfig(vocab_size=37, d_model=24, n_layers=2, d_state=4, expand=2, conv_kernel=3))
    input_ids = torch.randint(0, model.config.vocab_size, (2, 9))

    output = model(input_ids)

    assert output.logits.shape == (2, 8, model.config.vocab_size)
    assert output.final_state is None
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
