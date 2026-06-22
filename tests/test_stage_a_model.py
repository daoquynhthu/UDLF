from __future__ import annotations

import torch

from udlf.config import UDLFModelConfig
from udlf.llm import MambaLMConfig, MambaLMModel, MambaMixer
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


def test_slot_identity_is_persistent_and_trainable():
    torch.manual_seed(11)
    model = UDLFStageAModel(small_config(diffusion_mode="ode"))

    assert model.slot_identity.requires_grad
    assert 0.8 < float(model.initial_state.detach().std()) < 1.2
    assert model.slot_identity.detach().std() > 0
    assert torch.allclose(
        model.slot_identity_features().pow(2).mean(dim=-1),
        torch.ones(1, model.config.latent_slots),
        atol=1e-5,
    )

    output = model(torch.tensor([[1, 2, 3, 4]]))
    output.loss.backward()

    assert model.slot_identity.grad is not None
    assert torch.isfinite(model.slot_identity.grad).all()
    assert model.slot_identity.grad.abs().sum() > 0


def test_hierarchical_prior_uses_independent_blocks():
    config = UDLFModelConfig(
        vocab_size=32,
        latent_slots=4,
        latent_dim=16,
        embed_dim=16,
        ff_multiplier=2,
        latent_heads=4,
        readout_heads=2,
        prior_depth=3,
        solver_steps=1,
        diffusion_mode="ode",
    )
    model = UDLFStageAModel(config)
    assert len(model.prior.additional_cores) == 2
    assert model.prior.core.u.weight.data_ptr() != model.prior.additional_cores[0].u.weight.data_ptr()
    output = model(torch.randint(0, config.vocab_size, (2, 6)))
    output.loss.backward()
    assert model.prior.core.u.weight.grad is not None
    assert all(core.u.weight.grad is not None for core in model.prior.additional_cores)


def test_prior_depth_one_preserves_legacy_parameter_surface():
    model = UDLFStageAModel(small_config(prior_depth=1))
    keys = model.state_dict()
    assert not any("additional_cores" in key for key in keys)


def test_hierarchical_64m_candidate_is_parameter_matched():
    model = UDLFStageAModel(
        UDLFModelConfig(
            vocab_size=50257,
            latent_slots=16,
            latent_dim=488,
            embed_dim=512,
            ff_multiplier=4,
            latent_heads=8,
            readout_heads=8,
            prior_depth=4,
            solver_steps=1,
            diffusion_mode="ode",
            tie_embeddings=True,
        )
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count == 64_523_673
    assert abs(parameter_count - 64_000_000) / 64_000_000 < 0.01


def test_tied_embedding_initialization_has_language_model_scale():
    torch.manual_seed(12)
    model = UDLFStageAModel(small_config(tie_embeddings=True))

    assert model.output_weight is model.embedding.weight
    assert 0.015 < float(model.embedding.weight.detach().std()) < 0.025


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


def test_mamba_mixer_uses_official_dt_and_decay_parameterization():
    torch.manual_seed(9)
    config = MambaLMConfig(vocab_size=37, d_model=24, d_state=4, expand=2, dt_min=0.002, dt_max=0.05)
    mixer = MambaMixer(config)

    dt = torch.nn.functional.softplus(mixer.dt_proj.bias.detach())

    assert dt.min() >= config.dt_min * 0.999
    assert dt.max() <= config.dt_max * 1.001
    assert getattr(mixer.dt_proj.bias, "_no_reinit", False)
    assert getattr(mixer.A_log, "_no_weight_decay", False)
    assert getattr(mixer.D, "_no_weight_decay", False)


def test_mamba_lm_pads_internal_vocab_but_returns_real_vocab_logits():
    torch.manual_seed(10)
    model = MambaLMModel(
        MambaLMConfig(
            vocab_size=37,
            d_model=24,
            n_layers=2,
            d_state=4,
            expand=2,
            conv_kernel=3,
            pad_vocab_size_multiple=8,
        )
    )
    input_ids = torch.randint(0, model.config.vocab_size, (2, 9))

    output = model(input_ids)

    assert model.embedding.num_embeddings == 40
    assert model.lm_head.out_features == 40
    assert output.logits.shape == (2, 8, 37)
    assert output.loss is not None
    assert torch.isfinite(output.loss)


def test_mamba_forced_fused_backend_does_not_silently_fallback():
    import udlf.llm as llm_module

    if llm_module.causal_conv1d_fn is not None and llm_module.selective_scan_fn is not None:
        return
    try:
        MambaMixer(MambaLMConfig(vocab_size=37, d_model=24, d_state=4, backend="fused"))
    except RuntimeError as exc:
        assert "fused backend requires" in str(exc)
    else:
        raise AssertionError("expected forced fused backend to reject missing kernels")


def test_custom_mamba_kernel_is_restricted_to_supported_state_size():
    import udlf.llm as llm_module

    if llm_module.selective_scan_fn is not None:
        return
    try:
        MambaMixer(MambaLMConfig(vocab_size=37, d_model=24, d_state=8, backend="fused"))
    except RuntimeError as exc:
        assert "fused backend requires" in str(exc)
    else:
        raise AssertionError("expected custom kernel to reject d_state != 16")
