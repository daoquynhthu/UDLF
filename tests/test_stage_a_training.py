from __future__ import annotations

import json

import torch
from datasets import Dataset, DatasetDict

from udlf.config import UDLFModelConfig
from udlf.data import QueryRecallDataset, RealTokenQueryRecallDataset, RepeatingPatternDataset, TokenDatasetFromDisk
from udlf.model import UDLFStageAModel
from udlf.training.config import train_config_from_dict
from udlf.training.train import (
    _bounded_probe_candidate,
    _build_optimizer,
    _choose_segment_len,
    _parse_nvidia_smi_memory,
    _solver_adapter_metrics,
    _step_batch_schedule,
    run_stage_a,
)


def test_stage_a_training_writes_metrics(tmp_path):
    run_dir = tmp_path / "stage_a"
    config = {
        "mode": "stage-a",
        "device": "cpu",
        "allow_cpu_training": True,
        "vocab_size": 24,
        "seq_len": 8,
        "batch_size": 2,
        "steps": 2,
        "latent_slots": 4,
        "latent_dim": 16,
        "embed_dim": 16,
        "ff_multiplier": 2,
        "latent_heads": 4,
        "readout_heads": 2,
        "solver_steps": 1,
        "diffusion_mode": "ode",
        "dynamics_diagnostics": True,
        "segment_len": 3,
        "eval_every": 1,
        "eval_batches": 1,
        "eval_batch_size": 1,
        "intervention_pair_trials": 2,
        "intervention_mix_alpha": 0.25,
    }

    run_stage_a(config=config, run_dir=run_dir)

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert all(row["stage_a"] is True for row in rows)
    assert "UDLF stage A training finished" in (run_dir / "train.log").read_text(encoding="utf-8")
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "models" / "model_latest.pt").exists()
    assert rows[-1]["eval_batch_size"] == 1
    assert rows[-1]["eval_segment_len"] == 3
    assert rows[-1]["intervention_mix_alpha"] == 0.25
    assert rows[-1]["intervention_pair_trials"] == 2.0
    assert "intervention_mixed_loss" in rows[-1]
    assert "intervention_mixed_delta" in rows[-1]
    assert "intervention_mixed_delta_sem" in rows[-1]
    assert "intervention_mixed_delta_ci95_low" in rows[-1]
    assert "intervention_mixed_delta_ci95_high" in rows[-1]
    assert "intervention_temporal_mixed_loss" in rows[-1]
    assert "intervention_temporal_mixed_delta" in rows[-1]
    assert "dynamics_drift_rms" in rows[-1]
    assert "dynamics_sigma_min" in rows[-1]
    assert "dynamics_sigma_max" in rows[-1]
    assert "dynamics_jump_rms" in rows[-1]
    assert "dynamics_injection_relative_jump" in rows[-1]
    assert "dynamics_injection_alpha_entropy" in rows[-1]
    assert "dynamics_injection_gate_high_saturation" in rows[-1]
    assert "dynamics_injection_state_cosine" in rows[-1]


def test_stage_a_training_resumes_from_checkpoint(tmp_path):
    run_dir = tmp_path / "resume"
    base_config = {
        "mode": "stage-a",
        "device": "cpu",
        "allow_cpu_training": True,
        "vocab_size": 24,
        "seq_len": 8,
        "batch_size": 2,
        "steps": 2,
        "latent_slots": 4,
        "latent_dim": 16,
        "embed_dim": 16,
        "ff_multiplier": 2,
        "latent_heads": 4,
        "readout_heads": 2,
        "solver_steps": 1,
        "diffusion_mode": "ode",
        "async_checkpoint": False,
    }
    run_stage_a(config=base_config, run_dir=run_dir)

    resumed = dict(base_config)
    resumed["max_steps"] = 3
    resumed["resume"] = str(run_dir / "latest.pt")
    run_stage_a(config=resumed, run_dir=run_dir)

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["step"] == 3


def test_stage_a_multisample_prior_writes_path_metrics(tmp_path):
    run_dir = tmp_path / "multisample"
    config = {
        "mode": "stage-a",
        "device": "cpu",
        "allow_cpu_training": True,
        "vocab_size": 24,
        "seq_len": 8,
        "batch_size": 2,
        "steps": 1,
        "latent_slots": 4,
        "latent_dim": 16,
        "embed_dim": 16,
        "ff_multiplier": 2,
        "latent_heads": 4,
        "readout_heads": 2,
        "solver_steps": 1,
        "diffusion_mode": "fixed",
        "fixed_sigma": 0.01,
        "prior_path_samples": 2,
        "prior_state_selection": "mean",
        "async_checkpoint": False,
        "eval_every": 0,
    }

    run_stage_a(config=config, run_dir=run_dir)

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["prior_path_samples"] == 2.0
    assert "prior_path_weight_entropy" in rows[-1]
    assert "prior_path_logprob_gap" in rows[-1]


def test_stage_a_stability_diagnostics_are_opt_in(tmp_path):
    run_dir = tmp_path / "stability"
    config = {
        "mode": "stage-a",
        "device": "cpu",
        "allow_cpu_training": True,
        "vocab_size": 24,
        "seq_len": 8,
        "batch_size": 2,
        "steps": 1,
        "latent_slots": 4,
        "latent_dim": 16,
        "embed_dim": 16,
        "ff_multiplier": 2,
        "latent_heads": 4,
        "readout_heads": 2,
        "solver_steps": 1,
        "diffusion_mode": "ode",
        "dynamics_diagnostics": True,
        "stability_diagnostics": True,
        "stability_diagnostic_every": 1,
        "async_checkpoint": False,
        "eval_every": 0,
    }

    run_stage_a(config=config, run_dir=run_dir)

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "stability_injection_fd_gain" in rows[-1]
    assert "stability_drift_fd_gain" in rows[-1]
    assert "stability_ftle_proxy" in rows[-1]


def test_stage_b_training_writes_posterior_metrics_without_stage_a_flag(tmp_path):
    run_dir = tmp_path / "stage_b"
    config = {
        "mode": "stage-b",
        "architecture": "udlf",
        "device": "cpu",
        "allow_cpu_training": True,
        "vocab_size": 24,
        "seq_len": 8,
        "batch_size": 2,
        "steps": 1,
        "latent_slots": 4,
        "latent_dim": 16,
        "embed_dim": 16,
        "ff_multiplier": 2,
        "latent_heads": 4,
        "readout_heads": 2,
        "solver_steps": 1,
        "diffusion_mode": "fixed",
        "fixed_sigma": 0.01,
        "lambda_prior": 1.0,
        "lambda_posterior": 0.5,
        "lambda_kl": 0.1,
        "posterior_dropout": 0.0,
        "async_checkpoint": False,
        "eval_every": 0,
    }

    run_stage_a(config=config, run_dir=run_dir)

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["stage_a"] is False
    assert rows[-1]["training_mode"] == "stage-b"
    assert "loss_prior" in rows[-1]
    assert "loss_posterior" in rows[-1]
    assert "posterior_kl" in rows[-1]
    assert rows[-1]["posterior_used"] == 1.0
    assert "posterior_prior_state_gap" in rows[-1]


def test_stage_a_training_refuses_accidental_run_overwrite(tmp_path):
    run_dir = tmp_path / "guard"
    config = {
        "mode": "stage-a",
        "device": "cpu",
        "allow_cpu_training": True,
        "vocab_size": 24,
        "seq_len": 8,
        "batch_size": 2,
        "steps": 1,
        "latent_slots": 4,
        "latent_dim": 16,
        "embed_dim": 16,
        "ff_multiplier": 2,
        "latent_heads": 4,
        "readout_heads": 2,
        "solver_steps": 1,
        "diffusion_mode": "ode",
        "async_checkpoint": False,
    }
    run_stage_a(config=config, run_dir=run_dir)

    try:
        run_stage_a(config=config, run_dir=run_dir)
    except RuntimeError as exc:
        assert "refusing to start a fresh training run" in str(exc)
    else:
        raise AssertionError("expected run overwrite guard to fail")


def test_train_config_accepts_parameters_and_steps_alias():
    config = train_config_from_dict(
        {
            "parameters": {
                "mode": "stage-a",
                "steps": 7,
                "batch_size": 3,
            },
            "batch_size": 5,
        }
    )

    assert config.max_steps == 7
    assert config.batch_size == 5


def test_segment_schedule_uses_random_horizons_and_periodic_full_bptt():
    config = train_config_from_dict(
        {
            "seq_len": 512,
            "segment_len": 64,
            "segment_len_min": 64,
            "segment_len_max": 256,
            "segment_len_choices": [256, 64, 128],
            "segment_len_weights": [0.1, 0.6, 0.3],
            "full_bptt_every": 8,
        }
    )
    generator = torch.Generator().manual_seed(19)

    sampled = _choose_segment_len(config, generator, torch.device("cpu"), step=1)
    full = _choose_segment_len(config, generator, torch.device("cpu"), step=8)

    assert config.segment_len_choices == [64, 128, 256]
    assert config.segment_len_weights == [0.6, 0.3, 0.1]
    assert sampled in config.segment_len_choices
    assert full == 0


def test_full_bptt_step_uses_dedicated_micro_batch(tmp_path):
    run_dir = tmp_path / "full_bptt_batch"
    run_stage_a(
        config={
            "mode": "stage-a",
            "device": "cpu",
            "allow_cpu_training": True,
            "vocab_size": 24,
            "seq_len": 8,
            "batch_size": 2,
            "grad_accum_steps": 2,
            "steps": 1,
            "latent_slots": 4,
            "latent_dim": 16,
            "embed_dim": 16,
            "ff_multiplier": 2,
            "latent_heads": 4,
            "readout_heads": 2,
            "solver_steps": 1,
            "diffusion_mode": "ode",
            "segment_len": 3,
            "full_bptt_every": 1,
            "full_bptt_batch_size": 1,
            "async_checkpoint": False,
            "eval_every": 0,
        },
        run_dir=run_dir,
    )

    row = json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["train_full_bptt"] == 1.0
    assert row["train_step_batch_size"] == 1.0
    assert row["train_step_grad_accum"] == 4.0
    assert row["train_step_effective_batch_size"] == 4.0
    assert row["step_seconds"] > 0
    assert row["step_tokens_per_second"] > 0
    heartbeat = json.loads((run_dir / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["status"] == "completed"
    assert heartbeat["step"] == 1


def test_random_horizon_scales_batch_to_constant_activation_budget():
    config = train_config_from_dict(
        {
            "seq_len": 512,
            "batch_size": 64,
            "grad_accum_steps": 1,
            "segment_len": 64,
            "segment_len_min": 64,
            "segment_len_max": 256,
            "segment_len_choices": [64, 128, 256],
            "segment_len_weights": [0.6, 0.3, 0.1],
            "full_bptt_every": 32,
            "full_bptt_batch_size": 12,
        }
    )

    assert _step_batch_schedule(config, 64) == (64, 1)
    assert _step_batch_schedule(config, 128) == (32, 2)
    assert _step_batch_schedule(config, 256) == (16, 4)
    assert _step_batch_schedule(config, 0) == (12, 6)


def test_auto_batch_prediction_cannot_jump_past_probe_increment():
    assert _bounded_probe_candidate(64, best=8, upper=64, max_increment=8) == 16
    assert _bounded_probe_candidate(13, best=8, upper=64, max_increment=8) == 13
    assert _bounded_probe_candidate(64, best=60, upper=64, max_increment=8) == 64


def test_nvidia_smi_memory_parser_reports_actual_free_bytes():
    free_bytes, total_bytes = _parse_nvidia_smi_memory("24564, 12180\n")
    assert total_bytes == 24564 * 1024**2
    assert free_bytes == (24564 - 12180) * 1024**2


def test_solver_adapter_metrics_report_substep_divergence():
    config = UDLFModelConfig(
        vocab_size=32,
        latent_slots=4,
        latent_dim=16,
        embed_dim=16,
        ff_multiplier=2,
        latent_heads=4,
        readout_heads=2,
        solver_steps=2,
        solver_adapter_rank=4,
        diffusion_mode="ode",
    )
    model = UDLFStageAModel(config)
    assert _solver_adapter_metrics(model)["solver_adapter_difference_rms"] == 0.0
    with torch.no_grad():
        model.prior.solver_adapters[0][-1].weight.fill_(0.1)
    metrics = _solver_adapter_metrics(model)
    assert metrics["solver_adapter_output_rms"] > 0.0
    assert metrics["solver_adapter_difference_rms"] > 0.0


def test_segment_choices_reject_invalid_sequence_horizons():
    try:
        train_config_from_dict({"seq_len": 128, "segment_len_choices": [64, 128]})
    except ValueError as exc:
        assert "segment_len_choices" in str(exc)
    else:
        raise AssertionError("expected horizon equal to seq_len to be rejected")


def test_segment_weights_must_match_choices():
    try:
        train_config_from_dict(
            {"seq_len": 512, "segment_len_choices": [64, 128], "segment_len_weights": [1.0]}
        )
    except ValueError as exc:
        assert "segment_len_weights" in str(exc)
    else:
        raise AssertionError("expected mismatched horizon weights to be rejected")


def test_train_config_passes_mamba_official_alignment_parameters():
    config = train_config_from_dict(
        {
            "architecture": "mamba",
            "mamba_dt_min": 0.002,
            "mamba_dt_max": 0.05,
            "mamba_dt_init": "constant",
            "mamba_dt_scale": 0.5,
            "mamba_dt_init_floor": 1e-5,
            "mamba_conv_bias": False,
            "mamba_bias": True,
            "mamba_residual_in_fp32": False,
            "mamba_pad_vocab_size_multiple": 8,
            "mamba_backend": "torch",
        }
    )

    mamba_config = config.mamba_config()

    assert mamba_config.dt_min == 0.002
    assert mamba_config.dt_max == 0.05
    assert mamba_config.dt_init == "constant"
    assert mamba_config.dt_scale == 0.5
    assert mamba_config.dt_init_floor == 1e-5
    assert mamba_config.conv_bias is False
    assert mamba_config.bias is True
    assert mamba_config.residual_in_fp32 is False
    assert mamba_config.pad_vocab_size_multiple == 8
    assert mamba_config.backend == "torch"


def test_mamba_optimizer_excludes_a_and_d_from_weight_decay():
    from udlf.llm import MambaLMConfig, MambaLMModel

    config = train_config_from_dict({"architecture": "mamba", "weight_decay": 0.1})
    model = MambaLMModel(MambaLMConfig(vocab_size=37, d_model=24, n_layers=2, d_state=4))
    optimizer = _build_optimizer(model, config)
    no_decay_ids = {
        id(parameter)
        for group in optimizer.param_groups
        if group["weight_decay"] == 0.0
        for parameter in group["params"]
    }

    for block in model.blocks:
        assert id(block.mixer.A_log) in no_decay_ids
        assert id(block.mixer.D) in no_decay_ids


def test_train_config_rejects_invalid_intervention_mix_alpha():
    try:
        train_config_from_dict({"mode": "stage-a", "intervention_mix_alpha": 1.5})
    except ValueError as exc:
        assert "intervention_mix_alpha" in str(exc)
    else:
        raise AssertionError("expected invalid intervention_mix_alpha to fail")


def test_train_config_rejects_invalid_intervention_pair_trials():
    try:
        train_config_from_dict({"mode": "stage-a", "intervention_pair_trials": 0})
    except ValueError as exc:
        assert "intervention_pair_trials" in str(exc)
    else:
        raise AssertionError("expected invalid intervention_pair_trials to fail")


def test_train_config_rejects_invalid_console_log_mode():
    try:
        train_config_from_dict({"mode": "stage-a", "console_log_mode": "verbose"})
    except ValueError as exc:
        assert "console_log_mode" in str(exc)
    else:
        raise AssertionError("expected invalid console_log_mode to fail")


def test_train_config_rejects_invalid_architecture():
    try:
        train_config_from_dict({"mode": "stage-a", "architecture": "transformer"})
    except ValueError as exc:
        assert "architecture" in str(exc)
    else:
        raise AssertionError("expected invalid architecture to fail")


def test_cpu_training_requires_explicit_override(tmp_path):
    config = {
        "mode": "stage-a",
        "device": "cpu",
        "vocab_size": 24,
        "seq_len": 8,
        "batch_size": 2,
        "steps": 1,
        "latent_slots": 4,
        "latent_dim": 16,
        "embed_dim": 16,
        "ff_multiplier": 2,
        "latent_heads": 4,
        "readout_heads": 2,
        "solver_steps": 1,
        "diffusion_mode": "ode",
        "async_checkpoint": False,
        "eval_every": 0,
    }

    try:
        run_stage_a(config=config, run_dir=tmp_path / "cpu_guard")
    except RuntimeError as exc:
        assert "CPU training is disabled" in str(exc)
    else:
        raise AssertionError("expected CPU training to require explicit override")


def test_mamba_training_writes_metrics(tmp_path):
    run_dir = tmp_path / "mamba"
    config = {
        "mode": "stage-a",
        "architecture": "mamba",
        "device": "cpu",
        "allow_cpu_training": True,
        "vocab_size": 48,
        "seq_len": 8,
        "batch_size": 2,
        "steps": 2,
        "llm_dim": 24,
        "llm_layers": 2,
        "mamba_d_state": 4,
        "mamba_expand": 2,
        "mamba_conv_kernel": 3,
        "learning_rate": 0.001,
        "eval_every": 1,
        "eval_batches": 1,
        "eval_interventions": False,
        "async_checkpoint": False,
        "console_log_mode": "quiet",
    }

    run_stage_a(config=config, run_dir=run_dir)

    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["architecture"] == "mamba"
    assert rows[-1]["mamba_backend"] == "torch"
    assert rows[-1]["parameter_count"] > 0
    assert "eval_loss_lm" in rows[-1]
    assert "state_rms" not in rows[-1]
    assert "intervention_mixed_delta" not in rows[-1]


def test_token_dataset_from_disk_samples_batches(tmp_path):
    dataset_path = tmp_path / "tokens"
    DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": [[1, 2, 3, 4], [5, 6, 7, 8]]}),
            "validation": Dataset.from_dict({"input_ids": [[9, 10, 11, 12]]}),
        }
    ).save_to_disk(str(dataset_path))

    dataset = TokenDatasetFromDisk(str(dataset_path), "train", seq_len=3, seed=1)
    batch = dataset.sample(batch_size=2)

    assert batch.shape == (2, 3)
    assert batch.dtype == torch.long


def test_repeating_pattern_suffix_loss_mask():
    dataset = RepeatingPatternDataset(vocab_size=16, seq_len=8, seed=1, suffix_loss_only=True)

    mask = dataset.loss_mask(batch_size=2)

    assert mask is not None
    assert mask.shape == (2, 7)
    assert mask[:, :3].sum().item() == 0
    assert mask[:, 3:].all()


def test_query_recall_dataset_masks_query_answer_targets():
    dataset = QueryRecallDataset(vocab_size=64, seq_len=18, seed=1)

    batch = dataset.sample(batch_size=4)
    mask = dataset.loss_mask(batch_size=4)

    assert batch.shape == (4, 18)
    assert mask.shape == (4, 17)
    assert dataset.intervention_split() == dataset.memory_len
    query_positions = mask[0].nonzero().flatten().tolist()
    assert query_positions
    for position in query_positions:
        query_token = int(batch[0, position].item())
        answer_token = int(batch[0, position + 1].item())
        memory_index = query_token - dataset.query_base
        assert 0 <= memory_index < dataset.memory_len
        assert answer_token == int(batch[0, memory_index].item())


def test_real_token_query_recall_dataset_uses_saved_tokens(tmp_path):
    dataset_path = tmp_path / "real_tokens"
    DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": [[11, 12, 13, 14, 15, 16], [21, 22, 23, 24, 25, 26]]}),
            "validation": Dataset.from_dict({"input_ids": [[31, 32, 33, 34, 35, 36]]}),
        }
    ).save_to_disk(str(dataset_path))
    dataset = RealTokenQueryRecallDataset(str(dataset_path), "train", seq_len=12, vocab_size=64, seed=1)

    batch = dataset.sample(batch_size=2)
    mask = dataset.loss_mask(batch_size=2)

    assert batch.shape == (2, 12)
    assert mask.shape == (2, 11)
    assert dataset.intervention_split() == dataset.memory_len
    for row_index in range(batch.shape[0]):
        for position in mask[row_index].nonzero().flatten().tolist():
            query_token = int(batch[row_index, position].item())
            answer_token = int(batch[row_index, position + 1].item())
            memory_index = query_token - dataset.query_base
            assert 0 <= memory_index < dataset.memory_len
            assert answer_token == int(batch[row_index, memory_index].item())
