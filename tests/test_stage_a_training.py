from __future__ import annotations

import json

import torch
from datasets import Dataset, DatasetDict

from udlf.data import QueryRecallDataset, RepeatingPatternDataset, TokenDatasetFromDisk
from udlf.training.config import train_config_from_dict
from udlf.training.train import run_stage_a


def test_stage_a_training_writes_metrics(tmp_path):
    run_dir = tmp_path / "stage_a"
    config = {
        "mode": "stage-a",
        "device": "cpu",
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


def test_stage_a_training_resumes_from_checkpoint(tmp_path):
    run_dir = tmp_path / "resume"
    base_config = {
        "mode": "stage-a",
        "device": "cpu",
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


def test_stage_a_training_refuses_accidental_run_overwrite(tmp_path):
    run_dir = tmp_path / "guard"
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
