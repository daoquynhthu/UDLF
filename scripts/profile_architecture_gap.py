from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from udlf.llm import MambaLMModel
from udlf.model import UDLFStageAModel
from udlf.training.checkpoint import normalize_state_dict_for_model
from udlf.training.config import train_config_from_dict


def load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = train_config_from_dict(checkpoint["config"])
    model = UDLFStageAModel(config.model_config()) if config.architecture == "udlf" else MambaLMModel(config.mamba_config())
    state_dict = normalize_state_dict_for_model(model, checkpoint["model"])
    if config.architecture == "udlf" and "slot_identity" not in state_dict:
        state_dict["slot_identity"] = torch.zeros_like(model.slot_identity)
    model.load_state_dict(state_dict)
    return model.cuda().train(), config


def event_value(event, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def profile_step(model, batch: torch.Tensor, warmup: int) -> dict:
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(batch).loss
        loss.backward()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
    ) as profiler:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(batch).loss
        loss.backward()
        torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started

    events = list(profiler.key_averages())
    rows = []
    for event in events:
        rows.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "self_cuda_us": event_value(event, "self_device_time_total", "self_cuda_time_total"),
                "cuda_total_us": event_value(event, "device_time_total", "cuda_time_total"),
                "self_cpu_us": event_value(event, "self_cpu_time_total"),
                "cpu_total_us": event_value(event, "cpu_time_total"),
            }
        )
    top_cuda = sorted(rows, key=lambda row: row["self_cuda_us"], reverse=True)[:20]
    top_cpu = sorted(rows, key=lambda row: row["self_cpu_us"], reverse=True)[:20]
    tokens = batch.shape[0] * (batch.shape[1] - 1)
    return {
        "loss": float(loss.detach().cpu()),
        "batch_size": batch.shape[0],
        "seq_len": batch.shape[1] - 1,
        "tokens": tokens,
        "wall_seconds": wall_seconds,
        "tokens_per_second": tokens / wall_seconds,
        "operator_calls": sum(row["calls"] for row in rows),
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        "top_self_cuda": top_cuda,
        "top_self_cpu": top_cpu,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--udlf", type=Path, required=True)
    parser.add_argument("--mamba", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(811)
    batch = torch.randint(0, 50257, (args.batch_size, args.seq_len + 1), device="cuda")
    udlf, _ = load_model(args.udlf)
    report = {"udlf": profile_step(udlf, batch, args.warmup)}
    del udlf
    torch.cuda.empty_cache()
    mamba, _ = load_model(args.mamba)
    report["mamba"] = profile_step(mamba, batch, args.warmup)
    report["wall_time_ratio_udlf_over_mamba"] = report["udlf"]["wall_seconds"] / report["mamba"]["wall_seconds"]
    report["operator_call_ratio_udlf_over_mamba"] = report["udlf"]["operator_calls"] / report["mamba"]["operator_calls"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
