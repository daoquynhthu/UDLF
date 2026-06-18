"""UDLF training entrypoint."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from udlf.model import UDLFStageAModel
from udlf.training.checkpoint import AsyncCheckpointWriter, build_checkpoint_payload, load_checkpoint, save_payload
from udlf.training.config import UDLFTrainConfig, load_raw_config, train_config_from_dict
from udlf.training.logging_utils import JsonlMetricLogger, metrics_jsonl_to_csv, setup_logger
from udlf.training.runtime import build_datasets, build_scheduler, make_noise_generator, resolve_device, set_seed, write_json


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def run_smoke(run_dir: Path, steps: int, sleep_seconds: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log = run_dir / "train.log"
    metrics_path = run_dir / "metrics.jsonl"
    stop_file = run_dir / "STOP"

    _append(train_log, "UDLF smoke training started\n")
    start = time.time()
    for step in range(1, steps + 1):
        if stop_file.exists():
            _append(train_log, f"STOP file observed at step={step}\n")
            break
        loss = 1.0 / math.sqrt(step)
        elapsed = max(time.time() - start, 1e-9)
        metrics = {
            "step": step,
            "loss_lm": loss,
            "ppl_lm": math.exp(min(loss, 20.0)),
            "tokens_per_second": round(step * 1024 / elapsed, 3),
            "grad_norm": 0.0,
            "smoke": True,
        }
        _append(metrics_path, json.dumps(metrics, sort_keys=True) + "\n")
        _append(train_log, f"step={step} loss_lm={loss:.6f} smoke=true\n")
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    _append(train_log, "UDLF smoke training finished\n")


def _grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().norm().cpu())
        total += value * value
    return math.sqrt(total)


def _autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


def _sample_training_batch(dataset, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch = dataset.sample(batch_size, device=device)
    loss_mask = dataset.loss_mask(batch_size, device=device) if hasattr(dataset, "loss_mask") else None
    return batch, loss_mask


def _masked_sequence_loss(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor | None, vocab_size: int) -> torch.Tensor:
    losses = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none").reshape_as(targets)
    if loss_mask is None:
        return losses.mean()
    selected = losses[loss_mask]
    if selected.numel() == 0:
        raise ValueError("loss_mask selected no target tokens")
    return selected.mean()


def _forward_segmented(
    model: UDLFStageAModel,
    batch: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None = None,
    segment_len: int,
    generator: torch.Generator,
    detach_state_between_segments: bool,
    diagnostics: dict[str, list[torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if segment_len <= 0 or segment_len >= batch.shape[1] - 1:
        logits, final_state = model.forward_prefix(batch[:, :-1], generator=generator, diagnostics=diagnostics)
        loss = _masked_sequence_loss(logits, batch[:, 1:], loss_mask, model.config.vocab_size)
        return loss, final_state

    losses: list[torch.Tensor] = []
    state = None
    last_state = None
    for start in range(0, batch.shape[1] - 1, segment_len):
        end = min(start + segment_len, batch.shape[1] - 1)
        prefix = batch[:, start:end]
        targets = batch[:, start + 1 : end + 1]
        segment_mask = loss_mask[:, start:end] if loss_mask is not None else None
        logits, final_state = model.forward_prefix(prefix, state=state, generator=generator, diagnostics=diagnostics)
        last_state = final_state
        state = final_state.detach() if detach_state_between_segments else final_state
        if segment_mask is not None and not bool(segment_mask.any()):
            continue
        loss = _masked_sequence_loss(logits, targets, segment_mask, model.config.vocab_size)
        losses.append(loss)
    assert last_state is not None
    if not losses:
        raise ValueError("loss_mask selected no target tokens in the sequence")
    return torch.stack(losses).mean(), last_state


def _dynamics_summary(diagnostics: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    if not diagnostics:
        return {}
    summary: dict[str, float] = {}
    for key, values in diagnostics.items():
        if not values:
            continue
        stacked = torch.stack([value.float().detach().cpu() for value in values])
        if key.endswith("_min"):
            summary[f"dynamics_{key}"] = float(stacked.min())
        elif key.endswith("_max"):
            summary[f"dynamics_{key}"] = float(stacked.max())
        else:
            summary[f"dynamics_{key}"] = float(stacked.mean())
    return summary


def _seeded_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _paired_stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        raise ValueError("paired stats require at least one value")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0, mean, mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    sem = math.sqrt(variance / len(values))
    margin = 1.96 * sem
    return mean, sem, mean - margin, mean + margin


def _choose_segment_len(config: UDLFTrainConfig, generator: torch.Generator, device: torch.device) -> int:
    if config.segment_len_min > 0 and config.segment_len_max > 0:
        value = torch.randint(
            config.segment_len_min,
            config.segment_len_max + 1,
            (1,),
            generator=generator,
            device=device,
        )
        return int(value.item())
    return config.segment_len


@torch.no_grad()
def _evaluate_loss(
    model: UDLFStageAModel,
    dataset,
    *,
    batch_size: int,
    batches: int,
    device: torch.device,
    generator: torch.Generator,
    use_amp: bool,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        batch, loss_mask = _sample_training_batch(dataset, batch_size, device)
        with _autocast_context(device, use_amp):
            loss, _ = _forward_segmented(
                model,
                batch,
                loss_mask=loss_mask,
                segment_len=0,
                generator=generator,
                detach_state_between_segments=True,
            )
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def _evaluate_interventions(
    model: UDLFStageAModel,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
    use_amp: bool,
    shift_tokens: int,
    pair_trials: int,
    perturb_std: float,
    perturb_trials: int,
    mix_alpha: float,
) -> dict[str, float]:
    model.eval()
    batch_size = max(2, batch_size)
    batch, loss_mask = _sample_training_batch(dataset, batch_size, device)
    split = dataset.intervention_split() if hasattr(dataset, "intervention_split") else max(1, batch.shape[1] // 2)
    if split >= batch.shape[1] - 1:
        split = batch.shape[1] - 2
    context = batch[:, :split]
    suffix_prefix = batch[:, split:-1]
    suffix_targets = batch[:, split + 1 :]
    suffix_mask = loss_mask[:, split:] if loss_mask is not None else None

    with _autocast_context(device, use_amp):
        _, state = model.forward_prefix(context, generator=generator)
        if context.shape[1] > shift_tokens:
            _, shifted_state = model.forward_prefix(context[:, :-shift_tokens], generator=generator)
        else:
            shifted_state = torch.zeros_like(state)

        pair_seeds = torch.randint(
            0,
            2_147_483_647,
            (max(1, pair_trials),),
            generator=generator,
            device=device,
            dtype=torch.long,
        ).detach().cpu().tolist()

        def loss_for(candidate_state: torch.Tensor, suffix_seed: int) -> float:
            suffix_generator = _seeded_generator(device, suffix_seed)
            logits, _ = model.forward_prefix(suffix_prefix, state=candidate_state, generator=suffix_generator)
            loss = _masked_sequence_loss(logits, suffix_targets, suffix_mask, model.config.vocab_size)
            return float(loss.detach().cpu())

        pair_rows: list[dict[str, float]] = []
        for pair_index, suffix_seed in enumerate(pair_seeds):
            correct = loss_for(state, suffix_seed)
            zero = loss_for(torch.zeros_like(state), suffix_seed)
            swapped = loss_for(state.flip(0), suffix_seed)
            shifted = loss_for(shifted_state, suffix_seed)
            mixed = loss_for(state.lerp(state.flip(0), mix_alpha), suffix_seed)
            temporal_mixed = loss_for(state.lerp(shifted_state, mix_alpha), suffix_seed)
            attenuated = loss_for(state * 0.5, suffix_seed)
            inverted = loss_for(-state, suffix_seed)
            perturb_losses = []
            for perturb_index in range(max(1, perturb_trials)):
                perturb_generator = _seeded_generator(device, suffix_seed + 1_000_003 * (perturb_index + 1) + 97 * pair_index)
                noise = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=perturb_generator)
                perturb_losses.append(loss_for(state + perturb_std * noise, suffix_seed))
            perturbed = sum(perturb_losses) / len(perturb_losses)
            pair_rows.append(
                {
                    "correct": correct,
                    "zero": zero,
                    "swapped": swapped,
                    "shifted": shifted,
                    "mixed": mixed,
                    "temporal_mixed": temporal_mixed,
                    "perturbed": perturbed,
                    "perturbed_min": min(perturb_losses),
                    "perturbed_max": max(perturb_losses),
                    "attenuated": attenuated,
                    "inverted": inverted,
                    "zero_delta": zero - correct,
                    "swapped_delta": swapped - correct,
                    "shifted_delta": shifted - correct,
                    "mixed_delta": mixed - correct,
                    "temporal_mixed_delta": temporal_mixed - correct,
                    "perturbed_delta": perturbed - correct,
                    "perturbed_min_delta": min(perturb_losses) - correct,
                    "perturbed_max_delta": max(perturb_losses) - correct,
                    "attenuated_delta": attenuated - correct,
                    "inverted_delta": inverted - correct,
                }
            )
    model.train()
    metrics: dict[str, float] = {
        "intervention_pair_trials": float(len(pair_rows)),
        "intervention_perturb_std": perturb_std,
        "intervention_perturb_trials": float(max(1, perturb_trials)),
        "intervention_shift_tokens": float(shift_tokens),
        "intervention_mix_alpha": mix_alpha,
    }
    names = [
        "correct",
        "zero",
        "swapped",
        "shifted",
        "mixed",
        "temporal_mixed",
        "perturbed",
        "perturbed_min",
        "perturbed_max",
        "attenuated",
        "inverted",
    ]
    for name in names:
        metrics[f"intervention_{name}_loss"] = sum(row[name] for row in pair_rows) / len(pair_rows)
    delta_names = [
        "zero_delta",
        "swapped_delta",
        "shifted_delta",
        "mixed_delta",
        "temporal_mixed_delta",
        "perturbed_delta",
        "perturbed_min_delta",
        "perturbed_max_delta",
        "attenuated_delta",
        "inverted_delta",
    ]
    for name in delta_names:
        mean, sem, ci_low, ci_high = _paired_stats([row[name] for row in pair_rows])
        metric_name = f"intervention_{name}"
        metrics[metric_name] = mean
        metrics[f"{metric_name}_sem"] = sem
        metrics[f"{metric_name}_ci95_low"] = ci_low
        metrics[f"{metric_name}_ci95_high"] = ci_high
    return metrics


def _checkpoint_jobs(
    *,
    run_dir: Path,
    models_dir: Path,
    model: UDLFStageAModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: UDLFTrainConfig,
    metrics: dict[str, Any],
    best: bool = False,
) -> list[tuple[Path, dict[str, Any]]]:
    full = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        step=step,
        config=config.to_dict(),
        metrics=metrics,
        checkpoint_kind="full",
    )
    model_only = build_checkpoint_payload(
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        step=step,
        config=config.to_dict(),
        metrics=metrics,
        checkpoint_kind="model_only",
    )
    jobs = [
        (run_dir / "latest.pt", full),
        (models_dir / "model_latest.pt", model_only),
    ]
    if best:
        jobs.extend([(run_dir / "best.pt", full), (models_dir / "model_best.pt", model_only)])
    return jobs


def _save_jobs(writer: AsyncCheckpointWriter | None, jobs: list[tuple[Path, dict[str, Any]]]) -> None:
    if writer is None:
        for path, payload in jobs:
            save_payload(path, payload)
        return
    for path, payload in jobs:
        writer.submit(path, payload)


def run_stage_a(config: dict[str, Any] | UDLFTrainConfig, run_dir: Path | None = None) -> None:
    train_config = config if isinstance(config, UDLFTrainConfig) else train_config_from_dict(config)
    if run_dir is not None:
        train_config.run_dir = str(run_dir)
    run_dir = train_config.resolved_run_dir()
    models_dir = run_dir / "models"
    existing_artifacts = [run_dir / "latest.pt", run_dir / "metrics.jsonl", models_dir / "model_latest.pt"]
    if not train_config.resume and not train_config.allow_run_overwrite and any(path.exists() for path in existing_artifacts):
        found = ", ".join(str(path) for path in existing_artifacts if path.exists())
        raise RuntimeError(
            "refusing to start a fresh training run in a non-empty run_dir; "
            f"found existing artifacts: {found}. Set resume to continue or allow_run_overwrite=true to replace."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    set_seed(train_config.seed)
    device = resolve_device(train_config.device)
    model_config = train_config.model_config()
    model = UDLFStageAModel(model_config).to(device)
    if train_config.compile_model:
        model = torch.compile(model)  # type: ignore[assignment]

    train_dataset, eval_dataset = build_datasets(train_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(train_config.beta1, train_config.beta2),
    )
    scheduler = build_scheduler(
        optimizer,
        max_steps=train_config.max_steps,
        warmup_steps=train_config.warmup_steps,
        min_lr_ratio=train_config.min_lr_ratio,
    )
    start_step = 0
    if train_config.resume:
        start_step = load_checkpoint(
            Path(train_config.resume),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            strict=train_config.strict_resume,
        )

    noise_seed = train_config.noise_seed if train_config.noise_seed is not None else train_config.seed + 2
    noise_generator = make_noise_generator(device, noise_seed)
    logger = setup_logger(run_dir)
    metric_logger = JsonlMetricLogger(
        run_dir / "metrics.jsonl",
        flush_every=train_config.metrics_flush_every,
        fsync_every=train_config.metrics_fsync_every,
    )
    writer = (
        AsyncCheckpointWriter(max_queue=train_config.async_checkpoint_queue)
        if train_config.async_checkpoint
        else None
    )
    stop_file = Path(train_config.stop_file) if train_config.stop_file else run_dir / "STOP"
    write_json(run_dir / "config.json", train_config.to_dict())
    best_eval = float("inf")
    last_metrics: dict[str, Any] = {"step": start_step}

    logger.info(
        "UDLF stage A training started device=%s amp=%s data=%s resume_step=%d",
        device,
        train_config.amp and device.type == "cuda",
        "disk" if train_config.data_path else "synthetic",
        start_step,
    )
    start_time = time.time()
    step = start_step
    try:
        for step in range(start_step + 1, train_config.max_steps + 1):
            if step % train_config.stop_check_every == 0 and stop_file.exists():
                logger.info("STOP file observed at step=%d", step)
                break

            optimizer.zero_grad(set_to_none=True)
            final_state = None
            loss = None
            batch_tokens = 0
            dynamics_diagnostics: dict[str, list[torch.Tensor]] = {}
            for accum_index in range(train_config.grad_accum_steps):
                batch, loss_mask = _sample_training_batch(train_dataset, train_config.batch_size, device)
                batch_tokens += train_config.batch_size * (batch.shape[1] - 1)
                with _autocast_context(device, train_config.amp):
                    micro_loss, final_state = _forward_segmented(
                        model,
                        batch,
                        loss_mask=loss_mask,
                        segment_len=_choose_segment_len(train_config, noise_generator, device),
                        generator=noise_generator,
                        detach_state_between_segments=train_config.detach_state_between_segments,
                        diagnostics=dynamics_diagnostics,
                    )
                    scaled_loss = micro_loss / train_config.grad_accum_steps
                scaled_loss.backward()
                loss = micro_loss if loss is None else loss + micro_loss.detach()

            assert loss is not None
            assert final_state is not None
            loss_for_metrics = loss / train_config.grad_accum_steps
            grad_norm = _grad_norm(model.parameters())
            if train_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            elapsed = max(time.time() - start_time, 1e-9)
            current_loss = float(loss_for_metrics.detach().cpu())
            metrics: dict[str, Any] = {
                "step": step,
                "loss_lm": current_loss,
                "ppl_lm": float(math.exp(min(current_loss, 20.0))),
                "tokens_per_second": round((step - start_step) * batch_tokens / elapsed, 3),
                "grad_norm": grad_norm,
                "state_rms": float(final_state.detach().pow(2).mean().sqrt().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "diffusion_mode": model_config.diffusion_mode,
                "device": device.type,
                "amp": train_config.amp and device.type == "cuda",
                "stage_a": True,
            }
            metrics.update(_dynamics_summary(dynamics_diagnostics))
            if device.type == "cuda":
                metrics["cuda_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 3)

            save_best = False
            if train_config.eval_every > 0 and step % train_config.eval_every == 0:
                eval_loss = _evaluate_loss(
                    model,
                    eval_dataset,
                    batch_size=train_config.batch_size,
                    batches=train_config.eval_batches,
                    device=device,
                    generator=noise_generator,
                    use_amp=train_config.amp,
                )
                metrics["eval_loss_lm"] = eval_loss
                metrics["eval_ppl_lm"] = math.exp(min(eval_loss, 20.0))
                metrics.update(
                    _evaluate_interventions(
                        model,
                        eval_dataset,
                        batch_size=train_config.batch_size,
                        device=device,
                        generator=noise_generator,
                        use_amp=train_config.amp,
                        shift_tokens=train_config.intervention_shift_tokens,
                        pair_trials=train_config.intervention_pair_trials,
                        perturb_std=train_config.intervention_perturb_std,
                        perturb_trials=train_config.intervention_perturb_trials,
                        mix_alpha=train_config.intervention_mix_alpha,
                    )
                )
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    save_best = True

            metric_logger.write(metrics, force_sync=save_best)
            last_metrics = metrics
            if step % train_config.log_every == 0:
                logger.info(
                    "step=%d loss_lm=%.6f grad_norm=%.6f state_rms=%.6f tok_s=%.1f",
                    step,
                    metrics["loss_lm"],
                    metrics["grad_norm"],
                    metrics["state_rms"],
                    metrics["tokens_per_second"],
                )

            save_latest = (
                train_config.save_every > 0 and step % train_config.save_every == 0
            ) or (
                train_config.latest_every > 0 and step % train_config.latest_every == 0
            )
            if save_latest or save_best:
                metric_logger.flush(force_sync=save_best)
                _save_jobs(
                    writer,
                    _checkpoint_jobs(
                        run_dir=run_dir,
                        models_dir=models_dir,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        config=train_config,
                        metrics=metrics,
                        best=save_best,
                    ),
                )

        metric_logger.flush(force_sync=True)
        _save_jobs(
            writer,
            _checkpoint_jobs(
                run_dir=run_dir,
                models_dir=models_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                config=train_config,
                metrics=last_metrics,
                best=best_eval == float("inf"),
            ),
        )
        if writer is not None:
            writer.close()
        metric_logger.close()
        metrics_jsonl_to_csv(run_dir / "metrics.jsonl")
        logger.info("UDLF stage A training finished step=%d", step)
    except BaseException:
        try:
            failed_payload = build_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                step=step,
                config=train_config.to_dict(),
                metrics=last_metrics,
                checkpoint_kind="full",
            )
            save_payload(run_dir / "failed.pt", failed_payload)
        finally:
            if writer is not None:
                writer.close()
            metric_logger.close()
            metrics_jsonl_to_csv(run_dir / "metrics.jsonl")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--mode", choices=["smoke", "stage-a"], default="")
    args = parser.parse_args(argv)

    raw_config = load_raw_config(args.config)
    if args.mode:
        raw_config["mode"] = args.mode
    if args.run_dir:
        raw_config["run_dir"] = args.run_dir
    if "steps" not in raw_config and "max_steps" not in raw_config:
        raw_config["steps"] = args.steps
    if "sleep_seconds" not in raw_config:
        raw_config["sleep_seconds"] = args.sleep_seconds

    mode = str(raw_config.get("mode") or raw_config.get("parameters", {}).get("mode") or "smoke")
    if mode == "smoke":
        steps = int(raw_config.get("steps") or raw_config.get("parameters", {}).get("steps") or args.steps)
        if steps < 1:
            raise SystemExit("--steps must be >= 1")
        run_dir = Path(raw_config.get("run_dir") or raw_config.get("parameters", {}).get("run_dir") or "runs/udlf_smoke").resolve()
        sleep_seconds = float(raw_config.get("sleep_seconds") or raw_config.get("parameters", {}).get("sleep_seconds") or 0.0)
        run_smoke(run_dir=run_dir, steps=steps, sleep_seconds=sleep_seconds)
        return 0

    train_config = train_config_from_dict(raw_config)
    run_stage_a(train_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
