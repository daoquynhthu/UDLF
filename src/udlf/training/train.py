"""UDLF training entrypoint."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from udlf.llm import MambaLMModel
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
    total = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().norm()
        total = value * value if total is None else total + value * value
    if total is None:
        return 0.0
    return float(total.sqrt().cpu())


@torch.no_grad()
def _slot_geometry_metrics(state: torch.Tensor) -> dict[str, float]:
    state = state.detach().float()
    normalized = F.normalize(state, dim=-1)
    cosine = normalized @ normalized.transpose(-1, -2)
    slots = state.shape[-2]
    if slots > 1:
        off_diagonal = (cosine.sum(dim=(-2, -1)) - slots) / (slots * (slots - 1))
        pair_cosine = off_diagonal.mean()
    else:
        pair_cosine = cosine.new_ones(())

    centered = state - state.mean(dim=-2, keepdim=True)
    gram = centered @ centered.transpose(-1, -2)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    participation_rank = eigenvalues.sum(dim=-1).square() / eigenvalues.square().sum(dim=-1).clamp_min(1e-12)
    return {
        "slot_pair_cosine": float(pair_cosine.cpu()),
        "slot_centered_rms": float(centered.pow(2).mean().sqrt().cpu()),
        "slot_participation_rank": float(participation_rank.mean().cpu()),
    }


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _build_optimizer(model: nn.Module, config: UDLFTrainConfig) -> torch.optim.AdamW:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if getattr(parameter, "_no_weight_decay", False):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )


def _autocast_context(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


def _build_model(train_config: UDLFTrainConfig, device: torch.device) -> tuple[nn.Module, Any]:
    if train_config.architecture == "mamba":
        model_config = train_config.mamba_config()
        return MambaLMModel(model_config).to(device), model_config
    model_config = train_config.model_config()
    return UDLFStageAModel(model_config, enable_posterior=train_config.mode == "stage-b").to(device), model_config


def _clear_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        if torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats(device)


def _configure_cuda_memory_cap(config: UDLFTrainConfig, device: torch.device, logger) -> int:
    if device.type != "cuda" or not config.enforce_cuda_memory_cap:
        return 0
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    cap_bytes = int(free_bytes * config.vram_fraction)
    cap_fraction = min(1.0, cap_bytes / total_bytes)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(cap_fraction, device_index)
    logger.info(
        "CUDA allocator cap set available=%.2fGiB total=%.2fGiB cap=%.2fGiB fraction=%.4f",
        free_bytes / 1024**3,
        total_bytes / 1024**3,
        cap_bytes / 1024**3,
        cap_fraction,
    )
    return cap_bytes


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


def _prior_multisample_forward(
    model: UDLFStageAModel,
    batch: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None,
    path_samples: int,
    state_selection: str,
    generator: torch.Generator,
    diagnostics: dict[str, list[torch.Tensor]] | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    targets = batch[:, 1:]
    states: list[torch.Tensor] = []
    token_log_probs: list[torch.Tensor] = []
    for _ in range(path_samples):
        logits, final_state = model.forward_prefix(batch[:, :-1], generator=generator, diagnostics=diagnostics)
        log_prob = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        token_log_probs.append(log_prob)
        states.append(final_state)
    stacked_log_probs = torch.stack(token_log_probs, dim=0)
    marginal_log_probs = torch.logsumexp(stacked_log_probs, dim=0) - math.log(path_samples)
    if loss_mask is not None:
        selected = marginal_log_probs[loss_mask]
        if selected.numel() == 0:
            raise ValueError("loss_mask selected no target tokens")
        loss = -selected.mean()
    else:
        loss = -marginal_log_probs.mean()
    if state_selection == "mean":
        final_state = torch.stack(states, dim=0).mean(dim=0)
    else:
        final_state = states[0]
    weights = torch.softmax(stacked_log_probs.detach(), dim=0)
    metrics = {
        "prior_path_samples": float(path_samples),
        "prior_path_weight_entropy": float((-(weights * weights.clamp_min(1e-12).log()).sum(dim=0).mean()).detach().cpu()),
        "prior_path_logprob_gap": float((stacked_log_probs.max(dim=0).values - stacked_log_probs.min(dim=0).values).mean().detach().cpu()),
    }
    return loss, final_state, metrics


def _stage_b_forward(
    model: UDLFStageAModel,
    batch: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None,
    generator: torch.Generator,
    use_posterior: bool,
    diagnostics: dict[str, list[torch.Tensor]] | None,
    config: UDLFTrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    targets = batch[:, 1:]
    prior_generator = generator
    posterior_generator = generator
    output = model.forward_posterior_prefix(
        batch[:, :-1],
        targets,
        prior_generator=prior_generator,
        posterior_generator=posterior_generator,
        diagnostics=diagnostics,
    )
    prior_loss = _masked_sequence_loss(output.prior_logits, targets, loss_mask, model.config.vocab_size)
    posterior_loss = _masked_sequence_loss(output.posterior_logits, targets, loss_mask, model.config.vocab_size)
    posterior_weight = 0.0
    if use_posterior:
        posterior_weight = config.lambda_posterior / max(1e-8, 1.0 - config.posterior_dropout)
    total = config.lambda_prior * prior_loss
    if use_posterior:
        total = total + posterior_weight * (posterior_loss + config.lambda_kl * output.posterior_kl)
    metrics = {
        "loss_prior": float(prior_loss.detach().cpu()),
        "loss_posterior": float(posterior_loss.detach().cpu()),
        "posterior_kl": float(output.posterior_kl.detach().cpu()),
        "posterior_used": float(use_posterior),
        "posterior_weight": float(posterior_weight),
        "posterior_prior_state_gap": float((output.posterior_final_state - output.prior_final_state).detach().pow(2).mean().sqrt().cpu()),
    }
    return total, output.prior_final_state, metrics


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


def _segment_slices(batch: torch.Tensor, loss_mask: torch.Tensor | None, segment_len: int) -> list[tuple[int, int]]:
    ranges = [(start, min(start + segment_len, batch.shape[1] - 1)) for start in range(0, batch.shape[1] - 1, segment_len)]
    if loss_mask is None:
        return ranges
    return [(start, end) for start, end in ranges if bool(loss_mask[:, start:end].any())]


def _backward_segmented(
    model: UDLFStageAModel,
    batch: torch.Tensor,
    *,
    loss_mask: torch.Tensor | None,
    segment_len: int,
    generator: torch.Generator,
    diagnostics: dict[str, list[torch.Tensor]] | None,
    grad_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if segment_len <= 0 or segment_len >= batch.shape[1] - 1:
        loss, final_state = _forward_segmented(
            model,
            batch,
            loss_mask=loss_mask,
            segment_len=segment_len,
            generator=generator,
            detach_state_between_segments=True,
            diagnostics=diagnostics,
        )
        (loss * grad_scale).backward()
        return loss.detach(), final_state

    ranges = _segment_slices(batch, loss_mask, segment_len)
    if not ranges:
        raise ValueError("loss_mask selected no target tokens in the sequence")
    state = None
    final_state = None
    loss_values: list[torch.Tensor] = []
    segment_scale = grad_scale / len(ranges)
    for start, end in ranges:
        prefix = batch[:, start:end]
        targets = batch[:, start + 1 : end + 1]
        segment_mask = loss_mask[:, start:end] if loss_mask is not None else None
        logits, final_state = model.forward_prefix(prefix, state=state, generator=generator, diagnostics=diagnostics)
        state = final_state.detach()
        loss = _masked_sequence_loss(logits, targets, segment_mask, model.config.vocab_size)
        (loss * segment_scale).backward()
        loss_values.append(loss.detach())
    assert final_state is not None
    return torch.stack(loss_values).mean(), final_state.detach()


def _forward_causal_lm(model: nn.Module, batch: torch.Tensor, loss_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = model(batch)
    if output.loss is None:
        raise ValueError("causal LM model did not return a loss")
    if loss_mask is None:
        return output.loss, output.final_state
    logits = output.logits
    targets = batch[:, 1:]
    return _masked_sequence_loss(logits, targets, loss_mask, logits.shape[-1]), output.final_state


def _probe_auto_batch_size(train_config: UDLFTrainConfig, device: torch.device, logger) -> None:
    if not train_config.auto_batch:
        return
    if device.type != "cuda":
        logger.info("auto-batch skipped: device is not cuda")
        return

    original_batch = train_config.batch_size
    original_accum = train_config.grad_accum_steps
    target_micro_batches = max(1, original_batch * original_accum)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    budget_bytes = int(free_bytes * train_config.vram_fraction)
    selection_budget_bytes = budget_bytes
    probe_budget_bytes = int(selection_budget_bytes * train_config.auto_batch_probe_budget_fraction)
    upper = max(1, train_config.auto_batch_max)
    best = 0
    tried: dict[int, tuple[bool, int]] = {}

    logger.info(
        "auto-batch probing: gpu=%s total_vram=%.2fGiB free_vram=%.2fGiB budget=%.2fGiB select_budget=%.2fGiB probe_budget=%.2fGiB max_batch=%d",
        torch.cuda.get_device_name(device),
        total_bytes / 1024**3,
        free_bytes / 1024**3,
        budget_bytes / 1024**3,
        selection_budget_bytes / 1024**3,
        probe_budget_bytes / 1024**3,
        upper,
    )

    def successful_points() -> list[tuple[int, int]]:
        return sorted((batch, peak) for batch, (ok, peak) in tried.items() if ok and peak > 0)

    def memory_model() -> tuple[float, float] | None:
        successful = successful_points()
        if len(successful) < 2:
            return None
        (b1, p1), (b2, p2) = successful[-2], successful[-1]
        if b2 <= b1:
            return None
        slope = max(0.0, (p2 - p1) / (b2 - b1))
        if slope <= 0:
            return None
        intercept = p2 - slope * b2
        return intercept, slope

    def predicted_safe_cap() -> int:
        model = memory_model()
        if model is None:
            return upper
        intercept, slope = model
        raw_budget = selection_budget_bytes / train_config.auto_batch_predict_safety
        cap = math.floor((raw_budget - intercept) / slope)
        cap = max(best, min(upper, cap))
        logger.info(
            "auto-batch predicted safe cap batch=%d intercept=%.2fGiB slope=%.3fGiB/batch safety=%.2f",
            cap,
            intercept / 1024**3,
            slope / 1024**3,
            train_config.auto_batch_predict_safety,
        )
        return cap

    def try_batch(batch_size: int) -> tuple[bool, int]:
        if batch_size in tried:
            return tried[batch_size]

        _clear_cuda(device)
        model: nn.Module | None = None
        optimizer: torch.optim.Optimizer | None = None
        batch: torch.Tensor | None = None
        loss = None
        final_state = None
        ok = False
        peak = 0
        peak_allocated = 0
        peak_reserved = 0
        logger.info("auto-batch probe start batch=%d", batch_size)
        try:
            model, _model_config = _build_model(train_config, device)
            model.train()
            optimizer = _build_optimizer(model, train_config)
            batch = torch.randint(0, train_config.vocab_size, (batch_size, train_config.seq_len), device=device)
            generator = make_noise_generator(device, train_config.seed + 12345)
            with _autocast_context(device, train_config.amp):
                if isinstance(model, UDLFStageAModel):
                    probe_segment_len = (
                        0
                        if train_config.full_bptt_every > 0 and train_config.full_bptt_batch_size == 0
                        else train_config.segment_len
                    )
                    if probe_segment_len > 0 and train_config.detach_state_between_segments:
                        loss, final_state = _backward_segmented(
                            model,
                            batch,
                            loss_mask=None,
                            segment_len=probe_segment_len,
                            generator=generator,
                            diagnostics=None,
                            grad_scale=1.0,
                        )
                    else:
                        loss, final_state = _forward_segmented(
                            model,
                            batch,
                            loss_mask=None,
                            segment_len=probe_segment_len,
                            generator=generator,
                            detach_state_between_segments=train_config.detach_state_between_segments,
                            diagnostics=None,
                        )
                        loss.backward()
                else:
                    loss, final_state = _forward_causal_lm(model, batch, None)
                    loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            peak = max(peak_allocated, peak_reserved)
            ok = peak <= budget_bytes
        except torch.OutOfMemoryError:
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            peak = max(peak_allocated, peak_reserved)
            ok = False
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            peak = max(peak_allocated, peak_reserved)
            ok = False
        finally:
            del model, optimizer, batch, loss, final_state
            _clear_cuda(device)

        tried[batch_size] = (ok, peak)
        logger.info(
            "auto-batch probe batch=%d ok=%s peak=%.2fGiB allocated=%.2fGiB reserved=%.2fGiB",
            batch_size,
            ok,
            peak / 1024**3,
            peak_allocated / 1024**3,
            peak_reserved / 1024**3,
        )
        return tried[batch_size]

    failed_upper: int | None = None
    first = min(max(1, original_batch), upper)
    ok, peak = try_batch(first)
    if ok and peak <= selection_budget_bytes:
        best = first
    else:
        failed_upper = first
        if ok:
            logger.info(
                "auto-batch first candidate batch=%d peak=%.2fGiB exceeds select_budget=%.2fGiB",
                first,
                peak / 1024**3,
                selection_budget_bytes / 1024**3,
            )

    if best > 0 and best < upper:
        anchor = min(upper, max(best + 1, best * 2))
        ok, peak = try_batch(anchor)
        if ok and peak <= selection_budget_bytes:
            best = anchor
        else:
            failed_upper = anchor
            if ok:
                logger.info(
                    "auto-batch anchor batch=%d peak=%.2fGiB exceeds select_budget=%.2fGiB",
                    anchor,
                    peak / 1024**3,
                    selection_budget_bytes / 1024**3,
                )

    if failed_upper is not None:
        high = failed_upper - 1
    else:
        predicted_safe_cap()
        high = upper
    low = best + 1
    while low <= high:
        probe_high = high
        if best > 0:
            probe_high = min(probe_high, best + train_config.auto_batch_max_probe_increment)
        mid = (low + probe_high) // 2
        ok, peak = try_batch(mid)
        if ok and peak <= selection_budget_bytes:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    if best <= 0:
        logger.warning("auto-batch could not fit requested batch=%d; falling back to batch=1", original_batch)
        best = 1
    train_config.batch_size = best
    if train_config.auto_adjust_grad_accum:
        train_config.grad_accum_steps = max(1, math.ceil(target_micro_batches / best))
    logger.info(
        "auto-batch selected batch_size=%d grad_accum_steps=%d effective_micro_batches=%d original_effective_micro_batches=%d",
        train_config.batch_size,
        train_config.grad_accum_steps,
        train_config.batch_size * train_config.grad_accum_steps,
        target_micro_batches,
    )


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


def _unit_like(tensor: torch.Tensor) -> torch.Tensor:
    direction = torch.randn_like(tensor)
    return direction / (direction.norm() + 1e-12)


def _stability_diagnostics(
    model: UDLFStageAModel,
    batch: torch.Tensor,
    *,
    eps: float,
) -> dict[str, float]:
    model_was_training = model.training
    model.eval()
    sample = batch[:1, :1]
    token_embed = model.embedding(sample[:, 0]).detach()
    state = model.init_state(1, device=batch.device, dtype=model.embedding.weight.dtype).detach()

    with torch.no_grad():
        injection_direction = _unit_like(state)
        identity = model.slot_identity_features()
        injected = model.inject(state, token_embed, identity)
        injected_perturbed = model.inject(state + eps * injection_direction, token_embed, identity)
        injection_gain = (injected_perturbed - injected).norm() / (eps * injection_direction.norm() + 1e-12)

        drift_direction = _unit_like(injected)
        drift_base, _ = model.prior.drift_and_sigma(injected, token_embed, identity)
        drift_perturbed, _ = model.prior.drift_and_sigma(injected + eps * drift_direction, token_embed, identity)
        drift_gain = (drift_perturbed - drift_base).norm() / (eps * drift_direction.norm() + 1e-12)

        base = injected.detach()
        direction = _unit_like(base)
        perturbed = base + eps * direction
        ds = 1.0 / model.config.solver_steps
        z_base = base
        z_perturbed = perturbed
        for _ in range(model.config.solver_steps):
            drift_base, _ = model.prior.drift_and_sigma(z_base, token_embed, identity)
            drift_perturbed, _ = model.prior.drift_and_sigma(z_perturbed, token_embed, identity)
            z_base = z_base + drift_base * ds
            z_perturbed = z_perturbed + drift_perturbed * ds
        growth = (z_perturbed - z_base).norm() / (eps * direction.norm() + 1e-12)
        lyapunov_proxy = torch.log(growth.clamp_min(1e-12))

    if model_was_training:
        model.train()
    return {
        "stability_injection_fd_gain": float(injection_gain.detach().cpu()),
        "stability_drift_fd_gain": float(drift_gain.detach().cpu()),
        "stability_ftle_proxy": float(lyapunov_proxy.detach().cpu()),
    }


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


def _choose_segment_len(
    config: UDLFTrainConfig,
    generator: torch.Generator,
    device: torch.device,
    *,
    step: int,
) -> int:
    if config.full_bptt_every > 0 and step % config.full_bptt_every == 0:
        return 0
    if config.segment_len_choices:
        index = torch.randint(
            0,
            len(config.segment_len_choices),
            (1,),
            generator=generator,
            device=device,
        )
        return config.segment_len_choices[int(index.item())]
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


def _step_batch_schedule(config: UDLFTrainConfig, segment_len: int) -> tuple[int, int]:
    target_examples = config.batch_size * config.grad_accum_steps
    if segment_len == 0 and config.full_bptt_batch_size > 0:
        batch_size = config.full_bptt_batch_size
    elif segment_len > 0:
        base_horizon = (
            min(config.segment_len_choices)
            if config.segment_len_choices
            else config.segment_len_min or config.segment_len or segment_len
        )
        batch_size = max(1, min(config.batch_size, config.batch_size * base_horizon // segment_len))
    else:
        batch_size = config.batch_size
    grad_accum = math.ceil(target_examples / batch_size)
    return batch_size, grad_accum


@torch.no_grad()
def _evaluate_loss(
    model: nn.Module,
    dataset,
    *,
    batch_size: int,
    batches: int,
    device: torch.device,
    generator: torch.Generator,
    use_amp: bool,
    segment_len: int = 0,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(batches):
        batch, loss_mask = _sample_training_batch(dataset, batch_size, device)
        with _autocast_context(device, use_amp):
            if isinstance(model, UDLFStageAModel):
                loss, _ = _forward_segmented(
                    model,
                    batch,
                    loss_mask=loss_mask,
                    segment_len=segment_len,
                    generator=generator,
                    detach_state_between_segments=True,
                )
            else:
                loss, _ = _forward_causal_lm(model, batch, loss_mask)
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
    model: nn.Module,
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

    device = resolve_device(train_config.device)
    if device.type != "cuda" and not train_config.allow_cpu_training:
        raise RuntimeError("CPU training is disabled; set allow_cpu_training=true only for tests or tiny debugging")
    logger = setup_logger(run_dir, console_log_mode=train_config.console_log_mode)
    cuda_cap_bytes = _configure_cuda_memory_cap(train_config, device, logger)
    set_seed(train_config.seed)
    _probe_auto_batch_size(train_config, device, logger)
    if train_config.full_bptt_batch_size > train_config.batch_size:
        raise RuntimeError(
            "full_bptt_batch_size must not exceed the selected training batch_size; "
            "lower it or rerun the full-BPTT memory probe"
        )

    model, model_config = _build_model(train_config, device)
    if train_config.compile_model:
        model = torch.compile(model)  # type: ignore[assignment]

    train_dataset, eval_dataset = build_datasets(train_config)
    parameter_count = _parameter_count(model)
    optimizer = _build_optimizer(model, train_config)
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
    segment_generator = make_noise_generator(device, train_config.seed + 3)
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
        "UDLF training started mode=%s architecture=%s params=%d device=%s amp=%s data=%s resume_step=%d",
        train_config.mode,
        train_config.architecture,
        parameter_count,
        device,
        train_config.amp and device.type == "cuda",
        "disk" if train_config.data_path else "synthetic",
        start_step,
    )
    start_time = time.time()
    cumulative_tokens = 0
    previous_shape: tuple[int, int] | None = None
    heartbeat_path = run_dir / "heartbeat.json"
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
            architecture_metrics: dict[str, float] = {}
            dynamics_diagnostics: dict[str, list[torch.Tensor]] | None = {} if train_config.dynamics_diagnostics else None
            segment_len = _choose_segment_len(
                train_config,
                segment_generator,
                device,
                step=step,
            )
            full_bptt = segment_len == 0
            step_batch_size, step_grad_accum = _step_batch_schedule(train_config, segment_len)
            step_shape = (segment_len, step_batch_size)
            cache_released = False
            if (
                device.type == "cuda"
                and train_config.release_cuda_cache_on_shape_change
                and previous_shape is not None
                and step_shape != previous_shape
            ):
                torch.cuda.empty_cache()
                cache_released = True
            previous_shape = step_shape
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            step_start_time = time.time()
            write_json(
                heartbeat_path,
                {
                    "status": "running",
                    "step": step,
                    "segment_len": segment_len,
                    "batch_size": step_batch_size,
                    "grad_accum_steps": step_grad_accum,
                    "started_at_unix": step_start_time,
                },
            )
            architecture_metrics["train_segment_len"] = float(segment_len)
            architecture_metrics["train_full_bptt"] = float(full_bptt)
            architecture_metrics["train_step_batch_size"] = float(step_batch_size)
            architecture_metrics["train_step_grad_accum"] = float(step_grad_accum)
            architecture_metrics["train_step_effective_batch_size"] = float(step_batch_size * step_grad_accum)
            architecture_metrics["train_cuda_cache_released"] = float(cache_released)
            for accum_index in range(step_grad_accum):
                batch, loss_mask = _sample_training_batch(train_dataset, step_batch_size, device)
                batch_tokens += step_batch_size * (batch.shape[1] - 1)
                with _autocast_context(device, train_config.amp):
                    did_backward = False
                    if isinstance(model, UDLFStageAModel):
                        if train_config.mode == "stage-b":
                            dropout_value = torch.rand((), generator=noise_generator, device=device)
                            use_posterior = bool(dropout_value.item() >= train_config.posterior_dropout)
                            micro_loss, final_state, micro_metrics = _stage_b_forward(
                                model,
                                batch,
                                loss_mask=loss_mask,
                                generator=noise_generator,
                                use_posterior=use_posterior,
                                diagnostics=dynamics_diagnostics,
                                config=train_config,
                            )
                            architecture_metrics.update(micro_metrics)
                        elif train_config.prior_path_samples > 1:
                            micro_loss, final_state, micro_metrics = _prior_multisample_forward(
                                model,
                                batch,
                                loss_mask=loss_mask,
                                path_samples=train_config.prior_path_samples,
                                state_selection=train_config.prior_state_selection,
                                generator=noise_generator,
                                diagnostics=dynamics_diagnostics,
                            )
                            architecture_metrics.update(micro_metrics)
                        elif segment_len > 0 and train_config.detach_state_between_segments:
                            micro_loss, final_state = _backward_segmented(
                                model,
                                batch,
                                loss_mask=loss_mask,
                                segment_len=segment_len,
                                generator=noise_generator,
                                diagnostics=dynamics_diagnostics,
                                grad_scale=1.0 / step_grad_accum,
                            )
                            did_backward = True
                        else:
                            micro_loss, final_state = _forward_segmented(
                                model,
                                batch,
                                loss_mask=loss_mask,
                                segment_len=segment_len,
                                generator=noise_generator,
                                detach_state_between_segments=train_config.detach_state_between_segments,
                                diagnostics=dynamics_diagnostics,
                            )
                    else:
                        micro_loss, final_state = _forward_causal_lm(model, batch, loss_mask)
                    if not did_backward:
                        scaled_loss = micro_loss / step_grad_accum
                        scaled_loss.backward()
                loss = micro_loss if loss is None else loss + micro_loss.detach()

            assert loss is not None
            loss_for_metrics = loss / step_grad_accum
            grad_norm = _grad_norm(model.parameters())
            if train_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            elapsed = max(time.time() - start_time, 1e-9)
            step_elapsed = max(time.time() - step_start_time, 1e-9)
            cumulative_tokens += batch_tokens
            current_loss = float(loss_for_metrics.detach().cpu())
            metrics: dict[str, Any] = {
                "step": step,
                "loss_lm": current_loss,
                "ppl_lm": float(math.exp(min(current_loss, 20.0))),
                "tokens_per_second": round(cumulative_tokens / elapsed, 3),
                "step_seconds": round(step_elapsed, 6),
                "step_tokens_per_second": round(batch_tokens / step_elapsed, 3),
                "grad_norm": grad_norm,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "architecture": train_config.architecture,
                "parameter_count": parameter_count,
                "batch_size": train_config.batch_size,
                "grad_accum_steps": train_config.grad_accum_steps,
                "effective_batch_size": train_config.batch_size * train_config.grad_accum_steps,
                "device": device.type,
                "amp": train_config.amp and device.type == "cuda",
                "stage_a": train_config.mode == "stage-a",
                "training_mode": train_config.mode,
            }
            if isinstance(model, MambaLMModel):
                metrics["mamba_backend"] = model.backend
            metrics.update(architecture_metrics)
            if isinstance(model, UDLFStageAModel):
                assert final_state is not None
                metrics["diffusion_mode"] = model_config.diffusion_mode
                metrics["state_rms"] = float(final_state.detach().pow(2).mean().sqrt().cpu())
                metrics.update(_slot_geometry_metrics(final_state))
                if train_config.stability_diagnostics and (
                    train_config.stability_diagnostic_every == 0 or step % train_config.stability_diagnostic_every == 0
                ):
                    metrics.update(
                        _stability_diagnostics(
                            model,
                            batch,
                            eps=train_config.stability_diagnostic_eps,
                        )
                    )
            if dynamics_diagnostics is not None:
                metrics.update(_dynamics_summary(dynamics_diagnostics))
            if device.type == "cuda":
                metrics["cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3)
                metrics["cuda_memory_reserved_mb"] = round(torch.cuda.memory_reserved(device) / (1024 * 1024), 3)
                metrics["cuda_step_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 3)
                metrics["cuda_step_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved(device) / (1024 * 1024), 3)
                metrics["cuda_allocator_cap_mb"] = round(cuda_cap_bytes / (1024 * 1024), 3)

            save_best = False
            if train_config.eval_every > 0 and step % train_config.eval_every == 0:
                eval_batch_size = train_config.eval_batch_size or train_config.batch_size
                eval_segment_len = train_config.segment_len if isinstance(model, UDLFStageAModel) else 0
                eval_loss = _evaluate_loss(
                    model,
                    eval_dataset,
                    batch_size=eval_batch_size,
                    batches=train_config.eval_batches,
                    device=device,
                    generator=noise_generator,
                    use_amp=train_config.amp,
                    segment_len=eval_segment_len,
                )
                metrics["eval_batch_size"] = eval_batch_size
                metrics["eval_segment_len"] = eval_segment_len
                metrics["eval_loss_lm"] = eval_loss
                metrics["eval_ppl_lm"] = math.exp(min(eval_loss, 20.0))
                if isinstance(model, UDLFStageAModel) and train_config.eval_interventions:
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
            write_json(
                heartbeat_path,
                {
                    "status": "completed",
                    "step": step,
                    "segment_len": segment_len,
                    "batch_size": step_batch_size,
                    "grad_accum_steps": step_grad_accum,
                    "step_seconds": metrics["step_seconds"],
                    "completed_at_unix": time.time(),
                },
            )
            last_metrics = metrics
            if step % train_config.log_every == 0:
                logger.info(
                    "step=%d loss_lm=%.6f grad_norm=%.6f tok_s=%.1f",
                    step,
                    metrics["loss_lm"],
                    metrics["grad_norm"],
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
        if train_config.mode == "stage-a":
            logger.info("UDLF stage A training finished step=%d", step)
        else:
            logger.info("UDLF training finished mode=%s step=%d", train_config.mode, step)
    except BaseException as exc:
        try:
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
            except BaseException as checkpoint_exc:
                write_json(
                    run_dir / "failed_error.json",
                    {
                        "step": step,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "checkpoint_error_type": type(checkpoint_exc).__name__,
                        "checkpoint_error": str(checkpoint_exc),
                        "traceback_tail": traceback.format_exc()[-4000:],
                    },
                )
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
