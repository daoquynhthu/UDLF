from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from udlf.model import UDLFStageAModel
from udlf.training.checkpoint import load_checkpoint
from udlf.training.config import load_raw_config, train_config_from_dict
from udlf.training.runtime import build_datasets, make_noise_generator, resolve_device, set_seed
from udlf.training.train import _evaluate_interventions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Stage A state interventions from an existing checkpoint.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing config.json and checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path. Defaults to <run-dir>/latest.pt.")
    parser.add_argument("--config", type=Path, default=None, help="Config path. Defaults to <run-dir>/config.json.")
    parser.add_argument("--device", default="", help="Override eval device.")
    parser.add_argument("--batch-size", type=int, default=0, help="Override eval batch size.")
    parser.add_argument("--eval-batches", type=int, default=1, help="Must be 1 for paired CI reporting; increase --pair-trials for more suffix noise seeds.")
    parser.add_argument("--pair-trials", type=int, default=0, help="Override paired suffix noise trials.")
    parser.add_argument("--mix-alpha", type=float, default=None, help="Override intervention_mix_alpha for this eval only.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--print-json", action="store_true", help="Print the full JSON result to stdout. Default prints one compact progress line.")
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    config_path = args.config if args.config is not None else run_dir / "config.json"
    checkpoint_path = args.checkpoint if args.checkpoint is not None else run_dir / "latest.pt"
    raw_config = load_raw_config(str(config_path))
    train_config = train_config_from_dict(raw_config)
    if args.device:
        train_config.device = args.device
    if args.batch_size > 0:
        train_config.batch_size = args.batch_size
    if args.pair_trials > 0:
        train_config.intervention_pair_trials = args.pair_trials
    if args.mix_alpha is not None:
        if not 0.0 <= args.mix_alpha <= 1.0:
            raise ValueError("--mix-alpha must be between 0 and 1")
        train_config.intervention_mix_alpha = args.mix_alpha
    if args.eval_batches < 1:
        raise ValueError("--eval-batches must be >= 1")
    if args.eval_batches != 1:
        raise ValueError("--eval-batches > 1 is not supported for paired CI reporting; increase --pair-trials instead")

    set_seed(train_config.seed)
    device = resolve_device(train_config.device)
    model = UDLFStageAModel(train_config.model_config()).to(device)
    step = load_checkpoint(checkpoint_path, model=model, strict=train_config.strict_resume)
    _, eval_dataset = build_datasets(train_config)
    noise_seed = train_config.noise_seed if train_config.noise_seed is not None else train_config.seed + 2
    generator = make_noise_generator(device, noise_seed)

    rows = [
        _evaluate_interventions(
            model,
            eval_dataset,
            batch_size=train_config.batch_size,
            device=device,
            generator=generator,
            use_amp=train_config.amp,
            shift_tokens=train_config.intervention_shift_tokens,
            pair_trials=train_config.intervention_pair_trials,
            perturb_std=train_config.intervention_perturb_std,
            perturb_trials=train_config.intervention_perturb_trials,
            mix_alpha=train_config.intervention_mix_alpha,
        )
        for _ in range(args.eval_batches)
    ]
    metrics: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "checkpoint_step": step,
        "step": step,
        "eval_batches": args.eval_batches,
        "device": device.type,
    }
    numeric_keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, int | float)})
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            metrics[key] = sum(values) / len(values)
    metrics["eval_loss_lm"] = metrics.get("intervention_correct_loss")
    if args.output:
        _write_json(args.output, metrics)
    if args.print_json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        output_text = f" output={args.output}" if args.output else ""
        print(
            "CRN eval "
            f"run={run_dir.name} step={step} "
            f"eval={metrics.get('eval_loss_lm', float('nan')):.4f} "
            f"perturb={metrics.get('intervention_perturbed_delta', float('nan')):+.4f} "
            f"mix={metrics.get('intervention_mixed_delta', float('nan')):+.4f} "
            f"temporal={metrics.get('intervention_temporal_mixed_delta', float('nan')):+.4f}"
            f"{output_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
