"""Minimal UDLF training entrypoint.

This module currently provides a smoke runner for validating local and remote
workflow plumbing. It is not the stage A UDLF model implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    if key in config:
        return config[key]
    params = config.get("parameters")
    if isinstance(params, dict) and key in params:
        return params[key]
    return default


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    run_dir = Path(args.run_dir or _config_value(config, "run_dir", "runs/udlf_smoke")).resolve()
    steps = int(_config_value(config, "steps", args.steps))
    sleep_seconds = float(_config_value(config, "sleep_seconds", args.sleep_seconds))
    if steps < 1:
        raise SystemExit("--steps must be >= 1")

    run_smoke(run_dir=run_dir, steps=steps, sleep_seconds=sleep_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
