from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = {
    "intervention_zero_delta": 0.3,
    "intervention_swapped_delta": 0.03,
    "intervention_shifted_delta": 0.02,
    "intervention_perturbed_delta": 0.0,
}


def _last_eval_row(metrics_path: Path) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "eval_loss_lm" in row:
                last = row
    if last is None:
        raise RuntimeError(f"no eval rows found in {metrics_path}")
    return last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the latest Stage A state intervention metrics.")
    parser.add_argument("metrics", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--zero", type=float, default=DEFAULT_THRESHOLDS["intervention_zero_delta"])
    parser.add_argument("--swapped", type=float, default=DEFAULT_THRESHOLDS["intervention_swapped_delta"])
    parser.add_argument("--shifted", type=float, default=DEFAULT_THRESHOLDS["intervention_shifted_delta"])
    parser.add_argument("--perturbed", type=float, default=DEFAULT_THRESHOLDS["intervention_perturbed_delta"])
    args = parser.parse_args(argv)

    row = _last_eval_row(args.metrics)
    thresholds = {
        "intervention_zero_delta": args.zero,
        "intervention_swapped_delta": args.swapped,
        "intervention_shifted_delta": args.shifted,
        "intervention_perturbed_delta": args.perturbed,
    }
    failed: list[str] = []
    for key, threshold in thresholds.items():
        value = float(row.get(key, float("-inf")))
        if value < threshold:
            failed.append(f"{key}={value:.6f} < {threshold:.6f}")

    summary = {
        "step": row.get("step"),
        "eval_loss_lm": row.get("eval_loss_lm"),
        **{key: row.get(key) for key in thresholds},
        "passed": not failed,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failed:
        print("FAILED: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

