from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRIC_KEYS = [
    "eval_loss_lm",
    "intervention_zero_delta",
    "intervention_swapped_delta",
    "intervention_shifted_delta",
    "intervention_mixed_delta",
    "intervention_temporal_mixed_delta",
    "intervention_perturbed_delta",
    "intervention_attenuated_delta",
    "intervention_inverted_delta",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(input_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        metrics = _load_json(path)
        config = _load_json(Path(metrics["config"]))
        row: dict[str, Any] = {
            "artifact": str(path),
            "run": Path(metrics["run_dir"]).name,
            "seed": config.get("seed"),
            "diffusion_mode": config.get("diffusion_mode"),
            "solver_steps": config.get("solver_steps"),
            "fixed_sigma": config.get("fixed_sigma"),
            "checkpoint_step": metrics.get("checkpoint_step"),
            "pair_trials": metrics.get("intervention_pair_trials"),
            "perturb_trials": metrics.get("intervention_perturb_trials"),
            "mix_alpha": metrics.get("intervention_mix_alpha"),
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key)
            row[f"{key}_sem"] = metrics.get(f"{key}_sem")
            row[f"{key}_ci95_low"] = metrics.get(f"{key}_ci95_low")
            row[f"{key}_ci95_high"] = metrics.get(f"{key}_ci95_high")
        rows.append(row)
    return rows


def _float_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["diffusion_mode"])].append(row)
    summaries: list[dict[str, Any]] = []
    for mode, group_rows in sorted(groups.items()):
        summary: dict[str, Any] = {
            "diffusion_mode": mode,
            "runs": len(group_rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group_rows, key=lambda item: int(item["seed"]))),
            "solver_steps": ",".join(sorted({str(row["solver_steps"]) for row in group_rows})),
            "pair_trials": ",".join(sorted({str(row["pair_trials"]) for row in group_rows})),
        }
        for key in METRIC_KEYS:
            values = _float_values(group_rows, key)
            if values:
                summary[f"mean_{key}"] = mean(values)
                summary[f"min_{key}"] = min(values)
        summaries.append(summary)
    return summaries


def _fmt(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{decimals}f}"


def _table(rows: list[dict[str, Any]]) -> str:
    headers = ["mode", "runs", "seeds", "eval", "zero", "swap", "shift", "mix", "temporal", "perturb", "atten", "invert"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["diffusion_mode"]),
                    str(row["runs"]),
                    str(row["seeds"]),
                    _fmt(row.get("mean_eval_loss_lm")),
                    _fmt(row.get("mean_intervention_zero_delta")),
                    _fmt(row.get("mean_intervention_swapped_delta")),
                    _fmt(row.get("mean_intervention_shifted_delta")),
                    _fmt(row.get("mean_intervention_mixed_delta")),
                    _fmt(row.get("mean_intervention_temporal_mixed_delta")),
                    _fmt(row.get("mean_intervention_perturbed_delta")),
                    _fmt(row.get("mean_intervention_attenuated_delta")),
                    _fmt(row.get("mean_intervention_inverted_delta")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_markdown(path: Path, rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# CRN Diffusion Intervention Summary",
            "",
            "This summary is generated from read-only checkpoint intervention evaluation with common random numbers.",
            "",
            "Evaluation settings:",
            "",
            "- Checkpoints: query-recall diffusion ablation seeds `710-713`.",
            "- Pair trials: `4` suffix Brownian paths per run.",
            "- Perturb trials: `16` state-noise samples per suffix path.",
            "- Mixed alpha: `0.2`.",
            "",
            "## Group Means",
            "",
            _table(summaries),
            "",
            "## Current Read",
            "",
            "- Under CRN, state-dependent diffusion has the strongest mean perturbation delta in this matched set and stays positive on every seed.",
            "- Fixed diffusion is also positive on average, but one seed is near zero and its paired interval can overlap zero.",
            "- ODE perturbation deltas are near zero, as expected because suffix rollouts are deterministic and only the perturbation noise changes state.",
            "- Fixed diffusion remains the simpler smoke/default candidate, but it is no longer the strongest robustness candidate after CRN re-evaluation.",
            "- This does not close the broader robustness blocker; it only repairs the diffusion comparison that previously used unpaired suffix paths.",
            "",
            f"Detail runs summarized: {len(rows)}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize CRN checkpoint intervention evaluation artifacts.")
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/phase4_crn_diffusion"))
    parser.add_argument("--detail-csv", type=Path, default=Path("doc/phase4_crn_intervention_runs.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("doc/phase4_crn_intervention_summary.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("doc/phase4_crn_intervention_summary.md"))
    args = parser.parse_args(argv)

    rows = _load_rows(args.input_dir)
    if not rows:
        raise RuntimeError(f"no CRN intervention JSON artifacts found under {args.input_dir}")
    summaries = _aggregate(rows)
    _write_csv(args.detail_csv, rows)
    _write_csv(args.summary_csv, summaries)
    _write_markdown(args.markdown, rows, summaries)
    print(json.dumps({"detail_runs": len(rows), "groups": len(summaries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
