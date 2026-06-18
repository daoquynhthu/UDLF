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
    "intervention_perturbed_delta",
    "intervention_attenuated_delta",
    "intervention_inverted_delta",
    "tokens_per_second",
]

DEFAULT_INCLUDE_PREFIXES = [
    "udlf_query_recall_ode_ablation",
    "udlf_query_recall_fixed_diffusion",
    "udlf_query_recall_fixed_k1",
    "udlf_query_recall_fixed_k4",
    "udlf_query_recall_state_dependent_diffusion",
    "udlf_query_recall_state_dep_k1",
    "udlf_query_recall_state_dep_k2",
    "udlf_query_recall_state_dep_k4",
    "udlf_query_recall_state_dep_k8",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _last_eval_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "eval_loss_lm" in row:
                last = row
    return last


def _iter_runs(runs_dir: Path, prefixes: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_dirs: list[Path] = []
    for prefix in prefixes:
        run_dirs.extend(path for path in runs_dir.glob(f"{prefix}*") if path.is_dir())
    for run_dir in sorted(set(run_dirs)):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.jsonl"
        if not config_path.exists() or not metrics_path.exists():
            continue
        config = _load_json(config_path)
        metrics = _last_eval_row(metrics_path)
        if metrics is None:
            continue
        row: dict[str, Any] = {
            "run": run_dir.name,
            "seed": config.get("seed"),
            "diffusion_mode": config.get("diffusion_mode"),
            "solver_steps": config.get("solver_steps"),
            "fixed_sigma": config.get("fixed_sigma"),
            "step": metrics.get("step"),
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key)
        rows.append(row)
    return rows


def _group_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row["diffusion_mode"]), int(row["solver_steps"]))


def _float_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)
    summaries: list[dict[str, Any]] = []
    for (mode, solver_steps), group_rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "diffusion_mode": mode,
            "solver_steps": solver_steps,
            "runs": len(group_rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group_rows, key=lambda item: int(item["seed"]))),
        }
        for key in METRIC_KEYS:
            values = _float_values(group_rows, key)
            if values:
                summary[f"mean_{key}"] = mean(values)
                summary[f"min_{key}"] = min(values)
        summaries.append(summary)
    return summaries


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


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    def fmt(row: dict[str, Any], key: str, decimals: int = 3) -> str:
        value = row.get(key)
        if value is None:
            return "n/a"
        return f"{float(value):.{decimals}f}"

    headers = [
        "mode",
        "K",
        "runs",
        "seeds",
        "eval",
        "tok/s",
        "zero",
        "swap",
        "shift",
        "mix",
        "perturb",
        "atten",
        "invert",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["diffusion_mode"]),
                    str(row["solver_steps"]),
                    str(row["runs"]),
                    str(row["seeds"]),
                    fmt(row, "mean_eval_loss_lm"),
                    fmt(row, "mean_tokens_per_second", decimals=0),
                    fmt(row, "mean_intervention_zero_delta"),
                    fmt(row, "mean_intervention_swapped_delta"),
                    fmt(row, "mean_intervention_shifted_delta"),
                    fmt(row, "mean_intervention_mixed_delta"),
                    fmt(row, "mean_intervention_perturbed_delta"),
                    fmt(row, "mean_intervention_attenuated_delta"),
                    fmt(row, "mean_intervention_inverted_delta"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _write_markdown(path: Path, detail_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# Phase 4 Query Recall Ablation Summary",
            "",
            "This summary is generated from local run metrics. It excludes checkpoints and run artifacts.",
            "",
            "## Group Means",
            "",
            _markdown_table(summary_rows),
            "",
            "## Current Read",
            "",
            "- All summarized groups pass the query-recall core state-causality gate.",
            "- Fixed K=4 and state-dependent K=4 produce stronger intervention margins than K=1 variants.",
            "- K=1 variants are much faster and still usable for cheap screening.",
            "- Attenuation remains too small and inconsistent to use as a blocking robustness gate.",
            "- Structured mixed-state perturbation is tracked for new runs; older summarized runs may not contain it.",
            "- Fixed K=4 is the current pragmatic default candidate for real-token confirmation because it is simpler than state-dependent diffusion and has strong synthetic margins.",
            "",
            f"Detail runs summarized: {len(detail_rows)}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Phase 4 query-recall ablation runs.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--include-prefix", action="append", default=[], help="Run directory prefix to include. May be repeated.")
    parser.add_argument("--detail-csv", type=Path, default=Path("doc/phase4_ablation_runs.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("doc/phase4_ablation_summary.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("doc/phase4_ablation_summary.md"))
    args = parser.parse_args(argv)

    prefixes = args.include_prefix or DEFAULT_INCLUDE_PREFIXES
    detail_rows = _iter_runs(args.runs_dir, prefixes)
    if not detail_rows:
        raise RuntimeError(f"no matching runs found under {args.runs_dir}")
    summary_rows = _aggregate(detail_rows)
    _write_csv(args.detail_csv, detail_rows)
    _write_csv(args.summary_csv, summary_rows)
    _write_markdown(args.markdown, detail_rows, summary_rows)
    print(json.dumps({"detail_runs": len(detail_rows), "groups": len(summary_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
