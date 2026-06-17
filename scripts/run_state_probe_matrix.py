from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _last_eval_row(metrics_path: Path) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "eval_loss_lm" in row:
                last = row
    if last is None:
        raise RuntimeError(f"no eval rows found in {metrics_path}")
    return last


def _run(command: list[str], *, cwd: Path) -> None:
    env = dict(os.environ)
    src_path = str(cwd / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]
    completed = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(command)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Stage A suffix state probe across seeds.")
    parser.add_argument("--template", type=Path, default=Path("configs/training_templates/udlf_stage_a_suffix_probe.json"))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--run-prefix", default="runs/udlf_state_probe_matrix")
    parser.add_argument("--device", default="")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--summary", type=Path, default=Path("runs/udlf_state_probe_matrix_summary.json"))
    parser.add_argument("--check", action="store_true", help="Run check_state_probe.py after each seed.")
    parser.add_argument("--check-profile", choices=["all", "core", "robustness"], default="all")
    parser.add_argument("--resume-existing", action="store_true", help="Resume from latest.pt when the run directory already has one.")
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    template = _load_json(args.template)
    summaries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="udlf_state_probe_") as temp_dir:
        temp_root = Path(temp_dir)
        for seed in args.seeds:
            config = dict(template)
            run_dir = Path(f"{args.run_prefix}_seed{seed}")
            config["seed"] = seed
            config["run_dir"] = str(run_dir)
            latest = run_dir / "latest.pt"
            if args.resume_existing and latest.exists():
                config["resume"] = str(latest)
            if args.device:
                config["device"] = args.device
            if args.max_steps > 0:
                config["max_steps"] = args.max_steps
            config_path = temp_root / f"seed{seed}.json"
            config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _run([sys.executable, "-m", "udlf.training.train", "--config", str(config_path)], cwd=repo_root)
            metrics_path = run_dir / "metrics.jsonl"
            if args.check:
                _run(
                    [sys.executable, "scripts/check_state_probe.py", str(metrics_path), "--profile", args.check_profile],
                    cwd=repo_root,
                )
            row = _last_eval_row(metrics_path)
            summaries.append(
                {
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "step": row.get("step"),
                    "eval_loss_lm": row.get("eval_loss_lm"),
                    "zero_delta": row.get("intervention_zero_delta"),
                    "swapped_delta": row.get("intervention_swapped_delta"),
                    "shifted_delta": row.get("intervention_shifted_delta"),
                    "perturbed_delta": row.get("intervention_perturbed_delta"),
                    "attenuated_delta": row.get("intervention_attenuated_delta"),
                    "inverted_delta": row.get("intervention_inverted_delta"),
                }
            )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
