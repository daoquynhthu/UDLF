from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def creation_flags() -> int:
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW


def parse_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    index = 0
    while index < len(items):
        raw = items[index]
        if not raw.startswith("-"):
            raise ValueError(f"unexpected train override token: {raw}")
        key = raw.lstrip("-")
        if index + 1 >= len(items) or items[index + 1].startswith("-"):
            overrides[key] = True
            index += 1
            continue
        value = items[index + 1]
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
        index += 2
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one UDLF template training job under workspace-agent supervision.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("-Template", required=True)
    parser.add_argument("-RunName", default="")
    args, unknown = parser.parse_known_args()

    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    template = Path(args.Template)
    if not template.is_absolute():
        template = repo / "configs" / "training_templates" / args.Template
    if template.suffix == "":
        template = template.with_suffix(".json")
    config = json.loads(template.read_text(encoding="utf-8"))
    config.update(parse_overrides(unknown))
    config["run_dir"] = str(run_dir)
    generated = run_dir / "workspace_config.json"
    generated.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["UDLF_PYTHON"] = str(args.python)
    env["PYTHONPATH"] = str(repo / "src")
    env["PYTHON_EXECUTABLE"] = str(args.python)
    env["VIRTUAL_ENV"] = str(args.python.parent.parent)
    env["PATH"] = str(args.python.parent) + os.pathsep + env.get("PATH", "")
    command = [str(args.python), "-m", "udlf.training.train", "--config", str(generated)]
    (run_dir / "workspace_train_cmd.txt").write_text(" ".join(command), encoding="utf-8")
    return subprocess.run(command, cwd=repo, env=env, check=False, creationflags=creation_flags()).returncode


if __name__ == "__main__":
    raise SystemExit(main())
