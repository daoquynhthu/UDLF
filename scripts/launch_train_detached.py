"""Launch a UDLF Python training module without a visible Windows console."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _creationflags() -> int:
    flags = 0
    for name in ("CREATE_NEW_PROCESS_GROUP", "DETACHED_PROCESS", "CREATE_NO_WINDOW", "CREATE_BREAKAWAY_FROM_JOB"):
        flags |= getattr(subprocess, name, 0)
    flags |= getattr(subprocess, "HIGH_PRIORITY_CLASS", 0)
    return flags


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--module", default="udlf.training.train")
    parser.add_argument("module_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["UDLF_PYTHON"] = str(args.python)
    env["PYTHONPATH"] = str(repo / "src")
    env["PYTHON_EXECUTABLE"] = str(args.python)
    env["VIRTUAL_ENV"] = str(args.python.parent.parent)
    env["PATH"] = str(args.python.parent) + os.pathsep + env.get("PATH", "")

    module_args = list(args.module_args)
    if module_args and module_args[0] == "--":
        module_args = module_args[1:]
    if "--run-dir" not in module_args:
        module_args = ["--run-dir", str(run_dir), *module_args]

    command = [str(args.python), "-m", args.module, *module_args]
    (run_dir / "launch_cmd.txt").write_text(" ".join(command), encoding="utf-8")
    (run_dir / "module_args.txt").write_text("\n".join(module_args) + "\n", encoding="utf-8")

    stdout = open(run_dir / "launcher.stdout.log", "ab", buffering=0)
    stderr = open(run_dir / "launcher.stderr.log", "ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=str(repo),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        creationflags=_creationflags(),
        close_fds=True,
    )
    (run_dir / "daemon.pid").write_text(str(process.pid), encoding="utf-8")
    print(process.pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
