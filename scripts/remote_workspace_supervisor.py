from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def creation_flags() -> int:
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW


def main() -> None:
    parser = argparse.ArgumentParser(description="Restarting supervisor for the UDLF workspace HTTPS agent.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    service_root = args.config.parent
    stdout_path = service_root / "agent.stdout.log"
    stderr_path = service_root / "agent.stderr.log"
    command = [str(config["python"]), str(config["agent"]), *map(str, config["arguments"])]

    while True:
        with stdout_path.open("ab", buffering=0) as stdout, stderr_path.open("ab", buffering=0) as stderr:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            stdout.write(f"\n[{timestamp}] starting workspace agent\n".encode())
            process = subprocess.Popen(
                command,
                cwd=str(config["repo"]),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=creation_flags(),
            )
            exit_code = process.wait()
            stderr.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] agent exited {exit_code}; restarting\n".encode())
        time.sleep(3)


if __name__ == "__main__":
    main()
