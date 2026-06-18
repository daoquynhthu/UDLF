from __future__ import annotations

import csv
import json
import logging
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Any


def setup_logger(run_dir: Path, *, console_log_mode: str = "progress") -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("udlf.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    if console_log_mode not in {"progress", "quiet"}:
        raise ValueError("console_log_mode must be 'progress' or 'quiet'")

    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if console_log_mode == "progress":
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(console_formatter)
        logger.addHandler(stream)

    file_handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    return logger


class _FlushRequest:
    def __init__(self, *, force_sync: bool) -> None:
        self.force_sync = force_sync
        self.done = threading.Event()


class JsonlMetricLogger:
    def __init__(
        self,
        path: Path,
        *,
        flush_every: int = 50,
        fsync_every: int = 1000,
        max_queue: int = 8192,
        submit_timeout: float = 30.0,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self.fsync_every = max(0, int(fsync_every))
        self._submit_timeout = submit_timeout
        self._queue: queue.Queue[dict[str, Any] | _FlushRequest | None] = queue.Queue(maxsize=max_queue)
        self._error: BaseException | None = None
        self._writes = 0
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="udlf-metric-jsonl-writer", daemon=False)
        self._thread.start()

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            while True:
                item = self._queue.get()
                try:
                    if item is None:
                        file.flush()
                        if self.fsync_every > 0:
                            os.fsync(file.fileno())
                        return
                    if isinstance(item, _FlushRequest):
                        file.flush()
                        if item.force_sync:
                            os.fsync(file.fileno())
                        item.done.set()
                        continue
                    file.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                    self._writes += 1
                    should_flush = self._writes % self.flush_every == 0
                    should_fsync = self.fsync_every > 0 and self._writes % self.fsync_every == 0
                    if should_flush or should_fsync:
                        file.flush()
                    if should_fsync:
                        os.fsync(file.fileno())
                except BaseException as exc:
                    self._error = exc
                    if isinstance(item, _FlushRequest):
                        item.done.set()
                finally:
                    self._queue.task_done()

    def _submit(self, item: dict[str, Any] | _FlushRequest | None) -> None:
        self.raise_if_failed()
        if self._closed:
            raise RuntimeError("metric logger is closed")
        try:
            self._queue.put(item, timeout=self._submit_timeout)
        except queue.Full as exc:
            raise RuntimeError(f"metric logger queue stayed full for {self._submit_timeout:.0f}s") from exc
        self.raise_if_failed()

    def write(self, payload: dict[str, Any], *, force_sync: bool = False) -> None:
        self._submit(dict(payload))
        if force_sync:
            self.flush(force_sync=True)

    def flush(self, *, force_sync: bool = False) -> None:
        request = _FlushRequest(force_sync=force_sync)
        self._submit(request)
        request.done.wait()
        self.raise_if_failed()

    def close(self) -> None:
        if self._closed:
            return
        self.flush(force_sync=True)
        self._queue.put(None)
        self._thread.join()
        self._closed = True
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("metric logger failed") from self._error


def metrics_jsonl_to_csv(jsonl_path: Path, csv_path: Path | None = None) -> Path | None:
    if not jsonl_path.exists() or jsonl_path.stat().st_size == 0:
        return None
    csv_path = csv_path or jsonl_path.with_suffix(".csv")
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    seen: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    if not rows:
        return None
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return csv_path
