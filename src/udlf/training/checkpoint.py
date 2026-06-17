from __future__ import annotations

import queue
import random
import threading
import time
from pathlib import Path
from typing import Any

import torch

COMPILED_PREFIX = "_orig_mod."


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def normalize_state_dict_for_model(model: torch.nn.Module, state_dict: dict[str, Any]) -> dict[str, Any]:
    target_keys = list(model.state_dict().keys())
    source_keys = list(state_dict.keys())
    target_compiled = bool(target_keys) and all(key.startswith(COMPILED_PREFIX) for key in target_keys)
    source_compiled = bool(source_keys) and all(key.startswith(COMPILED_PREFIX) for key in source_keys)
    if source_compiled and not target_compiled:
        return {key.removeprefix(COMPILED_PREFIX): value for key, value in state_dict.items()}
    if target_compiled and not source_compiled:
        return {f"{COMPILED_PREFIX}{key}": value for key, value in state_dict.items()}
    return {key.removeprefix(COMPILED_PREFIX).replace(f".{COMPILED_PREFIX}", "."): value for key, value in state_dict.items()}


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _snapshot(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return value


def _atomic_save(path: Path, payload: dict[str, Any], retries: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        tmp = path.with_name(f"{path.name}.{threading.get_ident()}.{attempt}.tmp")
        try:
            torch.save(payload, tmp)
            tmp.replace(path)
            return
        except BaseException as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"failed to save checkpoint payload to {path}") from last_error


def build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    step: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    checkpoint_kind: str = "full",
) -> dict[str, Any]:
    model = unwrap_compiled_model(model)
    payload: dict[str, Any] = {
        "step": step,
        "model": _snapshot(normalize_state_dict_for_model(model, model.state_dict())),
        "config": dict(config),
        "metrics": dict(metrics),
        "checkpoint_kind": checkpoint_kind,
    }
    if checkpoint_kind == "full":
        payload["optimizer"] = _snapshot(optimizer.state_dict()) if optimizer is not None else None
        payload["scheduler"] = _snapshot(scheduler.state_dict()) if scheduler is not None else None
        payload["scaler"] = _snapshot(scaler.state_dict()) if scaler is not None else None
        payload["rng"] = _snapshot(rng_state())
    return payload


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    _atomic_save(path, payload)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    strict: bool = True,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(normalize_state_dict_for_model(model, checkpoint["model"]), strict=strict)
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if checkpoint.get("rng") is not None:
        load_rng_state(checkpoint["rng"])
    return int(checkpoint.get("step", 0))


class AsyncCheckpointWriter:
    def __init__(self, max_queue: int = 2, submit_timeout: float = 120.0) -> None:
        self._queue: queue.Queue[tuple[Path, dict[str, Any]] | None] = queue.Queue(maxsize=max_queue)
        self._submit_timeout = submit_timeout
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="udlf-checkpoint-writer", daemon=False)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, payload = item
                _atomic_save(path, payload)
            except BaseException as exc:
                self._error = exc
            finally:
                self._queue.task_done()

    def submit(self, path: Path, payload: dict[str, Any]) -> None:
        self.raise_if_failed()
        if self._closed:
            raise RuntimeError("async checkpoint writer is closed")
        try:
            self._queue.put((path, payload), timeout=self._submit_timeout)
        except queue.Full as exc:
            raise RuntimeError(f"async checkpoint writer queue stayed full for {self._submit_timeout:.0f}s") from exc
        self.raise_if_failed()

    def wait(self) -> None:
        self._queue.join()
        self.raise_if_failed()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.wait()
        finally:
            self._queue.put(None)
            self._thread.join()
            self._closed = True
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("async checkpoint writer failed") from self._error

