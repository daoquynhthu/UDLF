from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


PLACEHOLDER_PREFIX = "<"
PLACEHOLDER_SUFFIX = ">"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parameters(payload: dict[str, Any]) -> dict[str, Any]:
    params = payload.get("parameters")
    if isinstance(params, dict):
        return params
    return payload


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(PLACEHOLDER_PREFIX) and value.endswith(PLACEHOLDER_SUFFIX)


def _find_placeholders(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_find_placeholders(item, child_prefix))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_placeholders(item, f"{prefix}[{index}]"))
    elif _is_placeholder(value):
        found.append(f"{prefix}={value}")
    return found


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"override must use key=value form: {raw!r}")
    key, value = raw.split("=", 1)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return key, parsed


def _set_nested(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    current: dict[str, Any] = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a private remote smoke training config from a tracked template.")
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("configs/training_templates/udlf_remote_real_token_query_recall_smoke.example.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/training_templates/udlf_remote_real_token_query_recall_smoke.local.json"),
    )
    parser.add_argument("--data-path", default=os.environ.get("UDLF_REMOTE_SMOKE_DATASET", ""))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a parameter key.")
    parser.add_argument("--check-template", action="store_true", help="Validate the tracked template without writing output.")
    args = parser.parse_args(argv)

    payload = _load_json(args.template)
    params = _parameters(payload)
    if args.data_path:
        params["data_path"] = args.data_path
    for item in args.set:
        key, value = _parse_override(item)
        _set_nested(params, key, value)

    placeholders = _find_placeholders(payload)
    if args.check_template:
        print(json.dumps({"template": str(args.template), "placeholders": placeholders}, indent=2, sort_keys=True))
        return 0
    if placeholders:
        raise RuntimeError(
            "refusing to write remote smoke config with unresolved placeholders: "
            + "; ".join(placeholders)
            + ". Pass --data-path or --set key=value for private values."
        )
    _write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "run_dir": params.get("run_dir")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
