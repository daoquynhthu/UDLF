from __future__ import annotations

import json

from udlf.training.train import run_smoke


def test_run_smoke_writes_logs_and_metrics(tmp_path):
    run_dir = tmp_path / "run"

    run_smoke(run_dir=run_dir, steps=3, sleep_seconds=0.0)

    train_log = (run_dir / "train.log").read_text(encoding="utf-8")
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "UDLF smoke training started" in train_log
    assert "UDLF smoke training finished" in train_log
    assert [row["step"] for row in metrics] == [1, 2, 3]
    assert all(row["smoke"] is True for row in metrics)
