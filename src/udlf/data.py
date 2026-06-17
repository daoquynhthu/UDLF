from __future__ import annotations

import torch
from torch import Tensor


class RepeatingPatternDataset:
    """Synthetic stream where the second half repeats the first half.

    This is a minimal exact-memory probe for the first training harness. It is
    intentionally simple and deterministic under a seeded generator.
    """

    def __init__(self, vocab_size: int, seq_len: int, *, seed: int = 0) -> None:
        if seq_len < 4:
            raise ValueError("seq_len must be >= 4")
        if vocab_size <= 4:
            raise ValueError("vocab_size must be > 4")
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        half = self.seq_len // 2
        prefix = torch.randint(1, self.vocab_size, (batch_size, half), generator=self.generator)
        repeated = prefix[:, : self.seq_len - half]
        sequence = torch.cat([prefix, repeated], dim=1)
        return sequence.to(device)


class TokenDatasetFromDisk:
    """Random batches from a Hugging Face dataset saved with load_from_disk."""

    def __init__(self, path: str, split: str, seq_len: int, *, seed: int = 0, column: str = "input_ids") -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise RuntimeError("datasets is required for data_path training") from exc
        loaded = load_from_disk(path)
        self.dataset = loaded[split] if hasattr(loaded, "keys") else loaded
        if column not in self.dataset.column_names:
            raise ValueError(f"column {column!r} not found in dataset columns {self.dataset.column_names}")
        self.seq_len = seq_len
        self.column = column
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        indices = torch.randint(0, len(self.dataset), (batch_size,), generator=self.generator).tolist()
        rows = self.dataset.select(indices)
        sequences: list[list[int]] = []
        for values in rows[self.column]:
            if len(values) < self.seq_len:
                raise ValueError(f"dataset row has length {len(values)}, shorter than seq_len={self.seq_len}")
            sequences.append([int(token) for token in values[: self.seq_len]])
        return torch.tensor(sequences, dtype=torch.long, device=device)
