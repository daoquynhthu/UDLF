from __future__ import annotations

import torch
from torch import Tensor


class RepeatingPatternDataset:
    """Synthetic stream where the second half repeats the first half.

    This is a minimal exact-memory probe for the first training harness. It is
    intentionally simple and deterministic under a seeded generator.
    """

    def __init__(self, vocab_size: int, seq_len: int, *, seed: int = 0, suffix_loss_only: bool = False) -> None:
        if seq_len < 4:
            raise ValueError("seq_len must be >= 4")
        if vocab_size <= 4:
            raise ValueError("vocab_size must be > 4")
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.suffix_loss_only = suffix_loss_only
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        half = self.seq_len // 2
        prefix = torch.randint(1, self.vocab_size, (batch_size, half), generator=self.generator)
        repeated = prefix[:, : self.seq_len - half]
        sequence = torch.cat([prefix, repeated], dim=1)
        return sequence.to(device)

    def loss_mask(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor | None:
        if not self.suffix_loss_only:
            return None
        half = self.seq_len // 2
        mask = torch.zeros(batch_size, self.seq_len - 1, dtype=torch.bool)
        mask[:, max(0, half - 1) :] = True
        return mask.to(device)

    def intervention_split(self) -> int:
        return self.seq_len // 2


class QueryRecallDataset:
    """Synthetic key-value recall where query tokens request prior positions."""

    def __init__(self, vocab_size: int, seq_len: int, *, seed: int = 0) -> None:
        if seq_len < 12:
            raise ValueError("seq_len must be >= 12")
        self.memory_len = max(4, seq_len // 3)
        if vocab_size <= self.memory_len + 8:
            raise ValueError("vocab_size must leave room for query tokens and values")
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.query_base = vocab_size - self.memory_len
        self.generator = torch.Generator().manual_seed(seed)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        values = torch.randint(1, self.query_base, (batch_size, self.memory_len), generator=self.generator)
        sequence = values
        while sequence.shape[1] + 2 <= self.seq_len:
            indices = torch.randint(0, self.memory_len, (batch_size, 1), generator=self.generator)
            queries = self.query_base + indices
            answers = values.gather(1, indices)
            sequence = torch.cat([sequence, queries, answers], dim=1)
        if sequence.shape[1] < self.seq_len:
            filler = torch.randint(1, self.query_base, (batch_size, self.seq_len - sequence.shape[1]), generator=self.generator)
            sequence = torch.cat([sequence, filler], dim=1)
        return sequence.to(device)

    def loss_mask(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        mask = torch.zeros(batch_size, self.seq_len - 1, dtype=torch.bool)
        for position in range(self.memory_len, self.seq_len - 1, 2):
            mask[:, position] = True
        return mask.to(device)

    def intervention_split(self) -> int:
        return self.memory_len


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


class RealTokenQueryRecallDataset:
    """Query-recall task using real saved-token rows as memory values."""

    def __init__(self, path: str, split: str, seq_len: int, vocab_size: int, *, seed: int = 0, column: str = "input_ids") -> None:
        if seq_len < 12:
            raise ValueError("seq_len must be >= 12")
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
        self.memory_len = max(4, seq_len // 3)
        if vocab_size <= self.memory_len + 8:
            raise ValueError("vocab_size must leave room for query tokens")
        self.vocab_size = vocab_size
        self.query_base = vocab_size - self.memory_len
        self.generator = torch.Generator().manual_seed(seed)

    def _sample_memory(self, batch_size: int) -> Tensor:
        indices = torch.randint(0, len(self.dataset), (batch_size,), generator=self.generator).tolist()
        rows = self.dataset.select(indices)
        memories: list[list[int]] = []
        for values in rows[self.column]:
            if len(values) < self.memory_len:
                raise ValueError(f"dataset row has length {len(values)}, shorter than memory_len={self.memory_len}")
            memories.append([int(token) % self.query_base for token in values[: self.memory_len]])
        return torch.tensor(memories, dtype=torch.long)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        values = self._sample_memory(batch_size)
        sequence = values
        while sequence.shape[1] + 2 <= self.seq_len:
            positions = torch.randint(0, self.memory_len, (batch_size, 1), generator=self.generator)
            queries = self.query_base + positions
            answers = values.gather(1, positions)
            sequence = torch.cat([sequence, queries, answers], dim=1)
        if sequence.shape[1] < self.seq_len:
            filler_positions = torch.randint(0, self.memory_len, (batch_size, self.seq_len - sequence.shape[1]), generator=self.generator)
            filler = values.gather(1, filler_positions)
            sequence = torch.cat([sequence, filler], dim=1)
        return sequence.to(device)

    def loss_mask(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        mask = torch.zeros(batch_size, self.seq_len - 1, dtype=torch.bool)
        for position in range(self.memory_len, self.seq_len - 1, 2):
            mask[:, position] = True
        return mask.to(device)

    def intervention_split(self) -> int:
        return self.memory_len
