"""Privacy-safe receipt for row ordinals actually consumed by a training attempt."""

from __future__ import annotations

import hashlib
import json


class ConsumedRowOrder:
    """Stream private row ordinals after successful training microbatches."""

    def __init__(self, *, attempt_id: str, seed: int, shuffle: bool):
        if not attempt_id or not isinstance(seed, int):
            raise ValueError("row-order receipt requires an attempt identity and integer seed")
        self.attempt_id = attempt_id
        self.seed = seed
        self.shuffle = shuffle
        self._hash = hashlib.sha256(b"gliner2-consumed-row-order-v1\0")
        self.sample_count = 0
        self.batch_count = 0
        self.epochs: set[int] = set()

    def observe(self, *, epoch: int, ordinals: tuple[int, ...]) -> None:
        """Count one batch in the main trainer process after it is consumed."""
        if epoch < 0 or not ordinals or any(index < 0 for index in ordinals):
            raise ValueError("consumed row ordinals must be a nonempty batch")
        canonical = json.dumps(
            {"epoch": epoch, "ordinals": ordinals}, separators=(",", ":")
        ).encode("ascii")
        self._hash.update(len(canonical).to_bytes(8, "big"))
        self._hash.update(canonical)
        self.sample_count += len(ordinals)
        self.batch_count += 1
        self.epochs.add(epoch)

    def receipt(self) -> dict[str, object]:
        """Return only attempt binding, digest, counts, seed and shuffle policy."""
        if not self.batch_count:
            raise ValueError("no training batch was consumed")
        return {
            "source": "gliner2_consumed_training_batches_v1",
            "attempt_id": self.attempt_id,
            "sha256": "sha256:" + self._hash.hexdigest(),
            "sample_count": self.sample_count,
            "batch_count": self.batch_count,
            "epoch_count": len(self.epochs),
            "seed": self.seed,
            "shuffle": self.shuffle,
        }
