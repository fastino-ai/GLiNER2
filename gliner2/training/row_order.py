"""Privacy-safe receipt for the private row identities a training attempt consumed."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence

_IDENTITY = re.compile(r"[0-9a-f]{64}\Z")
_BINDING = re.compile(r"sha256:[0-9a-f]{64}\Z")
SOURCE = "gliner2_consumed_training_batches_v2"


class ConsumedRowOrder:
    """Hash caller-supplied private row identities in the order batches are consumed.

    ``row_identities[i]`` is an opaque 64-hex identity for source position ``i``
    of the training data handed to the trainer. Identities, ordinals and text
    never leave this object; the receipt carries only the digest and counts.
    """

    def __init__(
        self,
        *,
        attempt_id: str,
        seed: int,
        shuffle: bool,
        row_identities: Sequence[str],
        source_binding: str,
    ):
        if not attempt_id or not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("row-order receipt requires an attempt identity and integer seed")
        if not isinstance(source_binding, str) or _BINDING.fullmatch(source_binding) is None:
            raise ValueError("row-order receipt requires a sha256 source binding")
        identities = tuple(row_identities)
        if not identities or any(
            not isinstance(value, str) or _IDENTITY.fullmatch(value) is None for value in identities
        ):
            raise ValueError("row-order receipt requires one 64-hex identity per training row")
        self.attempt_id = attempt_id
        self.seed = seed
        self.shuffle = shuffle
        self.source_binding = source_binding
        self.sampler: str | None = None
        self._identities = identities
        self._hash = hashlib.sha256(b"gliner2-consumed-row-order-v2\0")
        self.sample_count = 0
        self.batch_count = 0
        self.epochs: set[int] = set()

    @property
    def row_count(self) -> int:
        return len(self._identities)

    def bind_sampler(self, sampler: str) -> None:
        """Record the sampler that orders training batches; it cannot change mid-attempt."""
        if not sampler or self.sampler not in (None, sampler):
            raise ValueError("row-order receipt sampler cannot change within an attempt")
        self.sampler = sampler

    def observe(self, *, epoch: int, positions: tuple[int, ...]) -> None:
        """Count one consumed batch by the source positions of its rows."""
        if epoch < 0 or not positions or any(
            not 0 <= index < len(self._identities) for index in positions
        ):
            raise ValueError("consumed row positions must be a nonempty batch of source rows")
        canonical = json.dumps(
            {"epoch": epoch, "rows": [self._identities[index] for index in positions]},
            separators=(",", ":"),
        ).encode("ascii")
        self._hash.update(len(canonical).to_bytes(8, "big"))
        self._hash.update(canonical)
        self.sample_count += len(positions)
        self.batch_count += 1
        self.epochs.add(epoch)

    def receipt(self) -> dict[str, object]:
        """Return the attempt binding, digest, counts, sampler, seed and source binding."""
        if not self.batch_count:
            raise ValueError("no training batch was consumed")
        if self.sampler is None:
            raise ValueError("row-order receipt has no bound sampler")
        return {
            "source": SOURCE,
            "attempt_id": self.attempt_id,
            "sha256": "sha256:" + self._hash.hexdigest(),
            "sample_count": self.sample_count,
            "batch_count": self.batch_count,
            "epoch_count": len(self.epochs),
            "seed": self.seed,
            "shuffle": self.shuffle,
            "sampler": self.sampler,
            "source_binding": self.source_binding,
        }
