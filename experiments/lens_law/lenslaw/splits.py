"""Deterministic dev/test assignment, fixed before any image is scored."""

from __future__ import annotations

import hashlib


def assign_split(group_key: str, salt: str = "lenslaw-v1", dev_fraction: float = 0.6) -> str:
    """Return "dev" or "test" for a scene, photographer or generator key.

    Everything that shares a key lands on the same side, so one photographer
    or one scene never appears in both splits.
    """
    digest = hashlib.sha256(f"{salt}/{group_key}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "dev" if fraction < dev_fraction else "test"
