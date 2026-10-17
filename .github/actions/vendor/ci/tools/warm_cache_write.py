"""Kache remote constants and the transfer summary parser for `build_cache.py`.

Kache 0.12.0 requires signing credentials. Public, nonsecret placeholders
satisfy its client-side SigV4 requirement. The gateway requires only network
membership, not those values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BUCKET = "sand-dollar-kache-production-010928226848"
ENDPOINT = "https://kache.mesh.cproof.ai"
REGION = "us-east-2"
CACHE_KEY = re.compile(r"[0-9a-f]{64}")
# Kache prints a carriage-return counter and then one summary per direction.
TRANSFER = re.compile(
    r"(?:^|[\r\n])\s*(Uploaded|Downloaded):\s+(\d+)/(\d+)"
    r"(?:\s+\((\d+) failed\))?\s*(?=\r|\n|$)"
)
# Kache 0.12.0 does not expose OpenDAL's skip_signature option, and its pinned
# reqsign-core 3.1.0 rejects missing credentials before invoking the signer.
# These PUBLIC protocol placeholders are not AWS credentials or gateway auth.
# Kache's static provider wins before any profile/broker credential lookup.
PUBLIC_SIGNING_VALUE = "tunnet-anonymous"


@dataclass(frozen=True, slots=True)
class Transfer:
    completed: int
    failed: int


def parse_transfer(stream: str, direction: str) -> Transfer | None:
    """Return the last summary for one direction, or None when Kache printed none."""
    found = [
        Transfer(int(match.group(2)), int(match.group(4) or 0))
        for match in TRANSFER.finditer(stream)
        if match.group(1) == direction
    ]
    return found[-1] if found else None
