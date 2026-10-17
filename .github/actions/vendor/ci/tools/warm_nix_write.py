"""Nix cache constants, the niks3 client pin, and path-info parsing for `build_cache.py`.

Tunnet membership is the only access requirement. The pinned niks3 CLI
requires a nonempty token argument, so it receives a public, nonsecret value
that the mesh server ignores.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# The cache serves its API on the private mesh name only.
SERVER_URL = "https://cache.mesh.cproof.ai"
# niks3 1.10.1 cmdutil.ResolveTokenSource rejects missing/empty tokens, even
# though the client library supports anonymous access. This public protocol
# value satisfies the CLI, not authentication. Only Tunnet admits the request.
# Explicit --auth-token also prevents fallback to a token in the host's XDG config.
# https://github.com/Mic92/niks3/blob/v1.10.1/cmdutil/cmdutil.go
ANONYMOUS_TOKEN = "tunnet-anonymous"
# The reviewed niks3 client release for Linux x86_64 and the SHA-256 of its archive.
NIKS3_VERSION = "1.10.1"
NIKS3_URL = (
    "https://github.com/Mic92/niks3/releases/download/v1.10.1/niks3_Linux_x86_64.tar.gz"
)
NIKS3_SHA256 = "4857272d97db083eb090eca30b6ad9c4c541b8c2ffa33e9bcc315945ec47284c"
SIGNING_KEY = "sand-dollar-niks3"
PUBLIC_KEY = "sand-dollar-niks3:NIdyQjpputnL1N5Utmm+4ve1/ugXmLLB6RSN1xnzBtI="
STORE_PATH = re.compile(
    r"/nix/store/[0-9abcdfghijklmnpqrsvwxyz]{32}-[A-Za-z0-9.+_?=-]{1,211}"
)
NAR_HASH = re.compile(r"sha256-[A-Za-z0-9+/]{43}=")


class WriteError(ValueError):
    """A fixed diagnostic without credential or endpoint detail."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise WriteError("nix-cache:" + code)


@dataclass(frozen=True, slots=True)
class PathRecord:
    path: str
    nar_hash: str
    signatures: tuple[str, ...]


def parse_path_info(document: str) -> tuple[PathRecord, ...]:
    """Parse `nix path-info --json` into bounded records.

    Nix prints one object keyed by store path. A later format wraps that
    object in a `paths` member beside a `version` member.
    """
    value = json.loads(document)
    require(isinstance(value, dict), "path-info-shape")
    if isinstance(value.get("paths"), dict):
        value = value["paths"]
    records: list[PathRecord] = []
    for path, info in value.items():
        require(
            isinstance(path, str)
            and STORE_PATH.fullmatch(path) is not None
            and isinstance(info, dict),
            "path-info-shape",
        )
        nar_hash = info.get("narHash")
        signatures = info.get("signatures", [])
        require(
            isinstance(nar_hash, str)
            and NAR_HASH.fullmatch(nar_hash) is not None
            and isinstance(signatures, list)
            and all(isinstance(item, str) for item in signatures),
            "path-info-shape",
        )
        records.append(PathRecord(path, nar_hash, tuple(signatures)))
    return tuple(sorted(records, key=lambda record: record.path))


def signature_keys(signatures: tuple[str, ...]) -> tuple[str, ...]:
    """The key names of the signatures on one path."""
    return tuple(sorted({item.split(":", 1)[0] for item in signatures if ":" in item}))
