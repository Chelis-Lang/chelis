"""Closed parser for the selected Devenv action profile record."""

from __future__ import annotations

import json
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn

PROFILE_SCHEMA = "chelis-devenv-profiles/v1"
MAX_PROFILE_BYTES = 64 * 1024
VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
SYSTEMS = frozenset({"aarch64-darwin", "x86_64-linux"})


@dataclass(frozen=True, slots=True)
class ProfileRecord:
    schema: str
    action: str
    profile: str
    systems: tuple[str, ...]
    constructor: str
    version: str
    digest: str | None
    runtime: Mapping[str, str]


def _fail(reason: str) -> NoReturn:
    raise ValueError(reason)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail("profile record contains a duplicate field")
        result[key] = value
    return result


def load_profile_record(
    env: Mapping[str, str],
    *,
    profile: str,
    action: str,
    constructor: str,
    systems: tuple[str, ...],
) -> ProfileRecord:
    """Parse the selected profile record at the action boundary."""
    if env.get("CHELIS_DEVENV_PROFILE") != profile:
        _fail("the selected Devenv profile is not reviewed")
    raw_path = env.get("CHELIS_DEVENV_PROFILE_RECORD")
    if not isinstance(raw_path, str) or not raw_path:
        _fail("the selected Devenv profile record is missing")
    path = Path(raw_path)
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            _fail("the selected Devenv profile record is not a regular file")
        if metadata.st_size > MAX_PROFILE_BYTES:
            _fail("the selected Devenv profile record exceeds the size limit")
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except ValueError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "the selected Devenv profile record cannot be parsed"
        ) from error
    if not isinstance(value, Mapping):
        _fail("the selected Devenv profile record is not an object")
    required = {
        "schema",
        "action",
        "profile",
        "systems",
        "constructor",
        "version",
        "digest",
    }
    if set(value) not in (required, required | {"runtime"}):
        _fail("the selected Devenv profile record has unknown or missing fields")
    raw_systems = value.get("systems")
    version = value.get("version")
    digest = value.get("digest")
    runtime = value.get("runtime", {})
    if (
        value.get("schema") != PROFILE_SCHEMA
        or value.get("action") != action
        or value.get("profile") != profile
        or value.get("constructor") != constructor
        or not isinstance(raw_systems, list)
        or tuple(raw_systems) != systems
        or any(system not in SYSTEMS for system in raw_systems)
        or not isinstance(version, str)
        or VERSION_PATTERN.fullmatch(version) is None
        or (
            digest is not None
            and (
                not isinstance(digest, str) or DIGEST_PATTERN.fullmatch(digest) is None
            )
        )
        or not isinstance(runtime, Mapping)
        or not all(
            isinstance(name, str)
            and isinstance(item, str)
            and VERSION_PATTERN.fullmatch(item) is not None
            for name, item in runtime.items()
        )
    ):
        _fail("the selected Devenv profile record does not match the action")
    return ProfileRecord(
        schema=PROFILE_SCHEMA,
        action=action,
        profile=profile,
        systems=tuple(raw_systems),
        constructor=constructor,
        version=version,
        digest=digest,
        runtime=MappingProxyType(dict(runtime)),
    )


def require_profile_match(
    record: ProfileRecord, *, version: str, digest: str | None = None
) -> None:
    """Require caller compatibility inputs to equal the selected profile."""
    if version != record.version or (digest is not None and digest != record.digest):
        _fail("the caller compatibility input does not match the selected profile")
