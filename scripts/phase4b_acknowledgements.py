"""Parse the closed Phase 4B pull-request acknowledgement grammar."""

from __future__ import annotations

import json
import re

from phase4b_contract_text import OracleError


ACKNOWLEDGEMENT_KEY = "Frozen-contract-change:"
ACKNOWLEDGEMENT_LINE = re.compile(
    r"^Frozen-contract-change: (?P<path>\S.*)$"
)
ACKNOWLEDGEMENT_NEAR_MISS = re.compile(
    r"^[\s>]*(?:[-*+]\s+)?frozen[-_ ]?contract[-_ ]?change\s*:",
    re.IGNORECASE,
)
FENCE_LINE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})")
ACKNOWLEDGEMENT_PATH = re.compile(
    r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$"
)


def acknowledgement_identity(value: str) -> tuple[str, str]:
    """Parse one file, atom, or region acknowledgement address."""
    if value.startswith("atom:"):
        identity = value[5:]
        if re.fullmatch(r"(?:04|05)-[A-Z]+-[1-9][0-9]*", identity):
            return "atom", identity
    elif value.startswith("region:"):
        try:
            identity = json.loads(value[7:])
        except (ValueError, TypeError):
            identity = None
        if (
            isinstance(identity, str)
            and identity.strip()
            and all(ord(char) >= 32 and ord(char) != 127 for char in identity)
        ):
            return "region", identity
    elif ACKNOWLEDGEMENT_PATH.fullmatch(value) and all(
        segment not in {".", ".."} for segment in value.split("/")
    ):
        return "file", value
    label = "identity" if value.startswith(("atom:", "region:")) else "path"
    raise OracleError(
        f"malformed frozen contract acknowledgement {label} {value!r}: expected a "
        "repo-relative POSIX path, atom:04/05-ID-N, or region:<JSON string>"
    )


def parse_acknowledgements(body: str) -> tuple[list[str], list[str]]:
    """Return valid addresses and errors from a pull-request body."""
    addresses: list[str] = []
    errors: list[str] = []
    open_fence: str | None = None
    for raw in body.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.rstrip()
        fence = FENCE_LINE.match(line)
        if fence is not None:
            run = fence.group("fence")
            if open_fence is None:
                open_fence = run
                continue
            if run[0] == open_fence[0] and len(run) >= len(open_fence):
                open_fence = None
            continue
        if open_fence is not None:
            continue
        match = ACKNOWLEDGEMENT_LINE.match(line)
        if match is None:
            if ACKNOWLEDGEMENT_NEAR_MISS.match(line):
                errors.append(
                    f"malformed frozen contract acknowledgement {line!r}: the "
                    f"only accepted form is '{ACKNOWLEDGEMENT_KEY} <address>' "
                    "at the start of a line, outside a code fence"
                )
            continue
        candidate = match.group("path")
        try:
            acknowledgement_identity(candidate)
        except OracleError as error:
            errors.append(str(error))
            continue
        addresses.append(candidate)
    if open_fence is not None:
        errors.append(
            f"unclosed {open_fence!r} code fence in the acknowledgement "
            "document: every line after it was ignored, so an acknowledgement "
            "there would be lost"
        )
    return addresses, errors
