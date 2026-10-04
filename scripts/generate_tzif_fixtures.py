#!/usr/bin/env python3
"""Generate the TZif fixtures for `Std.Datetime.Zone` (chelis#2862).

The fixtures are binary TZif files (RFC 9636) under
`crates/chelis-cli/tests/fixtures/tzif/`:

- `zones/<IANA name>.tzif`: real zones copied from one pinned release of the
  Python `tzdata` package, which ships the IANA database compiled by `zic`.
- `synthetic/*.tzif`: small files written by the TZif writer below, each
  exercising one rule of the parser: an empty footer, malformed headers,
  non-increasing transitions, out-of-range offsets, leap-second records, an
  inconsistent footer, malformed counts, time types, indicators and footer
  framing, and the version 3 transition-time extension.

`manifest.json` records the tzdata release and the SHA-256 of every file.

Usage:

    uv run --python 3.11 --with tzdata==2026.4 scripts/generate_tzif_fixtures.py --write
    .venv/bin/python scripts/generate_tzif_fixtures.py --check

`--write` needs the pinned tzdata release installed and rewrites every
fixture and the manifest. `--check` needs no tzdata: it regenerates the
synthetic files in memory and compares them, and checks every file against
the manifest.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import importlib.metadata
import importlib.resources
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "crates/chelis-cli/tests/fixtures/tzif"
MANIFEST = FIXTURE_DIR / "manifest.json"

TZDATA_PACKAGE_VERSION = "2026.4"
IANA_RELEASE = "2026d"

ZONES = (
    "America/New_York",
    "Europe/London",
    "Australia/Lord_Howe",
    "Pacific/Kiritimati",
    "Asia/Kolkata",
    "America/Sao_Paulo",
    "UTC",
)


def zone_path(name: str) -> str:
    return f"zones/{name}.tzif"


def tzif(
    version: bytes,
    times: Sequence[int],
    indices: Sequence[int],
    types: Sequence[tuple[int, int, int]],
    chars: bytes,
    footer: bytes,
    leaps: Sequence[tuple[int, int]] = (),
    isstd: Sequence[int] = (),
    isut: Sequence[int] = (),
) -> bytes:
    """A TZif file with an empty version 1 block (zic's "slim" layout) and the
    given version 2+ data block and footer."""

    def header(counts: tuple[int, int, int, int, int, int]) -> bytes:
        return b"TZif" + version + bytes(15) + struct.pack(">6L", *counts)

    v1 = header((0, 0, 0, 0, 1, 1)) + bytes(6) + bytes(1)
    if version == b"\x00":
        return v1
    v2 = header((len(isut), len(isstd), len(leaps), len(times), len(types), len(chars)))
    data = b"".join(struct.pack(">q", t) for t in times) + bytes(indices)
    data += b"".join(struct.pack(">lBB", *t) for t in types) + chars
    data += b"".join(struct.pack(">ql", *leap) for leap in leaps) + bytes(isstd) + bytes(isut)
    return v1 + v2 + data + b"\n" + footer + b"\n"


def two_type(version: bytes, times: Sequence[int], footer: bytes, second_offset: int = 3600) -> bytes:
    """A zone at UTC until its transitions, which alternate between offset
    `second_offset` and zero."""
    indices = [(k + 1) % 2 for k in range(len(times))]
    return tzif(version, times, indices, [(0, 0, 0), (second_offset, 1, 4)], b"STD\x00DST\x00", footer)


def with_version(data: bytes, version: bytes) -> bytes:
    """`data` with both headers' version byte replaced; the version 2+ header
    follows the 44-byte header and the 7-byte slim version 1 block."""
    return data[:4] + version + data[5:55] + version + data[56:]


def indicated(isstd: Sequence[int], isut: Sequence[int]) -> bytes:
    """A two-type zone with the given standard/wall and UT/local indicators."""
    return tzif(b"2", [1_000_000_000], [1], [(0, 0, 0), (3600, 1, 4)], b"STD\x00DST\x00", b"STD-1", isstd=isstd, isut=isut)


def synthetic() -> dict[str, bytes]:
    valid = two_type(b"2", [1_000_000_000], b"STD-1")
    second = 44 + 7  # the version 2+ header's offset in a slim file
    return {
        # Coverage ends at the last transition, 2001-09-09T01:46:40Z.
        "synthetic/empty_footer.tzif": two_type(b"2", [1_000_000_000], b""),
        "synthetic/bad_magic.tzif": b"TZiF" + valid[4:],
        "synthetic/version_1.tzif": tzif(b"\x00", [], [], [], b"", b""),
        "synthetic/version_5.tzif": with_version(valid, b"5"),
        "synthetic/truncated.tzif": valid[:60],
        "synthetic/zero_typecnt.tzif": tzif(b"2", [], [], [], b"UTC\x00", b"UTC0"),
        "synthetic/non_increasing.tzif": two_type(b"2", [1_000_000_000, 1_000_000_000], b"STD0"),
        "synthetic/decreasing.tzif": two_type(b"2", [1_000_000_000, 900_000_000], b"STD0"),
        "synthetic/offset_out_of_bounds.tzif": two_type(b"2", [1_000_000_000], b"", second_offset=86400),
        "synthetic/leap_seconds.tzif": tzif(b"2", [], [], [(0, 0, 0)], b"UTC\x00", b"UTC0", leaps=[(78796800, 1)]),
        "synthetic/inconsistent_footer.tzif": two_type(b"2", [1_000_000_000], b"STD0"),
        "synthetic/dst_without_rule.tzif": two_type(b"2", [1_000_000_000], b"STD-1DST"),
        # Asia/Jerusalem's rule: daylight time starts at 26:00 on the Thursday
        # before the last Sunday of March, which needs version 3.
        "synthetic/jerusalem_v3.tzif": tzif(b"3", [978_307_200], [0], [(7200, 0, 0), (10800, 1, 4)], b"IST\x00IDT\x00", b"IST-2IDT,M3.4.4/26,M10.5.0"),
        "synthetic/jerusalem_in_v2.tzif": tzif(b"2", [978_307_200], [0], [(7200, 0, 0), (10800, 1, 4)], b"IST\x00IDT\x00", b"IST-2IDT,M3.4.4/26,M10.5.0"),
        # RFC 9636 §3.3.1: daylight time all year.
        "synthetic/all_year_dst.tzif": tzif(b"3", [], [], [(-14400, 1, 0)], b"EDT\x00", b"EST5EDT,0/0,J365/25"),
        # One file per rejection of a malformed version 2+ header, data block or footer.
        "synthetic/bad_second_magic.tzif": valid[:second] + b"TZiF" + valid[second + 4:],
        "synthetic/zero_charcnt.tzif": tzif(b"2", [], [], [(0, 0, 0)], b"", b"UTC0"),
        "synthetic/short_isstdcnt.tzif": indicated([0], []),
        "synthetic/short_isutcnt.tzif": indicated([], [0]),
        "synthetic/truncated_data.tzif": valid[:second + 44 + 5],
        "synthetic/dst_flag.tzif": tzif(b"2", [1_000_000_000], [1], [(0, 0, 0), (3600, 2, 4)], b"STD\x00DST\x00", b"STD-1"),
        "synthetic/designation_index.tzif": tzif(b"2", [1_000_000_000], [1], [(0, 0, 0), (3600, 1, 8)], b"STD\x00DST\x00", b"STD-1"),
        "synthetic/isstd_value.tzif": indicated([0, 2], []),
        "synthetic/isut_value.tzif": indicated([0, 1], [0, 2]),
        "synthetic/isut_without_isstd.tzif": indicated([0, 0], [0, 1]),
        "synthetic/footer_without_newline.tzif": valid[:-7] + b"xSTD-1\n",
        "synthetic/footer_control_byte.tzif": two_type(b"2", [1_000_000_000], b"STD\x7f-1"),
        # Their accepted twins: indicators of 0 and 1 in every allowed
        # combination, and the last designation index.
        "synthetic/indicators.tzif": indicated([0, 1], [0, 1]),
        "synthetic/last_designation.tzif": tzif(b"2", [1_000_000_000], [1], [(0, 0, 0), (3600, 1, 7)], b"STD\x00DST\x00", b"STD-1"),
    }


def real_zones() -> dict[str, bytes]:
    installed = importlib.metadata.version("tzdata")
    if installed != TZDATA_PACKAGE_VERSION:
        raise SystemExit(f"tzdata {installed} is installed; the fixtures pin {TZDATA_PACKAGE_VERSION}")
    package = importlib.import_module("tzdata")
    if package.IANA_VERSION != IANA_RELEASE:
        raise SystemExit(f"tzdata ships IANA {package.IANA_VERSION}; the fixtures pin {IANA_RELEASE}")
    root = importlib.resources.files("tzdata.zoneinfo")
    return {zone_path(name): root.joinpath(*name.split("/")).read_bytes() for name in ZONES}


def manifest(files: dict[str, bytes]) -> dict[str, object]:
    return {
        "tzdata_package": TZDATA_PACKAGE_VERSION,
        "iana_release": IANA_RELEASE,
        "zones": list(ZONES),
        "sha256": {path: hashlib.sha256(data).hexdigest() for path, data in sorted(files.items())},
    }


def write() -> None:
    files = {**real_zones(), **synthetic()}
    for path, data in files.items():
        target = FIXTURE_DIR / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    MANIFEST.write_text(json.dumps(manifest(files), indent=2) + "\n", encoding="utf-8")


def check(fixture_dir: Path = FIXTURE_DIR) -> list[str]:
    """Every difference between the fixtures in `fixture_dir` and what this
    script produces, without tzdata."""
    problems = []
    recorded = json.loads((fixture_dir / MANIFEST.name).read_text(encoding="utf-8"))
    if (recorded["tzdata_package"], recorded["iana_release"]) != (TZDATA_PACKAGE_VERSION, IANA_RELEASE):
        problems.append("the manifest records a different tzdata release")
    if recorded["zones"] != list(ZONES):
        problems.append("the manifest records a different zone list")
    expected_paths = {zone_path(name) for name in ZONES} | set(synthetic())
    if set(recorded["sha256"]) != expected_paths:
        problems.append(f"the manifest lists {sorted(recorded['sha256'])}, expected {sorted(expected_paths)}")
    on_disk = {path.relative_to(fixture_dir).as_posix() for path in fixture_dir.rglob("*.tzif")}
    if on_disk != expected_paths:
        problems.append(f"the fixture directory holds {sorted(on_disk)}, expected {sorted(expected_paths)}")
    for path, digest in recorded["sha256"].items():
        target = fixture_dir / path
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            problems.append(f"{path} does not match its recorded SHA-256")
    for path, data in synthetic().items():
        target = fixture_dir / path
        if not target.exists() or target.read_bytes() != data:
            problems.append(f"{path} differs from its generator")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="rewrite every fixture from the pinned tzdata release")
    mode.add_argument("--check", action="store_true", help="verify the checked-in fixtures without tzdata")
    args = parser.parse_args(argv)
    if args.write:
        write()
    problems = check()
    for problem in problems:
        print(f"TZIF FIXTURES: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
