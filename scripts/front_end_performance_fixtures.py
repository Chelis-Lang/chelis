#!/usr/bin/env python3
"""Generate the deterministic chelis#1205 nested/flat performance corpus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class FixtureError(ValueError):
    """Raised when a fixture request is invalid or would overwrite evidence."""


@dataclass(frozen=True)
class Fixture:
    filename: str
    source: str


def _validated_counts(values: Sequence[int]) -> tuple[int, ...]:
    counts = tuple(values)
    if not counts:
        raise FixtureError("operation counts must contain at least one value")
    if any(value <= 0 for value in counts):
        raise FixtureError("operation counts must be positive")
    if len(set(counts)) != len(counts):
        raise FixtureError("operation counts must be unique")
    return counts


def _header(shape: str, operations: int, width: int) -> list[str]:
    return [
        f"module FrontEndPerformance.{shape}N{operations}",
        f"def bc(c: f32) -> tensor[{width}, f32] = "
        f"reshape(insert(to_tensor([c]), 0, {width}i64), [{width}i64])",
    ]


def _render_nested(operations: int, width: int) -> str:
    body = "s"
    for _ in range(operations):
        body = (
            f"mul(add({body}, bc(cast(1.0, f32))), "
            "bc(cast(0.5, f32)))"
        )
    lines = _header("Nested", operations, width)
    lines.extend(
        [
            f"def st(s: tensor[{width}, f32], i: i64) -> tensor[{width}, f32] = "
            f"if gte(i, 5i64) then s else st({body}, add(i, 1i64))",
            "r = index(to_list(st(bc(cast(1.0, f32)), 0i64)), 0i64)",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_flat(operations: int, width: int) -> str:
    lines = _header("Flat", operations, width)
    lines.append(
        f"def st(s: tensor[{width}, f32], i: i64) -> tensor[{width}, f32] = "
        "if gte(i, 5i64) then s else {"
    )
    previous = "s"
    for index in range(operations):
        lines.append(
            f"  t{index} = mul(add({previous}, bc(cast(1.0, f32))), "
            "bc(cast(0.5, f32)))"
        )
        previous = f"t{index}"
    lines.extend(
        [
            f"  st({previous}, add(i, 1i64))",
            "}",
            "r = index(to_list(st(bc(cast(1.0, f32)), 0i64)), 0i64)",
        ]
    )
    return "\n".join(lines) + "\n"


def build_fixtures(
    operation_counts: Sequence[int], *, width: int = 20_000
) -> tuple[Fixture, ...]:
    """Build the issue corpus in stable size-then-shape order."""

    counts = _validated_counts(operation_counts)
    if width <= 0:
        raise FixtureError("width must be positive")
    fixtures = []
    for operations in counts:
        fixtures.extend(
            [
                Fixture(
                    f"nested_n{operations}.ch",
                    _render_nested(operations, width),
                ),
                Fixture(f"flat_n{operations}.ch", _render_flat(operations, width)),
            ]
        )
    return tuple(fixtures)


def write_fixtures(output_dir: Path, fixtures: Sequence[Fixture]) -> tuple[Path, ...]:
    if output_dir.exists() and not output_dir.is_dir():
        raise FixtureError(f"output path is not a directory: {output_dir}")
    paths = tuple(output_dir / fixture.filename for fixture in fixtures)
    existing = [path for path in paths if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FixtureError(f"refusing to overwrite existing fixture(s): {rendered}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, fixture in zip(paths, fixtures, strict=True):
        path.write_text(fixture.source, encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--operations", required=True, nargs="+", type=int)
    parser.add_argument("--width", default=20_000, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        generated = build_fixtures(args.operations, width=args.width)
        paths = write_fixtures(args.output_dir, generated)
    except FixtureError as error:
        raise SystemExit(str(error)) from error
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
