#!/usr/bin/env python3
"""Generate deterministic chelis#1207 binding-count benchmark fixtures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


class FixtureError(ValueError):
    """Raised when a fixture request cannot be generated unambiguously."""


@dataclass(frozen=True)
class Fixture:
    """One generated fixture and its repository-independent file name."""

    filename: str
    source: str


def _require_positive_unique(values: Sequence[int], label: str) -> tuple[int, ...]:
    normalized = tuple(values)
    if not normalized:
        raise FixtureError(f"{label} must contain at least one value")
    if any(value <= 0 for value in normalized):
        raise FixtureError(f"{label} values must be positive")
    if len(set(normalized)) != len(normalized):
        raise FixtureError(f"{label} values must be unique")
    return normalized


def _render_flat(bindings: int) -> str:
    lines = [f"module TypecheckGeneralization.FlatN{bindings}"]
    for index in range(bindings):
        lines.append(
            f"def flat_{index}(x: f32) -> f32 = "
            f"mul(x, cast({index + 1}.0, f32))"
        )
    lines.append("out = flat_0(1.0)")
    return "\n".join(lines) + "\n"


def _render_shallow(bindings: int) -> str:
    lines = [
        f"module TypecheckGeneralization.ShallowN{bindings}",
        "def shared_base(x: f32) -> f32 = mul(x, cast(2.0, f32))",
    ]
    for index in range(bindings):
        lines.append(
            f"def shallow_{index}(x: f32) -> f32 = "
            f"add(x, cast({index + 1}.0, f32)) |> shared_base"
        )
    lines.append("out = shallow_0(1.0)")
    return "\n".join(lines) + "\n"


def _render_let_body(bindings: int, *, independent: bool) -> list[str]:
    lines = ["def main(x: f32) -> f32 = {"]
    for index in range(bindings):
        operand = "x" if independent or index == 0 else f"value_{index - 1}"
        lines.append(
            f"  value_{index} = add({operand}, cast({index + 1}.0, f32))"
        )
    lines.extend([f"  value_{bindings - 1}", "}"])
    return lines


def _render_lets(bindings: int, *, independent: bool) -> str:
    shape = "Letsind" if independent else "Lets"
    lines = [f"module TypecheckGeneralization.{shape}N{bindings}"]
    lines.extend(_render_let_body(bindings, independent=independent))
    lines.append("out = main(1.0)")
    return "\n".join(lines) + "\n"


def _render_split(total: int, split_count: int) -> str:
    per_definition = total // split_count
    lines = [f"module TypecheckGeneralization.SplitN{total}K{split_count}"]
    next_binding = 0
    for definition in range(split_count):
        lines.append(f"def split_{definition}(x: f32) -> f32 = {{")
        for local_index in range(per_definition):
            operand = "x" if local_index == 0 else f"value_{next_binding - 1}"
            lines.append(
                f"  value_{next_binding} = "
                f"add({operand}, cast({next_binding + 1}.0, f32))"
            )
            next_binding += 1
        lines.extend([f"  value_{next_binding - 1}", "}"])
    lines.append("out = split_0(1.0)")
    return "\n".join(lines) + "\n"


def build_fixtures(
    bindings: Sequence[int], split_total: int, split_counts: Sequence[int]
) -> tuple[Fixture, ...]:
    """Build fixtures in stable shape/size order without touching the filesystem."""

    binding_counts = _require_positive_unique(bindings, "bindings")
    counts = _require_positive_unique(split_counts, "split counts")
    if split_total <= 0:
        raise FixtureError("split total must be positive")
    for count in counts:
        if split_total % count != 0:
            raise FixtureError(
                f"split total {split_total} is not divisible by split count {count}"
            )

    renderers: tuple[tuple[str, Callable[[int], str]], ...] = (
        ("flat", _render_flat),
        ("shallow", _render_shallow),
        ("lets", lambda count: _render_lets(count, independent=False)),
        ("letsind", lambda count: _render_lets(count, independent=True)),
    )
    fixtures = [
        Fixture(f"{shape}_n{count}.ch", render(count))
        for shape, render in renderers
        for count in binding_counts
    ]
    fixtures.extend(
        Fixture(
            f"split_n{split_total}_k{count}.ch",
            _render_split(split_total, count),
        )
        for count in counts
    )
    return tuple(fixtures)


def write_fixtures(output_dir: Path, fixtures: Sequence[Fixture]) -> tuple[Path, ...]:
    """Write fixtures atomically with respect to pre-existing target paths."""

    if output_dir.exists() and not output_dir.is_dir():
        raise FixtureError(f"output path is not a directory: {output_dir}")
    targets = tuple(output_dir / fixture.filename for fixture in fixtures)
    existing = [target for target in targets if target.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FixtureError(f"refusing to overwrite existing fixture(s): {names}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for target, fixture in zip(targets, fixtures, strict=True):
        target.write_text(fixture.source, encoding="utf-8")
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bindings", required=True, nargs="+", type=int)
    parser.add_argument("--split-total", required=True, type=int)
    parser.add_argument("--split-counts", required=True, nargs="+", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        fixtures = build_fixtures(args.bindings, args.split_total, args.split_counts)
        paths = write_fixtures(args.output_dir, fixtures)
    except FixtureError as error:
        raise SystemExit(str(error)) from error
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
