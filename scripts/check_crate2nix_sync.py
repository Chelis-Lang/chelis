#!/usr/bin/env python3
"""Check the digest of the Cargo inputs that generated Cargo.nix."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
MARKER_PREFIX = "# chelis-crate2nix-input-sha256: "
MARKER_RE = re.compile(rf"(?m)^{re.escape(MARKER_PREFIX)}([0-9a-f]{{64}})$")


class InputParseError(ValueError):
    """A crate2nix graph input does not satisfy the repository contract."""


@dataclass(frozen=True)
class WorkspaceInputs:
    root: Path
    cargo_nix: Path
    graph_inputs: tuple[Path, ...]

    @classmethod
    def parse(cls, root: Path) -> WorkspaceInputs:
        resolved_root = root.resolve()
        root_manifest = resolved_root / "Cargo.toml"
        cargo_lock = resolved_root / "Cargo.lock"
        cargo_nix = resolved_root / "Cargo.nix"

        for required in (root_manifest, cargo_lock, cargo_nix):
            if not required.is_file():
                raise InputParseError(f"required file does not exist: {required}")

        try:
            document = tomllib.loads(root_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
            raise InputParseError(f"cannot parse {root_manifest}: {error}") from error

        workspace = document.get("workspace")
        if not isinstance(workspace, dict):
            raise InputParseError(f"{root_manifest}: workspace must be a table")
        raw_members = workspace.get("members")
        if not isinstance(raw_members, list) or not raw_members:
            raise InputParseError(f"{root_manifest}: workspace.members must be a nonempty array")

        member_manifests: list[Path] = []
        for raw_member in raw_members:
            if not isinstance(raw_member, str) or not raw_member:
                raise InputParseError(
                    f"{root_manifest}: each workspace member must be a nonempty string"
                )
            member = PurePosixPath(raw_member)
            if member.is_absolute() or ".." in member.parts:
                raise InputParseError(
                    "workspace member must stay below the repository root: "
                    f"{raw_member}"
                )
            manifest = (resolved_root / Path(*member.parts) / "Cargo.toml").resolve()
            if not manifest.is_relative_to(resolved_root):
                raise InputParseError(
                    "workspace member must stay below the repository root: "
                    f"{raw_member}"
                )
            if not manifest.is_file():
                raise InputParseError(f"workspace manifest does not exist: {manifest}")
            member_manifests.append(manifest)

        graph_inputs = tuple(
            sorted(
                {root_manifest, cargo_lock, *member_manifests},
                key=lambda path: path.relative_to(resolved_root).as_posix(),
            )
        )
        return cls(
            root=resolved_root,
            cargo_nix=cargo_nix,
            graph_inputs=graph_inputs,
        )

    def digest(self) -> str:
        digest = hashlib.sha256()
        for path in self.graph_inputs:
            relative = path.relative_to(self.root).as_posix().encode("utf-8")
            content = path.read_bytes()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest()

    def recorded_digest(self) -> str | None:
        text = self.cargo_nix.read_text(encoding="utf-8")
        match = MARKER_RE.search(text)
        return None if match is None else match.group(1)

    def write_digest(self) -> str:
        expected = self.digest()
        text = self.cargo_nix.read_text(encoding="utf-8")
        marker = f"{MARKER_PREFIX}{expected}"
        if MARKER_RE.search(text):
            updated = MARKER_RE.sub(marker, text, count=1)
        else:
            first_line, separator, remainder = text.partition("\n")
            if separator:
                updated = f"{first_line}\n{marker}\n{remainder}"
            else:
                updated = f"{text}\n{marker}\n"
        updated = updated.rstrip("\n") + "\n"
        self.cargo_nix.write_text(updated, encoding="utf-8")
        return expected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inputs = WorkspaceInputs.parse(args.root)
        if args.write:
            digest = inputs.write_digest()
            print(f"updated Cargo.nix input digest: {digest}")
            return 0

        recorded = inputs.recorded_digest()
        expected = inputs.digest()
    except (InputParseError, OSError, UnicodeError) as error:
        print(f"invalid crate2nix input: {error}", file=sys.stderr)
        return 2

    if recorded is None:
        print("Cargo.nix input digest is missing", file=sys.stderr)
        return 1
    if recorded != expected:
        print("Cargo.nix input digest is stale", file=sys.stderr)
        print(f"  recorded: {recorded}", file=sys.stderr)
        print(f"  expected: {expected}", file=sys.stderr)
        return 1

    print(f"Cargo.nix inputs match: {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
