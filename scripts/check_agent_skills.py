#!/usr/bin/env python3
"""Validate shared skill metadata, agent/embedded copies, and command wrappers.

CI runs this on documentation-only changes as well as code changes. The
Rust asset-drift and skill-uniformity tests separately exercise the compiled
embed and downstream distribution paths. Requires PyYAML 6.0.3; CI supplies
it through uv, without depending on a developer's installed agent tooling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import yaml

try:
    from .regenerate_conformance_assets import (
        SHARED_SKILLS,
        agent_surface_layout_reasons,
        is_stale,
        planned_agent_surface_copies,
        planned_skill_copies,
    )
except ImportError:
    from regenerate_conformance_assets import (
        SHARED_SKILLS,
        agent_surface_layout_reasons,
        is_stale,
        planned_agent_surface_copies,
        planned_skill_copies,
    )


class UniqueKeyLoader(yaml.SafeLoader):
    """Do not silently replace a skill field with a later duplicate."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            if key in result:
                raise ValueError(f"duplicate metadata key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def validate_skill(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
        match = re.match(r"\A---\n(.*?)\n---(?:\n|$)", text, re.DOTALL)
        if match is None:
            raise ValueError("missing YAML frontmatter delimiters")
        metadata = yaml.load(match.group(1), Loader=UniqueKeyLoader)
        if not isinstance(metadata, dict):
            raise ValueError("frontmatter must be a mapping")
        extra = set(metadata) - {"name", "description", "license", "allowed-tools", "metadata"}
        if extra:
            raise ValueError(f"unknown frontmatter fields: {sorted(extra)}")
        name = metadata.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) > 64:
            raise ValueError("name must be a lowercase hyphenated identifier of at most 64 characters")
        if name != path.parent.name:
            raise ValueError("name must match the skill directory")
        description = metadata.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a nonempty string")
        if len(description) > 1024 or "<" in description or ">" in description:
            raise ValueError("description must be at most 1024 characters with no angle brackets")
        if description.lstrip().startswith("[TODO:"):
            raise ValueError("description contains an unfinished placeholder")
        # Match the skill-creator validator's unfinished-instruction rule while
        # allowing literal placeholder examples inside Markdown code fences.
        fence_marker = None
        fence_length = 0
        for line in text[match.end():].splitlines():
            fence = re.match(r"^[ \t]*(?:(?:[-+*]|\d+[.)])[ \t]+)?(`{3,}|~{3,})(.*)$", line)
            if fence:
                marker = fence.group(1)
                if fence_marker is None:
                    fence_marker = marker[0]
                    fence_length = len(marker)
                elif marker[0] == fence_marker and len(marker) >= fence_length and not fence.group(2).strip():
                    fence_marker = None
                    fence_length = 0
                continue
            if fence_marker is None and re.fullmatch(r"[ ]{0,3}\[TODO:[^\n]*\][ \t]*", line):
                raise ValueError("instructions contain an unfinished TODO placeholder")
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        return [f"{path}: {exc}"]
    return []


def check(root: Path) -> list[str]:
    errors = []
    source = root / "agent-skills"
    try:
        names = {path.name for path in source.iterdir() if path.is_dir()}
        if names != set(SHARED_SKILLS):
            errors.append("agent-skills directory differs from the registered shared skill set")
        pairs = planned_skill_copies(root)
        for src, dest in pairs:
            errors.extend(validate_skill(src))
            errors.extend(validate_skill(dest))
        surface_pairs = planned_agent_surface_copies(root)
        for _, dest in surface_pairs:
            errors.extend(validate_skill(dest))
        errors.extend(
            is_stale(
                pairs + surface_pairs,
                root / "crates/chelis-conformance/assets/skills",
            )
        )
        errors.extend(agent_surface_layout_reasons(root))
    except (OSError, SystemExit) as exc:
        errors.append(str(exc))
    try:
        claude = (root / ".claude/commands/red-team.md").read_bytes()
        codex = (root / ".codex/commands/red-team.md").read_bytes()
        if claude != codex:
            errors.append("Claude and Codex red-team command wrappers differ")
    except OSError as exc:
        errors.append(str(exc))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    errors = check(parser.parse_args().root)
    for error in errors:
        print(error)
    print("AGENT SKILLS: FAIL" if errors else "AGENT SKILLS: PASS")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
