#!/usr/bin/env python3
"""Generate a lines-of-code report for the Chelis repository.

Runs tokei for standard languages, then counts Chelis-specific file types
(.ch, .dp, .scm) that tokei doesn't recognize. Outputs a markdown table
to docs/loc_report.md.

Usage:
    python scripts/loc_report.py              # write to docs/loc_report.md
    python scripts/loc_report.py --stdout     # print to stdout instead
    python scripts/loc_report.py --json       # print raw data as JSON
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LangEntry:
    language: str
    files: int
    code: int
    comments: int
    blanks: int
    notes: str = ""

    @property
    def total_lines(self) -> int:
        return self.code + self.comments + self.blanks


DEFAULT_OUTPUT = "docs/loc_report.md"

# tokei language names to keep, mapped to display names and notes
TOKEI_LANGUAGES: dict[str, tuple[str, str]] = {
    "Rust": ("Rust", "Compiler, CLI, runtime, backends, type checker"),
    "C": ("C", "Generated runtime"),
    "Markdown": ("Markdown", "Specs, design docs, plans"),
    "JSON": ("JSON", "Package metadata, test fixtures"),
    "Python": ("Python", "PyO3 bindings, scripts, benchmarks"),
    "C Header": ("C Header", "Runtime headers"),
    "JavaScript": ("JavaScript", "Tree-sitter grammar definitions"),
    "TOML": ("TOML", "Cargo/reef manifests"),
}

# tokei languages to skip (we count these manually with proper names)
TOKEI_SKIP = {"Pest", "Scheme"}

# Custom file types that tokei doesn't know about
CUSTOM_TYPES: list[tuple[str, str, str]] = [
    # (glob_pattern, display_name, notes)
    ("**/*.ch", "Chelis Surf (.ch)", "Examples, std library, test fixtures"),
    ("**/*.dp", "Chelis Deep (.dp)", "Deep test fixtures"),
    ("**/*.pest", "PEG Grammars (.pest)", "Validator grammars (Surf + Deep)"),
    ("**/*.scm", "Tree-sitter Queries (.scm)", "Syntax highlighting for Surf and Deep"),
]


def find_repo_root() -> Path:
    """Walk up from cwd to find the directory containing Cargo.toml workspace."""
    p = Path.cwd()
    while p != p.parent:
        if (p / "Cargo.toml").exists() and (p / "crates").is_dir():
            return p
        p = p.parent
    return Path.cwd()


def run_tokei(repo_root: Path) -> dict:
    """Run tokei --output json and return parsed results."""
    result = subprocess.run(
        ["tokei", "--output", "json", str(repo_root)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def parse_tokei(data: dict) -> list[LangEntry]:
    """Extract language entries from tokei JSON output."""
    entries = []
    for lang_name, (display, notes) in TOKEI_LANGUAGES.items():
        info = data.get(lang_name)
        if info is None:
            continue
        entries.append(
            LangEntry(
                language=display,
                files=len(info.get("reports", [])),
                code=info["code"],
                comments=info["comments"],
                blanks=info["blanks"],
                notes=notes,
            )
        )
    return entries


def count_files(repo_root: Path, glob_pattern: str) -> LangEntry:
    """Count lines in files matching a glob pattern under repo_root.

    Returns a LangEntry with code = total non-blank lines, blanks = blank lines,
    comments = 0 (we can't reliably detect comment syntax for custom types).
    """
    files = sorted(repo_root.glob(glob_pattern))
    # Exclude anything under target/ or .git/
    files = [
        f
        for f in files
        if not any(part in ("target", ".git") for part in f.parts)
    ]
    total_lines = 0
    blank_lines = 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            total_lines += 1
            if not line.strip():
                blank_lines += 1

    code = total_lines - blank_lines
    return LangEntry(
        language="",  # caller sets this
        files=len(files),
        code=code,
        comments=0,
        blanks=blank_lines,
    )


def build_report(repo_root: Path) -> list[LangEntry]:
    """Build the full LOC report: tokei languages + custom types."""
    tokei_data = run_tokei(repo_root)
    entries = parse_tokei(tokei_data)

    for glob_pat, display, notes in CUSTOM_TYPES:
        entry = count_files(repo_root, glob_pat)
        entry.language = display
        entry.notes = notes
        if entry.files > 0:
            entries.append(entry)

    # Sort by code descending
    entries.sort(key=lambda e: e.code, reverse=True)
    return entries


def format_number(n: int) -> str:
    """Format integer with comma separators."""
    return f"{n:,}"


def render_markdown(entries: list[LangEntry], repo_root: Path) -> str:
    """Render the report as a markdown document."""
    total_files = sum(e.files for e in entries)
    total_code = sum(e.code for e in entries)
    total_comments = sum(e.comments for e in entries)
    total_blanks = sum(e.blanks for e in entries)
    total_lines = sum(e.total_lines for e in entries)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Lines of Code Report",
        "",
        f"Generated: {timestamp}",
        "",
        "| Language | Files | Code | Comments | Blanks | Total | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for e in entries:
        lines.append(
            f"| {e.language} | {format_number(e.files)} "
            f"| {format_number(e.code)} | {format_number(e.comments)} "
            f"| {format_number(e.blanks)} | {format_number(e.total_lines)} "
            f"| {e.notes} |"
        )

    lines.append(
        f"| **Total** | **{format_number(total_files)}** "
        f"| **{format_number(total_code)}** "
        f"| **{format_number(total_comments)}** "
        f"| **{format_number(total_blanks)}** "
        f"| **{format_number(total_lines)}** | |"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LOC report for Chelis")
    parser.add_argument("--stdout", action="store_true", help="Print to stdout")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    repo_root = find_repo_root()
    entries = build_report(repo_root)

    if args.json:
        data = [
            {
                "language": e.language,
                "files": e.files,
                "code": e.code,
                "comments": e.comments,
                "blanks": e.blanks,
                "total": e.total_lines,
                "notes": e.notes,
            }
            for e in entries
        ]
        print(json.dumps(data, indent=2))
        return

    md = render_markdown(entries, repo_root)

    if args.stdout:
        print(md)
        return

    output_path = Path(args.output) if args.output else repo_root / DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
