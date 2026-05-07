#!/usr/bin/env python3
"""Generate a lines-of-code report for the Chelis repository.

Uses pygount for standard language counting with manual overrides for
Chelis-specific file types (.ch, .dp, .pest, .scm) that pygount doesn't
recognize. Outputs a markdown table to docs/archive/reports/loc_report.md.

Requirements: pygount (pip install pygount)

Usage:
    python scripts/loc_report.py              # write to docs/archive/reports/loc_report.md
    python scripts/loc_report.py --stdout     # print to stdout instead
    python scripts/loc_report.py --json       # print raw data as JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pygount import SourceAnalysis


@dataclass
class LangEntry:
    language: str
    files: int
    code: int
    comments: int
    blanks: int
    notes: str = ""
    bold: bool = False

    @property
    def total_lines(self) -> int:
        return self.code + self.comments + self.blanks


DEFAULT_OUTPUT = "docs/archive/reports/loc_report.md"

# Directories to always skip
SKIP_DIRS = {"target", ".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}

# Display names and notes for languages pygount recognizes
LANGUAGE_NOTES: dict[str, str] = {
    "Rust": "Compiler, CLI, runtime, backends, type checker",
    "C": "Generated runtime, headers",
    "Markdown": "Specs, design docs, plans",
    "JSON": "Package metadata, test fixtures",
    "Python": "PyO3 bindings, tools, benchmarks",
    "JavaScript": "Tree-sitter grammar definitions",
    "TOML": "Cargo/reef manifests",
    "YAML": "CI workflows",
}

# pygount language name normalization
LANGUAGE_RENAMES: dict[str, str] = {
    "C/C++ Header": "C",
}

# pygount language names to drop (misidentified or not useful)
LANGUAGE_DROP: set[str] = {"JavaScript+Genshi Text"}


@dataclass
class CustomLang:
    """A file type pygount doesn't know about."""

    glob: str
    display: str
    notes: str
    line_comment: str | None = None
    bold: bool = False


CUSTOM_LANGUAGES: list[CustomLang] = [
    CustomLang("*.ch", "**Chelis Surf** (.ch)",
               "Examples, std library, test fixtures", line_comment="--", bold=True),
    CustomLang("*.dp", "**Chelis Deep** (.dp)",
               "Deep test fixtures", line_comment=";;", bold=True),
    CustomLang("*.pest", "PEG Grammars (.pest)",
               "Validator grammars (Surf + Deep)", line_comment="//"),
    CustomLang("*.scm", "Tree-sitter Queries (.scm)",
               "Syntax highlighting for Surf and Deep", line_comment=";;"),
]

# Extensions handled by custom counting — skip these in pygount pass
CUSTOM_EXTENSIONS: set[str] = {".ch", ".dp", ".pest", ".scm"}


def find_repo_root() -> Path:
    """Walk up from cwd to find the directory containing Cargo.toml workspace."""
    p = Path.cwd()
    while p != p.parent:
        if (p / "Cargo.toml").exists() and (p / "crates").is_dir():
            return p
        p = p.parent
    return Path.cwd()


def should_skip(path: Path) -> bool:
    """Return True if path is under a directory we should skip."""
    return any(part in SKIP_DIRS for part in path.parts)


def collect_all_files(repo_root: Path) -> list[Path]:
    """Collect all non-hidden files under repo_root, excluding skip dirs."""
    files = []
    for f in sorted(repo_root.rglob("*")):
        if f.is_file() and not should_skip(f) and not f.name.startswith("."):
            files.append(f)
    return files


def count_with_pygount(files: list[Path]) -> dict[str, LangEntry]:
    """Count lines using pygount for files it recognizes."""
    by_lang: dict[str, LangEntry] = {}

    for f in files:
        if f.suffix in CUSTOM_EXTENSIONS:
            continue

        analysis = SourceAnalysis.from_file(str(f), group="chelis", encoding="utf-8")
        if analysis.language in ("__unknown__", "__binary__", "__empty__", "__error__"):
            continue

        lang = LANGUAGE_RENAMES.get(analysis.language, analysis.language)
        if lang in LANGUAGE_DROP:
            continue
        if lang not in by_lang:
            by_lang[lang] = LangEntry(
                language=lang,
                files=0,
                code=0,
                comments=0,
                blanks=0,
                notes=LANGUAGE_NOTES.get(lang, ""),
            )
        entry = by_lang[lang]
        entry.files += 1
        entry.code += analysis.code_count
        entry.comments += analysis.documentation_count
        entry.blanks += analysis.empty_count

    return by_lang


def count_custom_files(
    repo_root: Path, lang: CustomLang
) -> LangEntry:
    """Count lines for a custom file type using simple line-based heuristics."""
    files = sorted(
        f for f in repo_root.rglob(lang.glob)
        if f.is_file() and not should_skip(f)
    )
    total_code = 0
    total_comments = 0
    total_blanks = 0

    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                total_blanks += 1
            elif lang.line_comment and stripped.startswith(lang.line_comment):
                total_comments += 1
            else:
                total_code += 1

    return LangEntry(
        language=lang.display,
        files=len(files),
        code=total_code,
        comments=total_comments,
        blanks=total_blanks,
        notes=lang.notes,
        bold=lang.bold,
    )


def build_report(repo_root: Path) -> list[LangEntry]:
    """Build the full LOC report: pygount for standard + manual for custom."""
    all_files = collect_all_files(repo_root)
    by_lang = count_with_pygount(all_files)
    entries = list(by_lang.values())

    for custom in CUSTOM_LANGUAGES:
        entry = count_custom_files(repo_root, custom)
        if entry.files > 0:
            entries.append(entry)

    entries.sort(key=lambda e: e.code, reverse=True)
    return entries


def format_number(n: int) -> str:
    """Format integer with comma separators."""
    return f"{n:,}"


def render_markdown(entries: list[LangEntry]) -> str:
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
        if e.bold:
            b = "**"
            lines.append(
                f"| {e.language} | {b}{format_number(e.files)}{b} "
                f"| {b}{format_number(e.code)}{b} | {b}{format_number(e.comments)}{b} "
                f"| {b}{format_number(e.blanks)}{b} | {b}{format_number(e.total_lines)}{b} "
                f"| {b}{e.notes}{b} |"
            )
        else:
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

    md = render_markdown(entries)

    if args.stdout:
        print(md)
        return

    output_path = Path(args.output) if args.output else repo_root / DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
