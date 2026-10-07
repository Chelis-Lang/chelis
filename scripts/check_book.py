#!/usr/bin/env python3
"""Lint user-facing shell books against the writing standard (ADR 0013).

Canonical copy: Chelis-Lang/website red-team/gates/book_lint.py. Each shell
repo vendors it as scripts/check_book.py; update it here and copy it out.

A book page teaches a reader who writes Chelis code against the shell. It
never carries contributor material: issue numbers, repo-internal paths,
maintainer commands, process talk, status words or "see the source" in place
of documentation. Fenced code is exempt; inline code is checked, so a page
cannot hide `spec/05-types.md` in backticks.

Usage: book_lint.py [PATH ...]   (default: the chelis.ch docs of every shell
                                  in apps/chelis-site/src/data/books.json)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RULES: dict[str, re.Pattern[str]] = {
    "em-dash": re.compile(r"—|&mdash;|&#8212;|&#x2014;", re.I),
    "load-bearing": re.compile(r"\bload[- ]bearing\b", re.I),
    "issue or PR reference": re.compile(
        r"(?<![\w&/])[A-Za-z][\w.-]*#\d+\b|(?<![\w&/#])#\d{2,}\b|/(?:issues|pull)/\d+", re.I),
    "repo-internal path": re.compile(
        r"(?<![\w.-])(?:spec|crates|scripts|tools|xtask|\.github)/[\w./-]*"
        r"|(?<![\w.-])docs/[\w./-]+\.md\b"
        r"|\b(?:AGENTS|CLAUDE|CONTRIBUTING|UPSTREAM_BUGS|CHELIS_SURFACE)(?:\.md)?\b"
        r"|\bissue_drafts\b"),
    "spec section citation": re.compile(r"\bspec\s+0\d\b", re.I),
    "see the source": re.compile(
        r"\b(?:see|read|check|consult|refer to|look at)\s+(?:the\s+)?(?:source|implementation|`?src/?`?)(?:\s+(?:code|tree|for))?\b"
        r"|\bin the source tree\b", re.I),
    "status wording": re.compile(
        r"\b(?:planned|coming soon|not yet (?:implemented|supported|available|wired|ported)|roadmap|future work"
        r"|TODO|TBD|WIP|phase\s+\d+[a-z]?|milestones?|(?-i:M\d)\b|deferr(?:ed|al)"
        r"|work in progress|experimental)\b", re.I),
    "process talk": re.compile(
        r"\b(?:red[- ]team(?:ed|ing)?|upstream (?:bugs?|gaps?|issues?)|import surface|manual gates?"
        r"|release gates?|CI gates?|contributors? (?:should|must)|maintainers? (?:should|must|run))\b", re.I),
    "unpublished name": re.compile(
        r"(?<![A-Za-z])(?:whale|school|beacon|hydronnx|betting|voyage|sonar|hull|calcify|octant|cove|compass|burn-?in)(?![A-Za-z])(?!-White)",
        re.I),
}

# The language specifications are published documents: a page may link them as
# the complete rules, beside the guide that teaches them.
ALLOWED = re.compile(r"https://github\.com/Chelis-Lang/chelis/blob/main/spec/0\d-[\w-]+\.md")
FENCE = re.compile(r"^\s*(```|~~~)")
FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
COMMENT = re.compile(r"<!--.*?-->", re.S)


def prose_lines(text: str) -> list[tuple[int, str]]:
    """Lines outside fenced code, numbered as in the file."""
    offset = 0
    match = FRONTMATTER.match(text)
    if match:
        offset = match.group(0).count("\n")
        text = text[match.end():]
    text = COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    out, fence = [], None
    for number, line in enumerate(text.splitlines(), 1 + offset):
        m = FENCE.match(line)
        if m:
            fence = None if fence == m.group(1) else (fence or m.group(1))
            continue
        if fence is None:
            out.append((number, line))
    return out


# A wrapped paragraph line that starts with "1." and a space
# renders as a new list item and cuts the sentence in two.
WRAPPED_MARKER = re.compile(r"^\d+\.\s")
LIST_LINE = re.compile(r"^\s*(?:\d+\.|[-+*])\s|^\s*[|#>]|^\s{2,}")


def lint_text(text: str) -> list[tuple[int, str, str]]:
    found = []
    previous = ""
    for number, line in prose_lines(text):
        if WRAPPED_MARKER.match(line) and previous.strip() and not LIST_LINE.match(previous) \
                and not previous.rstrip().endswith(":"):
            found.append((number, "wrapped line starts a list", line[:20]))
        previous = line
        # Link targets are checked with their text: a link into src/ leaks too.
        line = ALLOWED.sub("", line)
        for label, pattern in RULES.items():
            m = pattern.search(line)
            if m:
                found.append((number, label, m.group(0)))
    return found


def lint_paths(paths: list[Path]) -> list[str]:
    files = sorted({f for p in paths for f in ([p] if p.is_file() else p.rglob("*"))
                    if f.is_file() and f.suffix in {".md", ".mdx"}})
    if not files:
        return ["book-lint: no pages found"]
    return [f"{f}:{n}: {label}: {hit}" for f in files for n, label, hit in lint_text(f.read_text())]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths
    if not paths:
        import json
        app = Path(__file__).resolve().parents[2] / "apps/chelis-site/src"
        books = json.loads((app / "data/books.json").read_text())["shells"].values()
        paths = [app / "content/docs" / b["siteDir"] for b in books]
    errors = lint_paths(paths)
    for error in errors:
        print(error)
    print(f"book-lint: {len(errors)} findings")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
