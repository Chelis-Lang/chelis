"""Tests for loc_report.py."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts/ to path so we can import the module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loc_report import (
    LangEntry,
    count_files,
    format_number,
    parse_tokei,
    render_markdown,
)


class TestLangEntry:
    def test_total_lines(self):
        e = LangEntry("Rust", files=10, code=100, comments=20, blanks=30)
        assert e.total_lines == 150

    def test_total_lines_zero(self):
        e = LangEntry("Empty", files=0, code=0, comments=0, blanks=0)
        assert e.total_lines == 0


class TestFormatNumber:
    def test_small(self):
        assert format_number(42) == "42"

    def test_thousands(self):
        assert format_number(1_234) == "1,234"

    def test_large(self):
        assert format_number(68_374) == "68,374"

    def test_zero(self):
        assert format_number(0) == "0"


class TestParseTokei:
    def test_extracts_known_languages(self):
        data = {
            "Rust": {
                "code": 68000,
                "comments": 1300,
                "blanks": 5000,
                "reports": [{"name": f"file{i}.rs"} for i in range(100)],
            },
            "Python": {
                "code": 1600,
                "comments": 10,
                "blanks": 300,
                "reports": [{"name": "a.py"}, {"name": "b.py"}],
            },
        }
        entries = parse_tokei(data)
        assert len(entries) == 2
        rust = next(e for e in entries if e.language == "Rust")
        assert rust.code == 68000
        assert rust.files == 100

    def test_skips_unknown_languages(self):
        data = {
            "Haskell": {
                "code": 50,
                "comments": 5,
                "blanks": 10,
                "reports": [{"name": "x.hs"}],
            }
        }
        entries = parse_tokei(data)
        assert len(entries) == 0

    def test_empty_input(self):
        assert parse_tokei({}) == []


class TestCountFiles:
    def test_counts_lines(self, tmp_path):
        (tmp_path / "a.ch").write_text("def foo -> Int = 42\n\ndef bar -> Int = 0\n")
        (tmp_path / "b.ch").write_text("let x = 1\n")
        entry = count_files(tmp_path, "**/*.ch")
        assert entry.files == 2
        assert entry.code == 3  # 3 non-blank lines
        assert entry.blanks == 1  # 1 blank line
        assert entry.total_lines == 4

    def test_empty_dir(self, tmp_path):
        entry = count_files(tmp_path, "**/*.ch")
        assert entry.files == 0
        assert entry.code == 0

    def test_excludes_target_dir(self, tmp_path):
        target = tmp_path / "target" / "debug"
        target.mkdir(parents=True)
        (target / "junk.ch").write_text("should not count\n")
        (tmp_path / "real.ch").write_text("should count\n")
        entry = count_files(tmp_path, "**/*.ch")
        assert entry.files == 1
        assert entry.code == 1


class TestRenderMarkdown:
    def test_produces_valid_table(self):
        entries = [
            LangEntry("Rust", 10, 1000, 50, 100, "Compiler"),
            LangEntry("Python", 2, 200, 5, 30, "Scripts"),
        ]
        md = render_markdown(entries, Path("/fake"))
        assert "# Lines of Code Report" in md
        assert "| Rust |" in md
        assert "| Python |" in md
        assert "| **Total** |" in md
        # Check totals
        assert "**12**" in md  # total files
        assert "**1,200**" in md  # total code

    def test_empty_entries(self):
        md = render_markdown([], Path("/fake"))
        assert "| **Total** |" in md
        assert "**0**" in md

    def test_table_row_count(self):
        entries = [LangEntry(f"Lang{i}", 1, 10 * i, 0, 0) for i in range(5)]
        md = render_markdown(entries, Path("/fake"))
        # header + separator + 5 data rows + total row = 8
        table_rows = [l for l in md.splitlines() if l.startswith("|")]
        assert len(table_rows) == 8
