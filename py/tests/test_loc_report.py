"""Tests for chelis_tools.loc_report."""

from __future__ import annotations

from pathlib import Path

import pytest

from chelis_tools.loc_report import (
    CustomLang,
    DEFAULT_OUTPUT,
    LangEntry,
    count_custom_files,
    format_number,
    render_markdown,
    should_skip,
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


class TestShouldSkip:
    def test_skips_target(self):
        assert should_skip(Path("/repo/target/debug/foo.rs"))

    def test_skips_git(self):
        assert should_skip(Path("/repo/.git/objects/abc"))

    def test_allows_normal(self):
        assert not should_skip(Path("/repo/crates/chelis-cli/src/main.rs"))

    def test_skips_pycache(self):
        assert should_skip(Path("/repo/__pycache__/foo.pyc"))

    def test_skips_claude_worktrees(self):
        assert should_skip(Path("/repo/.claude/worktrees/agent-a/target/debug/foo.rs"))

    def test_skips_codex_worktrees(self):
        assert should_skip(Path("/repo/.codex/worktrees/agent-a/crates/lib.rs"))

    def test_skips_generated_book_output(self):
        assert should_skip(Path("/repo/docs/book/book/index.html"))

    def test_allows_tracked_agent_commands(self):
        assert not should_skip(Path("/repo/.claude/commands/red-team.md"))


def test_default_output_matches_ci_commit_path():
    assert DEFAULT_OUTPUT == "docs/loc_report.md"


class TestCountCustomFiles:
    def test_counts_code_and_comments(self, tmp_path):
        lang = CustomLang("*.ch", "Surf", "test", line_comment="--")
        (tmp_path / "a.ch").write_text("def foo -> Int = 42\n-- comment\n\nlet x = 1\n")
        entry = count_custom_files(tmp_path, lang)
        assert entry.files == 1
        assert entry.code == 2
        assert entry.comments == 1
        assert entry.blanks == 1

    def test_multiple_files(self, tmp_path):
        lang = CustomLang("*.ch", "Surf", "test", line_comment="--")
        (tmp_path / "a.ch").write_text("line1\nline2\n")
        (tmp_path / "b.ch").write_text("line3\n")
        entry = count_custom_files(tmp_path, lang)
        assert entry.files == 2
        assert entry.code == 3

    def test_empty_dir(self, tmp_path):
        lang = CustomLang("*.ch", "Surf", "test", line_comment="--")
        entry = count_custom_files(tmp_path, lang)
        assert entry.files == 0
        assert entry.code == 0

    def test_excludes_target_dir(self, tmp_path):
        lang = CustomLang("*.ch", "Surf", "test", line_comment="--")
        target = tmp_path / "target" / "debug"
        target.mkdir(parents=True)
        (target / "junk.ch").write_text("should not count\n")
        (tmp_path / "real.ch").write_text("should count\n")
        entry = count_custom_files(tmp_path, lang)
        assert entry.files == 1
        assert entry.code == 1


class TestRenderMarkdown:
    def test_produces_valid_table(self):
        entries = [
            LangEntry("Rust", 10, 1000, 50, 100, "Compiler"),
            LangEntry("Python", 2, 200, 5, 30, "Scripts"),
        ]
        md = render_markdown(entries)
        assert "# Lines of Code Report" in md
        assert "| Rust |" in md
        assert "| Python |" in md
        assert "| **Total** |" in md
        assert "**12**" in md  # total files
        assert "**1,200**" in md  # total code

    def test_empty_entries(self):
        md = render_markdown([])
        assert "| **Total** |" in md
        assert "**0**" in md

    def test_table_row_count(self):
        entries = [LangEntry(f"Lang{i}", 1, 10 * i, 0, 0) for i in range(5)]
        md = render_markdown(entries)
        # header + separator + 5 data rows + total row = 8
        table_rows = [line for line in md.splitlines() if line.startswith("|")]
        assert len(table_rows) == 8
