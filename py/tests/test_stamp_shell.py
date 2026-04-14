"""Tests for chelis_tools.stamp_shell."""

from __future__ import annotations

from pathlib import Path

import pytest

from chelis_tools.stamp_shell import stamp_shell


def _make_fake_nautilus(root: Path) -> Path:
    src = root / "nautilus"
    (src / "src").mkdir(parents=True)
    (src / ".github/workflows").mkdir(parents=True)
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    (src / "reef.toml").write_text(
        '[package]\nname = "nautilus"\n'
        'version = "0.0.0"\n'
        'compiler = "=0.1.3"\n'
        'module_prefix = "Nautilus"\n'
    )
    (src / "src/core.ch").write_text(
        "module Nautilus.Core\nexport (version)\ndef version() -> i32 = 0\n"
    )
    (src / "README.md").write_text(
        "# Nautilus\n\nA Chelis shell repo.\n"
    )
    (src / "AGENTS.md").write_text(
        "# Nautilus Agent Contract\n\n"
        "Rotate with `gh secret set --repo Chelis-Lang/nautilus`.\n"
    )
    (src / ".github/workflows/ci.yml").write_text(
        "name: ci\njobs:\n  check:\n    steps:\n      - run: chelis check src/core.ch\n"
    )
    (src / ".gitignore").write_text("target/\n")
    (src / "LICENSE").write_text("MIT License\n")
    return src


class TestStampShell:
    def test_refuses_existing_destination(self, tmp_path: Path):
        src = _make_fake_nautilus(tmp_path)
        dst = tmp_path / "coral"
        dst.mkdir()
        with pytest.raises(FileExistsError):
            stamp_shell(src, dst, "nautilus", "coral")

    def test_refuses_non_shell_source(self, tmp_path: Path):
        src = tmp_path / "not-a-shell"
        src.mkdir()
        dst = tmp_path / "coral"
        with pytest.raises(FileNotFoundError):
            stamp_shell(src, dst, "nautilus", "coral")

    def test_rewrites_name_tokens(self, tmp_path: Path):
        src = _make_fake_nautilus(tmp_path)
        dst = tmp_path / "coral"
        stamp_shell(src, dst, "nautilus", "coral")

        assert 'name = "coral"' in (dst / "reef.toml").read_text()
        assert 'module_prefix = "Coral"' in (dst / "reef.toml").read_text()
        assert "module Coral.Core" in (dst / "src/core.ch").read_text()
        assert "# Coral\n" in (dst / "README.md").read_text()
        assert "# Coral Agent Contract" in (dst / "AGENTS.md").read_text()
        assert "Chelis-Lang/coral" in (dst / "AGENTS.md").read_text()

    def test_skips_git_directory(self, tmp_path: Path):
        src = _make_fake_nautilus(tmp_path)
        dst = tmp_path / "coral"
        stamp_shell(src, dst, "nautilus", "coral")
        assert not (dst / ".git").exists()

    def test_preserves_untouched_files_verbatim(self, tmp_path: Path):
        src = _make_fake_nautilus(tmp_path)
        dst = tmp_path / "coral"
        stamp_shell(src, dst, "nautilus", "coral")
        assert (dst / ".gitignore").read_text() == "target/\n"
        assert (dst / "LICENSE").read_text() == "MIT License\n"
        ci = (dst / ".github/workflows/ci.yml").read_text()
        assert "nautilus" not in ci.lower()
        assert "chelis check src/core.ch" in ci

    def test_returns_rewritten_paths(self, tmp_path: Path):
        src = _make_fake_nautilus(tmp_path)
        dst = tmp_path / "coral"
        rewritten = stamp_shell(src, dst, "nautilus", "coral")
        rewritten_names = {p.name for p in rewritten}
        assert rewritten_names == {"reef.toml", "core.ch", "README.md", "AGENTS.md"}

    def test_excludes_phase_spec_files(self, tmp_path: Path):
        src = _make_fake_nautilus(tmp_path)
        (src / "spec").mkdir()
        (src / "spec" / "phase3j.md").write_text("# Phase 3j — Nautilus\nnautilus body\n")
        (src / "spec" / "shared_notes.md").write_text("not a phase file\n")
        dst = tmp_path / "coral"
        stamp_shell(src, dst, "nautilus", "coral")

        # phase3j.md must not be copied — each shell owns its own phase spec
        assert not (dst / "spec" / "phase3j.md").exists()
        # non-phase files in spec/ are still copied verbatim
        assert (dst / "spec" / "shared_notes.md").read_text() == "not a phase file\n"
