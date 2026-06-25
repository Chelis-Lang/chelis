#!/usr/bin/env python3
"""Tests for scripts/setup_proof_toolchain.py and the Option-B proof scripts.

Run with: .venv/bin/python -m pytest scripts/test_setup_proof_toolchain.py

These cover the pure logic (dep probing, the execute.h patch idempotence, the
manifest/envelope consistency) without rebuilding the C toolchain, which is the
slow part exercised by CI's setup job, not the unit suite.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


setup = _load("setup_proof_toolchain")


def test_root_deps_table_is_complete():
    # The three root-requiring base deps the escalation named.
    assert set(setup.ROOT_DEPS) == {"mpfi", "libxml2", "gappa"}


def test_install_hints_cover_fedora_and_debian():
    assert "dnf" in setup.INSTALL_HINTS["fedora"]
    assert "apt-get" in setup.INSTALL_HINTS["debian"]
    # both name the three packages
    for hint in setup.INSTALL_HINTS.values():
        assert "mpfi" in hint and "gappa" in hint


def test_execute_h_patch_is_idempotent(tmp_path):
    src = tmp_path
    h = src / "execute.h"
    h.write_text("...\nextern int miniyyparse();\n...\n")
    setup._patch_sollya_execute_h(src)
    once = h.read_text()
    assert "miniyyparse(void *myScanner)" in once
    assert "miniyyparse();" not in once
    # re-applying does nothing
    setup._patch_sollya_execute_h(src)
    assert h.read_text() == once


def test_pinned_source_checksums_are_present():
    # The tarball pins must be real sha256 hex (64 chars), not placeholders.
    for sha in (setup.FPLLL_SHA256, setup.SOLLYA_SHA256):
        assert len(sha) == 64
        int(sha, 16)  # valid hex


def test_committed_manifest_matches_committed_envelope():
    # The committed Gappa manifest's central eps + coeffs + tail eps must equal
    # the committed envelope (the proof term and the consumed bound do not
    # drift). Mirrors the Rust consistency tests, guarding the data files
    # directly so a hand-edit is caught in the Python suite too.
    data = REPO_ROOT / "crates" / "chelis-prove" / "data"
    env = json.loads((data / "erf_envelope.json").read_text())
    manifest = json.loads((data / "erf_proof" / "manifest.json").read_text())
    central = [b for b in env["boxes"] if b["arm"]["kind"] == "central"][0]
    assert central["proof_kind"] == "gappa"
    assert float(central["eps"]) == float(manifest["central_eps"])
    assert central["arm"]["coeffs"] == [float(c) for c in manifest["coeffs"]]
    # All three arms are Gappa-proved now (tails too).
    sats = [b for b in env["boxes"] if b["arm"]["kind"] == "saturation"]
    assert sats and all(b["proof_kind"] == "gappa" for b in sats)
    upper = [b for b in sats if b["arm"]["value"] == 1.0][0]
    lower = [b for b in sats if b["arm"]["value"] == -1.0][0]
    assert float(upper["eps"]) == float(manifest["tail_pos_eps"])
    assert float(lower["eps"]) == float(manifest["tail_neg_eps"])


def test_envelope_provenance_pins_the_gappa_bundle():
    data = REPO_ROOT / "crates" / "chelis-prove" / "data"
    env = json.loads((data / "erf_envelope.json").read_text())
    manifest = json.loads((data / "erf_proof" / "manifest.json").read_text())
    assert (
        env["provenance"]["gappa_bundle_sha256"] == manifest["bundle_sha256"]
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
