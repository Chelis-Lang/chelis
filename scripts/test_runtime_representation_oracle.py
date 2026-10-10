"""Unit contracts for the chelis#893 Phase 0 acceptance oracle.

These cover the parts of the oracle that are not proved by running it: that the
scanned universe is every file under the frozen roots, that the freeze digest
binds the finished foundation, the roots, and the reviewed mutation contracts, that the runtime manifest
exactly describes the configuration the oracle executes, and that every
classifier has a controlled mutation.

The seam classification rules themselves are proved in the scanner's own suite
(`cargo nextest run -p chelis-repr-inventory --test inventory`), which is where
they belong: they are Rust and C parsing decisions, not Python ones.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import runtime_representation_oracle as oracle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def moved(source: Path, destination: Path) -> Iterator[None]:
    """Move a tracked file or directory away and restore it exactly."""

    assert source.exists() and not destination.exists()
    created = [parent for parent in reversed(destination.parents) if not parent.exists()]
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(source, destination)
    oracle._invalidate_inventory_cache()
    try:
        yield
    finally:
        os.rename(destination, source)
        for directory in reversed(created):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        oracle._invalidate_inventory_cache()


class SourceUniverseTests(unittest.TestCase):
    """The universe is every file on disk under the frozen roots."""

    def test_a_debt_file_leaving_the_roots_fails_until_retired(self) -> None:
        source = "crates/chelis-ir/src/ownership/verify.rs"
        baseline = oracle.load_baseline()
        self.assertTrue(
            any(f"|path={source}|" in row["identity"] for row in baseline["active_debt"])
        )
        with moved(
            REPO_ROOT / source,
            REPO_ROOT / "crates/chelis-departed-probe/src/verify.rs",
        ), tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(oracle.OracleFailure) as caught:
                oracle.validate_baseline(baseline, oracle.inventory_rows(REPO_ROOT))
            self.assertEqual(caught.exception.code, oracle.DEPARTED_FILE_FAILURE.code)
            self.assertEqual(
                str(caught.exception),
                f"{oracle.DEPARTED_FILE_FAILURE.reason_prefix}: {source}; "
                f"{oracle.DEPARTED_FILE_ACTION}",
            )
            # Regeneration may not drop that debt silently; naming the file
            # is the sanctioned retirement, and it leaves the digest alone.
            copy = Path(directory) / "baseline.json"
            copy.write_text(json.dumps(baseline), encoding="utf-8")
            with mock.patch.object(oracle, "BASELINE_PATH", copy):
                with self.assertRaises(oracle.OracleFailure) as refused:
                    oracle.regenerate()
                self.assertEqual(refused.exception.code, oracle.DEPARTED_FILE_FAILURE.code)
                self.assertEqual(json.loads(copy.read_text()), baseline)
                oracle.regenerate(retired_files=(source,))
                regenerated = json.loads(copy.read_text())
        self.assertFalse(
            any(f"|path={source}|" in row["identity"] for row in regenerated["active_debt"])
        )
        self.assertIn(source, regenerated["retired_files"])
        self.assertEqual(regenerated["foundation_rows"], baseline["foundation_rows"])
        self.assertEqual(regenerated["freeze_sha256"], baseline["freeze_sha256"])

    def test_retiring_a_file_still_in_the_universe_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "baseline.json"
            copy.write_text(oracle.BASELINE_PATH.read_text(), encoding="utf-8")
            with mock.patch.object(oracle, "BASELINE_PATH", copy):
                with self.assertRaisesRegex(
                    oracle.OracleFailure,
                    "only a departed referenced file can be retired: "
                    "crates/chelis-ir/src/dag.rs",
                ):
                    oracle.regenerate(retired_files=("crates/chelis-ir/src/dag.rs",))

    def test_a_crate_leaving_a_glob_root_fails_as_departed(self) -> None:
        baseline = oracle.load_baseline()
        rows = oracle.inventory_rows(REPO_ROOT)
        with moved(
            REPO_ROOT / "crates/chelis-backend-metal",
            REPO_ROOT / "crates/chelis-departed-metal",
        ):
            # The other backends still satisfy `crates/chelis-backend-*`.
            self.assertTrue(any(REPO_ROOT.glob("crates/chelis-backend-*/src")))
            with self.assertRaises(oracle.OracleFailure) as caught:
                oracle.validate_baseline(baseline, rows)
        self.assertEqual(caught.exception.code, oracle.DEPARTED_ROOT_FAILURE.code)
        self.assertEqual(
            str(caught.exception),
            f"{oracle.DEPARTED_ROOT_FAILURE.reason_prefix}: "
            "crates/chelis-backend-metal, crates/chelis-backend-metal/runtime, "
            f"crates/chelis-backend-metal/src; {oracle.ROOT_CHANGE_ACTION}",
        )

    def test_an_untracked_or_ignored_directory_is_not_a_root_directory(self) -> None:
        leftover = REPO_ROOT / "crates/chelis-backend-leftover-probe"
        ignored = REPO_ROOT / "crates/chelis-backend-ignored-probe"
        self.assertFalse(leftover.exists() or ignored.exists())
        baseline = oracle.load_baseline()
        rows = oracle.inventory_rows(REPO_ROOT)
        try:
            (leftover / "src").mkdir(parents=True)
            (ignored / "src").mkdir(parents=True)
            (ignored / ".gitignore").write_text("*\n", encoding="utf-8")
            (ignored / "src/lib.rs").write_text("pub fn ignored() {}\n", encoding="utf-8")
            self.assertEqual(oracle.root_directories(REPO_ROOT), baseline["source_inventory"]["roots"])
            self.assertNotIn(
                "crates/chelis-backend-ignored-probe/src/lib.rs",
                oracle.inventory_sources(REPO_ROOT),
            )
            oracle.validate_baseline(baseline, rows)
        finally:
            shutil.rmtree(leftover, ignore_errors=True)
            shutil.rmtree(ignored, ignore_errors=True)
            oracle._invalidate_inventory_cache()

    def test_an_untracked_crate_with_sources_is_a_new_root_directory(self) -> None:
        # The scan sees an untracked file, so the directory rule must too.
        crate = REPO_ROOT / "crates/chelis-backend-untracked-probe"
        self.assertFalse(crate.exists())
        baseline = oracle.load_baseline()
        rows = oracle.inventory_rows(REPO_ROOT)
        try:
            (crate / "src").mkdir(parents=True)
            (crate / "src/lib.rs").write_text("pub fn untracked() {}\n", encoding="utf-8")
            oracle._invalidate_inventory_cache()
            self.assertIn(
                "crates/chelis-backend-untracked-probe/src/lib.rs",
                oracle.inventory_sources(REPO_ROOT),
            )
            with self.assertRaises(oracle.OracleFailure) as caught:
                oracle.validate_baseline(baseline, rows)
        finally:
            shutil.rmtree(crate, ignore_errors=True)
            oracle._invalidate_inventory_cache()
        self.assertEqual(
            str(caught.exception),
            "inventory root directories do not match the frozen source universe; "
            "a new directory under a root is a freeze move: "
            "crates/chelis-backend-untracked-probe, "
            f"crates/chelis-backend-untracked-probe/src; {oracle.ROOT_CHANGE_ACTION}",
        )

    def test_a_frozen_directory_untracked_in_the_index_stays_present(self) -> None:
        # Its files stay on disk and visible, so they are still scanned and
        # the directory has not departed. The index edit happens in a
        # throwaway copy named by GIT_INDEX_FILE, never the live index.
        directory = "crates/chelis-backend-metal"
        baseline = oracle.load_baseline()
        rows = oracle.inventory_rows(REPO_ROOT)
        sources = oracle.inventory_sources(REPO_ROOT)
        live_index = Path(
            subprocess.run(
                ("git", "rev-parse", "--path-format=absolute", "--git-path", "index"),
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        live_bytes = live_index.read_bytes()
        with tempfile.TemporaryDirectory() as scratch:
            index = Path(scratch) / "index"
            shutil.copyfile(live_index, index)
            with mock.patch.dict(os.environ, {"GIT_INDEX_FILE": str(index)}):
                subprocess.run(
                    ("git", "rm", "-r", "-q", "--cached", directory),
                    cwd=REPO_ROOT,
                    check=True,
                    capture_output=True,
                )
                self.assertNotIn(
                    f"{directory}/src/lib.rs",
                    subprocess.run(
                        ("git", "ls-files", "--cached", "--", directory),
                        cwd=REPO_ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout,
                )
                oracle._invalidate_inventory_cache()
                try:
                    self.assertEqual(
                        oracle.root_directories(REPO_ROOT),
                        baseline["source_inventory"]["roots"],
                    )
                    self.assertEqual(oracle.inventory_sources(REPO_ROOT), sources)
                    oracle.validate_baseline(baseline, rows)
                finally:
                    oracle._invalidate_inventory_cache()
        self.assertEqual(live_index.read_bytes(), live_bytes)

    def test_a_frozen_directory_with_its_files_deleted_departs(self) -> None:
        # The index still holds the file; only the disk lost it.
        source = "crates/chelis-backend-metal/runtime/chelis_metal_runtime.h"
        baseline = oracle.load_baseline()
        rows = oracle.inventory_rows(REPO_ROOT)
        self.assertEqual(
            [path.name for path in (REPO_ROOT / source).parent.iterdir()],
            ["chelis_metal_runtime.h"],
        )
        with moved(REPO_ROOT / source, REPO_ROOT / "crates/chelis-departed-probe/runtime.h"):
            self.assertTrue((REPO_ROOT / source).parent.is_dir())
            with self.assertRaises(oracle.OracleFailure) as caught:
                oracle.validate_baseline(baseline, rows)
        self.assertEqual(caught.exception.code, oracle.DEPARTED_ROOT_FAILURE.code)
        self.assertEqual(
            str(caught.exception),
            f"{oracle.DEPARTED_ROOT_FAILURE.reason_prefix}: "
            f"crates/chelis-backend-metal/runtime; {oracle.ROOT_CHANGE_ACTION}",
        )

    def test_regeneration_drops_a_retired_file_that_returned(self) -> None:
        source = "crates/chelis-backend-c/src/random_observer.rs"
        baseline = oracle.load_baseline()
        self.assertEqual(baseline["retired_files"], [source])
        with oracle.temporary_mutation(
            REPO_ROOT / source, lambda _: "pub fn returned() {}\n", creates_file=True
        ), tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "baseline.json"
            copy.write_text(json.dumps(baseline), encoding="utf-8")
            with mock.patch.object(oracle, "BASELINE_PATH", copy):
                oracle.regenerate()
                regenerated = oracle.load_baseline()
            self.assertEqual(regenerated["retired_files"], [])
            oracle.validate_baseline(regenerated, oracle.inventory_rows(REPO_ROOT))

    def test_a_symlinked_crate_directory_is_outside_the_universe(self) -> None:
        # A link to a directory is never traversed, at any depth, and is not a
        # root directory; the oracle reads past it rather than failing.
        link = REPO_ROOT / "crates/chelis-backend-linkprobe"
        self.assertFalse(link.exists() or link.is_symlink())
        baseline = oracle.load_baseline()
        rows = oracle.inventory_rows(REPO_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "src").mkdir()
            (Path(directory) / "src/lib.rs").write_text(
                oracle.mutate_unregistered_inventory_source(""), encoding="utf-8"
            )
            link.symlink_to(directory, target_is_directory=True)
            oracle._invalidate_inventory_cache()
            try:
                sources = oracle.inventory_sources(REPO_ROOT)
                directories = oracle.root_directories(REPO_ROOT)
                oracle.validate_baseline(baseline, rows)
            finally:
                link.unlink()
                oracle._invalidate_inventory_cache()
        self.assertFalse([path for path in sources if "linkprobe" in path])
        self.assertEqual(directories, baseline["source_inventory"]["roots"])

    def test_a_nested_repository_is_outside_files_and_directories_alike(self) -> None:
        nested = REPO_ROOT / "crates/chelis-backend-nestedprobe"
        self.assertFalse(nested.exists())
        baseline = oracle.load_baseline()
        try:
            (nested / "src").mkdir(parents=True)
            (nested / "src/lib.rs").write_text("pub fn nested() {}\n", encoding="utf-8")
            subprocess.run(("git", "init", "-q", str(nested)), check=True, capture_output=True)
            oracle._invalidate_inventory_cache()
            sources = oracle.inventory_sources(REPO_ROOT)
            directories = oracle.root_directories(REPO_ROOT)
        finally:
            shutil.rmtree(nested, ignore_errors=True)
            oracle._invalidate_inventory_cache()
        self.assertFalse([path for path in sources if "nestedprobe" in path])
        self.assertEqual(directories, baseline["source_inventory"]["roots"])

    def test_ignored_names_are_excluded_whatever_their_spelling(self) -> None:
        # NUL-delimited git I/O: no quoting, whatever core.quotePath says.
        crate = REPO_ROOT / "crates/chelis-backend-\u00fcnicodeprobe"
        local = REPO_ROOT / "crates/chelis-ir/src/runtime_representation_ignored_names"
        self.assertFalse(crate.exists() or local.exists())
        baseline = oracle.load_baseline()
        try:
            (crate / "src").mkdir(parents=True)
            (crate / ".gitignore").write_text("*\n", encoding="utf-8")
            (crate / "src/\u00e9.rs").write_text("pub fn e() {}\n", encoding="utf-8")
            local.mkdir()
            (local / ".gitignore").write_text("*\n", encoding="utf-8")
            for name in ("\u00e9.rs", 'q"t.rs', "a b.rs", "plain.rs"):
                (local / name).write_text("pub fn f() {}\n", encoding="utf-8")
            oracle._invalidate_inventory_cache()
            sources = oracle.inventory_sources(REPO_ROOT)
            directories = oracle.root_directories(REPO_ROOT)
        finally:
            shutil.rmtree(crate, ignore_errors=True)
            shutil.rmtree(local, ignore_errors=True)
            oracle._invalidate_inventory_cache()
        self.assertFalse(
            [path for path in sources if "probe" in path or "ignored_names" in path]
        )
        self.assertEqual(directories, baseline["source_inventory"]["roots"])

    def test_visible_names_are_scanned_whatever_their_spelling(self) -> None:
        local = REPO_ROOT / "crates/chelis-ir/src/runtime_representation_visible_names"
        self.assertFalse(local.exists())
        names = ("\u00e9.rs", 'q"t.rs', "tab\tname.rs", "back\\slash.rs", "new\nline.rs")
        expected = {
            f"crates/chelis-ir/src/runtime_representation_visible_names/{name}"
            for name in names
        }
        try:
            local.mkdir()
            for name in names:
                (local / name).write_text("pub fn visible() {}\n", encoding="utf-8")
            oracle._invalidate_inventory_cache()
            with mock.patch.dict(
                os.environ,
                {
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "core.quotePath",
                    "GIT_CONFIG_VALUE_0": "true",
                },
            ):
                self.assertLessEqual(expected, set(oracle.inventory_sources(REPO_ROOT)))
        finally:
            shutil.rmtree(local, ignore_errors=True)
            oracle._invalidate_inventory_cache()

    def test_regeneration_prunes_unreferenced_retired_files(self) -> None:
        stale = "crates/chelis-ir/src/runtime_representation_unreferenced_probe.rs"
        baseline = oracle.load_baseline()
        self.assertNotIn(stale, oracle.referenced_files(baseline))
        recorded = json.loads(json.dumps(baseline))
        recorded["retired_files"] = sorted([*recorded["retired_files"], stale])
        with self.assertRaisesRegex(
            oracle.OracleFailure,
            f"retired_files names a file nothing references: {stale}",
        ):
            oracle.validate_baseline(recorded, oracle.inventory_rows(REPO_ROOT))
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "baseline.json"
            copy.write_text(json.dumps(recorded), encoding="utf-8")
            with mock.patch.object(oracle, "BASELINE_PATH", copy):
                oracle.regenerate()
            regenerated = json.loads(copy.read_text())
        self.assertEqual(regenerated["retired_files"], baseline["retired_files"])

    def test_a_final_form_file_leaving_the_roots_fails(self) -> None:
        source = "crates/chelis-backend-c/src/fp_env.rs"
        baseline = oracle.load_baseline()
        self.assertIn(source, oracle._owner_module_final_forms_manifest())
        self.assertFalse(
            any(f"|path={source}|" in row["identity"] for row in baseline["active_debt"])
        )
        with moved(REPO_ROOT / source, REPO_ROOT / "crates/chelis-departed-probe/fp_env.rs"):
            with self.assertRaises(oracle.OracleFailure) as caught:
                oracle.validate_baseline(baseline, oracle.inventory_rows(REPO_ROOT))
        self.assertEqual(caught.exception.code, oracle.DEPARTED_FILE_FAILURE.code)
        self.assertEqual(
            str(caught.exception),
            f"{oracle.DEPARTED_FILE_FAILURE.reason_prefix}: {source}; "
            f"{oracle.DEPARTED_FILE_ACTION}",
        )

    def test_a_retired_identity_file_leaving_the_roots_fails(self) -> None:
        # eval.rs holds only retired foundation identities; the reappearance
        # check needs it scanned.
        source = "crates/chelis-ir/src/eval.rs"
        baseline = oracle.load_baseline()
        self.assertTrue(
            any(f"|path={source}|" in row["identity"] for row in baseline["foundation_rows"])
        )
        self.assertFalse(
            any(f"|path={source}|" in row["identity"] for row in baseline["active_debt"])
        )
        with moved(REPO_ROOT / source, REPO_ROOT / "crates/chelis-departed-probe/eval.rs"):
            with self.assertRaises(oracle.OracleFailure) as caught:
                oracle.validate_baseline(baseline, oracle.inventory_rows(REPO_ROOT))
        self.assertEqual(caught.exception.code, oracle.DEPARTED_FILE_FAILURE.code)
        self.assertEqual(
            str(caught.exception),
            f"{oracle.DEPARTED_FILE_FAILURE.reason_prefix}: {source}; "
            f"{oracle.DEPARTED_FILE_ACTION}",
        )

    def test_a_recorded_departure_stays_departed(self) -> None:
        # random_observer.rs was deleted; its retired identities stay in the
        # foundation, and the baseline records the departure.
        source = "crates/chelis-backend-c/src/random_observer.rs"
        baseline = oracle.load_baseline()
        rows = oracle.inventory_rows(REPO_ROOT)
        self.assertEqual(baseline["retired_files"], [source])
        unrecorded = json.loads(json.dumps(baseline))
        unrecorded["retired_files"] = []
        with self.assertRaises(oracle.OracleFailure) as caught:
            oracle.validate_baseline(unrecorded, rows)
        self.assertEqual(caught.exception.code, oracle.DEPARTED_FILE_FAILURE.code)
        self.assertIn(source, str(caught.exception))
        with oracle.temporary_mutation(
            REPO_ROOT / source, lambda _: "pub fn returned() {}\n", creates_file=True
        ):
            with self.assertRaisesRegex(
                oracle.OracleFailure,
                f"a retired file is back in the inventory universe: {source}",
            ):
                oracle.validate_baseline(baseline, oracle.inventory_rows(REPO_ROOT))

    def test_a_missing_mutation_target_fails_rather_than_being_planted(self) -> None:
        target = "crates/chelis-ir/src/dag.rs"
        probes = [
            probe for probe in oracle.phase0_mutation_probes()
            if probe.path.as_posix() == target
        ]
        self.assertTrue(probes)
        baseline = oracle.load_baseline()
        rows = oracle.inventory_rows(REPO_ROOT)
        with moved(REPO_ROOT / target, REPO_ROOT / "crates/chelis-departed-probe/dag.rs"):
            for probe in probes:
                with self.subTest(witness=probe.witness_id):
                    with self.assertRaisesRegex(
                        oracle.OracleFailure,
                        f"mutation target does not exist: {target}",
                    ):
                        with oracle.temporary_mutation(REPO_ROOT / probe.path, probe.mutate):
                            pass
                    self.assertFalse((REPO_ROOT / target).exists())
            with self.assertRaises(oracle.OracleFailure) as caught:
                oracle.validate_baseline(baseline, rows)
        self.assertEqual(caught.exception.code, oracle.DEPARTED_FILE_FAILURE.code)
        self.assertEqual(
            str(caught.exception),
            f"{oracle.DEPARTED_FILE_FAILURE.reason_prefix}: {target}; "
            f"{oracle.DEPARTED_FILE_ACTION}",
        )

    def test_a_symlinked_directory_under_a_root_is_not_traversed(self) -> None:
        link = REPO_ROOT / "crates/chelis-ir/src/runtime_representation_linkdir_probe"
        self.assertFalse(link.exists() or link.is_symlink())
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "mod.rs").write_text(
                oracle.mutate_unregistered_inventory_source(""), encoding="utf-8"
            )
            link.symlink_to(directory, target_is_directory=True)
            oracle._invalidate_inventory_cache()
            try:
                self.assertTrue((link / "mod.rs").is_file())
                sources = oracle.inventory_sources(REPO_ROOT)
            finally:
                link.unlink()
                oracle._invalidate_inventory_cache()
        self.assertFalse(
            [source for source in sources if "runtime_representation_linkdir_probe" in source]
        )

    def test_the_universe_is_every_file_under_the_frozen_roots(self) -> None:
        sources = oracle.inventory_sources(REPO_ROOT)
        self.assertEqual(sources, oracle._inventory_candidates(REPO_ROOT))
        frozen = oracle.load_baseline()["source_inventory"]["roots"]
        self.assertEqual(frozen, oracle.root_directories(REPO_ROOT))
        self.assertEqual([row["pattern"] for row in frozen], list(oracle.INVENTORY_ROOTS))
        real_run = subprocess.run
        scanned: list[str] = []

        def run(arguments, **kwargs):
            if arguments[0] != "scanner":
                return real_run(arguments, **kwargs)
            scanned.extend(json.loads(kwargs["input"]))
            return subprocess.CompletedProcess(arguments, 0, '{"rows": []}', "")

        with (
            mock.patch.object(oracle, "_build_scanner", return_value=Path("scanner")),
            mock.patch.object(oracle.subprocess, "run", side_effect=run),
        ):
            oracle.scan_sources(REPO_ROOT)
        self.assertEqual(scanned, list(sources))

    def test_every_registered_source_exists_exactly_once(self) -> None:
        sources = oracle.inventory_sources(REPO_ROOT)
        self.assertEqual(len(sources), len(set(sources)), "duplicate scanned source")
        for relative in sources:
            self.assertTrue(
                (REPO_ROOT / relative).is_file(), f"scanned source missing: {relative}"
            )

    def test_a_new_file_under_a_root_is_scanned_without_registration(self) -> None:
        relative = "crates/chelis-ir/src/runtime_representation_new_source_probe.rs"
        baseline = oracle.load_baseline()
        with oracle.temporary_mutation(
            REPO_ROOT / relative,
            lambda _source: "//! A seam-free new source.\n\npub fn new_source_probe() {}\n",
            creates_file=True,
        ):
            self.assertIn(relative, oracle.inventory_sources(REPO_ROOT))
            oracle.validate_baseline(baseline, oracle.inventory_rows(REPO_ROOT))
        self.assertFalse((REPO_ROOT / relative).exists())

    def test_a_seam_in_a_new_file_fails_as_unclassified(self) -> None:
        probes = {probe.witness_id: probe for probe in oracle.phase0_mutation_probes()}
        for witness, kind, owner in (
            (
                "phase0.mutate_unregistered_inventory_source",
                "raw-element-pointer",
                "runtime_representation_phase0_unregistered",
            ),
            (
                "phase0.mutate_unregistered_subdirectory_source",
                "normalized-key-arithmetic",
                "runtime_representation_phase0_subdirectory",
            ),
        ):
            probe = probes[witness]
            with self.subTest(witness=witness):
                self.assertFalse((REPO_ROOT / probe.path).exists())
                self.assertTrue(probe.creates_file)
                self.assertIs(probe.expected_failure, oracle.UNCLASSIFIED_FAILURE)
                self.assertEqual(probe.expected_kind, kind)
                self.assertEqual(probe.expected_owners, (owner,))
                oracle._expect_mutation_rejected(probe)

    def test_a_symlinked_file_under_a_root_is_scanned_at_its_link_path(self) -> None:
        relative = "crates/chelis-ir/src/runtime_representation_symlink_probe.rs"
        link = REPO_ROOT / relative
        self.assertFalse(link.exists() or link.is_symlink())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "outside_the_roots.rs"
            target.write_text(
                oracle.mutate_unregistered_inventory_source(""), encoding="utf-8"
            )
            link.symlink_to(target)
            oracle._invalidate_inventory_cache()
            try:
                self.assertIn(relative, oracle.inventory_sources(REPO_ROOT))
                with self.assertRaises(oracle.OracleFailure) as caught:
                    oracle.validate_baseline(
                        oracle.load_baseline(), oracle.inventory_rows(REPO_ROOT)
                    )
                target.unlink()
                # A dangling link is not a file, and cargo cannot compile it.
                self.assertNotIn(relative, oracle.inventory_sources(REPO_ROOT))
            finally:
                link.unlink()
                oracle._invalidate_inventory_cache()
        self.assertEqual(caught.exception.code, oracle.UNCLASSIFIED_FAILURE.code)
        self.assertIn(
            f"kind=raw-element-pointer|path={relative}"
            "|owner=runtime_representation_phase0_unregistered",
            caught.exception.details,
        )

    def test_a_git_ignored_file_under_a_root_is_outside_the_universe(self) -> None:
        directory = REPO_ROOT / "crates/chelis-ir/src/runtime_representation_ignored_probe"
        relative = "crates/chelis-ir/src/runtime_representation_ignored_probe/generated.rs"
        self.assertFalse(directory.exists())
        directory.mkdir()
        oracle._invalidate_inventory_cache()
        try:
            (directory / ".gitignore").write_text("*\n", encoding="utf-8")
            (directory / "generated.rs").write_text(
                oracle.mutate_unregistered_inventory_source(""), encoding="utf-8"
            )
            self.assertNotIn(relative, oracle.inventory_sources(REPO_ROOT))
            oracle.validate_baseline(
                oracle.load_baseline(), oracle.inventory_rows(REPO_ROOT)
            )
            (directory / ".gitignore").unlink()
            # The same file, untracked but not ignored, is scanned.
            self.assertIn(relative, oracle.inventory_sources(REPO_ROOT))
        finally:
            shutil.rmtree(directory)
            oracle._invalidate_inventory_cache()

    def test_a_root_whose_directory_departed_fails_closed(self) -> None:
        departed = "crates/chelis-departed/src/**/*.rs"
        with mock.patch.object(
            oracle, "INVENTORY_ROOTS", (*oracle.INVENTORY_ROOTS, departed)
        ):
            with self.assertRaises(oracle.OracleFailure) as caught:
                oracle.inventory_sources(REPO_ROOT)
        self.assertEqual(caught.exception.code, oracle.DEPARTED_ROOT_FAILURE.code)
        self.assertEqual(
            str(caught.exception),
            f"{oracle.DEPARTED_ROOT_FAILURE.reason_prefix}: {departed}; "
            f"{oracle.ROOT_CHANGE_ACTION}",
        )
        # A build-script root names a file that may not exist yet; its crate
        # directory is what must exist.
        self.assertIn("crates/chelis-ir/build.rs", oracle.INVENTORY_ROOTS)
        self.assertFalse((REPO_ROOT / "crates/chelis-ir/build.rs").exists())
        oracle.inventory_sources(REPO_ROOT)

    def test_a_changed_root_moves_the_freeze(self) -> None:
        baseline = oracle.load_baseline()
        rows = oracle.inventory_rows(REPO_ROOT)
        for name, roots in (
            ("removed", oracle.INVENTORY_ROOTS[1:]),
            ("added", (*oracle.INVENTORY_ROOTS, "crates/chelis-vocab/tests/**/*.rs")),
            ("reordered", tuple(reversed(oracle.INVENTORY_ROOTS))),
        ):
            with self.subTest(roots=name), mock.patch.object(
                oracle, "INVENTORY_ROOTS", roots
            ):
                with self.assertRaisesRegex(
                    oracle.OracleFailure,
                    "inventory roots do not match the frozen source universe",
                ):
                    oracle.validate_baseline(baseline, rows)
        mutated = json.loads(json.dumps(baseline))
        mutated["source_inventory"]["roots"] = mutated["source_inventory"]["roots"][1:]
        with self.assertRaisesRegex(oracle.OracleFailure, "freeze digest"):
            oracle.validate_baseline(mutated, rows)

    def test_candidates_come_from_disk_not_the_git_index(self) -> None:
        # cargo compiles what is on disk, so an unstaged file is production
        # source, and a tracked file deleted from disk is not.
        unstaged = "crates/chelis-ir/src/runtime_representation_unstaged_probe.rs"
        deleted = "crates/chelis-ir/src/span_merge.rs"
        with oracle.temporary_mutation(
            REPO_ROOT / unstaged, lambda _: "pub fn unstaged() {}\n", creates_file=True
        ):
            self.assertIn(unstaged, oracle._inventory_candidates(REPO_ROOT))
        with moved(REPO_ROOT / deleted, REPO_ROOT / "crates/chelis-departed-probe/span_merge.rs"):
            self.assertNotIn(deleted, oracle._inventory_candidates(REPO_ROOT))

    def test_the_universe_holds_the_phase2_owned_sources(self) -> None:
        sources = oracle.inventory_sources(REPO_ROOT)
        for source in (
            "crates/chelis-backend-c/src/integer_float.rs",
            "crates/chelis-backend-hip/runtime/chelis_device_descriptor.h",
            "crates/chelis-backend-hip/runtime/chelis_device_owner.cpp",
            "crates/chelis-backend-hip/runtime/chelis_device_owner.h",
            "crates/chelis-backend-hip/runtime/chelis_hip_runtime.h",
            "crates/chelis-backend-metal/runtime/chelis_metal_runtime.h",
            "crates/chelis-ir/src/compositions.rs",
            "crates/chelis-python/src/dlpack.rs",
            "crates/chelis-python/src/native_tensor.rs",
            "crates/chelis-runtime/include/chelis_runtime_views.h",
        ):
            self.assertIn(source, sources)


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = oracle.load_baseline()
        self.rows = oracle.inventory_rows(REPO_ROOT)

    def test_committed_debt_is_exact_and_shrink_only(self) -> None:
        oracle.validate_baseline(self.baseline, self.rows)

    def test_every_identity_is_kind_path_owner_only(self) -> None:
        # No line number, no source offset, and no hash of the exact bytes:
        # reformatting must not move the freeze.
        for row in self.baseline["foundation_rows"]:
            identity = str(row["identity"])
            self.assertRegex(identity, r"^kind=[^|]+\|path=[^|]+\|owner=.+$")
            self.assertNotIn("occurrence=", identity)
            self.assertNotIn("signature=", identity)

    def test_every_row_has_a_phase_the_design_maps(self) -> None:
        for row in self.baseline["foundation_rows"]:
            self.assertIn(row["deletion_phase"], (1, 2, 3, 4))
        for kind, phase in oracle.DELETION_PHASE_BY_KIND.items():
            self.assertIn(phase, (1, 2, 3, 4), kind)

    def test_the_sample_is_evidence_and_never_identity(self) -> None:
        # Samples live only in the active-debt list, which is outside the
        # digest, so rewording a line cannot move the freeze.
        for row in self.baseline["active_debt"]:
            self.assertEqual(set(row), {"identity", "sample"})
        for row in self.baseline["foundation_rows"]:
            self.assertEqual(set(row), {"identity", "deletion_phase"})

    def test_foundation_digest_rejects_an_edited_row(self) -> None:
        mutated = json.loads(json.dumps(self.baseline))
        row = mutated["foundation_rows"][0]
        row["deletion_phase"] = 1 if row["deletion_phase"] == 4 else 4
        with self.assertRaisesRegex(oracle.OracleFailure, "freeze digest"):
            oracle.validate_baseline(mutated, self.rows)

    def test_baseline_contains_only_the_frozen_foundation_and_shrink_only_debt(
        self,
    ) -> None:
        self.assertEqual(
            set(self.baseline),
            {
                "schema_version",
                "freeze_sha256",
                "foundation_rows",
                "source_inventory",
                "active_debt",
                "retired_files",
            },
        )
        self.assertNotIn("coverage_manifest", self.baseline)

    def test_freeze_has_no_live_coverage_configuration_input(self) -> None:
        digest = oracle._freeze_digest(
            self.baseline["foundation_rows"],
            self.baseline["source_inventory"],
        )
        manifest = oracle.coverage_manifest()
        manifest["release_reproducers"] = []
        manifest["hardware_probes"] = []
        manifest["source_inventory"]["universe"]["closure_rule"] = ""
        manifest["acceptance"] = "ALWAYS PASS"
        self.assertEqual(
            digest,
            oracle._freeze_digest(
                self.baseline["foundation_rows"],
                self.baseline["source_inventory"],
            ),
        )

    def test_all_current_mutations_are_exactly_frozen(self) -> None:
        frozen = self.baseline["source_inventory"]["mutations"]
        current = oracle.frozen_mutation_rows(oracle.phase0_mutation_probes())
        self.assertEqual(len(frozen), 44)
        self.assertEqual(frozen, current)

    def test_mutation_implementation_drift_rejects_the_frozen_contract(self) -> None:
        probes = oracle.phase0_mutation_probes()

        def weakened(source: str) -> str:
            return source

        drifted = (
            oracle.MutationProbe(
                witness_id=probes[0].witness_id,
                expected_kind=probes[0].expected_kind,
                path=probes[0].path,
                mutate=weakened,
                expected_failure=probes[0].expected_failure,
                expected_owners=probes[0].expected_owners,
            ),
            *probes[1:],
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure,
            "frozen mutation contract",
        ):
            oracle.validate_baseline(self.baseline, self.rows, probes=drifted)

    def test_mutation_contract_fields_are_inside_the_freeze(self) -> None:
        cases = (
            ("witness_id", "phase0.reviewed_replacement"),
            ("implementation_sha256", "0" * 64),
            (
                "expected_failure",
                {"code": "different.code", "reason_prefix": "different reason"},
            ),
            ("command", "python scripts/runtime_representation_oracle.py --phase 0"),
        )
        original_digest = self.baseline["freeze_sha256"]
        for field, value in cases:
            with self.subTest(field=field):
                mutated = json.loads(json.dumps(self.baseline))
                mutated["source_inventory"]["mutations"][0][field] = value
                self.assertNotEqual(
                    original_digest,
                    oracle._freeze_digest(
                        mutated["foundation_rows"],
                        mutated["source_inventory"],
                    ),
                )
                with self.assertRaisesRegex(oracle.OracleFailure, "freeze digest"):
                    oracle.validate_baseline(mutated, self.rows)

    def test_schema_rejects_a_reintroduced_persisted_coverage_manifest(self) -> None:
        mutated = json.loads(json.dumps(self.baseline))
        mutated["coverage_manifest"] = oracle.coverage_manifest()
        with self.assertRaisesRegex(oracle.OracleFailure, "top-level fields"):
            oracle.validate_baseline(mutated, self.rows)

    def test_active_debt_rejects_ignored_authority_fields(self) -> None:
        for field, value in (
            ("coverage_manifest", {"acceptance": "ALWAYS PASS"}),
            ("authority", "trusted"),
        ):
            with self.subTest(field=field):
                mutated = json.loads(json.dumps(self.baseline))
                mutated["active_debt"][0][field] = value
                with self.assertRaisesRegex(
                    oracle.OracleFailure,
                    r"active_debt\[0\].*exact fields",
                ):
                    oracle.validate_baseline(mutated, self.rows)

    def test_every_persisted_row_requires_exact_fields_and_types(self) -> None:
        cases = (
            ("foundation extra", "foundation_rows", 0, "authority", "trusted"),
            ("foundation missing", "foundation_rows", 0, "deletion_phase", None),
            ("foundation identity type", "foundation_rows", 0, "identity", 7),
            ("foundation phase type", "foundation_rows", 0, "deletion_phase", "4"),
            ("foundation bool phase", "foundation_rows", 0, "deletion_phase", True),
            ("foundation phase value", "foundation_rows", 0, "deletion_phase", 0),
            ("active missing", "active_debt", 0, "sample", None),
            ("active identity type", "active_debt", 0, "identity", 7),
            ("active sample type", "active_debt", 0, "sample", {"text": "stored"}),
        )
        for name, row_class, index, field, value in cases:
            with self.subTest(name=name):
                mutated = json.loads(json.dumps(self.baseline))
                row = mutated[row_class][index]
                if "missing" in name:
                    del row[field]
                else:
                    row[field] = value
                with self.assertRaisesRegex(
                    oracle.OracleFailure,
                    rf"{row_class}\[{index}\]",
                ):
                    oracle.validate_baseline(mutated, self.rows)

    def test_duplicate_identities_are_rejected_before_digest_comparison(self) -> None:
        for row_class in ("foundation_rows", "active_debt"):
            with self.subTest(row_class=row_class):
                mutated = json.loads(json.dumps(self.baseline))
                mutated[row_class][1]["identity"] = mutated[row_class][0]["identity"]
                with self.assertRaisesRegex(oracle.OracleFailure, "duplicate identity"):
                    oracle.validate_baseline(mutated, self.rows)

    def test_mutation_rows_fail_closed_on_unsupported_fields_and_duplicates(
        self,
    ) -> None:
        mutated = json.loads(json.dumps(self.baseline))
        mutated["source_inventory"]["mutations"][0]["authority"] = "ignored"
        with self.assertRaisesRegex(
            oracle.OracleFailure,
            r"source_inventory\.mutations\[0\].*exact fields",
        ):
            oracle.validate_baseline(mutated, self.rows)

        mutated = json.loads(json.dumps(self.baseline))
        mutations = mutated["source_inventory"]["mutations"]
        mutations[1]["witness_id"] = mutations[0]["witness_id"]
        with self.assertRaisesRegex(oracle.OracleFailure, "duplicate witness_id"):
            oracle.validate_baseline(mutated, self.rows)

    def test_source_inventory_requires_the_exact_mutation_envelope(self) -> None:
        roots = oracle.root_directories(REPO_ROOT)
        first = roots[0]
        cases = (
            ("extra", {"mutations": [], "roots": roots, "authority": "ignored"}),
            ("missing", {}),
            ("roots missing", {"mutations": []}),
            ("mutations type", {"mutations": {}, "roots": roots}),
            ("roots type", {"mutations": [], "roots": "crates/**/*.rs"}),
            ("roots empty", {"mutations": [], "roots": []}),
            ("root type", {"mutations": [], "roots": [*roots, "crates/**/*.rs"]}),
            ("root field", {"mutations": [], "roots": [*roots, {"pattern": "x"}]}),
            ("pattern empty", {"mutations": [], "roots": [{**first, "pattern": ""}]}),
            ("directories empty", {"mutations": [], "roots": [{**first, "directories": []}]}),
            (
                "directories unsorted",
                {"mutations": [], "roots": [{**first, "directories": ["b", "a"]}]},
            ),
            ("pattern duplicate", {"mutations": [], "roots": [*roots, first]}),
        )
        for name, source_inventory in cases:
            with self.subTest(name=name):
                mutated = json.loads(json.dumps(self.baseline))
                mutated["source_inventory"] = source_inventory
                with self.assertRaisesRegex(
                    oracle.OracleFailure,
                    r"source_inventory",
                ):
                    oracle.validate_baseline(mutated, self.rows)

    def test_mutation_rows_require_exact_nested_fields_and_types(self) -> None:
        cases = (
            ("missing implementation", "implementation_sha256", None),
            ("implementation type", "implementation_sha256", 7),
            ("implementation spelling", "implementation_sha256", "not-a-sha"),
            ("witness type", "witness_id", 7),
            ("missing path", "path", None),
            ("path type", "path", 7),
            ("path empty", "path", ""),
            ("missing kind", "expected_kind", None),
            ("kind type", "expected_kind", 7),
            ("kind empty", "expected_kind", ""),
            ("missing owners", "expected_owners", None),
            ("owners type", "expected_owners", "owner"),
            ("owners entry type", "expected_owners", [7]),
            ("owners empty entry", "expected_owners", [""]),
            ("command type", "command", ["phase", "0"]),
            ("creates file type", "creates_file", "yes"),
            ("failure extra", "failure_extra", "ignored"),
            ("failure code type", "failure_code", 7),
            ("failure reason type", "failure_reason", 7),
        )
        for name, field, value in cases:
            with self.subTest(name=name):
                mutated = json.loads(json.dumps(self.baseline))
                row = mutated["source_inventory"]["mutations"][0]
                if field == "failure_extra":
                    row["expected_failure"]["ignored"] = value
                elif field == "failure_code":
                    row["expected_failure"]["code"] = value
                elif field == "failure_reason":
                    row["expected_failure"]["reason_prefix"] = value
                elif name.startswith("missing"):
                    del row[field]
                else:
                    row[field] = value
                with self.assertRaisesRegex(
                    oracle.OracleFailure,
                    r"source_inventory\.mutations\[0\]",
                ):
                    oracle.validate_baseline(mutated, self.rows)

    def test_top_level_fields_require_exact_types_before_digest_validation(self) -> None:
        cases = (
            ("missing freeze", "freeze_sha256", None),
            ("schema string", "schema_version", "7"),
            ("schema bool", "schema_version", True),
            ("freeze type", "freeze_sha256", 7),
            ("freeze spelling", "freeze_sha256", "not-a-sha"),
        )
        for name, field, value in cases:
            with self.subTest(name=name):
                mutated = json.loads(json.dumps(self.baseline))
                if name.startswith("missing"):
                    del mutated[field]
                else:
                    mutated[field] = value
                with self.assertRaisesRegex(oracle.OracleFailure, field):
                    oracle.validate_baseline(mutated, self.rows)

    def test_load_rejects_duplicate_json_object_keys(self) -> None:
        duplicates = (
            (
                "top-level",
                '{"schema_version":7,"freeze_sha256":"'
                + oracle.FREEZE_SHA256
                + '","foundation_rows":[],"source_inventory":{"mutations":[]},'
                '"active_debt":[],'
                '"active_debt":[{"identity":"x","sample":""}]}',
                "active_debt",
            ),
            (
                "nested",
                '{"schema_version":7,"freeze_sha256":"'
                + oracle.FREEZE_SHA256
                + '","foundation_rows":[],"source_inventory":{"mutations":[]},'
                '"active_debt":[{"identity":"x","identity":"y","sample":""}]}',
                "identity",
            ),
        )
        for name, duplicate, key in duplicates:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "baseline.json"
                path.write_text(duplicate, encoding="utf-8")
                with mock.patch.object(oracle, "BASELINE_PATH", path):
                    with self.assertRaisesRegex(
                        oracle.OracleFailure,
                        f"duplicate JSON object key: {key}",
                    ):
                        oracle.load_baseline()

    def test_schema_accepts_supported_neighbor_rows(self) -> None:
        supported = json.loads(json.dumps(self.baseline))
        supported["active_debt"][0]["sample"] = "a different reviewed sample"
        supported["foundation_rows"][0]["deletion_phase"] = 1
        supported["source_inventory"]["mutations"][0]["command"] = "reviewed command"
        oracle._validate_baseline_schema(supported)

    def test_regeneration_rejects_ignored_fields_before_deriving_or_writing(
        self,
    ) -> None:
        mutated = json.loads(json.dumps(self.baseline))
        mutated["active_debt"][0]["authority"] = "trusted"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "baseline.json"
            with (
                mock.patch.object(oracle, "load_baseline", return_value=mutated),
                mock.patch.object(oracle, "BASELINE_PATH", output),
                mock.patch.object(oracle, "inventory_rows") as inventory_rows,
            ):
                with self.assertRaisesRegex(
                    oracle.OracleFailure,
                    r"active_debt\[0\].*exact fields",
                ):
                    oracle.regenerate()
            inventory_rows.assert_not_called()
            self.assertFalse(output.exists())

    def test_regeneration_preserves_retired_foundation_rows(self) -> None:
        foundation = self.baseline["foundation_rows"]
        active_rows = self.rows[:1]
        regenerated = oracle.build_foundation_baseline(
            active_rows,
            foundation_rows=foundation,
            source_inventory=self.baseline["source_inventory"],
            active_debt_rows=self.baseline["active_debt"],
        )
        self.assertEqual(regenerated["foundation_rows"], foundation)
        self.assertEqual(
            regenerated["source_inventory"],
            self.baseline["source_inventory"],
        )
        self.assertEqual(
            regenerated["active_debt"],
            [row.to_active_dict() for row in active_rows],
        )

    def test_regeneration_adds_a_new_row_to_the_reviewed_foundation(self) -> None:
        foundation = self.baseline["foundation_rows"]
        added = oracle.InventoryRow(
            kind="direct-data-access",
            path="crates/chelis-runtime/src/invented.rs",
            owner="new_owner",
            deletion_phase=3,
            sample="invented",
        )
        regenerated = oracle.build_foundation_baseline(
            (*self.rows, added),
            foundation_rows=foundation,
            source_inventory=self.baseline["source_inventory"],
            active_debt_rows=self.baseline["active_debt"],
        )
        self.assertIn(added.to_baseline_dict(), regenerated["foundation_rows"])
        self.assertNotEqual(
            regenerated["freeze_sha256"],
            oracle._freeze_digest(foundation, self.baseline["source_inventory"]),
        )

    def test_regeneration_rejects_a_retired_identity_before_writing(self) -> None:
        foundation = self.baseline["foundation_rows"]
        active_ids = {
            str(row["identity"]) for row in self.baseline["active_debt"]
        }
        retired = next(
            row for row in foundation if str(row["identity"]) not in active_ids
        )
        identity = {
            part.split("=", 1)[0]: part.split("=", 1)[1]
            for part in str(retired["identity"]).split("|")
        }
        reactivated = oracle.InventoryRow(
            kind=identity["kind"],
            path=identity["path"],
            owner=identity["owner"],
            deletion_phase=int(retired["deletion_phase"]),
            sample="reviewer reactivation reproduction",
        )
        observed = (*self.rows, reactivated)

        with self.assertRaisesRegex(
            oracle.OracleFailure,
            "retired Phase 0 identity reappeared",
        ):
            oracle.build_foundation_baseline(
                observed,
                foundation_rows=foundation,
                source_inventory=self.baseline["source_inventory"],
                active_debt_rows=self.baseline["active_debt"],
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "baseline.json"
            with (
                mock.patch.object(oracle, "BASELINE_PATH", output),
                mock.patch.object(
                    oracle,
                    "load_baseline",
                    return_value=self.baseline,
                ),
                mock.patch.object(
                    oracle,
                    "inventory_rows",
                    return_value=observed,
                ),
            ):
                with self.assertRaisesRegex(
                    oracle.OracleFailure,
                    "retired Phase 0 identity reappeared",
                ):
                    oracle.regenerate()
            self.assertFalse(output.exists())

    def test_regeneration_rejects_current_mutation_drift_without_rewriting(
        self,
    ) -> None:
        probes = oracle.phase0_mutation_probes()

        def weakened(source: str) -> str:
            return source

        drifted = (
            oracle.MutationProbe(
                witness_id=probes[0].witness_id,
                expected_kind=probes[0].expected_kind,
                path=probes[0].path,
                mutate=weakened,
                expected_failure=probes[0].expected_failure,
                expected_owners=probes[0].expected_owners,
            ),
            *probes[1:],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "baseline.json"
            with (
                mock.patch.object(oracle, "BASELINE_PATH", output),
                mock.patch.object(oracle, "load_baseline", return_value=self.baseline),
                mock.patch.object(
                    oracle,
                    "phase0_mutation_probes",
                    return_value=drifted,
                ),
                mock.patch.object(oracle, "inventory_rows") as inventory_rows,
            ):
                with self.assertRaisesRegex(
                    oracle.OracleFailure,
                    "frozen mutation contract",
                ):
                    oracle.regenerate()
            inventory_rows.assert_not_called()
            self.assertFalse(output.exists())

    def test_active_debt_cannot_add_an_identity(self) -> None:
        mutated = json.loads(json.dumps(self.baseline))
        mutated["active_debt"].append(
            {"identity": "kind=direct-data-access|path=invented.rs|owner=f", "sample": ""}
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "outside the frozen foundation"):
            oracle.validate_baseline(mutated, self.rows)

    def test_active_debt_cannot_retain_a_stale_identity(self) -> None:
        mutated = json.loads(json.dumps(self.baseline))
        keep = mutated["foundation_rows"][0]["identity"]
        with self.assertRaisesRegex(oracle.OracleFailure, "stale active debt"):
            oracle.validate_baseline(
                mutated, tuple(row for row in self.rows if row.identity != keep)
            )

    def test_an_unclassified_hit_names_its_exact_identities(self) -> None:
        extra = oracle.InventoryRow(
            kind="direct-data-access",
            path="crates/chelis-ir/src/dag.rs",
            owner="runtime_representation_phase0_unit_probe",
            deletion_phase=3,
        )
        with self.assertRaises(oracle.OracleFailure) as caught:
            oracle.validate_baseline(self.baseline, (*self.rows, extra))
        self.assertEqual(caught.exception.code, oracle.UNCLASSIFIED_FAILURE.code)
        self.assertIn(extra.identity, caught.exception.details)

    def test_a_retired_foundation_identity_cannot_be_restored(self) -> None:
        restored = oracle.InventoryRow(
            kind="normalized-key-arithmetic",
            path="crates/chelis-ir/src/dag.rs",
            owner="DimExpr::normalized_key",
            deletion_phase=1,
        )
        self.assertIn(
            restored.identity,
            {row["identity"] for row in self.baseline["foundation_rows"]},
        )
        self.assertNotIn(
            restored.identity,
            {row["identity"] for row in self.baseline["active_debt"]},
        )
        with self.assertRaises(oracle.OracleFailure) as caught:
            oracle.validate_baseline(self.baseline, (*self.rows, restored))
        self.assertEqual(caught.exception.code, oracle.UNCLASSIFIED_FAILURE.code)
        self.assertIn(restored.identity, caught.exception.details)

    def test_only_explicit_exact_arithmetic_is_final_inside_the_capacity_owner(self) -> None:
        exact = oracle.InventoryRow(
            kind="exact-capacity-arithmetic",
            path=oracle.CAPACITY_KEY_OWNER,
            owner="ExactLiteralProduct::include",
            deletion_phase=None,
        )
        exact_wrong_owner = oracle.InventoryRow(
            kind="exact-capacity-arithmetic",
            path=oracle.CAPACITY_KEY_OWNER,
            owner="unreviewed_product",
            deletion_phase=None,
        )
        saturated = oracle.InventoryRow(
            kind="saturating-capacity-fold",
            path=oracle.CAPACITY_KEY_OWNER,
            owner="bad_fold",
            deletion_phase=1,
        )
        wrapping = oracle.InventoryRow(
            kind="wrapping-capacity-fold",
            path=oracle.CAPACITY_KEY_OWNER,
            owner="bad_product",
            deletion_phase=1,
        )
        sibling = oracle.InventoryRow(
            kind="normalized-key-arithmetic",
            path="crates/chelis-ir/src/dag.rs",
            owner="normalized_key",
            deletion_phase=1,
        )
        self.assertTrue(oracle.owner_module_final_form(exact.kind, exact.path, exact.owner))
        self.assertFalse(
            oracle.owner_module_final_form(
                exact_wrong_owner.kind, exact_wrong_owner.path, exact_wrong_owner.owner
            )
        )
        self.assertFalse(
            oracle.owner_module_final_form(saturated.kind, saturated.path, saturated.owner)
        )
        self.assertFalse(
            oracle.owner_module_final_form(wrapping.kind, wrapping.path, wrapping.owner)
        )
        self.assertFalse(
            oracle.owner_module_final_form(sibling.kind, sibling.path, sibling.owner)
        )
        self.assertFalse(
            oracle.owner_module_final_form(
                "legacy-capacity-key-use", oracle.CAPACITY_KEY_OWNER, "legacy"
            )
        )


class FrozenMutationContractTests(unittest.TestCase):
    """Exercise the reviewed contract without invoking the inventory scanner."""

    def setUp(self) -> None:
        self.probes = oracle.phase0_mutation_probes()
        self.baseline = oracle.load_baseline()
        # All foundation rows may retire without moving the mutation freeze.
        # This fixture checks contract validation, not current source inventory.
        self.baseline["active_debt"] = []
        self.probe = next(
            probe for probe in self.probes
            if probe.witness_id == "phase0.mutate_element_binding"
        )
        self.assertEqual(
            self.probe.expected_owners,
            ("Storage for UnregisteredElement", "private :: Sealed for UnregisteredElement"),
        )

    def test_unchanged_probes_match_the_reviewed_digest_and_contract(self) -> None:
        oracle.validate_baseline(self.baseline, (), probes=self.probes)
        self.assertEqual(
            self.baseline["source_inventory"]["mutations"],
            oracle.mutation_manifest(self.probes),
        )

    def test_rejection_obligation_drift_moves_digest_and_fails_comparison(self) -> None:
        variants = (
            ("path", replace(self.probe, path=Path(oracle.VOCAB_OWNER))),
            ("kind", replace(self.probe, expected_kind="direct-data-access")),
            ("first owner removed", replace(self.probe, expected_owners=self.probe.expected_owners[1:])),
            ("second owner removed", replace(self.probe, expected_owners=self.probe.expected_owners[:1])),
            ("all owners removed", replace(self.probe, expected_owners=())),
        )
        for name, replacement in variants:
            probes = tuple(
                replacement if probe is self.probe else probe for probe in self.probes
            )
            projected = oracle.frozen_source_inventory(probes)
            with self.subTest(drift=name, check="digest"):
                self.assertNotEqual(
                    self.baseline["freeze_sha256"],
                    oracle._freeze_digest(self.baseline["foundation_rows"], projected),
                )
            with self.subTest(drift=name, check="comparison"):
                with self.assertRaisesRegex(oracle.OracleFailure, "frozen mutation contract"):
                    oracle._validate_frozen_mutation_contract(
                        self.baseline["source_inventory"], probes,
                    )
            with self.subTest(drift=name, check="validator"):
                with self.assertRaisesRegex(oracle.OracleFailure, "frozen mutation contract"):
                    oracle.validate_baseline(self.baseline, (), probes=probes)

    def test_persisted_rejection_drift_requires_a_reviewed_digest(self) -> None:
        for field, value in (
            ("path", oracle.VOCAB_OWNER),
            ("expected_kind", "direct-data-access"),
            ("expected_owners", []),
        ):
            with self.subTest(field=field):
                baseline = json.loads(json.dumps(self.baseline))
                row = next(
                    row for row in baseline["source_inventory"]["mutations"]
                    if row["witness_id"] == self.probe.witness_id
                )
                row[field] = value
                digest = oracle._freeze_digest(
                    baseline["foundation_rows"], baseline["source_inventory"],
                )
                self.assertNotEqual(self.baseline["freeze_sha256"], digest)
                # Rewriting the stored digest alone cannot authorize contract drift.
                baseline["freeze_sha256"] = digest
                with self.assertRaisesRegex(oracle.OracleFailure, "freeze digest"):
                    oracle.validate_baseline(baseline, (), probes=self.probes)

    def test_owner_list_spelling_matches_the_existing_live_manifest(self) -> None:
        for owners in (
            tuple(reversed(self.probe.expected_owners)),
            (*self.probe.expected_owners, self.probe.expected_owners[0]),
        ):
            with self.subTest(owners=owners):
                probes = tuple(
                    replace(probe, expected_owners=owners) if probe is self.probe else probe
                    for probe in self.probes
                )
                rows = oracle.frozen_mutation_rows(probes)
                self.assertEqual(rows, oracle.mutation_manifest(probes))
                self.assertNotEqual(rows, oracle.frozen_mutation_rows(self.probes))
                with self.assertRaisesRegex(oracle.OracleFailure, "frozen mutation contract"):
                    oracle._validate_frozen_mutation_contract(
                        self.baseline["source_inventory"], probes,
                    )

    def test_live_configuration_does_not_enter_the_frozen_contract(self) -> None:
        original = oracle.coverage_manifest(self.probes)
        with (
            mock.patch.object(oracle, "phase0_legs", return_value=()),
            mock.patch.object(oracle, "hardware_probe_manifest", return_value=()),
        ):
            current = oracle.coverage_manifest(self.probes)
            self.assertNotEqual(original, current)
            self.assertEqual(
                self.baseline["freeze_sha256"],
                oracle._freeze_digest(
                    self.baseline["foundation_rows"],
                    oracle.frozen_source_inventory(self.probes),
                ),
            )
            oracle.validate_baseline(self.baseline, (), probes=self.probes)

    def test_schema_accepts_full_manifest_and_rejects_legacy_rows(self) -> None:
        baseline = json.loads(json.dumps(self.baseline))
        baseline["schema_version"] = 8
        baseline["source_inventory"] = oracle.frozen_source_inventory(self.probes)
        with self.subTest(schema="current"):
            oracle._validate_baseline_schema(baseline)
        baseline["schema_version"] = 7
        del baseline["source_inventory"]["roots"]
        with self.subTest(schema="schema 7 without frozen roots"):
            with self.assertRaisesRegex(oracle.OracleFailure, "schema_version"):
                oracle._validate_baseline_schema(baseline)
        baseline["schema_version"] = 8
        with self.subTest(schema="schema 7 relabeled current"):
            with self.assertRaisesRegex(oracle.OracleFailure, "source_inventory.*exact fields"):
                oracle._validate_baseline_schema(baseline)
        baseline["source_inventory"]["roots"] = oracle.root_directories(REPO_ROOT)
        baseline["schema_version"] = 6
        for row in baseline["source_inventory"]["mutations"]:
            for field in ("path", "expected_kind", "expected_owners"):
                del row[field]
        with self.subTest(schema="legacy"):
            with self.assertRaisesRegex(oracle.OracleFailure, "schema_version"):
                oracle._validate_baseline_schema(baseline)
        baseline["schema_version"] = 8
        with self.subTest(schema="legacy rows relabeled current"):
            with self.assertRaisesRegex(oracle.OracleFailure, "mutations.*exact fields"):
                oracle._validate_baseline_schema(baseline)

    def test_schema_rejects_malformed_rejection_fields_before_comparison(self) -> None:
        for field, values in (
            ("path", (None, 7, "")),
            ("expected_kind", (None, 7, "")),
            ("expected_owners", (None, "owner", [7], [""], [None])),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    baseline = json.loads(json.dumps(self.baseline))
                    row = baseline["source_inventory"]["mutations"][0]
                    row[field] = value
                    with self.assertRaisesRegex(oracle.OracleFailure, field):
                        oracle._validate_baseline_schema(baseline)
            with self.subTest(field=field, value="missing"):
                baseline = json.loads(json.dumps(self.baseline))
                baseline["source_inventory"]["mutations"][0].pop(field, None)
                with self.assertRaisesRegex(oracle.OracleFailure, "exact fields"):
                    oracle._validate_baseline_schema(baseline)


class MutationContractTests(unittest.TestCase):
    def test_every_mutation_applies_to_its_current_source(self) -> None:
        # Mutations execute only in the oracle proper, so without this an
        # anchor the tree has drifted from surfaces only in that heavy run.
        for probe in oracle.phase0_mutation_probes():
            path = REPO_ROOT / probe.path
            source = path.read_text(encoding="utf-8") if path.exists() else ""
            with self.subTest(witness=probe.witness_id):
                self.assertNotEqual(probe.mutate(source), source)

    def test_incomplete_dtype_mutation_survives_an_appended_dtype(self) -> None:
        declaration = "pub enum RuntimeDType {\n"
        source = (REPO_ROOT / "crates/chelis-vocab/src/lib.rs").read_text(encoding="utf-8")
        tail = source.index("\n}", source.index(declaration))
        appended = source[:tail] + "\n    Phase0Successor = 126," + source[tail:]
        for name, current in (("current", source), ("appended", appended)):
            with self.subTest(source=name):
                mutated = oracle.mutate_incomplete_dtype(current)
                start = mutated.index(declaration)
                body = mutated[start : mutated.index("\n}", start)]
                self.assertTrue(body.endswith("\n    Phase0Probe = 127,"), body[-80:])

    def test_header_mutations_remain_inside_their_include_guards(self) -> None:
        for probe in oracle.phase0_mutation_probes():
            if probe.path.suffix != ".h":
                continue
            source = (REPO_ROOT / probe.path).read_text()
            mutated = probe.mutate(source)
            original_guard_end = source.rstrip().rfind("#endif")
            mutated_guard_end = mutated.rstrip().rfind("#endif")
            self.assertGreaterEqual(original_guard_end, 0, probe.witness_id)
            self.assertGreaterEqual(mutated_guard_end, 0, probe.witness_id)
            self.assertEqual(
                mutated[mutated_guard_end + len("#endif") :],
                source[original_guard_end + len("#endif") :],
                f"{probe.witness_id} planted a repeated declaration after the guard",
            )

    def test_element_final_forms_are_exact_and_never_admit_raw_access(self) -> None:
        for owner in oracle.ELEMENT_FINAL_CONTRACT_OWNERS:
            self.assertTrue(oracle.owner_module_final_form(
                "dtype-contract", oracle.ELEMENT_OWNER, owner
            ))
            for kind, path, candidate in (
                ("dtype-contract", oracle.ELEMENT_OWNER, owner + "New"),
                ("dtype-contract", "crates/chelis-runtime/src/lib.rs", owner),
                ("raw-element-pointer", oracle.ELEMENT_OWNER, owner),
            ):
                self.assertFalse(oracle.owner_module_final_form(kind, path, candidate))
        self.assertTrue(oracle.owner_module_final_form(
            "width-arithmetic", oracle.ELEMENT_OWNER, "assert_registration"
        ))
        self.assertFalse(oracle.owner_module_final_form(
            "raw-element-pointer", oracle.ELEMENT_OWNER, "assert_registration"
        ))
        self.assertTrue(any("element_contract" in leg.argv for leg in oracle.phase0_legs()))
        probe = next(p for p in oracle.phase0_mutation_probes()
                     if p.witness_id == "phase0.mutate_element_binding")
        self.assertEqual(probe.expected_owners, (
            "Storage for UnregisteredElement", "private :: Sealed for UnregisteredElement",
        ))
        self.assertIn("impl Storage for UnregisteredElement", probe.mutate(""))

    def test_vocabulary_final_forms_do_not_admit_new_variants_or_owners(self) -> None:
        path = "crates/chelis-vocab/src/lib.rs"
        for owner in oracle.VOCAB_FINAL_CONTRACT_OWNERS:
            self.assertTrue(oracle.owner_module_final_form("dtype-contract", path, owner))
            self.assertFalse(
                oracle.owner_module_final_form("dtype-contract", path, owner + "New")
            )
            self.assertFalse(oracle.owner_module_final_form(
                "dtype-contract", "crates/chelis-runtime/src/lib.rs", owner
            ))
        self.assertTrue(oracle.owner_module_final_form(
            "width-arithmetic", path, "DTypeContract::byte_width"
        ))
        self.assertFalse(oracle.owner_module_final_form(
            "width-arithmetic", path, "DTypeContract::other_width"
        ))
        self.assertFalse(oracle.owner_module_final_form(
            "raw-element-pointer", path, "DTypeContract::byte_width"
        ))

    def test_every_seam_kind_in_the_ledger_has_a_mutation(self) -> None:
        baseline = oracle.load_baseline()
        kinds = {
            str(row["identity"]).split("|", 1)[0].removeprefix("kind=")
            for row in baseline["foundation_rows"]
        }
        covered = {probe.expected_kind for probe in oracle.phase0_mutation_probes()}
        self.assertEqual(
            kinds - covered,
            set(),
            "a seam class in the frozen ledger has no controlled mutation",
        )

    def test_witness_ids_are_unique(self) -> None:
        ids = [probe.witness_id for probe in oracle.phase0_mutation_probes()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_mutation_targets_a_registered_source(self) -> None:
        sources = oracle.inventory_sources(REPO_ROOT)
        for probe in oracle.phase0_mutation_probes():
            relative = probe.path.as_posix()
            if not probe.creates_file:
                self.assertIn(relative, sources, probe.witness_id)
                continue
            # A new-file witness: the file it creates must not exist yet and
            # must land inside the universe, or its rejection would prove
            # nothing about the scan.
            self.assertFalse((REPO_ROOT / probe.path).exists(), probe.witness_id)
            with self.subTest(witness=probe.witness_id), oracle.temporary_mutation(
                REPO_ROOT / probe.path, probe.mutate, creates_file=True
            ):
                self.assertIn(relative, oracle.inventory_sources(REPO_ROOT))

    def test_manifest_binds_exact_mutation_semantics_and_command(self) -> None:
        for entry in oracle.mutation_manifest(oracle.phase0_mutation_probes()):
            self.assertEqual(
                set(entry),
                {
                    "witness_id",
                    "expected_kind",
                    "path",
                    "implementation_sha256",
                    "expected_failure",
                    "expected_owners",
                    "command",
                    "creates_file",
                },
            )
            self.assertEqual(entry["command"], oracle.PHASE0_COMMAND)
            self.assertIs(type(entry["creates_file"]), bool)
            self.assertRegex(str(entry["implementation_sha256"]), r"^[0-9a-f]{64}$")

    def test_mutation_body_change_moves_the_runtime_manifest_and_frozen_projection(
        self,
    ) -> None:
        probes = oracle.phase0_mutation_probes()
        before = oracle.coverage_manifest(probes)
        frozen = oracle.frozen_mutation_rows(probes)

        def weakened(source: str) -> str:
            return source

        replaced = (
            oracle.MutationProbe(
                witness_id=probes[0].witness_id,
                expected_kind=probes[0].expected_kind,
                path=probes[0].path,
                mutate=weakened,
                expected_failure=probes[0].expected_failure,
            ),
            *probes[1:],
        )
        self.assertNotEqual(before, oracle.coverage_manifest(replaced))
        self.assertNotEqual(frozen, oracle.frozen_mutation_rows(replaced))

    def test_implementation_digest_is_independent_of_import_name(self) -> None:
        digest = oracle._mutation_implementation_sha256(oracle.mutate_direct_data_access)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(
            digest,
            oracle._mutation_implementation_sha256(oracle.mutate_direct_data_access),
        )

    def test_temporary_mutation_restores_original_bytes_after_failure(self) -> None:
        path = REPO_ROOT / "crates/chelis-runtime/src/decimal_parse.rs"
        original = path.read_bytes()
        with self.assertRaises(RuntimeError):
            with oracle.temporary_mutation(path, oracle.mutate_direct_data_access):
                raise RuntimeError("simulated failure inside the mutation window")
        self.assertEqual(path.read_bytes(), original)

    def test_temporary_mutation_refuses_a_dirty_source(self) -> None:
        # The dirty check lives inside temporary_mutation, so no caller, unit
        # tests included, can plant a mutation over uncommitted work.
        path = REPO_ROOT / "crates/chelis-runtime/src/decimal_parse.rs"
        with mock.patch.object(
            oracle.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(args=(), returncode=0, stdout=" M x\n", stderr=""),
        ):
            with self.assertRaisesRegex(oracle.OracleFailure, "dirty detector source"):
                with oracle.temporary_mutation(path, lambda source: source):
                    pass

    def test_a_created_probe_file_is_removed_not_left_behind(self) -> None:
        path = REPO_ROOT / "crates/chelis-ir/src/runtime_representation_unit_probe.rs"
        self.assertFalse(path.exists())
        with oracle.temporary_mutation(path, lambda _: "// probe\n", creates_file=True):
            self.assertTrue(path.exists())
        self.assertFalse(path.exists())
        with self.assertRaisesRegex(oracle.OracleFailure, "mutation target does not exist"):
            with oracle.temporary_mutation(path, lambda _: "// probe\n"):
                pass
        existing = REPO_ROOT / "crates/chelis-runtime/src/decimal_parse.rs"
        with self.assertRaisesRegex(
            oracle.OracleFailure, "new-file mutation target already exists"
        ):
            with oracle.temporary_mutation(existing, lambda _: "// probe\n", creates_file=True):
                pass


class ManifestTests(unittest.TestCase):
    def test_phase_two_final_forms_are_exact_closed_and_manifested(self) -> None:
        forms = set(oracle.PHASE2_FINAL_FORMS)
        self.assertEqual(len(forms), 35)
        for path, kind, owner in forms:
            self.assertTrue(oracle.owner_module_final_form(kind, path, owner))
            self.assertFalse(oracle.owner_module_final_form(kind, path, owner + "_unchecked"))
            self.assertFalse(oracle.owner_module_final_form(kind, path + ".other", owner))
        self.assertFalse(oracle.owner_module_final_form(
            "raw-element-pointer",
            "crates/chelis-backend-hip/runtime/chelis_hip_runtime.h",
            "chelis_hipblas_sgemm_row_major",
        ))
        manifest = oracle.coverage_manifest()["source_inventory"]["owner_module_final_forms"]
        manifested = {
            (path, row["kind"], row["owner"])
            for path, rows in manifest.items()
            for row in rows
        }
        self.assertTrue(forms <= manifested)

    def test_checked_host_metadata_has_paired_profiles_and_exact_width_owners(self) -> None:
        path = "crates/chelis-runtime/src/metadata.rs"
        self.assertIn(path, oracle.inventory_sources(REPO_ROOT))
        owners = {"ElementCount::bytes", "ElementCount::scratch_len"}
        for owner in owners:
            self.assertTrue(oracle.owner_module_final_form("width-arithmetic", path, owner))
            self.assertFalse(oracle.owner_module_final_form("raw-element-pointer", path, owner))
            self.assertFalse(oracle.owner_module_final_form("width-arithmetic", path + ".other", owner))
        self.assertFalse(oracle.owner_module_final_form("width-arithmetic", path, "new_width"))
        final = oracle.coverage_manifest()["source_inventory"]["owner_module_final_forms"]
        self.assertEqual(final[path], [
            {"kind": "width-arithmetic", "owner": owner} for owner in sorted(owners)
        ])
        commands = [" ".join(leg.argv) for leg in oracle.phase0_legs()]
        for test in ("checked_metadata", "metadata_compile", "checked_metadata_padding", "checked_c_metadata",
                     "checked_c_indexing", "checked_c_movement", "checked_c_reduction", "checked_c_sparse", "checked_c_matmul", "checked_c_window", "checked_c_literal", "exact_tagged_c_abi",
                     "op33_empty_tensor_axis_decomposition", "op33_tensor_validation",
                     "op33_legal_domain_matrix", "dim_carrier_int64",
                     "tensor_repurpose", "tensor_write_guard"):
            self.assertTrue(any(test in command and "--release" not in command for command in commands), test)
            self.assertTrue(any(test in command and "--release" in command for command in commands), test)
        self.assertTrue(any(
            probe.path == Path(path) and probe.expected_kind == "width-arithmetic"
            for probe in oracle.phase0_mutation_probes()
        ))

    def test_retired_capacity_projection_has_a_release_execution_leg(self) -> None:
        commands = [" ".join(leg.argv) for leg in oracle.phase0_legs()]
        self.assertTrue(any(
            "dim_expr_evaluation" in command and "--release" in command
            for command in commands
        ), commands)

    def test_checked_c_index_projection_owners_require_their_executable_control(self) -> None:
        manifest = oracle.coverage_manifest()
        forms = manifest["source_inventory"]["owner_module_final_forms"]
        for path, owner in oracle.C_INDEX_PROJECTION_OWNERS:
            self.assertTrue(oracle.owner_module_final_form("backend-element-spelling", path, owner))
            self.assertFalse(oracle.owner_module_final_form("width-arithmetic", path, owner))
            self.assertFalse(oracle.owner_module_final_form("backend-element-spelling", path, owner + "_unchecked"))
            self.assertIn({"kind": "backend-element-spelling", "owner": owner}, forms[path])
        commands = [leg.argv for leg in oracle.phase0_legs()]
        self.assertTrue(any("chelis-backend-c" in command and "checked_c_indexing" in command for command in commands))
        self.assertTrue(any("exec_compile" in command and "test(checked_c_indexing_)" in command for command in commands))

    def test_typed_nonnumeric_backend_owners_require_exact_execution_controls(self) -> None:
        manifest = oracle.coverage_manifest()
        forms = manifest["source_inventory"]["owner_module_final_forms"]
        self.assertNotIn(
            (
                "crates/chelis-backend-hip/src/kernels.rs",
                "backend-element-spelling",
                "NUMERIC_DEVICE_HELPERS",
            ),
            oracle.TYPED_NONNUMERIC_BACKEND_FINAL_FORMS,
        )
        for path, kind, owner in oracle.TYPED_NONNUMERIC_BACKEND_FINAL_FORMS:
            self.assertTrue(oracle.owner_module_final_form(kind, path, owner))
            self.assertFalse(oracle.owner_module_final_form(kind, path, owner + "_unchecked"))
            self.assertFalse(oracle.owner_module_final_form("width-arithmetic", path, owner))
            self.assertIn({"kind": kind, "owner": owner}, forms[path])
        commands = [" ".join(leg.argv) for leg in oracle.phase0_legs()]
        self.assertTrue(any(
            "logical_bool_semantics" in command for command in commands
        ))
        self.assertTrue(any(
            "exec_compile" in command
            and "typed_comparison_c_matrix_matches_evaluator_for_every_identity_and_dtype" in command
            and "typed_logical_c_truth_tables_are_bool8" in command
            and "typed_where_c_copies_selected_storage_bits_for_every_admitted_dtype" in command
            and "typed_nonnumeric_c_permuted_stepped_views_match_evaluator_and_preserve_bits" in command
            for command in commands
        ))
        self.assertTrue(any(
            "logical_comparison_where" in command
            and "chelis-backend-hip" in command
            for command in commands
        ))
        hardware = oracle.hardware_probe_manifest()
        self.assertTrue(any(
            "logical_comparison_where_gpu" in probe["command"]
            for probe in hardware
        ))

    def test_result_claim_metadata_owners_require_exact_execution_controls(self) -> None:
        expected = (
            (
                "crates/chelis-backend-c/src/host_emit.rs",
                "load-store-template",
                "append_host_result_claim_checks",
            ),
            (
                "crates/chelis-backend-c/src/host_emit.rs",
                "load-store-template",
                "append_host_result_interface_origin_support",
            ),
        )
        self.assertEqual(oracle.RESULT_CLAIM_METADATA_FINAL_FORMS, expected)
        forms = oracle.coverage_manifest()["source_inventory"]["owner_module_final_forms"]
        for path, kind, owner in expected:
            self.assertTrue(oracle.owner_module_final_form(kind, path, owner))
            self.assertFalse(oracle.owner_module_final_form(kind, path, owner + "_unchecked"))
            self.assertFalse(
                oracle.owner_module_final_form("backend-element-spelling", path, owner)
            )
            self.assertFalse(
                oracle.owner_module_final_form(
                    kind, path.replace("host_emit.rs", "emit.rs"), owner
                )
            )
            self.assertIn({"kind": kind, "owner": owner}, forms[path])
        commands = [" ".join(leg.argv) for leg in oracle.phase0_legs()]
        self.assertTrue(any(
            "issue_1771_callable_selected_result_claims" in command
            and "c_aggregate_interface_ingress_stamps_each_tensor_field_as_load" in command
            and "c_direct_list_skip_retains_selected_tail_producer" in command
            and "c_list_and_adt_projection_retains_selected_producer" in command
            and "c_nested_list_pattern_retains_selected_tail_producer" in command
            and "c_option_projection_distinguishes_local_and_formal_origins" in command
            and "c_aggregate_origin_arena_is_fresh_for_repeated_public_calls" in command
            for command in commands
        ))

    def test_named_list_entry_metadata_owners_require_exact_execution_controls(self) -> None:
        expected = (
            (
                "crates/chelis-backend-c/src/host_emit.rs",
                "backend-element-spelling",
                "HostResultClaim::frame_lines",
            ),
            (
                "crates/chelis-backend-c/src/host_emit/entry_walk.rs",
                "load-store-template",
                "EntryWalkers < 'a >::render",
            ),
            (
                "crates/chelis-backend-c/src/host_emit/entry_walk.rs",
                "backend-element-spelling",
                "retained_list_pass::walk",
            ),
        )
        self.assertEqual(oracle.LIST_ENTRY_METADATA_FINAL_FORMS, expected)
        forms = oracle.coverage_manifest()["source_inventory"]["owner_module_final_forms"]
        for path, kind, owner in expected:
            self.assertTrue(oracle.owner_module_final_form(kind, path, owner))
            self.assertFalse(oracle.owner_module_final_form(kind, path, owner + "_unchecked"))
            self.assertFalse(
                oracle.owner_module_final_form(kind, path.replace("host_emit", "emit"), owner)
            )
            other_kind = (
                "load-store-template"
                if kind == "backend-element-spelling"
                else "backend-element-spelling"
            )
            self.assertFalse(oracle.owner_module_final_form(other_kind, path, owner))
            self.assertIn({"kind": kind, "owner": owner}, forms[path])
        commands = [" ".join(leg.argv) for leg in oracle.phase0_legs()]
        self.assertTrue(any(
            "--test issue_2627_list_entry_extents" in command
            and "--test issue_1788_entry_obligations" in command
            for command in commands
        ))

    def test_correctly_rounded_c_owners_require_exact_execution_controls(self) -> None:
        expected = (
            (
                "crates/chelis-backend-c/src/fp_env.rs",
                "backend-element-spelling",
                "HELPERS",
            ),
            (
                "crates/chelis-backend-c/src/fp_env.rs",
                "backend-element-spelling",
                "prune_unused_nan_helpers",
            ),
            (
                "crates/chelis-backend-c/src/generated_header.rs",
                "backend-element-spelling",
                "validate_generated_include_set::ALLOWED_INCLUDES",
            ),
        )
        self.assertEqual(oracle.CORRECTLY_ROUNDED_C_FINAL_FORMS, expected)
        forms = oracle.coverage_manifest()["source_inventory"]["owner_module_final_forms"]
        for path, kind, owner in expected:
            self.assertTrue(oracle.owner_module_final_form(kind, path, owner))
            self.assertFalse(oracle.owner_module_final_form(kind, path, owner + "_unchecked"))
            self.assertFalse(
                oracle.owner_module_final_form(kind, path.replace("chelis-backend-c", "chelis-backend-hip"), owner)
            )
            other_kind = (
                "load-store-template"
                if kind == "backend-element-spelling"
                else "backend-element-spelling"
            )
            self.assertFalse(oracle.owner_module_final_form(other_kind, path, owner))
            self.assertIn({"kind": kind, "owner": owner}, forms[path])
        # The renamed fused step and the extracted activation body stay
        # transition debt rather than borrowing the final-form authority.
        for path, owner in (
            ("crates/chelis-backend-c/src/emit.rs", "CEmitter::scalar_step_raw_expr"),
            ("crates/chelis-backend-c/src/host_emit.rs", "activation_body"),
        ):
            self.assertFalse(
                oracle.owner_module_final_form("backend-element-spelling", path, owner)
            )
        commands = [" ".join(leg.argv) for leg in oracle.phase0_legs()]
        self.assertTrue(any(
            "-p chelis-backend-c --lib --test exec_compile" in command
            and "test(toolchain::tests::verify_compiler_)" in command
            and "direct_and_fused_float_arithmetic_finalizes_canonical_nan_at_every_width" in command
            and "direct_extrema_preserve_nan_payloads_and_lhs_signed_zero_at_every_float_width" in command
            for command in commands
        ))

    def test_correctly_rounded_pow_c_owner_requires_exact_execution_controls(self) -> None:
        path = "crates/chelis-backend-c/src/emit.rs"
        kind = "backend-element-spelling"
        owner = "CEmitter::emit_binary_kernel"
        self.assertEqual(oracle.CORRECTLY_ROUNDED_POW_C_FINAL_FORMS, ((path, kind, owner),))
        forms = oracle.coverage_manifest()["source_inventory"]["owner_module_final_forms"]
        self.assertTrue(oracle.owner_module_final_form(kind, path, owner))
        self.assertFalse(oracle.owner_module_final_form(kind, path, owner + "_unchecked"))
        self.assertFalse(oracle.owner_module_final_form(kind, path, "CEmitter::emit_binary_func"))
        self.assertFalse(oracle.owner_module_final_form("load-store-template", path, owner))
        self.assertFalse(
            oracle.owner_module_final_form(kind, path.replace("chelis-backend-c", "chelis-backend-hip"), owner)
        )
        self.assertIn({"kind": kind, "owner": owner}, forms[path])
        commands = [leg.argv for leg in oracle.phase0_legs()]
        self.assertIn(
            ("cargo", "nextest", "run", "-p", "chelis-backend-c", "--test", "pow_kernel_routes"),
            commands,
        )

    def test_key_callable_load_store_owner_requires_exact_execution_controls(self) -> None:
        path = "crates/chelis-backend-c/src/host_emit.rs"
        kind = "load-store-template"
        owner = "append_key_callable_helpers"
        self.assertEqual(oracle.KEY_CALLABLE_BACKEND_FINAL_FORMS, ((path, kind, owner),))
        forms = oracle.coverage_manifest()["source_inventory"]["owner_module_final_forms"]
        self.assertTrue(oracle.owner_module_final_form(kind, path, owner))
        self.assertFalse(oracle.owner_module_final_form(kind, path, owner + "_unchecked"))
        self.assertFalse(oracle.owner_module_final_form("backend-element-spelling", path, owner))
        self.assertFalse(oracle.owner_module_final_form(kind, path.replace("host_emit.rs", "emit.rs"), owner))
        self.assertIn({"kind": kind, "owner": owner}, forms[path])
        commands = [leg.argv for leg in oracle.phase0_legs()]
        self.assertIn(
            (
                "cargo", "nextest", "run", "-p", "chelis-compiler-api",
                "--features", "ownership-ledger", "--test", "key_tensor_forms",
            ),
            commands,
        )

    def test_exact_reduction_backend_owners_require_exact_execution_controls(self) -> None:
        expected = (
            (
                "crates/chelis-backend-c/src/emit.rs",
                "backend-element-spelling",
                "CEmitter::emit_mean_nonempty_guard",
            ),
            (
                "crates/chelis-backend-c/src/emit.rs",
                "backend-element-spelling",
                "CEmitter::emit_reduce_extreme",
            ),
        )
        self.assertEqual(oracle.EXACT_REDUCTION_BACKEND_FINAL_FORMS, expected)
        forms = oracle.coverage_manifest()["source_inventory"]["owner_module_final_forms"]
        for path, kind, owner in expected:
            self.assertTrue(oracle.owner_module_final_form(kind, path, owner))
            self.assertFalse(oracle.owner_module_final_form(kind, path, owner + "_unchecked"))
            self.assertFalse(oracle.owner_module_final_form("load-store-template", path, owner))
            self.assertIn({"kind": kind, "owner": owner}, forms[path])
        commands = [" ".join(leg.argv) for leg in oracle.phase0_legs()]
        self.assertTrue(any(
            "chelis-backend-c" in command
            and "issue_1281_exact_reductions" in command
            for command in commands
        ))

    def test_list_map_backend_owners_bind_to_exact_op55_execution(self) -> None:
        path = "crates/chelis-backend-c/src/emit.rs"
        expected = (
            (path, "load-store-template", "CEmitter::emit_list_map_capture"),
            (path, "backend-element-spelling", "CEmitter::emit_ordered_adjoint_sum"),
        )
        self.assertEqual(oracle.LIST_MAP_BACKEND_FINAL_FORMS, expected)
        forms = oracle.coverage_manifest()["source_inventory"]["owner_module_final_forms"]
        for owner_path, kind, owner in expected:
            self.assertTrue(oracle.owner_module_final_form(kind, owner_path, owner))
            self.assertFalse(oracle.owner_module_final_form(kind, owner_path, owner + "_unchecked"))
            self.assertFalse(oracle.owner_module_final_form("width-arithmetic", owner_path, owner))
            self.assertIn({"kind": kind, "owner": owner}, forms[owner_path])
        commands = [" ".join(leg.argv) for leg in oracle.phase0_legs()]
        self.assertTrue(any(
            "chelis-backend-c" in command
            and "issue_570_runtime_iota" in command
            and "ordered_cotangent_native_groups_check_actual_column_lengths" in command
            for command in commands
        ))
        self.assertTrue(any(
            "chelis-cli" in command
            and "issue_2419_range_tensor_ad" in command
            and "runtime_capture_tree_rounds_each_pair_at_the_capture_dtype" in command
            and "runtime_capture_tree_preserves_inactive_and_empty_rows" in command
            for command in commands
        ))

    def test_integer_unary_final_forms_are_exact_and_require_execution_controls(self) -> None:
        expected = (
            ("crates/chelis-backend-c/src/integer_float.rs", "backend-element-spelling", "integer_to_float_bits"),
            ("crates/chelis-backend-hip/src/kernels.rs", "backend-element-spelling", "cast_integer_to_float"),
            ("crates/chelis-backend-metal/src/emit.rs", "backend-element-spelling", "Emitter < 'plan >::emit_expand"),
            ("crates/chelis-backend-metal/src/emit.rs", "backend-element-spelling", "Emitter < 'plan >::emit_integer_float_cast"),
        )
        self.assertEqual(oracle.INTEGER_UNARY_BACKEND_FINAL_FORMS, expected)
        forms = oracle.coverage_manifest()["source_inventory"]["owner_module_final_forms"]
        for path, kind, owner in expected:
            self.assertTrue(oracle.owner_module_final_form(kind, path, owner))
            self.assertFalse(oracle.owner_module_final_form(kind, path, owner + "_unchecked"))
            self.assertFalse(oracle.owner_module_final_form(kind, path + ".other", owner))
            self.assertFalse(oracle.owner_module_final_form("load-store-template", path, owner))
            self.assertIn({"kind": kind, "owner": owner}, forms[path])
        commands = [" ".join(leg.argv) for leg in oracle.phase0_legs()]
        self.assertIn(
            "cargo nextest run -p chelis-backend-c --lib -E test(integer_float::tests::)",
            commands,
        )
        self.assertIn(
            "cargo nextest run -p chelis-backend-hip -p chelis-backend-metal "
            "--test integer_abs --test integer_abs_guard",
            commands,
        )
        probes = {probe["lane"]: probe for probe in oracle.hardware_probe_manifest()}
        for lane, command in (
            ("hip-integer-unary", "scripts/hip_test.py -p chelis-backend-hip --test integer_abs -- --ignored --test-threads=1"),
            ("metal-integer-unary", "cargo test -p chelis-backend-metal --test integer_abs_guard -- --ignored --test-threads=1"),
        ):
            self.assertEqual(probes[lane]["command"], command)
            self.assertEqual(probes[lane]["status"], "manual-required")

    def test_uniform_sampler_helper_keeps_op8_authority_and_gpu_execution(self) -> None:
        path, kind, owner = oracle.UNIFORM_RANDOM_BACKEND_FINAL_FORMS[0]
        self.assertEqual(owner, "NUMERIC_DEVICE_HELPERS")
        self.assertNotIn((path, kind, owner), oracle.TYPED_NONNUMERIC_BACKEND_FINAL_FORMS)
        self.assertTrue(oracle.owner_module_final_form(kind, path, owner))
        forms = oracle.coverage_manifest()["source_inventory"]["owner_module_final_forms"]
        self.assertIn({"kind": kind, "owner": owner}, forms[path])
        self.assertTrue(any(
            "gpu_correctness" in probe["command"]
            for probe in oracle.hardware_probe_manifest()
            if probe["lane"] == "hip"
        ))

    def test_direct_arithmetic_hip_owners_require_exact_execution_controls(self) -> None:
        manifest = oracle.coverage_manifest()
        forms = manifest["source_inventory"]["owner_module_final_forms"]
        for path, kind, owner in oracle.DIRECT_ARITHMETIC_BACKEND_FINAL_FORMS:
            self.assertTrue(oracle.owner_module_final_form(kind, path, owner))
            self.assertFalse(oracle.owner_module_final_form(kind, path, owner + "_unchecked"))
            self.assertIn({"kind": kind, "owner": owner}, forms[path])
        commands = [" ".join(leg.argv) for leg in oracle.phase0_legs()]
        self.assertTrue(any(
            "chelis-backend-hip" in command
            and "codegen_structure" in command
            and "direct_extrema_and_adjoint_emit_bit_preserving_kernels" in command
            and "direct_checked_signed_sub_emits_exact_always_on_trap_channel" in command
            and "direct_narrow_float_arithmetic_emits_f32_compute_and_raw_selection" in command
            and "direct_and_fused_wide_float_subtraction_emit_canonical_nan_finalization" in command
            for command in commands
        ))
        self.assertTrue(any(
            probe["lane"] == "hip-direct-arithmetic"
            and "gpu_correctness" in probe["command"]
            and "direct_" in probe["command"]
            for probe in oracle.hardware_probe_manifest()
        ))

    def test_utf8_string_byte_boundaries_are_exact_final_forms(self) -> None:
        manifest = oracle.coverage_manifest()
        forms = manifest["source_inventory"]["owner_module_final_forms"]
        for path, kind, owner in oracle.UTF8_STRING_FINAL_FORMS:
            self.assertTrue(oracle.owner_module_final_form(kind, path, owner))
            self.assertFalse(oracle.owner_module_final_form(kind, path, owner + "_unchecked"))
            self.assertIn({"kind": kind, "owner": owner}, forms[path])
        self.assertFalse(oracle.owner_module_final_form(
            "raw-element-pointer",
            "crates/chelis-runtime/include/chelis_runtime.h",
            "chelis_string_data",
        ))

    def test_checked_snapshot_observation_and_allocation_have_execution_receipts(self) -> None:
        commands = [leg.argv for leg in oracle.phase0_legs()]
        for profile in ((), ("--release",)):
            self.assertTrue(any("chelis-runtime" in command and "checked_c_alloc_like" in command and ("--release" in command) == bool(profile) for command in commands))
        self.assertTrue(any("chelis-backend-c" in command and "checked_c_snapshot_metadata" in command for command in commands))
        self.assertTrue(any("exec_compile" in command and "test(checked_snapshot_)" in command for command in commands))
        self.assertTrue(any("runtime_extent_slice_a" in command and "test(vmap_shape_bound_with_concrete_batch_emits_c_without_to_end_ice)" in command for command in commands))

    def test_checked_c_movement_has_delegation_mutations_and_native_execution(self) -> None:
        commands = [leg.argv for leg in oracle.phase0_legs()]
        self.assertTrue(any("chelis-backend-c" in command and "checked_c_movement" in command for command in commands))
        for selector in (
            "test(checked_c_movement_)",
            "test(a_local_class_guards)",
            "test(a_literal_claim_on_a_symbolic_input)",
            "test(numeric_local_extent_claims)",
        ):
            self.assertTrue(any("exec_compile" in command and any(selector in arg.split(" | ") for arg in command) for command in commands))
        self.assertTrue(any("chelis-backend-c" in command and "checked_c_movement_plans" in command for command in commands))
        self.assertTrue(any("chelis-runtime" in command and "checked_c_movement_plans" in command for command in commands))
        self.assertTrue(any("chelis-ir" in command and "movement_expansion_kind" in command for command in commands))
        self.assertTrue(any("chelis-cli" in command and "issue_616_runtime_movement_c_parity" in command for command in commands))

    def test_checked_json_scratch_requires_lifetime_controls_and_native_execution(self) -> None:
        commands = [leg.argv for leg in oracle.phase0_legs()]
        self.assertTrue(any("chelis-backend-c" in command and "checked_c_json_scratch" in command for command in commands))
        self.assertTrue(any("issue_1314_json_bigint" in command and "test(=json_object_serialization_is_recursive_canonical_unicode_order_in_eval_and_c)" in command for command in commands))

        self.assertTrue(any("issue_1314_json_bigint_ledger" in command and "ownership-ledger" in command and "test(=json_scratch_execution_detects_skipped_cleanup)" in command for command in commands))

    def test_checked_literals_require_delegation_and_native_execution(self) -> None:
        commands = [leg.argv for leg in oracle.phase0_legs()]
        self.assertTrue(any("chelis-backend-c" in command and "checked_c_host_metadata" in command for command in commands))
        self.assertTrue(any("exec_compile" in command and "test(checked_literals_)" in command for command in commands))

    def test_checked_windows_require_source_controls_and_native_execution(self) -> None:
        commands = [leg.argv for leg in oracle.phase0_legs()]
        self.assertTrue(any("chelis-backend-c" in command and "checked_c_window" in command for command in commands))
        self.assertTrue(any("exec_compile" in command and any("test(checked_windows_)" in arg for arg in command) for command in commands))

    def test_checked_c_blas_has_submission_native_and_vendor_controls(self) -> None:
        commands = [leg.argv for leg in oracle.phase0_legs()]
        self.assertTrue(any("chelis-backend-c" in command and "checked_c_blas" in command for command in commands))
        for selector in ("test(checked_blas_)", "test(blas_vendor_dimension_contract)"):
            self.assertTrue(any("exec_compile" in command and any(selector in arg for arg in command) for command in commands))

    def test_checked_c_sparse_has_delegation_native_and_host_execution(self) -> None:
        commands = [leg.argv for leg in oracle.phase0_legs()]
        self.assertTrue(any("chelis-backend-c" in command and "checked_c_sparse" in command for command in commands))
        self.assertTrue(any("exec_compile" in command and any("test(checked_c_sparse_)" in arg for arg in command) for command in commands))
        self.assertTrue(any("chelis-cli" in command and "cross_library_sparse_summaries" in command for command in commands))

    def test_checked_c_reduction_has_delegation_native_and_example_execution(self) -> None:
        commands = [leg.argv for leg in oracle.phase0_legs()]
        self.assertTrue(any("chelis-backend-c" in command and "checked_c_reduction" in command for command in commands))
        self.assertTrue(any("exec_compile" in command and any("test(checked_c_reduction_)" in arg for arg in command) for command in commands))
        self.assertTrue(any(any("test(parity_count_bool_axes)" in arg for arg in command) for command in commands))

    def test_release_reproducers_and_landed_receipts_are_named(self) -> None:
        commands = [
            entry["command"] for entry in oracle.coverage_manifest()["release_reproducers"]
        ]
        self.assertTrue(
            any("issue_888_capacity_collision" in command for command in commands),
            commands,
        )
        self.assertTrue(any("exact_tagged_c_abi" in command for command in commands), commands)
        self.assertTrue(
            any("chelis-repr-inventory" in command for command in commands), commands
        )
        # The reproducers that hide in a debug profile must run in release.
        for command in commands:
            if ("chelis-runtime" in command or "chelis-ir" in command) and "--release" not in command:
                self.assertIn(command.replace("nextest run", "nextest run --release"), commands)

    def test_hardware_manifest_cannot_misreport_ignored_tests_as_executed(self) -> None:
        for probe in oracle.hardware_probe_manifest():
            self.assertEqual(probe["status"], "manual-required")
            self.assertIn("--ignored", probe["command"])

    def test_the_manifest_records_the_closed_universe_rule(self) -> None:
        universe = oracle.coverage_manifest()["source_inventory"]["universe"]
        self.assertEqual(set(universe), {"roots", "closure_rule"})
        self.assertEqual(universe["roots"], list(oracle.INVENTORY_ROOTS))
        self.assertEqual(
            universe["closure_rule"],
            "the scanned sources are exactly the files on disk under the "
            "digest-bound roots; a new file is scanned without registration, "
            "a frozen root directory that departed fails, and a file a "
            "path-keyed registration names that left the universe fails "
            "until it is retired",
        )

    def test_no_libclang_or_configuration_enumeration_remains(self) -> None:
        # The front end is a `clang` subprocess reading one fixed configuration
        # per header; a libclang binding or a preprocessor-configuration
        # product is the earlier, undischargeable design and must not return.
        source = Path(oracle.__file__).read_text(encoding="utf-8").lower()
        for banned in (
            "libclang",
            "clang.cindex",
            "preprocessor configuration",
            "cartesian",
        ):
            self.assertNotIn(banned, source)

    def test_phase_index_names_the_authoritative_continuous_command(self) -> None:
        index = (REPO_ROOT / "docs/phase_oracles.md").read_text(encoding="utf-8")
        self.assertIn("scripts/runtime_representation_oracle.py --phase 0", index)
        self.assertIn("RUNTIME REPRESENTATION PHASE 0: PASS", Path(oracle.__file__).read_text())

    @mock.patch("runtime_representation_oracle.subprocess.run")
    def test_runner_stops_at_the_first_failed_leg(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess(args=(), returncode=1)
        with mock.patch.object(oracle, "validate_phase0_inventory"):
            with self.assertRaisesRegex(oracle.OracleFailure, "failed with exit 1"):
                oracle.run_phase0(run_mutations=False)
        self.assertEqual(run.call_count, 1)

    def test_runner_rejects_a_manifest_that_does_not_match_its_configuration(
        self,
    ) -> None:
        drifted = oracle.coverage_manifest()
        drifted["release_reproducers"] = drifted["release_reproducers"][:-1]
        with (
            mock.patch.object(oracle, "validate_phase0_inventory"),
            mock.patch.object(oracle, "coverage_manifest", return_value=drifted),
            mock.patch.object(oracle, "_run_leg") as run_leg,
        ):
            with self.assertRaisesRegex(oracle.OracleFailure, "coverage manifest drifted"):
                oracle.run_phase0(run_mutations=False)
        run_leg.assert_not_called()

    def test_runner_executes_every_manifested_mutation_and_reproducer(self) -> None:
        probes = oracle.phase0_mutation_probes()
        legs = oracle.phase0_legs()
        self.assertEqual(len(probes), 44)
        self.assertEqual(
            oracle.load_baseline()["source_inventory"]["mutations"],
            oracle.frozen_mutation_rows(probes),
        )
        with (
            mock.patch.object(oracle, "validate_phase0_inventory"),
            mock.patch.object(oracle, "_expect_mutation_rejected") as reject,
            mock.patch.object(oracle, "_run_leg") as run_leg,
        ):
            oracle.run_phase0()
        self.assertEqual(
            [call.args[0] for call in reject.call_args_list],
            list(probes),
        )
        self.assertEqual(
            [call.args[0] for call in run_leg.call_args_list],
            list(legs),
        )


if __name__ == "__main__":
    unittest.main()


class RedTeamRegressionTests(unittest.TestCase):
    """Regressions from the first red-team round on the rebuilt architecture."""

    def test_a_subdirectory_of_a_root_is_a_candidate(self) -> None:
        # Single-level globs did not see a file cargo compiles from a
        # subdirectory, so the closure check could be evaded by placing a seam
        # one directory down.
        for root in oracle.INVENTORY_ROOTS:
            # A root is recursive, or it names a crate's one build script
            # exactly; a single-level directory glob is the evadable shape.
            self.assertTrue(
                "**" in root or root.endswith("/build.rs"),
                f"inventory root is neither recursive nor a build script: {root}",
            )

    def test_a_stale_active_debt_sample_fails(self) -> None:
        baseline = oracle.load_baseline()
        rows = oracle.inventory_rows(REPO_ROOT)
        mutated = json.loads(json.dumps(baseline))
        mutated["active_debt"][0]["sample"] = "THIS SAMPLE IS A LIE"
        with self.assertRaisesRegex(oracle.OracleFailure, "sample is stale"):
            oracle.validate_baseline(mutated, rows)

    def test_the_c_front_end_has_witnesses_for_every_mis_modelled_form(self) -> None:
        # Each of these is a declaration form that carried a seam past the
        # hand-written token walk in one red-team round. The compiler-backed
        # reader closes them by construction; the witnesses keep it that way.
        witnesses = {probe.witness_id for probe in oracle.phase0_mutation_probes()}
        for expected in (
            "phase0.mutate_c_public_element_pointer_export",
            "phase0.mutate_c_body_direct_data_access",
            "phase0.mutate_c_extern_element_data",
            "phase0.mutate_c_non_descriptor_struct_field",
            "phase0.mutate_c_tagged_struct_field",
            "phase0.mutate_c_union_field",
            "phase0.mutate_c_macro_typed_carrier",
            "phase0.mutate_c_multi_declarator_data",
            "phase0.mutate_c_enum_width",
            "phase0.mutate_objc_element_pointer_parameter",
            "phase0.mutate_unknown_c_arithmetic_spelling",
            "phase0.mutate_c_carrier_in_simd_arm",
            "phase0.mutate_c_undeclared_conditional",
            "phase0.mutate_objc_method_carrier",
            "phase0.mutate_c_unclassified_cast_spelling",
            "phase0.mutate_c_pointer_to_element_array",
            "phase0.mutate_c_include_outside_universe",
            "phase0.mutate_c_include_in_dead_arm",
            "phase0.mutate_objc_block_parameter",
            "phase0.mutate_c_sizeof_in_array_bound",
            "phase0.mutate_rust_cast_turbofish",
            "phase0.mutate_c_int8_element_pointer",
            "phase0.mutate_c_elifdef_arm",
        ):
            self.assertIn(expected, witnesses)

    def test_a_multi_declarator_witness_demands_every_owner(self) -> None:
        probes = {probe.witness_id: probe for probe in oracle.phase0_mutation_probes()}
        pair = probes["phase0.mutate_c_multi_declarator_data"]
        self.assertEqual(len(pair.expected_owners), 2)
        manifest = {
            entry["witness_id"]: entry
            for entry in oracle.coverage_manifest()["source_inventory"]["mutations"]
        }
        self.assertEqual(
            manifest["phase0.mutate_c_multi_declarator_data"]["expected_owners"],
            list(pair.expected_owners),
        )

    def test_build_scripts_are_inside_the_universe(self) -> None:
        # A build script is compiled by cargo like any other source, so a root
        # that cannot see one is a closure hole.
        self.assertIn("crates/chelis-runtime/build.rs", oracle.inventory_sources(REPO_ROOT))
        self.assertTrue(
            any(root.endswith("build.rs") for root in oracle.INVENTORY_ROOTS),
            oracle.INVENTORY_ROOTS,
        )

    def test_the_closure_check_has_a_subdirectory_witness(self) -> None:
        paths = {probe.path.as_posix() for probe in oracle.phase0_mutation_probes()}
        self.assertTrue(
            any(path.count("/") > 3 and path.endswith("mod.rs") for path in paths),
            f"no subdirectory closure witness among {sorted(paths)}",
        )
