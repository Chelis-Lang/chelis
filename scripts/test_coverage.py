"""Unit tests for `coverage.py`.

Run via: `python3 -m unittest scripts.test_coverage` from repo root, or
`python3 scripts/test_coverage.py`. The `lint-and-unit` CI job discovers
this file through `python -m unittest discover -s scripts -p 'test_*.py'`;
that step is the authoritative acceptance oracle for the coverage tooling.

No test here invokes cargo. Every fixture -- the llvm-cov JSON export,
the workspace manifests, the baseline -- is built in a temporary
directory, so the suite runs in milliseconds on a machine with no
`cargo-llvm-cov` installed.

The cases mirror the `coverage-baseline` spec scenarios one for one, with
the fail-closed pair carrying the most weight: an empty filter result
must be a loud error, and a genuinely zero-covered file must be an
ordinary 0.0% row. Those two look identical in a percentage and must not
be allowed to collapse into each other.
"""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "coverage_tool", here / "coverage.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cov = _load_module()

PROVENANCE = {
    "package": "chelis-ir",
    "runner": "nextest",
    "recorded_at": "2026-07-27T00:00:00Z",
    "rustc_version": "1.90.0",
    "host_triple": "x86_64-unknown-linux-gnu",
    "llvm_version": "20.1.2",
}


def _export(files: list[tuple[str, int, int]]) -> str:
    """Build a minimal llvm-cov JSON export. Each file is
    `(filename, covered_regions, total_regions)`."""
    return json.dumps(
        {
            "type": "llvm.coverage.json.export",
            "version": "2.0.1",
            "data": [
                {
                    "files": [
                        {
                            "filename": name,
                            "summary": {
                                "regions": {
                                    "covered": covered,
                                    "count": total,
                                    "notcovered": total - covered,
                                    "percent": 0.0,
                                }
                            },
                        }
                        for name, covered, total in files
                    ]
                }
            ],
        }
    )


def _write_export(tmp: Path, files: list[tuple[str, int, int]]) -> Path:
    path = tmp / "export.json"
    path.write_text(_export(files))
    return path


def _baseline(files: dict[str, tuple[int, int]], **provenance_overrides) -> str:
    provenance = dict(PROVENANCE)
    provenance.update(provenance_overrides)
    return json.dumps(
        {
            "provenance": provenance,
            "files": {
                key: {"covered": c, "total": t} for key, (c, t) in files.items()
            },
        }
    )


def _workspace(root: Path, members: dict[str, str], with_src=("chelis-ir",)):
    """Create a fixture workspace. `members` maps repo-relative member
    directory -> package name."""
    member_list = ", ".join(f'"{d}"' for d in members)
    (root / "Cargo.toml").write_text(
        f"[workspace]\nmembers = [{member_list}]\n"
    )
    for directory, name in members.items():
        member = root / directory
        member.mkdir(parents=True, exist_ok=True)
        (member / "Cargo.toml").write_text(
            f'[package]\nname = "{name}"\nversion = "0.1.0"\n'
        )
        if name in with_src:
            (member / "src").mkdir(exist_ok=True)
    return root


class ParseExportTests(unittest.TestCase):
    def test_parses_region_counts_per_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_export(
                Path(tmp), [("/repo/crates/chelis-ir/src/lower.rs", 10, 40)]
            )
            self.assertEqual(
                cov.parse_export(path),
                {"/repo/crates/chelis-ir/src/lower.rs": (10, 40)},
            )

    def test_missing_export_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cov.CoverageError) as cm:
                cov.parse_export(Path(tmp) / "nope.json")
            self.assertIn("not found", str(cm.exception))

    def test_malformed_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "export.json"
            path.write_text('{"data": [{"files": ')
            with self.assertRaises(cov.CoverageError) as cm:
                cov.parse_export(path)
            self.assertIn("malformed", str(cm.exception))

    def test_export_without_data_key_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "export.json"
            path.write_text('{"type": "something-else"}')
            with self.assertRaises(cov.CoverageError) as cm:
                cov.parse_export(path)
            self.assertIn("not an llvm-cov JSON export", str(cm.exception))

    def test_export_with_no_files_raises(self):
        # An instrumented run that measured nothing is a broken run, not
        # a zero-coverage result.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_export(Path(tmp), [])
            with self.assertRaises(cov.CoverageError) as cm:
                cov.parse_export(path)
            self.assertIn("measured nothing", str(cm.exception))

    def test_file_entry_missing_regions_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "export.json"
            path.write_text(
                json.dumps(
                    {"data": [{"files": [{"filename": "a.rs", "summary": {}}]}]}
                )
            )
            with self.assertRaises(cov.CoverageError) as cm:
                cov.parse_export(path)
            self.assertIn("summary.regions", str(cm.exception))


class FilterTests(unittest.TestCase):
    """Spec: dependency sources excluded; filter matching nothing is an
    error; genuine zero coverage is an ordinary row."""

    def _roots(self, tmp: Path):
        src = tmp / "crates" / "chelis-ir" / "src"
        src.mkdir(parents=True)
        return src

    def test_keeps_only_measured_package_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            src = self._roots(root)
            entries = {
                str(src / "lower.rs"): (10, 40),
                str(src / "eval.rs"): (5, 20),
                str(root / "crates" / "chelis-deep" / "src" / "parser.rs"): (
                    99,
                    100,
                ),
                str(root / "crates" / "chelis-surf" / "src" / "lib.rs"): (1, 2),
            }
            kept = cov.filter_to_package(entries, src, root)
            self.assertEqual(
                set(kept),
                {
                    "crates/chelis-ir/src/lower.rs",
                    "crates/chelis-ir/src/eval.rs",
                },
            )

    def test_dependency_regions_do_not_reach_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            src = self._roots(root)
            entries = {
                str(src / "lower.rs"): (10, 40),
                str(root / "crates" / "chelis-deep" / "src" / "p.rs"): (
                    900,
                    1000,
                ),
            }
            kept = cov.filter_to_package(entries, src, root)
            self.assertEqual(sum(t for _, t in kept.values()), 40)

    def test_empty_result_is_an_error_not_zero_percent(self):
        # The core fail-closed rule. A moved source root or a renamed
        # crate must not render as a total coverage collapse.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            src = self._roots(root)
            entries = {
                str(root / "crates" / "chelis-deep" / "src" / "p.rs"): (1, 2)
            }
            with self.assertRaises(cov.CoverageError) as cm:
                cov.filter_to_package(entries, src, root)
            message = str(cm.exception)
            self.assertIn(str(src), message)
            self.assertIn("refusing to report 0.0%", message)

    def test_genuine_zero_coverage_is_kept_as_a_normal_row(self):
        # Distinguishes the case above from a file that really has no
        # covered regions.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            src = self._roots(root)
            kept = cov.filter_to_package(
                {str(src / "cold.rs"): (0, 37)}, src, root
            )
            self.assertEqual(kept, {"crates/chelis-ir/src/cold.rs": (0, 37)})
            self.assertEqual(cov.percent(0, 37), 0.0)

    def test_relative_filenames_are_resolved_against_repo_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            src = self._roots(root)
            kept = cov.filter_to_package(
                {"crates/chelis-ir/src/lower.rs": (3, 4)}, src, root
            )
            self.assertEqual(kept, {"crates/chelis-ir/src/lower.rs": (3, 4)})


class PackageResolutionTests(unittest.TestCase):
    """Spec: unknown package fails loudly; workspace-wide is refused."""

    def test_resolves_member_source_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _workspace(Path(tmp), {"crates/chelis-ir": "chelis-ir"})
            self.assertEqual(
                cov.package_source_root("chelis-ir", root),
                root / "crates" / "chelis-ir" / "src",
            )

    def test_directory_name_is_not_assumed_to_be_package_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _workspace(
                Path(tmp), {"crates/ir": "chelis-ir"}, with_src=("chelis-ir",)
            )
            self.assertEqual(
                cov.package_source_root("chelis-ir", root),
                root / "crates" / "ir" / "src",
            )

    def test_unknown_package_raises_naming_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _workspace(Path(tmp), {"crates/chelis-ir": "chelis-ir"})
            with self.assertRaises(cov.CoverageError) as cm:
                cov.package_source_root("chelis-nope", root)
            self.assertIn("chelis-nope", str(cm.exception))
            self.assertIn("not a workspace member", str(cm.exception))

    def test_member_without_src_directory_raises(self):
        # A member that exists but has no src/ would otherwise reach the
        # filter and be indistinguishable from lost coverage.
        with tempfile.TemporaryDirectory() as tmp:
            root = _workspace(
                Path(tmp), {"crates/chelis-ir": "chelis-ir"}, with_src=()
            )
            with self.assertRaises(cov.CoverageError) as cm:
                cov.package_source_root("chelis-ir", root)
            self.assertIn("no source directory", str(cm.exception))

    def test_empty_package_name_is_refused(self):
        for value in ("", "   "):
            with self.assertRaises(cov.CoverageError) as cm:
                cov.validate_package_argument(value)
            self.assertIn("single named package", str(cm.exception))

    def test_pattern_package_name_is_refused(self):
        for value in ("chelis-*", "chelis-?r", "chelis-[ab]"):
            with self.assertRaises(cov.CoverageError) as cm:
                cov.validate_package_argument(value)
            self.assertIn("single named package", str(cm.exception))


class ToolchainTests(unittest.TestCase):
    """Spec: cargo-llvm-cov absent fails loudly with its install hint."""

    def test_missing_cargo_llvm_cov_raises_with_install_hint(self):
        with mock.patch.object(cov.shutil, "which", return_value=None):
            with self.assertRaises(cov.CoverageError) as cm:
                cov.require_llvm_cov()
            message = str(cm.exception)
            self.assertIn("cargo-llvm-cov not found", message)
            self.assertIn("devenv", message)

    def test_present_cargo_llvm_cov_passes(self):
        with mock.patch.object(
            cov.shutil, "which", return_value="/nix/store/x/bin/cargo-llvm-cov"
        ):
            cov.require_llvm_cov()


class BaselineTests(unittest.TestCase):
    """Spec: malformed baseline fails loudly; provenance is required."""

    def _write(self, tmp: Path, text: str) -> Path:
        path = Path(tmp) / "baseline.json"
        path.write_text(text)
        return path

    def test_loads_provenance_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp, _baseline({"crates/chelis-ir/src/lower.rs": (10, 40)})
            )
            provenance, counts = cov.load_baseline(path)
            self.assertEqual(provenance["host_triple"], PROVENANCE["host_triple"])
            self.assertEqual(counts, {"crates/chelis-ir/src/lower.rs": (10, 40)})

    def test_missing_baseline_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cov.CoverageError) as cm:
                cov.load_baseline(Path(tmp) / "absent.json")
            self.assertIn("missing coverage baseline", str(cm.exception))

    def test_malformed_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "{not json")
            with self.assertRaises(cov.CoverageError) as cm:
                cov.load_baseline(path)
            self.assertIn("malformed", str(cm.exception))

    def test_baseline_without_provenance_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp, json.dumps({"files": {"a.rs": {"covered": 1, "total": 2}}})
            )
            with self.assertRaises(cov.CoverageError) as cm:
                cov.load_baseline(path)
            self.assertIn("no `provenance` object", str(cm.exception))

    def test_baseline_with_incomplete_provenance_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial = {k: v for k, v in PROVENANCE.items() if k != "host_triple"}
            path = self._write(
                tmp,
                json.dumps(
                    {
                        "provenance": partial,
                        "files": {"a.rs": {"covered": 1, "total": 2}},
                    }
                ),
            )
            with self.assertRaises(cov.CoverageError) as cm:
                cov.load_baseline(path)
            self.assertIn("host_triple", str(cm.exception))

    def test_baseline_with_corrupt_counts_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                json.dumps(
                    {
                        "provenance": PROVENANCE,
                        "files": {"a.rs": {"covered": "lots", "total": 2}},
                    }
                ),
            )
            with self.assertRaises(cov.CoverageError) as cm:
                cov.load_baseline(path)
            self.assertIn("integer", str(cm.exception))

    def test_empty_files_map_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp, json.dumps({"provenance": PROVENANCE, "files": {}})
            )
            with self.assertRaises(cov.CoverageError) as cm:
                cov.load_baseline(path)
            self.assertIn("non-empty `files`", str(cm.exception))

    def test_round_trips_through_write_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "written.json"
            counts = {"crates/chelis-ir/src/lower.rs": (10, 40)}
            cov.write_baseline(path, PROVENANCE, counts)
            provenance, loaded = cov.load_baseline(path)
            self.assertEqual(loaded, counts)
            self.assertEqual(provenance["runner"], "nextest")


class EnvironmentComparabilityTests(unittest.TestCase):
    """Spec: a cross-environment comparison warns, labels, and exits 0."""

    def test_same_environment_is_comparable(self):
        self.assertTrue(cov.environment_matches(PROVENANCE, dict(PROVENANCE)))

    def test_different_host_is_not_comparable(self):
        current = dict(PROVENANCE, host_triple="aarch64-apple-darwin")
        self.assertFalse(cov.environment_matches(PROVENANCE, current))

    def test_different_rustc_is_not_comparable(self):
        current = dict(PROVENANCE, rustc_version="1.91.0")
        self.assertFalse(cov.environment_matches(PROVENANCE, current))

    def test_cross_environment_summary_carries_a_warning(self):
        counts = {"crates/chelis-ir/src/lower.rs": (10, 40)}
        text = cov.render_summary(
            counts, PROVENANCE, baseline=counts, comparable=False
        )
        self.assertIn("CROSS-ENVIRONMENT", text)

    def test_same_environment_summary_has_no_warning(self):
        counts = {"crates/chelis-ir/src/lower.rs": (10, 40)}
        text = cov.render_summary(
            counts, PROVENANCE, baseline=counts, comparable=True
        )
        self.assertNotIn("CROSS-ENVIRONMENT", text)


class SummaryRenderingTests(unittest.TestCase):
    def test_sorted_by_uncovered_regions_descending(self):
        counts = {
            "a.rs": (90, 100),
            "b.rs": (0, 500),
            "c.rs": (5, 10),
        }
        text = cov.render_summary(counts, PROVENANCE)
        rows = [line for line in text.splitlines() if line.endswith(".rs")]
        self.assertEqual(
            [row.split()[-1] for row in rows], ["b.rs", "a.rs", "c.rs"]
        )

    def test_reports_totals_and_environment(self):
        counts = {"a.rs": (1, 4)}
        text = cov.render_summary(counts, PROVENANCE)
        self.assertIn("25.0%", text)
        self.assertIn("1/4 regions", text)
        self.assertIn("x86_64-unknown-linux-gnu", text)

    def test_new_and_gone_files_are_marked(self):
        text = cov.render_summary(
            {"new.rs": (1, 2)}, PROVENANCE, baseline={"gone.rs": (1, 2)}
        )
        self.assertIn("(new)", text)
        self.assertIn("(gone)", text)

    def test_file_with_no_regions_is_full_coverage(self):
        self.assertEqual(cov.percent(0, 0), 100.0)


class MainTests(unittest.TestCase):
    """End-to-end through `main`, with cargo stubbed out via `--export`."""

    def _run(self, argv, root=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            if root is None:
                code = cov.main(argv)
            else:
                with mock.patch.object(cov, "REPO_ROOT", root), mock.patch.object(
                    cov, "SCRIPTS_DIR", root / "scripts"
                ), mock.patch.object(
                    cov, "build_provenance", return_value=dict(PROVENANCE)
                ):
                    code = cov.main(argv)
        return code, out.getvalue(), err.getvalue()

    def _fixture(self, tmp):
        root = Path(tmp).resolve()
        _workspace(root, {"crates/chelis-ir": "chelis-ir"})
        (root / "scripts").mkdir()
        export = _write_export(
            root,
            [
                (str(root / "crates/chelis-ir/src/lower.rs"), 10, 40),
                (str(root / "crates/chelis-ir/src/eval.rs"), 0, 20),
                (str(root / "crates/chelis-deep/src/parser.rs"), 99, 100),
            ],
        )
        return root, export

    def test_valid_inputs_produce_a_summary_and_exit_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, export = self._fixture(tmp)
            code, out, _ = self._run(
                ["chelis-ir", "--export", str(export)], root
            )
            self.assertEqual(code, 0)
            self.assertIn("crates/chelis-ir/src/lower.rs", out)
            self.assertNotIn("chelis-deep", out)

    def test_missing_package_argument_exits_nonzero(self):
        code, _, err = self._run([])
        self.assertEqual(code, 2)
        self.assertIn("single named package", err)

    def test_workspace_flag_is_refused(self):
        for flag in ("--workspace", "--all"):
            code, _, err = self._run([flag])
            self.assertEqual(code, 2)
            self.assertIn("never the whole workspace", err)

    def test_unknown_package_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, export = self._fixture(tmp)
            code, _, err = self._run(
                ["chelis-nope", "--export", str(export)], root
            )
            self.assertEqual(code, 2)
            self.assertIn("chelis-nope", err)

    def test_broken_filter_exits_nonzero_without_printing_a_percentage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            _workspace(root, {"crates/chelis-ir": "chelis-ir"})
            (root / "scripts").mkdir()
            export = _write_export(
                root, [(str(root / "crates/chelis-deep/src/p.rs"), 1, 2)]
            )
            code, out, err = self._run(
                ["chelis-ir", "--export", str(export)], root
            )
            self.assertEqual(code, 2)
            self.assertIn("refusing to report 0.0%", err)
            self.assertNotIn("region coverage", out)

    def test_coverage_decrease_still_exits_zero(self):
        # No percentage is a threshold: a drop is reported, never failed.
        with tempfile.TemporaryDirectory() as tmp:
            root, export = self._fixture(tmp)
            baseline = root / "scripts" / "coverage_baseline_chelis_ir.json"
            baseline.write_text(
                _baseline(
                    {
                        "crates/chelis-ir/src/lower.rs": (39, 40),
                        "crates/chelis-ir/src/eval.rs": (19, 20),
                    }
                )
            )
            code, out, _ = self._run(
                ["chelis-ir", "--export", str(export)], root
            )
            self.assertEqual(code, 0)
            self.assertIn("-72.5 pts", out)

    def test_malformed_baseline_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, export = self._fixture(tmp)
            bad = root / "scripts" / "coverage_baseline_chelis_ir.json"
            bad.write_text("{not json")
            code, _, err = self._run(
                ["chelis-ir", "--export", str(export)], root
            )
            self.assertEqual(code, 2)
            self.assertIn("malformed coverage baseline", err)

    def test_absent_default_baseline_is_not_an_error(self):
        # Before the baseline is recorded the tool must still report.
        with tempfile.TemporaryDirectory() as tmp:
            root, export = self._fixture(tmp)
            code, out, _ = self._run(
                ["chelis-ir", "--export", str(export)], root
            )
            self.assertEqual(code, 0)
            self.assertIn("region coverage", out)

    def test_explicitly_named_absent_baseline_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, export = self._fixture(tmp)
            code, _, err = self._run(
                [
                    "chelis-ir",
                    "--export",
                    str(export),
                    "--baseline",
                    str(root / "scripts" / "absent.json"),
                ],
                root,
            )
            self.assertEqual(code, 2)
            self.assertIn("missing coverage baseline", err)

    def test_a_normal_run_does_not_rewrite_the_baseline(self):
        # Spec: no workflow rewrites the baseline. Only --update-baseline
        # writes, so a scheduled run cannot launder a coverage loss in.
        with tempfile.TemporaryDirectory() as tmp:
            root, export = self._fixture(tmp)
            baseline = root / "scripts" / "coverage_baseline_chelis_ir.json"
            original = _baseline({"crates/chelis-ir/src/lower.rs": (39, 40)})
            baseline.write_text(original)
            code, _, _ = self._run(
                ["chelis-ir", "--export", str(export)], root
            )
            self.assertEqual(code, 0)
            self.assertEqual(baseline.read_text(), original)

    def test_update_baseline_writes_the_default_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, export = self._fixture(tmp)
            code, out, _ = self._run(
                ["chelis-ir", "--export", str(export), "--update-baseline"],
                root,
            )
            self.assertEqual(code, 0)
            written = root / "scripts" / "coverage_baseline_chelis_ir.json"
            self.assertTrue(written.is_file())
            _, counts = cov.load_baseline(written)
            self.assertEqual(counts["crates/chelis-ir/src/eval.rs"], (0, 20))

    def test_cross_environment_comparison_warns_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, export = self._fixture(tmp)
            baseline = root / "scripts" / "coverage_baseline_chelis_ir.json"
            baseline.write_text(
                _baseline(
                    {"crates/chelis-ir/src/lower.rs": (10, 40)},
                    host_triple="aarch64-apple-darwin",
                )
            )
            code, out, _ = self._run(
                ["chelis-ir", "--export", str(export)], root
            )
            self.assertEqual(code, 0)
            self.assertIn("CROSS-ENVIRONMENT", out)


class BaselinePathTests(unittest.TestCase):
    def test_package_name_maps_to_underscored_baseline_file(self):
        self.assertEqual(
            cov.baseline_path_for("chelis-ir", Path("/s")).name,
            "coverage_baseline_chelis_ir.json",
        )


if __name__ == "__main__":
    unittest.main()
