"""Tests for the runtime-archive lookup guard (chelis#1354)."""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

if __package__:
    from . import check_runtime_archive_lookups as guard
else:
    import check_runtime_archive_lookups as guard

OWNER_FILES = {
    "crates/chelis-runtime-bundle/src/lib.rs": 'const RUNTIME_DIR_VARIABLE: &str = "CHELIS_RUNTIME_DIR";\n',
    "crates/chelis-runtime-bundle-macro/src/locate.rs": 'name.starts_with("libchelis_runtime")\n',
}
LOOKUP = "archive = next((target / 'debug' / 'deps').glob('libchelis_runtime-*.a'))\n"


class PlantedTree:
    """A temporary tree scanned with an explicit file list, owners included."""

    def __init__(self, test: unittest.TestCase, files: dict[str, str]) -> None:
        directory = tempfile.TemporaryDirectory()
        test.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.files = {**OWNER_FILES, **files}
        for name, text in self.files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def check(self, rows: tuple[guard.Row, ...] = ()) -> list[str]:
        return guard.check(self.root, rows, sorted(self.files))


def row(path: str, pattern: str, count: int = 1, **fields: str) -> guard.Row:
    values = {"disposition": "not-lookup", "reason": "reviewed", **fields}
    return guard.Row(path, pattern, count, **values)


class RepositoryTests(unittest.TestCase):
    def test_repository_has_only_reviewed_runtime_lookups(self) -> None:
        self.assertEqual(guard.check(), [])


class PatternTests(unittest.TestCase):
    def test_every_pattern_requires_its_token_and_trips_on_its_sample(self) -> None:
        for pattern in guard.PATTERNS:
            with self.subTest(pattern=pattern.name):
                self.assertIn(pattern.token, pattern.sample)
                self.assertIsNotNone(pattern.regex.search(pattern.sample))
                self.assertIsNone(
                    pattern.regex.search(pattern.sample.replace(pattern.token, "x"))
                )
                failures = PlantedTree(self, {"tools/sample.sh": pattern.sample}).check()
                self.assertTrue(
                    any(failure.startswith(f"{pattern.name}:") for failure in failures),
                    failures,
                )

    def test_planted_directory_scans_fail(self) -> None:
        scans = {
            "scripts/planted.py": LOOKUP,
            "crates/planted/tests/scan.rs": (
                "for entry in std::fs::read_dir(deps)? {\n"
                "    let name = entry?.file_name();\n"
                '    if name.to_string_lossy().starts_with("libchelis_runtime-") {}\n'
                "}\n"
            ),
            "nix/planted.nix": 'runtime = builtins.getEnv "CHELIS_RUNTIME_DIR";\n',
            ".github/workflows/planted.yml": "run: cp target/release/libchelis_runtime.a dist/\n",
            "crates/planted/src/link.rs": '.args(["-L", dir, "-l", "chelis_runtime"])\n',
            "scripts/nested.py": 'if message["target"]["name"] == "chelis_runtime":\n',
        }
        for name, text in scans.items():
            with self.subTest(file=name):
                failures = PlantedTree(self, {name: text}).check()
                self.assertTrue(any(name in failure for failure in failures), failures)

    def test_a_lookup_split_across_lines_is_found(self) -> None:
        failures = PlantedTree(
            self, {"crates/planted/src/link.rs": '.args([\n    "-l",\n    "chelis_runtime",\n])\n'}
        ).check()
        self.assertTrue(any(failure.startswith("linker-search:") for failure in failures))

    def test_staged_uses_and_rejections_are_not_lookups(self) -> None:
        # Real shapes of code that uses the staged archive by path or rejects a lookup.
        corpus = {
            "crates/c/tests/link.rs": (
                'cmd.arg(out_dir.join("libchelis_runtime.a"));\n'
                '.env("CHELIS_RUNTIME_DIR", dir.path())\n'
                '.env_remove("CHELIS_RUNTIME_DIR")\n'
                '.stderr(predicate::str::contains("Unset CHELIS_RUNTIME_DIR"));\n'
                'let staged = chelis_runtime_bundle::stage(dir.path()).unwrap();\n'
            ),
            "scripts/stage.py": (
                'runtime = output / "libchelis_runtime.a"\n'
                'environment.pop("CHELIS_RUNTIME_DIR", None)\n'
                "for name in (\"CHELIS_RUNTIME_DIR\", \"CHELIS_CC\"):\n"
            ),
            ".github/workflows/release.yml": (
                'run: cp "$RUNTIME_EXPORT/libchelis_runtime.a" "$DIST/lib/"\n'
            ),
            "nix/packages.nix": '$out/bin/chelis runtime export "$runtime/lib"\n',
        }
        self.assertEqual(PlantedTree(self, corpus).check(), [])

    def test_the_owner_crates_are_exempt_and_lookalikes_are_not(self) -> None:
        self.assertEqual(
            PlantedTree(self, {"crates/chelis-runtime-bundle/src/scan.rs": LOOKUP}).check(), []
        )
        failures = PlantedTree(
            self, {"crates/chelis-runtime-bundle-extra/src/scan.rs": LOOKUP}
        ).check()
        self.assertTrue(any("chelis-runtime-bundle-extra" in failure for failure in failures))

    def test_a_missing_owner_is_a_stale_exemption(self) -> None:
        tree = PlantedTree(self, {})
        failures = guard.check(tree.root, (), ["crates/chelis-runtime-bundle/src/lib.rs"])
        self.assertTrue(
            any("crates/chelis-runtime-bundle-macro/" in failure for failure in failures)
        )


class RowTests(unittest.TestCase):
    def test_a_row_must_match_the_exact_count(self) -> None:
        name = "scripts/planted.py"
        once = PlantedTree(self, {name: LOOKUP})
        self.assertEqual(once.check((row(name, "archive-name-match"), row(name, "build-tree-archive"))), [])
        stale = once.check(
            (row(name, "archive-name-match", 2), row(name, "build-tree-archive"))
        )
        self.assertTrue(any(failure.startswith("stale row") for failure in stale), stale)
        twice = PlantedTree(self, {name: LOOKUP * 2}).check(
            (row(name, "archive-name-match"), row(name, "build-tree-archive", 2))
        )
        self.assertTrue(
            any(failure.startswith("archive-name-match: 2 match(es)") for failure in twice),
            twice,
        )

    def test_a_row_for_a_removed_match_is_stale(self) -> None:
        failures = PlantedTree(self, {"scripts/clean.py": "pass\n"}).check(
            (row("scripts/clean.py", "linker-search"),)
        )
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0].startswith("stale row scripts/clean.py"))

    def test_malformed_rows_are_rejected(self) -> None:
        name = "scripts/planted.py"
        cases = {
            "duplicated": (row(name, "linker-search"), row(name, "linker-search")),
            "unknown pattern": (row(name, "directory-walk"),),
            "admits no reviewed rows": (row(name, "runtime-lib-variable"),),
            "exempts": (row("crates/chelis-runtime-bundle/src/lib.rs", "linker-search"),),
            "at least one": (row(name, "linker-search", 0),),
            "disposition": (row(name, "linker-search", disposition="allowed"),),
            "no reason": (row(name, "linker-search", reason=" "),),
            "tracking issue": (row(name, "linker-search", disposition="lookup"),),
        }
        for message, rows in cases.items():
            with self.subTest(case=message):
                self.assertTrue(
                    any(message in error for error in guard.row_errors(rows)),
                    guard.row_errors(rows),
                )
        tracked = row(name, "linker-search", disposition="lookup", tracking="chelis#1354")
        self.assertEqual(guard.row_errors((tracked,)), [])

    def test_the_reviewed_rows_are_well_formed(self) -> None:
        self.assertEqual(guard.row_errors(guard.REVIEWED), [])


class RepositoryFileTests(unittest.TestCase):
    def test_untracked_files_are_scanned_and_ignored_files_are_not(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        for name, text in {**OWNER_FILES, ".gitignore": "ignored/\n"}.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text)
        (root / "ignored").mkdir()
        (root / "ignored/scan.py").write_text(LOOKUP)
        self.assertEqual(guard.check(root, ()), [])
        (root / "scripts").mkdir()
        (root / "scripts/scan.py").write_text(LOOKUP)
        failures = guard.check(root, ())
        self.assertTrue(any("scripts/scan.py" in failure for failure in failures), failures)


if __name__ == "__main__":
    unittest.main()
