"""Tests for the runtime-archive lookup guard (chelis#1354)."""
from __future__ import annotations

from pathlib import Path
import re
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

# Lookups this repository shipped before chelis#1354 removed them, without the
# sibling lines (an environment read, a literal deps path) that also flagged them.
HISTORICAL = {
    # The CLI selector (`git show a8feaf19b^:crates/chelis-cli/src/main.rs`).
    "crates/planted/src/find_runtime.rs": (
        "fn find_runtime_library() -> Option<PathBuf> {\n"
        '    const LIB_NAME: &str = "libchelis_runtime.a";\n'
        '    const LIB_PREFIX: &str = "libchelis_runtime";\n'
        "    fn find_in_dir(dir: &Path) -> Option<PathBuf> {\n"
        "        for entry in fs::read_dir(dir).ok()?.flatten() {\n"
        "            let name = entry.file_name();\n"
        "            if name.to_str()?.starts_with(LIB_PREFIX) {}\n"
        "        }\n"
        "        None\n"
        "    }\n"
        '    find_in_dir(&exe_dir.join("deps"))\n'
        "}\n"
    ),
    # The Phase 1 oracle's Cargo read (`git show 2cfebb8c0^:scripts/runtime_representation_phase1.py`).
    "scripts/planted_phase1.py": (
        "for line in output.splitlines():\n"
        "    row = load_json(line)\n"
        "    if row.get('reason') != 'compiler-artifact' or row.get('target', {}).get('name') != 'chelis_runtime':\n"
        "        continue\n"
        "    artifacts.extend(Path(name) for name in row['filenames'] if name.endswith('.a'))\n"
    ),
    # The harness helper (`git show 2cfebb8c0^:crates/chelis-backend-c/tests/exec_compile.rs`).
    "crates/planted/tests/exec_compile.rs": (
        'let canonical = target_debug_dir().join("libchelis_runtime.a");\n'
    ),
}

# Lookup idioms in other syntaxes, one per line, each of which must fail.
IDIOMS = (
    'println!("cargo:rustc-link-lib=static=chelis_runtime");',
    '#[link(name = "chelis_runtime")]',
    'cmd.arg("-l").arg("chelis_runtime");',
    'cc main.c -L "$dir" -l chelis_runtime',
    'if name.contains("chelis_runtime") {',
    "find target -name libchelis_runtime.a",
    'archive = next(Path(target).glob("**/libchelis_runtime.a"))',
    'env = os.environ.copy(); runtime = env.get("CHELIS_RUNTIME_DIR")',
    'let dir = var_os("CHELIS_RUNTIME_DIR");',
    'env::vars().find(|(key, _)| key == "CHELIS_RUNTIME_DIR")',
    'archive = Path(target_dir) / profile / "libchelis_runtime.a"',
    'if message["package_id"].startswith("chelis-runtime "):',
    'if row.get("target", {}).get("name") == Some("chelis_runtime")',
    "export RUNTIME=$CHELIS_RUNTIME_DIR/libchelis_runtime.a",
    "runtime: ${{ env.CHELIS_RUNTIME_DIR }}",
)


class PlantedTree:
    """A temporary tree scanned with an explicit file list, owners included."""

    def __init__(self, test: unittest.TestCase, files: dict[str, str | bytes]) -> None:
        directory = tempfile.TemporaryDirectory()
        test.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.files = {**OWNER_FILES, **files}
        for name, text in self.files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(text, bytes):
                path.write_bytes(text)
            else:
                path.write_text(text)

    def check(self, rows: tuple[guard.Row, ...] = ()) -> list[str]:
        return guard.check(self.root, rows, sorted(self.files))


def row(path: str, pattern: str, *lines: str, **fields: str) -> guard.Row:
    values = {"disposition": "not-lookup", "reason": "reviewed", **fields}
    return guard.Row(path, pattern, lines, **values)


def matched_patterns(failures: list[str]) -> set[str]:
    return {failure.split(":", 1)[0] for failure in failures if "unreviewed match" in failure}


class RepositoryTests(unittest.TestCase):
    def test_repository_has_only_reviewed_runtime_lookups(self) -> None:
        self.assertEqual(guard.check(), [])


class PatternTests(unittest.TestCase):
    def test_every_sample_matches_and_needs_a_runtime_token(self) -> None:
        for pattern in guard.PATTERNS:
            for sample in pattern.samples:
                with self.subTest(pattern=pattern.name, sample=sample):
                    self.assertIsNotNone(pattern.regex.search(sample))
                    self.assertTrue(any(token in sample for token in guard.TOKENS))
                    # The scan skips files without a token, so no match may exist without one.
                    stripped = sample
                    for token in guard.TOKENS:
                        stripped = stripped.replace(token, "x")
                    self.assertIsNone(pattern.regex.search(stripped))

    def test_every_alternative_is_needed_by_a_sample(self) -> None:
        for pattern in guard.PATTERNS:
            for index, alternative in enumerate(pattern.alternatives):
                others = pattern.alternatives[:index] + pattern.alternatives[index + 1 :]
                without = re.compile("|".join(others)) if others else None
                with self.subTest(pattern=pattern.name, alternative=alternative):
                    self.assertTrue(
                        any(
                            without is None or without.search(sample) is None
                            for sample in pattern.samples
                        ),
                        "no sample fails without this alternative",
                    )

    def test_every_sample_fails_the_scan_under_its_pattern(self) -> None:
        for pattern in guard.PATTERNS:
            for sample in pattern.samples:
                with self.subTest(pattern=pattern.name, sample=sample):
                    failures = PlantedTree(self, {"tools/sample.sh": sample + "\n"}).check()
                    self.assertIn(pattern.name, matched_patterns(failures), failures)

    def test_lookups_this_repository_shipped_fail(self) -> None:
        for name, text in HISTORICAL.items():
            with self.subTest(file=name):
                failures = PlantedTree(self, {name: text}).check()
                self.assertTrue(any(name in failure for failure in failures), failures)

    def test_lookup_idioms_in_other_syntaxes_fail(self) -> None:
        for idiom in IDIOMS:
            with self.subTest(idiom=idiom):
                failures = PlantedTree(self, {"tools/idiom.rs": idiom + "\n"}).check()
                self.assertTrue(matched_patterns(failures), failures)

    def test_a_lookup_split_across_lines_is_found(self) -> None:
        split = {
            "crates/planted/src/link.rs": '.args([\n    "-l",\n    "chelis_runtime",\n])\n',
            "crates/planted/tests/join.rs": (
                'let archive = root\n    .join("debug")\n    .join("libchelis_runtime.a");\n'
            ),
        }
        for name, text in split.items():
            with self.subTest(file=name):
                self.assertTrue(matched_patterns(PlantedTree(self, {name: text}).check()))

    def test_staged_uses_are_not_lookups(self) -> None:
        # Real shapes of code that links or ships the staged archive by path.
        corpus = {
            "crates/c/tests/link.rs": (
                'cmd.arg(out_dir.join("libchelis_runtime.a"));\n'
                "let staged = chelis_runtime_bundle::stage(dir.path()).unwrap();\n"
                "cc.arg(&staged.archive);\n"
                'let receipt = out.join("chelis_runtime.receipt.json");\n'
                "pub use chelis_runtime::public_headers::PUBLIC_HEADERS;\n"
                'fs::write(dir.join("main.c"), "#include \\"chelis_runtime.h\\"\\n")?;\n'
            ),
            "scripts/stage.py": (
                'runtime = output / "libchelis_runtime.a"\n'
                'HEADERS = ("chelis_runtime.h", "chelis_runtime_views.h")\n'
            ),
            ".github/workflows/release.yml": (
                'run: cp "$RUNTIME_EXPORT/libchelis_runtime.a" "$DIST/lib/"\n'
            ),
            "nix/packages.nix": '$out/bin/chelis runtime export "$runtime/lib"\n',
        }
        self.assertEqual(PlantedTree(self, corpus).check(), [])

    def test_every_mention_of_the_runtime_directory_variable_needs_a_row(self) -> None:
        name = "crates/c/tests/reject.rs"
        text = '.env_remove("CHELIS_RUNTIME_DIR")\n'
        tree = PlantedTree(self, {name: text})
        self.assertEqual(matched_patterns(tree.check()), {"runtime-variable"})
        reviewed = row(name, "runtime-variable", '.env_remove("CHELIS_RUNTIME_DIR")')
        self.assertEqual(tree.check((reviewed,)), [])

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

    def test_an_unreadable_file_fails_by_name(self) -> None:
        name = "scripts/latin1.py"
        failures = PlantedTree(self, {name: b"# caf\xe9 CHELIS_RUNTIME_DIR\n"}).check()
        self.assertTrue(
            any(failure.startswith(f"{name}: cannot be read as UTF-8") for failure in failures),
            failures,
        )


class RowTests(unittest.TestCase):
    NAME = "crates/c/tests/link.rs"
    SEARCH = 'cmd.args(["-L.", "-lchelis_runtime"]);'
    ASSERTION = '.stdout(predicate::str::contains("-lchelis_runtime").not())'

    def test_a_row_pins_the_exact_lines(self) -> None:
        tree = PlantedTree(self, {self.NAME: self.ASSERTION + "\n" + self.ASSERTION + "\n"})
        pinned = row(self.NAME, "linker-search", self.ASSERTION, self.ASSERTION)
        self.assertEqual(tree.check((pinned,)), [])
        once = tree.check((row(self.NAME, "linker-search", self.ASSERTION),))
        self.assertTrue(any("1 unreviewed match(es)" in failure for failure in once), once)

    def test_a_swapped_line_fails_even_at_the_same_count(self) -> None:
        tree = PlantedTree(self, {self.NAME: self.ASSERTION + "\n" + self.SEARCH + "\n"})
        failures = tree.check((row(self.NAME, "linker-search", self.ASSERTION, self.ASSERTION),))
        self.assertTrue(any(self.SEARCH in failure for failure in failures), failures)
        self.assertTrue(any(failure.startswith("stale row") for failure in failures), failures)

    def test_a_row_for_a_removed_match_is_stale(self) -> None:
        failures = PlantedTree(self, {"scripts/clean.py": "pass\n"}).check(
            (row("scripts/clean.py", "linker-search", self.SEARCH),)
        )
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0].startswith("stale row scripts/clean.py"))

    def test_malformed_rows_are_rejected(self) -> None:
        name = "scripts/planted.py"
        line = self.SEARCH
        cases = {
            "duplicated": (row(name, "linker-search", line), row(name, "linker-search", line)),
            "unknown pattern": (row(name, "directory-walk", line),),
            "admits no reviewed rows": (row(name, "runtime-lib-variable", line),),
            "exempts": (row("crates/chelis-runtime-bundle/src/lib.rs", "linker-search", line),),
            "at least one line": (row(name, "linker-search"),),
            "not stripped": (row(name, "linker-search", " " + line),),
            "disposition": (row(name, "linker-search", line, disposition="allowed"),),
            "no reason": (row(name, "linker-search", line, reason=" "),),
            "tracking issue": (row(name, "linker-search", line, disposition="lookup"),),
        }
        for message, rows in cases.items():
            with self.subTest(case=message):
                self.assertTrue(
                    any(message in error for error in guard.row_errors(rows)),
                    guard.row_errors(rows),
                )
        tracked = row(name, "linker-search", line, disposition="lookup", tracking="chelis#1354")
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
