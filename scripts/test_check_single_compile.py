"""Tests for the single-compile guard (chelis#2928)."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

if __package__:
    from . import check_single_compile as guard
else:
    import check_single_compile as guard

# Double compiles this repository shipped before chelis#2928 removed them,
# reduced to the lines that build and compile.
SHIPPED = {
    # One helper builds and links (issue_1418_recursive_cast_targets.rs).
    "crates/planted/tests/parity.rs": (
        "fn parity(root: &Path, file: &str) {\n"
        '    let built = run(root, &["build", file, "--target", "c", "--output", "out"]);\n'
        "    success(&built);\n"
        '    assert!(common::link_generated(&root.join("out"), "probe.c", "probe").success());\n'
        '    let native = Command::new(root.join("out").join("probe")).output().unwrap();\n'
        "}\n"
    ),
    # A hand-written compile of the generated C (issue_1753_grad_result_order_cli.rs).
    "crates/planted/tests/order.rs": (
        "fn native_control(source: &str) {\n"
        '    run(&mut cli(temp.path(), &["build", "order.ch", "--target", "c", "--output", "native"]));\n'
        '    let mut cc = Command::new("cc");\n'
        '    cc.current_dir(&native).args(["-O2", "order.c", "libchelis_runtime.a", "-lm", "-o"]);\n'
        "    run(&mut cc);\n"
        "}\n"
    ),
    # The command is the first argument of a same-file helper
    # (issue_1942_declared_operation_bounds.rs).
    "crates/planted/tests/bounds.rs": (
        "fn cli(command: &str, path: &Path, output: &Path) -> Output {\n"
        '    let mut cli = Command::cargo_bin("chelis").unwrap();\n'
        "    cli.arg(command).arg(path);\n"
        '    if command == "build" {\n'
        '        cli.args(["--target", "c", "-o"]).arg(output);\n'
        "    }\n"
        "    cli.output().unwrap()\n"
        "}\n"
        "#[test]\n"
        "fn bounded() {\n"
        '    let built = cli("build", &path, &output_dir);\n'
        '    assert!(common::link_generated(&output_dir, "probe.c", "probe").success());\n'
        "}\n"
    ),
}

# Each must fail: a default build and a compile meet in one function.
DOUBLE = {
    "builder chain": (
        "fn lane() {\n"
        '    Command::cargo_bin("chelis").unwrap().arg("build").arg(&file).assert().success();\n'
        '    let status = StdCommand::new(&toolchain.compiler).arg("main.c").status().unwrap();\n'
        "}\n"
    ),
    "inherited from two helpers": (
        "fn build(path: &Path) {\n"
        '    chelis(&["build", path.to_str().unwrap()]);\n'
        "}\n"
        "fn link(out: &Path) {\n"
        '    assert!(link_generated(out, "main.c", "main").success());\n'
        "}\n"
        "#[test]\n"
        "fn both_lanes() {\n"
        "    build(&path);\n"
        "    link(&out);\n"
        "}\n"
    ),
    "inherited through a chain": (
        "fn build(path: &Path) {\n"
        '    chelis(vec!["build", path.to_str().unwrap()]);\n'
        "}\n"
        "fn prepare(path: &Path) {\n"
        "    build(path);\n"
        "}\n"
        "#[test]\n"
        "fn compiled() {\n"
        "    prepare(&path);\n"
        "    let compiler = chelis_backend_c::toolchain::c_compiler();\n"
        '    Command::new(compiler).args(["main.c", "-o", "main"]).status().unwrap();\n'
        "}\n"
    ),
    "a compiler literal in an expression": (
        "fn lane() {\n"
        '    run(&["build", "main.ch"]);\n'
        '    NativeCommand::new(std::env::var("CC").unwrap_or_else(|_| "cc".into())).status().unwrap();\n'
        "}\n"
    ),
    "a compile that links only the runtime archive": (
        "fn lane() {\n"
        '    run(&["build", "main.ch"]);\n'
        '    Command::new("cc").args(["main.c", "libchelis_runtime.a", "-o", "main"]).status().unwrap();\n'
        "}\n"
    ),
    "a compile inside a closure": (
        "fn lane() {\n"
        '    run(&["build", "main.ch"]);\n'
        "    let link = |out: &Path| common::link_generated(out, \"main.c\", \"main\");\n"
        "    assert!(link(&out).success());\n"
        "}\n"
    ),
}

# Each must pass.
SINGLE = {
    "source-only build compiled by the test": (
        "fn lane() {\n"
        '    run(&["build", "--emit-c", "main.ch", "-o", "out"]);\n'
        '    assert!(common::link_generated(&out, "main.c", "main").success());\n'
        "}\n"
    ),
    "default build run as published": (
        "fn lane() {\n"
        '    run(&["build", "main.ch", "-o", "out"]);\n'
        '    let native = Command::new(out.join("main")).output().unwrap();\n'
        "}\n"
    ),
    "published library linked into a driver": (
        "fn library() {\n"
        '    build(&file, &out).assert().success();\n'
        "    let status = Process::new(toolchain.compiler)\n"
        "        .arg(&driver)\n"
        '        .arg(out.join("libfunctions.a"))\n'
        '        .arg(out.join("libchelis_runtime.a"))\n'
        "        .status()\n"
        "        .unwrap();\n"
        "}\n"
        "fn build(file: &Path, out: &Path) -> Command {\n"
        '    let mut cmd = Command::cargo_bin("chelis").unwrap();\n'
        '    cmd.args(["build", file.to_str().unwrap(), "--output", out.to_str().unwrap()]);\n'
        "    cmd\n"
        "}\n"
    ),
    "source-only build through a helper": (
        "fn cli(command: &str, path: &Path) -> Output {\n"
        '    let mut cli = Command::cargo_bin("chelis").unwrap();\n'
        "    cli.arg(command);\n"
        '    if command == "build" {\n'
        '        cli.arg("--emit-c");\n'
        "    }\n"
        "    cli.arg(path).output().unwrap()\n"
        "}\n"
        "#[test]\n"
        "fn lane() {\n"
        '    cli("build", &path);\n'
        '    assert!(link_generated(&out, "main.c", "main").success());\n'
        "}\n"
    ),
    "a caller appends --emit-c to a default helper": (
        "fn build(file: &Path) -> Command {\n"
        '    let mut cmd = Command::cargo_bin("chelis").unwrap();\n'
        '    cmd.args(["build", file.to_str().unwrap()]);\n'
        "    cmd\n"
        "}\n"
        "#[test]\n"
        "fn lane() {\n"
        '    build(&file).arg("--emit-c").assert().success();\n'
        '    assert!(link_generated(&out, "main.c", "main").success());\n'
        "}\n"
    ),
    "a stage label names build": (
        "fn lane() {\n"
        '    let built = run(&["build", "--emit-c", "main.ch"]);\n'
        "    if !built.status.success() {\n"
        '        return receipt("build", &built);\n'
        "    }\n"
        '    assert!(link_generated(&out, "main.c", "main").success());\n'
        '    let run = Command::new(out.join("main")).output().expect("build");\n'
        "}\n"
        "fn receipt(stage: &str, output: &Output) -> Value {\n"
        "    json!({})\n"
        "}\n"
    ),
    "build and compile in unrelated functions": (
        "fn published() {\n"
        '    run(&["build", "main.ch"]);\n'
        "}\n"
        "fn emitted() {\n"
        '    run(&["build", "--emit-c", "other.ch"]);\n'
        '    assert!(link_generated(&out, "other.c", "other").success());\n'
        "}\n"
    ),
    "build and compile inside literals and comments": (
        "// run(&[\"build\", \"main.ch\"]); link_generated(&out, \"main.c\", \"main\");\n"
        "fn lane() {\n"
        '    let fixture = r#"run(&["build", "main.ch"]); Command::new("cc")"#;\n'
        "    /* link_generated(&out, \"main.c\", \"main\"); /* nested */ */\n"
        "    let quote = '\"';\n"
        "    let apostrophe = '\\'';\n"
        '    run(&["build", "main.ch"]);\n'
        "}\n"
        "fn compile<'a>(out: &'a Path) {\n"
        '    let message = "Command::new(\\"cc\\") \\"build\\"";\n'
        '    Command::new("cc").arg("main.c").status().unwrap();\n'
        "}\n"
    ),
}


class PlantedTree:
    """A temporary tree scanned with an explicit file list."""

    def __init__(self, test: unittest.TestCase, files: dict[str, str]) -> None:
        directory = tempfile.TemporaryDirectory()
        test.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.files = files
        for name, text in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def check(self) -> list[str]:
        return guard.check(self.root, sorted(self.files))


def flagged(failures: list[str]) -> set[str]:
    return {failure.split(": fn ", 1)[1].split(" ", 1)[0] for failure in failures if ": fn " in failure}


class RepositoryTests(unittest.TestCase):
    def test_repository_compiles_each_program_once(self) -> None:
        self.assertEqual(guard.check(), [])


class DoubleCompileTests(unittest.TestCase):
    def test_double_compiles_this_repository_shipped_fail(self) -> None:
        expected = {
            "crates/planted/tests/parity.rs": {"parity"},
            "crates/planted/tests/order.rs": {"native_control"},
            "crates/planted/tests/bounds.rs": {"bounded"},
        }
        for name, text in SHIPPED.items():
            with self.subTest(file=name):
                failures = PlantedTree(self, {name: text}).check()
                self.assertEqual(flagged(failures), expected[name], failures)
                self.assertTrue(all(failure.startswith(name) for failure in failures), failures)

    def test_every_double_compile_fails(self) -> None:
        for label, text in DOUBLE.items():
            with self.subTest(label):
                self.assertTrue(
                    flagged(PlantedTree(self, {"crates/planted/tests/case.rs": text}).check()),
                    text,
                )

    def test_helpers_are_reported_where_they_meet(self) -> None:
        failures = PlantedTree(
            self, {"crates/planted/tests/case.rs": DOUBLE["inherited from two helpers"]}
        ).check()
        self.assertEqual(flagged(failures), {"both_lanes"}, failures)
        self.assertIn("(line 2)", failures[0])
        self.assertIn("(line 5)", failures[0])

    def test_every_single_compile_passes(self) -> None:
        for label, text in SINGLE.items():
            with self.subTest(label):
                self.assertEqual(
                    PlantedTree(self, {"crates/planted/tests/case.rs": text}).check(), []
                )


class ScanTests(unittest.TestCase):
    def test_tokens_skip_comments_and_keep_literals(self) -> None:
        tokens = guard.tokenize(
            "a /* x /* y */ z */ b // c\n"
            "r#\"q\"u\"# br\"s\" c\"t\" '\\'' 'k' 'life \"d\\\"e\" r#type 9u8\n"
        )
        self.assertEqual(
            [(token.kind, token.text) for token in tokens],
            [
                ("ident", "a"),
                ("ident", "b"),
                ("str", 'q"u'),
                ("str", "s"),
                ("str", "t"),
                ("punct", "'"),
                ("ident", "life"),
                ("str", 'd\\"e'),
                ("ident", "r"),
                ("punct", "#"),
                ("ident", "type"),
                ("other", "9u8"),
            ],
        )
        self.assertEqual([token.line for token in tokens][-1], 2)

    def test_an_unreadable_file_fails_closed(self) -> None:
        for text in (
            'fn lane() { run(&["build"]);\n',
            'fn lane() { run(&["build"); }\n',
            'fn lane() { run(&["build"]); "x }\n',
            'fn lane() { run(&["build"]); r#"x" }\n',
            "fn lane() { run(&[\"build\"]); '\\\n",
        ):
            with self.subTest(text=text):
                failures = PlantedTree(self, {"crates/planted/tests/case.rs": text}).check()
                self.assertEqual(len(failures), 1, failures)
                self.assertIn("cannot scan", failures[0])

    def test_files_without_a_build_literal_are_skipped(self) -> None:
        text = DOUBLE["builder chain"].replace('"build"', '"check"')
        self.assertEqual(PlantedTree(self, {"crates/planted/tests/case.rs": text}).check(), [])

    def test_the_repository_scan_reads_rust_files_under_crates(self) -> None:
        files = guard.rust_files(guard.ROOT)
        self.assertIn("crates/chelis-cli/tests/common/mod.rs", files)
        self.assertTrue(all(path.startswith("crates/") and path.endswith(".rs") for path in files))


if __name__ == "__main__":
    unittest.main()
