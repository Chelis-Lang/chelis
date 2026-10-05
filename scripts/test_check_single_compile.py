"""Tests for the single-compile guard (chelis#2928)."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
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
    "a compiler held in a bare identifier": (
        "fn lane() {\n"
        '    run(&["build", "main.ch"]);\n'
        '    let cc = locate("clang-17");\n'
        '    StdCommand::new(&cc).arg("main.c").status().unwrap();\n'
        "}\n"
    ),
    "a method that links the default build": (
        "struct CBuild {\n"
        "    out: PathBuf,\n"
        "}\n"
        "impl CBuild {\n"
        "    fn link_and_run(&self) -> Output {\n"
        '        assert!(link_generated(&self.out, "prog.c", "prog").success());\n'
        '        Command::new(self.out.join("prog")).output().unwrap()\n'
        "    }\n"
        "}\n"
        "fn build_c(file: &Path) -> CBuild {\n"
        '    Command::new(chelis_bin()).args(["build", "--target", "c"]).arg(file).output().unwrap();\n'
        "    CBuild { out: PathBuf::new() }\n"
        "}\n"
        "#[test]\n"
        "fn lane() {\n"
        "    build_c(&file).link_and_run();\n"
        "}\n"
    ),
    "a method that takes the command": (
        "struct Harness;\n"
        "impl Harness {\n"
        "    fn cli(&self, command: &str, path: &Path) -> Output {\n"
        '        Command::cargo_bin("chelis").unwrap().arg(command).arg(path).output().unwrap()\n'
        "    }\n"
        "}\n"
        "#[test]\n"
        "fn lane() {\n"
        '    harness.cli("build", &path);\n'
        '    assert!(link_generated(&out, "main.c", "main").success());\n'
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
    "a method that links a source-only build": (
        "struct CBuild {\n"
        "    out: PathBuf,\n"
        "}\n"
        "impl CBuild {\n"
        "    fn link_and_run(&self) -> Output {\n"
        '        assert!(link_generated(&self.out, "prog.c", "prog").success());\n'
        '        Command::new(self.out.join("prog")).output().unwrap()\n'
        "    }\n"
        "}\n"
        "fn build_c(file: &Path) -> CBuild {\n"
        '    Command::new(chelis_bin()).args(["build", "--emit-c", "--target", "c"]).arg(file).output().unwrap();\n'
        "    CBuild { out: PathBuf::new() }\n"
        "}\n"
        "#[test]\n"
        "fn lane() {\n"
        "    build_c(&file).link_and_run();\n"
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


# A test that calls two build helpers and two link helpers, the later-defined
# of each first.
TWO_OF_EACH = (
    "fn build_a() {\n"
    '    run(&["build", "a.ch"]);\n'
    "}\n"
    "fn build_b() {\n"
    '    run(&["build", "b.ch"]);\n'
    "}\n"
    "fn link_a(out: &Path) {\n"
    '    assert!(link_generated(out, "a.c", "a").success());\n'
    "}\n"
    "fn link_b(out: &Path) {\n"
    '    assert!(link_generated(out, "b.c", "b").success());\n'
    "}\n"
    "#[test]\n"
    "fn lanes() {\n"
    "    build_b();\n"
    "    build_a();\n"
    "    link_b(&out);\n"
    "    link_a(&out);\n"
    "}\n"
)


# The business-day oracle's C lane before chelis#2928 gave it `--emit-c`
# (scripts/datetime_business_differential.py), reduced to its build and link.
PY_SHIPPED = f"""class Runner:
    def c_lane(self, program):
        app = self.app(program, "c")
        build = self.run([str(self.chelis), "build", "src/main.ch", "--target", "c", "--output", "out"], app)
        if build.returncode != 0:
            return LaneResult("c", build.returncode, build.stdout, build.stderr, "build")
        link = self.run([self.toolchain.compiler, *self.toolchain.compile_flags, "-Iout", "out/main.c",
                         "out/{guard.RUNTIME_ARCHIVE}", *self.toolchain.link_flags, "-o", "out/case"], app)
        return link
"""

# Each must fail: a default build and a compile meet in one Python function.
PY_DOUBLE = {
    "helpers split across methods": (
        "class Runner:\n"
        "    def build(self, app):\n"
        '        return self.run([self.chelis, "build", "src/main.ch", "-o", "out"], app)\n'
        "    def link(self, app):\n"
        '        return self.run([self.toolchain.compiler, "out/main.c", "-o", "out/case"], app)\n'
        "    def c_lane(self, app):\n"
        "        self.build(app)\n"
        "        self.link(app)\n"
    ),
    "helpers split across functions": (
        "def build(chelis, app):\n"
        '    subprocess.run([chelis, "build", "main.ch"], cwd=app, check=True)\n'
        "def link(app):\n"
        '    subprocess.run(["cc", "main.c", "-o", "main"], cwd=app, check=True)\n'
        "def c_lane(chelis, app):\n"
        "    build(chelis, app)\n"
        "    link(app)\n"
    ),
    "a compiler held in a bare name": (
        "def c_lane(chelis):\n"
        '    subprocess.run((chelis, "build", "main.ch"), check=True)\n'
        '    cc = shutil.which("clang-17")\n'
        '    subprocess.run([cc, "main.c", "-o", "main"], check=True)\n'
    ),
    "a compiler inside str() and the command first": (
        "def c_lane(compiler):\n"
        '    run_chelis(["build", "main.ch"])\n'
        '    subprocess.run([str(compiler), "main.c", "-o", "main"], check=True)\n'
    ),
    "module-level statements": (
        'subprocess.run([CHELIS, "build", "main.ch"], check=True)\n'
        'subprocess.run(["gcc", "main.c", "-o", "main"], check=True)\n'
    ),
}

# Each must pass.
PY_SINGLE = {
    "source-only build compiled by the harness": (
        "class Runner:\n"
        "    def c_lane(self, main, app):\n"
        '        build = self.run([str(self.chelis), "build", main, "--target", "c", "--emit-c", "--output", "out"], app)\n'
        '        return self.run([self.toolchain.compiler, "out/main.c", "-o", "out/case"], app)\n'
    ),
    "default build run as published": (
        "def c_lane(chelis, app):\n"
        '    subprocess.run([chelis, "build", "src/main.ch", "--output", "out"], cwd=app, check=True)\n'
        '    return subprocess.run([str(app / "out" / "main")], capture_output=True)\n'
    ),
    "another tool's build subcommand": (
        "def prepare(compiler):\n"
        '    subprocess.run(["cargo", "build", "-p", "chelis-cli"], check=True)\n'
        '    subprocess.run([compiler, "probe.c", "-o", "probe"], check=True)\n'
    ),
    "published library linked into a driver": (
        "def library(chelis, compiler, out):\n"
        '    subprocess.run([chelis, "build", "lib.ch", "--output", str(out)], check=True)\n'
        '    subprocess.run([compiler, "driver.c", str(out / "liblib.a"), "-o", "driver"], check=True)\n'
    ),
    "build and compile inside literals and comments": (
        "# subprocess.run([chelis, \"build\", \"main.ch\"])\n"
        "def c_lane(compiler):\n"
        "    note = '[chelis, \"build\", \"main.ch\"]'\n"
        '    subprocess.run([compiler, "main.c", "-o", "main"], check=True)\n'
    ),
}

# A Python harness that calls two build helpers and two link helpers, the
# later-defined of each first.
PY_TWO_OF_EACH = (
    "def build_a(chelis):\n"
    '    subprocess.run([chelis, "build", "a.ch"])\n'
    "def build_b(chelis):\n"
    '    subprocess.run([chelis, "build", "b.ch"])\n'
    "def link_a(compiler):\n"
    '    subprocess.run([compiler, "a.c"])\n'
    "def link_b(compiler):\n"
    '    subprocess.run([compiler, "b.c"])\n'
    "def lanes(chelis, compiler):\n"
    "    build_b(chelis)\n"
    "    build_a(chelis)\n"
    "    link_b(compiler)\n"
    "    link_a(compiler)\n"
)


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
    return {
        match.group(1)
        for failure in failures
        if (match := re.search(r": (?:fn|def) (\S+) runs", failure))
    }


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

    def test_the_cited_lines_do_not_depend_on_string_hashing(self) -> None:
        tree = PlantedTree(
            self,
            {"crates/planted/tests/case.rs": TWO_OF_EACH, "scripts/planted_lanes.py": PY_TWO_OF_EACH},
        )
        script = (
            "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "import check_single_compile as guard; "
            "print(guard.check(Path(sys.argv[2]), ['crates/planted/tests/case.rs', 'scripts/planted_lanes.py']))"
        )
        outputs = {
            subprocess.run(
                [sys.executable, "-B", "-c", script, str(Path(guard.__file__).parent), str(tree.root)],
                env={**os.environ, "PYTHONHASHSEED": str(seed)},
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            for seed in range(8)
        }
        self.assertEqual(len(outputs), 1, outputs)
        # The first build and link helpers `lanes` calls supply the cited lines.
        (output,) = outputs
        self.assertIn("case.rs:14: fn lanes runs a default `chelis build` (line 5) and compiles C itself (line 11)", output)
        self.assertIn("planted_lanes.py:9: def lanes runs a default `chelis build` (line 4) and compiles C itself (line 8)", output)

    def test_the_shipped_python_double_compile_fails(self) -> None:
        failures = PlantedTree(self, {"scripts/planted_differential.py": PY_SHIPPED}).check()
        self.assertEqual(flagged(failures), {"c_lane"}, failures)
        self.assertIn("(line 4) and compiles C itself (line 7)", failures[0])

    def test_every_python_double_compile_fails(self) -> None:
        for label, text in PY_DOUBLE.items():
            with self.subTest(label):
                self.assertTrue(
                    flagged(PlantedTree(self, {"scripts/planted.py": text}).check()), text
                )

    def test_python_helpers_are_reported_where_they_meet(self) -> None:
        for label, name in (
            ("helpers split across methods", "c_lane"),
            ("helpers split across functions", "c_lane"),
            ("module-level statements", "<module>"),
        ):
            with self.subTest(label):
                failures = PlantedTree(self, {"scripts/planted.py": PY_DOUBLE[label]}).check()
                self.assertEqual(flagged(failures), {name}, failures)

    def test_every_python_single_compile_passes(self) -> None:
        for label, text in PY_SINGLE.items():
            with self.subTest(label):
                self.assertEqual(PlantedTree(self, {"scripts/planted.py": text}).check(), [])

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
        failures = PlantedTree(
            self, {"scripts/planted.py": 'def c_lane(:\n    run([chelis, "build"])\n'}
        ).check()
        self.assertEqual(len(failures), 1, failures)
        self.assertIn("scripts/planted.py: cannot scan: not parseable Python", failures[0])

    def test_files_without_a_build_literal_are_skipped(self) -> None:
        text = DOUBLE["builder chain"].replace('"build"', '"check"')
        self.assertEqual(PlantedTree(self, {"crates/planted/tests/case.rs": text}).check(), [])

    def test_the_repository_scan_reads_rust_crates_and_python_scripts(self) -> None:
        files = guard.source_files(guard.ROOT)
        self.assertIn("crates/chelis-cli/tests/common/mod.rs", files)
        self.assertIn("scripts/datetime_business_differential.py", files)
        self.assertIn(".github/scripts/smoke_macos_metal.py", files)
        self.assertTrue(
            all(
                (path.startswith("crates/") and path.endswith(".rs"))
                or (path.startswith(("scripts/", ".github/scripts/")) and path.endswith(".py"))
                for path in files
            )
        )


if __name__ == "__main__":
    unittest.main()
