"""Unit tests for `scripts/regen_all.py`.

Run via `.venv/bin/python -m unittest scripts.test_regen_all` from the repo
root, or through the CI `unittest discover -s scripts` step.

Every test injects a recording runner in place of `subprocess.run`, so nothing
here needs cargo, a built `chelis`, or a network. The tier-2 hooks read files
from an injected temporary repository root rather than the live checkout.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent / "regen_all.py"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location("regen_all_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


regen_all = _load_module()

PYTHON = "/managed/python3.11"
GOOD_SHA = "a" * 64
OTHER_SHA = "b" * 64


class _Recorder:
    """A `subprocess.run` stand-in that records argv and env per call.

    `returncodes` maps a substring of the rendered argv to the exit code that
    call should report; unmatched calls exit 0. When `corpus` is set, a call
    to the corpus generator with `--out-dir` copies that tree into the
    requested directory so the custom check has something to compare.
    """

    def __init__(
        self,
        returncodes: dict[str, int] | None = None,
        corpus: Path | None = None,
        missing_programs: tuple[str, ...] = (),
    ):
        self.returncodes = returncodes or {}
        self.corpus = corpus
        self.missing_programs = missing_programs
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, argv, *, cwd=None, env=None, check=False, **_kwargs):
        argv = list(argv)
        self.calls.append((argv, dict(env or {})))
        if argv[0] in self.missing_programs:
            raise FileNotFoundError(2, "No such file or directory", argv[0])
        rendered = " ".join(argv)
        if "--out-dir" in argv and self.corpus is not None:
            target = Path(argv[argv.index("--out-dir") + 1])
            shutil.copytree(self.corpus, target, dirs_exist_ok=True)
        returncode = 0
        for needle, code in self.returncodes.items():
            if needle in rendered:
                returncode = code
        return subprocess.CompletedProcess(argv, returncode)

    def rendered(self) -> list[str]:
        return [" ".join(argv) for argv, _env in self.calls]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_repo(tmp: Path, *, todo_rows: int = 0, freeze_sha: str = GOOD_SHA) -> Path:
    """A minimal repository root carrying every file the tier-2 hooks read."""
    rows = [
        {"kind": "header-export", "id": f"row {n}", "flags": [], "citation": "TODO"}
        for n in range(todo_rows)
    ]
    rows.append({"kind": "header-export", "id": "classified", "flags": [], "citation": ""})
    _write(
        tmp / regen_all.CENSUS_JSON,
        json.dumps({"version": 3, "legs": [], "rows": rows}, indent=2) + "\n",
    )
    _write(
        tmp / regen_all.RUNTIME_REPRESENTATION_ORACLE,
        "REPO_ROOT = None\n"
        f'FREEZE_SHA256 = "{freeze_sha}"\n'
        "PHASE0_COMMAND = 'x'\n",
    )
    _write(
        tmp / regen_all.RUNTIME_REPRESENTATION_INVENTORY,
        json.dumps({"freeze_sha256": GOOD_SHA, "foundation_rows": []}, indent=2) + "\n",
    )
    corpus = tmp / regen_all.OPAQUE_CORPUS_DIR
    _write(corpus / "manifest.json", '{"schema_version": 1}\n')
    _write(corpus / "programs" / "a.ch", "def a() -> unit = ()\n")
    _write(corpus / "generate_corpus.py", "# stand-in\n")
    return tmp


def _run(
    argv, *, repo_root: Path, recorder: _Recorder, environ: dict[str, str] | None = None
) -> tuple[int, str]:
    out = io.StringIO()
    code = regen_all.main(
        argv,
        repo_root=repo_root,
        python=PYTHON,
        runner=recorder,
        environ={"PATH": "/usr/bin"} if environ is None else environ,
        out=out,
    )
    return code, out.getvalue()


class LegManifestTests(unittest.TestCase):
    def test_default_tiers_are_zero_and_one_in_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder()
            code, output = _run([], repo_root=root, recorder=recorder)
        self.assertEqual(code, 0, output)
        self.assertEqual(
            recorder.rendered(),
            [
                f"{PYTHON} scripts/generate_rejection_registries.py --write",
                f"{PYTHON} scripts/regenerate_conformance_assets.py",
                f"{PYTHON} scripts/generate_reviewed_unsupported_wording_snapshot.py",
                f"{PYTHON} tests/corpus/opaque_invariants/generate_corpus.py",
                f"{PYTHON} scripts/regenerate_chelis_std_bundle.py --debug",
            ],
        )
        self.assertNotIn("capacity_census_tripwire", output)
        self.assertNotIn("tree-sitter", output)
        self.assertTrue(output.rstrip().endswith("REGEN ALL: PASS"), output)

    def test_tier_zero_alone_runs_only_the_python_legs(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder()
            code, output = _run(["--tier", "0"], repo_root=root, recorder=recorder)
        self.assertEqual(code, 0, output)
        self.assertEqual(len(recorder.calls), 4)
        for argv, _env in recorder.calls:
            self.assertEqual(argv[0], PYTHON)
            self.assertNotEqual(argv[0], "cargo")

    def test_full_adds_tier_two_and_check_only_legs(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder()
            code, output = _run(["--full"], repo_root=root, recorder=recorder)
        self.assertEqual(code, 0, output)
        rendered = recorder.rendered()
        self.assertEqual(
            rendered[5:],
            [
                "cargo nextest run -p chelis-cli --test capacity_census_tripwire",
                f"{PYTHON} scripts/runtime_representation_oracle.py --phase 0 --regenerate",
                "cargo nextest run -p chelis-runtime --test runtime_dtype_generated_header",
                "cargo nextest run -p chelis-compiler-api --test capacity_census_wire",
            ],
        )
        self.assertIn("(no writer)", output)
        self.assertIn(regen_all.TREE_SITTER_NOTE, output)

    def test_tier_two_requires_full(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                regen_all.parse_args(["--tier", "2"])
        self.assertEqual(raised.exception.code, 2)
        args = regen_all.parse_args(["--tier", "2", "--full"])
        self.assertEqual(args.tiers, (2,))

    def test_legs_are_sorted_by_tier_and_named_uniquely(self):
        legs = regen_all.regen_legs(PYTHON)
        tiers = [leg.tier for leg in legs]
        self.assertEqual(tiers, sorted(tiers))
        names = [leg.name for leg in legs]
        self.assertEqual(len(names), len(set(names)))
        for leg in legs:
            if leg.note is None:
                self.assertTrue(
                    leg.write_argv is not None or leg.check_argv is not None,
                    f"{leg.name} can neither write nor check",
                )
            if leg.write_argv is None and leg.note is None:
                self.assertIsNotNone(
                    leg.manual_after, f"check-only leg {leg.name} names no manual action"
                )


class CheckModeTests(unittest.TestCase):
    def test_check_mode_swaps_every_writer_for_its_checker(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder(corpus=root / regen_all.OPAQUE_CORPUS_DIR)
            code, output = _run(["--check", "--full"], repo_root=root, recorder=recorder)
        self.assertEqual(code, 0, output)
        rendered = recorder.rendered()
        self.assertEqual(
            rendered[0], f"{PYTHON} scripts/generate_rejection_registries.py --check"
        )
        self.assertEqual(
            rendered[1], f"{PYTHON} scripts/regenerate_conformance_assets.py --check"
        )
        self.assertEqual(
            rendered[2],
            f"{PYTHON} scripts/generate_reviewed_unsupported_wording_snapshot.py --check",
        )
        self.assertTrue(
            rendered[3].startswith(
                f"{PYTHON} tests/corpus/opaque_invariants/generate_corpus.py --out-dir "
            ),
            rendered[3],
        )
        self.assertEqual(
            rendered[4], f"{PYTHON} scripts/regenerate_chelis_std_bundle.py --debug --check"
        )
        self.assertEqual(
            rendered[5], "cargo nextest run -p chelis-cli --test capacity_census_tripwire"
        )
        self.assertEqual(
            rendered[6], f"{PYTHON} scripts/runtime_representation_oracle.py --phase 0"
        )
        for line in rendered:
            self.assertNotIn("--write", line)
            self.assertNotIn("--regenerate", line)
        self.assertTrue(output.rstrip().endswith("REGEN ALL: PASS"), output)

    def test_check_mode_reports_every_stale_leg(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder(
                returncodes={
                    "generate_rejection_registries.py --check": 1,
                    "regenerate_chelis_std_bundle.py --debug --check": 3,
                },
                corpus=root / regen_all.OPAQUE_CORPUS_DIR,
            )
            code, output = _run(["--check"], repo_root=root, recorder=recorder)
        self.assertEqual(code, 1)
        # Every leg still ran (no fail-fast in check mode).
        self.assertEqual(len(recorder.calls), 5)
        self.assertTrue(
            output.rstrip().endswith("REGEN ALL: STALE (rejection-registry, std-bundle)"),
            output,
        )

    def test_check_only_leg_failure_in_check_mode_is_stale_with_the_manual_hint(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder(
                returncodes={"runtime_dtype_generated_header": 101},
                corpus=root / regen_all.OPAQUE_CORPUS_DIR,
            )
            code, output = _run(
                ["--check", "--full", "--tier", "2"], repo_root=root, recorder=recorder
            )
        self.assertEqual(code, 1)
        self.assertIn(regen_all.DTYPE_HEADER_MANUAL, output)
        self.assertTrue(output.rstrip().endswith("REGEN ALL: STALE (dtype-c-header)"), output)


class CensusEnvTests(unittest.TestCase):
    def test_census_write_leg_carries_the_env_and_check_does_not(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            writer = _Recorder()
            _run(["--full", "--tier", "2"], repo_root=root, recorder=writer)
            checker = _Recorder(corpus=root / regen_all.OPAQUE_CORPUS_DIR)
            _run(["--check", "--full", "--tier", "2"], repo_root=root, recorder=checker)
        write_env = [
            env for argv, env in writer.calls if "capacity_census_tripwire" in argv
        ]
        check_env = [
            env for argv, env in checker.calls if "capacity_census_tripwire" in argv
        ]
        self.assertEqual(len(write_env), 1)
        self.assertEqual(write_env[0].get(regen_all.CENSUS_WRITE_ENV), "1")
        self.assertEqual(len(check_env), 1)
        self.assertNotIn(regen_all.CENSUS_WRITE_ENV, check_env[0])
        # The env is scoped to that one leg.
        for argv, env in writer.calls:
            if "capacity_census_tripwire" not in argv:
                self.assertNotIn(regen_all.CENSUS_WRITE_ENV, env)

    def test_check_mode_child_env_never_carries_a_leg_env_key(self):
        # An ambient leftover from a manual census write must not turn the
        # check-mode tripwire into a writer that then compares against what
        # it just wrote.
        ambient = {
            "PATH": "/usr/bin",
            regen_all.CENSUS_WRITE_ENV: "1",
            "KEEP": "yes",
        }
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            checker = _Recorder(corpus=root / regen_all.OPAQUE_CORPUS_DIR)
            code, _out = _run(["--check", "--full"], repo_root=root, recorder=checker, environ=ambient)
        self.assertEqual(code, 0)
        self.assertGreaterEqual(len(checker.calls), 8)
        for argv, env in checker.calls:
            self.assertNotIn(regen_all.CENSUS_WRITE_ENV, env, argv)
            self.assertEqual(env.get("KEEP"), "yes")

    def test_write_mode_scrubs_the_seam_from_every_leg_but_its_owner(self):
        ambient = {
            "PATH": "/usr/bin",
            regen_all.CENSUS_WRITE_ENV: "1",
        }
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            writer = _Recorder()
            code, _out = _run(["--full"], repo_root=root, recorder=writer, environ=ambient)
        self.assertEqual(code, 0)
        census_carriers = [
            argv for argv, env in writer.calls if regen_all.CENSUS_WRITE_ENV in env
        ]
        self.assertEqual(len(census_carriers), 1)
        self.assertIn("capacity_census_tripwire", census_carriers[0])

    def test_scrub_environment_is_keyed_on_every_leg_declaration(self):
        keys = regen_all.leg_env_keys(regen_all.regen_legs(PYTHON))
        self.assertEqual(
            keys,
            frozenset({regen_all.CENSUS_WRITE_ENV}),
        )
        scrubbed = regen_all.scrub_environment(
            {
                "A": "1",
                regen_all.CENSUS_WRITE_ENV: "1",
            },
            keys,
        )
        self.assertEqual(scrubbed, {"A": "1"})

    def test_write_env_is_printed_on_the_leg_line(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder()
            _code, output = _run(["--full", "--tier", "2"], repo_root=root, recorder=recorder)
        self.assertIn(
            "capacity-census (cargo): CHELIS_CAPACITY_CENSUS_WRITE=1 cargo nextest run",
            output,
        )
        self.assertNotIn("capacity-census-bindings", output)


class ManualActionTests(unittest.TestCase):
    def test_todo_citations_after_census_write_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td), todo_rows=2)
            recorder = _Recorder()
            code, output = _run(["--full", "--tier", "2"], repo_root=root, recorder=recorder)
        self.assertEqual(code, 2)
        self.assertIn('2 capacity census row(s) landed with citation "TODO"', output)
        self.assertIn("header-export: row 0", output)
        self.assertIn("header-export: row 1", output)
        self.assertIn(regen_all.CENSUS_TODO_INSTRUCTION, output)
        self.assertTrue(
            output.rstrip().endswith("REGEN ALL: MANUAL ACTION REQUIRED (capacity-census)"),
            output,
        )
        # Write mode stops there: the runtime-representation leg never ran.
        self.assertEqual(len(recorder.calls), 1)
        self.assertIn("later leg(s) not run", output)

    def test_census_todo_scan_ignores_classified_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td), todo_rows=0)
            self.assertEqual(regen_all.census_todo_rows(root / regen_all.CENSUS_JSON), [])
            self.assertIsNone(regen_all.after_census_write(root))

    def test_freeze_sha_mismatch_prints_manual_action_and_exits_2(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td), freeze_sha=OTHER_SHA)
            recorder = _Recorder()
            code, output = _run(["--full", "--tier", "2"], repo_root=root, recorder=recorder)
        self.assertEqual(code, 2)
        self.assertIn(f"hashes to {GOOD_SHA}", output)
        self.assertIn(f'FREEZE_SHA256 = "{OTHER_SHA}"', output)
        self.assertIn("B1 freeze review", output)
        self.assertTrue(
            output.rstrip().endswith(
                "REGEN ALL: MANUAL ACTION REQUIRED (runtime-representation)"
            ),
            output,
        )
        # Census (0) and runtime-representation (1) ran; the check-only legs did not.
        self.assertEqual(len(recorder.calls), 2)

    def test_freeze_sha_match_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td), freeze_sha=GOOD_SHA)
            self.assertIsNone(regen_all.after_runtime_representation_write(root))
            recorder = _Recorder()
            code, output = _run(["--full", "--tier", "2"], repo_root=root, recorder=recorder)
        self.assertEqual(code, 0, output)
        self.assertTrue(output.rstrip().endswith("REGEN ALL: PASS"), output)

    def test_freeze_sha_literal_missing_is_a_manual_action(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            (root / regen_all.RUNTIME_REPRESENTATION_ORACLE).write_text("# no literal\n")
            message = regen_all.after_runtime_representation_write(root)
        self.assertIsNotNone(message)
        self.assertIn("FREEZE_SHA256", message)

    def test_wire_check_only_leg_reports_the_manual_action_in_write_mode(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder(returncodes={"capacity_census_wire": 101})
            code, output = _run(["--full", "--tier", "2"], repo_root=root, recorder=recorder)
        self.assertEqual(code, 2)
        self.assertIn(regen_all.CENSUS_WIRE_MANUAL, output)
        self.assertTrue(
            output.rstrip().endswith(
                "REGEN ALL: MANUAL ACTION REQUIRED (capacity-census-wire)"
            ),
            output,
        )


class LaunchFailureTests(unittest.TestCase):
    """A program that cannot be launched never escapes as a traceback: the
    run still ends in one of the four `REGEN ALL:` lines."""

    def test_missing_cargo_in_write_mode_fails_the_leg_with_the_final_line(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder(missing_programs=("cargo",))
            code, output = _run(["--full", "--tier", "2"], repo_root=root, recorder=recorder)
        self.assertEqual(code, 1)
        self.assertIn("failed: could not launch cargo", output)
        self.assertTrue(
            output.rstrip().endswith(
                f"REGEN ALL: FAIL (capacity-census, exit {regen_all.LAUNCH_FAILURE_EXIT})"
            ),
            output,
        )
        # Write mode stops at the first failure.
        self.assertEqual(len(recorder.calls), 1)

    def test_missing_cargo_in_check_mode_is_stale_with_the_launch_reason(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder(
                missing_programs=("cargo",), corpus=root / regen_all.OPAQUE_CORPUS_DIR
            )
            code, output = _run(["--check", "--full"], repo_root=root, recorder=recorder)
        self.assertEqual(code, 1)
        self.assertIn("stale: could not launch cargo", output)
        # Every selected leg still ran: the Python legs passed, the three
        # cargo legs are stale for the launch reason.
        self.assertTrue(
            output.rstrip().endswith(
                "REGEN ALL: STALE (capacity-census, dtype-c-header, "
                "capacity-census-wire)"
            ),
            output,
        )

    def test_missing_interpreter_for_the_corpus_check_is_stale_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder(missing_programs=(PYTHON,))
            code, output = _run(["--check", "--tier", "0"], repo_root=root, recorder=recorder)
        self.assertEqual(code, 1)
        self.assertIn(f"could not launch {PYTHON}", output)
        self.assertTrue(
            output.rstrip().endswith(
                "REGEN ALL: STALE (rejection-registry, conformance-assets, "
                "reviewed-unsupported-wording, opaque-corpus)"
            ),
            output,
        )

    def test_check_only_leg_launch_failure_in_write_mode_is_failed_not_manual(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder(returncodes={}, missing_programs=())
            # Only the check-only legs use cargo once the two writers are
            # stubbed as present: make cargo vanish for them alone.
            def runner(argv, *, cwd=None, env=None, check=False, **_kw):
                recorder.calls.append((list(argv), dict(env or {})))
                if "runtime_dtype_generated_header" in argv:
                    raise PermissionError(13, "Permission denied", argv[0])
                return subprocess.CompletedProcess(list(argv), 0)
            code, output = _run(["--full", "--tier", "2"], repo_root=root, recorder=runner)
        self.assertEqual(code, 1)
        self.assertNotIn("manual:", output)
        self.assertTrue(
            output.rstrip().endswith(
                f"REGEN ALL: FAIL (dtype-c-header, exit {regen_all.LAUNCH_FAILURE_EXIT})"
            ),
            output,
        )


class OpaqueCorpusCheckTests(unittest.TestCase):
    def _tree(self, root: Path, program: str) -> Path:
        _write(root / "manifest.json", '{"schema_version": 1}\n')
        _write(root / "programs" / "a.ch", program)
        return root

    def test_opaque_corpus_check_compares_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            committed = self._tree(Path(td) / "committed", "def a() -> unit = ()\n")
            same = self._tree(Path(td) / "same", "def a() -> unit = ()\n")
            differs = self._tree(Path(td) / "differs", "def a() -> unit = ( )\n")
            self.assertEqual(regen_all.compare_corpus_trees(committed, same), [])
            reasons = regen_all.compare_corpus_trees(committed, differs)
        self.assertEqual(reasons, ["committed program 'a.ch' is stale vs generate_corpus.py"])

    def test_opaque_corpus_check_reports_manifest_missing_and_extra_files(self):
        with tempfile.TemporaryDirectory() as td:
            committed = self._tree(Path(td) / "committed", "x\n")
            regenerated = self._tree(Path(td) / "regenerated", "x\n")
            _write(regenerated / "programs" / "b.ch", "y\n")
            _write(regenerated / "manifest.json", '{"schema_version": 2}\n')
            reasons = regen_all.compare_corpus_trees(committed, regenerated)
        self.assertEqual(
            reasons,
            [
                "committed manifest.json is stale vs generate_corpus.py",
                "regenerated program 'b.ch' is missing from programs/",
            ],
        )

    def test_stale_corpus_makes_check_mode_stale(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            fresh = Path(td) / "fresh"
            self._tree(fresh, "def a() -> unit = ( )\n")
            recorder = _Recorder(corpus=fresh)
            code, output = _run(["--check", "--tier", "0"], repo_root=root, recorder=recorder)
        self.assertEqual(code, 1)
        self.assertIn("stale: committed program 'a.ch' is stale", output)
        self.assertTrue(output.rstrip().endswith("REGEN ALL: STALE (opaque-corpus)"), output)

    def test_generator_failure_is_reported_as_stale(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder(returncodes={"generate_corpus.py": 7})
            code, output = _run(["--check", "--tier", "0"], repo_root=root, recorder=recorder)
        self.assertEqual(code, 1)
        self.assertIn("generate_corpus.py exited 7", output)


class WriteModeTests(unittest.TestCase):
    def test_tier_zero_owns_the_reviewed_unsupported_wording_snapshot(self):
        legs = regen_all.regen_legs(PYTHON)
        wording = next(
            leg for leg in legs if leg.name == "reviewed-unsupported-wording"
        )
        self.assertEqual(wording.tier, 0)
        self.assertIn(
            "scripts/generate_reviewed_unsupported_wording_snapshot.py",
            wording.write_argv,
        )
        self.assertIn("--check", wording.check_argv)

    def test_write_mode_stops_at_first_failure_and_names_the_leg(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder(returncodes={"regenerate_conformance_assets.py": 1})
            code, output = _run([], repo_root=root, recorder=recorder)
        self.assertEqual(code, 1)
        self.assertEqual(len(recorder.calls), 2)
        self.assertNotIn("generate_corpus.py", " ".join(recorder.rendered()))
        self.assertNotIn("regenerate_chelis_std_bundle.py", " ".join(recorder.rendered()))
        self.assertTrue(
            output.rstrip().endswith("REGEN ALL: FAIL (conformance-assets, exit 1)"), output
        )
        self.assertIn("3 later leg(s) not run", output)

    def test_leg_lines_name_tier_position_and_owned_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = _fake_repo(Path(td))
            recorder = _Recorder()
            _code, output = _run([], repo_root=root, recorder=recorder)
        self.assertIn(
            "[tier 0 1/5] rejection-registry (cargo + python):", output
        )
        self.assertIn("[tier 1 5/5] std-bundle (cargo):", output)
        self.assertIn(
            "owns: spec/design/loud_unsupported_issue_manifest.json, "
            "crates/chelis-types/src/rejection_registry_generated.rs",
            output,
        )
        self.assertIn("regen_all: write mode, tiers 0, 1, 5 leg(s)", output)


class NeverWritesFrozenArtifactsTests(unittest.TestCase):
    FORBIDDEN_WRITE_TARGETS = (
        "dtype_phase4b_oracle",
        "FROZEN_",
        "loud_unsupported_tripwire",
        "runtime_extent_oracle_baseline",
        "copy_drop_fixture_fitness_baseline",
        "test_timing_baseline",
        "capacity_census_bindings.json",
        "capacity_census_wire.json",
        "chelis_runtime_dtype.h",
        "tree-sitter generate",
    )

    def test_never_writes_frozen_artifacts(self):
        legs = regen_all.regen_legs(PYTHON)
        for leg in legs:
            argv_text = " ".join(leg.write_argv or ())
            env_text = " ".join(f"{k}={v}" for k, v in leg.env)
            for needle in self.FORBIDDEN_WRITE_TARGETS:
                self.assertNotIn(needle, argv_text, f"{leg.name} writes {needle}")
                self.assertNotIn(needle, env_text, f"{leg.name} env names {needle}")
        regenerate_legs = [
            leg.name for leg in legs if "--regenerate" in (leg.write_argv or ())
        ]
        self.assertEqual(regenerate_legs, ["runtime-representation"])
        for leg in legs:
            self.assertNotIn("--regenerate", leg.check_argv or ())
        # The artifacts with no writer are check-only legs, never write legs.
        no_writer = {leg.name for leg in legs if leg.write_argv is None}
        self.assertEqual(
            no_writer, {"dtype-c-header", "capacity-census-wire", "tree-sitter"}
        )

    def test_docstring_names_every_hand_maintained_artifact(self):
        doc = regen_all.__doc__ or ""
        for name in (
            "REQUIRED_ATOMS",
            "REQUIRED_REGIONS",
            "FREEZE_SHA256",
            "loud_unsupported_tripwire.rs",
            "runtime_extent_oracle_baseline.json",
            "copy_drop_fixture_fitness_baseline.json",
            "test_timing_baseline.json",
            "capacity_census_bindings.json",
            "capacity_census_wire.json",
            "chelis_runtime_dtype.h",
            "grammars/",
        ):
            self.assertIn(name, doc)


class LiveRepositoryShapeTests(unittest.TestCase):
    """Cheap disk reads against the real checkout: the legs name real files."""

    def test_every_leg_script_and_test_target_exists(self):
        for leg in regen_all.regen_legs(PYTHON):
            for argv in (leg.write_argv, leg.check_argv):
                if argv is None:
                    continue
                if argv[0] == PYTHON:
                    self.assertTrue((REPO_ROOT / argv[1]).is_file(), argv[1])
                elif argv[0] == "cargo":
                    package = None
                    for index, token in enumerate(argv):
                        if token == "-p":
                            package = argv[index + 1]
                        if token == "--test":
                            assert package is not None
                            target = REPO_ROOT / "crates" / package / "tests" / f"{argv[index + 1]}.rs"
                            self.assertTrue(target.is_file(), str(target))

    def test_live_hooks_read_the_real_files(self):
        self.assertIsNotNone(
            regen_all.reviewed_freeze_sha(
                (REPO_ROOT / regen_all.RUNTIME_REPRESENTATION_ORACLE).read_text()
            )
        )
        self.assertIsNone(regen_all.after_runtime_representation_write(REPO_ROOT))
        self.assertEqual(regen_all.census_todo_rows(REPO_ROOT / regen_all.CENSUS_JSON), [])


if __name__ == "__main__":
    unittest.main()
