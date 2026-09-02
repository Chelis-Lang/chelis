from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("runtime_extent_oracle.py")
SPEC = importlib.util.spec_from_file_location("runtime_extent_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORACLE
SPEC.loader.exec_module(ORACLE)


class RuntimeExtentOracleTests(unittest.TestCase):
    def test_generated_corpus_is_sorted_unique_and_covers_slice_a_boundaries(self) -> None:
        rows = ORACLE.generated_phase_a_corpus()
        ids = [row.id for row in rows]
        self.assertEqual(ids, sorted(set(ids)))
        for required in (
            "expand.literal_zero.eval_c",
            "expand.binder.eval_c",
            "expand.node.hip_device",
            "expand.node.metal_device",
            "grad.input_axis.runtime_movement_source.eval_c",
            "grad_vmap.backward_expand.eval_c",
            "ir.movement_input_cardinality",
            "reshape.negative.runtime_eval_c",
            "vmap.element_derived_extent",
            "vmap.shared_shape_bound",
            "vmap.shared_shape_bound.concrete_c_emit",
            "wire.input_axis.round_trip",
            "wire.movement_input_cardinality",
            "wire.capacity_census",
        ):
            self.assertIn(required, ids)

        by_id = {row.id: row for row in rows}
        self.assertEqual(
            by_id["vmap.element_derived_extent"].receipt,
            "cli.vmap_rejects_element_derived_extent_at_public_checker",
        )
        self.assertEqual(
            by_id["vmap.shared_shape_bound.concrete_c_emit"].phase_a,
            "silent_unguarded",
        )
        self.assertEqual(
            by_id["reshape.negative.runtime_eval_c"].phase_a,
            "rejects_exactly",
        )

    def test_checked_baseline_exactly_matches_generated_corpus(self) -> None:
        payload = ORACLE.load_and_validate_baseline()
        self.assertEqual(
            payload["corpus_sha256"],
            ORACLE.corpus_digest(ORACLE.generated_phase_a_corpus()),
        )

    def test_baseline_digest_mutation_fails_closed(self) -> None:
        payload = json.loads(ORACLE.BASELINE_PATH.read_text())
        payload["corpus_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ORACLE.OracleFailure, "corpus digest drift"):
                ORACLE.load_and_validate_baseline(path)

    def test_status_lattice_rejects_regression_and_unregistered_receipt(self) -> None:
        ORACLE.validate_transition("ice", "typed_unsupported(#1298)")
        ORACLE.validate_transition("typed_unsupported(#1298)", "executes_exactly")
        ORACLE.validate_transition("silent_unguarded", "rejects_exactly")
        with self.assertRaisesRegex(ORACLE.OracleFailure, "forbidden status transition"):
            ORACLE.validate_transition("executes_exactly", "lane_divergent")
        with self.assertRaisesRegex(ORACLE.OracleFailure, "unknown runtime-extent status"):
            ORACLE.validate_transition("ice", "typed_unsupported")

    def test_execution_receipt_rejects_zero_match_ignored_and_duplicates(self) -> None:
        with self.assertRaisesRegex(ORACLE.OracleFailure, "zero per-test receipts"):
            ORACLE.parse_test_receipt("running 0 tests\n")
        with self.assertRaisesRegex(ORACLE.OracleFailure, "did not pass"):
            ORACLE.parse_test_receipt("test row ... ignored\n")
        with self.assertRaisesRegex(ORACLE.OracleFailure, "duplicate"):
            ORACLE.parse_test_receipt("test row ... ok\ntest row ... ok\n")

    def test_execution_receipt_requires_the_exact_frozen_test_set(self) -> None:
        target = ORACLE.TestTarget(
            "sample", ("cargo", "test"), ("first", "second")
        )
        completed = subprocess.CompletedProcess(
            target.argv,
            0,
            stdout="test first ... ok\n",
            stderr="",
        )
        with self.assertRaisesRegex(ORACLE.OracleFailure, "test receipt drift"):
            ORACLE.validate_target_receipt(target, completed)

    def test_every_corpus_row_has_a_named_executable_receipt(self) -> None:
        ORACLE.validate_receipt_coverage(
            ORACLE.generated_phase_a_corpus(), ORACLE.automatic_targets("python")
        )
        with self.assertRaisesRegex(ORACLE.OracleFailure, "no executable receipt"):
            ORACLE.validate_receipt_coverage(
                ORACLE.generated_phase_a_corpus(),
                (ORACLE.TestTarget("empty", ("true",)),),
            )

    def test_validate_runs_every_target_and_prints_exact_head_digest(self) -> None:
        targets = ORACLE.automatic_targets("python")
        by_argv = {target.argv: target for target in targets}
        seen: list[tuple[str, ...]] = []

        def runner(argv: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            command = tuple(argv)
            seen.append(command)
            if command == ("git", "rev-parse", "HEAD"):
                return subprocess.CompletedProcess(command, 0, "a" * 40 + "\n", "")
            if command == ("git", "status", "--porcelain"):
                return subprocess.CompletedProcess(command, 0, "", "")
            target = by_argv[command]
            if target.list_only:
                stdout = "".join(f"{name}: test\n" for name in target.expected_tests)
            else:
                stdout = "".join(
                    f"test {name} ... ok\n" for name in target.expected_tests
                )
            return subprocess.CompletedProcess(command, 0, stdout, "")

        output = io.StringIO()
        with redirect_stdout(output):
            head, digest = ORACLE.validate(
                "a", runner=runner, targets=targets, require_clean=True
            )
        self.assertEqual(head, "a" * 40)
        self.assertEqual(digest, ORACLE.corpus_digest(ORACLE.generated_phase_a_corpus()))
        self.assertEqual(seen[2:], [target.argv for target in targets])
        self.assertEqual(output.getvalue().splitlines()[-1], ORACLE.PASS_MARKER)

    def test_unimplemented_later_phases_cannot_report_success(self) -> None:
        for phase in ("b", "c", "final"):
            with self.subTest(phase=phase):
                with self.assertRaisesRegex(
                    ORACLE.OracleFailure, "only Slice A can report PASS"
                ):
                    ORACLE.validate(phase)

    def test_authoritative_run_rejects_dirty_worktree(self) -> None:
        def runner(argv: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            command = tuple(argv)
            if command == ("git", "rev-parse", "HEAD"):
                return subprocess.CompletedProcess(command, 0, "b" * 40 + "\n", "")
            if command == ("git", "status", "--porcelain"):
                return subprocess.CompletedProcess(command, 0, " M changed.rs\n", "")
            raise AssertionError(command)

        with self.assertRaisesRegex(ORACLE.OracleFailure, "clean exact-head"):
            ORACLE.exact_head_receipt(runner, require_clean=True)


if __name__ == "__main__":
    unittest.main()
