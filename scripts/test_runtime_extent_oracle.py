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


# The digest Slice A froze. The multi-phase plumbing keys each row's exit
# state by its owning phase, so phase A's canonical bytes must be identical
# to what the merged Slice A oracle produced.
#
# Moved once, deliberately, for chelis#1588. chelis#1277 S2a renamed the test
# `shape_sourced_expand_rejects_wrong_rank_ascription` to its `insert`
# spelling, because the program it checks spells `insert`, and left the phase-a
# target list and the `expand.rank_ascription` row's receipt pointing at the
# old name. That row is at an exit state, so `validate_receipt_coverage`
# enforces its receipt and `--phase a` was RED on `main`.
#
# The receipt is inside the hashed bytes, so correcting the pointer moves the
# digest. What moved was measured rather than asserted: canonicalizing both
# corpora and comparing row by row gives 32 rows before and after, ONE row
# differing, and that row differing only in its `receipt` string. No row id, no
# baseline and no exit state changed, so the freeze still holds over everything
# it was put there to protect.
#
# Was 29a3bf773f1637a70ec533b1d24ecfac7d0f7ce637af98a117f612abafa044e1.
FROZEN_PHASE_A_DIGEST = (
    "cb8401a05a009c97450cb0eaf00b94e2686725a37304ec44d63ceba8e1b64320"
)


def _spec(
    phase: str,
    rows: tuple[object, ...],
    *,
    baseline_path: Path | None = None,
    targets: tuple[object, ...] = (),
    deferred: dict[str, str] | None = None,
) -> object:
    return ORACLE.PhaseSpec(
        phase=phase,
        corpus=rows,
        baseline_path=baseline_path or Path("/nonexistent"),
        targets=lambda: targets,
        deferred=deferred or {},
    )


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
            by_id["vmap.shared_shape_bound.concrete_c_emit"].exit_state,
            "silent_unguarded",
        )
        self.assertEqual(
            by_id["reshape.negative.runtime_eval_c"].exit_state,
            "rejects_exactly",
        )

    def test_generated_phase_b_corpus_declares_the_c4_and_c2_properties(self) -> None:
        rows = ORACLE.generated_phase_b_corpus()
        ids = [row.id for row in rows]
        self.assertEqual(ids, sorted(set(ids)))
        for required in (
            "class.load_load.c",
            "class.load_load.eval",
            "class.load_op_output.c",
            "class.load_op_output.eval",
            "class.no_movement_consumer.c",
            "class.no_movement_consumer.eval",
            "class.op_output_op_output.c",
            "class.op_output_op_output.eval",
            "class.shared_member_node.c",
            "class.shared_member_node.eval",
            "class.splice_f_of_n_n.c",
            "class.splice_f_of_n_n.eval",
            "expand.arith_size.named_claim.c",
            "expand.arith_size.named_claim.eval",
            "expand.foreign_claim.same_tensor_set_axis.c",
            "expand.foreign_claim.same_tensor_set_axis.eval",
            "expand.kept_axis.op_declared_source.c",
            "expand.kept_axis.op_declared_source.eval",
            "expand.literal_claim.cross_tensor_read.c",
            "expand.literal_claim.cross_tensor_read.eval",
            "expand.literal_claim.inlined_root.c",
            "expand.literal_claim.inlined_root.eval",
            "expand.named_claim.cross_tensor_read.c",
            "expand.named_claim.cross_tensor_read.eval",
            "expand.op_declared_source.hip_prologue",
            "expand.piped_shape_read.lint_fix",
            "expand.positional.replacement.c",
            "expand.positional.replacement.eval",
            "expand.positional.replacement.non_unit_source_static",
            "expand.positional.replacement.non_unit_source_traps.c",
            "expand.positional.replacement.non_unit_source_traps.eval",
            "expand.positional.replacement.shape_size.eval_c",
            "expand.positional.replacement_zero.c",
            "expand.positional.replacement_zero.eval",
            "expand.record_projection.size",
            "expand.shape_derived.declared_result_survives.c",
            "expand.shape_derived.declared_result_survives.eval",
            "guard_order.effect_after.eval",
            "guard.local.numeric_carriers.eval_c",
            "guard_order.effect_before.eval",
            "guard_order.trap_after.c",
            "guard_order.trap_after.eval",
            "guard_order.trap_before.c",
            "guard_order.trap_before.eval",
            "ir.axis_source.cardinality",
            "rebuild.classes_after_each_pass",
            "reshape.named_claim.node_target.c",
            "reshape.named_claim.node_target.eval",
            "shrink.elementwise_const.build",
            "shrink.to_end.nonzero_start",
        ):
            self.assertIn(required, ids)

        by_id = {row.id: row for row in rows}
        # The rows PR B1 moves, and the two states it moves them to.
        self.assertEqual(by_id["ir.axis_source.cardinality"].exit_state, "executes_exactly")
        self.assertEqual(
            by_id["shrink.elementwise_const.build"].exit_state,
            "typed_unsupported(#1482)",
        )
        # chelis#665 and the class/guard rows stay at their main baseline
        # until Slice B's second half. Both lanes carry the same baseline:
        # the row is split because its guard lands per lane, not because the
        # lanes start anywhere different.
        for lane in ("c", "eval"):
            row = by_id[f"expand.kept_axis.op_declared_source.{lane}"]
            self.assertEqual(row.baseline, "ice")
            self.assertEqual(row.exit_state, "ice")
        # The HIP prologue row is new in Slice B's second half and starts at
        # the same `require_load_source` panic, on the one lane that has it.
        hip = by_id["expand.op_declared_source.hip_prologue"]
        self.assertEqual(hip.baseline, "ice")

    def test_phase_a_bytes_and_digest_are_unchanged_by_multi_phase_plumbing(self) -> None:
        rows = ORACLE.generated_phase_a_corpus()
        payload = json.loads(ORACLE.canonical_corpus_bytes(rows, "a"))
        self.assertEqual(
            sorted(payload[0]), ["baseline", "id", "phase_a", "receipt"]
        )
        self.assertEqual(ORACLE.corpus_digest(rows, "a"), FROZEN_PHASE_A_DIGEST)
        self.assertEqual(
            json.loads(ORACLE.BASELINE_PATH.read_text())["corpus_sha256"],
            FROZEN_PHASE_A_DIGEST,
        )
        # The per-phase key is what distinguishes the two byte streams, so a
        # phase B corpus can never collide with phase A's frozen digest.
        self.assertNotEqual(
            ORACLE.corpus_digest(ORACLE.generated_phase_b_corpus(), "b"),
            FROZEN_PHASE_A_DIGEST,
        )

    def test_checked_baselines_exactly_match_their_generated_corpus(self) -> None:
        for phase in ("a", "b"):
            with self.subTest(phase=phase):
                spec = ORACLE.PHASE_REGISTRY[phase]
                payload = ORACLE.load_and_validate_baseline(spec)
                self.assertEqual(
                    payload["corpus_sha256"],
                    ORACLE.corpus_digest(spec.corpus, phase),
                )

    def test_baseline_digest_mutation_fails_closed(self) -> None:
        spec = ORACLE.PHASE_REGISTRY["a"]
        payload = json.loads(spec.baseline_path.read_text())
        payload["corpus_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ORACLE.OracleFailure, "corpus digest drift"):
                ORACLE.load_and_validate_baseline(spec, path)

    def test_baseline_row_keyed_by_the_wrong_phase_is_rejected(self) -> None:
        spec = ORACLE.PHASE_REGISTRY["b"]
        payload = json.loads(spec.baseline_path.read_text())
        payload["rows"][0]["phase_a"] = payload["rows"][0].pop("phase_b")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ORACLE.OracleFailure, "wrong fields"):
                ORACLE.load_and_validate_baseline(spec, path)

    def test_status_lattice_rejects_regression_and_unregistered_receipt(self) -> None:
        ORACLE.validate_transition("ice", "typed_unsupported(#1298)")
        ORACLE.validate_transition("typed_unsupported(#1298)", "executes_exactly")
        ORACLE.validate_transition("silent_unguarded", "rejects_exactly")
        with self.assertRaisesRegex(ORACLE.OracleFailure, "forbidden status transition"):
            ORACLE.validate_transition("executes_exactly", "lane_divergent")
        with self.assertRaisesRegex(ORACLE.OracleFailure, "unknown runtime-extent status"):
            ORACLE.validate_transition("ice", "typed_unsupported")

    def test_the_lattice_binds_the_whole_baseline_to_phase_b_chain(self) -> None:
        ORACLE.validate_phase_chain(ORACLE.registered_specs())
        leftward = (
            _spec("a", (ORACLE.CorpusRow("shared.row", "ice", "executes_exactly", "t.a"),)),
            _spec("b", (ORACLE.CorpusRow("shared.row", "ice", "lane_divergent", "t.b"),)),
        )
        with self.assertRaisesRegex(ORACLE.OracleFailure, "forbidden status transition"):
            ORACLE.validate_phase_chain(leftward)
        disagreeing = (
            _spec("a", (ORACLE.CorpusRow("shared.row", "ice", "executes_exactly", "t.a"),)),
            _spec(
                "b",
                (
                    ORACLE.CorpusRow(
                        "shared.row", "silent_unguarded", "executes_exactly", "t.b"
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ORACLE.OracleFailure, "two different baselines"):
            ORACLE.validate_phase_chain(disagreeing)

    def test_a_leftward_move_in_phase_b_fails_a_phase_a_invocation(self) -> None:
        registry = {
            "a": _spec(
                "a", (ORACLE.CorpusRow("shared.row", "ice", "executes_exactly", "t.a"),)
            ),
            "b": _spec(
                "b", (ORACLE.CorpusRow("shared.row", "ice", "lane_divergent", "t.b"),)
            ),
        }

        def runner(argv: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            raise AssertionError(f"no command should run: {argv}")

        with self.assertRaisesRegex(ORACLE.OracleFailure, "forbidden status transition"):
            ORACLE.validate("a", runner=runner, registry=registry, require_clean=False)

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

    def test_every_recorded_corpus_row_has_a_named_executable_receipt(self) -> None:
        for phase, targets in (
            ("a", ORACLE.phase_a_targets("python")),
            ("b", ORACLE.phase_b_targets("python")),
        ):
            with self.subTest(phase=phase):
                spec = ORACLE.PHASE_REGISTRY[phase]
                ORACLE.validate_receipt_coverage(ORACLE.rows_at_exit(spec), targets)
                with self.assertRaisesRegex(ORACLE.OracleFailure, "no executable receipt"):
                    ORACLE.validate_receipt_coverage(
                        ORACLE.rows_at_exit(spec),
                        (ORACLE.TestTarget("empty", ("true",)),),
                    )

    def test_receipt_coverage_is_per_phase(self) -> None:
        # Phase A's targets do not cover phase B's recorded rows, and the
        # reverse, so the coverage check cannot be satisfied by another
        # phase's suite.
        with self.assertRaisesRegex(ORACLE.OracleFailure, "no executable receipt"):
            ORACLE.validate_receipt_coverage(
                ORACLE.rows_at_exit(ORACLE.PHASE_REGISTRY["b"]),
                ORACLE.phase_a_targets("python"),
            )

    def test_host_actualization_owner_matrix_is_in_the_automatic_gate(self) -> None:
        targets = {target.id: target for target in ORACLE.phase_a_targets("python")}
        target = targets["host_actualization"]
        self.assertEqual(
            target.argv,
            (
                "cargo",
                "test",
                "-p",
                "chelis-ir",
                "--lib",
                "host::tests::tensor_helper_actualization_declines_input_axis_for_shrink_and_stride",
                "--",
                "--exact",
                "--nocapture",
            ),
        )
        self.assertEqual(
            target.expected_tests,
            (
                "host::tests::tensor_helper_actualization_declines_input_axis_for_shrink_and_stride",
            ),
        )

    def test_shared_targets_are_deduplicated_by_command(self) -> None:
        shared = ORACLE.self_test_target("python")
        self.assertIn(shared, ORACLE.phase_a_targets("python"))
        self.assertIn(shared, ORACLE.phase_b_targets("python"))
        combined = ORACLE.phase_a_targets("python") + ORACLE.phase_b_targets("python")
        deduped = ORACLE.dedupe_targets(combined)
        self.assertEqual(
            [target.argv for target in deduped].count(shared.argv), 1
        )

    def test_validate_runs_every_target_and_prints_exact_head_digest(self) -> None:
        targets = ORACLE.phase_a_targets("python")
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
        self.assertEqual(digest, FROZEN_PHASE_A_DIGEST)
        self.assertEqual(seen[2:], [target.argv for target in targets])
        self.assertEqual(output.getvalue().splitlines()[-1], ORACLE.PASS_MARKER)

    def test_an_unregistered_phase_cannot_report_success(self) -> None:
        for phase in ("c", "final"):
            with self.subTest(phase=phase):
                with self.assertRaisesRegex(
                    ORACLE.OracleFailure, "not implemented|requires every slice phase"
                ):
                    ORACLE.validate(phase)
        self.assertNotIn("c", ORACLE.PHASE_REGISTRY)

    def test_a_phase_with_a_row_short_of_its_exit_state_cannot_report_success(self) -> None:
        short = ORACLE.CorpusRow("short.row", "ice", "ice", "t.short")
        done = ORACLE.CorpusRow("done.row", "ice", "executes_exactly", "t.done")
        target = ORACLE.TestTarget("t", ("true",), ("done", "short"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            rows = tuple(sorted((short, done), key=lambda row: row.id))
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "corpus_sha256": ORACLE.corpus_digest(rows, "b"),
                        "rows": [
                            {
                                "id": row.id,
                                "baseline": row.baseline,
                                "phase_b": row.exit_state,
                                "receipt": row.receipt,
                            }
                            for row in rows
                        ],
                    }
                )
            )
            registry = {
                "b": _spec("b", rows, baseline_path=path, targets=(target,))
            }

            def runner(argv: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                command = tuple(argv)
                if command == ("git", "rev-parse", "HEAD"):
                    return subprocess.CompletedProcess(command, 0, "c" * 40 + "\n", "")
                if command == ("git", "status", "--porcelain"):
                    return subprocess.CompletedProcess(command, 0, "", "")
                return subprocess.CompletedProcess(
                    command, 0, "test done ... ok\ntest short ... ok\n", ""
                )

            with self.assertRaisesRegex(
                ORACLE.OracleFailure, r"short of an exit state: b:short\.row"
            ):
                ORACLE.validate("b", runner=runner, registry=registry, require_clean=False)

            # The positive control: the same registry passes once the row
            # reaches an exit state.
            exited = ORACLE.CorpusRow(
                "short.row", "ice", "typed_unsupported(#1482)", "t.short"
            )
            rows_exited = tuple(sorted((exited, done), key=lambda row: row.id))
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "corpus_sha256": ORACLE.corpus_digest(rows_exited, "b"),
                        "rows": [
                            {
                                "id": row.id,
                                "baseline": row.baseline,
                                "phase_b": row.exit_state,
                                "receipt": row.receipt,
                            }
                            for row in rows_exited
                        ],
                    }
                )
            )
            registry = {
                "b": _spec("b", rows_exited, baseline_path=path, targets=(target,))
            }
            output = io.StringIO()
            with redirect_stdout(output):
                ORACLE.validate(
                    "b", runner=runner, registry=registry, require_clean=False
                )
            self.assertEqual(
                output.getvalue().splitlines()[-1], ORACLE.PASS_MARKER
            )

    def test_phase_b_is_registered_but_not_yet_at_exit(self) -> None:
        # PR B1 records two moves and leaves the rest of Slice B's rows at
        # their main baseline, so `--phase b` must fail honestly.
        shortfall = ORACLE.exit_shortfall(ORACLE.PHASE_REGISTRY["b"])
        self.assertNotIn("ir.axis_source.cardinality", shortfall)
        self.assertNotIn("shrink.elementwise_const.build", shortfall)
        self.assertIn("expand.kept_axis.op_declared_source.c", shortfall)
        self.assertIn("expand.kept_axis.op_declared_source.eval", shortfall)
        self.assertEqual(ORACLE.exit_shortfall(ORACLE.PHASE_REGISTRY["a"]), ())

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
