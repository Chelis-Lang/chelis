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
            "guard.local.declaration_order.eval_c",
            "claim.literal.nested_and_unused.eval_c",
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
        # chelis#665's pair reaches exit in B2b-2, and the two lanes carry
        # DIFFERENT baselines, which is the point of the row rather than an
        # asymmetry to tidy away. The C lane aborted; the eval lane did not,
        # and its recorded `ice` baseline was corrected with the measurement
        # (`chelis eval --file` prints the program's shape and exits zero on
        # `3dc3f54f6`). The eval receipt is therefore a disposition lock and
        # the C receipt asserts byte-for-byte parity against it.
        c_row = by_id["expand.kept_axis.op_declared_source.c"]
        self.assertEqual(c_row.baseline, "ice")
        self.assertEqual(c_row.exit_state, ORACLE.EXECUTES)
        eval_row = by_id["expand.kept_axis.op_declared_source.eval"]
        self.assertEqual(eval_row.baseline, ORACLE.EXECUTES)
        self.assertEqual(eval_row.exit_state, ORACLE.EXECUTES)
        # The HIP prologue row is new in Slice B's second half and starts at
        # the same `require_load_source` panic, on the one lane that has it.
        hip = by_id["expand.op_declared_source.hip_prologue"]
        self.assertEqual(hip.baseline, "ice")
        # chelis#1742's labelling correction. Three C-lane receipts named the
        # wrong defect: #1374 IS the cross-tensor read under a NAMED claim and
        # #1376 IS the foreign claim over a same-tensor read, so the
        # `issue_1374_` receipt sat on the literal-control row and the
        # `issue_1376_` one on #1374's row. Every row now names the test its
        # own id describes, and each lane pair is a `_on_c`/`_on_eval` twin.
        # Disposition lock rather than a regression test: none of these rows
        # is at an exit state, so `validate_receipt_coverage` does not enforce
        # their receipts yet. What this pins is that the corrected binding
        # cannot silently swap back before the tests exist.
        for row_id, receipt in (
            (
                "expand.literal_claim.cross_tensor_read.c",
                "cli_slice_b.a_literal_claim_over_a_cross_tensor_read_traps_on_c",
            ),
            (
                "expand.literal_claim.cross_tensor_read.eval",
                "cli_slice_b.a_literal_claim_over_a_cross_tensor_read_traps_on_eval",
            ),
            (
                "expand.named_claim.cross_tensor_read.c",
                "cli_slice_b.issue_1374_cross_tensor_read_under_a_named_claim_is_guarded_on_c",
            ),
            (
                "expand.named_claim.cross_tensor_read.eval",
                "cli_slice_b.a_cross_tensor_read_under_a_named_claim_is_guarded_on_eval",
            ),
            (
                "expand.foreign_claim.same_tensor_set_axis.c",
                "cli_slice_b.issue_1376_same_tensor_read_under_a_foreign_claim_is_guarded_on_c",
            ),
            (
                "expand.foreign_claim.same_tensor_set_axis.eval",
                "cli_slice_b.a_same_tensor_read_under_a_foreign_claim_is_guarded_on_eval",
            ),
        ):
            self.assertEqual(by_id[row_id].receipt, receipt)

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
        # The names follow `--`, where the harness takes any number of
        # filters. Cargo takes one `[TESTNAME]` positional, so this is the
        # only placement that works for a row naming more than one test.
        self.assertEqual(
            target.argv,
            (
                "cargo",
                "test",
                "-p",
                "chelis-ir",
                "--lib",
                "--",
                "--exact",
                "--nocapture",
                "host::tests::tensor_helper_actualization_declines_input_axis_for_shrink_and_stride",
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
        # Slice B's rows reach exit one owning change at a time, so
        # `--phase b` must keep failing honestly while any remains short.
        shortfall = ORACLE.exit_shortfall(ORACLE.PHASE_REGISTRY["b"])
        self.assertNotIn("ir.axis_source.cardinality", shortfall)
        self.assertNotIn("shrink.elementwise_const.build", shortfall)
        # B2b-2's own pair left the shortfall with chelis#665's fix.
        self.assertNotIn("expand.kept_axis.op_declared_source.c", shortfall)
        self.assertNotIn("expand.kept_axis.op_declared_source.eval", shortfall)
        # A row whose owning slice has not started, named so this assertion
        # does not go stale every time a sibling change moves a row: #1266's
        # record projection is the provenance work B2b-2 explicitly defers.
        self.assertIn("expand.record_projection.size", shortfall)
        self.assertNotEqual(shortfall, ())
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

    # --- chelis#1742: the manifest is the single home of every expectation ---

    def test_every_cargo_target_is_backed_by_a_manifest_row_and_no_row_is_orphaned(
        self,
    ) -> None:
        # Disposition lock, not a regression test: before chelis#1742 the
        # expectations were Python tuples and there was no manifest to
        # reconcile against. What this pins is that the manifest stays the
        # only home for them, so a target cannot regrow a private tuple and a
        # row cannot linger after its target is retired.
        manifest = ORACLE.load_target_manifest()
        rows = {(row["phase"], row["id"]): row for row in manifest}
        covered: set[tuple[str, str]] = set()
        for phase in ("a", "b"):
            for target in ORACLE.PHASE_REGISTRY[phase].targets():
                if not target.expected_tests:
                    # The oracle's own unit suite is the one target with no
                    # per-test expectation, so it has nothing to drift.
                    self.assertEqual(target.id, "self_tests")
                    continue
                key = (phase, target.id)
                self.assertIn(key, rows, f"{target.id} has no manifest row")
                row = rows[key]
                self.assertEqual(target.expected_tests, tuple(row["expected"]))
                self.assertEqual(target.argv, ORACLE.target_argv(row))
                self.assertEqual(target.list_only, bool(row.get("list_only", False)))
                covered.add(key)
        self.assertEqual(covered, set(rows), "manifest rows with no target")

    def test_no_generated_command_passes_cargo_more_than_one_positional(self) -> None:
        # Regression test, measured: the eight-name `exec_c` row was
        # generated with its filters before `--`, and cargo rejected the
        # whole command with "unexpected argument" in 0.0s, so the target
        # never ran and the oracle reported a failed command rather than a
        # receipt. Cargo accepts a single `[TESTNAME]`; the harness after
        # `--` accepts any number.
        for phase in ("a", "b"):
            for target in ORACLE.manifest_targets(phase):
                argv = list(target.argv)
                separator = argv.index("--")
                head = argv[:separator]
                if "--test" in head:
                    positionals = head[head.index("--test") + 2 :]
                else:
                    positionals = head[head.index("--lib") + 1 :]
                self.assertLessEqual(
                    len(positionals),
                    1,
                    f"{target.id}: cargo takes one TESTNAME, got {positionals}",
                )
                self.assertTrue(
                    all(not item.startswith("-") for item in positionals),
                    f"{target.id}: {positionals}",
                )

    def test_a_manifest_row_naming_a_missing_file_fails_closed(self) -> None:
        # Regression test for the rename this issue is about: a row whose
        # source moved must stop the oracle rather than silently describe a
        # file that is not there.
        payload = json.loads(ORACLE.TARGETS_PATH.read_text())
        payload["targets"][0]["file"] = (
            f"crates/{payload['targets'][0]['package']}/tests/no_such_target.rs"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ORACLE.OracleFailure, "missing source file"):
                ORACLE.load_target_manifest(path)

    def test_a_manifest_row_with_the_wrong_shape_fails_closed(self) -> None:
        # Negative parity for the case above: every structural defect the
        # loader can see is fatal, not a skipped row.
        base = json.loads(ORACLE.TARGETS_PATH.read_text())

        def reject(mutate, pattern: str) -> None:
            payload = json.loads(json.dumps(base))
            mutate(payload)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "targets.json"
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ORACLE.OracleFailure, pattern):
                    ORACLE.load_target_manifest(path)

        reject(lambda p: p["targets"][0].pop("expected"), "wrong fields")
        reject(lambda p: p["targets"][0].update(surprise=1), "wrong fields")
        reject(
            lambda p: p["targets"][0].update(selector={"mode": "prefix", "value": "x"}),
            "unknown selector",
        )
        reject(lambda p: p["targets"][0].update(expected=[]), "at least one expected")
        reject(
            lambda p: p["targets"][0].update(expected=["b", "a"]),
            "sorted and unique",
        )
        reject(lambda p: p["targets"].append(p["targets"][0]), "repeats target")
        reject(
            lambda p: p["targets"][0].update(
                list_only=True, selector={"mode": "all"}
            ),
            "listed, which needs a substring selector",
        )
        reject(lambda p: p.update(schema_version=2), "unsupported target manifest")
        reject(lambda p: p.update(targets=[]), "non-empty targets list")

    def test_the_checked_baselines_are_byte_identical_to_their_rendering(self) -> None:
        # Disposition lock: both baselines are derived data, so a hand edit
        # that happens to keep a consistent digest is still a defect. The
        # writer that produces these bytes is `--write-baseline`.
        for phase in ("a", "b"):
            with self.subTest(phase=phase):
                spec = ORACLE.PHASE_REGISTRY[phase]
                self.assertEqual(
                    ORACLE.render_baseline(spec), spec.baseline_path.read_text()
                )

    def test_allow_shortfall_reports_rows_without_claiming_success(self) -> None:
        # Regression test for the nightly wiring: phase B's 19 recorded short
        # rows are the tracker's published state, not a drift, so a job must
        # be able to enforce every receipt while they land. What must never
        # happen is that the run then claims PASS.
        rows = (
            ORACLE.CorpusRow("done.row", "ice", "executes_exactly", "t.done"),
            ORACLE.CorpusRow("short.row", "ice", "ice", "t.short"),
        )
        target = ORACLE.TestTarget("t", ("true",), ("done", "short"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            spec = _spec("a", rows, baseline_path=path, targets=(target,))
            path.write_text(ORACLE.render_baseline(spec))
            registry = {"a": spec}

            def runner(argv, **_kwargs):
                command = tuple(argv)
                if command == ("git", "rev-parse", "HEAD"):
                    return subprocess.CompletedProcess(command, 0, "c" * 40 + "\n", "")
                if command == ("git", "status", "--porcelain"):
                    return subprocess.CompletedProcess(command, 0, "", "")
                return subprocess.CompletedProcess(
                    command, 0, "test done ... ok\ntest short ... ok\n", ""
                )

            with self.assertRaisesRegex(ORACLE.OracleFailure, "short of an exit state"):
                ORACLE.validate("a", runner=runner, registry=registry)

            output = io.StringIO()
            with redirect_stdout(output):
                ORACLE.validate(
                    "a", runner=runner, registry=registry, allow_shortfall=True
                )
            printed = output.getvalue().splitlines()
            self.assertEqual(printed[-1], ORACLE.SHORT_MARKER)
            self.assertNotIn(ORACLE.PASS_MARKER, printed)
            self.assertIn("runtime_extent_rows_short=1", printed)
            self.assertIn("runtime_extent_rows_short_list=a:short.row", printed)

    def test_the_final_completion_oracle_refuses_to_allow_a_shortfall(self) -> None:
        # Regression test: `--phase final` is chelis#1277's completion
        # command, so the flag that lets the nightly hold a still-landing
        # phase must be refused there rather than silently accepted. The
        # refusal precedes the unregistered-phase check, so it is the reason
        # reported rather than the missing phase `c`.
        with self.assertRaisesRegex(
            ORACLE.OracleFailure, "completion oracle and cannot allow a row shortfall"
        ):
            ORACLE.validate("final", allow_shortfall=True)
        self.assertEqual(ORACLE.main(["--phase", "final", "--allow-shortfall"]), 1)
        # Negative parity: a slice phase still accepts the flag, so the
        # refusal is about `final` and not about the flag existing.
        self.assertNotIn("final", ORACLE.SLICE_PHASES)

    def test_allow_shortfall_does_not_excuse_a_failing_receipt(self) -> None:
        # Negative parity: the flag downgrades the row shortfall and nothing
        # else. A drifted receipt under it must still fail the run, which is
        # the whole reason the nightly job can hold phase B at all.
        rows = (ORACLE.CorpusRow("done.row", "ice", "executes_exactly", "t.done"),)
        target = ORACLE.TestTarget("t", ("true",), ("done",))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            spec = _spec("a", rows, baseline_path=path, targets=(target,))
            path.write_text(ORACLE.render_baseline(spec))

            def runner(argv, **_kwargs):
                command = tuple(argv)
                if command == ("git", "rev-parse", "HEAD"):
                    return subprocess.CompletedProcess(command, 0, "c" * 40 + "\n", "")
                if command == ("git", "status", "--porcelain"):
                    return subprocess.CompletedProcess(command, 0, "", "")
                return subprocess.CompletedProcess(
                    command, 0, "test renamed ... ok\n", ""
                )

            with self.assertRaisesRegex(ORACLE.OracleFailure, "test receipt drift"):
                ORACLE.validate(
                    "a",
                    runner=runner,
                    registry={"a": spec},
                    allow_shortfall=True,
                )

    def test_writing_a_baseline_cannot_move_the_frozen_phase_a_digest(self) -> None:
        # The writer regenerates derived data; it is not an escape hatch from
        # the freeze. Rendering phase A and re-reading it must still produce
        # the digest this file pins.
        spec = ORACLE.PHASE_REGISTRY["a"]
        rendered = json.loads(ORACLE.render_baseline(spec))
        self.assertEqual(rendered["corpus_sha256"], FROZEN_PHASE_A_DIGEST)


if __name__ == "__main__":
    unittest.main()
