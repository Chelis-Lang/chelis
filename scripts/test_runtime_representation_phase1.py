"""Spec-derived Phase 1 receipt controls: selection is not execution."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock
from xml.sax.saxutils import escape
from scripts import runtime_representation_phase1 as oracle


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(mock.patch.dict(oracle.os.environ))
        oracle.os.environ.pop('CARGO_TARGET_DIR', None)

    def test_fixture_target_isolated_from_inherited_target_and_preserves_foreign_junit(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'external-target'
            sentinel = target / 'nextest' / oracle.PROFILE / 'junit.xml'
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b'foreign execution receipt')
            env = dict(oracle.os.environ, CARGO_TARGET_DIR=str(target))
            result = oracle.subprocess.run(
                [oracle.sys.executable, '-m', 'unittest',
                 'scripts.test_runtime_representation_phase1.ReceiptTests.test_selection_requires_current_binary_and_exact_nonempty_census',
                 'scripts.test_runtime_representation_phase1.ReceiptTests.test_fresh_execution_cannot_reuse_old_junit'],
                cwd=oracle.ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(sentinel.read_bytes(), b'foreign execution receipt')

    def test_native_controls_reject_a_runtime_directory_and_a_missing_compiler(self):
        watched = ('CHELIS_RUNTIME_DIR', 'CHELIS_TEST_CC')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            junit = root / 'junit.xml'

            def run_controls(evidence, runtime_directory):
                controls = iter(oracle.native_controls())
                current, observed = {}, []

                def command(argv, command_root, command_evidence, label, *, expected_exit=0):
                    command_evidence.mkdir(parents=True, exist_ok=True)
                    if label == 'list':
                        current['control'] = next(controls)
                    kind = current['control']['kind']
                    observed.append((label, kind, {name: oracle.os.environ.get(name) for name in watched}))
                    if label == 'run':
                        self.assertEqual(expected_exit, 100)
                        suite, case = current['control']['required'][0].rsplit('::', 1)
                        text = {'missing-c-compiler': 'No such file or directory'}.get(kind) or runtime_directory(evidence)
                        junit.write_text(f'<testsuites><testsuite name="{suite}"><testcase name="{case}">'
                                         f'<failure>{escape(text)}</failure></testcase></testsuite></testsuites>')
                    return '{}'

                with (mock.patch.object(oracle, 'command', side_effect=command),
                      mock.patch.object(oracle, 'selection', side_effect=lambda packet, root, expected: (expected, {})),
                      mock.patch.object(oracle, 'junit_path', return_value=junit),
                      redirect_stdout(io.StringIO())):
                    oracle.os.environ.pop('CHELIS_RUNTIME_DIR', None)
                    oracle.os.environ.pop('CHELIS_TEST_CC', None)
                    try:
                        return oracle.execute_native_controls(evidence), observed
                    finally:
                        self.assertNotIn('CHELIS_RUNTIME_DIR', oracle.os.environ)

            def rejected(evidence):
                bad = evidence / 'empty-runtime'
                return (f'stderr="error: CHELIS_RUNTIME_DIR is set ({bad}), but chelis stages the runtime '
                        'built into it and never takes one from a directory. Unset CHELIS_RUNTIME_DIR"')

            evidence = root / 'controls'
            receipts, observed = run_controls(evidence, rejected)
            self.assertEqual([row['outcome'] for row in receipts], ['rejected-by-execution'] * len(receipts))
            empty = evidence / 'empty-runtime/libchelis_runtime.a'
            self.assertEqual(empty.read_bytes(), b'!<arch>\n')
            expected = {
                'runtime-directory-rejected': {'CHELIS_RUNTIME_DIR': str(empty.parent), 'CHELIS_TEST_CC': None},
                'missing-c-compiler': {'CHELIS_RUNTIME_DIR': None, 'CHELIS_TEST_CC': str(evidence / 'missing-compiler')},
            }
            self.assertEqual({kind for _, kind, _ in observed}, set(expected))
            for label, kind, environment in observed:
                with self.subTest(label=label, kind=kind):
                    self.assertEqual(environment, expected[kind] if label == 'run' else
                                     {'CHELIS_RUNTIME_DIR': None, 'CHELIS_TEST_CC': None})
            # A CLI that honors the directory fails at the linker; one that both
            # rejects and links is not a rejection before staging.
            honored = lambda evidence: 'undefined reference to `chelis_alloc`'
            both = lambda evidence: rejected(evidence) + ' undefined reference to `chelis_alloc`'
            for name, outcome in (('honored', honored), ('linked', both)):
                with self.subTest(outcome=name), self.assertRaises(oracle.OracleFailure):
                    run_controls(root / name, outcome)

    def test_expected_mutation_failure_requires_the_exact_assertion_case(self):
        selected = ['p::contract::negative']
        valid = '<testsuites><testsuite name="p::contract"><testcase name="negative"><failure>assertion failed: distinct capacities</failure></testcase></testsuite></testsuites>'
        self.assertEqual(oracle.execution(valid, selected, failure_text='assertion failed: distinct capacities'), selected)
        for invalid in (valid.replace('distinct capacities', 'toolchain unavailable'),
                        valid.replace('name="negative"', 'name="unrelated"'),
                        valid.replace('<failure>assertion failed: distinct capacities</failure>', ''),
                        valid.replace('failure', 'error')):
            with self.subTest(xml=invalid), self.assertRaises(oracle.OracleFailure):
                oracle.execution(invalid, selected, failure_text='assertion failed: distinct capacities')

    def test_manifest_cannot_erase_a_selected_negative_or_mutation(self):
        valid = b'{"schema": 1, "legs": [{"selected": ["positive", "negative", "mutation"]}]}\n'
        digest = oracle.hashlib.sha256(valid).hexdigest()
        self.assertEqual(oracle.frozen_manifest(valid, digest)['schema'], 1)
        for changed in (valid.replace(b', "negative"', b''), valid.replace(b', "mutation"', b''), b'{}'):
            with self.subTest(changed=changed), self.assertRaises(oracle.OracleFailure):
                oracle.frozen_manifest(changed, digest)

    def test_manifest_requires_schema_two_sorted_identity_floors(self):
        packet = oracle.frozen_manifest(
            oracle.MANIFEST.read_bytes(),
            oracle.MANIFEST_SHA256,
        )
        oracle.validate_manifest(packet)
        for mutation in (
            'schema',
            'command',
            'empty',
            'legacy',
            'unsorted',
            'nonstring',
            'python',
        ):
            changed = copy.deepcopy(packet)
            if mutation == 'schema':
                changed['schema'] = 1
            elif mutation == 'command':
                changed['legs'][0]['args'].append('--changed')
            elif mutation == 'empty':
                changed['legs'][0]['required'] = []
            elif mutation == 'legacy':
                changed['legs'][0]['selected'] = changed['legs'][0].pop('required')
            elif mutation == 'unsorted':
                changed['legs'][0]['required'] = ['z', 'a']
            elif mutation == 'nonstring':
                changed['legs'][0]['required'] = [1]
            else:
                changed['python_required'] = []
            with self.subTest(mutation=mutation), self.assertRaises(oracle.OracleFailure):
                oracle.validate_manifest(changed)

    def test_exact_reduction_final_forms_have_a_frozen_phase_one_execution_leg(self):
        packet = oracle.frozen_manifest(
            oracle.MANIFEST.read_bytes(),
            oracle.MANIFEST_SHA256,
        )
        rows = {row["name"]: row for row in packet["legs"]}
        row = rows["exact C reduction stored-width and runtime-empty execution"]
        self.assertIn("issue_1281_exact_reductions", row["args"])
        self.assertIn(
            "chelis-backend-c::issue_1281_exact_reductions::"
            "runtime_empty_global_extrema_and_argument_reductions_trap_domain",
            row["required"],
        )
        self.assertIn(
            "chelis-backend-c::issue_1281_exact_reductions::"
            "global_extrema_and_argument_reductions_execute_at_every_remaining_storage_width",
            row["required"],
        )
        self.assertEqual(
            next(
                args
                for name, args in oracle.phase1_legs()
                if name == row["name"]
            ),
            tuple(row["args"]),
        )

    def test_list_map_numeric_forms_have_frozen_positive_and_negative_execution(self):
        packet = oracle.frozen_manifest(
            oracle.MANIFEST.read_bytes(),
            oracle.MANIFEST_SHA256,
        )
        rows = {row["name"]: row for row in packet["legs"]}
        expected = {
            "exact C List-map capture and ordered cotangent execution": [
                "chelis-backend-c::issue_570_runtime_iota::"
                "ordered_cotangent_native_groups_check_actual_column_lengths",
            ],
            "exact List-map capture compiled C parity and negative controls": [
                "chelis-cli::issue_2419_range_tensor_ad::"
                "captured_cotangent_keeps_the_false_forward_range_claim",
                "chelis-cli::issue_2419_range_tensor_ad::"
                "runtime_capture_accumulation_preserves_the_executed_consumer_tree",
                "chelis-cli::issue_2419_range_tensor_ad::"
                "runtime_capture_tree_preserves_inactive_and_empty_rows",
                "chelis-cli::issue_2419_range_tensor_ad::"
                "runtime_capture_tree_rounds_each_pair_at_the_capture_dtype",
            ],
        }
        for name, identities in expected.items():
            with self.subTest(name=name):
                row = rows[name]
                self.assertEqual(row["required"], identities)
                self.assertEqual(
                    tuple(row["args"]),
                    next(args for leg_name, args in oracle.phase1_legs() if leg_name == name),
                )
                altered = copy.deepcopy(packet)
                altered["legs"] = [leg for leg in altered["legs"] if leg["name"] != name]
                with self.assertRaises(oracle.OracleFailure):
                    oracle.validate_manifest(altered)
                altered = copy.deepcopy(packet)
                next(leg for leg in altered["legs"] if leg["name"] == name)["required"] = identities[:-1]
                with self.assertRaises(oracle.OracleFailure):
                    oracle.frozen_manifest(
                        oracle.json.dumps(altered).encode(),
                        oracle.MANIFEST_SHA256,
                    )
                with self.assertRaises(oracle.OracleFailure):
                    oracle.require_frozen_selection(identities[:-1], identities)

    def test_key_callable_inherited_leg_preserves_execution_and_rejection_floors(self):
        packet = oracle.frozen_manifest(oracle.MANIFEST.read_bytes(), oracle.MANIFEST_SHA256)
        name = 'checked key callable scalar and tensor C execution'
        row = next(row for row in packet['legs'] if row['name'] == name)
        self.assertEqual(row['args'], list(dict(oracle.phase1_legs())[name]))
        self.assertIn('ownership-ledger', row['args'])
        for test in (
            'unannotated_key_builtin_aliases_execute_in_eval_and_c',
            'locally_aggregated_key_builtin_alias_rejects_wrong_operand',
            'exported_aggregate_of_key_callables_rejects_before_public_c_abi',
            'typed_ordinary_function_tuple_remains_a_loud_c_rejection',
        ):
            identity = 'chelis-compiler-api::key_tensor_forms::' + test
            self.assertIn(identity, row['required'])
            with self.subTest(missing=identity), self.assertRaisesRegex(oracle.OracleFailure, 'lost identities'):
                oracle.require_frozen_selection([item for item in row['required'] if item != identity], row['required'])
        changed = copy.deepcopy(packet)
        changed['legs'] = [row for row in changed['legs'] if row['name'] != name]
        with self.assertRaisesRegex(oracle.OracleFailure, 'leg inventory drifted'):
            oracle.validate_manifest(changed)

    def test_integer_unary_inherited_legs_preserve_exact_positive_and_negative_floors(self):
        packet = oracle.frozen_manifest(oracle.MANIFEST.read_bytes(), oracle.MANIFEST_SHA256)
        rows = {row['name']: row for row in packet['legs']}
        expected = {
            'integer-to-float exact finalization execution': [
                'chelis-backend-c::integer_float::tests::emitted_integer_rounding_matches_finalizer_without_double_rounding',
                'chelis-backend-c::integer_float::tests::integer_rounding_rejects_non_float_target',
            ],
            'integer unary device lowering and trap controls': [
                'chelis-backend-hip::integer_abs::emitted_hip_integer_kernels_execute_exactly_with_device_intrinsic_shims',
                'chelis-backend-hip::integer_abs::integer_constants_stay_exact_before_abs',
                'chelis-backend-hip::integer_abs::lowered_integer_abs_gradient_emits_checked_hip_kernel_and_typed_cast',
                'chelis-backend-hip::integer_abs::uncanonicalized_integer_rounding_is_rejected_at_the_ir_boundary',
                'chelis-backend-metal::integer_abs_guard::activated_integer_abs_is_refused_before_emission_without_a_gate',
                'chelis-backend-metal::integer_abs_guard::empty_pointwise_dispatches_are_omitted_without_omitting_output_allocations',
                'chelis-backend-metal::integer_abs_guard::exact_integer_constants_and_abs_cover_scalar_empty_and_rank_two_shapes',
                'chelis-backend-metal::integer_abs_guard::integer_abs_then_float_cast_has_distinct_typed_kernels_at_every_width',
                'chelis-backend-metal::integer_abs_guard::integer_abs_uses_checked_kernel_and_fused_abs_stays_rejected',
                'chelis-backend-metal::integer_abs_guard::literal_integer_unary_gradients_lower_without_zero_placeholders',
                'chelis-backend-metal::integer_abs_guard::lowered_integer_abs_gradient_reaches_typed_metal_kernels',
            ],
        }
        inherited = dict(oracle.phase1_legs())
        for name, required in expected.items():
            self.assertTrue(name in rows, name)
            self.assertEqual(rows[name]['required'], required)
            self.assertEqual(rows[name]['args'], list(inherited[name]))
            self.assertEqual(oracle.require_frozen_selection(required, required), [])
            removed_leg = copy.deepcopy(packet)
            removed_leg['legs'] = [row for row in removed_leg['legs'] if row['name'] != name]
            with self.subTest(removed_leg=name), self.assertRaisesRegex(oracle.OracleFailure, 'leg inventory drifted'):
                oracle.validate_manifest(removed_leg)
            for identity in required:
                with self.subTest(missing=identity), self.assertRaisesRegex(oracle.OracleFailure, 'lost identities'):
                    oracle.require_frozen_selection([case for case in required if case != identity], required)
                changed = copy.deepcopy(packet)
                next(row for row in changed['legs'] if row['name'] == name)['required'].remove(identity)
                with self.subTest(tampered_floor=identity), self.assertRaisesRegex(oracle.OracleFailure, 'reviewed selection digest differs'):
                    oracle.frozen_manifest((json.dumps(changed, indent=2) + '\n').encode(), oracle.MANIFEST_SHA256)

    def test_failed_process_cannot_publish_a_passing_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fallback = root / '.venv/bin/python'
            fallback.parent.mkdir(parents=True)
            fallback.symlink_to(Path(oracle.sys.executable))
            prefix = oracle.subprocess.check_output(
                [str(fallback), '-I', '-c', 'import sys; print(sys.prefix)'],
                text=True,
            ).strip()
            probe = (
                'import json, os; '
                'print(json.dumps({name: os.environ[name] '
                'for name in ("PYO3_PYTHON", "VIRTUAL_ENV")}))'
            )
            with mock.patch.dict(oracle.os.environ, {}, clear=True):
                output = oracle.command(
                    [oracle.sys.executable, '-c', probe],
                    root,
                    root / 'managed',
                    'environment',
                )
                self.assertEqual(json.loads(output), {
                    'PYO3_PYTHON': str(fallback),
                    'VIRTUAL_ENV': prefix,
                })
                with mock.patch.dict(
                    oracle.os.environ,
                    {'PYO3_PYTHON': str(root / 'missing-python')},
                ):
                    with self.assertRaisesRegex(
                        oracle.OracleFailure,
                        'PYO3_PYTHON.*missing-python',
                    ):
                        oracle.command(
                            [oracle.sys.executable, '-c', 'pass'],
                            root,
                            root / 'invalid',
                            'environment',
                        )
                with self.assertRaises(oracle.OracleFailure):
                    oracle.command(
                        [oracle.sys.executable, '-c', 'print("PASS"); raise SystemExit(1)'],
                        root,
                        root / 'failing',
                        'failing',
                    )
            self.assertEqual(
                json.loads((root / 'failing/failing.process.json').read_text())['returncode'],
                1,
            )

    def test_python_selection_executes_and_reports_additions_but_blocks_every_nonpass(self):
        class Fixture(unittest.TestCase):
            def test_positive(self):
                self.assertEqual(1 + 1, 2)
            def test_addition(self):
                self.assertEqual(2 + 2, 4)
            def test_failure(self):
                self.fail('wrong outcome')
            @unittest.skip('cannot count')
            def test_ignored(self):
                pass
        positive = Fixture('test_positive')
        addition = Fixture('test_addition')
        receipt = oracle.python_execution(
            unittest.TestSuite([positive, addition]),
            [positive.id()],
        )
        self.assertEqual(receipt, {
            'required': [positive.id()],
            'selected': sorted([positive.id(), addition.id()]),
            'additions': [addition.id()],
            'executed': sorted([positive.id(), addition.id()]),
        })
        invalid = (
            unittest.TestSuite([Fixture('test_addition')]),
            unittest.TestSuite([Fixture('test_positive'), Fixture('test_ignored')]),
            unittest.TestSuite([Fixture('test_positive'), Fixture('test_failure')]),
            unittest.TestSuite(),
            unittest.TestSuite([Fixture('test_positive'), Fixture('test_positive')]),
        )
        for suite in invalid:
            with (
                self.subTest(suite=suite),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
                self.assertRaises(oracle.OracleFailure),
            ):
                oracle.python_execution(suite, [positive.id()])

    def test_fresh_execution_cannot_reuse_old_junit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            junit = root / 'target/nextest' / oracle.PROFILE / 'junit.xml'
            junit.parent.mkdir(parents=True)
            junit.write_text('<testsuites><testsuite name="p"><testcase name="positive"/></testsuite></testsuites>')
            with mock.patch.object(oracle, 'ROOT', root), mock.patch.object(oracle, 'selection', return_value=(['p::positive'], {})), mock.patch.object(oracle, 'command', return_value='{}'):
                with self.assertRaisesRegex(oracle.OracleFailure, 'fresh execution'):
                    oracle.execute_leg('fixture', (), ['p::positive'], root / 'evidence')
            self.assertFalse(junit.exists())

    def test_junit_receipt_is_workspace_owned_when_build_target_is_external(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'checkout'
            external = Path(directory) / 'external-target'
            foreign = external / 'nextest' / oracle.PROFILE / 'junit.xml'
            foreign.parent.mkdir(parents=True)
            foreign.write_bytes(b'foreign receipt')
            workspace = root / 'target' / 'nextest' / oracle.PROFILE / 'junit.xml'
            with mock.patch.object(oracle, 'ROOT', root), mock.patch.dict(
                oracle.os.environ, {'CARGO_TARGET_DIR': str(external)}, clear=True
            ):
                self.assertEqual(oracle.junit_path(), workspace)
            self.assertEqual(foreign.read_bytes(), b'foreign receipt')

    def test_frozen_selection_accepts_additions_but_blocks_loss_or_rename(self):
        frozen = ['p::contract::negative', 'p::contract::positive']
        addition = 'p::contract::new_negative'
        self.assertEqual(oracle.require_frozen_selection(frozen, frozen), [])
        self.assertEqual(
            oracle.require_frozen_selection(sorted([*frozen, addition]), frozen),
            [addition],
        )
        for selected in (
            frozen[1:],
            [frozen[1], addition],
            [*frozen, frozen[0]],
            list(reversed(frozen)),
        ):
            with self.subTest(selected=selected), self.assertRaises(oracle.OracleFailure):
                oracle.require_frozen_selection(selected, frozen)
        for invalid_frozen in (
            [],
            [frozen[0], frozen[0]],
            list(reversed(frozen)),
            [1],
        ):
            with self.subTest(frozen=invalid_frozen), self.assertRaises(oracle.OracleFailure):
                oracle.require_frozen_selection(frozen, invalid_frozen)

    def test_runtime_manifest_rejects_stale_replaced_python_identities(self):
        packet = oracle.frozen_manifest(
            oracle.MANIFEST.read_bytes(),
            oracle.MANIFEST_SHA256,
        )
        selected = oracle.python_selection(oracle.python_suite())
        replacements = (
            (
                'scripts.test_runtime_representation_oracle.'
                'FrozenMutationContractTests.'
                'test_rejection_obligation_drift_moves_digest_and_fails_comparison',
                'scripts.test_runtime_representation_oracle.'
                'BaselineTests.'
                'test_freeze_digest_rejects_an_edited_coverage_manifest',
            ),
            (
                'scripts.test_runtime_representation_oracle.'
                'MutationContractTests.'
                'test_mutation_body_change_moves_the_runtime_manifest_and_frozen_projection',
                'scripts.test_runtime_representation_oracle.'
                'MutationContractTests.'
                'test_mutation_body_change_moves_the_freeze_digest',
            ),
        )
        for current, stale in replacements:
            with self.subTest(stale=stale):
                self.assertIn(current, packet['python_required'])
                self.assertIn(current, selected)
                self.assertNotIn(stale, selected)
                stale_required = sorted(
                    stale if identity == current else identity
                    for identity in packet['python_required']
                )
                with self.assertRaisesRegex(
                    oracle.OracleFailure,
                    re.escape(stale),
                ):
                    oracle.require_frozen_selection(selected, stale_required)

    def test_runtime_manifest_rejects_stale_replaced_native_identities(self):
        # The oracle's nextest listing proves the replacements are current;
        # here the leg's reviewed floor stands in for that listing.
        packet = oracle.frozen_manifest(
            oracle.MANIFEST.read_bytes(),
            oracle.MANIFEST_SHA256,
        )
        replacements = (
            (
                'chelis-backend-c::emit::tests::'
                'path_sensitive_uniform_reads_uint8_bool',
                'chelis-backend-c::emit::tests::'
                'path_sensitive_uniform_reads_float_backed_bool_and_gates_counter',
            ),
            (
                'chelis-backend-c::emit::tests::'
                'path_sensitive_uniform_reads_uint8_bool',
                'chelis-backend-c::emit::tests::'
                'path_sensitive_uniform_reads_uint8_bool_and_gates_counter',
            ),
            (
                'chelis-cli::capacity_census_tripwire::'
                'prepared_random_kernel_boundaries_have_exact_semantic_authority',
                'chelis-cli::capacity_census_tripwire::'
                'prepared_dropout_kernel_boundaries_have_exact_semantic_authority',
            ),
            (
                'chelis-cli::issue_1314_json_bigint_ledger::'
                'json_scratch_execution_detects_skipped_cleanup',
                'chelis-cli::issue_1314_json_bigint::'
                'json_scratch_execution_detects_skipped_cleanup',
            ),
        )
        for current, stale in replacements:
            with self.subTest(stale=stale):
                legs = [row['required'] for row in packet['legs']]
                self.assertFalse(any(stale in required for required in legs))
                owners = [required for required in legs if current in required]
                self.assertTrue(owners)
                for required in owners:
                    stale_required = sorted(
                        stale if identity == current else identity
                        for identity in required
                    )
                    with self.assertRaisesRegex(
                        oracle.OracleFailure,
                        re.escape(stale),
                    ):
                        oracle.require_frozen_selection(required, stale_required)

    def test_native_and_mutation_selections_remain_exact(self):
        expected = ['p::contract::negative']
        oracle.require_exact_selection(expected, expected)
        for selected in (
            [],
            [*expected, 'p::contract::addition'],
            [expected[0], expected[0]],
        ):
            with self.subTest(selected=selected), self.assertRaises(oracle.OracleFailure):
                oracle.require_exact_selection(selected, expected)

    def test_execution_receipt_names_and_reports_an_added_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            junit = root / 'junit.xml'
            frozen = ['p::contract::negative']
            addition = 'p::contract::new_negative'
            selected = sorted([*frozen, addition])
            xml = (
                '<testsuites><testsuite name="p::contract">'
                '<testcase name="negative"/><testcase name="new_negative"/>'
                '</testsuite></testsuites>'
            )

            def command(argv, command_root, evidence, label):
                self.assertEqual(command_root, root)
                evidence.mkdir(parents=True, exist_ok=True)
                if label == 'run':
                    junit.write_text(xml)
                return '{}'

            with (
                mock.patch.object(oracle, 'ROOT', root),
                mock.patch.object(oracle, 'command', side_effect=command),
                mock.patch.object(oracle, 'selection', return_value=(selected, {})),
                mock.patch.object(oracle, 'junit_path', return_value=junit),
            ):
                receipt = oracle.execute_leg(
                    'fixture',
                    (),
                    frozen,
                    root / 'evidence',
                )

            self.assertEqual(receipt['selected'], selected)
            self.assertEqual(receipt['additions'], [addition])
            self.assertEqual(
                receipt['executed'],
                [{'id': identity, 'outcome': 'passed'} for identity in selected],
            )

    def test_added_test_binary_is_hashed_and_checked_after_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            required_binary = root / 'target/debug/deps/required'
            added_binary = root / 'target/debug/deps/added'
            required_binary.parent.mkdir(parents=True)
            required_binary.write_bytes(b'required')
            added_binary.write_bytes(b'added')

            def suite(identity, binary, package, case):
                return {
                    'binary-id': identity,
                    'binary-path': str(binary),
                    'package-name': package,
                    'cwd': str(root / 'crates' / package),
                    'status': 'listed',
                    'testcases': {
                        case: {
                            'ignored': False,
                            'filter-match': {'status': 'matches'},
                        },
                    },
                }

            packet = {
                'rust-build-meta': {'target-directory': str(root / 'target')},
                'rust-suites': {
                    'p::required': suite(
                        'p::required', required_binary, 'p', 'negative'
                    ),
                    'q::added': suite('q::added', added_binary, 'q', 'new_negative'),
                },
            }
            selected, artifacts = oracle.selection(packet, root)
            self.assertEqual(
                oracle.require_frozen_selection(
                    selected,
                    ['p::required::negative'],
                ),
                ['q::added::new_negative'],
            )
            self.assertEqual(
                set(artifacts),
                {str(required_binary.resolve()), str(added_binary.resolve())},
            )
            oracle.verify_artifacts(artifacts)
            added_binary.write_bytes(b'changed')
            with self.assertRaises(oracle.OracleFailure):
                oracle.verify_artifacts(artifacts)

    def test_selection_requires_current_binary_and_exact_nonempty_census(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'target/debug/deps/contract'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'compiled fixture')
            packet = {'rust-build-meta': {'target-directory': str(root / 'target')},
                      'rust-suites': {'p::contract': {'binary-id': 'p::contract',
                        'binary-path': str(binary), 'package-name': 'p',
                        'cwd': str(root / 'crates/p'), 'status': 'listed',
                        'testcases': {'positive': {'ignored': False, 'filter-match': {'status': 'matches'}},
                                      'negative': {'ignored': False, 'filter-match': {'status': 'matches'}}}}}}
            expected = ['p::contract::negative', 'p::contract::positive']
            self.assertEqual(oracle.selection(packet, root, expected)[0], expected)
            for change in ('empty', 'ignored', 'missing', 'external', 'wrong_census', 'addition'):
                changed = copy.deepcopy(packet)
                suite = changed['rust-suites']['p::contract']
                if change == 'empty': suite['testcases'] = {}
                if change == 'ignored': suite['testcases']['positive']['ignored'] = True
                if change == 'missing': suite['binary-path'] += '-missing'
                if change == 'external': suite['cwd'] = '/elsewhere/crates/p'
                if change == 'wrong_census': del suite['testcases']['negative']
                if change == 'addition':
                    suite['testcases']['addition'] = {
                        'ignored': False,
                        'filter-match': {'status': 'matches'},
                    }
                with self.subTest(change=change), self.assertRaises(oracle.OracleFailure):
                    oracle.selection(changed, root, expected)

    def test_selection_accepts_only_ignored_rows_when_explicitly_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'target/debug/deps/contract'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'compiled fixture')
            packet = {
                'rust-build-meta': {'target-directory': str(root / 'target')},
                'rust-suites': {
                    'p::contract': {
                        'binary-id': 'p::contract',
                        'binary-path': str(binary),
                        'package-name': 'p',
                        'cwd': str(root / 'crates/p'),
                        'status': 'listed',
                        'testcases': {
                            'ignored': {
                                'ignored': True,
                                'filter-match': {'status': 'matches'},
                            },
                            'active': {
                                'ignored': False,
                                'filter-match': {'status': 'mismatch'},
                            },
                        },
                    },
                },
            }
            expected = ['p::contract::ignored']
            self.assertEqual(
                oracle.selection(
                    packet,
                    root,
                    expected,
                    include_ignored=True,
                )[0],
                expected,
            )

    def test_execution_requires_one_passing_framework_case_per_selection(self):
        selected = ['p::contract::negative', 'p::contract::positive']
        valid = '<testsuites><testsuite name="p::contract"><testcase name="positive"/><testcase name="negative"/></testsuite></testsuites>'
        self.assertEqual(oracle.execution(valid, selected), selected)
        for invalid in ('<testsuites/>', valid.replace('<testcase name="negative"/>', ''),
                        valid.replace('<testcase name="positive"/>', '<testcase name="positive"><skipped/></testcase>'),
                        valid.replace('<testcase name="positive"/>', '<testcase name="positive"><failure/></testcase>'),
                        valid.replace('<testcase name="negative"/>', '<testcase name="positive"/>')):
            with self.subTest(xml=invalid), self.assertRaises(oracle.OracleFailure):
                oracle.execution(invalid, selected)

    def test_duplicate_json_fields_are_not_a_valid_selection(self):
        with self.assertRaises(oracle.OracleFailure):
            oracle.load_json('{"rust-suites": {}, "rust-suites": {}}')

    def test_public_command_rejects_development_shortcuts_before_source_validation(self):
        for option in ("--skip-mutations", "--regenerate"):
            result = oracle.subprocess.run([oracle.sys.executable, str(oracle.ROOT / "scripts/runtime_representation_oracle.py"), "--phase", "1", option], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Phase 1 requires every mutation", result.stderr)
            self.assertNotIn("PHASE 1: PASS", result.stdout)

    def test_phase_one_rejects_development_shortcuts(self):
        for argv in (['--phase', '1', '--skip-mutations'], ['--phase', '1', '--regenerate']):
            with self.subTest(argv=argv), self.assertRaises(oracle.OracleFailure):
                oracle.check_options(argv)
        oracle.check_options(['--phase', '1'])

    def test_missing_or_changed_binary_after_execution_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'binary'
            binary.write_bytes(b'first')
            before = oracle.artifact_hashes([binary])
            oracle.verify_artifacts(before)
            binary.write_bytes(b'second')
            with self.assertRaises(oracle.OracleFailure): oracle.verify_artifacts(before)
            binary.unlink()
            with self.assertRaises(oracle.OracleFailure): oracle.verify_artifacts(before)


if __name__ == '__main__':
    unittest.main()
