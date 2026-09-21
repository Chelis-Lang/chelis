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

    def test_runtime_archive_requires_current_cargo_artifact_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'target/debug/libchelis_runtime.a'
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b'current artifact')
            packet = {'reason': 'compiler-artifact',
                      'manifest_path': str(root / 'crates/chelis-runtime/Cargo.toml'),
                      'target': {'name': 'chelis_runtime', 'kind': ['rlib', 'staticlib'],
                                 'src_path': str(root / 'crates/chelis-runtime/src/lib.rs')},
                      'features': ['ownership-ledger'], 'filenames': [str(archive)]}
            with mock.patch.object(oracle, 'ROOT', root), mock.patch.dict(oracle.os.environ, {}, clear=True):
                self.assertEqual(oracle.runtime_artifact(json.dumps(packet)), archive.resolve())
                for defect in ('absent', 'ambiguous', 'other_source', 'other_target', 'missing_file', 'missing_ledger'):
                    changed = copy.deepcopy(packet)
                    if defect == 'other_source': changed['target']['src_path'] = '/old/src/lib.rs'
                    if defect == 'other_target': changed['filenames'] = ['/old/libchelis_runtime.a']
                    if defect == 'missing_file': changed['filenames'][0] += '-absent.a'
                    if defect == 'missing_ledger': changed['features'] = []
                    text = json.dumps(changed)
                    if defect == 'absent': text = '{}'
                    if defect == 'ambiguous': text += '\n' + text
                    with self.subTest(defect=defect), self.assertRaises(oracle.OracleFailure):
                        oracle.runtime_artifact(text)
                evidence = root / 'evidence'
                evidence.mkdir()
                with mock.patch.object(oracle, 'command', return_value=json.dumps(packet)):
                    with oracle.runtime_pin(evidence) as receipt:
                        pinned = evidence / 'runtime/libchelis_runtime.a'
                        self.assertEqual(oracle.os.environ['CHELIS_RUNTIME_DIR'], str(pinned.parent))
                        self.assertEqual(oracle.os.environ['CHELIS_RUNTIME_LIB'], str(pinned))
                        self.assertEqual(list(receipt['pinned_artifact']), [str(pinned)])
                    self.assertNotIn('CHELIS_RUNTIME_DIR', oracle.os.environ)
                    self.assertNotIn('CHELIS_RUNTIME_LIB', oracle.os.environ)

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
