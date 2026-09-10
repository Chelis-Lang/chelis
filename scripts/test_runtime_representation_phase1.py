"""Spec-derived Phase 1 receipt controls: selection is not execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from scripts import runtime_representation_phase1 as oracle


class ReceiptTests(unittest.TestCase):
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

    def test_failed_process_cannot_publish_a_passing_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(oracle.OracleFailure):
                oracle.command([oracle.sys.executable, '-c', 'print("PASS"); raise SystemExit(1)'], Path(directory), Path(directory), 'failing')
            self.assertEqual(json.loads((Path(directory) / 'failing.process.json').read_text())['returncode'], 1)

    def test_python_skip_missing_and_duplicate_success_fail_receipts(self):
        class Fixture(unittest.TestCase):
            def test_positive(self):
                self.assertEqual(1 + 1, 2)
            @unittest.skip('cannot count')
            def test_ignored(self):
                pass
        positive = Fixture('test_positive')
        oracle.python_execution(unittest.TestSuite([positive]), [positive.id()])
        with self.assertRaises(oracle.OracleFailure):
            oracle.python_execution(unittest.TestSuite([Fixture('test_ignored')]), [Fixture('test_ignored').id()])
        with self.assertRaises(oracle.OracleFailure):
            oracle.python_execution(unittest.TestSuite(), [positive.id()])
        with self.assertRaises(oracle.OracleFailure):
            oracle.python_execution(unittest.TestSuite([Fixture('test_positive'), Fixture('test_positive')]), [positive.id()])

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
            for change in ('empty', 'ignored', 'missing', 'external', 'wrong_census'):
                changed = copy.deepcopy(packet)
                suite = changed['rust-suites']['p::contract']
                if change == 'empty': suite['testcases'] = {}
                if change == 'ignored': suite['testcases']['positive']['ignored'] = True
                if change == 'missing': suite['binary-path'] += '-missing'
                if change == 'external': suite['cwd'] = '/elsewhere/crates/p'
                if change == 'wrong_census': del suite['testcases']['negative']
                with self.subTest(change=change), self.assertRaises(oracle.OracleFailure):
                    oracle.selection(changed, root, expected)

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
