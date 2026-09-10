"""Execution receipts for the runtime-representation Phase 1 acceptance surface.

The manifest freezes selection, not outcomes. Every invocation builds and lists
current test artifacts and obtains fresh framework results; saved PASS packets
are outputs only. Phase 0's source inventory and real mutation probes remain
mandatory prerequisites. Device descriptors and lane element access are later
phases and are not claimed by this host/C vocabulary receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import uuid
import xml.etree.ElementTree as ET

from scripts import runtime_representation_oracle as phase0
from scripts.dtype_builtin_atom_closure_oracle import source_identity

ROOT = phase0.REPO_ROOT
OracleFailure = phase0.OracleFailure
PROFILE = 'runtime-representation'
MANIFEST = ROOT / 'spec/design/runtime_representation_phase1_tests.json'
MANIFEST_SHA256 = '98e708189bcabb4f34198dfaa3f35d805913cacfc4db21907cb11282b8dbb436'
MANUAL_TEST = 'an_allocation_above_int32_elements_reports_its_true_extent'


def load_json(source: str):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise OracleFailure(f'duplicate JSON field: {name}')
            result[name] = value
        return result
    try:
        return json.loads(source, object_pairs_hook=unique)
    except (ValueError, TypeError) as error:
        raise OracleFailure(f'invalid JSON receipt: {error}') from error


def check_options(argv):
    if '--skip-mutations' in argv or '--regenerate' in argv:
        raise OracleFailure('Phase 1 requires every mutation and current execution; development shortcuts cannot pass')


def frozen_manifest(data, digest):
    if hashlib.sha256(data).hexdigest() != digest:
        raise OracleFailure('Phase 1 reviewed selection digest differs')
    return load_json(data.decode())


class PythonReceipts(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.executed.append(test.id())


def python_execution(suite, expected):
    result = unittest.TextTestRunner(verbosity=1, resultclass=PythonReceipts).run(suite)
    if (not result.wasSuccessful() or not expected or len(set(expected)) != len(expected)
            or len(result.executed) != len(set(result.executed))
            or sorted(result.executed) != sorted(expected)):
        raise OracleFailure('Python selection lacks exact successful framework execution')
    return sorted(result.executed)


def python_suite():
    return unittest.defaultTestLoader.loadTestsFromNames([
        'scripts.test_runtime_representation_phase1',
        'scripts.test_runtime_representation_oracle',
    ])


def artifact_hashes(paths):
    try:
        return {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}
    except OSError as error:
        raise OracleFailure(f'cannot read test artifact: {error}') from error


def verify_artifacts(expected):
    if artifact_hashes(expected) != expected:
        raise OracleFailure('test artifact changed between selection and execution')


def selection(packet, root: Path, expected):
    selected, binaries = [], []
    target = Path(os.environ.get('CARGO_TARGET_DIR', root / 'target')).resolve()
    try:
        if Path(packet['rust-build-meta']['target-directory']).resolve() != target:
            raise OracleFailure('nextest selected a different target directory')
        for identity, suite in packet['rust-suites'].items():
            if identity != suite['binary-id'] or suite['status'] != 'listed':
                raise OracleFailure('invalid nextest binary identity/status')
            binary = Path(suite['binary-path']).resolve()
            if not binary.is_file() or not binary.is_relative_to(target):
                raise OracleFailure('test binary missing or outside current target')
            package = root / 'crates' / suite['package-name']
            if Path(suite['cwd']).resolve() != package.resolve():
                raise OracleFailure('test binary belongs to another source checkout')
            used = False
            for name, case in suite['testcases'].items():
                if case['filter-match']['status'] != 'matches':
                    continue
                if case['ignored'] is not False:
                    raise OracleFailure(f'ignored test selected: {identity}::{name}')
                selected.append(f'{identity}::{name}')
                used = True
            if used:
                binaries.append(binary)
    except (KeyError, TypeError, ValueError) as error:
        raise OracleFailure(f'malformed nextest selection: {error}') from error
    selected.sort()
    if not selected or len(set(selected)) != len(selected) or selected != expected:
        raise OracleFailure('empty, duplicate, or drifted frozen test selection')
    return selected, artifact_hashes(binaries)


def execution(xml: str, selected, *, failure_text=None):
    try:
        root = ET.fromstring(xml)
        if root.tag != 'testsuites':
            raise OracleFailure('missing nextest JUnit root')
        executed = []
        for suite in root.findall('testsuite'):
            for case in suite.findall('testcase'):
                forbidden = ('error', 'skipped', 'rerunFailure', 'rerunError', 'flakyFailure', 'flakyError')
                if any(case.find(tag) is not None for tag in forbidden):
                    raise OracleFailure('non-passing framework outcome')
                failed = case.find('failure') is not None
                if failure_text is None and failed:
                    raise OracleFailure('non-passing framework outcome')
                if failure_text is not None and (not failed or failure_text not in ''.join(case.itertext())):
                    raise OracleFailure('mutation did not fail its exact assertion')
                executed.append(f"{suite.attrib['name']}::{case.attrib['name']}")
    except (ET.ParseError, KeyError) as error:
        raise OracleFailure(f'malformed framework execution: {error}') from error
    executed.sort()
    if not selected or len(set(executed)) != len(executed) or executed != selected:
        raise OracleFailure('empty, duplicate, missing or unexpected framework execution')
    return executed


def phase1_legs():
    # Inherit the reviewed behavioral/mutation obligations without maintaining a
    # second hand-copied list. The checked-in per-leg census freezes that list.
    legs = []
    for leg in phase0.phase0_legs():
        argv = list(leg.argv)
        if argv[:3] == ['cargo', 'nextest', 'run']:
            args = argv[3:]
        elif '--doc' in argv:
            # The public opaque-key compile failures remain a supporting phase-0
            # obligation, executed separately. The key's complete unit contract
            # below owns Phase 1's exact key selection and receipt.
            continue
        elif argv[:2] == ['cargo', 'test'] and argv[-1] == 'capacity_key::tests::':
            args = [*argv[2:-1], '--lib', '-E', 'test(capacity_key::tests::)']
        else:
            raise OracleFailure(f'unclassified inherited test command: {leg.name}')
        exclusion = f'not test(={MANUAL_TEST})'
        if '-E' in args:
            index = args.index('-E') + 1
            args[index] = f'({args[index]}) & ({exclusion})'
        else:
            args += ['-E', exclusion]
        legs.append((leg.name, tuple(args)))
    # Exact-key and planner equality must be exercised with both compilation
    # profiles, independently of the already dual-profile runtime contracts.
    for name, args in tuple(legs):
        if '--release' in args and ('capacity' in name or 'finite projection' in name):
            legs.append((name + ' (debug)', tuple(x for x in args if x != '--release')))
    for profile, flags in (('debug', ()), ('release', ('--release',))):
        legs.append((f'exact storage proof and representation controls {profile}',
                     (*flags, '-p', 'chelis-ir', '--lib', '-E', 'test(ownership::storage::tests::)')))
        legs.append((f'Metal no-reuse projection {profile}',
                     (*flags, '-p', 'chelis-backend-metal', '--test', 'never_reuse_boundary')))
    legs.append(('closed representation vocabulary release',
                 ('--release', '-p', 'chelis-vocab', '--test', 'dtype_contract', '--test', 'dtype_contract_compile')))
    legs.append(('sealed runtime element contract debug',
                 ('-p', 'chelis-runtime', '--test', 'element_contract')))
    return tuple(legs)


def command(argv, root, directory: Path, label: str, *, expected_exit=0):
    directory.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, 'PYO3_PYTHON': sys.executable, 'VIRTUAL_ENV': sys.prefix}
    with (directory / f'{label}.stdout').open('w') as stdout, (directory / f'{label}.stderr').open('w') as stderr:
        result = subprocess.run(argv, cwd=root, env=environment, stdout=stdout, stderr=stderr, check=False)
    (directory / f'{label}.process.json').write_text(json.dumps({'argv': argv, 'cwd': str(root), 'returncode': result.returncode}, indent=2) + '\n')
    if result.returncode != expected_exit:
        raise OracleFailure(f'{label} failed ({result.returncode}); see {directory}')
    return (directory / f'{label}.stdout').read_text()


def nextest_command(action, args):
    return ['cargo', 'nextest', action, '--locked', '--profile', PROFILE,
            '--ignore-default-filter', *args]


def execute_leg(name, args, expected, directory):
    print(f'+ {name}: list and execute {len(expected)} frozen tests', flush=True)
    listed = command([*nextest_command('list', args), '--message-format', 'json'], ROOT, directory, 'list')
    selected, artifacts = selection(load_json(listed), ROOT, expected)
    target = Path(os.environ.get('CARGO_TARGET_DIR', ROOT / 'target'))
    junit = target / 'nextest' / PROFILE / 'junit.xml'
    junit.unlink(missing_ok=True)
    command([*nextest_command('run', args), '--no-fail-fast', '--retries', '0'], ROOT, directory, 'run')
    if not junit.is_file():
        raise OracleFailure('nextest produced no fresh execution receipt')
    xml = junit.read_text()
    (directory / 'execution.xml').write_text(xml)
    executed = execution(xml, selected)
    verify_artifacts(artifacts)
    return {'name': name, 'selected': selected, 'executed': [{'id': name, 'outcome': 'passed'} for name in executed], 'artifacts': artifacts}


def planner_mutations():
    predicate = 'self.repr == other.repr && self.key.prove_equal(&other.key).is_ok()'
    return (
        ('erase-exact-representation', predicate, 'self.key.prove_equal(&other.key).is_ok()',
         'expired_owned_slots_require_exact_representation_in_both_lanes'),
        ('erase-exact-capacity', predicate, 'self.repr == other.repr',
         'distinct_large_products_do_not_reuse_storage'),
    )


def execute_planner_mutations(directory):
    source = ROOT / 'crates/chelis-ir/src/ownership/storage.rs'
    receipts = []
    for name, before, after, test in planner_mutations():
        def mutate(text):
            if text.count(before) != 1:
                raise OracleFailure('production storage equality mutation anchor drifted')
            return text.replace(before, after)
        args = ('--release', '-p', 'chelis-ir', '--test', 'issue_888_capacity_collision', '-E', f'test(={test})')
        expected = [f'chelis-ir::issue_888_capacity_collision::{test}']
        evidence = directory / name
        with phase0.temporary_mutation(source, mutate):
            listed = command([*nextest_command('list', args), '--message-format', 'json'], ROOT, evidence, 'list')
            selected, artifacts = selection(load_json(listed), ROOT, expected)
            target = Path(os.environ.get('CARGO_TARGET_DIR', ROOT / 'target'))
            junit = target / 'nextest' / PROFILE / 'junit.xml'
            junit.unlink(missing_ok=True)
            command([*nextest_command('run', args), '--retries', '0'], ROOT, evidence, 'run', expected_exit=100)
            xml = junit.read_text()
            (evidence / 'execution.xml').write_text(xml)
            execution(xml, selected, failure_text='assertion `left == right` failed')
            verify_artifacts(artifacts)
            receipts.append({'id': name, 'outcome': 'rejected-by-exact-assertion',
                             'selected': selected, 'artifacts': artifacts})
        # Rebuild and run the same actual test after restoration, so a mutant
        # artifact cannot survive as the last execution of this obligation.
        execute_leg(name + ' restored', args, expected, evidence / 'restored')
    return receipts


def run():
    check_options(sys.argv[1:])
    identity = source_identity(ROOT)
    packet = frozen_manifest(MANIFEST.read_bytes(), MANIFEST_SHA256)
    expected_mutations = [{'id': n, 'before': a, 'after': z, 'test': t} for n, a, z, t in planner_mutations()]
    if packet.get('planner_mutations') != expected_mutations:
        raise OracleFailure('Phase 1 production mutation manifest drifted')
    legs = phase1_legs()
    if packet.get('schema') != 1 or [row.get('name') for row in packet.get('legs', [])] != [name for name, _ in legs]:
        raise OracleFailure('Phase 1 leg inventory drifted')
    for row, (_, args) in zip(packet['legs'], legs, strict=True):
        if row.get('args') != list(args):
            raise OracleFailure('Phase 1 frozen command drifted')
    run_id = str(uuid.uuid4())
    directory = ROOT / 'target/runtime-representation-phase1' / run_id
    directory.mkdir(parents=True, exist_ok=False)
    python_cases = python_execution(python_suite(), packet['python_selected'])
    phase0.validate_phase0_inventory()
    phase0.run_phase0_mutations()
    executions = []
    for index, (row, (name, args)) in enumerate(zip(packet['legs'], legs, strict=True)):
        executions.append(execute_leg(name, args, row['selected'], directory / str(index)))
    planner_receipts = execute_planner_mutations(directory / 'planner-mutations')
    for leg in phase0.phase0_legs():
        if '--doc' in leg.argv:
            command(list(leg.argv), ROOT, directory / 'supporting-doc-privacy', 'run')
    if source_identity(ROOT) != identity:
        raise OracleFailure('source changed during execution')
    receipt = {'schema': 1, 'head': identity[0], 'source_digest': identity[1], 'run_id': run_id,
               'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
               'python_executed': [{'id': name, 'outcome': 'passed'} for name in python_cases],
               'mutation_probes': [{'id': probe.witness_id, 'outcome': 'rejected',
                                    'expected_class': probe.expected_kind}
                                   for probe in phase0.phase0_mutation_probes()],
               'manual_exclusions': [{'test': MANUAL_TEST, 'issue': 1112,
                                      'reason': 'allocates more than 8 GiB; not an executed receipt'}],
               'planner_mutations': planner_receipts,
               'legs': executions}
    (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Current execution receipt: {directory / "receipt.json"}')
    print('RUNTIME REPRESENTATION PHASE 1: PASS')
