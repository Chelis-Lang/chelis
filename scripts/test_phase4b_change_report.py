"""Committed atom/region review cues, with retained guards as independent controls."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import yaml

import phase4b_change_report as report
import dtype_phase4b_oracle as oracle


OP = "spec/05-risc-primitives.md"
TYPES = "spec/04-type-system.md"
REGISTRY = "spec/registry/c_scalar_carrier.md"
BUILTINS = "spec/registry/builtin_semantic_identities.md"
DESIGN = "spec/design/example.md"
ATOM = "> **[05-OP-31]** A scalar contract.\n> Exact values.\n"
REGION = "## Start\nThe contract.\n## End\nOutside.\n"
BUILTIN_TEXT = "| Identity | Atom |\n|---|---|\n| `Numeric:scalar:Scalar` | [05-OP-31] |\n"


def inventory(*, atoms=None, regions=None, registries=None):
    return (
        f"CONTRACT_FILES = {(OP, TYPES, REGISTRY, BUILTINS, DESIGN)!r}\n"
        f"FROZEN_ATOM_DIGESTS = {({'05-OP-31': '0' * 64} if atoms is None else atoms)!r}\n"
        f"FROZEN_REGION_DIGESTS = {({'example': (DESIGN, '## Start', '## End', '0' * 64)} if regions is None else regions)!r}\n"
        f"OP_MANIFEST_REGISTRY_FILES = {({'05-OP-31': REGISTRY} if registries is None else registries)!r}\n"
    )


def current_inventory(*, atoms=("05-OP-31",), regions=None):
    return (
        f"CONTRACT_FILES = {(OP, TYPES, REGISTRY, BUILTINS, DESIGN)!r}\n"
        f"REQUIRED_ATOMS = {atoms!r}\n"
        f"REQUIRED_REGIONS = {({'example': (DESIGN, '## Start', '## End')} if regions is None else regions)!r}\n"
        f"OP_MANIFEST_REGISTRY_FILES = {{'05-OP-31': {REGISTRY!r}}}\n"
    )


class InventoryRetirementTests(unittest.TestCase):
    def test_old_and_current_formats_have_identical_protection(self):
        old = report.read_inventory(inventory())
        new = report.read_inventory(current_inventory())
        self.assertEqual(old, new)
        self.assertEqual(new["REQUIRED_ATOMS"], ("05-OP-31",))
        self.assertEqual(new["REQUIRED_REGIONS"], {"example": (DESIGN, "## Start", "## End")})

    def test_mixed_duplicate_mutated_and_incomplete_formats_fail(self):
        for source in (
            current_inventory() + "FROZEN_ATOM_DIGESTS = {}\n",
            inventory() + "REQUIRED_REGIONS = {}\n",
            current_inventory(atoms=("05-OP-31", "05-OP-31")),
            current_inventory(atoms=["05-OP-31"]),
            current_inventory().replace("REQUIRED_REGIONS =", "MISSING_REGIONS ="),
            current_inventory() + "REQUIRED_ATOMS += ('05-OP-32',)\n",
            current_inventory() + "REQUIRED_REGIONS.clear()\n",
            current_inventory() + "alias = REQUIRED_REGIONS\nalias.clear()\n",
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                report.read_inventory(source)


class GitChanges(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.git("init", "--initial-branch=fixture")
        self.git("config", "user.name", "Phase 4B fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.files = {report.ORACLE: inventory(), OP: ATOM,
                      TYPES: "> **[04-NUM-8]** Declared widths.\n",
                      REGISTRY: "# Scalars\n| callable | signature |\n| item | `scalar()` |\n",
                      BUILTINS: BUILTIN_TEXT, DESIGN: REGION}

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-c", "maintenance.auto=false", "-C", str(self.root), *args],
            text=True, stderr=subprocess.PIPE,
        ).strip()

    def commit(self, edits=None):
        self.files.update(edits or {})
        for name, source in self.files.items():
            path = self.root / name
            if source is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source)
        self.git("add", ".")
        self.git("commit", "--allow-empty", "-m", "fixture")
        return self.git("rev-parse", "HEAD")

    def compare(self, base, candidate="HEAD"):
        return report.changed_contracts(self.root, base, candidate)

    def identities(self, result):
        return {(row["kind"], row["identity"]) for row in result["changes"]}

    def enforce(self, base, values=(), *, body=None, extra=()):
        args = ["--base", base, "--require-acknowledgement", *extra]
        for value in values:
            args += ["--acknowledge", value]
        if body is not None:
            args += ["--acknowledgements-env", "PHASE4B_TEST_BODY"]
        with mock.patch.object(oracle, "CONTRACT_FILES", tuple(p for p, text in self.files.items() if p != report.ORACLE and text is not None)), \
             mock.patch.object(oracle, "run_oracle") as run, \
             mock.patch.dict(oracle.os.environ, {"PHASE4B_TEST_BODY": body or ""}), \
             redirect_stdout(io.StringIO()):
            oracle.main(args, root=self.root)
            run.assert_called_once()

    def test_retirement_changes_no_identity_and_later_edits_require_acknowledgement(self):
        base = self.commit()
        transition = self.commit({report.ORACLE: current_inventory()})
        self.assertEqual(self.compare(base)["changes"], [])
        self.commit({OP: ATOM.replace("Exact values.", "Reviewed exact values.")})
        self.assertIn(("atom", "05-OP-31"), self.identities(self.compare(transition)))
        with self.assertRaises(SystemExit):
            self.enforce(transition, (OP,))
        self.enforce(transition, (OP, "atom:05-OP-31"))

    def test_granular_acknowledgements_require_each_changed_identity(self):
        base = self.commit()
        self.commit({OP: ATOM.replace("Exact", "Rounded"), DESIGN: REGION.replace("The", "Another")})
        values = [OP, DESIGN, "atom:05-OP-31", 'region:"example"']
        self.enforce(base, values)
        for missing in values:
            with self.subTest(missing=missing), self.assertRaisesRegex(SystemExit, "unacknowledged"):
                self.enforce(base, [v for v in values if v != missing])
        result = self.compare(base)
        self.assertEqual(set(result["required_acknowledgements"]),
                         {"Frozen-contract-change: " + value for value in values})

    def test_granular_stale_unknown_duplicate_and_aliases_are_rejected(self):
        base = self.commit()
        self.commit({DESIGN: REGION.replace("The", "Another")})
        valid = [DESIGN, 'region:"example"']
        self.enforce(base, valid)
        for extra, reason in [('atom:04-NUM-8', "stale"), ('atom:05-OP-999', "stale"),
                              ('region:"missing"', "stale"), ('region:"example"', "duplicate"),
                              ('region:"ex\\u0061mple"', "duplicate")]:
            with self.subTest(extra=extra), self.assertRaisesRegex(SystemExit, reason):
                self.enforce(base, valid + [extra])

    def test_granular_removed_and_protection_only_identities_cannot_disappear(self):
        base = self.commit()
        self.commit({report.ORACLE: inventory(atoms={}, regions={})})
        result = self.compare(base)
        self.assertEqual(result["changed_contract_files"], [])
        self.enforce(base, ["atom:05-OP-31", 'region:"example"'])
        with self.assertRaisesRegex(SystemExit, "unacknowledged.*atom:05-OP-31"):
            self.enforce(base, ['region:"example"'])
        with self.assertRaisesRegex(SystemExit, "unacknowledged.*region:"):
            self.enforce(base, ["atom:05-OP-31"])

    def test_removed_contract_file_still_owes_file_and_region_acknowledgement(self):
        base = self.commit()
        source = inventory(regions={}).replace(
            repr((OP, TYPES, REGISTRY, BUILTINS, DESIGN)), repr((OP, TYPES, REGISTRY, BUILTINS))
        )
        self.commit({report.ORACLE: source, DESIGN: None})
        self.enforce(base, [DESIGN, 'region:"example"'])
        # Retiring the inventory declaration also owes the file cue if its bytes remain.
        self.commit({DESIGN: REGION})
        self.enforce(base, [DESIGN, 'region:"example"'])
        with self.assertRaisesRegex(SystemExit, "unacknowledged.*example.md"):
            self.enforce(base, ['region:"example"'])

    def test_explicit_empty_or_invalid_pr_head_never_uses_default_base(self):
        base = self.commit()
        self.git("update-ref", "refs/remotes/origin/main", base)
        for head in ("", "short", base):
            with self.subTest(head=head), mock.patch.object(oracle, "run_oracle") as run:
                with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                    oracle.main(["--pr-head", head, "--require-acknowledgement"], root=self.root)
                run.assert_not_called()

    def test_granular_registry_only_edit_requires_owning_atom(self):
        base = self.commit()
        self.commit({REGISTRY: "# Revised scalar rules\n"})
        self.enforce(base, [REGISTRY, "atom:05-OP-31"])
        with self.assertRaisesRegex(SystemExit, "unacknowledged.*atom:05-OP-31"):
            self.enforce(base, [REGISTRY])

    def test_granular_body_grammar_and_fences_fail_closed(self):
        base = self.commit()
        self.commit({OP: ATOM.replace("Exact", "Rounded")})
        self.enforce(base, body=f"Frozen-contract-change: {OP}\nFrozen-contract-change: atom:05-OP-31\n")
        for value in ['atom:[05-OP-31]', 'atom:05-OP-0', 'atom:05-op-31',
                      'region:example', 'region:["example"]', 'region:""',
                      'region:"example" trailing', 'region:"bad\\nlabel"']:
            with self.subTest(value=value), self.assertRaisesRegex(SystemExit, "malformed"):
                self.enforce(base, [OP, "atom:05-OP-31", value])
        for body in ['```\nFrozen-contract-change: atom:05-OP-31\n```',
                     '~~~\nFrozen-contract-change: atom:05-OP-31\n~~~']:
            with self.assertRaisesRegex(SystemExit, "unacknowledged.*atom:05-OP-31"):
                self.enforce(base, [OP], body=body)

    def test_granular_region_label_round_trip_and_body_cli_duplicate(self):
        label = 'exact "quoted" region: contract'
        base = self.commit({report.ORACLE: inventory(regions={label: (DESIGN, "## Start", "## End", "0" * 64)})})
        self.commit({DESIGN: REGION.replace("The", "Another")})
        address = "region:" + json.dumps(label)
        body = report.acknowledgement_line("region", label)
        self.enforce(base, [DESIGN], body=body)
        with self.assertRaisesRegex(SystemExit, "duplicate"):
            self.enforce(base, [DESIGN, address], body=body)
        with self.assertRaisesRegex(SystemExit, "unclosed"):
            self.enforce(base, [DESIGN, address], body="```\n")

    def test_granular_advisory_reports_missing_and_strict_unreadable_fails(self):
        base = self.commit()
        self.commit({OP: ATOM.replace("Exact", "Rounded")})
        output = io.StringIO()
        with mock.patch.object(oracle, "run_oracle"), redirect_stdout(output):
            oracle.main(["--base", base], root=self.root)
        self.assertIn("NEEDS Frozen-contract-change: atom:05-OP-31", output.getvalue())
        with self.assertRaisesRegex(SystemExit, "cannot|resolve"):
            self.enforce("absent-base")

    def test_granular_oracle_validates_synthetic_merge_event_head(self):
        base = self.commit()
        self.git("checkout", "-b", "pr")
        head = self.commit({OP: ATOM.replace("Exact", "Rounded")})
        self.git("checkout", "fixture")
        advanced = self.commit({OP: ATOM, DESIGN: REGION.replace("The", "Main")})
        self.git("merge", "--no-ff", "pr", "-m", "synthetic")
        # The candidate uses the first parent; main-only region edits owe no PR acknowledgement.
        with mock.patch.object(oracle, "run_oracle") as run, redirect_stdout(io.StringIO()):
            oracle.main(["--pr-head", head, "--require-acknowledgement",
                         "--acknowledge", OP, "--acknowledge", "atom:05-OP-31"], root=self.root)
            run.assert_called_once()
        with self.assertRaisesRegex(SystemExit, "second parent|event pull-request head"):
            oracle.main(["--pr-head", advanced, "--require-acknowledgement"], root=self.root)

    def test_unchanged_and_working_tree_edits(self):
        base = self.commit()
        (self.root / OP).write_text("uncommitted invalid text")
        result = self.compare(base)
        self.assertEqual(result["changes"], [])
        self.assertEqual(result["changed_contract_files"], [])
        self.assertEqual(result["base"], base)

    def test_atom_and_region_changes_have_owners_and_locations(self):
        base = self.commit()
        self.commit({OP: ATOM.replace("Exact", "Rounded"), DESIGN: REGION.replace("The", "Another")})
        result = self.compare(base)
        self.assertEqual(self.identities(result), {("atom", "05-OP-31"), ("region", "example")})
        atom = next(row for row in result["changes"] if row["kind"] == "atom")
        self.assertEqual(atom["before"]["path"], OP)
        self.assertEqual(atom["after"]["lines"], [1, 2])
        self.assertEqual(set(result["changed_contract_files"]), {OP, DESIGN})

    def test_registry_text_and_builtin_membership_name_their_owning_atom(self):
        base = self.commit()
        for path, value in [(REGISTRY, "# Scalar contract changed\n"),
                            (BUILTINS, BUILTIN_TEXT.replace("Numeric:scalar:Scalar", "Numeric:scalar:Other"))]:
            with self.subTest(path=path):
                original = self.files[path]
                self.commit({path: value})
                result = self.compare(base)
                self.assertEqual(self.identities(result), {("atom", "05-OP-31")})
                self.assertIn("registry changed", result["changes"][0]["changes"])
                self.commit({path: original})

    def test_new_unfrozen_atoms_and_removed_definitions_are_visible(self):
        base = self.commit()
        self.commit({OP: ATOM + "\n> **[05-OP-90]** New contract.\n"})
        self.assertEqual(self.identities(self.compare(base)), {("atom", "05-OP-90")})
        added = self.git("rev-parse", "HEAD")
        self.commit({OP: ATOM})
        self.assertIn("removed", self.compare(added)["changes"][0]["changes"])

    def test_builtin_rehome_names_both_owners_and_preserves_unrelated_atoms(self):
        self.commit({OP: ATOM + "\n> **[05-OP-90]** A different contract.\n"})
        base = self.git("rev-parse", "HEAD")
        self.commit({BUILTINS: BUILTIN_TEXT.replace("05-OP-31", "05-OP-90")})
        self.assertEqual(self.identities(self.compare(base)), {("atom", "05-OP-31"), ("atom", "05-OP-90")})

    def test_atom_registry_reference_is_discovered_without_a_parallel_mapping(self):
        self.commit({OP: ATOM + f"> Incorporates `{REGISTRY}`.\n",
                     report.ORACLE: inventory(registries={})})
        base = self.git("rev-parse", "HEAD")
        self.commit({REGISTRY: "# Revised normative registry\n"})
        self.assertEqual(self.identities(self.compare(base)), {("atom", "05-OP-31")})
        self.commit({OP: ATOM + "> Incorporates `spec/registry/missing.md`.\n"})
        with self.assertRaises(ValueError):
            self.compare(base)

    def test_inventory_removal_cannot_hide_a_region_or_atom_removal(self):
        base = self.commit()
        self.commit({report.ORACLE: inventory(atoms={}, regions={}, registries={}),
                     OP: "# Removed atom\n", BUILTINS: "| Identity | Atom |\n|---|---|\n"})
        self.assertEqual(self.identities(self.compare(base)), {("atom", "05-OP-31"), ("region", "example")})

    def test_marker_moves_and_protection_changes_are_named(self):
        base = self.commit()
        self.commit({report.ORACLE: inventory(atoms={}, regions={"example": (DESIGN, "## New start", "## End", "0" * 64)}),
                     DESIGN: REGION.replace("## Start", "## New start")})
        result = self.compare(base)
        by_id = {row["identity"]: row for row in result["changes"]}
        self.assertIn("protection changed", by_id["05-OP-31"]["changes"])
        self.assertIn("boundaries changed", by_id["example"]["changes"])

    def test_outside_region_edit_is_file_cue_without_false_region_change(self):
        base = self.commit()
        self.commit({DESIGN: "Introduction.\n" + REGION + "Appended contradiction.\n"})
        result = self.compare(base)
        self.assertEqual(result["changes"], [])
        self.assertEqual(result["changed_contract_files"], [DESIGN])

    def test_invalid_or_missing_candidate_evidence_fails_closed(self):
        base = self.commit()
        cases = [(OP, "# Required atom missing\n"), (OP, ATOM + ATOM),
                 (DESIGN, REGION.replace("## End", "Missing")),
                 (DESIGN, REGION + "## Start\n"), (REGISTRY, None),
                 (BUILTINS, BUILTIN_TEXT + "| broken | row |\n"),
                 (BUILTINS, BUILTIN_TEXT.replace("05-OP-31", "05-OP-99")),
                 (report.ORACLE, "# No inventory\n")]
        for path, source in cases:
            with self.subTest(path=path, source=source):
                old = self.files[path]
                self.commit({path: source})
                with self.assertRaises(ValueError):
                    self.compare(base)
                self.commit({path: old})

    def test_literal_inventory_does_not_execute_historical_code(self):
        self.commit({report.ORACLE: "raise RuntimeError('never execute')\n" + inventory()})
        self.assertEqual(self.compare("HEAD")["changes"], [])
        for source in [inventory() + "FROZEN_REGION_DIGESTS = {}\n",
                       inventory().replace("'example':", "'example': ('x', 'a', 'b', '0'), 'example':"),
                       inventory() + "FROZEN_REGION_DIGESTS.update({})\n",
                       inventory() + "alias = FROZEN_REGION_DIGESTS\nalias.clear()\n",
                       inventory(regions={"example": ("../escape.md", "a", "b", "0" * 64)}),
                       inventory().replace("CONTRACT_FILES = (", "CONTRACT_FILES = tuple("),
                       inventory(atoms={"not-an-atom": "0" * 64})]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                report.read_inventory(source)

    def test_unique_merge_base_and_explicit_pr_parent_validation(self):
        event_base = self.commit()
        self.git("checkout", "-b", "candidate")
        head = self.commit()
        self.git("checkout", "fixture")
        advanced = self.commit({OP: ATOM.replace("Exact", "Rounded")})
        self.git("merge", "--no-ff", "--no-edit", "candidate")
        merge = self.git("rev-parse", "HEAD")
        base, candidate = report.resolve_comparison(self.root, "HEAD", "", head)
        self.assertEqual((base, candidate), (advanced, merge))
        self.assertEqual(self.compare(base, candidate)["changes"], [])
        self.assertTrue(self.compare(event_base, candidate)["changes"])
        self.assertEqual(self.compare(advanced, head)["changes"], [])
        for candidate, wrong_head in [(head, head), (merge, event_base), (merge, "bad")]:
            with self.subTest(candidate=candidate, head=wrong_head), self.assertRaises(ValueError):
                report.resolve_comparison(self.root, candidate, "", wrong_head)
        with self.assertRaises(ValueError):
            report.resolve_comparison(self.root, merge, advanced, head)

    def test_unrelated_history_and_multiple_merge_bases_are_invalid(self):
        base = self.commit()
        tree = self.git("rev-parse", f"{base}^{{tree}}")
        left = self.git("commit-tree", tree, "-p", base, "-m", "left")
        right = self.git("commit-tree", tree, "-p", base, "-m", "right")
        a = self.git("commit-tree", tree, "-p", left, "-p", right, "-m", "a")
        b = self.git("commit-tree", tree, "-p", right, "-p", left, "-m", "b")
        for comparison, candidate in [(a, b), (base, self.git("commit-tree", tree, "-m", "orphan")), ("missing", base)]:
            with self.subTest(comparison=comparison), self.assertRaises(ValueError):
                self.compare(comparison, candidate)

    def test_cli_publishes_provenance_doctrine_and_removes_stale_output(self):
        base = self.commit()
        self.commit({DESIGN: REGION.replace("The", "New")})
        output = self.root / "report.json"
        with mock.patch.object(report, "ROOT", self.root), redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
            self.assertEqual(report.main(["--base", base, "--output", str(output)]), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["candidate"], self.git("rev-parse", "HEAD"))
            self.assertIn("adversarial mutation", stdout.getvalue())
            self.assertEqual(report.main(["--base", "missing", "--output", str(output)]), 1)
            self.assertFalse(output.exists())


class RetainedMutationTests(unittest.TestCase):
    def test_every_required_atom_and_region_edit_keeps_enforcing_review(self):
        root = Path(__file__).resolve().parents[1]
        documents = {name: (root / name).read_text() for name in oracle.CONTRACT_FILES}
        documents[report.ORACLE] = (root / report.ORACLE).read_text()
        baseline_failures = []
        oracle.validate_required_contract(documents, baseline_failures)
        self.assertEqual(baseline_failures, [])
        before = report.snapshot(documents.__getitem__)
        cases = []
        for atom in oracle.REQUIRED_ATOMS:
            path = TYPES if atom.startswith("04-") else OP
            block = oracle.strict_atom_block(documents[path], atom)
            cases.append(("atom", atom, path, block, block.rstrip() + "\n> Mutated obligation.\n"))
        for label, (path, start, end) in oracle.REQUIRED_REGIONS.items():
            block = oracle.frozen_region(documents[path], start, end, label)
            cases.append(("region", label, path, block, block + "Mutated obligation.\n"))
        for kind, identity, path, old, new in cases:
            with self.subTest(kind=kind, identity=identity):
                mutated = dict(documents)
                mutated[path] = mutated[path].replace(old, new, 1)
                failures = []
                oracle.validate_required_contract(mutated, failures)
                oracle.validate_normative_contract(mutated, failures)
                oracle.validate_schema_and_consumers(mutated, failures)
                self.assertEqual(failures, [], "wording has no mechanical digest requirement")
                changes = report.compare_snapshots(before, report.snapshot(mutated.__getitem__))
                self.assertIn((kind, identity), {(row["kind"], row["identity"]) for row in changes})
                result = {"changes": changes}
                named = [(row["kind"], row["identity"]) for row in changes]
                self.assertEqual(report.identity_acknowledgement_violations(result, named), [])
                failures = report.identity_acknowledgement_violations(result, [key for key in named if key != (kind, identity)])
                self.assertTrue(any(report.acknowledgement_line(kind, identity) in failure for failure in failures))
                address = report.acknowledgement_line(kind, identity).removeprefix("Frozen-contract-change: ")
                self.assertEqual(oracle.acknowledgement_identity(address), (kind, identity))


    def test_missing_or_ambiguous_required_definitions_and_boundaries_still_fail(self):
        root = Path(__file__).resolve().parents[1]
        documents = {name: (root / name).read_text() for name in oracle.CONTRACT_FILES}
        cases = []
        for atom in oracle.REQUIRED_ATOMS:
            path = TYPES if atom.startswith("04-") else OP
            block = oracle.strict_atom_block(documents[path], atom)
            cases.extend((atom, path, block, replacement) for replacement in ("", block + block))
        for label, (path, start, end) in oracle.REQUIRED_REGIONS.items():
            cases.extend((label, path, marker, replacement)
                         for marker in (start, end) for replacement in ("REMOVED", marker + marker))
        for identity, path, old, new in cases:
            with self.subTest(identity=identity, replacement=new[:30]):
                mutated = dict(documents)
                mutated[path] = mutated[path].replace(old, new, 1)
                failures = []
                oracle.validate_required_contract(mutated, failures)
                self.assertTrue(any(identity in failure for failure in failures), failures)



class WorkflowTests(unittest.TestCase):
    command = '.venv/bin/python scripts/phase4b_change_report.py --base "$BASE_REF" --pr-head "$PR_HEAD" --output target/phase4b-contract-changes.json'

    def workflow(self):
        return yaml.safe_load((Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text())

    def check(self, workflow):
        job = workflow["jobs"]["docs"]
        self.assertNotIn("continue-on-error", job)
        steps = job["steps"]
        self.assertEqual(next(s for s in steps if s.get("uses", "").startswith("actions/checkout@"))["with"]["fetch-depth"], 0)
        reports = [s for s in steps if "phase4b_change_report.py" in s.get("run", "")]
        self.assertEqual(len(reports), 1)
        step = reports[0]
        self.assertEqual(step["run"], self.command)
        self.assertEqual(step["env"], {"BASE_REF": "${{ github.event_name == 'push' && github.event.before || '' }}", "PR_HEAD": "${{ github.event.pull_request.head.sha }}"})
        for key in ("if", "continue-on-error", "shell"):
            self.assertNotIn(key, step)
        artifact = next(s for s in steps if s.get("with", {}).get("name") == "phase4b-contract-changes")
        self.assertTrue(artifact["uses"].startswith("actions/upload-artifact@"))
        self.assertEqual(artifact["if"], "${{ always() }}")
        self.assertEqual(artifact["with"]["path"], "target/phase4b-contract-changes.json")
        self.assertEqual(artifact["with"]["if-no-files-found"], "error")

    def test_acknowledgement_step_requires_the_same_event_evidence(self):
        workflow = self.workflow()
        step = next(s for s in workflow["jobs"]["docs"]["steps"]
                    if s.get("name") == "Require frozen contract acknowledgements")
        self.assertEqual(step["if"], "github.event_name == 'pull_request'")
        self.assertEqual(step["env"], {"PR_BODY": "${{ github.event.pull_request.body }}",
                                      "PR_HEAD": "${{ github.event.pull_request.head.sha }}"})
        self.assertEqual(step["run"], 'uv run --managed-python --python 3.11 --no-project python scripts/dtype_phase4b_oracle.py --pr-head "$PR_HEAD" --require-acknowledgement --acknowledgements-env PR_BODY')
        self.assertNotIn("continue-on-error", step)

    def test_report_has_required_execution_and_publication(self):
        self.check(self.workflow())

    def test_synchronize_before_is_not_allowed_to_select_push_comparison(self):
        workflow = self.workflow()
        step = next(s for s in workflow["jobs"]["docs"]["steps"]
                    if "phase4b_change_report.py" in s.get("run", ""))
        step["env"]["BASE_REF"] = "${{ github.event.before }}"
        with self.assertRaises(AssertionError):
            self.check(workflow)

    def test_skipped_noop_or_suppressed_execution_does_not_satisfy_contract(self):
        for mutation in ("if", "continue-on-error", "run"):
            workflow = self.workflow()
            for step in workflow["jobs"]["docs"]["steps"]:
                if "phase4b_change_report.py" in step.get("run", ""):
                    step[mutation] = "true # " + self.command if mutation == "run" else True
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                self.check(workflow)


if __name__ == "__main__":
    unittest.main()
