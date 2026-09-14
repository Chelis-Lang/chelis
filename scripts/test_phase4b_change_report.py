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
    def test_every_current_frozen_atom_and_region_keeps_independent_detection(self):
        root = Path(__file__).resolve().parents[1]
        documents = {name: (root / name).read_text() for name in oracle.CONTRACT_FILES}
        documents[report.ORACLE] = (root / report.ORACLE).read_text()
        baseline_failures = []
        oracle.validate_frozen_contract(documents, baseline_failures)
        self.assertEqual(baseline_failures, [])
        before = report.snapshot(documents.__getitem__)
        cases = []
        for atom in oracle.FROZEN_ATOM_DIGESTS:
            path = TYPES if atom.startswith("04-") else OP
            block = oracle.strict_atom_block(documents[path], atom)
            cases.append(("atom", atom, path, block, block.rstrip() + "\n> Mutated obligation.\n"))
        for label, (path, start, end, _) in oracle.FROZEN_REGION_DIGESTS.items():
            block = oracle.frozen_region(documents[path], start, end, label)
            cases.append(("region", label, path, block, block + "Mutated obligation.\n"))
        for kind, identity, path, old, new in cases:
            with self.subTest(kind=kind, identity=identity):
                mutated = dict(documents)
                mutated[path] = mutated[path].replace(old, new, 1)
                failures = []
                oracle.validate_frozen_contract(mutated, failures)
                self.assertTrue(any(identity in failure for failure in failures))
                changes = report.compare_snapshots(before, report.snapshot(mutated.__getitem__))
                self.assertIn((kind, identity), {(row["kind"], row["identity"]) for row in changes})


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
        self.assertEqual(step["env"], {"BASE_REF": "${{ github.event.before }}", "PR_HEAD": "${{ github.event.pull_request.head.sha }}"})
        for key in ("if", "continue-on-error", "shell"):
            self.assertNotIn(key, step)
        artifact = next(s for s in steps if s.get("with", {}).get("name") == "phase4b-contract-changes")
        self.assertTrue(artifact["uses"].startswith("actions/upload-artifact@"))
        self.assertEqual(artifact["if"], "${{ always() }}")
        self.assertEqual(artifact["with"]["path"], "target/phase4b-contract-changes.json")
        self.assertEqual(artifact["with"]["if-no-files-found"], "error")

    def test_report_has_required_execution_and_publication(self):
        self.check(self.workflow())

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
