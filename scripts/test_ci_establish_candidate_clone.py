"""The candidate clone's stated invariant, and its confinement."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tempfile
import unittest

import yaml

from scripts import ci_candidate_lifecycle as lifecycle
from scripts import ci_establish_candidate_clone as clone


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
ABSENT = "0" * 39 + "1"


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


class AdvancedTargetOrigin:
    """chelis#2234's topology: the target advanced past the branch point.

    `before` sits on the old base, `head` on the advanced target, so
    classifying the update needs a merge base the head deepen's own
    boundary grafts away.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        git(root, "init", "-b", "main")
        git(root, "config", "user.name", "Candidate Clone Test")
        git(root, "config", "user.email", "clone@example.invalid")
        self.commit("root.txt", "root\n")
        self.branch_point = self.commit("base.txt", "base\n")
        git(root, "checkout", "-b", "old-head")
        self.before = self.commit("feature.txt", "one\n")
        git(root, "checkout", "main")
        self.target = self.commit("target.txt", "advanced\n")
        git(root, "checkout", "-b", "new-head")
        self.head = self.commit("feature.txt", "one\n")
        git(root, "checkout", "-b", "candidate", self.target)
        git(root, "merge", "--no-ff", "new-head", "-m", "synthetic candidate")
        self.candidate = git(root, "rev-parse", "HEAD")

    def commit(self, name: str, body: str) -> str:
        (self.root / name).write_text(body, encoding="utf-8")
        git(self.root, "add", name)
        git(self.root, "commit", "-m", name)
        return git(self.root, "rev-parse", "HEAD")

    def checkout_candidate(self, destination: Path, *, depth: int | None = 1) -> Path:
        git(destination, "init", "-b", "main")
        git(destination, "remote", "add", "origin", self.root.as_uri())
        arguments = ["fetch", "--no-tags"]
        if depth is not None:
            arguments.append(f"--depth={depth}")
        git(destination, *arguments, "origin", self.candidate)
        git(destination, "checkout", self.candidate)
        return destination


class OrdinaryPushOrigin:
    """The common synchronize event: a descendant push, base unchanged.

    The escalation's cost argument rests on how often it fires, so the
    frequency has to be a tested property rather than an assertion.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        git(root, "init", "-b", "main")
        git(root, "config", "user.name", "Candidate Clone Test")
        git(root, "config", "user.email", "clone@example.invalid")
        self.commit("root.txt", "root\n")
        self.base = self.commit("base.txt", "base\n")
        git(root, "checkout", "-b", "feature")
        self.before = self.commit("feature.txt", "one\n")
        self.head = self.commit("feature.txt", "two\n")
        git(root, "checkout", "-b", "candidate", self.base)
        git(root, "merge", "--no-ff", "feature", "-m", "synthetic candidate")
        self.candidate = git(root, "rev-parse", "HEAD")

    def commit(self, name: str, body: str) -> str:
        (self.root / name).write_text(body, encoding="utf-8")
        git(self.root, "add", name)
        git(self.root, "commit", "-m", name)
        return git(self.root, "rev-parse", "HEAD")


class CandidateCloneInvariantTests(unittest.TestCase):
    def origin_and_clone(self, *, depth: int | None = 1):
        origin = tempfile.TemporaryDirectory()
        checkout = tempfile.TemporaryDirectory()
        self.addCleanup(origin.cleanup)
        self.addCleanup(checkout.cleanup)
        source = AdvancedTargetOrigin(Path(origin.name))
        return source, source.checkout_candidate(
            Path(checkout.name), depth=depth
        )

    def test_the_cheap_shape_fails_and_deepening_repairs_it(self) -> None:
        """The whole defect and its repair, in one test so neither passes alone."""

        source, checkout = self.origin_and_clone()

        established, notes = clone.establish(
            checkout,
            base=source.target,
            head=source.head,
            before=source.before,
            target_tip=source.target,
            commits=1,
        )

        deepened = [note for note in notes if "deepening" in note]
        self.assertEqual(len(deepened), 1, notes)
        self.assertIn("have no merge base this clone can walk to", deepened[0])
        self.assertFalse((checkout / ".git" / "shallow").exists())
        self.assertEqual(established["candidate first parent"], source.target)
        # The property is only worth asserting if it buys the verifier
        # something, so check the verifier, not just the clone.
        self.assertEqual(
            lifecycle.classify_update(
                before=source.before,
                head=source.head,
                base=source.target,
                graph=lifecycle.GitGraph(checkout),
            ),
            "base-rebase",
        )

    def test_a_clone_that_already_satisfies_it_is_not_deepened(self) -> None:
        """The escalation is the exception, not the cost of every run."""

        source, checkout = self.origin_and_clone(depth=None)

        _, notes = clone.establish(
            checkout,
            base=source.target,
            head=source.head,
            before=source.before,
            target_tip=source.target,
            commits=1,
        )

        self.assertEqual([note for note in notes if "deepening" in note], [])


    def test_an_ordinary_descendant_push_does_not_deepen(self) -> None:
        """The escalation is the exception, on a shallow clone too.

        `--unshallow` is bounded below by the whole history, so the
        argument for it can only be about how rarely it fires. This is the
        common synchronize event: the pre-push head is an ancestor of the
        head and the target has not advanced, so the deepened range already
        contains every comparison and nothing escalates.
        """

        origin = tempfile.TemporaryDirectory()
        checkout = tempfile.TemporaryDirectory()
        self.addCleanup(origin.cleanup)
        self.addCleanup(checkout.cleanup)
        source = OrdinaryPushOrigin(Path(origin.name))
        destination = Path(checkout.name)
        git(destination, "init", "-b", "main")
        git(destination, "remote", "add", "origin", source.root.as_uri())
        git(destination, "fetch", "--no-tags", "--depth=1", "origin", source.candidate)
        git(destination, "checkout", source.candidate)

        _, notes = clone.establish(
            destination,
            base=source.base,
            head=source.head,
            before=source.before,
            target_tip=source.base,
            commits=2,
        )

        self.assertTrue(
            (destination / ".git" / "shallow").exists(),
            "this clone must still be shallow, or the case is not the cheap one",
        )
        self.assertEqual([note for note in notes if "deepening" in note], [])

    def test_an_unreachable_commit_fails_with_the_pair_and_the_shape(
        self,
    ) -> None:
        """Negative control: unmet is a loud failure, never a quiet pass."""

        source, checkout = self.origin_and_clone()

        with self.assertRaises(clone.CloneError) as raised:
            clone.establish(
                checkout,
                base=ABSENT,
                head=source.head,
                before=None,
                target_tip=None,
                commits=1,
            )

        message = str(raised.exception)
        self.assertIn("does not support the verifiers that read it", message)
        self.assertIn(f"base {ABSENT} is not in the clone", message)
        self.assertRegex(message, r"it is (shallow|complete) after deepening")

    def test_the_first_parent_is_read_through_a_graft(self) -> None:
        source, checkout = self.origin_and_clone()

        self.assertEqual(
            git(checkout, "rev-list", "--parents", "-n", "1", source.candidate),
            source.candidate,
            "this clone must be grafted for the test to mean anything",
        )
        self.assertEqual(clone.first_parent(checkout), source.target)

    def test_a_failed_fetch_is_recorded_rather_than_discarded(self) -> None:
        """No silent fetch here either; chelis#2229's defect was the silence."""

        source, checkout = self.origin_and_clone(depth=None)
        notes: list[str] = []

        clone._fetch(
            checkout, "--no-tags", "origin", ABSENT, label="probe", notes=notes
        )

        self.assertEqual(len(notes), 1, notes)
        self.assertTrue(notes[0].startswith("probe failed: "), notes[0])


class CandidateCloneConfinementTests(unittest.TestCase):
    """An invariant a later fetch can violate reads as a guarantee it lost.

    A fetch can only add history, with one exception: a depth-limited fetch
    adds a graft, which is what broke the property in chelis#2228 and again
    in chelis#2234. So the establishing step owns every fetch in its job,
    and this fails if another step acquires one.
    """

    DEPTH_ARGUMENT = re.compile(r"--(depth|deepen|shallow-since|shallow-exclude)\b")

    def changes_steps(self, workflow: str) -> list[dict]:
        parsed = yaml.safe_load((WORKFLOWS / workflow).read_text())
        return parsed["jobs"]["changes"]["steps"]

    def test_only_the_establishing_step_fetches(self) -> None:
        for workflow in ("ci.yml", "conformance.yml"):
            with self.subTest(workflow=workflow):
                steps = self.changes_steps(workflow)
                owners = [step for step in steps if step.get("id") == "clone"]
                self.assertEqual(len(owners), 1, "one step owns the clone")
                strays = [
                    (step.get("name"), line.strip())
                    for step in steps
                    if step.get("id") != "clone"
                    for line in step.get("run", "").splitlines()
                    if "git fetch" in line
                ]
                self.assertEqual(strays, [])

    def test_only_the_establishing_step_may_limit_depth(self) -> None:
        """The narrower property, and the one that actually bites.

        A depth-less fetch cannot graft, so it cannot violate the
        invariant. A depth-limited one can, wherever it is written.
        """

        for workflow in ("ci.yml", "conformance.yml"):
            with self.subTest(workflow=workflow):
                limited = [
                    (step.get("name"), line.strip())
                    for step in self.changes_steps(workflow)
                    if step.get("id") != "clone"
                    for line in step.get("run", "").splitlines()
                    if "git fetch" in line and self.DEPTH_ARGUMENT.search(line)
                ]
                self.assertEqual(limited, [])

    def test_the_verifiers_consume_the_established_clone(self) -> None:
        for workflow in ("ci.yml", "conformance.yml"):
            steps = {
                step.get("id"): step.get("run", "")
                for step in self.changes_steps(workflow)
            }
            with self.subTest(workflow=workflow):
                self.assertIn(
                    "ci_establish_candidate_clone.py", steps.get("clone", "")
                )
                self.assertIn(
                    "steps.clone.outputs.target_tip",
                    steps.get("candidate-lifecycle", ""),
                )


if __name__ == "__main__":
    unittest.main()
