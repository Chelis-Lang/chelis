"""The candidate clone's stated invariant, and its confinement."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
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
    """An invariant a later step can violate reads as a guarantee it lost.

    A fetch can only add history, with two exceptions: a depth-limited
    fetch adds a graft, and a checkout re-clones. Both broke the property,
    in chelis#2228 and again in chelis#2234. So the establishing step owns
    both, and these fail if another step acquires either.

    Reading `run:` blocks alone is not enough and was the first version's
    hole: `actions/checkout` takes `fetch-depth` as an input, so the
    command that creates the original graft never appears in a run line at
    all.
    """

    SHAPE_SUBCOMMANDS = {"fetch", "pull", "clone"}
    DEPTH_FLAGS = ("--depth", "--deepen", "--shallow-since", "--shallow-exclude")

    def changes_steps(self, workflow: str) -> list[dict]:
        parsed = yaml.safe_load((WORKFLOWS / workflow).read_text())
        return parsed["jobs"]["changes"]["steps"]

    @staticmethod
    def commands(body: str) -> list[str]:
        """Fold continuations and collapse whitespace before matching.

        A line-scoped, literal scan is evaded by the spellings this file
        already uses elsewhere: a backslash continuation, a second space,
        and `git -C <dir>`. Normalising once beats adding a witness per
        spelling, which has no end.
        """

        joined: list[str] = []
        pending = ""
        for line in body.splitlines():
            stripped = line.rstrip()
            if stripped.endswith("\\"):
                pending += stripped[:-1].rstrip() + " "
                continue
            joined.append((pending + stripped.strip()).strip())
            pending = ""
        if pending:
            joined.append(pending.strip())
        return [" ".join(command.split()) for command in joined if command]

    @classmethod
    def git_invocations(cls, command: str) -> list[list[str]]:
        """Every `git ...` in one command, with its subcommand resolved.

        `git -C dir fetch` and `git fetch` must look the same here, so the
        global options and their values are skipped rather than matched.
        """

        try:
            tokens = shlex.split(command, comments=True)
        except ValueError:
            tokens = command.split()
        found: list[list[str]] = []
        for index, token in enumerate(tokens):
            if token != "git" and not token.endswith("/git"):
                continue
            rest = tokens[index + 1:]
            position = 0
            while position < len(rest) and rest[position].startswith("-"):
                # `-C <path>` and `-c <name>=<value>` take a value.
                position += 2 if rest[position] in {"-C", "-c"} else 1
            if position < len(rest):
                found.append([rest[position], *rest[position + 1:]])
        return found

    def shape_changing(self, body: str) -> list[tuple[str, list[str]]]:
        out = []
        for command in self.commands(body):
            for invocation in self.git_invocations(command):
                if invocation[0] in self.SHAPE_SUBCOMMANDS:
                    out.append((command, invocation))
        return out

    def test_only_the_establishing_step_changes_the_clone_shape(self) -> None:
        for workflow in ("ci.yml", "conformance.yml"):
            with self.subTest(workflow=workflow):
                steps = self.changes_steps(workflow)
                owners = [step for step in steps if step.get("id") == "clone"]
                self.assertEqual(len(owners), 1, "one step owns the clone")
                strays = [
                    (step.get("name"), command)
                    for step in steps
                    if step.get("id") != "clone"
                    for command, _ in self.shape_changing(step.get("run", ""))
                ]
                self.assertEqual(strays, [])

    def test_only_the_establishing_step_may_limit_depth(self) -> None:
        """The narrower rule, and the one that actually bites.

        A depth-less fetch cannot graft, so it cannot violate the
        invariant. A depth-limited one can, wherever it is written and
        however it is spelled.
        """

        for workflow in ("ci.yml", "conformance.yml"):
            with self.subTest(workflow=workflow):
                limited = [
                    (step.get("name"), command)
                    for step in self.changes_steps(workflow)
                    if step.get("id") != "clone"
                    for command, invocation in self.shape_changing(
                        step.get("run", "")
                    )
                    if any(
                        token.startswith(self.DEPTH_FLAGS) for token in invocation
                    )
                ]
                self.assertEqual(limited, [])

    def test_no_checkout_can_reclone_over_the_established_candidate(
        self,
    ) -> None:
        """`actions/checkout` never appears in a run line, and it grafts.

        It takes `fetch-depth` as an input, so a second checkout at the
        candidate path after the establishing step would replace the clone
        whose shape was just asserted, and every `run:`-scoped guard would
        miss it. The rule is positional: every checkout runs before the
        owner, and only the sanctioned one targets the candidate path.
        """

        for workflow in ("ci.yml", "conformance.yml"):
            steps = self.changes_steps(workflow)
            owner = next(
                index for index, step in enumerate(steps)
                if step.get("id") == "clone"
            )
            checkouts = [
                (index, step)
                for index, step in enumerate(steps)
                if "actions/checkout" in str(step.get("uses", ""))
            ]
            with self.subTest(workflow=workflow):
                self.assertTrue(checkouts, "the job must check something out")
                late = [
                    step.get("name") or step.get("id")
                    for index, step in checkouts
                    if index > owner
                ]
                self.assertEqual(late, [], "a checkout after the owner reclones")
                # Matched by name rather than id, because the step has no
                # id on this branch and a guard should not require one.
                candidate_path = [
                    step.get("name")
                    for _, step in checkouts
                    if (step.get("with") or {}).get("path") == "candidate"
                ]
                self.assertEqual(candidate_path, ["Checkout exact candidate"])

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


class EstablishingStepPreambleTests(unittest.TestCase):
    """The preamble must reach the module, not die above it.

    This step's outcome gates the candidate preflight and the step after it
    reads the clone, so a preamble command exiting under errexit closes
    required contexts saying nothing. The repair for that was verified once
    by hand, which is the same shape as the finding that prompted it, so
    the probe is encoded here instead.
    """

    def clone_step_body(self) -> str:
        parsed = yaml.safe_load((WORKFLOWS / "ci.yml").read_text())
        return next(
            step["run"]
            for step in parsed["jobs"]["changes"]["steps"]
            if step.get("id") == "clone"
        )

    def run_preamble(self, *, path: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate"
            (candidate / "scripts").mkdir(parents=True)
            shutil.copy(
                ROOT / "scripts/ci_establish_candidate_clone.py",
                candidate / "scripts",
            )
            git(candidate, "init", "-b", "main")
            git(candidate, "config", "user.name", "Preamble Test")
            git(candidate, "config", "user.email", "preamble@example.invalid")
            git(candidate, "commit", "--allow-empty", "-m", "root")
            git(candidate, "remote", "add", "origin", "file:///no-such-remote")
            body = root / "step.sh"
            body.write_text(self.clone_step_body(), encoding="utf-8")
            temp = root / "runner-temp"
            temp.mkdir()
            return subprocess.run(
                ["bash", "-e", str(body)],
                cwd=candidate,
                text=True,
                capture_output=True,
                env={
                    "PATH": path,
                    "RUNNER_TEMP": str(temp),
                    "GITHUB_OUTPUT": str(root / "out.txt"),
                    "GITHUB_REPOSITORY": "Chelis-Lang/chelis",
                    "PR_NUMBER": "1",
                    "ACTION": "synchronize",
                    "BASE": "1" * 40,
                    "HEAD": "2" * 40,
                    "BEFORE": "3" * 40,
                    "COMMITS": "1",
                },
            )

    def test_the_preamble_reaches_the_module_when_every_tool_fails(
        self,
    ) -> None:
        """No `gh`, no `jq`, unreachable remote: the module must still run."""

        completed = self.run_preamble(path="/usr/bin:/bin")

        self.assertIn("could not read the live pull request", completed.stderr)
        # Reaching the module is the property. It then fails closed *with
        # information*, which is what the preamble must not pre-empt.
        self.assertIn("CANDIDATE CLONE: FAIL", completed.stderr)
        # bash still reports the missing tool; what matters is that the
        # step's exit came from the module (1) rather than from a preamble
        # command dying under errexit (127 for a missing tool, 5 for jq on
        # a malformed document, 128 for git).
        self.assertEqual(completed.returncode, 1)

    def test_the_preamble_survives_a_malformed_pull_request_document(
        self,
    ) -> None:
        """`// ""` defaults a null field; it does not survive an array."""

        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory)
            (fake / "gh").write_text("#!/bin/sh\nprintf '[]'\n", encoding="utf-8")
            (fake / "gh").chmod(0o755)
            completed = self.run_preamble(
                path=f"{fake}:{os.environ.get('PATH', '/usr/bin:/bin')}"
            )

        self.assertIn("could not resolve the target tip", completed.stderr)
        self.assertIn("CANDIDATE CLONE: FAIL", completed.stderr)


if __name__ == "__main__":
    unittest.main()
