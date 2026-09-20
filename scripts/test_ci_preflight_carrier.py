"""The changes job always completes, and one context says why it failed."""

from __future__ import annotations

from pathlib import Path
import re
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
DETECTOR_WORKFLOWS = ("ci.yml", "conformance.yml")


def workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text())


def changes_steps(name: str) -> list[dict]:
    return workflow(name)["jobs"]["changes"]["steps"]


class AlwaysCompletesTests(unittest.TestCase):
    """A required context that never reports cannot be recovered.

    chelis#2228's pull request was unmergeable rather than merely red: the
    identity step raised, the `changes` job failed, the planner produced no
    output, and `Integration Tests (Linux)` had zero check runs. A context
    with no run cannot be waited out, re-run into existence, or overridden.
    So the job completes whatever happens and always emits its outputs, and
    the verdict says which step could not be evaluated.
    """

    def test_no_step_can_fail_the_changes_job(self) -> None:
        """Literal `true`, because anything else is not decidable here.

        `step.get("continue-on-error")` is truthy for any non-empty string,
        so a quoted `'false'` and an expression that evaluates to false at
        run time both satisfy a truthiness check while leaving the step
        able to fail the job. The same reason this file rejects an
        expression in a concurrency field: only a literal is verifiable
        without evaluating something this test cannot see.
        """

        for name in DETECTOR_WORKFLOWS:
            with self.subTest(workflow=name):
                intolerant = [
                    f"{step.get('name') or step.get('uses')} "
                    f"({step.get('continue-on-error')!r})"
                    for step in changes_steps(name)
                    if step.get("continue-on-error") is not True
                ]
                self.assertEqual(intolerant, [])

    def test_the_verdict_and_gate_run_even_after_a_failure(self) -> None:
        for name in DETECTOR_WORKFLOWS:
            steps = {step.get("id"): step for step in changes_steps(name)}
            for step_id in ("preflight-gate", "candidate-preflight"):
                with self.subTest(workflow=name, step=step_id):
                    self.assertEqual(steps[step_id].get("if"), "always()")

    def test_the_job_publishes_the_reason(self) -> None:
        for name in DETECTOR_WORKFLOWS:
            outputs = workflow(name)["jobs"]["changes"]["outputs"]
            with self.subTest(workflow=name):
                self.assertIn("preflight_reason", outputs)
                self.assertIn(
                    "steps.candidate-preflight.outputs.preflight_reason",
                    outputs["preflight_reason"],
                )


class ToleratedStepsAreReadTests(unittest.TestCase):
    """Tolerating a step must not make its failure ignorable.

    Every step is now `continue-on-error`, which is what keeps the job
    completing. The cost is that a failure is invisible unless something
    reads it, and "somebody wires the next one" is a hope rather than a
    mechanism. This makes forgetting a build failure instead.
    """

    #: Steps whose outcome the gate has no reason to read.
    EXEMPT = {
        # The gate and the verdict are the readers; a step cannot gate on
        # itself, and the verdict reads the gate by output rather than by
        # outcome.
        "preflight-gate",
        "candidate-preflight",
        # Read by the verdict rather than the gate, because they run
        # between the two.
        "identity",
        "upload-identity",
    }

    def gate_body(self, name: str) -> str:
        return next(
            step["run"]
            for step in changes_steps(name)
            if step.get("id") == "preflight-gate"
        )

    def test_every_tolerated_step_reaches_the_gate(self) -> None:
        for name in DETECTOR_WORKFLOWS:
            body = self.gate_body(name)
            unread = []
            for step in changes_steps(name):
                step_id = step.get("id")
                if step_id is None:
                    unread.append(
                        f"{step.get('name') or step.get('uses')} has no id, "
                        "so the gate cannot read its outcome"
                    )
                    continue
                if step_id in self.EXEMPT:
                    continue
                read = re.search(
                    r"(\w+)=\"\$\{\{ steps\." + re.escape(step_id)
                    + r"\.outcome \}\}\"",
                    body,
                )
                if read is None:
                    unread.append(f"{step_id} is tolerated but never read")
                    continue
                # A read that is never compared is a dead read. Deleting
                # the branch and leaving the assignment kept this green
                # until review mutated it, which is the difference between
                # proving the variable exists and proving it decides
                # something.
                variable = read.group(1)
                if not re.search(
                    r"\[ \"\$" + re.escape(variable)
                    + r"\" (?:=|!=) \"(?:failure|success)\" \]",
                    body,
                ):
                    unread.append(
                        f"{step_id} is read into ${variable} and never "
                        "compared, so its failure changes nothing"
                    )
            with self.subTest(workflow=name):
                self.assertEqual(unread, [])

    def test_the_identity_outcome_reaches_the_verdict(self) -> None:
        for name in DETECTOR_WORKFLOWS:
            verdict = next(
                step["run"]
                for step in changes_steps(name)
                if step.get("id") == "candidate-preflight"
            )
            with self.subTest(workflow=name):
                self.assertIn("steps.identity.outcome", verdict)
                self.assertIn("steps.upload-identity.outcome", verdict)

    def test_every_failure_branch_states_a_reason(self) -> None:
        """A verdict of failure with an empty reason is the old behaviour."""

        for name in DETECTOR_WORKFLOWS:
            body = self.gate_body(name)
            with self.subTest(workflow=name):
                bare = re.findall(r'^\s*fail\s*$|^\s*fail\s+""\s*$', body, re.M)
                self.assertEqual(bare, [])
                self.assertGreater(len(re.findall(r'\bfail "', body)), 8)


class IdentityStepOrderingTests(unittest.TestCase):
    """Moving a step earlier must not run it in a state nobody designed for.

    The identity step used to sit after the verdict and was gated on it. It
    now sits before, so the verdict can read its outcome, which means its
    own gate has to reproduce the condition it previously inherited rather
    than simply becoming unconditional.
    """

    def step_order(self, name: str) -> list[str]:
        return [
            step.get("id") or (step.get("name") or "")
            for step in changes_steps(name)
        ]

    def test_identity_precedes_the_verdict_and_follows_the_gate(self) -> None:
        for name in DETECTOR_WORKFLOWS:
            order = self.step_order(name)
            with self.subTest(workflow=name):
                self.assertLess(
                    order.index("preflight-gate"), order.index("identity")
                )
                self.assertLess(
                    order.index("identity"), order.index("candidate-preflight")
                )

    def test_identity_still_runs_only_behind_a_satisfied_gate(self) -> None:
        for name in DETECTOR_WORKFLOWS:
            identity = next(
                step
                for step in changes_steps(name)
                if step.get("id") == "identity"
            )
            with self.subTest(workflow=name):
                condition = identity.get("if", "")
                self.assertIn("steps.preflight-gate.outputs.gate == 'success'", condition)
                self.assertIn("github.event_name == 'pull_request'", condition)

    def test_the_upload_follows_the_identity_it_uploads(self) -> None:
        for name in DETECTOR_WORKFLOWS:
            upload = next(
                step
                for step in changes_steps(name)
                if "upload-artifact" in str(step.get("uses", ""))
            )
            with self.subTest(workflow=name):
                self.assertIn("steps.identity.outcome == 'success'", upload["if"])


class SingleCarrierTests(unittest.TestCase):
    """One context states the reason; Hull stops adding a silent red.

    Not "the rest skip": on a code pull request the two required
    aggregators go red as well, because they do not gate on the verdict
    and `ci_require_success.py` fails on a skipped dependency. They report
    the dependency, this one reports the cause. Hull is the one removed,
    because its red said nothing the aggregators did not.

    The count is also the margin. A skipped required context satisfies
    branch protection here, so reducing the reds to one would leave a
    single context standing between an unevaluable candidate and a merge.
    """

    def test_docs_carries_the_reason(self) -> None:
        docs = workflow("ci.yml")["jobs"]["docs"]
        step = next(
            s for s in docs["steps"]
            if s.get("name", "").startswith("Require candidate lifecycle")
        )

        self.assertIn("preflight_reason", str(step["env"]))
        self.assertIn("could not be evaluated", step["run"])
        self.assertIn("PREFLIGHT_REASON", step["run"])
        self.assertIn("exit 1", step["run"])

    def test_hull_skips_a_failed_verdict_but_not_a_failed_job(self) -> None:
        condition = workflow("conformance.yml")["jobs"]["conformance"]["if"]

        # Skips when the verdict failed: the carrier reports it instead.
        self.assertNotIn("candidate_preflight != 'success'", condition)
        self.assertIn("candidate_preflight == 'success'", condition)
        # Still runs when the job itself failed, so a broken detector can
        # never skip the gate on a code pull request.
        self.assertIn("needs.changes.result != 'success'", condition)


if __name__ == "__main__":
    unittest.main()
