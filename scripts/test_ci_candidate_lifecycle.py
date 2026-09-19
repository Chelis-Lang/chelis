"""Candidate-history classification and acknowledgement controls."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts import ci_candidate_lifecycle as lifecycle


OLD_BASE = "1" * 40
NEW_BASE = "2" * 40
BEFORE = "3" * 40
HEAD = "4" * 40
MERGE = "5" * 40
TARGET = "6" * 40


class FakeGraph:
    def __init__(
        self,
        *,
        ancestors: set[tuple[str, str]],
        merge_bases: dict[tuple[str, str], str] | None = None,
        merges: list[tuple[str, tuple[str, ...]]] | None = None,
    ) -> None:
        self.ancestors = ancestors
        self.merge_bases = merge_bases or {}
        self.merges = merges or []

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return ancestor == descendant or (ancestor, descendant) in self.ancestors

    def merge_base(self, left: str, right: str) -> str:
        return self.merge_bases[(left, right)]

    def new_merge_commits(
        self, before: str, head: str
    ) -> list[tuple[str, tuple[str, ...]]]:
        self.assert_range = (before, head)
        return self.merges


class UnavailableGraph(FakeGraph):
    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        raise lifecycle.GraphInspectionError("old force-pushed head is unavailable")


def payload(*, body: str = "", action: str = "synchronize") -> dict:
    return {
        "action": action,
        "before": BEFORE,
        "after": HEAD,
        "pull_request": {
            "body": body,
            "base": {"sha": NEW_BASE},
            "head": {"sha": HEAD},
        },
    }


def current_pr(*, body: str = "", head: str = HEAD) -> dict:
    return {
        "body": body,
        "base": {"sha": NEW_BASE, "ref": "main"},
        "head": {"sha": head},
    }


class CandidateLifecycleTests(unittest.TestCase):
    def test_non_pr_event_needs_no_candidate_declaration(self) -> None:
        self.assertEqual(
            lifecycle.validate_payload(
                {"ref": "refs/heads/main"},
                FakeGraph(ancestors=set()),
            ),
            "unchanged-candidate",
        )

    def test_opened_candidate_needs_no_history_acknowledgement(self) -> None:
        self.assertEqual(
            lifecycle.validate_payload(payload(action="opened"), FakeGraph(ancestors=set())),
            "initial-candidate",
        )

    def test_ordinary_review_repair_push_is_not_a_base_update(self) -> None:
        graph = FakeGraph(ancestors={(BEFORE, HEAD)}, merges=[])
        self.assertEqual(
            lifecycle.validate_payload(payload(), graph),
            "review-repair",
        )

    def test_merge_from_base_requires_exact_head_bound_reason(self) -> None:
        graph = FakeGraph(
            ancestors={(BEFORE, HEAD), (NEW_BASE, NEW_BASE)},
            merges=[(MERGE, (BEFORE, NEW_BASE))],
        )
        with self.assertRaisesRegex(ValueError, "Candidate-base-update"):
            lifecycle.validate_payload(payload(), graph)
        with self.assertRaisesRegex(ValueError, "current head"):
            lifecycle.validate_payload(
                payload(
                    body=(
                        f"Candidate-base-update: {BEFORE} "
                        "required conflict resolution"
                    )
                ),
                graph,
            )
        self.assertEqual(
            lifecycle.validate_payload(
                payload(
                    body=(
                        f"Candidate-base-update: {HEAD} "
                        "required conflict resolution"
                    )
                ),
                graph,
            ),
            "base-merge",
        )

    def test_rebase_onto_newer_base_requires_base_update_reason(self) -> None:
        graph = FakeGraph(
            ancestors={(OLD_BASE, NEW_BASE)},
            merge_bases={
                (BEFORE, NEW_BASE): OLD_BASE,
                (HEAD, NEW_BASE): NEW_BASE,
            },
        )
        with self.assertRaisesRegex(ValueError, "Candidate-base-update"):
            lifecycle.validate_payload(payload(), graph)
        self.assertEqual(
            lifecycle.validate_payload(
                payload(
                    body=(
                        f"Candidate-base-update: {HEAD} "
                        "base-sensitive overlap required a rebase"
                    )
                ),
                graph,
            ),
            "base-rebase",
        )

    def test_other_history_rewrite_requires_its_own_exact_reason(self) -> None:
        graph = FakeGraph(
            ancestors=set(),
            merge_bases={
                (BEFORE, NEW_BASE): OLD_BASE,
                (HEAD, NEW_BASE): OLD_BASE,
            },
        )
        with self.assertRaisesRegex(ValueError, "Candidate-history-rewrite"):
            lifecycle.validate_payload(payload(), graph)
        self.assertEqual(
            lifecycle.validate_payload(
                payload(
                    body=(
                        f"Candidate-history-rewrite: {HEAD} "
                        "approved consolidation before review"
                    )
                ),
                graph,
            ),
            "history-rewrite",
        )

    def test_live_pr_body_repairs_a_rerun_of_the_same_event(self) -> None:
        graph = FakeGraph(
            ancestors={(OLD_BASE, NEW_BASE)},
            merge_bases={
                (BEFORE, NEW_BASE): OLD_BASE,
                (HEAD, NEW_BASE): NEW_BASE,
            },
        )
        event = payload(body="")
        self.assertEqual(
            lifecycle.validate_payload(
                event,
                graph,
                current_pr=current_pr(
                    body=(
                        f"Candidate-base-update: {HEAD} "
                        "base-sensitive overlap required a rebase"
                    )
                ),
            ),
            "base-rebase",
        )

    def test_live_pr_head_must_still_match_the_event_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not equal current PR head"):
            lifecycle.validate_payload(
                payload(),
                FakeGraph(ancestors={(BEFORE, HEAD)}),
                current_pr=current_pr(head=MERGE),
            )

    def test_live_target_tip_controls_classification_not_event_base_snapshot(
        self,
    ) -> None:
        graph = FakeGraph(
            ancestors={(OLD_BASE, TARGET)},
            merge_bases={
                (BEFORE, TARGET): OLD_BASE,
                (HEAD, TARGET): TARGET,
            },
        )
        self.assertEqual(
            lifecycle.validate_payload(
                payload(
                    body=(
                        f"Candidate-base-update: {HEAD} "
                        "base-sensitive overlap required a rebase"
                    )
                ),
                graph,
                current_pr=current_pr(
                    body=(
                        f"Candidate-base-update: {HEAD} "
                        "base-sensitive overlap required a rebase"
                    )
                ),
                target_tip=TARGET,
            ),
            "base-rebase",
        )

    def test_force_push_declaration_survives_body_edits_and_later_pushes(
        self,
    ) -> None:
        graph = FakeGraph(ancestors={(MERGE, HEAD)}, merges=[])
        edited = payload(action="edited")
        edited["pull_request"]["head"]["sha"] = HEAD
        timeline = [
            [
                {
                    "event": "head_ref_force_pushed",
                    "commit_id": MERGE,
                }
            ]
        ]
        with self.assertRaisesRegex(ValueError, "Candidate-base-update.*or"):
            lifecycle.validate_payload(
                edited,
                graph,
                current_pr=current_pr(),
                target_tip=TARGET,
                timeline=timeline,
            )
        self.assertEqual(
            lifecycle.validate_payload(
                edited,
                graph,
                current_pr=current_pr(
                    body=(
                        f"Candidate-base-update: {MERGE} "
                        "semantic conflict resolution required the rebase"
                    )
                ),
                target_tip=TARGET,
                timeline=timeline,
            ),
            "unchanged-candidate",
        )
        self.assertEqual(
            lifecycle.validate_payload(
                edited,
                graph,
                current_pr=current_pr(
                    body="\n".join(
                        [
                            (
                                f"Candidate-base-update: {MERGE} "
                                "semantic conflict resolution required the rebase"
                            ),
                            (
                                f"Candidate-history-rewrite: {HEAD} "
                                "approved consolidation followed"
                            ),
                        ]
                    )
                ),
                target_tip=TARGET,
                timeline=timeline,
            ),
            "unchanged-candidate",
        )

    def test_prior_base_merge_declaration_survives_later_pushes(self) -> None:
        graph = FakeGraph(
            ancestors={(MERGE, HEAD), (NEW_BASE, TARGET)},
            merges=[(MERGE, (BEFORE, NEW_BASE))],
        )
        reopened = payload(action="reopened")
        with self.assertRaisesRegex(ValueError, "Candidate-base-update"):
            lifecycle.validate_payload(
                reopened,
                graph,
                current_pr=current_pr(),
                target_tip=TARGET,
                timeline=[],
            )
        self.assertEqual(
            lifecycle.validate_payload(
                reopened,
                graph,
                current_pr=current_pr(
                    body=(
                        f"Candidate-base-update: {MERGE} "
                        "main overlap required the merge"
                    )
                ),
                target_tip=TARGET,
                timeline=[],
            ),
            "unchanged-candidate",
        )

    def test_unavailable_old_head_requires_explicit_history_rewrite_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "Candidate-history-rewrite"):
            lifecycle.validate_payload(payload(), UnavailableGraph(ancestors=set()))
        self.assertEqual(
            lifecycle.validate_payload(
                payload(
                    body=(
                        f"Candidate-history-rewrite: {HEAD} "
                        "approved force push after the old head became unavailable"
                    )
                ),
                UnavailableGraph(ancestors=set()),
            ),
            "history-rewrite-unverifiable",
        )


    def test_an_uninspectable_pre_push_head_says_so_rather_than_blaming_the_author(
        self,
    ) -> None:
        """The demand must name its cause, not read as a forgotten line.

        chelis#2229: a clean forward rebase produced
        `Candidate-history-rewrite: requires exactly one exact-head line in
        the PR body` while the body already carried a correct
        `Candidate-base-update:` line. The author has no way to tell from
        that message that the checkout never obtained the pre-push head.
        """

        with self.assertRaises(ValueError) as raised:
            lifecycle.validate_payload(
                payload(
                    body=f"Candidate-base-update: {HEAD} a real conflict"
                ),
                UnavailableGraph(ancestors=set()),
            )

        message = str(raised.exception)
        self.assertIn(f"the pre-push head {BEFORE} could not be inspected", message)
        self.assertIn("old force-pushed head is unavailable", message)
        self.assertIn("cannot be classified as a base update", message)
        # The stricter declaration is still what unblocks it: fail-safe.
        self.assertIn("Candidate-history-rewrite:", message)

    def test_empty_or_duplicate_acknowledgements_fail_closed(self) -> None:
        graph = FakeGraph(
            ancestors={(BEFORE, HEAD)},
            merges=[(MERGE, (BEFORE, NEW_BASE))],
        )
        for body in (
            f"Candidate-base-update: {HEAD}",
            "\n".join(
                [
                    f"Candidate-base-update: {HEAD} conflict one",
                    f"Candidate-base-update: {HEAD} conflict two",
                ]
            ),
        ):
            with self.subTest(body=body), self.assertRaises(ValueError):
                lifecycle.validate_payload(payload(body=body), graph)

    def test_malformed_synchronize_payload_fails_closed(self) -> None:
        malformed = payload()
        del malformed["before"]
        with self.assertRaisesRegex(ValueError, "before"):
            lifecycle.validate_payload(malformed, FakeGraph(ancestors=set()))


class GitGraphIntegrationTests(unittest.TestCase):
    def git(self, root: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()

    def commit(self, root: Path, name: str, content: str) -> str:
        (root / name).write_text(content, encoding="utf-8")
        self.git(root, "add", name)
        self.git(root, "commit", "-m", name)
        return self.git(root, "rev-parse", "HEAD")

    def initialized_repo(self, root: Path) -> str:
        self.git(root, "init", "-b", "main")
        self.git(root, "config", "user.name", "Candidate Lifecycle Test")
        self.git(root, "config", "user.email", "candidate@example.invalid")
        return self.commit(root, "root.txt", "root\n")

    def test_real_base_merge_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialized_repo(root)
            self.git(root, "checkout", "-b", "feature")
            before = self.commit(root, "feature.txt", "feature\n")
            self.git(root, "checkout", "main")
            base = self.commit(root, "base.txt", "base\n")
            self.git(root, "checkout", "feature")
            self.git(root, "merge", "--no-ff", "main", "-m", "merge base")
            head = self.git(root, "rev-parse", "HEAD")

            self.assertEqual(
                lifecycle.classify_update(
                    before=before,
                    head=head,
                    base=base,
                    graph=lifecycle.GitGraph(root),
                ),
                "base-merge",
            )

    def test_real_base_rebase_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialized_repo(root)
            self.git(root, "checkout", "-b", "feature")
            before = self.commit(root, "feature.txt", "feature\n")
            self.git(root, "checkout", "main")
            base = self.commit(root, "base.txt", "base\n")
            self.git(root, "checkout", "feature")
            self.git(root, "rebase", "main")
            head = self.git(root, "rev-parse", "HEAD")

            self.assertEqual(
                lifecycle.classify_update(
                    before=before,
                    head=head,
                    base=base,
                    graph=lifecycle.GitGraph(root),
                ),
                "base-rebase",
            )



class LifecycleInvocationTests(unittest.TestCase):
    """Every invocation must hand the classifier the head it needs.

    `pr-contract-acknowledgements.yml` checks out at `fetch-depth: 0`, but a
    full clone holds only what refs reach, and a force-pushed-away head is
    reachable from none, so that step has to fetch it by SHA. The other two
    invocations already fetch it; what they did was hide the failure. The
    script now reports an uninspectable pre-push head and repeats git's
    reason, so a `2>/dev/null` at any call site throws away the half of that
    diagnosis git holds and leaves the same script diagnosing itself on one
    path and going quiet on another (chelis#2229).
    """

    WORKFLOWS = Path(__file__).resolve().parents[1] / ".github/workflows"

    def acknowledgement_step(self) -> str:
        text = (self.WORKFLOWS / "pr-contract-acknowledgements.yml").read_text()
        start = text.index("Require persistent candidate lifecycle declaration")
        end = text.index("- name: Require protected-test acknowledgements")
        return text[start:end]

    def detector_step(self, workflow: str) -> str:
        text = (self.WORKFLOWS / workflow).read_text()
        start = text.index("- name: Validate candidate lifecycle")
        end = text.index("- name: Select targeted rebase lane")
        return text[start:end]

    def all_steps(self) -> dict[str, str]:
        return {
            "pr-contract-acknowledgements.yml": self.acknowledgement_step(),
            "ci.yml": self.detector_step("ci.yml"),
            "conformance.yml": self.detector_step("conformance.yml"),
        }

    def test_the_acknowledgement_step_fetches_the_pre_push_head(self) -> None:
        body = self.acknowledgement_step()

        self.assertIn("BEFORE: ${{ github.event.before }}", body)
        self.assertIn("ACTION: ${{ github.event.action }}", body)
        self.assertIn('git fetch --no-tags origin "$BEFORE"', body)

    def test_no_invocation_silences_its_pre_push_head_fetch(self) -> None:
        for workflow, body in self.all_steps().items():
            fetches = [
                line
                for line in body.splitlines()
                if "git fetch" in line and '"$BEFORE"' in line
            ]
            with self.subTest(workflow=workflow):
                self.assertTrue(fetches, "no pre-push head fetch")
                for line in fetches:
                    self.assertNotIn("2>/dev/null", line)
                    self.assertNotIn("|| true", line)


if __name__ == "__main__":
    unittest.main()
