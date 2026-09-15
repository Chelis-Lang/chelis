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
        "base": {"sha": NEW_BASE},
        "head": {"sha": head},
    }


class CandidateLifecycleTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
