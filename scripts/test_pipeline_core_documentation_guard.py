#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_core_documentation_guard as guard


class PipelineCoreDocumentationGuardTests(unittest.TestCase):
    def test_inventory_that_states_the_std_requirement_passes(self) -> None:
        guard.validate_text(
            """# Standard-library blocker inventory

`chelis-pipeline-core` requires `std`.

Collections and allocation
Global state
Stack support
Panic behavior
Operating-system use
Dependency features

`chelis-deep`
`chelis-types`
`chelis-effects`
`chelis-ir`
`chelis-unord`
"""
        )

    def test_false_current_no_std_claim_is_rejected(self) -> None:
        claims = (
            "`chelis-pipeline-core` supports `#![no_std]`.",
            "The core is no_std compatible.",
            "The core supports `#![no_std]` and is future-ready.",
            "The core supports `#![no_std]`; it no longer requires std.",
        )
        for claim in claims:
            with self.subTest(claim=claim), self.assertRaisesRegex(
                guard.DocumentationBoundaryError, "false current portability claim"
            ):
                guard.validate_text(f"# Portability\n\n{claim}\n")

    def test_environment_phrasings_are_rejected(self) -> None:
        # These are genuine false portability claims that the previous keyword
        # heuristic let through.
        claims = (
            "`chelis-pipeline-core` is portable to no_std targets.",
            "The core works in a no_std environment.",
            "The core runs in a no_std context.",
            "The core compiles as a no_std crate.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                self.assertTrue(guard.false_no_std_claim(claim), claim)

    def test_redteam_bypass_phrasings_are_rejected(self) -> None:
        # A fresh-context red team found these natural false-claim phrasings
        # bypassing the first-cut heuristic (the hyphen `no-std` spelling was a
        # whole-class escape, and `has no_std support` / `is #![no_std]` slipped
        # through). They must now be caught.
        claims = (
            "`chelis-pipeline-core` is no-std compatible.",
            "The core is portable to no-std targets.",
            "The core runs in a no-std environment.",
            "The core has no_std support.",
            "The core has no-std support.",
            "The crate is `#![no_std]`.",
            "The crate is `#![no-std]`.",
            "The core now provides no_std support.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                self.assertTrue(guard.false_no_std_claim(claim), claim)

    def test_broadening_keeps_zero_false_positives(self) -> None:
        # Honest blocker-analysis and status lines that mention no_std must stay
        # unflagged, including the hypothetical `would require` framing and a
        # bare `no_std blockers` topic mention.
        honest = (
            "no_std support would require a custom allocator.",
            "A no_std port would replace the thread-local guard.",
            "std is required; no_std is out of scope for this crate.",
            "The no_std blockers are collections, allocation, and global state.",
            "Porting to no_std is not possible while it needs alloc.",
            "`chelis-pipeline-core` requires `std`, not no-std.",
        )
        for line in honest:
            with self.subTest(line=line):
                self.assertFalse(guard.false_no_std_claim(line), line)

    def test_future_target_is_not_a_false_current_claim(self) -> None:
        self.assertFalse(
            guard.false_no_std_claim(
                "`#![no_std]` support is a future target, not a current capability."
            )
        )

    def test_true_requires_std_statement_is_not_a_false_claim(self) -> None:
        # The previous heuristic false-positived on this true statement because
        # the affirmative `supports ... no_std` pattern fired before the bare
        # `not no_std` negation was recognized.
        acceptable = (
            "`chelis-pipeline-core` supports std only, not no_std.",
            "`chelis-pipeline-core` requires `std`.",
            "It does not support no_std yet.",
            "no_std support is not available in this crate.",
        )
        for line in acceptable:
            with self.subTest(line=line):
                self.assertFalse(guard.false_no_std_claim(line), line)

    def test_blocker_analysis_mention_is_not_a_false_claim(self) -> None:
        # The inventory must be free to discuss what no_std would require
        # without the discussion being read as a current capability claim.
        analysis = (
            "Collections and allocation: BTreeMap needs alloc; no_std would "
            "require a custom allocator.",
            "Global state: a no_std port would replace the thread-local guard.",
        )
        for line in analysis:
            with self.subTest(line=line):
                self.assertFalse(guard.false_no_std_claim(line), line)

    def test_true_statement_inside_a_full_inventory_passes(self) -> None:
        guard.validate_text(
            """# Standard-library blocker inventory

`chelis-pipeline-core` requires `std`. It supports std only, not no_std.

Collections and allocation
Global state
Stack support
Panic behavior
Operating-system use
Dependency features

`chelis-deep`
`chelis-types`
`chelis-effects`
`chelis-ir`
`chelis-unord`
"""
        )


if __name__ == "__main__":
    unittest.main()
