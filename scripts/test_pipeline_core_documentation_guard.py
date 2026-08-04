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

    def test_future_target_is_not_a_false_current_claim(self) -> None:
        self.assertFalse(
            guard.false_no_std_claim(
                "`#![no_std]` support is a future target, not a current capability."
            )
        )


if __name__ == "__main__":
    unittest.main()
