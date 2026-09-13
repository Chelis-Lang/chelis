#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_reviewed_unsupported_wording_snapshot as generator


class ReviewedUnsupportedWordingSnapshotTests(unittest.TestCase):
    def test_write_check_and_byte_idempotence(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "snapshot.json"
            self.assertEqual(generator.run(output, check=False), 0)
            first = output.read_bytes()
            self.assertEqual(generator.run(output, check=True), 0)
            self.assertEqual(generator.run(output, check=False), 0)
            self.assertEqual(output.read_bytes(), first)

    def test_check_rejects_stale_wording(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "snapshot.json"
            output.write_text("{}\n", encoding="utf-8")
            self.assertEqual(generator.run(output, check=True), 1)

    def test_softmax_wording_has_one_reviewed_source_row(self) -> None:
        rows = generator.reviewed_expectation_rows()
        self.assertEqual(list(rows), ["annotated_concat_softmax_c"])
        self.assertTrue(rows["annotated_concat_softmax_c"].startswith("error: unsupported:"))

    def test_provenance_does_not_claim_production_derivation(self) -> None:
        self.assertEqual(
            generator.SNAPSHOT_PROVENANCE, "centralized-reviewed-expectation"
        )


if __name__ == "__main__":
    unittest.main()
