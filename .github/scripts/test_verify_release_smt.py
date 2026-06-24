"""Unit tests for verify_release_smt.py's record classification.

These cover the pure parsing/classification surface (no chelis binary
needed): the cvc5-smt record detector and the NDJSON parser. The
end-to-end behavior (running a real binary) is exercised by the release
workflow itself and validated locally during WS-4 against both an
smt-enabled and a feature-less binary.

The JSON shapes below are copied verbatim from real
`chelis prove --json --tier auto` output (chelis 0.10.0):
  * a cvc5-discharged producer obligation (smt-enabled binary), and
  * the smt-disabled warning record (feature-less binary).
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent / "verify_release_smt.py"
    spec = importlib.util.spec_from_file_location("verify_release_smt", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VRS = _load_module()


# A real cvc5-discharged obligation record from the smt-enabled binary.
CVC5_OBLIGATION = {
    "arith_model": "real",
    "assumptions": [
        {
            "discharge": {
                "evidence": {
                    "arith_model": "real",
                    "obligation": "invariant:Unit:scale_half",
                    "status": "proved",
                },
                "method": "smt",
            },
            "discharge_tier": {
                "engine": "cvc5",
                "guarantee": "smt",
                "source": "invariant:Unit:scale_half",
            },
            "name": "invariant:Unit:scale_half",
        }
    ],
    "kind": "obligation",
    "proof_tier": "smt",
    "status": "passed",
}

# A fuzz-tier property record (the kind the solver-free path emits).
FUZZ_PROPERTY = {
    "assumptions": [
        {
            "discharge_tier": {
                "engine": "fuzz-sampler",
                "guarantee": "fuzz",
                "source": "invariant:Probability:binder:p",
            },
        }
    ],
    "kind": "property",
    "proof_tier": "fuzz",
    "status": "passed",
}

# The structured warning the feature-less binary emits on the
# producer-obligation path.
SMT_DISABLED_WARNING = {
    "kind": "warning",
    "reason": (
        "producer obligation verification requires the smt-enabled build "
        "(--features smt); obligations were not SMT-verified in this build"
    ),
    "skipped": 1,
    "stage": "obligations",
}


class RecordIsCvc5SmtTest(unittest.TestCase):
    def test_cvc5_smt_obligation_is_detected(self) -> None:
        self.assertTrue(VRS.record_is_cvc5_smt(CVC5_OBLIGATION))

    def test_fuzz_record_is_not_cvc5_smt(self) -> None:
        self.assertFalse(VRS.record_is_cvc5_smt(FUZZ_PROPERTY))

    def test_smt_tier_without_cvc5_engine_is_rejected(self) -> None:
        # proof_tier says smt but the engine is not cvc5: must not pass.
        record = {
            "proof_tier": "smt",
            "assumptions": [
                {"discharge_tier": {"engine": "z3", "guarantee": "smt"}}
            ],
        }
        self.assertFalse(VRS.record_is_cvc5_smt(record))

    def test_cvc5_engine_without_smt_tier_is_rejected(self) -> None:
        # engine cvc5 but proof_tier is not smt: must not pass.
        record = {
            "proof_tier": "fuzz",
            "assumptions": [
                {"discharge_tier": {"engine": "cvc5", "guarantee": "smt"}}
            ],
        }
        self.assertFalse(VRS.record_is_cvc5_smt(record))

    def test_summary_record_is_not_cvc5_smt(self) -> None:
        self.assertFalse(
            VRS.record_is_cvc5_smt({"kind": "summary", "obligations": 2})
        )


class ParseRecordsTest(unittest.TestCase):
    def test_skips_blank_and_non_json_lines(self) -> None:
        stdout = (
            "\n"
            "warning: a human-readable warning line\n"
            '{"kind":"summary","obligations":2}\n'
            "  \n"
        )
        records = VRS.parse_records(stdout)
        self.assertEqual(records, [{"kind": "summary", "obligations": 2}])

    def test_parses_multiple_ndjson_records(self) -> None:
        import json

        stdout = json.dumps(CVC5_OBLIGATION) + "\n" + json.dumps(FUZZ_PROPERTY)
        records = VRS.parse_records(stdout)
        self.assertEqual(len(records), 2)


class MarkerTest(unittest.TestCase):
    def test_marker_present_in_disabled_warning(self) -> None:
        import json

        self.assertIn(VRS.SMT_DISABLED_MARKER, json.dumps(SMT_DISABLED_WARNING))

    def test_marker_absent_from_cvc5_obligation(self) -> None:
        import json

        self.assertNotIn(VRS.SMT_DISABLED_MARKER, json.dumps(CVC5_OBLIGATION))


class FindChelisInTreeTest(unittest.TestCase):
    """The --tarball mode extracts a staged release .tar.gz and locates
    `<staging>/bin/chelis`. Cover the locator without a real binary."""

    def test_finds_staging_bin_chelis(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "chelis-0.10.0-linux-x86_64" / "bin" / "chelis"
            target.parent.mkdir(parents=True)
            target.write_text("#!/bin/sh\n")
            found = VRS.find_chelis_in_tree(root)
            self.assertEqual(found, target)

    def test_returns_none_when_no_chelis(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "staging" / "lib").mkdir(parents=True)
            (root / "staging" / "lib" / "libchelis_runtime.a").write_text("x")
            self.assertIsNone(VRS.find_chelis_in_tree(root))

    def test_prefers_bin_chelis_over_stray_file(self) -> None:
        # A stray file literally named `chelis` elsewhere must not win
        # over the canonical `*/bin/chelis` staging path.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stray = root / "docs" / "chelis"
            stray.parent.mkdir(parents=True)
            stray.write_text("not the binary")
            canonical = root / "chelis-x" / "bin" / "chelis"
            canonical.parent.mkdir(parents=True)
            canonical.write_text("#!/bin/sh\n")
            self.assertEqual(VRS.find_chelis_in_tree(root), canonical)


if __name__ == "__main__":
    unittest.main()
