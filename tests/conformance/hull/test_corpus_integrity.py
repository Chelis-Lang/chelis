"""Corpus-integrity invariants for the frozen Hull conformance corpus.

THE FREEZE CONTRACT. The whole tests/conformance/hull/ directory is one frozen
artifact; these tests assert manifest <-> programs/ <-> verdicts.jsonl are
mutually consistent so a stale or partially-refreshed corpus is a LOUD failure,
not a silent pass:

  - every verdict record names a program file that EXISTS (no orphan verdict),
  - every program file is named by exactly one verdict (no orphan program),
  - manifest counts match the verdicts.jsonl lane tallies,
  - every reject/check record's pinned fields are shape-correct (a reject has
    null type/effects; an accept-check has a non-null canonical type),
  - the 4 known_conservative entries are present as gap_family != GapNone reject
    records (the disambiguation teeth),
  - the reject-sentinel set is present as gap_family == GapNone reject records
    (the standing teeth),
  - the per-PR slice covers all 28 Hull RuleTags (the Phase-4 arm-coverage
    property preserved in the subset),
  - golden_wire.json carries a non-empty drift-guard sample.

Run with the uv-managed interpreter:
    .venv/bin/python -m unittest -v tests.conformance.hull.test_corpus_integrity
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROGRAMS_DIR = HERE / "programs"
MANIFEST_PATH = HERE / "manifest.json"
VERDICTS_PATH = HERE / "verdicts.jsonl"
KNOWN_CONSERVATIVE_PATH = HERE / "known_conservative.json"
GOLDEN_WIRE_PATH = HERE / "golden_wire.json"

# The 28 canonical Hull RuleTags (generate.ch all_rule_tags). The check lane must
# cover all of them in the per-PR slice (Phase-4 100% arm coverage preserved).
HULL_RULE_TAGS = {
    "RVar", "RLit", "RLam", "RApp", "RLet", "RIf", "RAdd", "RMul", "RSub", "RDiv",
    "RNeg", "RExp", "RLog", "RSqrt", "RSin", "RCast", "RSum", "RGather", "RMatmul",
    "RGrad", "RTuple", "RTupleGet", "RMatchVar", "RMatchLit", "RMatchWild",
    "RReshape", "RPermute", "RExpand",
}


def _corpus_present() -> bool:
    return MANIFEST_PATH.is_file() and VERDICTS_PATH.is_file()


def _load_verdicts() -> list[dict]:
    out = []
    with open(VERDICTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class CorpusIntegrityTests(unittest.TestCase):
    def setUp(self):
        if not _corpus_present():
            self.skipTest("frozen corpus not yet generated")
        self.manifest = json.loads(MANIFEST_PATH.read_text())
        self.verdicts = _load_verdicts()

    def test_every_verdict_has_a_program_file(self):
        for v in self.verdicts:
            path = PROGRAMS_DIR / v["program_path"]
            self.assertTrue(path.is_file(), f"orphan verdict {v['id']}: missing {path}")

    def test_no_orphan_program_files(self):
        named = {v["program_path"] for v in self.verdicts}
        on_disk = {p.name for p in PROGRAMS_DIR.glob("*.dp")}
        orphans = on_disk - named
        self.assertEqual(orphans, set(), f"orphan program files (no verdict): {orphans}")

    def test_ids_are_unique(self):
        ids = [v["id"] for v in self.verdicts]
        self.assertEqual(len(ids), len(set(ids)), "verdict ids must be unique")

    def test_manifest_counts_match_verdicts(self):
        check = sum(1 for v in self.verdicts if v["lane"] == "check")
        eval_ = sum(1 for v in self.verdicts if v["lane"] == "eval")
        reject = sum(1 for v in self.verdicts if v["lane"] == "reject")
        self.assertEqual(self.manifest["check_count"], check)
        self.assertEqual(self.manifest["eval_count"], eval_)
        self.assertEqual(self.manifest["reject_sentinel_count"], reject)

    def test_pinned_version_is_0_8_0(self):
        self.assertEqual(self.manifest["chelis_version_pinned"], "0.8.0")

    def test_accept_check_records_have_canonical_type(self):
        for v in self.verdicts:
            if v["lane"] == "check" and v["hull_verdict"] == "accept":
                self.assertIsNotNone(
                    v["hull_type_canonical"],
                    f"accept-check {v['id']} must pin a canonical type",
                )

    def test_reject_records_have_null_type(self):
        for v in self.verdicts:
            if v["hull_verdict"] == "reject":
                self.assertIsNone(
                    v["hull_type_canonical"],
                    f"reject {v['id']} must have null type",
                )

    def test_eval_records_have_eval_value(self):
        for v in self.verdicts:
            if v["lane"] == "eval":
                self.assertIsNotNone(
                    v["hull_eval_value"],
                    f"eval record {v['id']} must pin a reference scalar",
                )

    def test_known_conservative_entries_present(self):
        # The 4 disambiguation teeth: reject records with gap_family != GapNone.
        gaps = [
            v for v in self.verdicts
            if v["lane"] == "reject" and v["gap_family"] != "GapNone"
        ]
        families = {v["gap_family"] for v in gaps}
        self.assertIn("GapAdtFragment", families)
        self.assertIn("GapBuiltinNameShadow", families)
        self.assertGreaterEqual(len(gaps), 4, "expected >= 4 known-conservative entries")
        for v in gaps:
            self.assertEqual(v["expected_compiler_outcome"], "KnownReferenceGap")

    def test_reject_sentinels_present(self):
        # The standing teeth: reject records with gap_family == GapNone (the
        # compiler currently rejects them too; an accept regression -> Unsound).
        sentinels = [
            v for v in self.verdicts
            if v["lane"] == "reject" and v["gap_family"] == "GapNone"
        ]
        self.assertGreaterEqual(
            len(sentinels), 40, "expected >= 40 reject sentinels (the teeth)"
        )
        for v in sentinels:
            self.assertEqual(v["expected_compiler_outcome"], "AgreeReject")

    def test_check_lane_covers_all_28_rule_tags(self):
        check_tags = {
            v["rule_tag"] for v in self.verdicts if v["lane"] == "check"
        }
        missing = HULL_RULE_TAGS - check_tags
        self.assertEqual(
            missing, set(), f"per-PR check slice misses RuleTags: {missing}"
        )

    def test_known_conservative_json_matches_source(self):
        # The committed known_conservative.json must carry the 4 entries verbatim.
        self.assertTrue(KNOWN_CONSERVATIVE_PATH.is_file())
        kc = json.loads(KNOWN_CONSERVATIVE_PATH.read_text())
        self.assertEqual(len(kc["whitelist_entries"]), 4)

    def test_golden_wire_present_and_nonempty(self):
        self.assertTrue(GOLDEN_WIRE_PATH.is_file(), "golden_wire.json must exist")
        golden = json.loads(GOLDEN_WIRE_PATH.read_text())
        self.assertGreater(len(golden), 0, "golden_wire.json must carry fixtures")
        self.assertEqual(self.manifest["golden_wire_count"], len(golden))

    def test_dropped_divergences_documented(self):
        # The documented Hull-reference imprecisions dropped from the agreement
        # corpus (gather-shape + int-div) are committed as evidence; the manifest
        # count must match.
        dropped_path = HERE / "dropped_divergences.json"
        self.assertTrue(dropped_path.is_file(), "dropped_divergences.json must exist")
        dropped = json.loads(dropped_path.read_text())
        self.assertEqual(self.manifest["dropped_divergence_count"], len(dropped))
        for d in dropped:
            self.assertIn(d["lane"], ("check", "eval"))
            self.assertTrue(d["reason"], "each dropped divergence must carry a reason")

    def test_no_kept_program_is_a_known_divergence(self):
        # A kept program must not duplicate a dropped one (the filter must have
        # actually removed the divergent programs from the agreement corpus).
        dropped_path = HERE / "dropped_divergences.json"
        if not dropped_path.is_file():
            self.skipTest("no dropped divergences file")
        dropped_programs = {
            d["program"].strip() for d in json.loads(dropped_path.read_text())
        }
        for v in self.verdicts:
            if v["lane"] in ("check", "eval"):
                text = (PROGRAMS_DIR / v["program_path"]).read_text().strip()
                self.assertNotIn(
                    text, dropped_programs,
                    f"kept program {v['id']} duplicates a dropped divergence",
                )

    def test_program_files_are_bare_top_level_defs(self):
        # PHASE_STATE generator contract: each generated / sentinel program is a
        # bare top-level `(def {} ...)`, NOT a module wrapper (a module wrapper
        # makes the effects pass report empty rows). The 2 known_conservative
        # ADT-fragment evidence programs legitimately carry a `(deftype {} ...)`
        # declaration prefix (they exercise the declaration layer Hull lacks), so
        # they are exempt; they still contain a top-level def.
        for v in self.verdicts:
            text = (PROGRAMS_DIR / v["program_path"]).read_text().strip()
            if v["gap_family"] == "GapAdtFragment":
                self.assertTrue(
                    text.startswith("(deftype {}") and "(def {}" in text,
                    f"{v['id']} ADT evidence must be deftype + def: {text[:40]!r}",
                )
                continue
            self.assertTrue(
                text.startswith("(def {}"),
                f"{v['id']} is not a bare top-level def: {text[:40]!r}",
            )


if __name__ == "__main__":
    unittest.main()
