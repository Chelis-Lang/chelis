"""Unit tests for the opaque-invariants corpus generator (no binary needed).

Run with the uv-managed interpreter:
    .venv/bin/python -m unittest -v \
        tests.corpus.opaque_invariants.test_generate_corpus
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_corpus as gc  # noqa: E402


class GeneratorTests(unittest.TestCase):
    def test_programs_have_unique_ids_and_filenames(self):
        progs = gc.build_programs()
        ids = [p.id for p in progs]
        files = [p.filename for p in progs]
        self.assertEqual(len(set(ids)), len(ids), "ids must be unique")
        self.assertEqual(len(set(files)), len(files), "filenames must be unique")

    def test_every_program_has_at_least_one_target(self):
        for p in gc.build_programs():
            self.assertTrue(p.targets, f"{p.id} has no targets")

    def test_lanes_are_known(self):
        for p in gc.build_programs():
            self.assertIn(p.lane, ("check", "prove"), f"{p.id} bad lane {p.lane}")

    def test_check_lane_files_are_dp_or_ch(self):
        for p in gc.build_programs():
            if p.lane == "check":
                self.assertTrue(p.filename.endswith((".dp", ".ch")), p.filename)

    def test_prove_lane_files_are_ch(self):
        for p in gc.build_programs():
            if p.lane == "prove":
                self.assertTrue(p.filename.endswith(".ch"), p.filename)

    def test_six_opacity_rejections_are_present(self):
        ids = {p.id for p in gc.build_programs()}
        required = {
            "check_rej_record",
            "check_rej_ctor_app",
            "check_rej_ctor_ref",
            "check_rej_pat_record",
            "check_rej_pat_ctor",
            "check_rej_field_access",
            "check_rej_record_update",
            "check_rej_cast_into",
            "check_rej_lit_forge",
            "check_rej_sixth_unexported",
        }
        self.assertTrue(required.issubset(ids), required - ids)

    def test_all_wf_error_classes_are_present(self):
        ids = {p.id for p in gc.build_programs()}
        required = {
            "check_wf_invariant_no_opaque",
            "check_wf_two_variants",
            "check_wf_value_class",
            "check_wf_grammar",
            "check_wf_free_var",
            "check_wf_non_boolean",
            "check_wf_amenability_mismatch",
            "check_wf_missing_amenability",
        }
        self.assertTrue(required.issubset(ids), required - ids)

    def test_obligation_statuses_are_targeted(self):
        targets = {t for p in gc.build_programs() for t in p.targets}
        for s in ("status:passed", "status:failed", "status:unsupported", "status:error"):
            self.assertIn(s, targets, f"no program targets {s}")

    def test_producer_positions_are_covered(self):
        targets = {t for p in gc.build_programs() for t in p.targets}
        for pos in ("position:Direct", "position:Option", "position:tuple", "position:constant"):
            self.assertIn(pos, targets, f"no program targets {pos}")

    def test_covered_or_rejected_containers_are_present(self):
        ids = {p.id for p in gc.build_programs()}
        self.assertTrue({
            "prove_error_list_container",
            "prove_error_record_wrapper",
            "prove_error_alias_record_field",
        }.issubset(ids))

    def test_positive_cases_present_in_both_lanes(self):
        progs = gc.build_programs()
        self.assertTrue([p for p in progs if p.lane == "check" and p.kind_positive])
        self.assertTrue([p for p in progs if p.lane == "prove" and p.kind_positive])

    def test_write_corpus_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            progs = gc.build_programs()
            manifest = gc.write_corpus(progs, out_dir=tmp)
            self.assertEqual(manifest["program_count"], len(progs))
            for rec in manifest["programs"]:
                self.assertTrue((tmp / "programs" / rec["filename"]).is_file())
            union = sorted({t for p in progs for t in p.targets})
            self.assertEqual(manifest["targets"], union)
            on_disk = json.loads((tmp / "manifest.json").read_text())
            self.assertEqual(on_disk, manifest)

    def test_write_corpus_does_not_touch_committed_tree(self):
        # Generating into a temp dir must leave the committed manifest intact.
        committed = (gc.HERE / "manifest.json").read_text()
        with tempfile.TemporaryDirectory() as d:
            gc.write_corpus(gc.build_programs(), out_dir=Path(d))
        self.assertEqual((gc.HERE / "manifest.json").read_text(), committed)

    def test_expect_exit_values_are_valid(self):
        for p in gc.build_programs():
            self.assertIn(p.expect_exit, (0, 1, 2, 3), f"{p.id}: {p.expect_exit}")
            if p.lane == "check":
                self.assertIn(p.expect_exit, (0, 2), f"{p.id}: check exit must be 0/2")


if __name__ == "__main__":
    unittest.main()
