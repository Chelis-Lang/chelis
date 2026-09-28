"""The final wire census is issued by execution, never by baseline editing."""

import copy
import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from capacity_census_graph import GraphError, Leaf
from capacity_census_wire_authority import LeafClassification

ROOT = Path(__file__).resolve().parent.parent


class FinalWireCensus(unittest.TestCase):
    def test_descriptor_and_saved_receipt_cannot_construct_a_wire_witness(self):
        from capacity_census_wire_verifier import VerifiedWireCensus

        for descriptor in (
            {},
            {"kind": "transport", "id": "Span.offset"},
            {"version": 2, "rows": [], "evidence": "passed"},
        ):
            with self.assertRaisesRegex(TypeError, "actual wire verification"):
                VerifiedWireCensus(descriptor)

    def test_final_rows_preserve_derived_capacity_and_exact_identity(self):
        from capacity_census_wire_verifier import final_rows

        source = LeafClassification(
            Leaf("schema::Span.offset", "u64"),
            "TaggedTransport",
            "source-byte-coordinate",
        )
        axis = LeafClassification(
            Leaf("schema::Shape.axis", "i32"), "NumericOperation", "[05-OP-7]"
        )
        rows = final_rows((source, axis))
        self.assertEqual(
            [r["id"] for r in rows],
            ["schema::Shape.axis: i32", "schema::Span.offset: u64"],
        )
        self.assertTrue(all(r["flags"] == ["numeric-field"] for r in rows))
        for dtype in ("f16", "bf16", "f32", "f64"):
            floating = replace(
                source,
                leaf=Leaf("codec::Float.$number", dtype),
                contract="stored-value/" + dtype,
            )
            self.assertEqual(final_rows((floating,))[0]["flags"], ["float-carrier"])
        for candidates in (
            (),
            (source, source),
            (replace(source, authority="Nonnumeric"),),
            (replace(source, contract=""),),
            (replace(source, leaf=Leaf("schema::Span.offset", "bool")),),
        ):
            with self.subTest(candidates=candidates), self.assertRaises(GraphError):
                final_rows(candidates)

    def test_baseline_comparison_cannot_grant_exception_or_new_leaf_authority(self):
        from capacity_census_wire_verifier import compare_baseline, final_rows

        rows = final_rows(
            (
                LeafClassification(
                    Leaf("schema::Span.offset", "u64"),
                    "TaggedTransport",
                    "source-byte-coordinate",
                ),
            )
        )
        baseline = {"version": 2, "rows": rows}
        compare_baseline(rows, json.dumps(baseline))
        changes = []
        for key in (
            "citation",
            "grandfather",
            "successor_overrides",
            "integer_plumbing",
        ):
            changed = copy.deepcopy(baseline)
            changed[key] = "reviewed"
            changes.append(changed)
        for field, value in (
            ("id", "Metadata.value: f64"),
            ("flags", []),
            ("authority", "permanent-disposition"),
            ("contract", ""),
        ):
            changed = copy.deepcopy(baseline)
            changed["rows"][0][field] = value
            changes.append(changed)
        changes += [
            {"version": version, "rows": rows} for version in (1, 2.0, "2", True, None)
        ]
        changes += [{"version": 2, "rows": []}]
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaises(GraphError):
                compare_baseline(rows, json.dumps(changed))

    def test_wire_cli_rejects_supplied_artifact_before_attempting_discovery(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/capacity_census_typed.py"),
                "wire",
                "--rustdoc-json",
                "/missing-supplied-wire.json",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "wire authority requires actual artifact and codec execution", result.stderr
        )
        self.assertNotIn("No such file", result.stderr)


if __name__ == "__main__":
    unittest.main()
