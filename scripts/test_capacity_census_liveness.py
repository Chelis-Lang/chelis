#!/usr/bin/env python3
"""Unit tests for the capacity-census liveness gate's pure verdict logic.

Run: .venv/bin/python scripts/test_capacity_census_liveness.py
"""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import capacity_census_liveness

from capacity_census_liveness import (
    IssueKind,
    IssueRecord,
    IssueState,
    adjudicate,
    extract_issue_refs,
    fetch_issue,
    load_census_rows,
)


RETIRED_NATIVE_DISPOSITION = (
    "permanent-disposition(C6 registered PyO3 signature surface complete descriptor set ratified 2026-08-04)"
)


def row(
    citation: str,
    row_id: str = "x.h: void f(void);",
    census_family: str = "primary",
) -> dict:
    return {
        "kind": "header-export",
        "id": row_id,
        "citation": citation,
        "_census_family": census_family,
    }


def final_wire_row(authority: str = "TaggedTransport") -> dict:
    return {
        "kind": "wire-schema-numeric-field",
        "id": "fixture::WireValue.value: u64",
        "flags": ["numeric-field"],
        "authority": authority,
        "contract": "source-offset" if authority == "TaggedTransport" else "[05-OP-65]",
    }


def load_wire_json(source: str) -> list[dict]:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        census = Path("capacity_census_wire.json")
        (root / census).write_text(source)
        return load_census_rows(root, (census,))


def load_wire(payload: object) -> list[dict]:
    return load_wire_json(json.dumps(payload))


class ExtractIssueRefs(unittest.TestCase):
    def test_extracts_and_dedupes_in_order(self) -> None:
        refs = extract_issue_refs("seam unwinds per chelis#893/chelis#894; see chelis#893")
        self.assertEqual(refs, [893, 894])

    def test_baseline_tag_has_no_refs(self) -> None:
        self.assertEqual(extract_issue_refs("baseline-2026-07-30"), [])


class LoadCensusRows(unittest.TestCase):
    def test_binding_final_shape_keeps_execution_evidence_out_of_baseline(self) -> None:
        transport = {"kind": "binding-pyfunction", "id": "chelis_python::eval_json(typed)",
                     "flags": ["float-carrier", "numeric-param", "numeric-return"],
                     "authority": "TaggedTransport",
                     "contract": "compiler-json/chelis_python::eval_json"}
        nonnumeric = {"kind": "binding-pymethod", "id": "chelis_python::CompiledModel::path(typed)",
                      "flags": [], "authority": "nonnumeric"}
        valid = {"version": 2, "rows": [nonnumeric, transport]}
        capacity_census_liveness.validate_binding_baseline(valid)
        for key in ("citation", "graph_identity", "source_sha256", "evidence",
                    "successor_overrides"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                capacity_census_liveness.validate_binding_baseline({**valid, key: "supplied"})
            with self.assertRaises(ValueError):
                capacity_census_liveness.validate_binding_baseline({"version": 2, "rows": [{**transport, key: "supplied"}]})

    def test_every_binding_row_rejects_all_legacy_citations(self) -> None:
        root = Path(__file__).resolve().parents[1]
        baseline = json.loads((root / "spec/design/capacity_census_bindings.json").read_text())
        for source in baseline["rows"]:
            for citation in ("", "chelis#893", RETIRED_NATIVE_DISPOSITION):
                with self.subTest(identity=source["id"], citation=citation):
                    changed = {**source, "citation": citation}
                    with self.assertRaises(ValueError):
                        capacity_census_liveness.validate_binding_baseline({"version": 2, "rows": [changed]})
                    problems = adjudicate(
                        [{**changed, "_census_family": "bindings"}],
                        {893: IssueRecord(IssueKind.ISSUE, IssueState.OPEN)},
                    )
                    self.assertTrue(problems)
                    legacy = {key: value for key, value in source.items()
                              if key not in {"authority", "contract"}}
                    legacy["citation"] = citation
                    with self.assertRaises(ValueError):
                        capacity_census_liveness.validate_binding_baseline({"version": 2, "rows": [legacy]})

    def test_final_native_contracts_and_shape_operation_match_the_rust_format(self) -> None:
        cases = (
            ("CompiledModel::__call__", "TaggedTransport", "native/compiled-tensor-call"),
            ("NativeTensor::__dlpack__", "TaggedTransport", "native/dlpack-capsule"),
            ("NativeTensor::__dlpack_device__", "TaggedTransport", "native/dlpack-device"),
            ("NativeTensor::shape", "NumericOperation", "[05-OP-45]"),
        )
        for owner, authority, contract in cases:
            source = {"kind": "binding-pymethod", "id": f"chelis_python::{owner}(typed)",
                      "flags": ["numeric-return"], "authority": authority, "contract": contract}
            with self.subTest(owner=owner):
                capacity_census_liveness.validate_binding_baseline({"version": 2, "rows": [source]})
                self.assertEqual(adjudicate([{**source, "_census_family": "bindings"}], {}), [])
                for change in ({"id": "chelis_python::unknown(typed)"},
                               {"kind": "binding-pyfunction"}, {"contract": "other"},
                               {"contract": "[05-OP-999]"}, {"flags": []},
                               {"flags": ["unknown-capacity"]}, {"flags": ["numeric-return", "numeric-return"]},
                               {"authority": "legacy"}, {"evidence": "passed"}):
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        capacity_census_liveness.validate_binding_baseline({"version": 2, "rows": [{**source, **change}]})

    def test_binding_shape_rejects_missing_contract_erased_capacity_and_duplicates(self) -> None:
        row = {"kind": "binding-pyfunction", "id": "chelis_python::check_json(typed)",
               "flags": ["float-carrier", "numeric-return"], "authority": "TaggedTransport",
               "contract": "compiler-json/chelis_python::check_json"}
        for change in ({"flags": []}, {"contract": "other"}, {"authority": "permanent-disposition"},
                       {"graph_identity": "a" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                capacity_census_liveness.validate_binding_baseline({"version": 2, "rows": [{**row, **change}]})
        with self.assertRaises(ValueError):
            capacity_census_liveness.validate_binding_baseline({"version": 2, "rows": [row, row]})

    def test_wire_final_authorities_need_no_issue_lookup(self) -> None:
        for authority in ("TaggedTransport", "NumericOperation"):
            for primitive, flag in (("u64", "numeric-field"), ("f64", "float-carrier")):
                with self.subTest(authority=authority, primitive=primitive):
                    source = {
                        **final_wire_row(authority),
                        "id": f"fixture::WireValue.value: {primitive}",
                        "flags": [flag],
                    }
                    rows = load_wire({"version": 2, "rows": [source]})
                    self.assertEqual(rows, [{**source, "_census_family": "wire"}])
                    self.assertEqual(adjudicate(rows, {}), [])

    def test_wire_duplicate_json_keys_cannot_overwrite_the_contract(self) -> None:
        valid = json.dumps({"version": 2, "rows": [final_wire_row()]})
        for source in (
            valid.replace('"version": 2', '"version": 1, "version": 2'),
            valid.replace('"authority":', '"authority": "Grandfather", "authority":'),
        ):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "DUPLICATE"):
                load_wire_json(source)

    def test_wire_inventory_size_is_not_a_liveness_authority(self) -> None:
        # Completeness is the live graph verifier's responsibility, not a
        # hard-coded row count in the issue liveness ledger.
        for count in (0, 1, 3):
            with self.subTest(count=count):
                rows = [
                    {**final_wire_row(), "id": f"fixture::Offsets.offset_{n}: u64"}
                    for n in range(count)
                ]
                self.assertEqual(len(load_wire({"version": 2, "rows": rows})), count)

    def test_wire_baseline_requires_exact_version_2_envelope(self) -> None:
        valid = {"version": 2, "rows": [final_wire_row()]}
        malformed = [{"rows": valid["rows"]}]
        malformed.extend({**valid, "version": value} for value in (1, 3, "2", 2.0, True))
        malformed.extend({**valid, "rows": value} for value in (None, {}, "rows"))
        malformed.extend({**valid, key: value} for key, value in (
            ("citation", ""), ("citation", "chelis#1288"),
            ("source_sha256", "baseline cannot supply execution evidence"),
        ))
        malformed.append([])
        for payload in malformed:
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "WIRE BASELINE"):
                load_wire(payload)

    def test_wire_rows_reject_legacy_fields_and_nonfinal_shapes(self) -> None:
        valid = final_wire_row()
        malformed = []
        for key in valid:
            incomplete = dict(valid)
            del incomplete[key]
            malformed.append(incomplete)
        malformed.extend({**valid, key: value} for key, value in (
            ("citation", ""),
            ("citation", "chelis#1288"),
            ("disposition", "permanent-disposition(reviewed)"),
            ("authority", "Nonnumeric"),
            ("authority", "Grandfather"),
            ("authority", ["TaggedTransport", "NumericOperation"]),
            ("kind", "binding-parameter"),
            ("id", ""),
            ("id", 12),
            ("flags", []),
            ("flags", ["numeric-field", "float-carrier"]),
            ("flags", ["numeric-field", "numeric-field"]),
            ("flags", ["raw-dtype-int"]),
            ("contract", ""),
            ("contract", None),
        ))
        malformed.append(None)
        for source in malformed:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "WIRE BASELINE"):
                load_wire({"version": 2, "rows": [source]})

    def test_wire_numeric_operation_requires_atom_grammar(self) -> None:
        for contract in ("chelis#1288", "[05-OBS-1]", "05-OP-65", "[05-OP-N]"):
            source = {**final_wire_row("NumericOperation"), "contract": contract}
            with self.subTest(contract=contract), self.assertRaisesRegex(ValueError, "WIRE BASELINE"):
                load_wire({"version": 2, "rows": [source]})

    def test_wire_duplicate_identity_cannot_be_hidden_by_different_authority(self) -> None:
        for second in (final_wire_row(), final_wire_row("NumericOperation")):
            with self.subTest(second=second), self.assertRaisesRegex(ValueError, "DUPLICATE"):
                load_wire({"version": 2, "rows": [final_wire_row(), second]})

    def test_top_level_citation_is_inherited_without_overriding_row_citation(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            census = root / "typed.json"
            census.write_text(
                '{"citation":"chelis#729","rows":['
                '{"kind":"wire","id":"inherited"},'
                '{"kind":"wire","id":"specific","citation":"chelis#893"}'
                "]}"
            )
            rows = load_census_rows(root, (Path("typed.json"),))

        self.assertEqual(rows[0]["citation"], "chelis#729")
        self.assertEqual(rows[1]["citation"], "chelis#893")


class Adjudicate(unittest.TestCase):
    def test_open_citation_passes(self) -> None:
        problems = adjudicate(
            [row("chelis#893")],
            {893: IssueRecord(kind=IssueKind.ISSUE, state=IssueState.OPEN)},
        )
        self.assertEqual(problems, [])

    def test_retired_native_disposition_no_longer_supplies_admission(self) -> None:
        self.assertTrue(adjudicate([row(RETIRED_NATIVE_DISPOSITION, census_family="bindings")], {}))

    def test_retired_primary_disposition_fails_closed(self) -> None:
        disposition = (
            "permanent-disposition(C6 initial non-seam complete descriptor set "
            "ratified 2026-08-04)"
        )
        problems = adjudicate([row(disposition)], {})
        self.assertEqual(len(problems), 1)
        self.assertIn("UNRECOGNIZED disposition", problems[0])

    def test_invented_refless_disposition_fails_closed(self) -> None:
        problems = adjudicate([row("permanent-disposition(reviewed and fine)")], {})
        self.assertEqual(len(problems), 1)
        self.assertIn("UNRECOGNIZED disposition", problems[0])

    def test_near_miss_permanent_disposition_fails_closed(self) -> None:
        disposition = RETIRED_NATIVE_DISPOSITION
        problems = adjudicate([row(f"{disposition} copied")], {})
        self.assertEqual(len(problems), 1)
        self.assertIn("UNRECOGNIZED disposition", problems[0])

    def test_permanent_disposition_cannot_move_between_census_families(self) -> None:
        bindings = RETIRED_NATIVE_DISPOSITION
        for family in ("primary", "unregistered"):
            with self.subTest(family=family):
                problems = adjudicate([row(bindings, census_family=family)], {})
                self.assertEqual(len(problems), 1)
                self.assertIn("UNRECOGNIZED disposition", problems[0])

    def test_wire_cannot_admit_legacy_even_with_an_open_issue(self) -> None:
        citations = (
            "chelis#1288",
            "permanent-disposition(C6 dtype-tagged wire schema complete descriptor set ratified 2026-08-04)",
            RETIRED_NATIVE_DISPOSITION,
        )
        for citation in citations:
            with self.subTest(citation=citation):
                source = {**final_wire_row(), "citation": citation, "_census_family": "wire"}
                problems = adjudicate(
                    [source], {1288: IssueRecord(kind=IssueKind.ISSUE, state=IssueState.OPEN)}
                )
                self.assertEqual(len(problems), 1)
                self.assertIn("WIRE BASELINE", problems[0])

    def test_wire_missing_citation_is_not_itself_final_authority(self) -> None:
        malformed = {**final_wire_row(), "_census_family": "wire"}
        del malformed["authority"]
        self.assertIn("WIRE BASELINE", adjudicate([malformed], {})[0])

    def test_closed_citation_fails_with_readjudication_message(self) -> None:
        problems = adjudicate(
            [row("chelis#893")],
            {893: IssueRecord(kind=IssueKind.ISSUE, state=IssueState.CLOSED)},
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("STALE citation", problems[0])
        self.assertIn("re-adjudicated", problems[0])

    def test_one_closed_ref_among_open_still_fails(self) -> None:
        problems = adjudicate(
            [row("chelis#893 and chelis#894")],
            {
                893: IssueRecord(kind=IssueKind.ISSUE, state=IssueState.OPEN),
                894: IssueRecord(kind=IssueKind.ISSUE, state=IssueState.CLOSED),
            },
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("chelis#894", problems[0])

    def test_todo_fails_but_missing_citation_is_final_authority_owned(self) -> None:
        problems = adjudicate([row("TODO"), row("")], {})
        self.assertEqual(len(problems), 1)
        self.assertIn("TODO legacy disposition", problems[0])

    def test_unresolvable_citation_fails(self) -> None:
        problems = adjudicate([row("chelis#999999")], {})
        self.assertEqual(len(problems), 1)
        self.assertIn("UNRESOLVABLE", problems[0])

    def test_open_pull_request_is_not_an_open_issue(self) -> None:
        problems = adjudicate(
            [row("chelis#956")],
            {956: IssueRecord(kind=IssueKind.PULL_REQUEST, state=IssueState.OPEN)},
        )
        self.assertEqual(len(problems), 1)
        self.assertIn("PULL REQUEST", problems[0])
        self.assertIn("not an OPEN issue", problems[0])


class Main(unittest.TestCase):
    def test_current_final_baselines_pass_without_querying_the_tracker(self) -> None:
        output = io.StringIO()
        with patch.object(capacity_census_liveness, "fetch_issue") as fetch, contextlib.redirect_stdout(output):
            self.assertEqual(capacity_census_liveness.main(), 0, output.getvalue())
        fetch.assert_not_called()
        self.assertTrue(output.getvalue().endswith("CAPACITY CENSUS LIVENESS: PASS\n"))

    def test_final_wire_rows_pass_without_network_lookup(self) -> None:
        rows = load_wire({"version": 2, "rows": [final_wire_row()]})
        output = io.StringIO()
        with (
            patch.object(capacity_census_liveness, "load_census_rows", return_value=rows),
            patch.object(capacity_census_liveness, "fetch_issue") as fetch,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(capacity_census_liveness.main(), 0)
        fetch.assert_not_called()
        self.assertTrue(output.getvalue().endswith("CAPACITY CENSUS LIVENESS: PASS\n"))

    def test_malformed_wire_baseline_fails_before_network_lookup(self) -> None:
        output = io.StringIO()
        with (
            patch.object(
                capacity_census_liveness,
                "load_census_rows",
                side_effect=ValueError("WIRE BASELINE must use version 2"),
            ),
            patch.object(capacity_census_liveness, "fetch_issue") as fetch,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(capacity_census_liveness.main(), 1)
        fetch.assert_not_called()
        self.assertIn("WIRE BASELINE", output.getvalue())
        self.assertTrue(output.getvalue().endswith("CAPACITY CENSUS LIVENESS: FAIL\n"))


class FetchIssue(unittest.TestCase):
    def test_rest_issue_payload_is_typed(self) -> None:
        calls: list[list[str]] = []

        def run(args: list[str], **_: object) -> object:
            calls.append(args)
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": '{"state":"open","number":729}'},
            )()

        record = fetch_issue(729, run=run)
        self.assertEqual(
            record, IssueRecord(kind=IssueKind.ISSUE, state=IssueState.OPEN)
        )
        self.assertEqual(
            calls,
            [["gh", "api", "repos/Chelis-Lang/chelis/issues/729"]],
            "the REST issues endpoint exposes pull_request identity unlike gh issue view",
        )

    def test_rest_pull_request_payload_is_rejected_by_adjudication(self) -> None:
        def run(_: list[str], **__: object) -> object:
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": (
                        '{"state":"open","number":956,'
                        '"pull_request":{"url":"https://api.github.test/pulls/956"}}'
                    ),
                },
            )()

        record = fetch_issue(956, run=run)
        self.assertEqual(
            record,
            IssueRecord(kind=IssueKind.PULL_REQUEST, state=IssueState.OPEN),
        )
        problems = adjudicate([row("chelis#956")], {956: record})
        self.assertEqual(len(problems), 1)
        self.assertIn("PULL REQUEST", problems[0])

    def test_closed_and_unresolvable_have_negative_parity(self) -> None:
        def closed(_: list[str], **__: object) -> object:
            return type(
                "Completed",
                (),
                {"returncode": 0, "stdout": '{"state":"closed","number":729}'},
            )()

        def missing(_: list[str], **__: object) -> object:
            return type("Completed", (), {"returncode": 1, "stdout": ""})()

        self.assertEqual(
            fetch_issue(729, run=closed),
            IssueRecord(kind=IssueKind.ISSUE, state=IssueState.CLOSED),
        )
        self.assertIsNone(fetch_issue(999999, run=missing))


if __name__ == "__main__":
    unittest.main()
