#!/usr/bin/env python3
"""Contract tests for the CI JUnit shard merger and timing report."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "ci_test_telemetry", here / "ci_test_telemetry.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


telemetry = _load_module()


def _load_timing_check_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "test_timing_check_impl", here / "test_timing_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


timing_check = _load_timing_check_module()


def _junit(cases: list[tuple[str, str, float]]) -> str:
    rows = "\n".join(
        f'<testcase classname="{binary}" name="{name}" time="{seconds}" />'
        for binary, name, seconds in cases
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<testsuites><testsuite name="nextest-run">'
        f"{rows}"
        "</testsuite></testsuites>\n"
    )


class ShardParsingTests(unittest.TestCase):
    def test_missing_malformed_and_empty_reports_fail_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed = root / "malformed.xml"
            malformed.write_text("<broken", encoding="utf-8")
            empty = root / "empty.xml"
            empty.write_text(_junit([]), encoding="utf-8")
            for path, fragment in (
                (root / "missing.xml", "not found"),
                (malformed, "malformed"),
                (empty, "no testcases"),
            ):
                with self.subTest(path=path.name):
                    with self.assertRaisesRegex(telemetry.TelemetryError, fragment):
                        telemetry.parse_shard("workspace-1", path)

    def test_duplicate_test_identity_inside_one_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "junit.xml"
            path.write_text(
                _junit([("bin", "same", 1.0), ("bin", "same", 2.0)]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(telemetry.TelemetryError, "duplicate"):
                telemetry.parse_shard("workspace-1", path)

    def test_invalid_times_fail_closed(self) -> None:
        for seconds, fragment in (
            ("not-a-number", "nonnumeric time"),
            ("NaN", "invalid time"),
            ("Infinity", "invalid time"),
            ("+Infinity", "invalid time"),
            ("-Infinity", "invalid time"),
            ("-0.001", "invalid time"),
        ):
            with self.subTest(seconds=seconds), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "junit.xml"
                path.write_text(
                    _junit(
                        [("bin", "test", seconds)]  # type: ignore[list-item]
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(telemetry.TelemetryError, fragment):
                    telemetry.parse_shard("workspace-1", path)


class MergeContractTests(unittest.TestCase):
    def _shards(self, root: Path):
        one = root / "one.xml"
        two = root / "two.xml"
        one.write_text(_junit([("bin-a", "fast", 0.5)]), encoding="utf-8")
        two.write_text(_junit([("bin-b", "slow", 30.48)]), encoding="utf-8")
        return (
            telemetry.parse_shard("workspace-1", one),
            telemetry.parse_shard("workspace-2", two),
        )

    def test_disjoint_shards_merge_to_the_exact_union(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = telemetry.build_report(self._shards(Path(tmp)), require_disjoint=True)
            self.assertEqual(report.observation_count, 2)
            self.assertEqual(report.unique_test_count, 2)
            self.assertEqual(report.overlaps, {})
            self.assertEqual(report.slowest[0].test_id, "bin-b::slow")

    def test_partition_overlap_names_both_owners_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = root / "one.xml"
            two = root / "two.xml"
            xml = _junit([("bin", "same", 1.0)])
            one.write_text(xml, encoding="utf-8")
            two.write_text(xml, encoding="utf-8")
            shards = (
                telemetry.parse_shard("workspace-1", one),
                telemetry.parse_shard("workspace-2", two),
            )
            with self.assertRaisesRegex(
                telemetry.TelemetryError, "workspace-1.*workspace-2"
            ):
                telemetry.build_report(shards, require_disjoint=True)

    def test_cross_lane_report_retains_expected_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = root / "one.xml"
            two = root / "two.xml"
            one.write_text(_junit([("bin", "same", 1.0)]), encoding="utf-8")
            two.write_text(_junit([("bin", "same", 2.0)]), encoding="utf-8")
            shards = (
                telemetry.parse_shard("workspace", one),
                telemetry.parse_shard("dtype", two),
            )
            report = telemetry.build_report(shards, require_disjoint=False)
            self.assertEqual(report.observation_count, 2)
            self.assertEqual(report.unique_test_count, 1)
            self.assertEqual(report.overlaps, {"bin::same": ("workspace", "dtype")})
            self.assertEqual(report.slowest[0].seconds, 2.0)


class OutputContractTests(unittest.TestCase):
    def test_merged_junit_is_canonical_across_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def report_for(prefix: str, *, reversed_order: bool):
                one = root / f"{prefix}-one.xml"
                two = root / f"{prefix}-two.xml"
                cases_one = [("bin-b", "two", 2.0), ("bin-a", "one", 1.0)]
                cases_two = [("bin-d", "four", 4.0), ("bin-c", "three", 3.0)]
                if reversed_order:
                    cases_one.reverse()
                    cases_two.reverse()
                one.write_text(_junit(cases_one), encoding="utf-8")
                two.write_text(_junit(cases_two), encoding="utf-8")
                shards = (
                    telemetry.parse_shard("workspace-1", one),
                    telemetry.parse_shard("workspace-2", two),
                )
                if reversed_order:
                    shards = tuple(reversed(shards))
                return telemetry.build_report(shards, require_disjoint=True)

            first = root / "first.xml"
            second = root / "second.xml"
            first_json = root / "first.json"
            second_json = root / "second.json"
            first_report = report_for("first", reversed_order=False)
            second_report = report_for("second", reversed_order=True)
            telemetry.write_merged_junit(first_report, first)
            telemetry.write_merged_junit(second_report, second)
            telemetry.write_json_report(first_report, first_json)
            telemetry.write_json_report(second_report, second_json)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(first_json.read_bytes(), second_json.read_bytes())

    def test_json_overlap_owners_are_canonical_across_shard_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = root / "one.xml"
            two = root / "two.xml"
            one.write_text(_junit([("bin", "same", 1.0)]), encoding="utf-8")
            two.write_text(_junit([("bin", "same", 2.0)]), encoding="utf-8")
            shard_one = telemetry.parse_shard("workspace", one)
            shard_two = telemetry.parse_shard("dtype", two)
            forward = telemetry.build_report(
                (shard_one, shard_two), require_disjoint=False
            )
            reverse = telemetry.build_report(
                (shard_two, shard_one), require_disjoint=False
            )
            forward_json = root / "forward.json"
            reverse_json = root / "reverse.json"
            telemetry.write_json_report(forward, forward_json)
            telemetry.write_json_report(reverse, reverse_json)

            self.assertEqual(forward_json.read_bytes(), reverse_json.read_bytes())

    def test_outputs_are_machine_readable_and_timing_check_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = MergeContractTests()._shards(root)
            report = telemetry.build_report(shards, require_disjoint=True)
            merged = root / "merged.xml"
            machine = root / "report.json"
            summary = root / "summary.md"
            telemetry.write_merged_junit(report, merged)
            telemetry.write_json_report(report, machine)
            telemetry.write_markdown_summary(report, summary)

            parsed = telemetry.parse_shard("merged", merged)
            self.assertEqual(parsed.timings, {"bin-a::fast": 0.5, "bin-b::slow": 30.48})
            payload = json.loads(machine.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["observation_count"], 2)
            rendered = summary.read_text(encoding="utf-8")
            self.assertIn("30.48s", rendered)
            self.assertIn("bin-b::slow", rendered)

    def test_merged_junit_preserves_just_over_ceiling_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.xml"
            merged = root / "merged.xml"
            just_over_ceiling = 30.00000001
            source.write_text(
                _junit([("bin", "boundary", just_over_ceiling)]),
                encoding="utf-8",
            )
            report = telemetry.build_report(
                (telemetry.parse_shard("workspace", source),),
                require_disjoint=True,
            )

            telemetry.write_merged_junit(report, merged)

            timings = timing_check.parse_junit(merged)
            self.assertEqual(timings["bin::boundary"], just_over_ceiling)
            flags = timing_check.evaluate(
                timings,
                {},
                tolerance=2.0,
                absolute_ceiling=30.0,
                min_regression_delta=0.05,
            )
            self.assertEqual(len(flags), 1)
            self.assertEqual(flags[0].kind, timing_check.Flag.OVER_CEILING)

    def test_every_durable_output_preserves_binary64_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.xml"
            merged = root / "merged.xml"
            machine = root / "report.json"
            summary = root / "summary.md"
            values = {
                "subnormal": 5e-324,
                "boundary": 30.0,
                "adjacent-above": 30.000000000000004,
                "maximum": float.fromhex("0x1.fffffffffffffp+1023"),
            }
            source.write_text(
                _junit(
                    [("bin", name, seconds) for name, seconds in values.items()]
                ),
                encoding="utf-8",
            )
            report = telemetry.build_report(
                (telemetry.parse_shard("workspace", source),),
                require_disjoint=True,
            )

            telemetry.write_merged_junit(report, merged)
            telemetry.write_json_report(report, machine)
            telemetry.write_markdown_summary(report, summary)

            self.assertEqual(
                timing_check.parse_junit(merged),
                {f"bin::{name}": seconds for name, seconds in values.items()},
            )
            payload = json.loads(machine.read_text(encoding="utf-8"))
            json_values = {
                item["test_id"].removeprefix("bin::"): item["seconds"]
                for item in payload["slowest"]
            }
            self.assertEqual(json_values, values)
            rendered = summary.read_text(encoding="utf-8")
            for seconds in values.values():
                self.assertIn(f"| {seconds!r}s |", rendered)

    def test_cli_requires_unique_labels_and_writes_all_requested_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = root / "one.xml"
            two = root / "two.xml"
            one.write_text(_junit([("a", "one", 1.0)]), encoding="utf-8")
            two.write_text(_junit([("b", "two", 2.0)]), encoding="utf-8")
            merged = root / "merged.xml"
            machine = root / "report.json"
            summary = root / "summary.md"
            rc = telemetry.main(
                [
                    "--shard",
                    f"workspace-1={one}",
                    "--shard",
                    f"workspace-2={two}",
                    "--require-disjoint",
                    "--merged-junit",
                    str(merged),
                    "--json",
                    str(machine),
                    "--summary",
                    str(summary),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(merged.is_file())
            self.assertTrue(machine.is_file())
            self.assertTrue(summary.is_file())

            duplicate_rc = telemetry.main(
                [
                    "--shard",
                    f"same={one}",
                    "--shard",
                    f"same={two}",
                ]
            )
            self.assertEqual(duplicate_rc, 2)


if __name__ == "__main__":
    unittest.main()
