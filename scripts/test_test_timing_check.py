"""Unit tests for `test_timing_check.py`.

Run via: `python3 -m unittest scripts.test_test_timing_check` from repo
root, or `python3 scripts/test_test_timing_check.py`.

Covers the four cases the brief pins:
  - an over-threshold (regressed) test is flagged;
  - all-under-budget produces an empty report and exit 0;
  - a NEW test over the absolute ceiling is flagged;
  - malformed / missing JUnit XML produces a clear error, not a
    silent pass.
"""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "test_timing_check", here / "test_timing_check.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tc = _load_module()


def _junit(cases: list[tuple[str, str, float]]) -> str:
    """Build a minimal nextest-shaped JUnit XML string. Each case is
    `(classname, name, time_seconds)`."""
    body = []
    for classname, name, seconds in cases:
        body.append(
            f'    <testcase classname="{classname}" name="{name}" '
            f'time="{seconds}"></testcase>'
        )
    inner = "\n".join(body)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<testsuites>\n'
        '  <testsuite name="nextest-run" tests="1">\n'
        f"{inner}\n"
        "  </testsuite>\n"
        "</testsuites>\n"
    )


class ParseJunitTests(unittest.TestCase):
    def test_parses_composite_key_and_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "junit.xml"
            p.write_text(_junit([("bin_a", "test_one", 1.5)]))
            timings = tc.parse_junit(p)
            self.assertEqual(timings, {"bin_a::test_one": 1.5})

    def test_missing_file_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.xml"
            with self.assertRaises(tc.TimingError) as cm:
                tc.parse_junit(missing)
            self.assertIn("not found", str(cm.exception))

    def test_malformed_xml_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "junit.xml"
            p.write_text("<testsuites><testcase not closed")
            with self.assertRaises(tc.TimingError) as cm:
                tc.parse_junit(p)
            self.assertIn("malformed", str(cm.exception))

    def test_empty_junit_raises_clear_error(self):
        # A JUnit file with zero <testcase> elements is suspicious
        # (the run produced no per-test timing); flag it loudly rather
        # than silently reporting "0 tests, all OK".
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "junit.xml"
            p.write_text(_junit([]))
            with self.assertRaises(tc.TimingError) as cm:
                tc.parse_junit(p)
            self.assertIn("no <testcase>", str(cm.exception))

    def test_non_numeric_time_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "junit.xml"
            p.write_text(_junit([("bin_a", "test_one", "fast")]))  # type: ignore[list-item]
            with self.assertRaises(tc.TimingError) as cm:
                tc.parse_junit(p)
            self.assertIn("non-numeric time", str(cm.exception))


class EvaluateTests(unittest.TestCase):
    def test_all_under_budget_is_empty(self):
        timings = {"bin_a::test_one": 1.0, "bin_a::test_two": 2.0}
        baseline = {"bin_a::test_one": 1.0, "bin_a::test_two": 2.0}
        flags = tc.evaluate(timings, baseline, tolerance=2.0, absolute_ceiling=30.0)
        self.assertEqual(flags, [])

    def test_regressed_test_is_flagged(self):
        # baseline 1.0s, tolerance 2.0x -> budget 2.0s; observed 5.0s.
        timings = {"bin_a::slow": 5.0}
        baseline = {"bin_a::slow": 1.0}
        flags = tc.evaluate(timings, baseline, tolerance=2.0, absolute_ceiling=30.0)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].kind, tc.Flag.REGRESSED)
        self.assertEqual(flags[0].key, "bin_a::slow")

    def test_regressed_exactly_at_budget_is_not_flagged(self):
        # observed == tolerance x baseline is within budget (strict >).
        timings = {"bin_a::edge": 2.0}
        baseline = {"bin_a::edge": 1.0}
        flags = tc.evaluate(timings, baseline, tolerance=2.0, absolute_ceiling=30.0)
        self.assertEqual(flags, [])

    def test_new_test_over_absolute_ceiling_is_flagged(self):
        # Not in baseline + over the 30s ceiling -> flagged as new.
        timings = {"bin_a::brand_new": 45.0}
        baseline = {"bin_a::existing": 1.0}
        flags = tc.evaluate(timings, baseline, tolerance=2.0, absolute_ceiling=30.0)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].kind, tc.Flag.NEW_OVER_CEILING)
        self.assertEqual(flags[0].key, "bin_a::brand_new")

    def test_new_test_under_absolute_ceiling_is_not_flagged(self):
        # A fast new test is fine; it gets absorbed at the next
        # explicit baseline regeneration.
        timings = {"bin_a::brand_new": 3.0}
        baseline = {"bin_a::existing": 1.0}
        flags = tc.evaluate(timings, baseline, tolerance=2.0, absolute_ceiling=30.0)
        self.assertEqual(flags, [])

    def test_flags_sorted_slowest_first(self):
        timings = {
            "bin_a::a": 10.0,
            "bin_a::b": 50.0,
            "bin_a::c": 25.0,
        }
        baseline = {"bin_a::a": 1.0, "bin_a::b": 1.0, "bin_a::c": 1.0}
        flags = tc.evaluate(timings, baseline, tolerance=2.0, absolute_ceiling=30.0)
        observed = [f.observed for f in flags]
        self.assertEqual(observed, sorted(observed, reverse=True))


class ConfigTests(unittest.TestCase):
    def _write_config(self, tmp: str, payload) -> Path:
        p = Path(tmp) / "config.json"
        p.write_text(payload if isinstance(payload, str) else json.dumps(payload))
        return p

    def test_loads_valid_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_config(tmp, {"tolerance": 2.0, "absolute_ceiling": 30.0})
            tolerance, ceiling = tc.load_config(p)
            self.assertEqual((tolerance, ceiling), (2.0, 30.0))

    def test_missing_config_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(tc.TimingError):
                tc.load_config(Path(tmp) / "nope.json")

    def test_malformed_config_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_config(tmp, "{not json")
            with self.assertRaises(tc.TimingError):
                tc.load_config(p)

    def test_config_missing_key_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_config(tmp, {"tolerance": 2.0})
            with self.assertRaises(tc.TimingError):
                tc.load_config(p)

    def test_tolerance_below_one_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write_config(tmp, {"tolerance": 0.5, "absolute_ceiling": 30.0})
            with self.assertRaises(tc.TimingError):
                tc.load_config(p)

    def test_committed_config_is_valid(self):
        # The committed scripts/test_timing_config.json must load.
        tolerance, ceiling = tc.load_config()
        self.assertGreaterEqual(tolerance, 1.0)
        self.assertGreater(ceiling, 0.0)


class BaselineTests(unittest.TestCase):
    def test_loads_valid_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "baseline.json"
            p.write_text(json.dumps({"bin_a::t": 1.5}))
            self.assertEqual(tc.load_baseline(p), {"bin_a::t": 1.5})

    def test_missing_baseline_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(tc.TimingError):
                tc.load_baseline(Path(tmp) / "nope.json")

    def test_non_object_baseline_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "baseline.json"
            p.write_text(json.dumps(["not", "an", "object"]))
            with self.assertRaises(tc.TimingError):
                tc.load_baseline(p)

    def test_committed_baseline_is_valid(self):
        # The committed scripts/test_timing_baseline.json must load and
        # be non-empty (a real current-state baseline).
        baseline = tc.load_baseline()
        self.assertGreater(len(baseline), 0, "committed baseline is empty")

    def test_update_baseline_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            junit = Path(tmp) / "junit.xml"
            junit.write_text(_junit([("bin_a", "t1", 1.0), ("bin_a", "t2", 2.0)]))
            baseline = Path(tmp) / "baseline.json"
            rc = tc.update_baseline(junit, baseline)
            self.assertEqual(rc, 0)
            written = tc.load_baseline(baseline)
            self.assertEqual(written, {"bin_a::t1": 1.0, "bin_a::t2": 2.0})


class MainExitCodeTests(unittest.TestCase):
    def test_all_under_budget_exits_zero_with_empty_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            junit = Path(tmp) / "junit.xml"
            junit.write_text(_junit([("bin_a", "t1", 1.0)]))
            baseline = Path(tmp) / "baseline.json"
            baseline.write_text(json.dumps({"bin_a::t1": 1.0}))
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({"tolerance": 2.0, "absolute_ceiling": 30.0}))
            saved = (tc.CONFIG_PATH, tc.BASELINE_PATH)
            tc.CONFIG_PATH, tc.BASELINE_PATH = config, baseline
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = tc.main(["--junit", str(junit)])
            finally:
                tc.CONFIG_PATH, tc.BASELINE_PATH = saved
            self.assertEqual(rc, 0)
            self.assertIn("OK", buf.getvalue())

    def test_over_budget_exits_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            junit = Path(tmp) / "junit.xml"
            junit.write_text(_junit([("bin_a", "slow", 99.0)]))
            baseline = Path(tmp) / "baseline.json"
            baseline.write_text(json.dumps({"bin_a::slow": 1.0}))
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({"tolerance": 2.0, "absolute_ceiling": 30.0}))
            saved = (tc.CONFIG_PATH, tc.BASELINE_PATH)
            tc.CONFIG_PATH, tc.BASELINE_PATH = config, baseline
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = tc.main(["--junit", str(junit)])
            finally:
                tc.CONFIG_PATH, tc.BASELINE_PATH = saved
            self.assertEqual(rc, 1)
            self.assertIn("over budget", buf.getvalue())

    def test_missing_junit_exits_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = tc.main(["--junit", str(Path(tmp) / "nope.xml")])
            self.assertEqual(rc, 2)

    def test_malformed_junit_exits_two_not_silent_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            junit = Path(tmp) / "junit.xml"
            junit.write_text("<broken")
            rc = tc.main(["--junit", str(junit)])
            self.assertEqual(rc, 2, "malformed XML must not be a silent pass")


if __name__ == "__main__":
    unittest.main()
