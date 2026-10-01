"""Manual tools take sibling checkouts as arguments, never as machine defaults."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import bench_phase_j, build_wrapped_spans_sidecar

BENCH_ARGS = [
    "--binary",
    "chelis",
    "--label",
    "post",
    "--chelis-repo",
    "chelis",
    "--out",
    "bench.json",
]


class BenchPhaseJTests(unittest.TestCase):
    def test_coral_checkout_is_required(self):
        with (
            patch.object(bench_phase_j, "run_benches") as run,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            bench_phase_j.main(BENCH_ARGS)
        run.assert_not_called()

    def test_coral_checkout_reaches_the_benchmarks(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "bench.json"
            argv = BENCH_ARGS[:-1] + [str(out), "--coral-dir", "/srv/coral"]
            with (
                patch.object(bench_phase_j, "run_benches", return_value={}) as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(bench_phase_j.main(argv), 0)
        self.assertEqual(run.call_args.kwargs["coral_dir"], Path("/srv/coral"))


class WrappedSpansSidecarTests(unittest.TestCase):
    def test_octant_checkout_is_required(self):
        with (
            patch.object(build_wrapped_spans_sidecar, "build_sidecar") as build,
            patch.object(build_wrapped_spans_sidecar.sys, "argv", ["sidecar"]),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            build_wrapped_spans_sidecar.main()
        build.assert_not_called()

    def test_octant_paths_derive_from_the_argument(self):
        with tempfile.TemporaryDirectory() as root:
            argv = ["sidecar", "--octant-repo", root]
            with (
                patch.object(build_wrapped_spans_sidecar.sys, "argv", argv),
                self.assertRaises(SystemExit) as error,
            ):
                build_wrapped_spans_sidecar.main()
            expected = Path(root).resolve() / "target/release/octant"
            self.assertIn(str(expected), str(error.exception))


if __name__ == "__main__":
    unittest.main()
