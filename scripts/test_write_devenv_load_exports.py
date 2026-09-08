#!/usr/bin/env python3
"""Tests for atomic publication of Devenv's shell export payload."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_devenv_load_exports import publish


class LoadExportsPublicationTests(unittest.TestCase):
    def test_publish_preserves_the_exact_payload_and_executable_mode(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            destination = Path(raw_directory) / "state" / "load-exports"
            payload = "export CHELIS_PROBE='alpha beta'\n"

            publish(destination, payload)

            self.assertEqual(destination.read_text(encoding="utf-8"), payload)
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o700)

    def test_concurrent_publishers_never_expose_a_partial_payload(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            destination = Path(raw_directory) / "load-exports"
            payloads = [
                f"export CHELIS_WRITER={index}\n# {'x' * (4096 + index)}\n"
                for index in range(24)
            ]
            allowed = {payload.encode() for payload in payloads}
            observations: list[bytes] = []
            stop = threading.Event()

            def read_while_publishing() -> None:
                while not stop.is_set():
                    try:
                        observations.append(destination.read_bytes())
                    except FileNotFoundError:
                        pass

            reader = threading.Thread(target=read_while_publishing)
            reader.start()
            try:
                with ThreadPoolExecutor(max_workers=12) as pool:
                    list(pool.map(lambda payload: publish(destination, payload), payloads))
            finally:
                stop.set()
                reader.join()

            self.assertTrue(observations)
            self.assertTrue(all(observation in allowed for observation in observations))
            self.assertIn(destination.read_bytes(), allowed)
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o700)
            self.assertEqual(
                list(Path(raw_directory).glob(".load-exports.*")),
                [],
            )

    @unittest.skipUnless(os.name == "posix", "directory fsync is POSIX-only")
    def test_replacing_an_existing_payload_remains_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            destination = Path(raw_directory) / "load-exports"
            publish(destination, "export FIRST=1\n")
            publish(destination, "export SECOND=2\n")
            self.assertEqual(destination.read_text(), "export SECOND=2\n")


if __name__ == "__main__":
    unittest.main()
