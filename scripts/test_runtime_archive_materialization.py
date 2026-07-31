"""Prevent parallel tests from sharing a runtime-archive temporary path."""

from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
CRATES = REPO / "crates"
UNSAFE_TEMP = 'with_extension("a.tmp")'
ARCHIVE_DISCOVERY = 'name.starts_with("libchelis_runtime-")'
ATOMIC_TEMP = "NEXT_TEMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed)"


class RuntimeArchiveMaterializationTests(unittest.TestCase):
    def test_every_archive_materializer_uses_an_invocation_unique_temp(self) -> None:
        helpers = []
        for source in CRATES.rglob("*.rs"):
            text = source.read_text(encoding="utf-8")
            self.assertNotIn(UNSAFE_TEMP, text, source.relative_to(REPO))
            if ARCHIVE_DISCOVERY in text and "fs::rename(&tmp, canonical)" in text:
                helpers.append(source)
                self.assertIn(ATOMIC_TEMP, text, source.relative_to(REPO))

        self.assertGreater(len(helpers), 0, "no runtime materialization helpers found")


if __name__ == "__main__":
    unittest.main()
