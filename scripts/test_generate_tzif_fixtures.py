"""Tests for scripts/generate_tzif_fixtures.py (chelis#2862)."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
import zoneinfo

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_tzif_fixtures as fixtures  # noqa: E402


def offset_at(path: Path, name: str, when: datetime.datetime) -> datetime.timedelta | None:
    with path.open("rb") as handle:
        zone = zoneinfo.ZoneInfo.from_file(handle, key=name)
    return when.astimezone(zone).utcoffset()


class CheckedInFixtures(unittest.TestCase):
    def test_fixtures_match_the_generator_and_the_manifest(self) -> None:
        self.assertEqual(fixtures.check(), [])

    def test_manifest_records_the_pinned_release(self) -> None:
        recorded = json.loads(fixtures.MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(recorded["tzdata_package"], fixtures.TZDATA_PACKAGE_VERSION)
        self.assertEqual(recorded["iana_release"], fixtures.IANA_RELEASE)
        self.assertEqual(recorded["zones"], list(fixtures.ZONES))

    def test_real_zones_load_with_zoneinfo(self) -> None:
        july = datetime.datetime(2026, 7, 1, tzinfo=datetime.timezone.utc)
        expected = {
            "America/New_York": -4 * 3600,
            "Europe/London": 3600,
            "Australia/Lord_Howe": 37800,
            "Pacific/Kiritimati": 14 * 3600,
            "Asia/Kolkata": 19800,
            "America/Sao_Paulo": -3 * 3600,
            "UTC": 0,
        }
        for name in fixtures.ZONES:
            with self.subTest(zone=name):
                path = fixtures.FIXTURE_DIR / fixtures.zone_path(name)
                self.assertEqual(offset_at(path, name, july), datetime.timedelta(seconds=expected[name]))


class CheckDetectsDrift(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = Path(tempfile.mkdtemp(prefix="tzif-fixtures-"))
        self.copy = self.scratch / "tzif"
        shutil.copytree(fixtures.FIXTURE_DIR, self.copy)

    def tearDown(self) -> None:
        shutil.rmtree(self.scratch)

    def flip_last_byte(self, relative: str) -> None:
        path = self.copy / relative
        data = bytearray(path.read_bytes())
        data[-2] ^= 1
        path.write_bytes(bytes(data))

    def test_a_copy_checks_clean(self) -> None:
        self.assertEqual(fixtures.check(self.copy), [])

    def test_a_changed_zone_breaks_its_hash(self) -> None:
        self.flip_last_byte(fixtures.zone_path("Asia/Kolkata"))
        problems = fixtures.check(self.copy)
        self.assertIn("zones/Asia/Kolkata.tzif does not match its recorded SHA-256", problems)

    def test_a_changed_synthetic_file_differs_from_its_generator(self) -> None:
        self.flip_last_byte("synthetic/empty_footer.tzif")
        self.assertIn("synthetic/empty_footer.tzif differs from its generator", fixtures.check(self.copy))

    def test_a_missing_file_is_reported(self) -> None:
        (self.copy / "synthetic/leap_seconds.tzif").unlink()
        problems = fixtures.check(self.copy)
        self.assertTrue(any("fixture directory holds" in problem for problem in problems), problems)

    def test_a_stray_file_is_reported(self) -> None:
        (self.copy / "synthetic/stray.tzif").write_bytes(b"TZif")
        problems = fixtures.check(self.copy)
        self.assertTrue(any("fixture directory holds" in problem for problem in problems), problems)


class Writer(unittest.TestCase):
    def test_headers_carry_the_counts(self) -> None:
        data = fixtures.tzif(b"2", [10, 20], [0, 1], [(0, 0, 0), (3600, 1, 4)], b"STD\x00DST\x00", b"STD-1")
        self.assertEqual(data[:5], b"TZif2")
        second = 44 + 7
        self.assertEqual(data[second:second + 5], b"TZif2")
        self.assertEqual(struct.unpack(">6L", data[second + 20:second + 44]), (0, 0, 0, 2, 2, 8))
        self.assertTrue(data.endswith(b"\nSTD-1\n"))

    def test_a_version_1_file_has_no_second_header(self) -> None:
        data = fixtures.tzif(b"\x00", [], [], [], b"", b"")
        self.assertEqual(len(data), 44 + 7)
        self.assertNotIn(b"TZif", data[4:])

    def test_with_version_changes_only_the_two_version_bytes(self) -> None:
        valid = fixtures.two_type(b"2", [1_000_000_000], b"STD-1")
        changed = fixtures.with_version(valid, b"5")
        differing = [k for k, (a, b) in enumerate(zip(valid, changed)) if a != b]
        self.assertEqual(differing, [4, 55])

    def test_the_valid_synthetic_zones_load_with_zoneinfo(self) -> None:
        july = datetime.datetime(2026, 7, 1, tzinfo=datetime.timezone.utc)
        for name, hours in (("jerusalem_v3", 3), ("all_year_dst", -4)):
            with self.subTest(zone=name):
                path = fixtures.FIXTURE_DIR / f"synthetic/{name}.tzif"
                self.assertEqual(offset_at(path, name, july), datetime.timedelta(hours=hours))

    def test_a_bad_magic_does_not_load_with_zoneinfo(self) -> None:
        with (fixtures.FIXTURE_DIR / "synthetic/bad_magic.tzif").open("rb") as handle:
            with self.assertRaises(ValueError):
                zoneinfo.ZoneInfo.from_file(handle, key="bad_magic")


if __name__ == "__main__":
    unittest.main()
