"""Adversarial coverage for the heavy-e2e nextest profile split.

Run via: `python3 -m unittest scripts.test_nextest_profile_partition`
from repo root, or `python3 scripts/test_nextest_profile_partition.py`.

PR #126 (refined by #127) split the heavyweight end-to-end suite off the
per-PR integration gate. `.config/nextest.toml` carries three profiles:

  - `default` and `ci` share a `default-filter` that EXCLUDES an
    explicitly-named heavy-e2e set;
  - `nightly` carries the EXACT SAME set as a positive filter, and the
    `Heavy E2E` workflow runs `cargo nextest run --profile nightly`.

The invariant this file locks: **every non-ignored test is on the
per-PR gate XOR the nightly gate -- never neither, never both.** A test
that falls into NEITHER profile silently stopped running; a test in
BOTH wastes the per-PR gate budget the split exists to protect. Nothing
in the merged change tested this invariant -- it was asserted in prose
in `.config/nextest.toml` and `heavy-e2e.yml` but never executed.

Two tiers of check:

  - `FilterTextTests` is a fast, no-compile lock on the *text* of the
    three filter blocks: the `default` and `ci` exclusion blocks must be
    byte-identical, and the `nightly` positive filter must equal the
    negated inner set of the exclusion block. A hand-edit that drifts
    one block trips this immediately.
  - `ProfilePartitionTests` is the real set-math oracle: it runs
    `cargo nextest list` for the `ci` and `nightly` profiles plus the
    full unfiltered list and asserts the partition. It is skipped when
    `cargo`/`cargo nextest` is unavailable, and is `slow`-tolerant
    (listing compiles test binaries on a cold tree).
"""

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NEXTEST_TOML = REPO_ROOT / ".config" / "nextest.toml"


def _filter_blocks() -> list[str]:
    """Return the raw bodies of every `default-filter = '''...'''` block
    in `.config/nextest.toml`, in file order (default, ci, nightly)."""
    text = NEXTEST_TOML.read_text()
    return re.findall(r"default-filter = '''(.*?)'''", text, re.DOTALL)


def _norm(s: str) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    return re.sub(r"\s+", " ", s).strip()


class FilterTextTests(unittest.TestCase):
    """No-compile lock on the three filter blocks' text."""

    def test_nextest_toml_exists(self):
        self.assertTrue(NEXTEST_TOML.is_file(), f"missing {NEXTEST_TOML}")

    def test_exactly_three_filter_blocks(self):
        blocks = _filter_blocks()
        self.assertEqual(
            len(blocks),
            3,
            "expected exactly three default-filter blocks "
            "(default, ci, nightly)",
        )

    def test_default_and_ci_exclusion_blocks_are_byte_identical(self):
        # The `default` and `ci` exclusion blocks are hand-duplicated.
        # Any drift between them is a real bug: the local dev gate and
        # the CI gate would then run different test sets.
        default_block, ci_block, _nightly = _filter_blocks()
        self.assertEqual(
            default_block,
            ci_block,
            "the `default` and `ci` default-filter blocks have drifted; "
            "they are hand-duplicated and must stay byte-identical",
        )

    def test_nightly_filter_is_the_negated_exclusion_set(self):
        # `default`/`ci` exclude `not ( <set> )`; `nightly` must select
        # exactly `<set>`. If these drift, a test lands in neither
        # profile (dropped coverage) or both (double-run).
        default_block, _ci, nightly_block = _filter_blocks()
        inner = re.search(r"not\s*\((.*)\)\s*$", default_block, re.DOTALL)
        self.assertIsNotNone(
            inner,
            "the `default` filter is not in the expected "
            "`not ( <set> )` shape",
        )
        self.assertEqual(
            _norm(inner.group(1)),
            _norm(nightly_block),
            "the `nightly` positive filter does not equal the negated "
            "inner set of the `default`/`ci` exclusion filter; a test "
            "now falls into neither profile or both",
        )


def _have_nextest() -> bool:
    if shutil.which("cargo") is None:
        return False
    try:
        out = subprocess.run(
            ["cargo", "nextest", "--version"],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0


def _list_profile(profile: str | None) -> dict[str, tuple[str, bool]]:
    """Run `cargo nextest list` for `profile` (or the implicit default
    when `None`) and return `binary_id::test -> (filter_status,
    ignored)`.

    nextest's `list --message-format json` annotates each testcase with
    `filter-match.status` ("matches" / "mismatch") relative to the
    profile's `default-filter`, and an `ignored` flag.
    """
    cmd = ["cargo", "nextest", "list", "--workspace", "--message-format", "json"]
    if profile is not None:
        cmd += ["--profile", profile]
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=900
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(cmd)}` failed (exit {result.returncode}): "
            f"{result.stderr[-2000:]}"
        )
    data = json.loads(result.stdout)
    out: dict[str, tuple[str, bool]] = {}
    for binary_id, suite in data.get("rust-suites", {}).items():
        for test_name, info in suite.get("testcases", {}).items():
            key = f"{binary_id}::{test_name}"
            status = info.get("filter-match", {}).get("status")
            out[key] = (status, bool(info.get("ignored")))
    return out


@unittest.skipUnless(
    _have_nextest(), "cargo nextest unavailable; skipping set-math oracle"
)
class ProfilePartitionTests(unittest.TestCase):
    """The real oracle: `cargo nextest list` set math across profiles.

    The invariant: every non-ignored test is in exactly one of `ci` /
    `nightly`. `#[ignore]`-d tests (HIP/Metal manual gates and friends)
    are intentionally on neither automated profile and are excluded
    from the partition.
    """

    @classmethod
    def setUpClass(cls):
        cls.ci = _list_profile("ci")
        cls.nightly = _list_profile("nightly")
        cls.full = _list_profile(None)

    def _sets(self):
        ci_matches = {k for k, (s, _) in self.ci.items() if s == "matches"}
        nightly_matches = {
            k for k, (s, _) in self.nightly.items() if s == "matches"
        }
        # The true universe: every test that appears under any profile's
        # listing. `nightly`'s json only enumerates binaries its filter
        # can match, so union all three to get the full picture.
        universe = set(self.ci) | set(self.nightly) | set(self.full)
        ignored = {
            k
            for src in (self.ci, self.nightly, self.full)
            for k, (_s, ign) in src.items()
            if ign
        }
        non_ignored = universe - ignored
        return ci_matches, nightly_matches, non_ignored

    def test_ci_and_nightly_are_disjoint(self):
        ci_matches, nightly_matches, _ = self._sets()
        overlap = ci_matches & nightly_matches
        self.assertEqual(
            overlap,
            set(),
            f"{len(overlap)} test(s) are on BOTH the per-PR `ci` gate and "
            f"the `nightly` gate (double-run, wastes the per-PR budget): "
            f"{sorted(overlap)[:20]}",
        )

    def test_no_non_ignored_test_falls_into_neither_profile(self):
        ci_matches, nightly_matches, non_ignored = self._sets()
        gap = non_ignored - ci_matches - nightly_matches
        self.assertEqual(
            gap,
            set(),
            f"{len(gap)} non-ignored test(s) are on NEITHER the per-PR `ci` "
            f"gate nor the `nightly` gate -- they silently stopped running "
            f"(dropped coverage): {sorted(gap)[:20]}",
        )

    def test_ci_union_nightly_covers_every_non_ignored_test(self):
        # Belt-and-braces statement of the partition: the union of the
        # two automated profiles is exactly the non-ignored universe.
        ci_matches, nightly_matches, non_ignored = self._sets()
        covered = (ci_matches | nightly_matches) & non_ignored
        self.assertEqual(
            covered,
            non_ignored,
            "the union of the `ci` and `nightly` profiles does not cover "
            "every non-ignored test",
        )

    def test_nightly_set_is_nonempty(self):
        # A `nightly` profile that matched nothing would mean the split
        # silently dropped the entire heavy-e2e suite.
        _ci, nightly_matches, _ = self._sets()
        self.assertGreater(
            len(nightly_matches),
            0,
            "the `nightly` profile matched zero tests; the heavy-e2e "
            "suite is not running anywhere",
        )

    def test_no_nightly_test_is_ignored(self):
        # A heavy-e2e test that is also `#[ignore]`-d would never run --
        # not on the per-PR gate (excluded) and not nightly (ignored).
        nightly_ignored = {
            k
            for k, (s, ign) in self.nightly.items()
            if s == "matches" and ign
        }
        self.assertEqual(
            nightly_ignored,
            set(),
            f"{len(nightly_ignored)} `nightly`-selected test(s) are also "
            f"`#[ignore]`-d, so they never run: {sorted(nightly_ignored)}",
        )


if __name__ == "__main__":
    unittest.main()
