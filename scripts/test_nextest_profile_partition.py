"""Adversarial coverage for the heavy-e2e nextest profile split.

Run via: `python3 -m unittest scripts.test_nextest_profile_partition`
from repo root, or `python3 scripts/test_nextest_profile_partition.py`.

PR #126 (refined by #127) split the heavyweight end-to-end suite off the
per-PR integration gate. `.config/nextest.toml` carries three profiles:

  - `default` excludes an explicitly-named heavy-e2e set;
  - `ci` excludes that same set plus every complete test binary named by a
    required Phase 0-3 oracle `--test` argument;
  - `nightly` carries the EXACT SAME set as a positive filter, and the
    `Heavy E2E` workflow runs `cargo nextest run --profile nightly`.

This file locks the original workspace/nightly split plus the delegation
contract for binaries named by oracle `--test` arguments. Selector-based
oracle legs still overlap the workspace lane, including `-E` expressions with
whole-binary arms, so the dtype oracle is not a third disjoint profile.

Two tiers of check:

  - `FilterTextTests` is a fast, no-compile lock on the *text* of the
    three filter blocks: `ci` may add only binaries named by oracle `--test`
    arguments to the default exclusion, and the `nightly` positive filter must
    equal the negated inner set of the default exclusion block.
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
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NEXTEST_TOML = REPO_ROOT / ".config" / "nextest.toml"
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import dtype_phase0_oracle  # noqa: E402
import dtype_phase1_oracle  # noqa: E402
import dtype_phase2_oracle  # noqa: E402
import dtype_phase3_oracle  # noqa: E402
import faithful_observation_phase3_oracle  # noqa: E402


def _filter_blocks() -> list[str]:
    """Return the raw bodies of every `default-filter = '''...'''` block
    in `.config/nextest.toml`, in file order (default, ci, nightly)."""
    text = NEXTEST_TOML.read_text()
    return re.findall(r"default-filter = '''(.*?)'''", text, re.DOTALL)


def _norm(s: str) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    return re.sub(r"\s+", " ", s).strip()


def _negative_filter_inner(block: str) -> str:
    match = re.search(r"not\s*\((.*)\)\s*$", block, re.DOTALL)
    if match is None:
        raise AssertionError("filter is not in the expected `not ( <set> )` shape")
    return match.group(1)


def _required_phase3_commands() -> tuple[tuple[str, ...], ...]:
    """Resolve every command inherited by the required Phase 3 oracle."""
    python = sys.executable
    phase3 = dtype_phase3_oracle.oracle_legs(python)
    if phase3[0].argv != (python, "scripts/dtype_phase2_oracle.py"):
        raise AssertionError("Phase 3 no longer inherits dtype_phase2_oracle.py")
    if phase3[1].argv != (
        python,
        "scripts/faithful_observation_phase3_oracle.py",
    ):
        raise AssertionError(
            "Phase 3 no longer inherits faithful_observation_phase3_oracle.py"
        )
    phase2 = dtype_phase2_oracle.oracle_legs(python)
    if phase2[0].argv != (python, "scripts/dtype_phase1_oracle.py"):
        raise AssertionError("Phase 2 no longer inherits dtype_phase1_oracle.py")
    phase1 = dtype_phase1_oracle.oracle_legs(python)
    if phase1[0].argv != (python, "scripts/dtype_phase0_oracle.py"):
        raise AssertionError("Phase 1 no longer inherits dtype_phase0_oracle.py")

    return (
        *(leg.argv for leg in phase3),
        *(leg.argv for leg in phase2),
        *(leg.argv for leg in phase1),
        *(leg.argv for leg in dtype_phase0_oracle.oracle_legs()),
        *(argv for _name, argv in faithful_observation_phase3_oracle.SUITE_COMMANDS),
    )


def _oracle_selected_test_binaries() -> set[str]:
    """Return complete binaries explicitly named by oracle `--test` arguments."""
    selected: set[str] = set()
    for argv in _required_phase3_commands():
        if "-p" not in argv:
            continue
        package = argv[argv.index("-p") + 1]
        for index, argument in enumerate(argv[:-1]):
            if argument == "--test":
                selected.add(f"{package}::{argv[index + 1]}")
    return selected


ORACLE_OWNED_BINARY_IDS = frozenset(_oracle_selected_test_binaries())
ORACLE_OWNED_FILTERS = tuple(
    f"binary_id(/^{binary_id}$/)"
    for binary_id in sorted(ORACLE_OWNED_BINARY_IDS)
)


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

    def test_ci_adds_only_oracle_owned_binaries_to_default_exclusion(self):
        default_block, ci_block, _nightly = _filter_blocks()
        expected = _norm(_negative_filter_inner(default_block))
        expected += " + " + " + ".join(ORACLE_OWNED_FILTERS)
        self.assertEqual(
            _norm(_negative_filter_inner(ci_block)),
            expected,
            "the `ci` filter must differ from `default` only by the exact "
            "complete binaries named by required dtype oracle `--test` arguments",
        )

    def test_ci_only_exclusions_are_executed_by_the_dtype_oracle(self):
        self.assertEqual(
            ORACLE_OWNED_BINARY_IDS,
            _oracle_selected_test_binaries(),
            "an excluded binary is not selected by the required dtype oracle",
        )

    def test_inherited_phase0_and_observation_binaries_are_delegated(self):
        self.assertTrue(
            {
                "chelis-cli::domain_checker",
                "chelis-cli::parity",
                "chelis-e2e::eval_agreement",
                "chelis-cli::issue_687_rejected_cells_corpus",
                "chelis-types::agreement_tolerance",
            }
            <= ORACLE_OWNED_BINARY_IDS
        )

    def test_nightly_filter_is_the_negated_exclusion_set(self):
        # `default`/`ci` exclude `not ( <set> )`; `nightly` must select
        # exactly `<set>`. If these drift, a test lands in neither
        # profile (dropped coverage) or both (double-run).
        default_block, _ci, nightly_block = _filter_blocks()
        self.assertEqual(
            _norm(_negative_filter_inner(default_block)),
            _norm(nightly_block),
            "the `nightly` positive filter does not equal the negated "
            "inner set of the `default` exclusion filter; a test "
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

    The invariant: every non-ignored test is in the workspace `ci` profile,
    the `nightly` profile, or an explicitly selected required-oracle binary.
    `#[ignore]`-d tests are intentionally outside this partition.
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
        oracle_matches = {
            key
            for key in non_ignored
            if any(
                key.startswith(f"{binary_id}::")
                for binary_id in ORACLE_OWNED_BINARY_IDS
            )
        }
        return ci_matches, nightly_matches, oracle_matches, non_ignored

    def test_ci_and_nightly_are_disjoint(self):
        ci_matches, nightly_matches, _oracle_matches, _ = self._sets()
        overlap = ci_matches & nightly_matches
        self.assertEqual(
            overlap,
            set(),
            f"{len(overlap)} test(s) are on BOTH the per-PR `ci` gate and "
            f"the `nightly` gate (double-run, wastes the per-PR budget): "
            f"{sorted(overlap)[:20]}",
        )

    def test_no_non_ignored_test_falls_into_neither_profile(self):
        ci_matches, nightly_matches, oracle_matches, non_ignored = self._sets()
        gap = non_ignored - ci_matches - nightly_matches - oracle_matches
        self.assertEqual(
            gap,
            set(),
            f"{len(gap)} non-ignored test(s) are on neither the workspace, "
            f"nightly, nor required-oracle lanes -- they silently stopped running "
            f"(dropped coverage): {sorted(gap)[:20]}",
        )

    def test_profiles_plus_delegated_binaries_cover_every_non_ignored_test(self):
        # Belt-and-braces coverage statement; this is not a disjoint
        # three-lane partition because the dtype oracle overlaps `ci` elsewhere.
        ci_matches, nightly_matches, oracle_matches, non_ignored = self._sets()
        covered = (ci_matches | nightly_matches | oracle_matches) & non_ignored
        self.assertEqual(
            covered,
            non_ignored,
            "the union of workspace, nightly, and required-oracle tests does "
            "not cover every non-ignored test",
        )

    def test_nightly_set_is_nonempty(self):
        # A `nightly` profile that matched nothing would mean the split
        # silently dropped the entire heavy-e2e suite.
        _ci, nightly_matches, _oracle_matches, _ = self._sets()
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
