"""Unit tests for scripts/ci_cache_prune.py's deletion policy."""

from __future__ import annotations

import unittest

import ci_cache_prune as mod


def _c(cid, key, ref, acc, size=500_000_000):
    return {"id": cid, "key": key, "ref": ref, "lastAccessedAt": acc, "sizeInBytes": size}


class CachePrefixTests(unittest.TestCase):
    def test_strips_trailing_hex_generations(self):
        self.assertEqual(
            mod.cache_prefix("v0-rust-smt-smt-build-Linux-x64-fa01682d-b5d55b63"),
            "v0-rust-smt-smt-build-Linux-x64",
        )

    def test_leaves_schema_suffix_alone(self):
        # cvc5 keys end in -v1/-v2, not a hex generation.
        self.assertEqual(
            mod.cache_prefix("cvc5-prebuilt-linux-x86_64-cvc5sys0.3.1-v2"),
            "cvc5-prebuilt-linux-x86_64-cvc5sys0.3.1-v2",
        )


class SelectDeletionsTests(unittest.TestCase):
    def test_keeps_newest_main_generation_deletes_older(self):
        caches = [
            _c(1, "v0-rust-lint-Linux-x64-aaaaaa-bbbbbb", "refs/heads/main", "2026-07-01T10:00:00Z"),
            _c(2, "v0-rust-lint-Linux-x64-cccccc-dddddd", "refs/heads/main", "2026-06-20T10:00:00Z"),
            _c(3, "v0-rust-lint-Linux-x64-eeeeee-ffffff", "refs/heads/main", "2026-06-10T10:00:00Z"),
        ]
        victims = mod.select_deletions(caches, open_pr_numbers=set(), keep_per_prefix=1)
        self.assertEqual(victims, [2, 3])  # newest (id 1) kept

    def test_keep_per_prefix_two(self):
        caches = [
            _c(1, "v0-rust-lint-Linux-x64-aaaaaa-111111", "refs/heads/main", "2026-07-01T10:00:00Z"),
            _c(2, "v0-rust-lint-Linux-x64-bbbbbb-222222", "refs/heads/main", "2026-06-20T10:00:00Z"),
            _c(3, "v0-rust-lint-Linux-x64-cccccc-333333", "refs/heads/main", "2026-06-10T10:00:00Z"),
        ]
        self.assertEqual(
            mod.select_deletions(caches, set(), keep_per_prefix=2), [3]
        )

    def test_deletes_closed_pr_caches(self):
        caches = [
            _c(10, "v0-rust-lint-Linux-x64-a-aaaaaa", "refs/pull/500/merge", "2026-06-01T10:00:00Z"),
            _c(11, "v0-rust-lint-Linux-x64-b-bbbbbb", "refs/pull/999/merge", "2026-06-01T10:00:00Z"),
        ]
        # PR 999 open, 500 closed.
        victims = mod.select_deletions(caches, open_pr_numbers={999}, keep_per_prefix=1)
        self.assertEqual(victims, [10])

    def test_none_open_prs_skips_pr_pruning(self):
        # Fail-safe: a failed open-PR lookup must NOT mass-delete PR caches.
        caches = [
            _c(10, "k-a-aaaaaa", "refs/pull/500/merge", "2026-06-01T10:00:00Z"),
            _c(11, "k-b-bbbbbb", "refs/pull/501/merge", "2026-06-01T10:00:00Z"),
        ]
        self.assertEqual(mod.select_deletions(caches, open_pr_numbers=None), [])

    def test_never_deletes_cvc5_prebuilt(self):
        caches = [
            # Even a duplicate main generation of a cvc5 cache is protected.
            _c(20, "cvc5-prebuilt-linux-x86_64-cvc5sys0.3.1-v2", "refs/heads/main", "2026-07-01T10:00:00Z"),
            _c(21, "cvc5-prebuilt-linux-x86_64-cvc5sys0.3.0-v2", "refs/heads/main", "2026-05-01T10:00:00Z"),
            # A closed-PR cvc5 cache is also protected.
            _c(22, "cvc5-prebuilt-darwin-arm64-cvc5sys0.3.1-v2", "refs/pull/500/merge", "2026-05-01T10:00:00Z"),
        ]
        self.assertEqual(mod.select_deletions(caches, open_pr_numbers=set()), [])

    def test_open_pr_and_other_branch_caches_untouched(self):
        caches = [
            _c(30, "k-a-aaaaaa", "refs/pull/999/merge", "2026-06-01T10:00:00Z"),  # open PR
            _c(31, "k-b-bbbbbb", "refs/heads/feature-x", "2026-06-01T10:00:00Z"),  # non-main branch
        ]
        # Open PR kept; non-main branch not in scope of rule 2 (main-only).
        self.assertEqual(mod.select_deletions(caches, open_pr_numbers={999}), [])


if __name__ == "__main__":
    unittest.main()
