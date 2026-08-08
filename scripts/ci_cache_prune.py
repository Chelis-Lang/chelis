"""Prune stale GitHub Actions caches so the repo stays well under the 10GB
per-repo LRU budget.

Why
---
The repo's Actions-cache pool is dominated by `Swatinem/rust-cache` `target/`
snapshots -- one per (job, Cargo.lock/rustc generation). Each merge strands the
previous generation, and every closed PR leaves its `refs/pull/N/merge` caches
behind, so the pool creeps over 10GB and GitHub LRU-evicts whatever is least
recently used. This job deletes the provably-dead entries on a schedule so
useful caches are never the eviction victim.

Deletion policy (`select_deletions`, pure + unit-tested)
--------------------------------------------------------
* Delete caches on a CLOSED/merged PR ref (`refs/pull/N/merge`, N not in the
  currently-open set). A closed PR's caches will never be read again. Skipped
  entirely if the open-PR set could not be fetched (fail safe: never mass-
  delete on a lookup error).
* On `refs/heads/main`, keep the most-recently-accessed `keep_per_prefix`
  generation(s) per job prefix and delete older duplicates. The prefix strips
  trailing hex generation segments, so `...smt-build-Linux-x64-<a>-<b>` groups
  by `...smt-build-Linux-x64`.
* NEVER delete a durable-cvc5 fallback cache (`cvc5-prebuilt-*`); those are the
  secondary link source and are tiny.

Default is a DRY RUN (print the plan). `--apply` performs the deletes.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys


# The cvc5-prebuilt prefix is protected as pinned data
# (adopt-shared-ci-actions). Its producer machinery retired with the
# SMT-lane Nix convergence, so no key under it is created any more;
# protecting the absent family is harmless and keeps the shared-action
# input stable.
PROTECTED_PREFIXES = ("cvc5-prebuilt-",)

# Strip one-or-more trailing "-<hex6+>" generation segments so all Cargo.lock/
# rustc generations of the same job collapse to one prefix.
_GENERATION_SUFFIX = re.compile(r"(-[0-9a-fA-F]{6,})+$")
_PR_REF = re.compile(r"^refs/pull/(\d+)/")


def cache_prefix(key: str) -> str:
    return _GENERATION_SUFFIX.sub("", key)


def _protected(key: str) -> bool:
    return any(key.startswith(p) for p in PROTECTED_PREFIXES)


def select_deletions(
    caches: list[dict],
    open_pr_numbers: set[int] | None,
    keep_per_prefix: int = 1,
) -> list[int]:
    """Return the cache ids to delete under the policy above.

    `caches` items are dicts with at least `id`, `key`, `ref`,
    `lastAccessedAt`. `open_pr_numbers` is the set of currently-open PR numbers,
    or None to skip closed-PR pruning entirely (fail-safe on lookup error).
    """
    to_delete: set[int] = set()

    # Rule 1: closed-PR-ref caches (only when the open set is known).
    if open_pr_numbers is not None:
        for c in caches:
            if _protected(c["key"]):
                continue
            m = _PR_REF.match(c.get("ref", ""))
            if m and int(m.group(1)) not in open_pr_numbers:
                to_delete.add(c["id"])

    # Rule 2: stale duplicate main generations.
    groups: dict[str, list[dict]] = {}
    for c in caches:
        if _protected(c["key"]) or c.get("ref") != "refs/heads/main":
            continue
        groups.setdefault(cache_prefix(c["key"]), []).append(c)
    for group in groups.values():
        newest_first = sorted(
            group, key=lambda c: c.get("lastAccessedAt", ""), reverse=True
        )
        for c in newest_first[keep_per_prefix:]:
            to_delete.add(c["id"])

    return sorted(to_delete)


def _gh_json(args: list[str]) -> list[dict]:
    out = subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out) if out.strip() else []


def list_caches() -> list[dict]:
    return _gh_json(
        [
            "cache",
            "list",
            "--limit",
            "200",
            "--json",
            "id,key,ref,lastAccessedAt,sizeInBytes",
        ]
    )


def open_pr_numbers() -> set[int] | None:
    """Currently-open PR numbers, or None if the lookup fails (fail-safe)."""
    try:
        prs = _gh_json(["pr", "list", "--state", "open", "--limit", "500", "--json", "number"])
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError) as exc:
        print(f"could not list open PRs ({exc}); skipping closed-PR pruning", file=sys.stderr)
        return None
    return {int(p["number"]) for p in prs}


def delete_cache(cache_id: int) -> None:
    subprocess.run(["gh", "cache", "delete", str(cache_id)], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="delete (default: dry-run print only)"
    )
    parser.add_argument("--keep-per-prefix", type=int, default=1)
    args = parser.parse_args(argv)

    caches = list_caches()
    by_id = {c["id"]: c for c in caches}
    victims = select_deletions(caches, open_pr_numbers(), args.keep_per_prefix)

    reclaimed = sum(by_id[i].get("sizeInBytes", 0) for i in victims)
    print(
        f"{len(caches)} caches; selecting {len(victims)} for deletion "
        f"(~{reclaimed / 1e9:.2f} GB){' [DRY RUN]' if not args.apply else ''}"
    )
    for i in victims:
        c = by_id[i]
        print(f"  {'delete' if args.apply else 'would delete'} {c['key']} ({c.get('ref')})")
        if args.apply:
            delete_cache(i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
