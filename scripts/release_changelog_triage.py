#!/usr/bin/env python3
"""Every commit in a release range lands in exactly one bucket.

Cutting 0.18.7 found the same defect three times: user-visible work shipped
with no changelog entry. Each time the triage was a human reading commit
subjects or bodies and judging. The first pass filtered on conventional-commit
prefixes and missed 32 commits that have none. The second screened bodies for
phrases like "no longer" and dismissed two PRs as noise that between them
closed three `release blocking` soundness issues.

The judgement was the defect, so this removes it. A commit is REPORTED if the
changelog section cites its PR or any issue it closes, and OMITTED otherwise.
Closing a `soundness` or `release blocking` issue while OMITTED is a hard
failure, not a warning: those are exactly the entries a downstream shell
maintainer reads before a pin bump.

Usage:
    release_changelog_triage.py --from v0.18.6 --to HEAD --section 0.18.7
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "Chelis-Lang/chelis"
BLOCKING_LABELS = {"soundness", "release blocking"}
CLOSES = re.compile(r"\b(?:closes|fixes|resolves)\s+(?:chelis)?#(\d+)", re.I)
PR_IN_SUBJECT = re.compile(r"\(#(\d+)\)\s*$")


def _gh_json(args: list[str]) -> dict | list | None:
    done = subprocess.run(["gh", *args], capture_output=True, text=True)
    if done.returncode != 0:
        return None
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError:
        return None


def commits(lo: str, hi: str) -> list[tuple[str, str]]:
    out = subprocess.run(
        ["git", "log", "--format=%h\t%s", f"{lo}..{hi}"],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for line in out.splitlines():
        sha, _, subject = line.partition("\t")
        rows.append((sha, subject))
    return rows


def section_text(changelog: Path, version: str) -> str:
    text = changelog.read_text()
    start = text.index(f"## [{version}]")
    rest = text[start + 1 :]
    end = rest.index("\n## [")
    return rest[:end]


def triage(lo: str, hi: str, version: str, cache: Path) -> int:
    body_cache: dict[str, str] = json.loads(cache.read_text()) if cache.is_file() else {}
    exempt_path = Path(__file__).with_name("release_changelog_exemptions.json")
    exempt = json.loads(exempt_path.read_text())["exemptions"] if exempt_path.is_file() else {}
    section = section_text(Path("CHANGELOG.md"), version)
    reported, omitted, blocking = [], [], []

    for sha, subject in commits(lo, hi):
        m = PR_IN_SUBJECT.search(subject)
        pr = m.group(1) if m else None
        body = ""
        if pr:
            if pr not in body_cache:
                data = _gh_json(["pr", "view", pr, "--repo", REPO, "--json", "body"])
                body_cache[pr] = (data or {}).get("body", "") if isinstance(data, dict) else ""
            body = body_cache[pr]

        closed = sorted(set(CLOSES.findall(body)))
        cited = (pr and f"#{pr}" in section) or any(f"#{n}" in section for n in closed)

        risky = []
        for issue in closed:
            key = f"issue:{issue}"
            if key not in body_cache:
                data = _gh_json(["issue", "view", issue, "--repo", REPO, "--json", "labels"])
                labels = [l["name"] for l in (data or {}).get("labels", [])] if isinstance(data, dict) else []
                body_cache[key] = ",".join(labels)
            if set(body_cache[key].split(",")) & BLOCKING_LABELS:
                risky.append(issue)

        row = (sha, pr, subject[:64], closed, risky)
        (reported if cited else omitted).append(row)
        if risky and not cited and sha not in exempt:
            blocking.append(row)

    cache.write_text(json.dumps(body_cache))

    print(f"range {lo}..{hi}  section [{version}]")
    print(f"  reported: {len(reported)}")
    print(f"  omitted:  {len(omitted)}")
    if blocking:
        print(f"\nFAIL: {len(blocking)} omitted commit(s) close a "
              f"{' or '.join(sorted(BLOCKING_LABELS))} issue:\n")
        for sha, pr, subject, closed, risky in blocking:
            print(f"  {sha}  PR #{pr}  closes {', '.join('#'+r for r in risky)}")
            print(f"      {subject}")
        return 1
    if exempt:
        print(f"  exempt:   {len(exempt)} (see release_changelog_exemptions.json)")
    print("\nOK: no omitted commit closes a soundness or release-blocking issue.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="lo", required=True)
    ap.add_argument("--to", dest="hi", default="HEAD")
    ap.add_argument("--section", required=True)
    ap.add_argument("--cache", default="target/changelog-triage-cache.json")
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    return triage(args.lo, args.hi, args.section, cache)


if __name__ == "__main__":
    sys.exit(main())
