#!/usr/bin/env python3
"""Capacity census liveness gate (chelis#729, dtype_semantics.md section C6).

The CI tripwire (crates/chelis-cli/tests/capacity_census_tripwire.rs) proves
the census matches the enumerated surface and that no row is uncited. This
script checks the leg CI cannot: that every cited issue is still OPEN. A row
citing a closed issue has lost its justification - the seam it excused was
supposedly unwound - and must be re-adjudicated (un-censused, or re-cited to
live work) rather than grandfathered forever. This is the known-red-ledger
pattern of the chelis#732 Phase 2 oracle, applied to the census.

Manual gate (needs network + gh auth, so it does not run in default CI):

    .venv/bin/python scripts/capacity_census_liveness.py

Success is exit 0 with the final line `CAPACITY CENSUS LIVENESS: PASS`.
Run it at every release cut and in red-team passes over the numeric surface.
Every sanctioned citation names at least one chelis#N reference (the Rust
tripwire enforces this, maintainer-override included), so every row is
liveness-bound: the seam rows cite chelis#893 and the plain baseline rows
cite chelis#729, which means the entire baseline comes up for
re-adjudication when the plan closes.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "Chelis-Lang/chelis"
CENSUS_REL = Path("spec/design/capacity_census.json")
ISSUE_REF = re.compile(r"chelis#(\d+)")


def extract_issue_refs(citation: str) -> list[int]:
    """Issue numbers referenced by a citation string, in order, deduplicated."""
    seen: list[int] = []
    for match in ISSUE_REF.finditer(citation):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def adjudicate(rows: list[dict], issue_states: dict[int, str]) -> list[str]:
    """Pure verdict logic: return the list of problems (empty means pass).

    `issue_states` maps issue number -> "OPEN" | "CLOSED".
    """
    problems: list[str] = []
    for row in rows:
        citation = str(row.get("citation", "")).strip()
        row_id = f"[{row.get('kind', '?')}] {row.get('id', '?')}"
        if not citation or citation == "TODO":
            problems.append(f"UNCITED row (CI tripwire should have caught this): {row_id}")
            continue
        for number in extract_issue_refs(citation):
            state = issue_states.get(number)
            if state is None:
                problems.append(f"UNRESOLVABLE citation chelis#{number}: {row_id}")
            elif state != "OPEN":
                problems.append(
                    f"STALE citation: chelis#{number} is {state}, so this row's "
                    f"justification no longer stands and it must be re-adjudicated "
                    f"(un-census the surface, or re-cite live work): {row_id}"
                )
    return problems


def fetch_issue_state(number: int) -> str | None:
    """Query GitHub for an issue's state; None when the lookup fails."""
    result = subprocess.run(
        ["gh", "issue", "view", str(number), "--repo", REPO, "--json", "state"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)["state"]
    except (json.JSONDecodeError, KeyError):
        return None


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    census_path = root / CENSUS_REL
    rows = json.loads(census_path.read_text())["rows"]

    numbers = sorted({n for row in rows for n in extract_issue_refs(str(row.get("citation", "")))})
    issue_states = {n: fetch_issue_state(n) for n in numbers}
    problems = adjudicate(rows, {n: s for n, s in issue_states.items() if s is not None})

    if problems:
        print(
            "capacity census liveness violation "
            "(spec/design/dtype_semantics.md section C6; chelis#729):"
        )
        for problem in problems:
            print(f"  {problem}")
        print("CAPACITY CENSUS LIVENESS: FAIL")
        return 1
    print(f"checked {len(rows)} rows, {len(numbers)} cited issues, all OPEN")
    print("CAPACITY CENSUS LIVENESS: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
