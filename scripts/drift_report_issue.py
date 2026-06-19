#!/usr/bin/env python3
"""Open/close a chelis-HEAD drift tracking issue in a downstream shell repo.

Used by the ecosystem drift canary (.github/workflows/ecosystem-drift.yml).
Each shell matrix leg runs this once, after its build+test gate, with a
per-shell App token scoped to that single repo (issues:write). It mirrors
the open-on-fail / close-on-pass tracking-issue pattern in heavy-e2e.yml's
`report` job, but files the issue IN THE SHELL'S OWN repo (so the signal
lands where that shell's maintainers see it) and uses the `gh` CLI with the
minted token rather than github-script (the canary job has no `github`
context object for a cross-repo issue write).

Behavior:
  - JOB_STATUS == "failure": if no open "chelis HEAD drift: <shell>" issue
    exists, open one; if one already exists, append a comment noting the
    repeat failure (so the issue stays current without churning duplicates).
  - JOB_STATUS == "success": if an open drift issue exists, comment that
    HEAD is green again and close it. Otherwise no-op.
  - Any other JOB_STATUS (cancelled / skipped): no-op (no signal either way).

Idempotency: the open/closed search is by exact title in the shell repo, so
re-runs never create a second issue for the same shell.

Env:
  GH_TOKEN       App token scoped to the shell repo (issues:write).
  SHELL_REPO     The shell repo name, e.g. "nautilus" (under Chelis-Lang).
  JOB_STATUS     The matrix leg's aggregate status ("success"/"failure"/...).
  HEAD_VERSION   The HEAD chelis version the shell was built against.
  RUN_URL        URL of this canary run (for the issue body).
  CHELIS_SHA     The chelis HEAD commit the toolchain was built from.

Exit codes:
  0  reported (or intentional no-op).
  2  missing required environment.
  3  a `gh` invocation failed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ORG = "Chelis-Lang"
LABEL = "chelis-head-drift"


def _run_gh(args: list[str], *, capture: bool = False) -> str:
    """Run `gh <args>`; return stdout when capture=True. Raises on failure."""
    proc = subprocess.run(
        ["gh", *args],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"`gh {' '.join(args)}` failed (exit {proc.returncode}):\n"
            f"{proc.stderr.strip()}"
        )
    return proc.stdout if capture else ""


def find_open_issue(repo: str, title: str) -> int | None:
    """Return the number of the open issue with the exact title, or None."""
    out = _run_gh(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--search",
            f'in:title "{title}"',
            "--json",
            "number,title",
            "--limit",
            "50",
        ],
        capture=True,
    )
    for item in json.loads(out or "[]"):
        if item.get("title") == title:
            return int(item["number"])
    return None


def main() -> int:
    try:
        shell = os.environ["SHELL_REPO"]
        status = os.environ["JOB_STATUS"]
        head_version = os.environ["HEAD_VERSION"]
        run_url = os.environ["RUN_URL"]
        chelis_sha = os.environ["CHELIS_SHA"]
    except KeyError as exc:
        print(f"ERROR: missing required env {exc}", file=sys.stderr)
        return 2

    repo = f"{ORG}/{shell}"
    title = f"chelis HEAD drift: {shell}"

    # A cancelled/skipped leg carries no drift signal either way — do not
    # touch `gh` at all (so a cancelled run never opens or closes anything).
    if status not in ("failure", "success"):
        print(f"Job status {status!r}: no drift signal; skipping report.")
        return 0

    try:
        existing = find_open_issue(repo, title)

        if status == "failure":
            body = (
                f"The **ecosystem drift canary** found that `{shell}@main` no "
                f"longer builds/tests against **chelis HEAD** "
                f"(`{head_version}`, commit `{chelis_sha}`).\n\n"
                f"This means a chelis change on `main` has broken this shell "
                f"before any release. See the failing run for the exact "
                f"`chelis reef build` / `chelis test` output:\n\n"
                f"Run: {run_url}\n\n"
                f"Auto-opened by Chelis-Lang/chelis ecosystem-drift.yml; "
                f"auto-closes when the canary next sees this shell green "
                f"against chelis HEAD."
            )
            if existing is None:
                # Best-effort label; `--label` fails if the label does not
                # exist in the repo, so fall back to a label-less create.
                try:
                    _run_gh(
                        [
                            "issue",
                            "create",
                            "--repo",
                            repo,
                            "--title",
                            title,
                            "--body",
                            body,
                            "--label",
                            LABEL,
                        ]
                    )
                except RuntimeError:
                    _run_gh(
                        [
                            "issue",
                            "create",
                            "--repo",
                            repo,
                            "--title",
                            title,
                            "--body",
                            body,
                        ]
                    )
                print(f"Opened drift issue in {repo}.")
            else:
                _run_gh(
                    [
                        "issue",
                        "comment",
                        str(existing),
                        "--repo",
                        repo,
                        "--body",
                        f"Still drifting against chelis HEAD "
                        f"(`{head_version}`, `{chelis_sha}`). Run: {run_url}",
                    ]
                )
                print(f"Updated existing drift issue #{existing} in {repo}.")

        elif status == "success":
            if existing is not None:
                _run_gh(
                    [
                        "issue",
                        "comment",
                        str(existing),
                        "--repo",
                        repo,
                        "--body",
                        f"`{shell}@main` is green against chelis HEAD again "
                        f"(`{head_version}`). Run: {run_url}. Closing.",
                    ]
                )
                _run_gh(["issue", "close", str(existing), "--repo", repo])
                print(f"Closed drift issue #{existing} in {repo}.")
            else:
                print(f"{repo} green and no open drift issue; nothing to do.")

    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
