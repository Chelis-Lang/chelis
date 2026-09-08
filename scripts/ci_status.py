#!/usr/bin/env python3
"""Report required checks separately from other CI work on one exact PR head.

This reports status-check readiness only. It neither merges nor substitutes for
review, mergeability, or the repository's validation contract. API failures and
changes to the PR identity abort the watch instead of producing a green report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from urllib.parse import quote


def requirements(classic: dict, rules: list[dict]) -> list[dict]:
    result = {(c["context"], c.get("app_id")) for c in classic.get("checks", [])}
    bound = {context for context, _ in result}
    result.update((name, None) for name in classic.get("contexts", []) if name not in bound)
    for rule in rules:
        if rule["type"] == "required_status_checks":
            result.update((c["context"], c.get("integration_id")) for c in
                          rule["parameters"]["required_status_checks"])
    return [{"context": name, "app_id": app} for name, app in
            sorted(result, key=lambda item: (item[0], item[1] or 0))]


def classify(required: list[dict], checks: list[dict], statuses: list[dict]) -> dict:
    if not required:
        raise ValueError("No required status checks found; refusing to report readiness")
    latest = {}
    for check in checks:
        key = ("check", check["name"], check["app"]["id"])
        if key not in latest or check["id"] > latest[key]["id"]:
            latest[key] = check
    for status in statuses:
        key = ("status", status["context"], None)
        if key not in latest or status["id"] > latest[key]["id"]:
            latest[key] = status

    def state(key, value):
        if key[0] == "status":
            return "passed" if value["state"] == "success" else (
                "pending" if value["state"] == "pending" else "failed")
        if value["status"] != "completed":
            return "pending"
        return "passed" if value["conclusion"] in {"success", "neutral", "skipped"} else "failed"

    result = {f"required_{kind}": [] for kind in ("pending", "missing", "failed")}
    result.update(other_pending=[], other_failed=[])
    matched = set()
    for requirement in required:
        name, app = requirement["context"], requirement.get("app_id")
        candidates = {key for key in latest if key[1] == name and
                      (app in (None, -1) or key[2] == app)}
        matched.update(candidates)
        if not candidates:
            result["required_missing"].append(name)
        else:
            states = {state(key, latest[key]) for key in candidates}
            for kind in ("failed", "pending"):
                if kind in states:
                    result[f"required_{kind}"].append(name)
    for key in latest.keys() - matched:
        kind = state(key, latest[key])
        if kind in ("pending", "failed"):
            result[f"other_{kind}"].append(key[1])
    result = {key: sorted(set(value)) for key, value in result.items()}
    result["required_passed"] = not any(result[key] for key in
                                       ("required_pending", "required_missing", "required_failed"))
    return result


def check_identity(original: dict, current: dict) -> None:
    if original["state"] != "OPEN" or any(original[key] != current[key] for key in
                                            ("headRefOid", "baseRefOid", "baseTipOid", "baseRefName", "state")):
        raise ValueError("PR head, base, or open state changed; restart with the intended head")


def gh(*args: str):
    completed = subprocess.run(["gh", *args], text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "GitHub command failed")
    return json.loads(completed.stdout)


def pr_identity(repo: str, pr: int) -> dict:
    identity = gh("pr", "view", str(pr), "--repo", repo, "--json",
                  "headRefOid,baseRefOid,baseRefName,state")
    # PR baseRefOid can retain the comparison base after the branch advances.
    # Read the actual ref as well so a watch cannot miss that movement.
    base = quote(identity["baseRefName"], safe="")
    identity = {**identity, "baseTipOid": gh(
        "api", f"repos/{repo}/git/ref/heads/{base}")["object"]["sha"]}
    return identity


def snapshot(repo: str, pr: int, original: dict) -> dict:
    prefix = f"repos/{repo}"
    base = quote(original["baseRefName"], safe="")
    classic = gh("api", f"{prefix}/branches/{base}/protection/required_status_checks")
    rules = gh("api", "--paginate", "--slurp", f"{prefix}/rules/branches/{base}?per_page=100")
    head = original["headRefOid"]
    check_pages = gh("api", "--paginate", "--slurp",
                     f"{prefix}/commits/{head}/check-runs?filter=latest&per_page=100")
    status_pages = gh("api", "--paginate", "--slurp",
                      f"{prefix}/commits/{head}/statuses?per_page=100")
    result = classify(requirements(classic, [rule for page in rules for rule in page]),
                      [check for page in check_pages for check in page["check_runs"]],
                      [status for page in status_pages for status in page])
    check_identity(original, pr_identity(repo, pr))
    return {"repo": repo, "pr": pr, "head": head,
            "base_tip": original["baseTipOid"], **result}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr", type=int)
    parser.add_argument("--repo", default="Chelis-Lang/chelis")
    parser.add_argument("--head", help="Expected full head SHA; a mismatch aborts")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=55)
    args = parser.parse_args(argv)
    if args.interval < 5:
        parser.error("--interval must be at least 5 seconds")
    try:
        original = pr_identity(args.repo, args.pr)
        check_identity(original, original)
        if args.head and args.head != original["headRefOid"]:
            raise ValueError("PR head does not match --head")
        last = None
        while True:
            result = snapshot(args.repo, args.pr, original)
            if result != last:
                print(json.dumps({"observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                  **result}, sort_keys=True), flush=True)
                last = result
            if result["required_failed"] or result["other_failed"]:
                return 1
            if result["required_passed"]:
                return 0
            if not args.watch:
                return 2
            time.sleep(args.interval)
    except (RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"CI status unavailable: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
