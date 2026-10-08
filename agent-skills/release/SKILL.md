---
name: release
description: Use when cutting a Chelis compiler release (a new vX.Y.Z tag of chelis and chelisup) or when asked to bump the compiler version, assemble release notes, or tag main. Covers the release PR, the merge-then-tag sequence, the release and Release E2E workflows, and recovery when a tag fails. Compiler repository only; shells never receive this skill.
---

# Release

A release is one squash-merged pull request that bumps the version and assembles the
notes, followed by an annotated tag on that merge commit. The tag push runs
`release.yml`, which builds every platform, runs the installed-artifact canary, and
publishes the GitHub Release; `release-e2e.yml` then installs the published release on
the hosts the install guide names. Nothing in the release PR is written by hand: two
scripts produce it, and the PR exists so CI tests the bumped tree before it becomes
`main`.

The release PR follows the ordinary [PR lifecycle](../../AGENTS.md#pull-request-lifecycle)
and [the PR-author guide](../../docs/guard_changes_for_pr_authors.md). This skill adds
only what is specific to a release.

## Decide The Version

The user names the version. Never start a release the user did not ask for.

- `workspace.package.version` in the root `Cargo.toml` is the previous release, and the
  tag is `v` plus the new version.
- Before 1.0, a minor bump may break documented behavior. A patch bump is reserved for
  bug fixes that do not change documented behavior (`phase3j_pre_release.md`), so a
  pending `*.breaking.md` fragment in `changelog.d/` rules out a patch release, and a
  pending `added` or `changed` fragment needs the user's confirmation that a patch is
  still intended. If the named version conflicts with the pending fragments, stop and
  ask.

## Prepare On Current Main

Work in a fresh worktree on `origin/main`, with its own `uv venv --python 3.11`.

1. Record the state of `main` for the PR body: the head SHA, the CI and Hull runs on that
   SHA (`gh run list --branch main --commit <sha>`), and the open `failing-on-main`
   issues (`gh issue list --label failing-on-main`). Dispatch
   `gh workflow run loud-unsupported-nightly.yml`, which the PR-author guide asks to be
   run by hand before a release, and record its result. Red results do not block on
   their own; the user decides, so list them.
2. Bump and assemble:

   ```sh
   .venv/bin/python scripts/bump_compiler_pins.py X.Y.Z
   .venv/bin/python scripts/changelog.py check
   .venv/bin/python scripts/changelog.py build --version X.Y.Z --date YYYY-MM-DD
   .venv/bin/python scripts/changelog.py build --version X.Y.Z --date YYYY-MM-DD --write
   .venv/bin/python scripts/changelog.py extract --version vX.Y.Z --output <scratch>/notes.md
   ```

   The bump rewrites the workspace version, every hand-pinned `package.compiler`, the
   pipeline-parity captures, the Hull corpus pin, the root and compile-fail fixture
   `Cargo.lock` files, and regenerates the conformance assets. Read the preview before
   `--write`. The date is the release day.
3. Check the diff touches only `Cargo.toml`, `Cargo.lock` files, `reef.toml` and pinned
   fixture JSON files, `tests/conformance/hull/manifest.json`, `CHANGELOG.md`, and
   deleted `changelog.d/` fragments. Lock diffs change only local package `version`
   lines: no `checksum` or `source` line moves. Anything else means a script or the tree
   is wrong; stop and report it rather than editing around it.
4. Run `.venv/bin/python -m unittest scripts.test_bump_compiler_pins scripts.test_changelog`
   and `python3 scripts/gate.py --fast`.

A release whose version bump already merged in an earlier PR skips the bump; the
changelog check accepts a notes-only release PR.

## Open And Review The PR

- Title and squash subject: `chore(release): prepare X.Y.Z`. No changelog fragment: the
  assembled section is the release's changelog content.
- The body records the base `main` SHA and its CI and Hull runs, the fragments consumed,
  the open `failing-on-main` issues and the liveness result, and the commands run.
- One red-team round through the `redteam-exec` skill. The brief asks the reviewer to
  rebuild the section independently from the base fragments, compare the pin inventory
  and lock diffs with step 3, and run `changelog.py extract` on the head.
- The head changes `Cargo.toml` and lock files, so dispatch `PR Package Expansion` on
  the settled head as the PR-author guide describes.
- Optional, before merge, when the release changes packaging, install, or build targets:
  `gh workflow run release.yml --ref <branch>` builds the candidate without publishing,
  and `gh workflow run release-e2e.yml -f candidate_run=<that run's id>` installs those
  artifacts on every host except the steps that need a published release.

Fragments that merge to `main` while the PR is open are not in the assembled section,
and the tag's preflight refuses a commit that still carries fragments. Before merging,
fetch and run `git diff --name-only <pr-base> origin/main -- changelog.d/`. If it lists a
fragment, rebuild the branch on `origin/main` by rerunning the steps above, declare the
`Candidate-base-update:` line in the body, and push with `--force-with-lease`.

## Merge And Tag

Merging and tagging need the user's go-ahead; the release is outward-facing.

```sh
gh pr merge <N> --squash
git fetch origin main
sha=$(gh pr view <N> --json mergeCommit --jq .mergeCommit.oid)
git merge-base --is-ancestor "$sha" origin/main
git show "$sha":Cargo.toml | grep -m1 '^version'
git tag -a vX.Y.Z "$sha" -m "Chelis X.Y.Z"
git push origin vX.Y.Z
```

Tag the merge commit by SHA, never a moving `main`, and never tag the PR branch.

## Watch The Release

1. Find the tag's `release.yml` run and watch it with one background waiter
   (`gh run watch <id> --exit-status`). Its preflight extracts the notes within about a
   minute; the platform builds and canary take about 35 minutes.
2. `gh release view vX.Y.Z --json assets`: every build job in `release.yml` contributes a
   tarball and its `.sha256`, plus the `chelisup-*` executables and `chelisup.sh`.
   Compare with the previous release's asset list and any build job added since.
3. `release-e2e.yml` starts when the tag run succeeds. Read its summary table. It reports
   and does not gate publishing; a failing host becomes an issue, not a re-release.
4. Report to the user: the tag, the release URL, the release run, and the E2E table.

## When A Tag Fails

- A flaky build or canary job: `gh run rerun <id> --failed`.
- The preflight refuses the notes, or a real build failure: the release is unpublished.
  Do not move or delete the tag without the user's approval. The fix is a new PR on
  `main`; once it merges, the user decides whether to delete and re-push the same tag
  on the new merge commit or to release the next patch version.
- Never edit a published release's assets by hand.

## After The Release

Propagating the new compiler to shells, and updating the book and chelis.ch for the
release, are separate tasks. Hand them back to the user; do not start a shell bump wave
as part of the release.
