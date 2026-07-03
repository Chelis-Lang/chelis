---
name: issue-resolution
description: Use when picking up a GitHub issue to investigate or fix. Claims the issue via assignee before any code is written so others can see it is in progress, scopes the work from the full issue thread, and links the PR with honest Closes/Part-of wording.
---

# Issue Resolution

Use this skill whenever a session or subagent picks up a GitHub issue
(in chelis or a shell repo) to investigate or fix.

## Claim The Issue First

- Before writing any code, check the claim state:
  `gh issue view <N> --repo <owner>/<repo> --json state,title,assignees`
- If the issue is already assigned to someone other than the repo owner
  operating this workstation, stop and surface that instead of starting
  work; a live assignee means someone else may be mid-flight.
- If it is unassigned (or assigned only to the operator), claim it:
  `gh issue edit <N> --repo <owner>/<repo> --add-assignee @me`
  `@me` resolves to the authenticated `gh` account (the workstation
  owner, e.g. `rlronan`), never to an agent identity. The assignment is
  the public "in progress" signal for outside collaborators.
- Claim after deciding to work the issue and before branching or
  editing, so the claim window matches the work window. Multiple agents
  on one workstation share one assignee account; branch names
  (`agent/<N>-<slug>`) disambiguate the parallel efforts internally.
- If `gh issue edit` fails for lack of triage rights (for example an
  upstream repo worked from a shell), leave a claiming comment on the
  issue naming the working branch instead.
- Do not claim a closed issue. If the problem persists at head, file a
  residue issue that links the closed one, then claim the new issue.

## Scope From The Whole Thread

- Read the body and every comment (`gh issue view <N> --comments`)
  before scoping. Late comments often rescope an issue: partial fixes
  already landed, escalations, added reproducers, or narrowed intent.
- Check linked PRs and any expected-to-fail pins the thread names; run
  those pins first to see the live failure mode before changing code.

## Land With Honest Linkage

- Follow the repo contract (`AGENTS.md`) for the change itself:
  spec-first tests, negative-test parity, public-surface sync.
- PR bodies use `Closes #N` only when the PR fully resolves the issue;
  otherwise `Part of #N` plus an explicit statement of what remains.
- Keep the assignee in place while the PR is open; merging or closing
  the issue releases the claim naturally.

## Release Stale Claims

- If the work is abandoned with nothing pushed, remove the assignment
  (`gh issue edit <N> --remove-assignee @me`) or comment with current
  status, so a stale claim does not block others from picking it up.
