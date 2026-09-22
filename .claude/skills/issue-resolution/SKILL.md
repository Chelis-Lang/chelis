---
name: issue-resolution
description: Use when picking up, resolving, addressing, fixing, working, tackling, triaging, or closing out one or more GitHub issues (tickets, bug reports, feature requests) in chelis or a shell repo, whether referenced by number (#N) or URL or handed as a batch, and even when a PR already exists. Claims each issue via assignee (gh issue edit --add-assignee @me) before branching or writing any code so others can see it is in progress, scopes the work from the full issue thread (body and every comment), and links the PR with honest Closes/Part-of wording.
---

# Issue Resolution

Use this skill whenever a session or subagent picks up, resolves,
addresses, fixes, works, tackles, or closes out one or more GitHub
issues (tickets, bug reports, feature requests) in chelis or a shell
repo, whether given an issue number, a URL, or a batch to work through,
and even when a PR for the issue already exists.

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
- Automatic issue closure is disabled for this repository: `Closes #N`
  in a PR body or commit message does nothing. Say `Closes #N` only when
  the PR fully resolves the issue, and `Part of #N` plus an explicit
  statement of what remains otherwise. Either way the sentence is a claim
  for the reviewer, not an instruction to GitHub.
- Merging does not close the issue. After the PR merges, close it as a
  separate step once the work is verified (`gh issue close <N>` with a
  comment naming the PR), or leave it open and say what remains.
- Keep the assignee in place while the PR is open; closing the issue
  releases the claim naturally.

## Release Stale Claims

- If the work is abandoned with nothing pushed, remove the assignment
  (`gh issue edit <N> --remove-assignee @me`) or comment with current
  status, so a stale claim does not block others from picking it up.
