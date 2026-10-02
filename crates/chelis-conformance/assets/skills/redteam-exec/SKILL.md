---
name: redteam-exec
description: Run a compliant Chelis red-team round, or send a local repair back to the standing reviewer for verification before push. A round starts with a fresh local subagent working an inline brief against the pushed head, then keeps that reviewer alive through the local repair loop. Anything else is not a red team.
---

# Red Team Exec

Use this skill when the user asks for a red team, an adversarial review, a fresh-context
validation pass, or verification of a fix that a red team reported.

## Repository Contract

1. A **round** starts with a fresh local subagent reviewing from the inline brief below
   and continues through the repair loop. **Verification** is the reviewer that reported
   a finding checking the exact local repair commit before it is pushed. Neither is a
   main-thread pass, and a phase or pull request is red-teamed only when the subagent
   actually ran the validation work.
2. Keep the reviewer alive while the author and reviewer serially hand off the supplied
   worktree. A confirmed in-scope P0 or P1 earns a fresh round after the current round
   closes and its exact verified repair head is pushed, subject to the cap below.
   Unmigrated assertions, old comments, minor documentation drift, and P2-or-lower
   findings do not alone earn another round. A rebase with substantial conflicts or
   semantic overlap gets a targeted red team focused on the overlap; it needs no
   separate permission and does not count toward the fresh-round cap.
3. A pull request gets at most three fresh rounds by default; a fourth needs the user's
   explicit approval. A prose-only pull request, design documents included, gets one,
   and a second needs the same approval. Rounds run from any platform count, and the
   pull request's round record is the counter. Verification does not count against the
   cap; the end-of-pull-request round does.
4. Context budgets and time limits are optional, with no default. When set, include
   them in the brief. The reviewer reports what it has when an explicit limit is
   reached; an unfinished check is "unvalidated", not a finding.
5. The initial head is pushed before the round starts, so CI runs while the first review
   runs. Repair commits remain local until the standing reviewer is satisfied. Push the
   exact verified repair head once after the round closes; CI then validates that head.
   CI is watched by at most one background waiter, never a foreground sleep or poll loop.
6. If every local subagent path is unavailable, state that red-team validation is
   blocked. Do not substitute an external agent CLI or main-thread validation.

## Execution Order: New Round

1. Confirm the candidate is committed and pushed, and count this pull request's fresh
   rounds against the cap. Stop and ask before a round past the cap.
2. Inventory subagent handles created in your own current session. Stop or interrupt
   and retire only stale or failed handles that will not be used again. A standing
   reviewer awaiting a fix is neither; leave it and every other developer's handles
   alone. The platform need not support deleting a retired handle from its listing.
3. Choose the worktree and target: an existing worktree pinned to the exact review head
   with a known clean baseline and no concurrent writer or build owner, together with
   its warm target, or a new isolated one. Decide whether the target is free right now.
4. Fill the round brief template below. Every placeholder is required, and the brief
   does not open with "read `AGENTS.md`".
5. Spawn a new local subagent with the brief as its whole prompt. In Codex sessions, use
   `list_agents`, `interrupt_agent`, `spawn_agent`, `send_message` or `followup_task`,
   and `wait_agent` rather than shelling out to `claude`, `codex exec`, or other
   external CLIs. If the spawn routes to remote infrastructure, errors, or comes back
   broken, retire that handle and retry, or report the red team blocked.
6. Record the initial report in the pull request, assigning a class to any finding the
   report left unlabeled, and keep the reviewer's handle through the repair loop. The
   round closes only when that reviewer is satisfied or the reviewer-unavailable fallback
   below has been recorded.

## Execution Order: Verify My Fix

1. Consolidate the current round's repairs into a local commit. Name the finding and its
   class, that exact unpushed head, and every repair since the initial report. After a
   rebase, also name the hand-resolved intersection; do not treat every upstream file
   brought in by the new base as changed review scope.
2. Re-run `worktree_status.py` after the author hands off the worktree, paste its output
   into the verification brief, and send the brief to the standing reviewer. Wait for
   closed or not closed with the commands it ran.
3. If the reviewer reports another issue, amend or replace the local repair commit and
   repeat verification. Do not push the repair while the round remains open.
4. Record every verification under the same round. When the reviewer is satisfied, push
   the exact verified head once. If the round confirmed an in-scope P0 or P1, start the
   owed fresh round after that push, subject to the cap.
5. If the reviewer is gone, run its exact reproduction yourself and record the commands
   with "reviewer unavailable". This fallback can close the current repair finding but
   does not replace an owed fresh round after a confirmed P0 or P1.

## Worktree And Build Reuse

- Freshness is a property of the reviewer context, not the checkout or build cache.
- Prefer an existing worktree and warm target artifacts when it is pinned to the exact
  review head, its baseline status is known and clean, and no concurrent agent or build
  owns it. Otherwise create an isolated worktree or target.
- The author and standing reviewer may serially reuse the round's worktree. Each handoff
  requires a fresh busy signal and a clean exact local head; author and reviewer never
  write or build there concurrently.
- The brief pastes the busy signal for that target instead of asserting it:
  `.venv/bin/python scripts/worktree_status.py [--path PATH] [--json] [--quiet]`.
  That pasted output is the heavyweight-command handshake for that target. It answers
  free, busy, unknown, or not clean, withholds free when evidence is missing, and prints
  a finished gate report under a history label rather than as current state, so treat
  unknown as busy. Free is its best answer rather than a proof; the probe documents the
  residual case its fail-safe does not reach. The reviewer uses a free target without a cold rebuild and asks
  before starting any other heavyweight build.
- A reviewer whose probes mutate tracked source gets its own worktree, whatever the
  signal reports. This is an exception to reuse, not a caveat on it: sequencing narrows
  the window in which an inserted variant reaches someone else's compile, and a
  separate worktree removes it.
- Reusing a worktree must not relax exact-head verification, adversarial execution, or
  restoration proof. Restore temporary tests, fixtures, and mutations after the pass
  and report the final worktree status unless the user explicitly asks to retain them.

## Round Brief Template

Send this as the subagent's prompt, with every placeholder filled.
Add a context budget or deadline only when one is set for the round.

```text
Red-team round <N> of <cap> for <PR number or branch>: <one-line subject>.

Head: <full SHA> on <branch>, pushed; CI is running on it.
Files under review: <every changed file, one per line>.
In-scope claims: <the pull request's stated claims, one per line>. A finding is in
scope only when this pull request introduces it, worsens it, or claims to correct it;
anything else is out of scope and gets an issue link or "untracked", not a repair.
Worktree: <absolute path>, at the head above, baseline <clean | describe>.
Target: <absolute path>, warm. Busy signal, taken <HH:MM local>, pasted verbatim from
`.venv/bin/python scripts/worktree_status.py --path <target>`:
<paste the command's output here; do not summarise it>
Use it as is; do not rebuild cold, and ask before starting any other heavyweight build.
Report budget: <N> characters.
Deliver by: <SendMessage to <name> | final report>. Nothing else counts as delivery.

Run code, not just eyes: execute tests and commands, add an adversarial probe where
coverage is thin, check positive and negative cases, and compare docs and examples
with shipped behavior. Restore every probe and mutation before you report.

Severity: P0/P1 blocks merge; P2 is residual work; a P3 is one line, no reproduction.
A construct no maintainer would write is not a finding. For prose and design documents
a wrong normative rule, or a plan that cannot close a named in-scope deliverable, is
P1; misdescribing current `main` or a seam between delivery slices is P2; staleness
against a sibling pull request's moving head, wording, pull-request-body line-level
accuracy, and anything whose fix would add text without a necessity sentence are out
of scope.

Report: findings by severity, each with its scope classification and its class (the
defect category or unmet obligation, not the file or line); commands run for every
P0-P2; coverage against the in-scope claims; anything unvalidated; the reviewed head
and final worktree status. Stay available afterwards: a confirmed P0/P1 comes back to
you to verify.
```

## Verification Brief Template

```text
Verify the local fix for <finding id: class> on <PR number or branch>.

Local head: <full SHA>, committed and not pushed. Changed since your initial report:
<files, one per line>.
Worktree: <absolute path>, clean at the local head above. Busy signal, taken <HH:MM
local>, pasted verbatim from `.venv/bin/python scripts/worktree_status.py --path
<worktree>`:
<paste the command's output here; do not summarise it>
The author has handed off the worktree and will not write or build there during
verification. Re-run your exact reproduction against this head. Report closed or not
closed, the commands you ran, and the head.
Deliver by: <SendMessage to <name> | final report>.
Acknowledge this message and confirm what you will do.
```

## Finding Discipline

- Every pull request, documentation-only work included, needs at least one compliant
  round before it merges. Only a confirmed in-scope P0/P1 blocks, and the standing
  reviewer's local verification closes the finding inside the current round. A
  confirmed in-scope P0/P1 then earns a fresh round on the pushed repair head, subject
  to the cap.
- Absent an in-scope P0 or P1 finding, scale rounds to the change. Minor updates, bug
  fixes, and textual changes do not inherently merit another round. If another change
  already requires a push or CI rerun, include every known P2-or-lower repair in that
  local candidate before the standing reviewer closes the round.
- A rebase with substantial conflicts or semantic overlap gets a targeted red team
  focused on that overlap and does not count toward the cap. A rebase with no such
  overlap does not automatically require review.
- Classify every finding against the pull request's stated scope. Mere discovery,
  including an unrelated pre-existing spec/implementation mismatch, does not bring it
  into scope. Do not repair an out-of-scope finding in the pull request; link its
  existing issue or file one when it is not already tracked.
- A gate repair may correct, remove, or narrow existing pull-request content. It must
  not add design scope, implementation responsibilities, inventories, mechanisms, or
  promises merely to absorb a finding. State every claim at the granularity its oracle
  proves; an unbounded universal claim invites sampling in every round.
- Record the class of every finding in the round record. When two consecutive rounds
  report the same class, or replace a repaired finding with a different class, stop
  patching witnesses: change the representation, the oracle, the claim, or the brief
  before another round. Do not keep expanding the pull request to satisfy a moving
  brief.
- Separability sizes a pull request. Slices that must ship together are commits
  inside one pull request, and slices that can ship apart are separate pull requests.
  About 1,000 hand-written changed lines, regenerated artifacts excluded, is the point
  at which the author owes a sentence justifying one shippable slice, not a threshold.
  Size is in scope for the round: the ratio of cases a claim covers to cases its tests
  prove is the reportable signal, and a low one is a finding whatever the line count.

## Minimum Deliverable

- findings ordered by severity, each with its class and its in-scope or out-of-scope
  classification, with linked issue status for every out-of-scope defect
- exact commands run for every P0, P1, and P2; a P3 is one line with no reproduction
- coverage against the in-scope claims and the active acceptance oracle
- exact reviewed commit and final worktree status; deadline met or missed when one was set
- explicit note of anything unvalidated

## Validation Discipline

- run code and commands, not just source inspection
- add adversarial probes where coverage is thin
- verify positive and negative cases
- check examples, docs, and CLI behavior against shipped behavior
