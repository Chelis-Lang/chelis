---
name: redteam-exec
description: Run a compliant Chelis red-team round, or send a fix back to the standing reviewer for verification. A round is a fresh local subagent working an inline brief against the pushed head, reusing a clean exact-head worktree and warm target; a fix is verified by the reviewer that reported it. Anything else is not a red team.
---

# Red Team Exec

Use this skill when the user asks for a red team, an adversarial review, a fresh-context
validation pass, or verification of a fix that a red team reported.

## Repository Contract

1. Two activities exist. A **round** is a fresh local subagent reviewing from the inline
   brief below. **Verification** is the reviewer that reported a finding checking the
   repair. Neither is a main-thread pass, and a phase or pull request is red-teamed only
   when the subagent actually ran the validation work.
2. A confirmed in-scope P0 or P1 goes back to the reviewer that reported it. A fresh
   round is owed only when the fix introduces a new mechanism or touches files the
   standing reviewer did not read, or once at the end of a long pull request before
   ready-for-review. A one-word or one-line repair inside the files the reviewer read
   never earns a fresh round.
3. A pull request gets at most three fresh rounds by default; a fourth needs the user's
   explicit approval. A prose-only pull request, design documents included, gets one,
   and a second needs the same approval. Rounds run from any platform count, and the
   pull request's round record is the counter. Verification does not count against the
   cap; the end-of-pull-request round does.
4. Context budgets and time limits are optional, with no default. When set, include
   them in the brief. The reviewer reports what it has when an explicit limit is
   reached; an unfinished check is "unvalidated", not a finding.
5. The head under review is pushed before the round starts, so CI runs on it while the
   review runs. CI is watched by at most one background waiter, never a foreground
   sleep or poll loop.
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
6. Record the round in the pull request, assigning a class to any finding the report
   left unlabeled, and keep the reviewer's handle for verification.

## Execution Order: Verify My Fix

1. Name the finding and its class, the new pushed head, and every file changed since
   the round. If the fix adds a mechanism or touches files the reviewer did not read,
   stop: that owes a fresh round, not a verification.
2. Send the verification brief below to the standing reviewer's handle and wait for
   closed or not closed with the commands it ran.
3. Record the result under the same round in the pull request. If the reviewer is gone,
   run its exact reproduction yourself, record the commands with "reviewer unavailable",
   and let the end-of-pull-request round cover the fix if one is owed. A lost reviewer
   does not earn a fresh round.

## Worktree And Build Reuse

- Freshness is a property of the reviewer context, not the checkout or build cache.
- Prefer an existing worktree and warm target artifacts when it is pinned to the exact
  review head, its baseline status is known and clean, and no concurrent agent or build
  owns it. Otherwise create an isolated worktree or target.
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
Verify the fix for <finding id: class> on <PR number or branch>.

New head: <full SHA>, pushed. Changed since your round: <files, one per line>.
Re-run your exact reproduction against this head. Report closed or not closed, the
commands you ran, and the head. If the repair adds a mechanism or touches a file you
did not read, say so instead of verifying; that owes a fresh round.
Deliver by: <SendMessage to <name> | final report>.
```

## Finding Discipline

- Every pull request, documentation-only work included, needs at least one compliant
  round before it merges. Only a confirmed in-scope P0/P1 blocks, and the standing
  reviewer's verification closes it, not another round.
- Absent an in-scope P0 or P1 finding, scale rounds to the change. Minor updates, bug
  fixes, and textual changes do not inherently merit another round. A rebase whose
  overlap with the reviewed work is significant, in changed lines or in semantics, may
  merit a fresh round on the intersection; one that only picks up a change clearly
  consistent with, or irrelevant to, the reviewed files does not, and neither does one
  whose only hand-resolved conflicts are generated or digest lines the owning script
  resolves.
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
