# `/red-team`

Use the shared `redteam-exec` skill. The word after the command selects the mode:
`verify` sends a fix to the standing reviewer; anything else, or nothing, starts a fresh
round.

## `/red-team` (fresh round)

1. Confirm the head is committed and pushed and that CI is running on it. Count the
   fresh rounds this pull request has had: the default cap is three, one for a prose-only
   pull request, and a round past the cap needs the user's explicit approval before you
   continue.
2. Inventory subagent handles created in your own current session. Stop or interrupt
   and retire only stale or failed handles that will not be used again. A standing
   reviewer awaiting a fix is neither; do not disturb it or another developer's handles.
3. Pick the worktree and target: a clean, exact-head, idle worktree and its warm target
   when one exists, otherwise a new isolated one. Decide whether the target is free.
4. Fill the round brief template from the skill: head SHA, files, in-scope claims,
   worktree, the target with the output of
   `.venv/bin/python scripts/worktree_status.py --path <target>` pasted verbatim and the
   time you took it, the report-length budget, and delivery channel. Context budgets
   and time limits are optional, with no default; include them only when set. Give a
   reviewer whose probes will mutate tracked source its own worktree.
   Do not tell the reviewer to read `AGENTS.md`.
5. Spawn a **new local subagent with fresh context** with the brief as its prompt. In
   Codex, use `list_agents`, `interrupt_agent`, `spawn_agent`, `send_message` or
   `followup_task`, and `wait_agent` rather than shelling out to `claude`, `codex exec`,
   or other external agent CLIs. If the spawn routes to remote infrastructure or errors,
   retire that handle and retry, or state that the red team is blocked.
6. Record the initial report in the pull request: head, verdict, commands, finding
   classes and scope classifications, accepted-no-action observations, linked issues,
   residual scope, and any explicit deadline. Keep the reviewer alive through the
   local repair loop.

## `/red-team verify`

1. Consolidate the current round's repairs into a local commit. Name the finding being
   closed, its class, the exact unpushed head SHA, and the files changed since the
   initial report.
2. Hand off the worktree, rerun `worktree_status.py`, and send the local verification
   brief from the skill to the standing reviewer. Do not push while the round is open.
3. Repeat local repair and verification until the reviewer is satisfied, then push the
   exact verified head once. A confirmed in-scope P0 or P1 earns a fresh round after
   that push, subject to the cap.
4. Record each result under the same round in the pull request. If the reviewer is gone,
   rerun its exact reproduction yourself and record the result as "reviewer
   unavailable"; this does not replace an owed fresh round after a confirmed P0 or P1.
5. When two consecutive rounds report the same finding class, or replace a repaired
   finding with a different class, stop patching witnesses and change the
   representation, the oracle, the claim, or the brief.

Do **not** fall back to main-thread validation and call it a red team. A round counts
only when the fresh-context local subagent actually ran the validation work and
delivered findings and evidence.
