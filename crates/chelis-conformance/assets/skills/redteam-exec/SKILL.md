---
name: redteam-exec
description: Run a compliant Chelis red-team pass. Requires retiring stale, no-longer-needed review handles from the current session and using a fresh local subagent, which may reuse a clean exact-head worktree and warm build artifacts; anything else is blocked, not a valid red team.
---

# Red Team Exec

Use this skill when the user asks for a red team, adversarial review, or a fresh-context
validation pass.

## Repository Contract

1. Inventory review and subagent handles created in your own current session. Stop or
   interrupt and retire only stale or failed handles that will not be used again; do not
   disturb another developer's handles or a handle reserved for follow-up work. The
   platform does not need to support deleting a retired handle from its listing.
2. Spawn a new local subagent with fresh context for the validation pass.
3. Give that fresh subagent a reusable exact-head worktree and its warm target cache when
   one is clean, idle, and available. Fresh review context does not require a fresh
   checkout or cold build.
4. If the built-in subagent path routes to remote infrastructure, errors, or is otherwise
   broken, retry with another fresh local subagent path.
5. Do not substitute CLI fallback or main-thread validation and call it a red team.
6. If every fresh-local subagent path is unavailable, state that red-team validation is
   blocked.

## Preferred Execution Order

1. Use the platform's agent inventory to find stale or failed handles from your own
   current session that will not be used again. Stop or interrupt those handles, then
   treat them as retired and do not reuse them; leave all other handles alone.
2. Spawn a fresh local subagent via the platform tool when it is actually local and
   working. In Codex sessions, use `list_agents`, `interrupt_agent`, `spawn_agent`,
   `send_message` or `followup_task`, and `wait_agent` rather than shelling out to
   `claude`, `codex exec`, or other external CLIs.
3. Hand the subagent the selected worktree path, exact commit, baseline status, and
   target path so it can reuse compiled artifacts safely.
4. If that path is broken, stop or interrupt and retire the failed handle, then retry
   with another fresh local subagent.
5. Only count the review as a red team when the fresh-context subagent actually ran the
   commands and reported findings.

## Worktree And Build Reuse

- Freshness is a property of the reviewer context, not the checkout or build cache.
- Prefer an existing worktree and warm target artifacts when it is pinned to the exact
  review head, its baseline status is known and clean, and no concurrent agent or build
  owns it. Otherwise create an isolated worktree or target.
- Reusing a worktree must not relax exact-head verification, adversarial execution, or
  restoration proof. Restore temporary tests, fixtures, and mutations after the pass
  and report the final worktree status unless the user explicitly asks to retain them.

## Explicit Non-Goals

- Do not use external agent CLIs as a substitute for the built-in local subagent path
  unless the user explicitly asks for that toolchain.
- Do not treat a remote 404 / deployment error as a successful spawn. Stop or interrupt
  and retire that handle, then retry or report the red team blocked.

## Minimum Deliverable

- findings ordered by severity
- exact commands run
- coverage against the active spec and acceptance oracle
- exact reviewed commit and final worktree status
- explicit note of anything unvalidated

## Validation Discipline

- run code and commands, not just source inspection
- add adversarial probes where coverage is thin
- verify positive and negative cases
- check examples, docs, and CLI behavior against shipped behavior
