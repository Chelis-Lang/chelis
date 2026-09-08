## Context

The pipeline-core extraction shipped three boundary guards and a documentation guard.
Review executed the documentation guard against adversarial strings and confirmed
`false_no_std_claim` is unreliable in both directions:

```text
True   <= "chelis-pipeline-core supports std only, not no_std."   (a TRUE statement, wrongly flagged)
False  <= "chelis-pipeline-core is portable to no_std targets."   (a FALSE claim, wrongly allowed)
False  <= "The core works in a no_std environment."              (a FALSE claim, wrongly allowed)
```

The false positive comes from the positive pattern `supports? ... no_std` firing before
the negation check, whose regex only matches `does not|cannot|is not` and misses a bare
`not no_std`. The false negatives come from a capability vocabulary that omits
`portable to`, `works in`, `runs in`, and similar environment phrasings.

Separately, `scripts/gate.py` is the single source of truth for the per-PR gate. Its
`lint-and-unit` stage runs the workspace build, clippy, fmt, `chelis lint`, the three
rustdoc doctest commands, and the raw-checkpoint compile-fail fixture. It does **not**
run the three pipeline-core guards. A repository-wide search confirms none of the guards
or `compiler_pipeline_oracle.py` is referenced under `.github/`; the oracle is a
documented manual runner. The result: the dependency boundary, the no_std doc contract,
and the cross-crate facade compile-fail fixture are enforced only when a human runs the
oracle.

## Goals / Non-Goals

**Goals:**

- The documentation guard reliably rejects a current no_std support claim regardless of
  adjective, verb, or environment phrasing, and does not reject a true requires-std
  statement or a future-target non-goal.
- The dependency guard, documentation guard, and pipeline-artifact compile-fail fixture
  run in the per-PR gate, so hosted CI enforces them.
- The gate parity test covers the new commands.

**Non-Goals:**

- No new subject matter for any guard: the approved dependency set, required inventory
  sections, and required compile-fail diagnostics are unchanged.
- No `#![no_std]` support.
- The manual oracle stays; it gains no exclusivity and loses no coverage.

## Decisions

### D1: Subject-scoped, negation-first false-claim detection

A prose heuristic over an inventory that legitimately discusses no_std blockers cannot be
fail-closed on every `no_std` mention — the inventory names no_std throughout. The
detector is therefore scoped to lines that assert the **crate subject** has a **present**
no_std **capability**, and it checks negation and future-target framing **before** any
positive match:

1. Normalize the line (lowercase, strip backticks, collapse `#![no_std]` to `no_std`,
   collapse whitespace).
2. If the line carries an explicit negation or future-target frame, it is **not** a false
   claim. Recognize at least: `requires std`, `require std`, `not no_std`,
   `no_std ... not`, `does not|do not|cannot|can not|is not|are not|no longer ... no_std`,
   `future target`, `future capability`, and `not ... current`.
3. Otherwise, the line is a false claim iff it pairs a present-tense capability with
   no_std. Broaden the capability vocabulary to cover the missed forms:
   `is|supports|provides|enables|has|uses|runs in|works in|compiles as|portable to`
   plus the adjective forms `no_std (compatible|ready|supported|capable)`.

This makes false negatives (a real claim slipping through) require a phrasing outside a
deliberately broad vocabulary, and makes the "supports std only, not no_std" false
positive impossible because the negation frame `not no_std` is checked first.

*Alternative considered:* fail-closed rejection of any un-sanctioned `no_std` mention.
Rejected — the inventory is required to discuss no_std blockers extensively, so
fail-closed on any mention would reject the honest inventory the guard exists to protect.

*Primary guarantee stays robust:* the required-statement, required-section, and
required-crate substring checks are exact and are the load-bearing part of the guard. The
false-claim scan is the secondary tripwire; D1 makes it materially better and locks the
review's phrasings, but the exact-substring checks remain the guarantee a reader relies
on.

*Red-team follow-up (fresh-context subagent, 2026-08-04):* an adversarial pass confirmed
the first-cut rewrite had zero false positives but overstated completeness — 14 natural
false phrasings still bypassed it, the largest class being the hyphen spelling `no-std`
(normalization never mapped it to `no_std`, so the `no_std`-absent early return fired) and
forms like `has no_std support` and `is #![no_std]`. Fixed: normalize `no-std` and
`#![no-std]` to `no_std`; add hypothetical framing (`no_std would/could require …`,
`out of scope`) to the negation set; re-add the `no_std support` adjective and an adjacent
`is/are no_std`. A 16-false/12-honest probe now shows 0 false negatives and 0 false
positives, and the spec requirement is reworded to claim coverage of the common phrasings
and both spellings rather than "any" phrasing, with the exact-substring checks named as
the guarantee.

### D2: Enforce the three guards in the per-PR gate, not only the oracle

Add to `STAGES["lint-and-unit"]` in `scripts/gate.py`, in this order after the existing
doctest and checkpoint commands:

- `<managed-python> scripts/pipeline_core_dependency_guard.py`
- `<managed-python> scripts/pipeline_core_documentation_guard.py`
- `<managed-python> scripts/check_pipeline_core_compile_fail.py`

CI already invokes `python3 scripts/gate.py lint-and-unit`, so the additions run in
hosted CI with no workflow edit. The interpreter token matches the existing
`CHECKPOINT_COMPILE_FAIL` entry (`.venv/bin/python`) so a single managed interpreter runs
every Python gate command.

*Cost:* the dependency guard is one `cargo metadata --locked`; the documentation guard is
pure Python; the pipeline-artifact fixture is one `cargo check` on an out-of-workspace
package that reuses the already-built `chelis-compiler-api`. The `lint-and-unit` stage
already builds the workspace (so libpython/pyo3 is present), which is exactly the
precedent the existing checkpoint fixture relies on.

*Alternative considered:* leave enforcement to the manual oracle and document it as
manual. Rejected — the extraction's whole point is a Cargo-inexpressible boundary; a
boundary that only a human runs is one refactor away from silent erosion, and the repo's
"Do Not Trust Green" standard wants the machine to catch it.

### D3: Gate parity covers the new commands

`scripts/test_gate.py` asserts CI hand-inlines no gate command the script does not
produce. Extend it so the three new `lint-and-unit` commands are part of the produced set
and the workflow's `lint-and-unit` job is asserted to call `gate.py lint-and-unit`
(already true), keeping the source-of-truth invariant honest for the additions.

## Risks / Trade-offs

- [The pipeline-artifact fixture adds a fresh out-of-workspace `cargo check` to every PR]
  → It reuses the built `chelis-compiler-api`; the cost mirrors the existing
  checkpoint fixture already in the same stage. If wall-clock is a concern, it stays in
  `lint-and-unit` (already the compile-heavy stage), not a new job.
- [The broadened capability vocabulary could false-positive on an unusual honest line]
  → The negation/future frame is checked first and is broad; the positive tests in this
  change include honest lines (blocker analysis, requires-std, future target) as
  regression anchors, and the exact-substring checks remain the primary guarantee.
- [A prose heuristic is still imperfect] → Acknowledged; D1 does not claim completeness.
  The required-statement check is the guarantee; the scan is a best-effort tripwire whose
  known-miss phrasings are now locked by tests.

## Migration Plan

No data or install migration. The added gate commands run on the next PR. Rollback is a
revert; no persisted state. A legitimately honest current inventory passes the rewritten
guard unchanged (verified by the reviewer-read positive scenario).

## Open Questions

None.
