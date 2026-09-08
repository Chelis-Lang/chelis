## Context

The numbered chapters `spec/01`–`spec/11` have been captured as OpenSpec capabilities. Capture is not transfer: both trees now describe the same subjects, and nothing states which controls. `spec/design/spec_provenance.md` § OpenSpec boundary says OpenSpec is "not the runtime authority format" and "not an input to canonical graph identity" — a document inside the tree being retired, forbidding the tree replacing it.

## Goals / Non-Goals

**Goals:**

- Define the unit, ordering, and preconditions of authority transfer.
- Make "which tree controls subject X" answerable by reading one thing.
- Keep the legacy chapter controlling by default, so a half-finished migration is safe.

**Non-Goals:**

- Transferring any chapter in this change.
- Requiring a capture to be lossless.
- Deleting, rewriting, or reformatting `spec/**`.
- Deciding whether atom IDs (`[04-NUM-1]`) survive the move; that is downstream of the first transfer and is recorded as an open question.

## Decisions

### 1. The chapter is the unit of transfer

Authority moves per numbered chapter, not per requirement and not in one cutover. A chapter is small enough to review against its capability and large enough that the boundary stays legible.

**Alternative rejected:** Big-bang transfer. It makes the review unbounded and offers no safe intermediate state — every chapter would be ambiguous at once.

### 2. Legacy controls until transfer is recorded

Absence of a transfer record means the `spec/` chapter controls. The default is the status quo, so an abandoned or partial migration degrades to today's behavior rather than to ambiguity.

**Alternative rejected:** Presumption in favor of the capability once captured. Capture is mechanical and reviewable in bulk; that is precisely why it should not by itself move authority.

### 3. Capture may be lossy, but not silently

Requiring losslessness would stall the migration on exactly the cases worth surfacing. The capture pass already found divergences between chapter and implementation — no `wrap_*` builtins exist, `decode_adt_value` has only test callers, jit evaluates as a transparent no-op, and the shipped lexer promotes 24 keywords matching neither §1 nor the PEG grammar. Those are findings, not blockers. What is required is that a known divergence be written down before transfer, so the gap moves with the subject instead of being lost in it.

**Alternative rejected:** A mechanical completeness oracle over chapter text. Nothing available compares prose to requirements without false confidence, and a green check that cannot see meaning is worse than a recorded gap.

### 4. Provenance is a precondition, not a nicety

A capability with no source citation cannot be reviewed against its chapter, so it cannot be transferred. The citation is what makes the migration auditable.

### 5. No transfer while a controlling document forbids it

Stated generally rather than as a one-off task, because the condition can recur: any document still in force that denies OpenSpec authority over a subject blocks transfer of that subject until amended. Today that is `spec_provenance.md` § OpenSpec boundary and the AGENTS.md documentation hierarchy.

## Risks / Trade-offs

- **[The migration stalls half-done, leaving both trees populated indefinitely]** → Acceptable and explicitly designed for: untransferred chapters control, so a stalled migration is today's behavior plus reference material.
- **[Lossy capture loses a normative statement nobody notices]** → Partly mitigated by recording known divergence; not eliminated. Accepted deliberately for now, and the reason transfer is per-chapter and human-reviewed rather than bulk.
- **[Two trees are edited concurrently during the window]** → Supersession marking on transferred chapters, and legacy-controls-by-default before that, keep the answer unambiguous even while both files exist.

## Open Questions

- Do atom IDs (`[04-NUM-1]`) survive into capabilities, move to a new scheme, or stay behind with the legacy chapters? Bears on `spec_provenance.md` and any Buoy integration; deferred until the pilot transfer shows what breaks.
- Does `spec/design/` migrate at all, or only the numbered chapters? The design tree is not captured and is a different kind of document.
