# Docs Sync: Lazy List Fusion + SIMD Support

## Summary

Add two new recorded future-work items across the active in-repo design docs:

1. **Lazy list fusion** — a future compiler optimization that fuses chained host-lane list operations (map, filter, fold) into single-pass traversals, eliminating intermediate list allocations. Same principle as tensor DAG fusion, applied to the host lane.

2. **SIMD support plan** — a four-level roadmap for improving CPU vectorization in the C backend, from auto-vectorizer hints (Level 1) through vectorized math library integration (Level 3).

Also import one new standalone design doc:
- `spec/design/chelis_simd_plan.md` — the comprehensive SIMD plan

This is a docs-only sync. No code changes.

## Key Changes

### 1. Import the SIMD plan as a standalone design doc

Create `spec/design/chelis_simd_plan.md` from the downloaded draft.

This doc covers:
- Current state (zero explicit SIMD, auto-vectorization only)
- Level 1: `restrict` pointers, alignment attributes, SIMD pragmas (small effort, ship with next codegen pass)
- Level 2: Hand-written SIMD reductions in the runtime (medium effort, when profiling shows need)
- Level 3: Sleef on Linux + Accelerate vForce on macOS for vectorized math in fused kernels (medium effort)
- Level 4: Full SIMD-width-aware codegen (large effort, only if Levels 1-3 leave gaps)
- Relationship to GPU backends (SIMD is CPU path; GPU backends are separate)

### 2. Update the project plan

Edit `spec/design/chelis_project_plan.md` in place.

Find the "Future Compiler Optimizations" section (under Phase 5, before Ecosystem Library Decisions). It should already have brief notes on lazy list fusion and SIMD. Replace the SIMD paragraph with a more complete summary and add a reference to the standalone SIMD plan doc:

Content for the SIMD entry should cover:
- Four levels (auto-vectorizer hints → SIMD reductions → vectorized math library → full SIMD codegen)
- `restrict` annotation leverages linearity (the type system has proved no aliasing)
- Sleef on Linux, Accelerate vForce on macOS
- Level 1 ships with the next codegen improvement pass
- Level 4 only if profiling demands it
- Full design: `spec/design/chelis_simd_plan.md`

The lazy list fusion entry should already be present. Verify it says:
- Compiler optimization pass, not a user-facing API
- Recognizes chains of `map`/`filter`/`fold` on lists and fuses to single-pass
- Same principle as GHC's `foldr/build` fusion
- Low priority, gated on profiling evidence from Coral string columns or Hull AST processing

### 3. Update the canonical reference

Edit `spec/design/chelis_canonical_reference.md` in place.

Find the Cross-Cutting Design Decisions section. It should already have a lazy list fusion entry. Verify it's present and accurate.

Add a SIMD entry in the same section if not already present:

> **SIMD support (four-level plan).** Level 1: `restrict` + alignment + pragmas in generated C (leverages linearity — the type system proves no aliasing, justifying `restrict`). Level 2: hand-written SIMD reductions in the runtime. Level 3: vectorized math library integration (Sleef on Linux, Accelerate vForce on macOS) for SIMD-width math in fused kernels. Level 4: full SIMD-width-aware codegen (only if Levels 1-3 leave gaps). Full design: `chelis_simd_plan.md`.

### 4. Update the Phase 3 plan

Edit `spec/design/chelis_phase3_plan.md` in place.

SIMD and lazy list fusion are NOT Phase 3 items. They should not appear as Phase 3 sub-phases.

However, if the Phase 3 plan has a section on performance or codegen improvements (e.g., near the existing OpenMP or `static inline` discussion), add a forward reference:

> SIMD support improvements and lazy list fusion are recorded as future compiler optimizations post-Phase 3. See `spec/design/chelis_simd_plan.md` and the project plan's Future Compiler Optimizations section.

If no natural insertion point exists, don't force it. These are Phase 5 / future items, not Phase 3.

## Verification

- `rg -n "SIMD\|simd\|restrict\|Sleef\|vForce"` across all five target docs shows expected coverage
- `rg -n "lazy.*list.*fusion\|foldr.*build\|host.*lane.*fusion"` confirms lazy list fusion is present in project plan and canonical reference
- The standalone SIMD plan doc exists at `spec/design/chelis_simd_plan.md`
- SIMD and lazy list fusion are never described as Phase 3 items or as active/in-progress

## Assumptions

- The repo's current docs are the canonical base; downloaded drafts are source material
- No code changes in this round
- SIMD and lazy list fusion are both future/deferred items, not active development
- The SIMD plan doc is imported as-is from the draft (it's a complete standalone spec)
