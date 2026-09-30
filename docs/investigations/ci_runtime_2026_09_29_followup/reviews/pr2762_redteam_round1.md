# PR #2762 — red-team round 1

**Reviewed head:** `20f726163a2b0aa942725f6f347da98d78b43b1f` on `perf/extent-export-preparation-20260929`.
**Verdict:** Satisfied. No in-scope P0–P2 or P3 findings. No out-of-scope defect identified.

## Execution and coverage

- `cargo test -p chelis-cli --test runtime_extent_claim_preparation claimed_extent_contract -- --exact --nocapture` — PASS: 55 cases, zero unmet contract cells; six C binaries compiled and six checked builds reused. Both `named.export.satisfied` and `named.export.mismatch` executed and met their separate value or trap/context expectations. The existing matrix also covers the other exported good/bad pairs and non-export routes.
- `cargo test -p chelis-cli --test runtime_extent_claim_preparation literal_result_claim_contract -- --exact --nocapture` — PASS: six positive/negative export, binding, and root cases, with independent checker and execution assertions.
- `cargo test -p chelis-cli --test runtime_extent_claim_preparation_redteam_probe redteam_cache_boundary_probe -- --exact --nocapture` — PASS on a temporary untracked copy of the test file. Its C caller printed its PID. Identical Surf source and caller with changed scalar bits reused preparation, returned the changed expected values, and ran in a different PID. The mismatching extent reused preparation but produced its own required trap and context. Changing caller rank produced a second linked binary and preparation without a reuse hit; adding a blank line to Surf source produced another checked build without a reuse hit. Binding and root cases passed without entering the export preparation cache. The probe file was removed.
- Inspected `observe_host_lane`, `dynamic_driver`, `ExportBinaryCache`, `contract_failures`, the PR body, and C5 in `spec/design/runtime_extents.md`. The C5 sentence accurately describes the bounded preparation suite: reuse needs identical Surf source and generated caller; each runtime input gets `Command::new(...).output()` and its own observation/contract check; other routes follow their own build path.

## Limits and status

The remaining preparation tests, hosted CI, the phase-final oracle, and HIP/Metal hardware receipts were not run by this reviewer. The supplied gate-fast result was treated as prior evidence; this review's exact-head acceptance rests on the commands above.

Final status: `20f726163a2b0aa942725f6f347da98d78b43b1f`, clean (zero modified, staged, untracked, or unmerged files); `worktree_status.py` reported FREE with no scoped process at 2026-09-29 16:00:52 UTC. Available for same-reviewer repair verification if needed.
