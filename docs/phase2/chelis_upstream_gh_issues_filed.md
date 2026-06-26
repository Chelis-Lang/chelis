# FlukeBall Upstream Issues — Resolution Tracker

Filed from the FlukeBall Phase 2 hardening review. All 13 issues resolved 2026-06-25.

## Chelis (Chelis-Lang/chelis)

| Issue | Title | PR | Status |
|-------|-------|-----|--------|
| #486 | Diagnostics: make common agent-repair failures actionable | #512 | Merged |
| #487 | prove: resolve Reef package imports or provide a package proof mode | #509 | Merged |
| #488 | prove: expose machine-readable capabilities and JSON schema metadata | #509 | Merged |
| #489 | prove: emit agent-oriented failure summaries and structured tier degradation | #510 | Merged |
| #490 | prove: expose property dependency or attached-target metadata for admission policies | #510 | Merged |
| #491 | reef: export pinned package bundles for hermetic agent sandboxes | #508 | Merged |
| #492 | Export machine-readable ABI and package schema for authoring harnesses | #508 | Merged |

## Whale (Chelis-Lang/whale)

| Issue | Title | PR | Status |
|-------|-------|-----|--------|
| #10 | Expose machine-readable opaque-domain constructor metadata | #11 | Merged |

## Beacon (Chelis-Lang/beacon)

| Issue | Title | PR | Status |
|-------|-------|-----|--------|
| #22 | Expose BeaconShim protocol/version compatibility preflight | #31 | Merged |
| #23 | Add a proof-profile linter for bounded Beacon roots | #32 | Merged |
| #24 | Normalize Beacon discharge status and soundness fields | #31 | Merged |
| #25 | Emit proof bundles with root, source, input-box, and toolchain provenance | #32 | Merged |

## Live Blocker (resolved)

The Beacon `arb-oracle` feature could not link due to cached `libflint.a` and
`libarb.a` built without `-fPIC`. Fixed by deleting `~/.cache/flint-sys/` and
`~/.cache/arb-sys/`, then rebuilding with `CFLAGS="-fPIC"`. The binary now
reports `dispatch_shim_default_verified_arb_box: true`.

## Advisory follow-ups

- chelis #510: `collect_expr_refs` in property dependency extraction skips
  `Match`, `Record`, `Grad`, and other compound expressions. Non-blocking
  (best-effort extraction) but could miss edges in complex properties.
- chelis #508: `reef schema` subcommand has no CLI integration test coverage.
- beacon #24: `NormalizedDischarge` derives `Deserialize`; a crafted JSON could
  bypass the transport/verdict invariant. In practice beacon only serializes,
  never deserializes external normalized-discharge. Consider a validating
  deserializer for defense-in-depth.
