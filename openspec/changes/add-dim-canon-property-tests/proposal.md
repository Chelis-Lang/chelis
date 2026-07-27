# Proposal: add-dim-canon-property-tests

## Why

`DimExpr::normalized_key` (`crates/chelis-ir/src/dag.rs:283`) is the sound-equality oracle for symbolic dimension expressions. Its rustdoc (`dag.rs:147-161`) makes a semantic claim, not a structural one: the key is "mathematically equivalent for positive-integer-valued dim expressions". Nothing in the test suite checks that claim as a claim.

The consequence of a wrong answer is not cosmetic. `capacity_fits` in `crates/chelis-backend-c/src/memory.rs:315-324` compares concrete element counts when both sides are concrete, and otherwise falls back to `slot_key == req_key`. The HIP backend carries the same fallback at `crates/chelis-backend-hip/src/memory.rs:360-362`. A key that reports two different dimension expressions as equal therefore reuses a buffer slot for a request whose real element count differs, in generated C and generated HIP. `specialize.rs:533` uses the same key to decide whether two dims are interchangeable.

The existing evidence is roughly 50 hand-picked expression pairs across `crates/chelis-ir/tests/dim_canonicalization.rs` and `crates/chelis-ir/tests/dim_canon_adversarial.rs`. Those tests already frame the risk correctly — `dim_canonicalization.rs:277-279` names its fixtures `unsound_a` and `unsound_b` — but an example test only rejects the collisions somebody already imagined. The failure mode that matters is the collision nobody imagined.

The oracle needed to check the claim already exists in the same file. `DimExpr::evaluate` (`dag.rs:171-192`) returns `Err` on division by zero and on inexact division, so "these two expressions denote the same size under this binding" is decidable and unambiguous. The missing piece is a generator, not an oracle.

## What Changes

- New `crates/chelis-ir/tests/dim_canon_properties.rs`: property tests for `normalized_key`, checked against `evaluate` as the reference oracle.
- New dev-dependency on `hegel` (`hegeldev/hegel-rust`) for `chelis-ir`. Property testing is not a new category of dependency in this workspace: `proptest` is already a dev-dependency of `chelis-deep` (`Cargo.toml:16`) and `chelis-surf` (`Cargo.toml:18`).
- The specified property is the **soundness direction only**: equal keys SHALL imply equal values. See Non-Goals for why the converse is deliberately unspecified.
- A generator health check. The property is conditioned on both expressions evaluating successfully, so a generator that mostly produces inexact division would satisfy the property vacuously while testing nothing. The suite SHALL fail when the proportion of usable cases falls below a declared floor, rather than reporting green.
- Counterexamples SHALL be committed as ordinary example tests in the existing `dim_canon_adversarial.rs`, so a found defect stays pinned by a deterministic test and does not depend on a future random draw.
- New `docs/property_testing.md`: how to run the suite, how to reproduce a failure from a recorded seed, and how to promote a counterexample into a committed regression test.

### Non-Goals

- **No completeness claim.** Key inequality does not imply value inequality, and this change does not require that it should. `normalized_key` is deliberately incomplete: `(n * 3) / 2` stays structural because divisibility of `n` is unknown, and the rustdoc says so. Specifying completeness would forbid a normalizer the codebase intends to keep. Only the sound direction is normative.
- **No change to `normalized_key` behavior.** This change adds evidence. If a property fails, the finding is recorded and routed to a separate change; this proposal does not pre-authorize a semantics change to the canonicalizer, because a fix to a soundness hole in dimension equality affects generated backend code and deserves its own review.
- **No replacement of the existing example tests.** `dim_canonicalization.rs` and `dim_canon_adversarial.rs` stay exactly as they are. Property tests and example tests answer different questions, and the example tests document intent that a generator cannot express.
- **No new gate stage, and no new required check.** The suite is an ordinary `cargo nextest` test in `chelis-ir` and runs inside the existing `integration` stage. It needs no workflow.
- **No workspace-wide property-testing programme.** Scope is one function and its helpers.
- **No product behavior change.** Compiler, runtime, CLI, backend, package, and generated-code behavior are untouched. This is a contributor-process change that adds tests.
- **No `spec/**` change.** No normative specification is modified and no existing contract is weakened. The requirements below restate the contract already published in the `DimExprKey` rustdoc; they do not extend it.

## Capabilities

### New Capabilities

- `dim-canon-properties`: what the dimension-canonicalization property suite checks, the direction of the soundness claim, the domain restriction it generates within, the generator health check that prevents a vacuous pass, and how a counterexample is recorded and reproduced.

### Modified Capabilities

None. No existing requirement changes.

## Impact

- `crates/chelis-ir/tests/dim_canon_properties.rs` — new test binary; the authoritative acceptance oracle for this change.
- `crates/chelis-ir/Cargo.toml` — `hegel` added under `[dev-dependencies]`.
- `crates/chelis-ir/tests/dim_canon_adversarial.rs` — gains committed counterexamples if the suite finds any. No existing test is modified or removed.
- `Cargo.lock` — new dev-dependency graph.
- `docs/property_testing.md`, `AGENTS.md` — runbook and a pointer to it.
- `CHANGELOG.md` — contributor-tooling entry.
- No non-test source file under `crates/` is modified. `crates/chelis-ir/src/dag.rs` is read by the tests and not changed.
- Relationship to `add-coverage-baseline-chelis-ir`: independent, and neither blocks the other. That change measures which regions the suite reaches; this one measures whether reaching them proves anything. Its validation record reads `dag.rs` at 84.9% region coverage, which is a reason to expect the canonicalizer is reached, not evidence that its contract is checked.
