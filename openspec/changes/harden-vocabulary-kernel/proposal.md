# Proposal: harden-vocabulary-kernel

## Why

Three hardening mechanisms are queued for this repository and none of them has a pilot:

- **FCIS boundary enforcement.** Barnacle's `adopt-fcis-boundary-lints` change specs a `fcis_safety` Dylint library and states that the Chelis-side adoption "happens in Chelis' own change once this library ships." That change does not exist.
- **`no_std` purity.** Zero crates in this workspace declare `no_std`. There is no worked example of what a pure Chelis leaf crate looks like.
- **Mechanized verification.** Lean formalization is Phase `5h` (`spec/design/chelis_project_plan.md:1266`). It has no executable path, no candidate crate, and no acceptance oracle.

Each is currently argued at whole-compiler scale against 328k lines of Rust across 28 crates. None has been demonstrated end to end on anything. A pilot on the smallest real crate converts three open-ended architecture arguments into one reviewable change, and produces the toolchain and gate wiring the larger crates would otherwise each invent.

**`chelis-vocab` is that crate.** 195 lines, one file, `#![forbid(unsafe_code)]`, and its `Cargo.toml` has **no `[dependencies]` section at all**. It is also genuinely load-bearing rather than a toy: it owns the `RuntimeDType` tag that crosses the C ABI and the `EffectKind` vocabulary the checker and effect engine share.

### The crate already fails its own stated contract, in three mechanically detectable ways

Its module doc reads: *"This crate deliberately has no dependencies and no fallback vocabulary variants. It is the single authority for identifiers that cross compiler, runtime, and generated-code boundaries."*

**1. "No dependencies" is true of Cargo and false of `std`.** `use std::error::Error`, `use std::fmt`, `String`, and `format!` all appear. The claim is about the dependency graph; the reader takes it as a purity claim. Nothing enforces either reading.

**2. The crate contains a code generator.** `render_runtime_dtype_c_header() -> String` (`lib.rs:174`) builds C source text with `format!` and `String::push_str`. A closed vocabulary is data plus total decode functions; emitting a file is a shell concern. This is a textbook functional-core/imperative-shell violation sitting in the crate whose entire purpose is to be the pure bottom of the graph. It has exactly one consumer — `crates/chelis-runtime/tests/runtime_dtype_generated_header.rs` — so the split is cheap.

**3. The closed-vocabulary invariant is enforced by substring matching.** `crates/chelis-cli/tests/closed_vocabulary_architecture.rs` asserts required and forbidden *string literals* against consumer source files:

```rust
required: &["use chelis_vocab::EffectKind", "decode_effect_kind("],
forbidden: &["EffectKind::from_symbol", ".and_then(EffectKind"],
```

This is blind to semantics, passes on a comment containing the right words, and fails on reformatting. Barnacle's `dispatch_safety_wildcard_enum_match` is the semantic form of precisely this rule — it rejects a catch-all arm over a configured enum, which is what "no fallback vocabulary variants" means when stated mechanically rather than in prose.

### One further defect the same pass closes

`byte_width(self) -> usize` (`lib.rs:130`) returns an **architecture-sized integer for an ABI quantity**. The byte width of `f32` in the Chelis runtime ABI does not vary with the host pointer width, and the value flows into `chelis-runtime`'s allocation path (`lib.rs:220`). Barnacle's `tiger_style_architecture_sized_integer` names this directly. `ground-tensor-extent-arithmetic` makes the same width-chain argument one level up and is where the general form of this rule belongs; this change fixes the instance in the crate it is piloting on.

### Why this is a pilot and not a one-off

The point is not that `chelis-vocab` is dangerous. It is not — it is 195 lines of closed enums. The point is that the four mechanisms under consideration have never been run against anything in this repository, and their real costs (a second Rust toolchain for Dylint, a Lean toolchain, the `no_std` split discipline, the FCIS scope manifest) are currently estimates. Paying them once on the smallest crate makes them measured.

## What Changes

- **`chelis-vocab` becomes `#![no_std]` unconditionally — not behind a `std` feature.** `std::error::Error` moves to `core::error::Error`; `std::fmt` to `core::fmt`. The crate SHALL NOT depend on `alloc` — which requires removing the one allocation in the decode path (below) and relocating the code generator. See "Why no feature flag" below.
- **`EffectKindDecodeError::Unknown` stops allocating.** It currently owns `String` via `symbol.to_owned()`. It becomes a borrow of the input lifetime, which removes the only heap allocation reachable from `EffectKind::decode` and makes the decode path total and allocation-free.
- **The C-header generator moves out of the vocabulary crate.** `render_runtime_dtype_c_header` relocates to a shell-side location; its single test consumer follows. The vocabulary crate retains the data the generator reads.
- **Barnacle is adopted, scoped to this crate.** A `dylint.toml` configures `dispatch_safety` over `EffectKind` and `RuntimeDType`, and `tiger_style_architecture_sized_integer`. The substring-matching architecture test is replaced by the semantic lint where the lint subsumes it, and retained where it does not.
- **`byte_width` returns a fixed-width integer.** `usize` becomes `u32`, with its consumers converted at the call site.
- **An Aeneas/Charon translation gate over the numeric ABI core.** `RuntimeDType::{id, decode_id, byte_width}` and `ALL` become a Charon-extractable, Aeneas-translatable subset, with the round-trip and exhaustive-rejection properties proved in Lean. See the design document for the honest scope boundary — the `&'static str` accessors and `EffectKind::decode`'s string matching are **not** in the verified subset.
- **A bounded pilot surface for chelis#803 (cargo-llvm-cov) and chelis#808 (cargo-mutants).** Both are staged rollouts with captured baselines and four sub-issues each. This change does not re-propose them; it gives each an early instance small enough to run in seconds, and scopes them so they measure what the proof does not. See "Layered evidence" below.
- **A `[lints]` table.** There is no `[workspace.lints]` and no crate declares one, so `cargo clippy -- -D warnings` runs default groups only across all 28 crates. `clippy::cast_possible_truncation` and `clippy::arithmetic_side_effects` are off. This change adds the table for `chelis-vocab`; whether it goes workspace-wide is a separate argument.
- **An exhaustive check over the whole `i32` tag domain**, and **a differential test against the generated C decoder** — the two cheaper oracles that turn out to matter more than the proof does. See below.
- **Compile-time layout assertions** pinning `RuntimeDType`'s size and discriminants, so `repr` drift fails the build rather than a test.

### The proof is not what establishes the headline property

An earlier draft of this change justified the Aeneas work on exhaustive rejection: `∀ i : i32`, the decoder rejects every non-tag — "the property enumerated tests cannot establish," quantified over 2^32.

That justification is wrong, and correcting it is more useful than defending it. `i32` has 4.29 billion values and `decode_id` is a nine-arm match over constants. A release-mode loop over the entire domain runs in seconds. **The property is establishable by brute force**, and an exhaustive check is not a sample — it is a complete proof by exhaustion, carrying the same guarantee as the Lean theorem.

So the exhaustive check is the default oracle, per `route-proof-obligations`' cheaper-oracle rule, and this change adds it regardless of what happens to the proof lane. It is also the honest fallback if the Aeneas lane is declined: without it, declining would leave the gap open with nothing filling it.

The Aeneas work is re-justified accordingly, and narrowly: it exists to **establish and measure the prover lane** on the smallest crate that can carry it. That is a legitimate reason, it is the reason `route-proof-obligations` Phase 3 needs a first obligation, and it is materially weaker than the claim it replaces. The documentation must say so, and the anti-overclaim checker enforces it.

### Why no feature flag

The conventional pattern is `#![cfg_attr(not(feature = "std"), no_std)]` with an additive `std` feature. It is the wrong choice here, for four reasons specific to this crate:

- **The motivating need is gone.** A `std` feature usually exists to restore `std::error::Error`. `core::error::Error` has been stable since Rust 1.81, so the `Error` impls work unconditionally.
- **Nothing else needs `std`.** After the generator relocates and the decode error borrows, no item in the crate has a `std`-only dependency. A feature gating nothing is ceremony.
- **It would multiply this change's own cost by two.** A feature flag means two build configurations, and this change is piloting four mechanisms across them: the lint job, coverage, mutation, and Aeneas extraction. Feature-conditional compilation is also a documented source of confusion for coverage instantiation counts (chelis#803) and for mutation viability classification (chelis#808). Doubling the matrix to serve a configuration nobody consumes is a poor first pilot.
- **There is no consumer in the other world.** Feature flags earn their keep on published library crates whose users may be embedded or hosted. `chelis-vocab` is an internal workspace crate; every consumer is hosted.

If a future consumer genuinely needs `std`-only behavior here, adding the feature then is a small, reversible change. Adding it now is speculative generality of exactly the kind that makes a pilot unrepresentative.

**A note on the broader question.** `no_std` is tempting as the mechanical marker for "this is functional core." It is a useful signal — it blocks `std::io` and `HashMap` for free — but it is **necessary, not sufficient**. `no_std` does not prevent global mutable state, does not prevent panics, and with `alloc` does not prevent `format!`. Treating it as the FCIS boundary would be a mechanism that reads stronger than it is, which is the failure mode this change repeatedly guards against. The actual boundary enforcement belongs to Barnacle's `fcis_safety` protected-scope manifest, driven by `establish-pure-compiler-core`. `no_std` is corroborating evidence, not the contract.

### Layered evidence, not stacked tools

The three mechanisms answer different questions, and the change is only worth the toolchain cost if they are scoped so they do not answer the same one twice.

| mechanism | question | scope here |
| --- | --- | --- |
| exhaustive `i32` check | is the property true for every input? | `decode_id` — the whole tag domain, by enumeration |
| differential test | does the emitted C decoder agree with the Rust one? | the ABI boundary the proof does **not** cross |
| compile-time assertions | has the layout drifted? | `size_of`, discriminant-to-C-macro equality |
| Lean proof | same question as the exhaustive check, by a different route | the numeric ABI core — present to establish and measure the lane |
| cargo-llvm-cov | did the suite execute this code? | runs whole-crate; **reported** for the proof's complement |
| cargo-mutants | would the suite notice if this code were wrong? | whole crate, before and after |
| `[lints]` table | is a lint class simply switched off? | the crate, at compile time |

Once exhaustive rejection is proved over all 2^32 tag values, line coverage of `decode_id` is strictly weaker information about the same function. Running coverage there and reporting a percentage would dilute the stronger result. So coverage is scoped to the items the proof deliberately excludes, which is exactly where it earns its place — and gives chelis#803 a worked example of coverage as a complement to proof rather than a blunt workspace percentage.

Mutation testing is the one that pays immediately, before any proof exists. `crates/chelis-vocab/tests/closed_vocabulary.rs:90` probes a **single** unknown tag id. A mutant that narrows `decode_id`'s rejection range should survive that test. Running cargo-mutants first therefore produces an empirical demonstration of the gap the proof closes, and re-running it after gives a before/after measurement — which is stronger evidence for the whole verification argument than the proof alone.

### Aeneas is the strategic lane; Verus is the specialist lane

`prove-dim-canon-overflow-safety` proposes a **Verus** spike over `usize_gcd`'s termination and the canonicalizer's replacement multiplication. This change proposes **Aeneas**. Both are adopted; `route-proof-obligations` owns the routing rule and the conditions each lane must satisfy, so this change does not restate them.

What matters here is why this obligation lands in the Aeneas lane, and it is settled by the sibling change's own D5: `normalized_key` is inexpressible in Verus because it carries `String`, `Vec`, and `Box` and recurses over a heap-allocated enum. That shape is Aeneas' central case, not an awkward one. The split is a measured property of the code rather than a preference.

The two obligations differ accordingly:

| | `prove-dim-canon-overflow-safety` (Verus) | this change (Aeneas) |
| --- | --- | --- |
| obligation | `usize_gcd` termination; multiplication cannot fold a wrong constant | encode/decode round-trip, injectivity, exhaustive rejection |
| shape | arithmetic with pre/postconditions over machine integers | total functions over a closed nine-variant ADT |
| tool fit | SMT-backed contracts on existing Rust, in place | translation to a pure functional model |
| toolchain | Verus' own fork, new | Lean 4.29.0, **already installed and green locally** |

That last row is the asymmetry worth stating plainly: the Verus spike proposes a toolchain the repository does not have, while the Lean toolchain already exists here because LaCaDiLE runs on it. It is also why Aeneas is the strategic lane — its output lands in the same logic as LaCaDiLE and `spec/design/verification_stack_sketch.md`'s L5 ladder, so an implementation proof and a metatheory proof can eventually meet. A Verus contract composes with nothing outside Verus.

**What this change commits to.** The Aeneas obligation, its kill criterion, and the isolation condition that the repository must build and test with neither Charon, Aeneas, nor Lean installed. `route-proof-obligations` owns the lane policy; this change owns one obligation inside it. If this spike fails its criterion, the `no_std` / FCIS / Barnacle work still lands — the phases are already ordered for that. Adoption of both lanes SHALL NOT be read as establishing a verified-code programme for Chelis, and an unverified component is not thereby a defect.

### What to take from LaCaDiLE

LaCaDiLE (`../LaCaDiLE`) is a green Lean 4 mechanization: ~86k lines across 32 files, ~1,460 theorems, ~530 definitions, **zero `sorry`/`admit`**, and ~60 negative `not_…` theorems that pin the supported fragment from the outside. It ships an acceptance oracle, `scripts/check_final_proof.py`, running seven audits.

**Take the oracle architecture. Do not take the proofs.** LaCaDiLE proves metatheory about the LaCaDiLE calculus — progress, preservation, dimension safety, effect correctness, AD correctness. None of that is about `RuntimeDType`'s ABI tag, and importing any of it here would be cargo-culting. The oracle, by contrast, solves a problem this change already has and currently answers only in prose.

Four of the seven audits are generic and port directly:

| LaCaDiLE audit | what it does | why this change needs it |
| --- | --- | --- |
| admission tokens | no `sorry` / `admit` in Lean source | a proof with an admission is not a proof; nothing here checks that today |
| banned declaration forms | no `axiom`, with one **explicit reviewed whitelist** for what Mathlib forces | same shape as this repo's narrowing-citation rule — forbid, then justify each exception at its site |
| required proof-surface | named theorems exist **and are non-vacuous** | the anti-vacuity half is the point; a theorem that typechecks but says nothing passes a naive existence check |
| banned doc snippets | documentation **cannot assert more than Lean proves** | this is the executable form of three requirements this change currently states as prose |

That last one is the highest-value adaptation and the reason this section exists. LaCaDiLE's own notes call it "the project's immune system: it makes it mechanically impossible for the prose to drift ahead of the Lean." This change's spec already requires that no text describe the crate as verified, that each evidence mechanism report at its own strength, and that residual unproved boundaries be named. Those are currently three prose requirements enforced by review. LaCaDiLE demonstrates the same class of requirement enforced by a script with its own unit tests.

**Explicitly not taken:** the ~7,450-line body of `check_final_proof.py` is dominated by hardcoded tables of LaCaDiLE theorem names, declaration shapes, and doc snippets. Those are its content, not its design. Adapting means writing a small checker with the same four mechanisms over this change's surface, with its own tests per the repository's Python-with-tests policy — not vendoring or generalizing LaCaDiLE's.

**A toolchain consequence.** Lean 4.29.0 is already installed and working on this workstation via elan, with Mathlib. The Lean half of the verification toolchain is therefore **not a new cost**, which materially improves this change's pin budget. Phase 0 must still check whether Aeneas's Lean pin and `Aeneas.Std` are compatible with 4.29.0 — elan resolves toolchains per Lake project, so divergent pins can coexist, but a skew is worth discovering before Phase 5 rather than during it.

LaCaDiLE also ships `scripts/prove.py`, an automated proof-filling loop driving Leanstral through `lean-lsp-mcp`. If it applies to Aeneas-generated obligations, the proof-engineering estimate in this change drops substantially. Phase 0 probes it; the change does not depend on it.

### The residual the proof cannot reach, and the test that can

The generated C header carries its own decoder — a `switch` with an `abort()` default — emitted by `render_runtime_dtype_c_header`. Proving the Rust decoder total says nothing about the C one, and the existing test compares the two **as text**, which is evidence of agreement rather than a check of behavior.

A differential test closes it: run the Rust `decode_id` and the compiled C `chelis_runtime_dtype_size_checked` over every valid tag and a sample of invalid ones, and require agreement. That converts a named residual into a checked one, and it is the only mechanism in this change that crosses the language boundary the ABI actually lives on.

This is worth more than the proof. The proof establishes something an exhaustive loop also establishes; the differential test establishes something nothing else in the repository does.

### Miri is deliberately excluded

Miri detects undefined behavior: out-of-bounds access, use-after-free, invalid transmutes, uninitialized reads, aliasing violations, data races. `chelis-vocab` declares `#![forbid(unsafe_code)]`, contains no raw pointers, no FFI, no transmutes, and after Phase 1 no allocation. **Miri would find nothing here**, and adding it would be evidence theater — a green result that reads as a safety claim while testing for a defect class the crate cannot contain.

It would also cost a fourth toolchain pin. Miri is nightly-only, and this change already carries `stable` plus Dylint's `nightly-2026-04-16` plus a possible Charon pin, which Phase 0.2 flags as the primary re-scoping risk.

The real Miri target is `chelis-runtime`: 4,162 lines, unsafe-heavy, raw-pointer arithmetic throughout, the C ABI boundary on the other side of the very tag this change proves, and 45.67% line coverage in chelis#803's baseline — the fourth-lowest crate in the workspace. That is a genuine and probably valuable change. It is a different one, it has no sibling issue today, and it should be filed rather than folded in here. Note when filing that Miri interprets Rust only, so the C side of that boundary stays uncovered by it.

### The decision this change must record

**Whether a second Rust toolchain enters this repository's gate.** Dylint requires `nightly-2026-04-16` with `rustc-dev`; `rust-toolchain.toml` pins `stable`. Barnacle solves this with a Nix flake and runs `nix flake check` as its only CI command. Chelis runs `python3 scripts/gate.py`. Three options, to be decided in Phase 0 and not presumed here:

- **Option A — advisory, out of gate.** Barnacle lints run in a separate optional CI job. Cheapest; no pin coupling. Weakest: an advisory architecture lint is one that eventually fails and gets muted.
- **Option B — gating, separate job, Nix-provided toolchain.** A dedicated CI job supplies the nightly via Nix, as Barnacle does. The developer-runnable `scripts/gate.py --local` does not change. Costs a Nix dependency in CI.
- **Option C — gating, in `scripts/gate.py`.** Every developer needs the nightly. Highest friction; contradicts the local-gate design in `docs/local_macos_environment.md`.

This proposal recommends **Option B**, because `scripts/test_gate.py` already asserts CI hand-inlines no command the gate script does not produce, and a Nix-provided lint job is the smallest addition that keeps that invariant intact. Phase 0 records the decision either way.

### Non-Goals

- **Not the FCIS compiler core.** `establish-pure-compiler-core` (recovered on `rescue/fcis-2026-07-16`) is the real functional-core work. This change is its pilot, not its replacement, and SHALL NOT presume its design.
- **Not general extent or width hygiene.** `ground-tensor-extent-arithmetic` owns the width chain and the `Numel`/`Extent` types. This change fixes one `usize` inside the crate it is piloting on and adds no general rule.
- **Not a Lean formalization of the type system.** Phase `5h` and the LaCaDiLE metatheory prove the typing *rules* sound. Aeneas here proves a shipped Rust *implementation* matches its model, over nine enum variants. Different rung of the `spec/design/verification_stack_sketch.md` L5 ladder; this one does not advance the other.
- **Not importing LaCaDiLE's proofs, model, or Lake project.** Only the oracle's four generic audit mechanisms are adapted, reimplemented against this change's surface. This change SHALL NOT add a dependency on the LaCaDiLE repository.
- **Not a `std` feature flag.** See above. Unconditional `no_std`; revisit only if a real consumer appears.
- **No new vocabulary variants, and no change to the ABI tag values.** `RuntimeDType`'s discriminants are a shipped C ABI. They do not move.
- **Not a `no_std` policy for the workspace.** One crate. Whether any other crate follows is a later argument with its own evidence.
- **Not Miri.** See above. The crate cannot contain the defect class Miri detects. `chelis-runtime` can, and warrants its own change.
- **Not the chelis#803 or chelis#808 rollouts.** Both own staged plans, captured baselines, four sub-issues each, and their own parent acceptance oracles. This change consumes them on one crate and SHALL NOT alter their thresholds, sharding, exclusion policy, or enforcement decisions. If the pilot produces evidence relevant to those decisions, it is reported to those issues, not settled here.
- **Not the verification-lane policy.** `route-proof-obligations` owns the routing rule, the per-lane adoption conditions, the double-counting prohibition, and the one-time calibration. This change discharges one obligation inside that policy and SHALL NOT restate or amend it.
- **Not a verified-code programme.** Adopting two lanes creates no obligation to verify existing or new code.
- **Not new mutation or coverage infrastructure.** `add-mutation-baseline-dim-canon` defines `scripts/mutants.py`, the committed-baseline convention, the fail-closed rules, and the workflow classification. This change consumes that shape and SHALL NOT fork it.

## Capabilities

### New Capabilities

- `vocabulary-kernel-purity`: the requirement that the dependency-bottom vocabulary crate be free of the standard library and of allocation, that code generation not live inside it, that its stated purity claims be mechanically enforced rather than asserted in a doc comment, and that its lint configuration be declared rather than inherited by default.
- `vocabulary-abi-roundtrip`: the requirement that the runtime dtype tag's encode/decode pair be total, injective, and exhaustively rejecting, that this be established by mechanized proof rather than by enumerated tests over the nine known variants, and that each evidence mechanism's guarantee be reported at its own strength rather than aggregated.

### Modified Capabilities

None. No existing requirement changes. The closed-vocabulary behavior this change hardens is currently stated only in a module doc comment and a substring-matching test; there is no requirement text to modify.

## Impact

- `crates/chelis-vocab/src/lib.rs` — `no_std`, borrowed decode error, `byte_width` width, generator removal.
- `crates/chelis-vocab/Cargo.toml` — remains dependency-free; gains no `[dependencies]` section.
- `crates/chelis-vocab/tests/closed_vocabulary.rs` — asserts the borrowed error and the new `byte_width` type; gains the exhaustive `i32` check and the compile-time layout assertions.
- `crates/chelis-vocab/Cargo.toml` — gains a `[lints]` table. No `[dependencies]` section is added.
- A differential test against the compiled generated header, sited with the generator after Phase 2.
- `crates/chelis-runtime/src/lib.rs:220` — `byte_width` call-site conversion.
- `crates/chelis-runtime/tests/runtime_dtype_generated_header.rs` — follows the generator to its new home.
- `crates/chelis-cli/tests/closed_vocabulary_architecture.rs` — the `EffectKind` / `RuntimeDType` substring assertions that `dispatch_safety` subsumes are removed; the rest stay, with a comment recording which lint replaced what.
- `crates/chelis-backend-metal/tests/dtype_abi_width_parity.rs` — **currently untracked in the working tree** and reads `byte_width` at two sites. Coordinate before landing; do not overwrite.
- `dylint.toml` (new, repository root) — `dispatch_safety` and `tiger_style` configuration, scoped to this crate.
- `.github/workflows/` — one new lint job under Option B.
- `scripts/gate.py`, `scripts/test_gate.py` — unchanged under Option B; the test asserting CI inlines no unknown gate command must continue to pass with the new job present.
- `spec/` — no change. The vocabulary contract is an implementation invariant, not a language-surface one; if Phase 0 finds it belongs in `spec/`, that is a follow-up.
- Relationship to Barnacle's `adopt-fcis-boundary-lints`: that change ships `fcis_safety`. This change consumes `dispatch_safety` and `tiger_style`, which ship today. It is therefore **not blocked** on the Barnacle change, and deliberately does not adopt `fcis_safety` — the FCIS scope manifest belongs with `establish-pure-compiler-core`.
- Relationship to `ground-tensor-extent-arithmetic`: independent. That change explicitly declares "No Barnacle dependency" as a non-goal and argues `tiger_style_architecture_sized_integer` would not have caught chelis#888. Both statements remain true; this change does not contradict them.
- `.cargo/mutants.toml` — created by chelis#811 if that lands first; otherwise this change creates it scoped to `chelis-vocab` only, and hands it over. Coordinate; do not fork the configuration.
- Relationship to chelis#803 / chelis#808: this change is a **consumer**, and both branches (`agent/803-cargo-llvm-cov`, `agent/808-cargo-mutants`) are currently empty placeholders. If either rollout lands its runner and configuration first, this change adopts it rather than building a parallel one. If this change lands first, its per-crate invocation is written so the rollout can absorb it.
- Relationship to `prove-dim-canon-overflow-safety`: the paired verification spike. Different lane, different obligation, no dependency in either direction. Its D5 supplies the empirical reason the lanes are split.
- Relationship to `route-proof-obligations`: that change owns the lane policy this one operates under. It also names `usize_gcd` as the single calibration obligation deliberately discharged in both lanes; this change's obligation is **not** part of that overlap.
- Relationship to `add-mutation-baseline-dim-canon`: this change **adopts its conventions** rather than inventing parallel ones — the Python wrapper shape, the committed-baseline discipline, fail-closed on a zero-mutant run, and survivor classification against coverage. It also inherits that change's sequencing rule (coverage before mutation), which reordered this change's phases.
- `scripts/test_gate.py` — `NON_GATE_WORKFLOWS` must gain any new workflow file this change adds, or `test_all_workflow_files_are_scope_classified` fails. This applies to the Option B lint job.
- `devenv.nix` — the established mechanism for providing `cargo-llvm-cov` and `cargo-mutants`, and the natural home for the Dylint toolchain entry under Option B.
