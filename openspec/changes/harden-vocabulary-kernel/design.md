# Design: harden-vocabulary-kernel

## Why this crate and not a smaller one

Three crates in the workspace have no `chelis-*` dependency:

| crate | lines | external deps | load-bearing? |
| --- | --- | --- | --- |
| `chelis-version` | 83 | — | version strings; no semantic content |
| `chelis-shell` | 92 | `serde`, `bincode` | not dependency-free; serde derives are Charon-hostile |
| `chelis-vocab` | 195 | **none** | the C ABI dtype tag and the effect vocabulary |

`chelis-version` is smaller but proves nothing: there is no invariant to verify and no boundary to enforce. `chelis-shell` carries `serde` and `bincode`, and derive-generated code is exactly the friction an Aeneas pilot should avoid on its first run. `chelis-vocab` is the smallest crate that is simultaneously dependency-free, semantically load-bearing, and already the declared "single authority" for values crossing an ABI.

## The `no_std` split, concretely

Four `std` uses, three of which are trivially `core`:

| site | current | after |
| --- | --- | --- |
| `lib.rs:9` | `use std::error::Error` | `core::error::Error` (stable since 1.81) |
| `lib.rs:10` | `use std::fmt` | `core::fmt` |
| `lib.rs:45` | `symbol.to_owned()` → `String` | borrow `&'a str` |
| `lib.rs:174` | `render_runtime_dtype_c_header() -> String` | relocates out of the crate |

The third is the only one with a design consequence. `EffectKindDecodeError::Unknown { symbol: String }` owns its symbol so the error outlives the input. Every current consumer formats the error immediately, so a borrow suffices:

```rust
pub enum EffectKindDecodeError<'a> {
    Missing,
    Malformed,
    Unknown { symbol: &'a str },
}
```

This adds a lifetime parameter to a public type. That is the real cost of the change and should be weighed in Phase 0 — if a consumer needs to store the error past the input's lifetime, the alternative is `no_std` **with** `alloc`, which keeps `String` and is a weaker but still meaningful boundary. The tasks below verify the consumer set before committing.

**Why no `alloc`.** `no_std + alloc` still provides `format!`, `String`, and `Vec`. For a crate whose contract is "closed vocabulary, total decode functions," allocation-free is the claim worth making, and it is achievable here. A larger crate would not have that luxury; this one does, which is part of why it is the pilot.

## The FCIS boundary this crate actually has

The functional core is the vocabulary: two closed enums, their total accessors, and their decode functions. The imperative shell is `render_runtime_dtype_c_header`, which builds C source text.

That function is not merely "uses `String`." It is a **code generator whose output is checked into the tree** and compared by `crates/chelis-runtime/tests/runtime_dtype_generated_header.rs`. The generator/checked-in-artifact pattern is shell work by construction: it has a filesystem artifact as its contract. Its data — `ALL`, `c_macro()`, `id()`, `byte_width()` — stays in the core, and the shell reads it.

Note that this is deliberately **not** adopting Barnacle's `fcis_safety` library. That library enforces protected-scope manifests with anti-vacuity minimums, and configuring it meaningfully requires the scope taxonomy that `establish-pure-compiler-core` defines. Adopting it here would mean inventing that taxonomy for one 195-line crate and then rewriting it. The FCIS split in this change is structural, enforced by the crate boundary and `#![no_std]`, which is a real mechanical guarantee and does not pre-commit the taxonomy.

## The Aeneas subset — and what is honestly outside it

Aeneas translates a Rust subset via Charon into a pure functional model in Lean. It does not do well with `&str` pattern matching, `format!`, or trait machinery. The whole crate is therefore not the verification target.

**In the verified subset:**

```rust
impl RuntimeDType {
    pub const ALL: [Self; 9];
    pub const fn id(self) -> i32;
    pub const fn byte_width(self) -> u32;
    pub const fn decode_id(id: i32) -> Result<Self, RuntimeDTypeDecodeError>;
}
```

Nine-variant fieldless enum, `i32`/`u32` arithmetic-free accessors, `Result` return. No allocation, no strings, no floats, no loops, no recursion. This is close to the easiest possible Aeneas target while still being a real ABI boundary.

**Explicitly outside it:** `symbol()`, `name()`, `c_macro()` (return `&'static str`), `EffectKind::decode` (matches on `&str`), every `Display` impl, and the header generator. These stay covered by the existing tests. The change SHALL NOT claim the crate is verified — it claims the numeric ABI core is.

### The properties

Two theorems, both meaningful at the C boundary because `chelis-runtime`'s `chelis_alloc` path decodes an `i32` that arrived from generated C:

1. **Round-trip.** `∀ d : RuntimeDType, decode_id(id(d)) = Ok(d)`
2. **Exhaustive rejection.** `∀ i : i32, (∀ d, id(d) ≠ i) → decode_id(i) = Err(InvalidId { id: i })`

Together these say the tag space is exactly the nine variants, with no aliasing and no silently-accepted unknown. Property 2 is the one that matters and the one enumerated tests cannot establish: `crates/chelis-vocab/tests/closed_vocabulary.rs:90` currently probes a *single* unknown id (the first one past the end). The Lean statement quantifies over all 2^32.

That gap is the honest justification for the whole exercise on this crate. It is not that the current code is wrong — inspection says it is right. It is that "the decoder rejects every non-tag" is a universally quantified claim currently backed by one sample.

### What this does not prove

The generated C header contains its own `switch` decoder with an `abort()` default. Proving the Rust decoder total says nothing about the C one. The existing generated-header test compares them textually, which is evidence of agreement but not proof. Closing that gap is out of scope and should be named as residual rather than quietly implied.

## Why three evidence mechanisms and not one

Each answers a different question, and stacking them without scoping produces a false sense of depth.

**The proof answers: is it true for every input.** Universally quantified, but only over the subset that extracts.

**Coverage answers: did the suite execute this.** Strictly weaker than the proof on the same code. The distinction that matters is **run scope versus report scope**: coverage *runs* over the whole crate, because Phase 5 needs its region export to tell a surviving mutant apart from an unreached line. It is *reported* only for the proof's complement — `symbol()`, `name()`, `c_macro()`, both `Display` impls, and `EffectKind::decode`. Publishing a whole-crate percentage after proving part of the crate would average a universal guarantee together with an execution trace, which is precisely the aggregation chelis#803 warns against when it restricts the enforcement floor to line coverage and keeps region/function/instantiation as diagnostics.

**Mutation answers: would the suite notice if this were wrong.** This is the one that pays before any proof exists, and it is the reason to run it before Phase 7.

### The ordering, and where it came from

An earlier draft of this change ran mutation before coverage. That was wrong, and the correction is inherited rather than invented. `add-coverage-baseline-chelis-ir` rejected `verify.rs` as a mutation target because "320 unreached regions means a mutation run there would waste effort on code no test executes. Coverage first, mutants second, in that order."

The rule generalizes: a mutant surviving in a region no test reaches tells you nothing about assertion strength. It restates the coverage number at the cost of a full mutation run. So the sequence is **coverage → mutation → extraction → proof**, with mutation still ahead of the proof so its result cannot be retrofitted to the conclusion.

This also means `add-mutation-baseline-dim-canon`'s survivor-classification requirement applies here directly: survivors are reported in two classes, reached and unreached, and only the first is evidence.

### The mutation result this change expects to find

`crates/chelis-vocab/tests/closed_vocabulary.rs:90` establishes rejection by probing one value:

```rust
let first_unknown = RuntimeDType::ALL.iter().map(|dtype| dtype.id()) ... // one sample
```

A mutant that narrows `decode_id`'s catch-all — for instance accepting one unassigned discriminant — should **survive** that test. If it does, the pilot has produced an empirical instance of the exact weakness the exhaustive-rejection proof closes, on a crate small enough that the whole argument fits on one screen.

If it does *not* survive, that is also a finding worth recording: it means the enumerated test is stronger than it looks, and the argument for the proof rests on the 2^32 quantification alone rather than on a demonstrated gap. Either outcome is reportable. The tasks below require running mutation **before** Phase 5 so the result is not retrofitted to the conclusion.

chelis#808 also notes that equivalent mutants should first prompt simplification rather than a test written to kill the mutant. That applies here: if a survivor turns out to be behaviorally equivalent, the response is to simplify `decode_id`, not to add an assertion.

## Why not Miri

Miri detects undefined behavior. The defect classes it finds — out-of-bounds, use-after-free, invalid transmute, uninitialized read, aliasing violation under Stacked or Tree Borrows, data race — require `unsafe`, raw pointers, FFI, or concurrency. `chelis-vocab` has `#![forbid(unsafe_code)]` at `lib.rs:7`, no pointers, no FFI, no threads, and after Phase 1 no allocation. There is no reachable UB for Miri to find.

The near-miss worth naming: `RuntimeDType` is `#[repr(i32)]`, so an invalid-discriminant `transmute` would be UB and Miri would catch it. But `decode_id` is a `match` over `i32` returning `Result`, not a transmute — which is the safe construction, and is what the exhaustive-rejection proof is about. Miri would be checking for a mistake the code deliberately does not make, and the proof already rules out the failure mode that mistake would cause.

Running it anyway would produce a green result that reads as a safety claim, on a crate that cannot fail it. That is the failure mode the repository's "Do Not Trust Green" rule exists to prevent, so the honest move is to exclude it and say why.

**Where it belongs instead.** `chelis-runtime` is 4,162 lines of raw-pointer arithmetic across the C ABI — `(*tensor).ndim`, `shape.as_ptr()`, `chelis_alloc` — and sits at 45.67% line coverage in chelis#803's baseline. That is a real Miri target and should be filed as its own issue. Two caveats for whoever files it: Miri interprets Rust only, so calls into actual C are not covered; and Miri is nightly-only, which is a fourth pin on top of the three this change already tracks.

## Toolchain cost, measured not estimated

Phase 0 produces numbers, not opinions:

- Dylint under Nix: wall-clock for a cold and warm `dispatch_safety` + `tiger_style` run over `chelis-vocab` alone, and over the workspace.
- Charon + Aeneas: whether the crate extracts at all on the pinned stable toolchain, and whether Charon needs its own nightly. **This is the primary Phase 0 risk** — if Charon requires a third toolchain pin, Option B's cost roughly doubles and the verification half of this change may need to split out.
- Lean: proof-script length for the two properties. Expected small; if it is not, that is a finding worth recording before any larger crate is attempted.
- cargo-mutants: mutants generated, caught, missed, unviable, timeout — reported as separate counts per chelis#808's category policy, never as one percentage. Expected to be fast; the chelis#808 pilot generated 11 mutants on a comparable file in 3m20s.
- cargo-llvm-cov: wall-clock and line coverage for a single-crate run. chelis#803's workspace run is 28 minutes and 10 GiB; a one-crate run should be a rounding error against that, and confirming so is what makes this a useful pilot for #806's CI sizing.

### Pin budget

The running count, which Phase 0 must not exceed without an explicit decision:

| pin | required by | status |
| --- | --- | --- |
| `stable` | `rust-toolchain.toml` | shipped |
| `nightly-2026-04-16` + `rustc-dev` | Dylint / Barnacle | new, Option B scopes it to one CI job |
| Charon's pin, if any | Aeneas extraction | **unknown — Phase 0.2** |
| Lean `v4.29.0` via elan | the proofs | **already installed** — LaCaDiLE runs green on it locally |
| nightly | Miri | **excluded, and this is part of why** |

cargo-llvm-cov and cargo-mutants both run on stable and add no pin. That is a quiet point in their favor and worth stating, because it is the reason they can join this change while Miri cannot.

The Lean row is the pleasant surprise: it was assumed to be a new cost and is not. elan resolves toolchains per Lake project from a checked-in `lean-toolchain`, so this change's Lake project and LaCaDiLE's can hold different pins without conflict. **The residual risk is Aeneas's own pin.** `Aeneas.Std` is built against a specific Lean and Mathlib; if it demands something incompatible with 4.29.0, the consequence is a second Lean toolchain download, not a blocked change — but it should be discovered in Phase 0, not in Phase 6.

## Adapting LaCaDiLE's oracle

LaCaDiLE's `scripts/check_final_proof.py` is ~7,450 lines with a companion `test_check_final_proof.py`. Almost all of that volume is hardcoded tables naming LaCaDiLE theorems, declaration shapes, and doc snippets. Those are content. The design underneath is four generic mechanisms, and those are what this change reimplements at its own much smaller scale.

### The four mechanisms

**1. Admission scan.** `ADMISSION_RE = re.compile(r"\b(?:sorry|admit)\b")` over all Lean sources. Trivial to implement, and the single most important check: a proof containing `sorry` typechecks and proves nothing. Without this, Phase 6 could report green on an unfinished proof.

**2. Banned declaration forms with a justified allowlist.** LaCaDiLE bans `axiom` outright, then maintains `ALLOWED_NONCOMPUTABLE_DECLARATIONS` — a named, reviewed set of the noncomputable real-valued denotations Mathlib's `ℝ` forces. The shape is exactly this repository's narrowing-citation rule: forbid the construct, then justify each exception at its site rather than widening the ban. For this change the ban is `axiom` and `unsafe`, and the allowlist is expected to be **empty** — nine fieldless variants over `i32` need no escape hatch. An allowlist that stays empty is a stronger result than one that fills up, and is worth asserting as such.

**3. Required proof-surface, with shape.** LaCaDiLE keeps both `REQUIRED_THEOREM_DECLARATIONS` (the name exists) and `REQUIRED_DECLARATION_SHAPES` (the statement has the expected form). The second is the load-bearing half. A theorem named `exhaustive_rejection` that actually states `True` passes an existence check and proves nothing; checking the signature is what makes the surface non-vacuous. This change has three theorems, so the table is three entries — but it needs both halves.

**4. Banned and required doc snippets.** `BANNED_DOC_SNIPPETS` / `BANNED_DOC_PATTERNS` reject documentation sentences claiming more than the Lean establishes; `REQUIRED_DOC_SNIPPETS` require the boundary to be stated. LaCaDiLE's notes call this the immune system that keeps prose from drifting ahead of the proofs.

### Why mechanism 4 is the one that matters here

This change's `vocabulary-abi-roundtrip` spec already contains three requirements that are, in substance, anti-overclaim rules: the verified subset must be stated, residual boundaries must be named, and evidence mechanisms must report at their own strength rather than aggregating. All three are currently enforced by a reviewer noticing.

That is the weakest link in the whole change. Everything else here has a mechanical gate — the compiler enforces `no_std`, Dylint enforces the closed vocabulary, Lean enforces the theorems, mutation enforces test strength. The honesty requirements have nothing, and they are precisely the requirements most likely to erode, because overclaiming is never deliberate and never fails a build.

LaCaDiLE demonstrates the fix at a far larger scale. Concretely, the checker for this change should reject documentation asserting:

- that `chelis-vocab` is verified (only the numeric ABI core is);
- that the ABI decode path is proved correct end to end (the generated C decoder is not covered);
- any combined proof/coverage/mutation score.

And should require that the verified-subset boundary and the C-decoder residual are stated somewhere.

### What is deliberately not adapted

- LaCaDiLE's theorem content, model, Lake project, or any dependency on that repository.
- Its `debt-surface` audit (leftover interim theorem names). Real for a project that went through several proof-surface revisions; premature for three theorems written once.
- Its `required Lake targets` audit. Meaningful across many sidecar modules; trivial for one.

These should be revisited if this change's Lean surface ever grows past a handful of theorems — not adopted preemptively.

### The proof-filling harness

LaCaDiLE also ships `scripts/prove.py`, driving Leanstral via `lean-lsp-mcp` in a closed loop to discharge `sorry`s, with `check_toolchain.py` verifying the setup. If that loop applies to Aeneas-generated obligations, the proof-engineering cost of Phase 6 drops substantially from a hand-written estimate.

Two caveats before counting on it. Aeneas emits proof obligations in its own idiom over `Aeneas.Std` types, which is a different shape from the hand-written Mathlib-backed goals `prove.py` was tuned against; and it depends on a Mistral API key plus an org-level Labs toggle, which is an external dependency a repository gate should not require. Treat it as a labor accelerator during development, never as part of the acceptance oracle.

If Phase 0 shows Charon cannot extract the crate on an acceptable toolchain, the `no_std` + FCIS + Barnacle half of this change still stands on its own and should land without the verification half. The phases are ordered so that failure is cheap and visible.

## The `no_std` / `std` split, decided

Unconditional `#![no_std]`. No `std` feature, no `core`/`std` crate pair.

The three candidate shapes and why the first wins here:

| shape | fits when | applies? |
| --- | --- | --- |
| unconditional `no_std` | nothing in the crate needs `std` | **yes** |
| `#![cfg_attr(not(feature = "std"), no_std)]` | published crate, consumers in both worlds | no external consumers |
| `foo-core` + `foo` crate pair | a substantial `std`-only surface must ship alongside | the only `std`-only item is one generator, and it is leaving anyway |

The third is worth a second look, because it is what the FCIS instinct reaches for: a `chelis-vocab-core` and a `chelis-vocab` wrapper. It is the wrong shape here for a concrete reason — after Phase 2 relocates the C-header generator, the `std` side of that pair would be **empty**. A crate pair whose shell half has nothing in it is worse than no split, because it implies a boundary that carries no content.

The generator's correct destination is the component that owns the generated artifact, not a sibling of the vocabulary. That is the FCIS split actually doing work: the shell is a place that already exists, not one invented to balance a diagram.

### Where a feature flag would have cost real money

Worth recording, because "just add a feature, it's cheap" is the reflex. This change runs four mechanisms over the crate, and a feature flag doubles the configuration space for each:

- **Coverage.** chelis#803 documents instantiation coverage moving with monomorphization; feature-conditional items add a second denominator with no stable relationship to the first.
- **Mutation.** chelis#808 classifies mutants that fail to compile as `unviable` — inconclusive, neither caught nor missed. A mutation valid under one feature set and unviable under another muddies exactly the category boundary that issue insists on keeping clean.
- **Aeneas.** Charon extracts one configuration. A proof would then cover one half of a crate that ships two, and stating that boundary honestly is harder than not creating it.
- **Lint.** Dylint would need to run per configuration to be exhaustive.

None of these is fatal. Together they are a real multiplier on a change whose entire purpose is to measure those four costs cleanly for the first time.

## The property is brute-forceable, and that reorders everything

The original justification for the Aeneas work was exhaustive rejection over 2^32 values — "the property enumerated tests cannot establish." That is false, and noticing it is worth more than the proof it undercuts.

`i32` admits 4,294,967,296 values. `decode_id` is a nine-arm match over constants returning a `Result`. An enumeration of the whole domain is not a sample; it is a complete proof by exhaustion, and it carries exactly the guarantee the Lean theorem carries.

### Measured, 2026-07-27 (task 0.10, aarch64-apple-darwin, 10 cores)

| configuration | wall-clock |
| --- | --- |
| release, single-threaded | **10.4 s** |
| debug, single-threaded | **128.5 s** |
| debug, 10 threads (`std::thread::scope`) | **35.1 s** |

All three visited the full domain and reported 9 accepted, 4,294,967,287 rejected. `std::hint::black_box` wraps the argument, so the loop is not elided; without it the measurement would be meaningless.

**The estimate of 4–20 seconds was right for release and wrong for the case that matters.** The gate runs `cargo nextest run --workspace` in **debug**, where the same loop is 128.5 s — more than double the ~60 s budget `AGENTS.md` sets for the whole workspace inner loop.

Threading does not rescue it. The 3.7× speedup (not 10×) reflects this machine's performance/efficiency core split, and 35 s is still over half the entire workspace budget for one test. Worse, the threaded version **saturates all ten cores**, and nextest runs test binaries concurrently — so a test that monopolizes the machine for 35 s effectively serializes the suite around itself. The threaded form is the right shape for a manual gate and the wrong shape for a default one.

### What this changes

**The exhaustive check is a documented manual gate, not an ordinary test.** Per `AGENTS.md`, ignored tests are permitted "only when they clearly mirror a documented manual gate or an environment-dependent prerequisite," so Phase 6 must site it with a concrete command and success condition, and the default test surface keeps a bounded check — every valid tag, the boundary values, and sampled invalid ones.

**The claim "declining the prover lane costs no guarantee" needs qualifying.** The guarantee still exists, but it is enforced by a gate that runs on demand rather than on every change. That is weaker than an ordinary test and stronger than nothing, and the honest statement is the middle one.

**The proof's relative value rises slightly, and only slightly.** Once written, re-checking a small Lean file is milliseconds, where the exhaustive check is 10–128 s. Both sit outside the local inner loop; both are affordable in a per-PR CI job, since CI is not bound by the 60 s local budget. So this is a modest argument for the proof, not a reversal — the cheaper-oracle finding stands, with the cost now measured instead of guessed.

Three consequences, all of which the task ordering now reflects:

**The exhaustive check is the primary oracle**, per `route-proof-obligations`' cheaper-oracle requirement. It lands in Phase 6, before any extraction.

**Declining the prover lane costs no guarantee.** Task 0.12 can fail and the change still delivers every headline property. That was not true of the earlier draft, where declining left the gap open with nothing filling it — a fragility that existed only because the wrong oracle was load-bearing.

**The mutation prediction changes attribution.** The predicted survivor — a mutant narrowing `decode_id`'s rejection arm, surviving the single-sample probe at `closed_vocabulary.rs:90` — is now closed by the exhaustive check in Phase 6, not by the proof in Phase 8. The baseline must therefore be captured in Phase 5, *before* the oracles land, or the gap cannot be demonstrated at all. Task 6.8 requires the closure be attributed to the check.

What remains for Aeneas is establishing and measuring the lane on the smallest crate that can carry it, which is a real reason and the one `route-proof-obligations` Phase 3 needs a first obligation for. It is materially weaker than the claim it replaces, the documentation must say so, and 9.11 makes the anti-overclaim checker enforce it.

## The differential test is the only thing crossing the boundary that matters

The ABI this change is about is a Rust/C boundary. Every other mechanism here lives entirely on the Rust side.

`render_runtime_dtype_c_header` emits a `switch` with an `abort()` default. Nothing establishes that it agrees with `decode_id`; `crates/chelis-runtime/tests/runtime_dtype_generated_header.rs` compares the generated text to a checked-in artifact, which proves the generator is deterministic, not that the two decoders behave alike. A generator bug that emitted a wrong `case` value would produce a stable artifact and a passing test.

Executing both — every valid tag, a set of invalid ones, agreement on acceptance, decoded result, byte width, and rejection — is the only check in this change that touches the C side at all. It establishes something no other mechanism here does, which is more than can be said for the proof.

It does not make the C decoder proved. Phase 9.3 records that boundary: agreement checked over the whole valid domain and sampled invalid values is stronger than a text comparison and weaker than a proof, and the honest statement is the middle one.

## Lint configuration is off by default across the whole workspace

There is no `[workspace.lints]` table in the root `Cargo.toml`, and no crate declares one. `cargo clippy --workspace --all-targets -- -D warnings` — a gate stage — therefore runs **default groups only** across all 28 crates. `clippy::cast_possible_truncation` (pedantic) and `clippy::arithmetic_side_effects` (restriction) are both off.

`ground-tensor-extent-arithmetic` identified this while explaining why review did not catch chelis#888, and correctly noted that neither lint would have caught that specific defect. That is true and is not a reason to leave them off; it is a reason not to claim they are a substitute for the type-level fix that change proposes.

For a 195-line crate the table costs almost nothing and is the highest value-per-effort item in this change. Task 3.3 deliberately stops short of the workspace: enabling pedantic and restriction lints across 28 crates and 328k lines is a different argument with its own evidence, and inheriting it from a pilot would be exactly the overreach this change otherwise avoids.

## One verification lane, and why the second was deferred

`prove-dim-canon-overflow-safety` proposes Verus. This change proposes Aeneas. `route-proof-obligations` settles it: one lane, Aeneas, with a stated entry criterion for a second.

The reasoning is recorded there and not repeated here. The part that belongs in this document is why *this* obligation is the lane's first, and what that costs.

**Why this obligation.** It is the smallest real one available: a nine-variant fieldless enum over `i32`, no allocation, no strings, no floats, no loops, no recursion. If the lane cannot discharge this, it cannot discharge anything, and the negative result arrives cheaply. If it can, Phase 8.7 produces the cost figure that sizes every later proposal.

**What it costs to be honest about.** The obligation is no longer load-bearing. Phase 6 establishes the same properties by enumeration, so Phase 8 is measurement, not assurance. A change that ran the proof and quietly let readers infer it was necessary would be more impressive and less true. The anti-overclaim checker exists because that inference is the natural one and nobody makes it deliberately.

## Alternatives considered

**Verify `DimExprKey` normalization instead** (`chelis-ir/src/dag.rs:295-510`). Richer theorem, and the soundness claim in its doc comment is genuinely unproven. Rejected for *this* change: it lives inside a 46k-line crate with `serde` and `HashMap`, so it requires an extraction refactor before Aeneas can see it, and its arithmetic is being rewritten by `ground-tensor-extent-arithmetic` anyway. Proving properties of `saturating_mul` calls that are scheduled for deletion is wasted effort. Revisit after that change lands.

**Skip the pilot, go straight to `establish-pure-compiler-core`.** Rejected because that change's toolchain costs are currently unmeasured, and it is large enough that discovering a Charon blocker mid-implementation would be expensive.

**Adopt Barnacle wholesale across the workspace.** Rejected as a separate, larger decision. `tiger_style_architecture_sized_integer` alone would flag `usize` across most of the tree, and `tiger_style_direct_recursion` conflicts with recursive functions this repository legitimately has (`normalize_dim_quotient`, `push_product_atoms`). Note that conflict is real and unresolved: TigerStyle forbids direct recursion, Aeneas accepts it given a termination proof. Scoping Barnacle to one crate defers that argument instead of losing it.
