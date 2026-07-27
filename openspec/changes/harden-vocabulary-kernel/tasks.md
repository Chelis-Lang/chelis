# Tasks: harden-vocabulary-kernel

Phases are ordered so that the costly unknowns — the Dylint toolchain and the
Charon/Aeneas toolchain — are probed before any source changes, and so that a failure in
either leaves the rest of the change landable on its own.

Three ordering rules, all inherited rather than invented:

- **Coverage before mutation**, per `add-coverage-baseline-chelis-ir` as quoted in
  `add-mutation-baseline-dim-canon`: "Coverage first, mutants second, in that order." A
  surviving mutant in an unreached region is not a finding about assertion strength; it is
  the coverage number restated in a more expensive form.
- **Mutation before the cheaper oracles.** The mutation baseline must be captured while
  the crate still has only its single-sample rejection probe, or the gap those oracles
  close cannot be demonstrated.
- **Cheaper oracles before the proof.** Per `route-proof-obligations`, an exhaustive check
  over a finite domain is the default. The proof exists to establish and measure the lane,
  and its phase is ordered so that declining it costs no guarantee.

## Phase 0 — Decide and measure (blocking)

- [ ] **0.1** Record the toolchain decision from the proposal (Option A / B / C). Recommendation is B: gating, separate CI job, Nix-provided nightly. Write the decision and its rationale into this change before any other phase begins.
- [ ] **0.2** Probe Charon on `chelis-vocab` as it exists today. Record: does it extract; does it require its own toolchain pin; does that pin conflict with `rust-toolchain.toml` (`stable`) or with Dylint's `nightly-2026-04-16`. **If Charon needs a third pin, stop and re-scope** — Phases 7 and 8 split into their own change and the rest lands alone.
- [ ] **0.3** Probe Barnacle's `dispatch_safety` and `tiger_style` against `chelis-vocab` from a local path dependency. Record cold and warm wall-clock, and the exact diagnostics produced.
- [ ] **0.4** Enumerate every consumer of `EffectKindDecodeError`. Confirm none stores it past the decoded input's lifetime. **If one does**, switch the design to `no_std` + `alloc` and amend the purity spec's allocation requirement before proceeding.
- [ ] **0.5** Coordinate on `crates/chelis-backend-metal/tests/dtype_abi_width_parity.rs`, which is untracked in the working tree and reads `byte_width` at two sites. Do not overwrite it.
- [ ] **0.6** Check whether `add-mutation-baseline-dim-canon` or `add-coverage-baseline-chelis-ir` have landed their wrappers (`scripts/mutants.py`, `scripts/coverage.py`) and `devenv.nix` entries. **If so, adopt them.** Do not build a parallel runner or a second configuration file.
- [ ] **0.7** Confirm cargo-mutants and cargo-llvm-cov both run on `stable` for this crate, adding no pin. Record the pin budget table from the design document with actual findings.
- [ ] **0.8** Check Aeneas's Lean and Mathlib pin against the locally installed `leanprover/lean4:v4.29.0` that LaCaDiLE builds green on. Record whether `Aeneas.Std` is compatible.
- [ ] **0.9** Probe whether LaCaDiLE's `scripts/prove.py` loop applies to Aeneas-generated obligations. Record the finding. **It is a development accelerator only** — it depends on an external API key and an org-level Labs toggle, so it SHALL NOT appear in the acceptance oracle.
- [x] **0.10** Measure the exhaustive `i32` check. **Done 2026-07-27** (aarch64-apple-darwin, 10 cores): release single-threaded **10.4 s**; debug single-threaded **128.5 s**; debug 10-threaded **35.1 s**. All visited the full domain (9 accepted, 4,294,967,287 rejected) with `black_box` preventing elision. Release compile of the crate is ~3 s. **Conclusion: per-PR CI step, release, single-threaded, ~14 s total** — cheaper than adding the threaded debug form to the existing job, and it catches regressions on the causing PR rather than the next nightly.
- [ ] **0.11** Confirm this obligation's routing against `route-proof-obligations`, including its cheaper-oracle rule. Record that the exhaustive check is the primary oracle and the proof is lane-establishment.
- [ ] **0.12** Satisfy the lane isolation condition: the repository must build and test on pinned stable with neither Charon, Aeneas, nor Lean installed. **If that cannot be arranged cleanly, decline the lane** and land Phases 1–6 without the proof work. No guarantee is lost by declining.

## Phase 1 — `no_std` purity

- [ ] **1.1** Write the failing test first: a compile-level assertion that the crate builds without the standard library.
- [ ] **1.2** Move `std::error::Error` → `core::error::Error`, `std::fmt` → `core::fmt`.
- [ ] **1.3** Change `EffectKindDecodeError::Unknown` to borrow its symbol; add the lifetime parameter; update consumers.
- [ ] **1.4** Add `#![no_std]` **unconditionally**. No `std` feature, no `cfg_attr`. Confirm the crate has no `alloc` dependency.
- [ ] **1.5** Negative test: reintroducing a `std` path fails the build.
- [ ] **1.6** Negative test: decoding an unknown symbol allocates nothing on the decode path.
- [ ] **1.7** Confirm the manifest declares no features at all, so every mechanism below runs over exactly one configuration.

## Phase 2 — FCIS relocation

- [ ] **2.1** Decide the destination for `render_runtime_dtype_c_header`. It belongs with the component owning the generated artifact; confirm against `crates/chelis-runtime`.
- [ ] **2.2** Move the generator and `crates/chelis-runtime/tests/runtime_dtype_generated_header.rs` together.
- [ ] **2.3** Assert the generated header is byte-identical to the pre-move output. This is the regression oracle for the relocation and must be checked against the artifact as it exists on `main`.
- [ ] **2.4** Confirm every tag identifier, value, and byte width in the generator still derives from `chelis-vocab` rather than being restated locally.
- [ ] **2.5** Negative test: the vocabulary crate contains no function returning generated source text.

## Phase 3 — Lint configuration and Barnacle, scoped

- [ ] **3.1** Add a `[lints]` table to `crates/chelis-vocab/Cargo.toml`. At minimum enable the classes governing lossy numeric conversion and unchecked arithmetic — `clippy::cast_possible_truncation` and `clippy::arithmetic_side_effects` are both off today because nothing declares them on.
- [ ] **3.2** Fix or explicitly justify every diagnostic the new table produces. A relaxation is scoped to one item and carries a recorded reason.
- [ ] **3.3** Record whether the table should go workspace-wide. **Do not do it here** — 28 crates is a separate argument with its own evidence.
- [ ] **3.4** Add `dylint.toml` at the repository root configuring `dispatch_safety.enum_paths` for `EffectKind` and `RuntimeDType`, and `tiger_style_architecture_sized_integer`.
- [ ] **3.5** Change `byte_width` from `usize` to `u32`; convert at `crates/chelis-runtime/src/lib.rs:220` and in the vocab tests. Reconcile with the untracked metal test from 0.5.
- [ ] **3.6** Wire the lint job per the Phase 0 decision. Under Option B: a new CI job with a `devenv.nix`-provided toolchain, leaving `scripts/gate.py` untouched.
- [ ] **3.7** Add the new workflow file to `NON_GATE_WORKFLOWS` in `scripts/test_gate.py`. Without this, `test_all_workflow_files_are_scope_classified` fails on any unclassified workflow.
- [ ] **3.8** Confirm `scripts/test_gate.py` still passes — it also asserts CI hand-inlines no command the gate script does not produce.
- [ ] **3.9** Remove the `EffectKind` / `RuntimeDType` substring assertions in `crates/chelis-cli/tests/closed_vocabulary_architecture.rs` that `dispatch_safety` subsumes. Record which lint replaced which assertion.
- [ ] **3.10** For each retained substring assertion, record why no semantic rule covers it.
- [ ] **3.11** Negative test: a planted catch-all arm over `RuntimeDType` fails the lint job.
- [ ] **3.12** Negative test: a planted `usize` in the vocabulary crate fails the lint job.

## Phase 4 — Coverage (first, per the inherited rule)

- [ ] **4.1** Run cargo-llvm-cov scoped to `chelis-vocab`, reporting line coverage per chelis#803's enforcement-floor policy. Region, function, and instantiation totals stay diagnostic.
- [ ] **4.2** Produce the region-coverage export Phase 5 needs to classify survivors. This is the reason coverage runs first; without it a survivor cannot be told apart from an unreached line.
- [ ] **4.3** Close any coverage gap with tests, positive and negative per the repository's parity rule, **before** mutation runs.
- [ ] **4.4** Report wall-clock and artifact size to chelis#806, which is sizing CI for the coverage workflow.
- [ ] **4.5** Do not commit HTML, JSON, `.profraw`, or merged profile data, per chelis#803's tracked-vs-generated policy.

## Phase 5 — Mutation baseline, captured before the gap is closed

Run this **before** Phase 6. The predicted survivor only exists while the single-sample
rejection probe is the crate's strongest rejection evidence.

- [ ] **5.1** Run cargo-mutants scoped to `chelis-vocab`. Report generated / caught / missed / unviable / timeout as **separate counts**, per chelis#808's category policy. A single percentage is not an acceptable report.
- [ ] **5.2** Confirm the unmutated baseline passes under the documented test selection before trusting any mutation result.
- [ ] **5.3** Fail closed when the run generates zero mutants, per `add-mutation-baseline-dim-canon`. A zero-mutant run exits zero and reads as a perfect score; that reading must be impossible.
- [ ] **5.4** Classify each survivor against the Phase 4 coverage export. A survivor in an unreached region is the coverage number restated, not evidence about assertion strength.
- [ ] **5.5** Record whether a mutant narrowing `decode_id`'s rejection arm **survives** `crates/chelis-vocab/tests/closed_vocabulary.rs:90`, which probes a single unknown id. This is the predicted finding.
- [ ] **5.6** **If it does not survive**, record that too, and state plainly that the single-sample probe was stronger than expected. Do not adjust the prediction after the fact.
- [ ] **5.7** Triage every survivor. Per chelis#808, an equivalent mutant prompts simplification of the implementation first; a test written solely to kill a mutant is not the default response.
- [ ] **5.8** Document the test selection used. A package-only run must not be presented as proving the full acceptance surface catches the mutant.
- [ ] **5.9** Add no exclusion or `#[mutants::skip]` to obtain a green result.

## Phase 6 — The cheaper oracles

These carry the guarantees. Everything after this phase is lane-establishment.

- [ ] **6.1** Add the exhaustive check over the entire `i32` domain: every value round-trips or is rejected with the invalid-tag error carrying that value. Single-threaded, in its own test binary so a filterset can name it precisely. Wrap the argument in `black_box` — without it the loop is elided and the test proves nothing.
- [ ] **6.1a** Add the bounded companion that runs in the **local** `default` profile: every valid tag, the boundary values (`-1`, `9`, `i32::MIN`, `i32::MAX`), and sampled invalid ones. The exhaustive check is the complete oracle; this is what keeps the local inner loop inside its ~60 s budget.
- [ ] **6.1b** Add a dedicated per-PR CI step: `cargo nextest run -p chelis-vocab --release -E 'binary_id(/^chelis-vocab::<binary>$/)'`. Release, single-threaded, ~14 s including compile. **Do not** add it to the debug workspace job.
- [ ] **6.1c** Exclude the exhaustive binary from the `default` profile's filter so a local `cargo nextest run` does not pick it up. This makes `ci` a **superset** of `default` for the first time — rewrite the `.config/nextest.toml` header comment, which currently states the two share a filter verbatim and that a test is on the per-PR gate XOR the nightly gate. That invariant gains a third category and the file's own rule requires the selection stay explicit.
- [ ] **6.1d** Record that `.config/nextest.toml` sets no `slow-timeout`, so nextest warns at 60 s and does not terminate. If a `terminate-after` is ever added there, this test needs an explicit override or it starts being killed.
- [ ] **6.1e** If the release step is later found awkward, the fallback is the threaded debug form under a nextest test group with `threads-required = 'num-cpus'`, which schedules it exclusively rather than oversubscribing. Recorded so the option is not rediscovered from scratch.
- [ ] **6.2** Add the injectivity check over `ALL`, pairwise.
- [ ] **6.3** Add compile-time layout assertions: the vocabulary's size, and each variant's tag value against the constant emitted for it in generated code.
- [ ] **6.4** Add the differential test against the compiled generated C decoder. Execute both over every valid tag and a set of invalid ones; require agreement on acceptance, decoded result, byte width, and rejection. **Comparing generated text to a checked-in artifact does not satisfy this** — both decoders must run.
- [ ] **6.5** Negative test: a planted generated-decoder arm accepting a value the Rust decoder rejects fails the differential test.
- [ ] **6.6** Negative test: a planted `repr` or discriminant change fails compilation.
- [ ] **6.7** Negative test: a planted extra `decode_id` arm accepting an unassigned value fails the exhaustive check.
- [ ] **6.8** Re-run cargo-mutants. The survivor from 5.5 must now be caught. **Attribute the closure to the exhaustive check**, which is what closed it — not to the proof, which has not run yet.
- [ ] **6.9** Record that at this point every headline guarantee in this change is established and enforced on every pull request, and that the remaining phases add no guarantee.

## Phase 7 — Aeneas extraction (lane establishment)

- [ ] **7.1** Confirm the verified subset extracts: `RuntimeDType::{ALL, id, byte_width, decode_id}`. Items outside it may remain in the crate but must not block extraction.
- [ ] **7.2** If the `&'static str` accessors or `EffectKind::decode` block extraction, isolate the subset behind a module boundary rather than deleting them.
- [ ] **7.3** Record the extraction command and its expected output as the acceptance oracle for this phase.

## Phase 8 — The proofs

- [ ] **8.1** Prove round-trip: `∀ d, decode_id(id(d)) = Ok(d)`.
- [ ] **8.2** Prove injectivity: no two variants share a tag value.
- [ ] **8.3** Prove exhaustive rejection: `∀ i : i32, (∀ d, id(d) ≠ i) → decode_id(i) = Err(InvalidId { id: i })`.
- [ ] **8.4** Wire extraction and proof checking into the gate chosen in Phase 0.
- [ ] **8.5** Negative test: a planted aliased discriminant fails the injectivity proof.
- [ ] **8.6** Negative test: a planted extra `decode_id` arm accepting an unassigned value fails the exhaustive-rejection proof.
- [ ] **8.7** Report the lane cost measurement into `route-proof-obligations` Phase 3: effort, toolchain friction, proof-artifact length, and whether the extracted model is reusable by a later Lean theorem.
- [ ] **8.8** Confirm no document claims these proofs made a property available that Phase 6 had not already established.

### Oracle, adapted from LaCaDiLE

Reimplemented against this change's surface, with tests per the repository's Python-with-tests policy. Do **not** vendor, generalize, or depend on `../LaCaDiLE`.

- [ ] **8.9** Admission scan: no `sorry` or `admit` in any Lean source.
- [ ] **8.10** Banned declaration forms: no `axiom`, no `unsafe`. The justified-exception allowlist is expected to be **empty**; assert its emptiness rather than leaving the list open.
- [ ] **8.11** Required proof-surface **with shape**: each theorem exists *and* its signature matches the expected statement. A theorem stating `True` must not pass.
- [ ] **8.12** Negative test: a theorem renamed, deleted, or weakened to a vacuous statement fails the oracle.
- [ ] **8.13** Negative test: a planted `sorry` fails the oracle.
- [ ] **8.14** Negative test: a planted `axiom` fails the oracle, and adding it to the allowlist without a recorded reason also fails.

## Phase 9 — Honesty pass

- [ ] **9.1** Document the verified subset and the excluded items with reasons. Confirm no text describes the crate as verified.
- [ ] **9.2** State that the tag properties are established by the exhaustive check, and that the proof establishes the lane. Do not present the proof as the source of the guarantee.
- [ ] **9.2a** State the enforcement frequency precisely: the complete oracle runs per PR in a dedicated release step, the local profile carries the bounded companion, and neither is a manual gate. Do not describe the local `cargo nextest run` as covering the whole domain.
- [ ] **9.3** Record the residual after the differential test: agreement is checked over every valid tag and sampled invalid ones, which is stronger than the previous text comparison but is not a proof of the C decoder.
- [ ] **9.4** Confirm every item excluded from the verified subset retains its existing test coverage.
- [ ] **9.5** Confirm no document reports proof, coverage, and mutation results as a single combined score, and that each is labelled with the question it answers.
- [ ] **9.6** Report coverage figures **only** for items outside the verified subset.
- [ ] **9.7** Record why Miri is absent, so a later reader does not read its absence as an oversight.
- [ ] **9.8** File the `chelis-runtime` Miri issue. Include: 4,162 lines, raw-pointer C ABI surface, 45.67% line coverage in chelis#803's baseline, and the two caveats — Miri does not interpret the C side, and it needs a nightly pin.
- [ ] **9.9** Report the pilot's findings to chelis#803 and chelis#808 rather than settling their threshold, sharding, or enforcement decisions here.
- [ ] **9.10** Name the acceptance oracle for the change as a whole, per the one-oracle-per-phase rule.

### Anti-overclaim enforcement (the weakest link, mechanized)

Every other requirement in this change has a mechanical gate. The honesty requirements have only review, and overclaiming never fails a build.

- [ ] **9.11** Extend the Phase 8 oracle with banned documentation patterns. At minimum, reject text asserting: that `chelis-vocab` is verified rather than its numeric ABI core; that the dtype ABI path is proved end to end; that the proof established a property the exhaustive check did not; or any combined proof/coverage/mutation score.
- [ ] **9.12** Add required documentation snippets: the verified-subset boundary, the C-decoder residual, and the statement that the exhaustive check is the primary oracle.
- [ ] **9.13** Negative test: a planted sentence claiming the crate is verified fails the check.
- [ ] **9.14** Negative test: deleting the residual statement fails the check.
- [ ] **9.15** Unit-test the checker itself. It must be discovered by the existing `python -m unittest discover -s scripts -p 'test_*.py'` step in the `lint-and-unit` job, so it needs no new gate wiring.

## Deferred, deliberately

- **Miri on this crate.** `#![forbid(unsafe_code)]`, no pointers, no FFI, no allocation after Phase 1. There is no reachable UB for it to find, and it would cost a fourth toolchain pin. `chelis-runtime` is the real target — see 9.8.
- **Fuzzing.** Subsumed here by the exhaustive check; the whole input domain is enumerable. Valuable for the parser crates, which is a different change.
- **`cargo-deny`.** Vacuous on a zero-dependency crate. Worth arguing workspace-wide, separately.
- **A workspace-wide `[lints]` table.** 3.3 records the recommendation; 28 crates is its own argument.
- **A `std` feature flag.** Nothing needs `std` once the generator relocates, and a second configuration would double the matrix for every mechanism above.
- **Verus, Kani, or any second prover.** Owned by `route-proof-obligations`: a second lane needs a named obligation the first cannot express.
- **LaCaDiLE's proofs, model, or Lake project.** Only the four generic oracle mechanisms are adapted.
- **`prove.py` in the acceptance oracle.** Development accelerator only.
- **Workspace-wide coverage or mutation enforcement** — owned by chelis#807 and chelis#810.
- **Verifying `DimExprKey` normalization** — owned by `prove-dim-canon-overflow-safety`.
