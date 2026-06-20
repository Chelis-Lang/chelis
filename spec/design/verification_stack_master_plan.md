# Chelis Verification Stack: Master Implementation Plan

Intended location: `spec/design/verification_stack_master_plan.md` (chelis repo).
Companion documents: the whole-stack sketch (`spec/design/verification_stack_sketch.md`), the dependency map (`spec/design/verification_stack_dependency_map.md`), the Beacon engine plan (`spec/design/beacon_plan.md`), and the VNN-LIB front-end placeholder (`spec/design/vnnlib_frontend_placeholder.md`).

## 1. Purpose and scope

This plan covers the chelis-side buildout of the verification orchestrator: the compiler and IR changes, the discharge-engine interface, the honesty layer, the dispatcher, and the integration of the external solver and library backends into core chelis as optional, slim-buildable dependencies. Beacon has its own plan. This document specifies everything Beacon depends on, plus everything that is not Beacon.

The backend and library components evaluated separately (Sollya, Arb/FLINT, Z3, Clarabel, Carcara) are not separate repositories. They are dependencies brought into core chelis behind optional build features, with their own deployment and build configuration. That integration is part of this plan (sections 4.5 and 4.6). Downstream pickup by Shoals and the other shells is noted at the end and deferred.

## 2. Context

The stack verifies properties of numerical programs by dispatching each goal to the engine whose method fits its shape, then aggregating the heterogeneous results behind one honest composite verdict, all on the chelis tensor IR so the verified artifact is the executed artifact. cvc5 is the exact engine for the polynomial and logical core. Beacon is the in-house, IR-native, sound bound-propagation engine that scales where SMT cannot and runs the same on finance pricing graphs and neural networks. The remaining engines fill what those two cannot close. The composite never launders a weak guarantee into a strong one. Background design is in the whole-stack sketch (`spec/design/verification_stack_sketch.md`) and the component evaluation, with the dependency structure in the dependency map (`spec/design/verification_stack_dependency_map.md`); this plan is the build-of-record.

## 3. How to read this plan

Work items are numbered for reference, not sequence. Each item records what it depends on. The dependency edges, not the numbering, define what must precede what. Sequencing and timeboxing are the team's; this document fixes structure and technical content, not schedule. Items in a group with no dependency edge between them are independent.

## 4. Work items

### 4.1 IR and lowering foundation

**WI-1 Lowering and cost-path fixes.** Resolve the `chelis check` stack-overflow on real Shoals source and the `ReservedLinkerName` checker error on the cost path, and provide a package-aware, import-resolved lowering entry point that emits the full resolved RISC DAG for a named entry (forward and gradient). Produces clean DAGs for downstream consumption. Depends on nothing; partly in-flight.

The stack-overflow is in the *type checker*, not lowering: `chelis_types::infer::infer_expr` and `infer_app` (`crates/chelis-types/src/infer.rs`) mutually recurse one native frame per AST level over the deeply-nested `app` trees the reef-linked pricer desugars into (gdb-confirmed on the real pricer; the crash is a `SIGSEGV`/`abort`, not lowering). The recursion is *finite* — a 512 MiB thread stack completes the real pricer — but `chelis check` runs the checker on the process main thread (default 8 MiB), which overflows partway through. A native stack overflow aborts the process (Rust's handler is `abort()`, not `panic()`), so it cannot be caught after the fact. The Phase-1 close adds a shared `stacker::remaining_stack()` budget guard (a ~128 KiB red zone, with a conservative depth-counter fallback where the platform cannot report remaining stack) at every self-recursive `deep::Expr` walker in the *type checker's* `infer_program` pipeline — `infer_expr` and its ~30 sibling walkers. When the budget is exhausted a walker records the bail (with the offending walker site and source span) into a thread-local flag and stops recursing; every public check entry drains that flag before its empty-errors gate, so a bail in the checker's own inference recursion surfaces as a hard, located check failure rather than a SIGSEGV — and never as a silent green or partial result (covered-or-rejected). Concretely, the real Shoals pricer's overflow (gdb-pinned to `infer_expr`) is now a clean located checker diagnostic. A byte budget (not a depth constant) is used because the overflow depth is build- and stack-profile dependent (the same chain overflows at very different depths under debug-vs-release and 2/8/32 MiB stacks); `remaining_stack()` measures the actual resource, the same mechanism rustc uses.

This bounds the checker's *own* recursion only. It is **not** a general "the type checker never SIGSEGVs on deep input" guarantee: the foundational `chelis_deep::ast::Expr` type's derived `Clone`, `Drop`, and `PartialEq`, plus `chelis_deep::validate::validate_expr`, still recurse over the nested structure and overflow on synthetic input deeper than the pricer, and the checker unavoidably clones and drops deep `Expr`s. A per-site guard in `infer.rs` cannot reach a derived `Drop`. Closing that residual (clone / drop / validate) is the deferred follow-up's job — see below.

The full pricer checking end-to-end is the deferred capability, tracked as a stack-growing (`stacker::maybe_grow`) follow-up; see `docs/investigations/wi1_infer_recursion_depth.md`. The follow-up is the right home because the residual deep-input overflow surface is *outside* the type checker, in the `chelis_deep` core AST crate: the derived `Clone`, `Drop`, and `PartialEq` for `Expr`, plus `chelis_deep::validate::validate_expr`, all recurse over the nested structure and overflow on a deeply-nested `Expr` regardless of the checker's per-walker guards (the checker unavoidably clones and drops deep `Expr`s). Per-site guards in `infer.rs` cannot reach a derived `Drop`; `maybe_grow` at the check entry grows the stack transparently for inference, cloning, dropping, and validation alike, which is why it — not more per-site guards — is the deferred fix.

**WI-2 IR op-subset versioning.** Pin and version the RISC op subset the verifier and Beacon target. The IR is an append-only topo-ordered SSA-like DAG with serde-derived `WireDag`/`WireRiscOp` JSON and no schema version field today; add a version so Beacon's transformers pin to a known surface. Produces a stable target surface. Depends on nothing.

### 4.2 Discharge seam and engine interface

**WI-3 Graph-extraction seam.** A property-to-IR path that does not route through `SmtProperty`. Today the proof path lowers a property to variables, preconditions, and a postcondition, discarding the IR at the SMT boundary; Beacon needs the DAG plus the input box plus the output-range assertion. This item provides that path, which the dispatcher forks to above SMT lowering. Produces the goal form Beacon consumes. Depends on WI-1. This is the seam the recon flagged as the consequential architecture fact.

**WI-4 Discharge-engine interface.** Generalize the current `Solver`/`SmtExpr` boundary to a canonical, engine-independent goal that retains an IR back-reference, where each engine implements a fitness predicate and a `discharge` returning a discharge object carrying a soundness flag and a qualifier set. cvc5 becomes one engine owning its own `Goal`-to-`SmtExpr` lowering rather than the trait's currency. Produces the seam every engine plugs into. Depends on nothing structurally (a refactor of shipped code).

**WI-5 Box and output-range goal schema.** A first-class goal shape for interval-box inputs and output-range assertions, Beacon's native verdict form. Scalar input boxes and output-range assertions are expressible today as ordinary preconditions and postconditions; this adds the first-class form so Beacon-bound goals are expressible directly. Depends on nothing.

### 4.3 Honesty layer

**WI-6 Qualifier-set verdict algebra.** Replace the scalar verdict-status precedence (`Error > Failed > Unsupported > Passed`) with a rollup that is the union of qualifiers and the minimum soundness across the discharges in a goal's dependency set. The guarantee kinds (exact, delta-complete, special-function-certified, sound-over-approximation, certificate-bearing, fuzz, axiom) are incomparable on one axis, so this is a lattice, not a chain; implementing it as a total order would launder a weaker guarantee into a stronger badge. Each discharge carries `(soundness, qualifier_set)`, and verdict rendering is a function of the rolled-up set. Integrity-critical. Depends on WI-4.

**WI-7 Non-vacuity guard.** Confirm the assumed invariant set is jointly satisfiable before a green is trusted. The recon confirmed no such check exists today, so this is added, not preserved. Vacuous preconditions otherwise prove anything. Depends on WI-4 and the prover.

**WI-8 Per-assumption discharge provenance.** Thread each assumption's discharge tier from c-earchin emission through to the artifact. Today assumption injection is implicit (binder-type matching, rendered `injected: true`) and carries no per-assumption tier; c-earchin emits source IDs but no discharge tier. This adds the field across the prover, the artifact schema, and c-earchin emission so the WI-6 rollup can aggregate across kinds. Depends on WI-4 and c-earchin.

### 4.4 Dispatcher and AD target

**WI-9 Dispatcher / orchestrator.** Route each goal to the fitting engine by goal shape (polynomial and logical to cvc5; polynomial-NRA to Z3; special-function inequalities to the envelope path or MetiTarski fallback; large-graph and range to Beacon; polynomial with certificate to the SoS backend), forking above SMT lowering so a Beacon-bound goal takes the graph path and a cvc5-bound goal takes the SmtProperty path. Provide a deep-prove lane distinct from the interactive lane for heavy backends (async, cancellable, result-cached). Depends on WI-3, WI-4, WI-5.

**WI-10 AD as verification target.** Make the existing `grad` adjoint graph dispatchable like any property, so verified bounded sensitivities (Greeks over a region) dispatch to Beacon over the gradient graph. The combined forward-and-backward graph already lowers in the same RISC IR; this wires it into dispatch. Depends on WI-9.

### 4.5 Backend and library integration (swept-in scope)

These bring the evaluated best-in-class components into core chelis behind the WI-4 interface. Each is an optional dependency (4.6). Technical context per backend follows.

**WI-11 cvc5 (exact core, existing).** Already linked in-process via the `cvc5-sys`/`cvc5_rs` FFI wrapping cvc5 1.3.1, selecting `QF_NRA`/`QF_NRAT`/`NRA`/`NRAT` logics. The only new work is the product build configuration: the shippable build must avoid the GPL-linked CLN path and use GMP, since the GPL-via-CLN binary is not distributable in the product. Effectively always-on rather than feature-gated.

**WI-12 Z3 in-process adapter.** Add the `z3` crate as the polynomial-NRA engine and the in-process real-closed-field backend, removing any need to shell to QEPCAD. The bake-off found Z3's nlsat closed a polynomial-NRA goal cvc5 timed out on, but on an old cvc5 binary, so re-confirm the cvc5-versus-Z3 split on the linked cvc5 1.3.x before fixing engine roles. Z3 does not support `exp`, so transcendental goals route to cvc5 or, preferably, to the envelope-plus-polynomial decomposition (WI-13): replace the transcendental with its certified envelope bound and hand the residual polynomial fact to Z3, which the bake-off showed closes instantly where the full transcendental goal does not. Depends on WI-4.

**WI-13 Special-function envelope library (Sollya, offline).** Generate certified sound envelopes for `erf` (and later the activation family) using Sollya offline. The bake-off derived the real `erf`-argument range from the desk box as roughly plus or minus one hundred, up to plus or minus three hundred on the route box, so a single polynomial is far too loose and saturation is mandatory: outside roughly plus or minus six, `erf` is within tiny tolerance of plus or minus one, so the envelope is the saturating constant; inside the central box, use a Sollya `remez` polynomial with a certified `supnorm` error bound. Emit the envelopes as embedded Rust constants with the certified epsilon per box, and retain the Gappa proof artifacts. Sollya is not linked at runtime (offline generation), so its CeCILL-C license is a build-tooling dependency, not a distribution concern. Consumed by Beacon's special-function relaxations and by the WI-12 envelope-plus-polynomial decomposition. The approximation-versus-mathematical-erf bridge bound for the existing pricer is handled by bounding `erf64` directly (Beacon) or via the Arb oracle (WI-14), not via Sollya `supnorm`, which does not fit the non-polynomial A-S expression. Depends on nothing.

**WI-14 Arb/FLINT oracle wrapper.** Wrap `arb-sys` with a narrow safe Rust layer exposing rigorous enclosures (`rigorous_erf_enclosure(x, prec)` and the rest) plus a containment helper, for the soundness oracle and the A-S bridge bound. The bake-off confirmed `arb-sys` 0.3.6 builds against FLINT/Arb 3.x and exposes `arb_hypgeom_erf` and interval extraction, with an include-order requirement (`mpfr.h` before `flint/arb.h`); a direct FLINT FFI is a viable fallback. FLINT/Arb LGPL obligations are recorded and deferred. Depends on nothing.

**WI-15 SoS certificate backend (Clarabel).** Use the `clarabel` crate as the SDP core for the sum-of-squares certificate path on the nonlinear-polynomial fragment. Implement the SoS-to-SDP reduction (Markov-Lukacs for univariate-on-interval) and, the real work, the Peyrl-Parrilo exact rational repair and projection plus boundary-degeneracy handling: the bake-off found nondegenerate instances solve to high accuracy but near-boundary instances produce small negative PSD coordinates and naive rounding reconstructs nothing exactly. The discharge is tagged certificate-bearing and covers the polynomial fragment cvc5's proof output cannot certify. Depends on WI-4. Parallel to Beacon.

**WI-16 Carcara auditability.** Build Carcara from source (it is a Rust project, not a crates.io crate) and route cvc5's Alethe proofs through it for independent in-process re-checking. cvc5's Alethe output covers equality-with-uninterpreted-functions, linear arithmetic, bit-vectors, and parts of strings, not nonlinear real arithmetic, so this is the auditability path for the non-transcendental fragment only; the transcendental fragment's auditability comes from WI-15 certificates and the WI-13 Gappa proofs. Depends on WI-4. Parallel.

**WI-17 MetiTarski fallback (optional shell).** A shell adapter to MetiTarski with its Z3 backend, for special-function inequalities the WI-13 envelope library cannot cover. Expected rare, and the only CLI-shell engine in the set. Depends on WI-4. Optional and low-reach.

### 4.6 Build, features, and deployment infra

**WI-18 Optional engine features.** Gate each backend behind a cargo feature so a slim build excludes the heavy dependencies. The always-on core is cvc5 plus the IR, the interface, the honesty layer, and the dispatcher. Z3, Clarabel, Carcara, and MetiTarski are feature-gated. Beacon ships as a separate ecosystem shell (`chelis-lang/beacon`) rather than an in-core cargo feature; it integrates through the discharge-engine interface (WI-4), so a slim core build excludes it by construction. The envelope library is build-time tooling that produces embedded constants, so it is not a runtime dependency at all and adds no link-time weight to a deployed build. Depends on WI-4 and the relevant backend items.

**WI-19 Build and vendor configuration.** The cvc5 non-GPL build configuration (GMP, not CLN); FLINT/Arb vendoring with the include-order handled; the inari interval crate's requirement for a modern CPU target (`target-cpu=haswell` or equivalent), which constrains deployment targets and raises an open ARM question for on-prem and sovereign serving; and the Sollya offline toolchain as a build-only dependency. License obligations (FLINT/Arb LGPL, Sollya CeCILL-C, the cvc5 build) are recorded and deferred per direction. Depends on the relevant backend items.

**WI-20 Deployment surface.** Distinguish the in-process engine set from the offline envelope toolchain, and the slim default build from the full build. Record the CPU-target constraints and the trust-base posture that in-process Rust and in-house engines are cleaner for the audit story than shelled binaries, so the shell path (MetiTarski) stays a rare fallback. Depends on WI-18, WI-19.

## 5. Downstream pickup (deferred)

Recorded for continuity, revisited once the stack is built. Shoals dispatches the European pricer through the orchestrator, bounding `erf64` directly via Beacon, which yields verified pricing bounds and, through WI-10, verified bounded Greeks over the trading range; C Proof is the commercial vehicle around those guarantees. Other shells follow the same dispatch pattern. Path-dependent and Monte Carlo pricing need the host/tensor seam specified in the Beacon plan and are a later fragment. None of this is built against in this plan.

## Appendix: dependency summary

- Foundation, no internal dependencies: WI-1, WI-2, WI-13, WI-14. The last two live outside the compiler and parallelize fully.
- Substrate, what the rest hangs off: WI-3, WI-4, WI-5.
- Honesty and dispatch, on the substrate: WI-6, WI-7, WI-8, WI-9, WI-10, and in parallel WI-12, WI-15, WI-16, WI-17.
- Build and deployment, alongside the backend items: WI-18, WI-19, WI-20.
- WI-11 is existing plus a build-config change.

Within each group, items with no edge between them are independent. The numbering carries no schedule.
