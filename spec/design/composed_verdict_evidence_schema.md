# Composed Verdict: Canonical Evidence Schema and Honesty Requirements

Intended location: `spec/design/composed_verdict_evidence_schema.md` (chelis repo).
Companion documents: the whole-stack sketch (`spec/design/verification_stack_sketch.md`), the master plan (`spec/design/verification_stack_master_plan.md`), the dependency map (`spec/design/verification_stack_dependency_map.md`), and the trust-stack architecture (`docs/trust_stack_verification.md`).

This is a requirements document, not a build plan. It states the substrate-side
obligations the Chelis verification stack must meet so a downstream legibility
layer can render the composed verdict honestly. The build-of-record for the
machinery that produces these records is the master plan (WI-4 through WI-8);
this document fixes the *shape and integrity* of what those work items emit, from
the point of view of the consumer that has to trust it.

## 1. Why this exists

Chelis is the **composed multi-method verification substrate**: one dispatcher
that routes each goal to the engine whose paradigm fits its shape, then
aggregates the heterogeneous results behind a single honest verdict (see the
sketch §"The stack, in layers"). The methods are first-class peers, not a
hierarchy with footnotes:

- the **type system** (LaCaDiLE dimensions, effects, linearity) — positive
  discharge at compile time, no runtime cost;
- **fuzzing and property-based testing** — statistical validation;
- **SMT over the reals** (cvc5, master WI-11) — exact discharge of the
  polynomial and logical core;
- **abstract interpretation and bound propagation** (Beacon, its own plan) —
  sound over-approximation that scales where SMT cannot, on the same IR for a
  pricing graph and a neural network;
- **specialized backend solvers** (Z3, Clarabel, certificate-bearing SoS,
  MetiTarski as a shelled fallback, master WI-12 through WI-17).

Neural-verification methods — abstract interpretation, bound propagation — are
**first-class method types in the dispatcher**, with their own evidence, not an
afterthought bolted beside the algebraic path. A verified robustness or
reachability bound is the same kind of record as an SMT-discharged inequality.

The substrate is the source of trust. A downstream legibility layer (the C Note
notebook is the first such consumer) renders that trust but cannot manufacture
it: it can only be as honest as the records it is handed. This document is the
contract between producer and consumer.

## 2. The canonical evidence record

Every verification result, **whatever method produced it**, carries one
canonical shape with method-specific extensions. This uniformity is the
foundation the legibility layer depends on: a consumer registers one renderer
per method against a known envelope, rather than special-casing each engine's
ad-hoc output.

A record carries:

- **claim** — the **discharged proposition (goal)** in a form the consumer can
  display directly, not reconstruct from source (chelis#436). The proven thing
  and the displayed thing must be the same thing; if the consumer has to
  re-derive the goal from the user's source, the two can drift.
- **method** — which engine established the result (type system, fuzz, SMT,
  bound propagation, certificate, …). One of the first-class method types above.
- **tier** — the certainty class of the discharge (exact, delta-complete,
  special-function-certified, sound-over-approximation, certificate-bearing,
  fuzz, axiom). These are the guarantee kinds the verdict algebra treats as
  incomparable (master WI-6).
- **scope** — the domain over which the claim holds: **over the reals** versus
  **over floats**. An SMT proof over the reals and a fuzz campaign over IEEE
  floats are different guarantees about different objects, and a record that
  omits scope lets a consumer conflate them.
- **assumptions** — the invariant set the discharge rests on, each with its own
  per-assumption discharge provenance (which engine closed it, with what
  guarantee kind, keyed to source identity; master WI-8). c-earchin emits source
  identity only; the tier is stamped prover-side at discharge time.
- **dependencies** — the other claims this one rests on, so the consumer can
  build the cross-property dependency graph and flag dependents for
  re-verification when an assumption changes status.
- **evidence** — the method-appropriate artifact: an SMT **model** /
  counterexample, a propagated **bound** and its tightness, a **proof reference**
  or certificate, the **validation parameters** (seed, sample count, shrunk
  counterexample) for a fuzz result, or the type rule that fired.
- **failure mode** — present when the result is not a clean discharge:
  disproved (with counterexample), unsupported (with reason), timeout/unknown,
  or vacuous. Absent only when there is genuinely no failure.

Nothing in the legibility layer can exist until results share this
representation (C Note product spec §4, §8). A field that is absent must be
**representable as absent** — never defaulted to a plausible value — so the
consumer can render absent as absent rather than fabricating a field.

## 3. Honesty requirements on emitted records

The composite never launders a weak guarantee into a strong one (sketch §"The
honest composite"). Concretely, the records the substrate emits must satisfy:

1. **No record over-claims.** The verdict on a record is exactly the property
   that was discharged, by the method that discharged it. A **pure-fuzz result
   must not be labeled with a "proven … contract" verdict**: emitting
   `proven_modulo_fuzz_validated_contract` for a result with no SMT-discharged
   contract is the over-claim this rule forbids (chelis#435). A fuzz-tier result
   is a fuzz-tier result.

2. **Verdict derives from the record, not from a label.** The composed verdict
   is a function of the rolled-up `(soundness, qualifier_set)` across the
   dependency set (master WI-6) — method, tier, samples, and assumptions — not a
   verdict string that can be wrong independently of them. A consumer that
   recomputes state from method and tier and a consumer that reads the verdict
   string must agree. Where they cannot, the structured fields are authoritative
   and the string is the defect.

3. **Method, tier, and scope are mandatory.** Every record carries its method,
   its tier, and its scope (over the reals vs over floats). "Proven" is
   **incomplete without "by this method, at this tier, over the reals, modulo
   this assumption."** Provenance of trust is the unit, not a single colored
   verdict.

4. **The goal is emitted, not reconstructed.** The discharged proposition
   travels with the record (chelis#436) so the consumer never has to rebuild it
   from source and risk displaying something other than what was proved.

5. **Soundness dependence is preserved.** Corrupting an assumption in a record's
   hypothesis must change the record's verdict. The non-vacuity guard (master
   WI-7) belongs to the producer: a green that survives corruption of its
   premises is empty, and the substrate must not emit one. A vacuous assumption
   set yields `invalid`, not a green; an unknown/timeout from the vacuity check
   is never silently green.

6. **Terminal failures dominate and are reported, not narrated away.** A
   disproof (`Failed`) and a vacuous assumption set (`Invalid`) sit off the
   guarantee lattice and dominate any green in the rollup (master WI-6). The
   substrate emits the failure mode and its evidence; it never downgrades a
   crash, an unsupported result, or a timeout into a pass.

## 4. Known substrate gaps that bound the contract

These are filed upstream and bound what the substrate can honestly emit today.
A consumer must degrade gracefully against them rather than assume the gap
away — and the gaps are de-narrowing targets, not permanent shape.

- **Transcendental / contract lowering to SMT is incomplete** (chelis#434).
  Transcendental finance properties (e.g. Black-Scholes positivity through the
  normal CDF) do not yet lower to SMT in standalone prove; such goals fall to
  fuzz or to the bound-propagation layer, and the record must say so via method
  and tier rather than presenting a non-SMT result as an SMT proof. The
  first-class special-function layer (sketch §L1) and Beacon's relaxations are
  the path that closes this.
- **`prove --json` over-claim on pure fuzz** (chelis#435): RESOLVED. A
  pure-fuzz base now renders the honest `fuzz_validated` badge (the
  `FuzzValidatedEmpirical` verdict carrying `fuzz_base`), never a `proven_*`
  badge; `proven_modulo_fuzz_validated_contract` is reserved for an SMT base
  discharged modulo a fuzz-validated contract. See §3.1.
- **`prove --json` omits the discharged proposition** (chelis#436): RESOLVED.
  Every property and obligation record now carries a `goal` field with the
  discharged proposition in canonical text, so a consumer never reconstructs it
  from source. The goal is the EXACT proposition discharged: an unguarded
  property's body; a GUARDED property's full `forall(...) where <pre>: <body>`
  form (the prover discharges `(/\ preconditions) => body`, so the bare body
  would over-claim an unconditional result); and an obligation's invariant
  predicate (metadata-stripped). The `goal` text re-parses to what was
  discharged -- the canonical formatter parenthesizes a compound expression
  used as a binary operand (chelis#461) so the displayed proposition cannot
  drift from the discharged one on re-parse. See §2 (claim) and §3.4.
- **Finance notation in authoring** (chelis#437): the front end rejects
  finance-standard uppercase single-letter value identifiers (`S`, `K`, `T`,
  `N`). This is upstream of the evidence schema — it constrains what models can
  be authored at all — but it is part of the same legibility story, because a
  claim the user cannot write is a claim the substrate never gets to discharge.

## 5. Relationship to existing specs

This document does not replace existing verification specs; it states the
consumer-facing integrity contract they must satisfy.

- `docs/trust_stack_verification.md` and `spec/design/chelis_trust_stack.md`
  describe the tiers and the layered trust story.
- `spec/design/verification_stack_master_plan.md` is the build-of-record:
  WI-4 (engine-independent goal), WI-6 (qualifier-set verdict algebra), WI-7
  (non-vacuity guard), and WI-8 (per-assumption discharge provenance) are the
  work items that produce records meeting this contract.
- `spec/design/verification_stack_sketch.md` is the narrative those work items
  formalize, including the honest-composite framing.
- `spec/design/beacon_plan.md` is the bound-propagation engine whose records
  are first-class peers under §2.

The single addition here is the **canonical record shape and its honesty
requirements stated as a producer/consumer contract**, so a downstream
legibility layer has a fixed surface to depend on.
