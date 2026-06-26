# Goal Transformation Layer: Design Spec

Intended location: `spec/design/transformation_layer.md` (chelis repo). This is a unified spec spanning the orchestrator (chelis) and the engines it routes to (cvc5, Z3, Clarabel, Beacon); it is written to be exploded into chelis-side and Beacon-side work later. The Why3 platform is the design reference for the transformation catalog and the driver concept; none of its code crosses over (see Section 8).

## 1. Problem

The orchestrator today answers one question per goal: which engine discharges it. There is a prior question it does not yet answer in general: what shape does the goal need to be in for that engine to win. A raw proof obligation rarely goes to an engine in the form the property lowering produced it. Between the goal and a discharge sits a set of semantics-preserving rewrites that massage the goal into a form a specific engine can close.

We have exactly one such rewrite today, discovered empirically: a goal with a transcendental subterm that cvc5 times out on and Z3 cannot parse becomes trivial when the transcendental is replaced by its certified envelope bound and the residual polynomial fact is handed to Z3. That move is one instance of a general pattern. This spec defines the layer that makes such rewrites a first-class, reusable, sound, composable catalog rather than a collection of hand-found special cases.

## 2. Shape of the solution

A transformation is a sound function from a proof goal to a list of proof goals. It is semantics-preserving in the proof direction: if every output goal discharges, the input goal discharges. Transformations compose. The orchestrator applies a sequence of transformations to a goal before (or as part of) routing it to an engine, and recombines the sub-results under the existing guarantee lattice.

The layer has three components, and they have different build modes, which is the central design fact of this spec:

- The transformation catalog: the named rewrites themselves. Created by design, executed deterministically.
- The soundness harness: the mechanism that validates a transformation preserves discharge. Deterministic infrastructure, built first.
- The driver/policy layer: which transformations to apply before which engine. Authored initially, mechanically fitted once a corpus exists.

## 3. The transformation type

A transformation is a pure function `Goal -> Vec<Goal>` over the canonical engine-independent Goal (the existing discharge-seam Goal that retains an IR back-reference and carries the box/output-range shape). Properties:

- Pure and deterministic: same input goal yields the same output goals. This is what makes application reproducible and replayable.
- Sound in the proof direction: if all outputs discharge, the input is discharged. A transformation must never make an unprovable input look provable through any output.
- Composable: transformations chain, and a recorded sequence is replayable, so a discharge carries the transformation pipeline that produced it as provenance.
- Lattice-aware at recombination: when a transformation splits one goal into several that route to different engines, the sub-discharges recombine under the existing qualifier-set lattice (union of qualifiers, minimum soundness). A split never launders: a goal split into a Beacon sub-goal and a cvc5 sub-goal rolls up to the weaker of the two kinds, never to the stronger.

The catalog worth building for the current fragment, in rough priority:

- Goal splitting. A conjunctive goal becomes N sub-goals dispatched independently. This is the highest-value transformation and the one that is genuinely novel here (Section 4).
- Controlled definition inlining. Unfold a helper before dispatch, with the inline-trivial-versus-inline-all distinction: inline only non-recursive trivial definitions versus everything. Too much inlining buries the engine in irrelevant terms; too little hides the structure the engine needs. This is plausibly part of the nested-helper lowering gaps already tracked.
- Abstract-subterm. Replace a subterm an engine cannot handle with a sound over-approximation (a certified envelope for a transcendental), leaving a residual the engine can close. This generalizes the existing envelope-plus-Z3 decomposition into a reusable transformation.
- Compute/partial-evaluation. Discharge the parts of a goal that are concrete arithmetic by computing them rather than asking the engine to reason about them symbolically, shrinking what reaches search.
- Construct-elimination. Lower a construct a target engine lacks a theory for into one it has. The general shape of existing routing decisions (host control flow rewritten to tensor ops for Beacon; f64 SMT form rewritten to exact-rational polynomial form for Clarabel).

## 4. Goal splitting and the lattice: the part that is ours to invent

Why3 splits goals to route among provers of the same kind (all theorem provers). Here the portfolio is heterogeneous in kind: exact solvers, an SDP-backed certificate engine, a bound-propagation engine, and empirical methods. So splitting does something it cannot do in Why3: it routes one property to several different kinds of engine and recombines under the guarantee lattice.

Concretely, a conjunctive postcondition where one conjunct is polynomial (to cvc5 or Z3), one is a special-function bound (to the envelope path), and one is a range claim (to Beacon) is split so each conjunct reaches its own engine, and the composite verdict is the lattice rollup of the three sub-discharges. This is the marriage of Why3's transformation idea with the guarantee algebra, and neither system has it alone: Why3 has the splitting but a single-kind verdict; the orchestrator has the lattice but applies it today only across whole goals, not across the parts of a split goal.

Because this is where a split could launder a weak sub-result into a strong composite, splitting is built by hand first, with focused attention on the recombination, not folded into an agentic loop. The recombination invariant: a split goal's verdict is the lattice rollup over its sub-goal discharges, at minimum soundness, with the union of qualifiers, and is never stronger than the weakest sub-discharge.

## 5. The soundness harness: deterministic, built first

Every transformation carries a soundness obligation: it must preserve discharge in the proof direction. This is a property, and properties are mechanically checkable, so this is the highest-leverage deterministic infrastructure and it is built before any transformation is ported.

The harness, structurally identical to Beacon's Arb containment oracle pattern:

- A goal generator and a curated goal corpus (drawn from the chelis test suite, the finance fragment, and the existing prove corpus).
- An applicator that runs a candidate transformation against a goal.
- An oracle that confirms the soundness relation held: a goal that is discharged stays discharged after the transformation, and, the critical direction, the transformation never causes an undischargeable goal to be reported discharged.

The harness inverts the trust problem. The agentic part of the work (writing a rewrite) is gated by a deterministic part (the harness certifying it sound on the corpus). Soundness is not trusted from the author's reasoning; it is certified by the harness. An unsound rewrite fails the harness and does not ship.

Honest boundary: a corpus check is evidence, not proof. The harness validates a transformation on a finite corpus; the airtight soundness theorem for a transformation is a Lean-level obligation (the LaCaDiLE-adjacent long game). The harness is the practical gate; mechanized proof of the foundational transformations is the eventual hardening, in the same relationship interval/Arb evidence has to mechanized transformer soundness in Beacon.

## 6. The driver/policy layer: authored, then fitted

A driver is the per-engine policy: which transformation pipeline to apply before sending a goal to a given engine, plus that engine's own goal lowering and result parsing. The per-engine lowering already exists, each engine owns shaping a canonical Goal for itself (cvc5 keeps its SmtProperty pipeline internal). What is new is the shared, named, composable transformation pipeline applied above the per-engine lowering, plus the policy of which pipeline precedes which engine.

This has a deterministic endgame, but only after a corpus exists. Initially the policy is authored by hand or by an agent ("goals shaped like this, going to this engine, want this pipeline"). Once the catalog exists and each transformation is sound-by-harness, the question "which transformation sequence makes engine E discharge goal G" becomes an empirical, searchable question over a finite transformation space against a concrete success metric (discharge achieved, time to discharge). The driver becomes a fitted artifact derived from measured outcomes on the corpus, not a hand-asserted guess. The policy layer is authored initially and mechanically tuned once the corpus and harness are in place.

## 7. Build staging: what is deterministic, what is agentic, in what order

The split between deterministic and agentic work is sharp, and the failure mode is doing either half in the wrong mode. The deterministic half done agentically is slow, non-reproducible, and untrustworthy on soundness; the agentic half done deterministically is impossible, because it is design.

Stage 1, deterministic infrastructure, before any porting:
- The transformation type (`Goal -> Vec<Goal>`) and the application/recording machinery.
- The soundness harness (Section 5).
- The goal corpus.
None of this is agentic; it is the substrate.

Stage 2, agentic over a bounded list:
- Point a coding agent at Why3's transformation catalog and its cvc5 and Alt-Ergo drivers, and port the effects one transformation at a time, each implemented against the canonical Goal type, each gated by the soundness harness.
- The loop is bounded (a known, finite list of transformations) and trust-bounded (the harness, not the agent, certifies soundness).
- Start with the few that matter for the fragment: goal-splitting (highest value; built by hand first per Section 4), controlled inlining, abstract-subterm.

Stage 3, deterministic again, once the catalog and corpus exist:
- Mechanically fit the per-engine driver policy by searching transformation sequences against measured discharge outcomes on the corpus.
- This is deterministic optimization over a finite transformation space, not an agentic loop. The policy is derived, not authored.

The irreducibly agentic-or-human core is narrow: the design of each transformation against the IR (the semantic mapping Why3's representation cannot supply mechanically) and the soundness argument for each (harness-validated on the corpus; Lean-mechanized for the foundational ones as the long game). Everything around that core, validation, policy-fitting, application, is deterministic.

The discipline that gates the whole thing: the agent must never both write a transformation and judge its soundness in the same loop. That is where an unsound rewrite ships wearing a green. The harness is built first precisely so the agentic porting is gated by a check it cannot talk past.

## 8. Why this is a reference to Why3, not an adoption of it

Why3 is a deductive verification platform that generates verification conditions and dispatches them through composable transformations to a herd of external provers, with a driver per prover. Architecturally it is a proof orchestrator, and its accumulated value is precisely the transformation catalog and the driver knowledge: the decade of learned answers to which rewrite pays off, in what order, gated on which prover, and what breaks.

The catalog is harvested as ideas, not code, for three reasons:

- The transformations are entangled with Why3's task representation and WhyML's term language. Lifting the code means importing Why3's IR, which defeats the IR-native position the whole stack rests on.
- The insights, which rewrites pay off and in what order, the inline-too-much brittleness, the per-prover policy, are language-independent and are the actual value.
- The catalog here needs entries Why3 does not have, because the portfolio includes kinds it lacks. The abstract-a-transcendental-with-a-certified-envelope-and-route-the-residual transformation, and any rewrite toward Beacon's box/range goal shape, are ours to invent. Why3's catalog is the template for how to structure a library of such things, not a source to copy them from.

So the reference value is: read Why3's transformation list and its cvc5/Alt-Ergo drivers as a design document for this layer, a shared, composable set of sound goal-rewrites applied during routing, with each engine declaring its preferred pipeline. The provers are commodities; the transformation-and-driver knowledge is the hard-won part, and it is reusable as architecture and as a catalog of which-rewrite-when even though none of the code crosses over.

## 9. Where this touches chelis versus Beacon

To be exploded into per-repo work later; recorded here so the seam is explicit.

Chelis (the orchestrator side, most of the layer):
- The transformation type and the application/recording machinery, above the discharge-engine interface.
- The transformation catalog (splitting, inlining, abstract-subterm, compute, construct-elimination).
- The soundness harness and the goal corpus.
- The driver/policy layer and its eventual mechanical fitting.
- The lattice-aware recombination for split goals (an extension of the existing qualifier-set rollup to operate across the parts of a split goal, not only across whole goals).

Beacon (the engine side, consumer of the layer):
- The box/range goal shape is one of the split targets, so the abstract-subterm and construct-elimination transformations that produce Beacon-bound sub-goals must emit the box/range form Beacon consumes through the existing seam.
- No transformation logic lives in Beacon; Beacon receives shaped goals like any engine. The Beacon-relevant work is ensuring the transformations that target it produce goals in the frozen seam shape (name-keyed box, serialized WireDag addressed by hash, root index), and that a Beacon sub-discharge carries its real (soundness, qualifier) into the split recombination.

Shared invariant across the seam: a transformation that routes a sub-goal to Beacon and another to an exact solver recombines under the lattice at minimum soundness, so a split that touches Beacon can never render stronger than Beacon's sound-over-approximation qualifier allows.

## 10. Acceptance shape

Not a build plan; the gates that define the layer working.

- Stage 1 is done when a transformation can be defined, applied, recorded, and validated against the corpus by the harness, with no transformation yet ported.
- A transformation is admitted only when it passes the soundness harness on the corpus; an unsound rewrite is structurally unable to ship.
- Goal-splitting is admitted only with its recombination invariant proven on the corpus: a split goal's verdict is the lattice rollup of its sub-discharges, never stronger than the weakest, with non-vacuity preserved per sub-goal.
- The driver policy is fitted, not asserted, once the corpus is large enough to measure discharge outcomes across transformation sequences.
- The existing prove corpus remains byte-identical for goals the layer does not intentionally transform; introducing the layer changes no current verdict except where a transformation is deliberately applied and measured to help.
