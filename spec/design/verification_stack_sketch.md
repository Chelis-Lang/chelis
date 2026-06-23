# The Chelis Verification Stack: Whole-Stack Sketch

Intended location: `spec/design/verification_stack_sketch.md` (chelis repo).
Companion documents: the master plan (`spec/design/verification_stack_master_plan.md`), the dependency map (`spec/design/verification_stack_dependency_map.md`), the Beacon engine plan (`spec/design/beacon_plan.md`), the VNN-LIB front-end placeholder (`spec/design/vnnlib_frontend_placeholder.md`), and the composed-verdict evidence-schema contract (`spec/design/composed_verdict_evidence_schema.md`), which fixes the canonical evidence record and honesty requirements a downstream legibility consumer depends on.

A high-level architecture sketch, not a build plan. It states the shape of what we are proposing, what is already shipped versus changing versus new, and the dependency relationships between pieces. Sequencing, and the deeper component-evaluation research, are named as separate passes at the end. The numbered build-of-record is the master plan and the dependency structure is the dependency map; this sketch is the narrative those two formalize.

## Thesis

One stack that verifies properties of numerical programs by dispatching each goal to the engine whose paradigm fits its shape, and aggregating the heterogeneous results behind a single honest verdict, running on the Chelis tensor IR, so the thing verified is provably the thing executed, and so the same engine verifies a pricing surface and a neural network. The crown is Beacon, an in-house, IR-native, sound bound-propagation engine that makes finance verification and ML verification the same problem. The external specialists fill what Beacon and cvc5 cannot close, and the long-run direction is to pull those capabilities in-process and in-house wherever a Rust or Chelis-native path exists.

## What Chelis uniquely does versus using these tools raw

This is the moat, and it is the reason the proposal is not "wire up cvc5, dReal, and MetiTarski yourself." Anyone can download those solvers. What no one else has is the integrated compiler and IR underneath them.

**Gap-free: the verified artifact is the executed artifact.** Using solvers raw, you hand-translate your program into each solver's input language, and that translation is itself unverified: you prove things about an SMT encoding that may or may not match the code that runs. In Chelis the program lowers once to the tensor IR, and every engine consumes a goal derived from that single IR through a compiler pass, which can itself be made trustworthy (tested semantics via Hull, eventually mechanized). There is no gap between what you proved and what runs. Raw tool use can never close that gap.

**One substrate makes the engines composable and makes finance equal ML.** Because cvc5, Beacon, and the rest all dispatch from one IR-derived goal, the orchestrator can route by goal shape and aggregate honestly. And because Beacon runs on the tensor IR, the same engine bounds a pricing graph and a neural net, since Hydronnx lowers ONNX into the identical RISC DAG, so this is mechanism, not slogan. Raw tools share no substrate; you cannot compose their verdicts honestly, and you cannot point one engine at both domains.

**First-class special functions the compiler reasons through.** The compiler recognizes erf, the normal CDF, and the activation family as primitives carrying their true contracts and sound relaxations, with the implementation-versus-primitive error bound discharged once and quarantined. Used raw, every proof re-derives properties from an open-coded approximation, dragging its branches through every downstream proof. This is a language and compiler capability that has no equivalent at the solver level. (This first-class-primitive layer is a generalization off the near-term path: it is featured here and in the dependency map but is not carried as a numbered work item in the master plan, whose WI set targets the orchestrator, the honesty layer, Beacon's dependencies, and the swept-in backends. The existing pricer bounds `erf64` directly and does not need it.)

**Verified automatic differentiation.** `grad` produces the adjoint as a graph in the same RISC IR, so Beacon over the gradient graph yields provably bounded Greeks (master WI-10). The differentiation and the verification share one representation. Used raw, your AD tool and your verifier never talk, and verified sensitivities are simply unavailable.

**The honest composite.** A verdict algebra that aggregates exact, delta-complete, sound-over-approximation, certificate-bearing, and fuzz discharges without ever laundering a weak guarantee into a strong badge (master WI-6), possible only because every engine feeds a verdict into one schema derived from one property surface. Used raw, you get a pile of incomparable answers and no principled way to combine them.

Net: the compiler and IR turn a set of incompatible raw solvers into a composable, honestly aggregated, gap-free verification stack where the verified thing is the executed thing, and the same substrate spans finance, ML, and scientific computing. That is the defensible position.

## The stack, in layers

Marked by state: shipped, changing, or new. Arrows are dependency edges, not a schedule. Master-plan work items (WI-N) and Beacon work items (WI-B1 through WI-B9) are cited inline so the layers tie back to the numbered plans.

**L0, the substrate: the Chelis language and tensor IR (RISC DAG).** Shipped. Changing: stabilize and version the op subset the verifier targets, so Beacon's transformers pin to a known surface (master WI-2).

**L1, compiler features for verification.** This is where most of the new compiler work lives, and it is cross-cutting, not a bolt-on.
- *First-class special functions* (new, generalization, not a numbered master-plan WI): a recognition layer (type system and verifier see `erf`/Phi/activations with their contracts and relaxations) over a codegen layer (lowering to a chosen implementation), bridged by a single discharged error-bound lemma per function. Curated set, not all of `scipy.special`; the normal family and the activation family are the natural first tier, oscillatory functions later or never.
- *Verified AD* (shipped; confirmed `grad` emits an adjoint RISC graph). Changing: treat the gradient graph as a first-class verification target so bounded Greeks dispatch like any other property (master WI-10).
- *Graph-extraction seam* (new, master WI-3): the recon confirmed the IR is discarded at the SMT-lowering boundary, so a property bound for Beacon needs a path to its IR DAG that does not route through the SMT property. The dispatcher therefore forks above SMT lowering.
- *Property goal schema* (changing, master WI-5): scalar input boxes and output-range assertions are expressible today as ordinary preconditions and postconditions; a native interval-box / output-range goal shape (Beacon's verdict form) is schema work on top.

**L2, engines (discharge backends behind one interface).**
- *cvc5* (master WI-11): shipped, in-process via FFI. The exact engine for the polynomial and logical core. Stays.
- *Beacon* (its own plan, WI-B1 through WI-B9): new, the centerpiece. In-house Rust, IR-native bound propagation in the CROWN lineage (interval to zonotope to linear relaxation, with branch-and-bound). Its real work, per the recon, is a perturbed-branch policy (the same problem as ReLU), reciprocal and cast transformers, the special-function relaxations from L1, and a transformer-soundness oracle built on the concrete IR evaluator (not Hull, which is a language-conformance fuzzer). It plugs into the orchestrator and also runs standalone on Hydronnx-derived graphs for the VNN path.
- *The specialist capabilities* (master WI-12 through WI-17): the roster evolves here (next section).

**L3, orchestrator / dispatcher.**
- *Backend interface* (master WI-4): each engine takes a canonical, engine-independent goal (built on the property shape but retaining the IR back-reference Beacon needs) and returns a discharge carrying a soundness flag and a qualifier set.
- *Dispatch by goal shape* (master WI-9): polynomial and logical to cvc5; special-function inequalities to the native bound layer (MetiTarski as fallback); bounded-box transcendental to Beacon's interval layer or cvc5; large-graph and range properties to Beacon; polynomial-with-receipt to the certificate backend.
- *The honest composite* (new): a qualifier-set rollup (union of qualifiers, minimum soundness) rather than a scalar chain (master WI-6); a non-vacuity guard (the recon confirmed none exists, so this is added, not preserved) (master WI-7); and per-assumption discharge-tier provenance threaded from c-earchin emission through to the artifact (also confirmed absent today) (master WI-8).
- *Deep-prove lane* versus the interactive lane: shared async infrastructure for the heavy backends (master WI-9).

**L4, domain shells / products.**
- *Shoals* (finance): pricers in Chelis using the special-function primitives, verified through the orchestrator; C Proof as the commercial vehicle around verified AD/Greeks and verified pricing bounds. (Downstream pickup, master plan section 5.)
- *Hydronnx + Beacon* (NN verification): the same IR and the same engine; the entry into the VNN community. A VNN-LIB front-end is the missing piece for a competition entry (see the VNN-LIB front-end placeholder).

**L5, the trust ladder.** Tested semantics (Hull) for the lowering and the transformers; mechanized soundness (LaCaDiLE / Lean) for the transformers and the metatheory; certificate-bearing discharges for independent auditability. This ladder is what lets the product say "verified" rather than "we ran some checks."

## Engine roster evolution

The set of engines we investigated is a starting point to be improved, not a commitment. Two forces reshape it.

**Consolidate toward in-process Rust.** cvc5 (FFI, shipped), Z3 (mature crate), and Clarabel (Rust-native SDP) all link in-process; Beacon is in-house Rust; the interval substrate has good Rust crates. The shell-to-CLI pattern (serialize to a text format, run a process, parse stdout, vendor an unmaintained binary) is reserved for research tools with no in-process equivalent, which in practice is MetiTarski (special-function proofs) and, rarely, full CAD. dReal is the clearest drop: it is CLI-only, unmaintained, lacks erf, and its bounded-box-transcendental capability is largely covered in-process by Beacon's interval layer plus cvc5/Z3. This is also a trust-story decision: shipping and trusting stale research binaries is a real supply-chain cost in a verified-finance product, so in-process and in-house is cleaner for the audit narrative as well as the engineering.

**The special-function convergence is the prime rewrite opportunity.** The sound polynomial and rational envelopes you must build for Beacon's special-function relaxations are the same objects MetiTarski uses as its function bounds, and the same objects the L1 special-function primitive layer needs for its contracts. Build that envelope library once in Chelis/Rust and you get Beacon's relaxations, the primitive contracts, and most of MetiTarski's value from a single investment, with an in-process RCF discharge via Z3/cvc5 rather than a shelled QEPCAD. MetiTarski then narrows to a fallback for the special-function inequalities the native bounds cannot yet close. The certificate backend is similarly Rust-native: Clarabel for the SDP core, with the SoS reduction and exact rationalization written on top.

So the roster moves from "cvc5 plus three external CLI binaries plus in-house Beacon" toward "cvc5, Z3, and Clarabel in-process, Beacon and a native special-function-bounds layer in-house, and MetiTarski only as a shelled fallback." A deeper component-evaluation pass (below) should confirm the best-in-class foundation for each capability rather than defaulting to this set.

*Reconciliation (current state).* The deeper component-evaluation pass named below has since been run (the solver and library bake-off). Its settled roster and per-backend technical context are the build-of-record in the master plan section 4.5: cvc5 (WI-11), Z3 (WI-12), the Sollya envelope library (WI-13), the Arb/FLINT oracle (WI-14), Clarabel (WI-15), Carcara (WI-16), and MetiTarski as a shelled fallback (WI-17), with dReal dropped. This section records the reasoning that led there; the master plan records the conclusions and the build and deployment infra (WI-18 through WI-20).

## What has to change in the ecosystem to land it

The proposal is not a module you add beside the compiler; it grows a verification-aware middle through the compiler. Concretely, the changes that ripple outward: the special-function primitive layer (front-end, type system, verifier, codegen, AD); the graph-extraction seam and the property goal schema (so Beacon can be dispatched at all); the verdict algebra, vacuity guard, and discharge-provenance threading through c-earchin (the honesty machinery the original framing assumed existed and the recon showed is mostly to-build); the IR op-subset stabilization Beacon pins to; Beacon itself with its internal ladder and soundness oracle; and the domain shells (Shoals, Hydronnx) wired to dispatch through the orchestrator. The verdict-status precedence that ships today is the floor of the honesty layer; the guarantee-kind aggregation on top of it is the new part.

## How it maps to the audience

The three audiences and the unique value line up, with the special-function layer as the connective tissue.

Quant research and systematic trading at small-to-mid funds get a guarantee class they have never had: not "we tested it," but verified Greeks and verified pricing bounds over the trading range, on the function that actually runs, the C Proof angle. The NN-verification academic community recognizes Beacon as their own field's machinery (the CROWN lineage) on a new substrate, which is the credibility and recruiting story and the VNN-COMP path. Scientific programmers get first-class, verified special functions, which is close to table stakes for that domain and absent from anything statically typed today. The connective fact is that a named special-function primitive with a sound relaxation and an adjoint is the same object whether it is a finance Phi, an ML activation, or a scientific erf, so one layer serves all three, and the cross-domain claim is mechanism rather than marketing.

## Deliberately left for the next passes

- *Sequencing.* This sketch states dependencies, not order. Turning the layer dependencies into an execution plan is a separate pass; the dependency map gives the topological tiers.
- *Deeper component research.* A real evaluation of best-in-class foundations for each capability (bound-propagation cores, SDP solvers, in-process RCF, special-function bound libraries worth porting) rather than the ad-hoc candidate set named here. This pass was run as the bake-off; its conclusions are the master plan section 4.5 roster.
- *The one gating product decision.* Whether a contract certifies the shipped approximation or the mathematical function. The approximation is the honest target because it is what executes and it is a closed form reachable in-process; the first-class-primitive design quarantines the approximation-versus-real bridge to a single lemma, which is what makes this decision tractable rather than pervasive.
