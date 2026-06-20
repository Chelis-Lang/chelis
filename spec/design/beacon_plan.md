# Beacon Engine: Specification and Implementation Plan

Intended location: `spec/design/beacon_plan.md` (chelis repo).
Companion documents: the master plan (`spec/design/verification_stack_master_plan.md`), the whole-stack sketch (`spec/design/verification_stack_sketch.md`), the dependency map (`spec/design/verification_stack_dependency_map.md`), and the VNN-LIB front-end placeholder (`spec/design/vnnlib_frontend_placeholder.md`).

## 1. What Beacon is

Beacon is an in-house, IR-native engine that proves output bounds for numerical programs by forward propagation of sound bounds through the chelis tensor IR. Given an input box, it pushes a numeric envelope forward through the RISC DAG using abstract domains in the CROWN lineage (interval, then zonotope, then linear relaxation), tightening with branch-and-bound, and reads a provable output range off the end. It is not a solver Beacon calls; it is an engine Beacon owns, operating on its own IR.

Two properties make it the centerpiece. It scales: bound propagation is close to a forward pass over the graph rather than a combinatorial search, so it bounds large numerical programs where cvc5 and even bounded-domain SMT never finish. And because it runs on the tensor IR, the same engine that bounds a pricing surface bounds a neural network; Hydronnx lowers ONNX into the identical RISC DAG, confirmed by recon, so this is mechanism, not slogan. Every Beacon discharge is sound (the true value is always inside the returned bound) and is tagged sound-over-approximation, because the bound may be looser than reality; branch-and-bound tightens it.

## 2. Scope and interfaces

Beacon consumes: the graph-extraction seam (master WI-3) for the DAG, input box, and output-range goal; the special-function envelope library (master WI-13) for its transcendental relaxations; the Arb/FLINT oracle (master WI-14) for the soundness oracle; and the versioned IR op subset (master WI-2) as its target surface. It produces sound-over-approximation discharges through the discharge-engine interface (master WI-4), and it also runs standalone against Hydronnx ONNX-derived IR for the neural-network path. The concrete IR evaluator (`eval_tensor`) already exists and backs the soundness oracle.

## 3. Architecture

### 3.1 The abstract-domain ladder

Beacon's domain support is a ladder, and the complexity is uneven across it. The interval domain is contained: simple per-op interval arithmetic, sound but loose, useless on a deep graph because of the dependency (wrapping) problem. The zonotope domain is substantial: it tracks linear dependence between variables, so it does not lose the slack interval arithmetic does on correlated quantities. The linear-relaxation domain with a backward pass is the research-grade core: it is the CROWN method, producing tight linear bounds, and it is what makes Beacon prove anything useful on a real graph.

Each stage is shippable behind the same discharge interface. The interval stage lands the interface integration, the soundness oracle, and the honesty wiring while proving nothing tight; the tight-bounds math comes after and behind the same seam. This decoupling is deliberate: Beacon-as-a-registered-engine and Beacon-with-useful-bounds are separable.

### 3.2 Per-op transformers

A corpus of sound abstract transformers, one family per RISC op Beacon covers. `Add`, `Mul`, `Exp`, `Log`, and `Sqrt` are ports of the relaxations the auto_LiRPA corpus already implements. `Div` and `Recip` need a reciprocal transformer with a denominator-excludes-zero precondition. `Cast` raises the float-versus-real question (section 6): real-valued first, with a roundoff-aware cast transformer as a later layer. Any op with no published relaxation is an in-house transformer, not a blocker. Transformers are kept small and pure so the soundness oracle can test each, and so each stays individually mechanizable for an eventual Lean path.

### 3.3 Perturbed-branch policy

This is the actual hard part of the finance fragment, and it is research-grade. The pricer's `erf64` is an Abramowitz-Stegun approximation with input-dependent branches: a sign fold at `x < 0` and a small-`x` branch, lowered as `CmpLt` predicates. These are the same class of object as a ReLU: a piecewise definition with an input-dependent predicate. Bounding through them soundly requires branch-and-bound that splits on the predicate, in the manner of GenBaB's general-nonlinearity branch-and-bound in the alpha-beta-CROWN line. The policy decides when to split, how to relax the unsplit branch soundly, and how to recombine, and it is shared with any other input-dependent branching in a target graph.

### 3.4 Special-function relaxations

Beacon's `erf` relaxation consumes the master WI-13 saturation-plus-central envelopes. Outside roughly plus or minus six, the envelope is the saturating constant bound near plus or minus one; inside the central box, the envelope is the certified polynomial pair. The real `erf`-argument range on the desk box reaches the hundreds, so the saturation arm is not an optimization, it is required for the relaxation to be sound and not absurdly loose over the actual range. The activation family follows the same pattern when the primitive layer arrives, but Beacon does not require that layer: for the existing pricer it bounds the `erf64` graph directly, treating the approximation as the function it is.

### 3.5 Soundness oracle

The practical guardrail against an unsound transformer. For a claimed bound over a box, sample concrete inputs, run the concrete IR via `eval_tensor`, compute the rigorous true enclosure via the Arb oracle (master WI-14), and assert the concrete output sits inside Beacon's claimed bound. This does not prove a transformer sound, but it catches unsoundness empirically and cheaply, and it reuses existing concrete evaluation. Build it alongside the interval stage so the harness exists before the hard transformers do.

### 3.6 Dispatch integration and standalone

Beacon plugs into the orchestrator (master WI-9) as a sound-over-approximation discharge engine through the discharge-engine interface, receiving a goal as DAG plus input box plus output-range assertion from the graph-extraction seam. It also runs standalone against Hydronnx ONNX-derived IR, which is the same RISC DAG type, for the neural-network path; that standalone path needs neither the orchestrator nor a finance graph and is the cleanest way to validate the transformer math in isolation. The VNN-LIB front-end that would make the standalone path a competition entry is a separate placeholder document.

## 4. Algorithm corpus and validation

Beacon ports the CROWN, auto_LiRPA, and GenBaB algorithm corpus (the five-time VNN-COMP-winning alpha-beta-CROWN lineage), which is Python and PyTorch and neural-network-coupled, so it is a port-the-algorithm relationship, not a dependency. Two validation tracks. First, a differential oracle: run alpha-beta-CROWN with GenBaB on shared ONNX, including the Black-Scholes pricer graph the bake-off already exported faithfully (numerical fidelity to the real pricer confirmed), and compare Beacon's bounds against it. Second, evaluate Luna, a 2026 bound propagator built specifically to be integrated into other tools with low overhead, as a possible foundation to build on rather than reimplementing from CROWN. The bake-off prepared the faithful pricer ONNX but did not complete the alpha-beta-CROWN run, so the achievable-tightness-on-the-real-branch-structure measurement is the still-open input to the port-versus-build-on-Luna decision; that measurement is the first thing the build resolves, and it is cheap now because the ONNX is ready.

## 5. Work items

Numbered for reference, not sequence; the dependency edges define what precedes what.

**WI-B1 Interval forward stage.** Interval-domain forward propagation over the RISC DAG, wired through the discharge interface, with the soundness oracle. Proves nothing tight; lands the seam, the oracle, and the honesty integration. Depends on master WI-2, WI-3, WI-4, and WI-14.

**WI-B2 Per-op transformer corpus.** The transformer families for the covered ops, ported and in-house, each oracle-tested. Depends on WI-B1.

**WI-B3 Zonotope domain.** The dependence-tracking domain. Depends on WI-B1, WI-B2.

**WI-B4 Linear-relaxation domain with backward pass.** The CROWN-method core that produces useful bounds. Research-grade. Depends on WI-B3.

**WI-B5 Branch-and-bound and the perturbed-branch policy.** Splitting on input-dependent predicates, relaxing the unsplit branch, recombining. The finance-fragment hard part. Depends on WI-B4.

**WI-B6 Special-function relaxation wiring.** Consume the saturation-plus-central envelopes as the `erf` relaxation. Depends on WI-B2 and master WI-13.

**WI-B7 Soundness oracle.** Box sampling, concrete evaluation via `eval_tensor`, rigorous enclosure via Arb, containment assertion. Built alongside WI-B1. Depends on master WI-14 and `eval_tensor`.

**WI-B8 Dispatch integration.** Register Beacon as a sound-over-approximation engine behind the discharge interface; the gradient-graph path for verified Greeks. Depends on master WI-4, WI-9, and WI-B4.

**WI-B9 Standalone and Hydronnx path.** Run Beacon directly on Hydronnx ONNX-derived IR for the neural-network track. Depends on WI-B4. The VNN-LIB front-end is a separate placeholder.

## 6. The float-versus-real decision

Beacon bounds over the reals first, as the entire CROWN lineage and the reference verifiers do; this is the standard first target and it is what the verified-Greeks and verified-pricing-bounds story needs. Floating-point roundoff soundness is a later, separate layer, surfaced through the `Cast` transformer (section 3.2): a roundoff-aware cast transformer would model machine roundoff and make the bound sound with respect to the executed floating-point program rather than its real-valued idealization. Its cost is scoped against the real-valued baseline once that baseline exists; until then, Beacon discharges carry the real-valued qualifier in their qualifier set so the composite never implies roundoff soundness it does not have.
