# End-to-End Differentiable Programming in Chelis

## Goal

Make Chelis a fully differentiable language: AD composes through arbitrary program structure (control flow, user-defined functions, custom data, effects, fixed points), not just tensor operations. The compiler treats AD as a first-class language transformation rather than a library feature, with type-level expression of differentiability and verified properties about gradient behavior.

## Audience and value proposition

The target audience is researchers and engineers working on problems where the standard ML framework boundary doesn't fit:

- Differentiable physics, rendering, and scientific simulation
- Probabilistic programming with rich inference machinery
- Neural-symbolic integration
- Optimization-as-a-layer
- Inverse problems with verified bounds
- Algorithmic differentiation through dynamic programming, sorting, ranking

These users today work around limitations in PyTorch and JAX: rewriting simulators as tensor operations, replacing real algorithms with soft relaxations, hand-coding backward passes for non-tensor components, restructuring control flow to fit a tracing framework's constraints. A fully differentiable Chelis lets them write programs as programs and get gradients without the workarounds.

The differentiator versus existing differentiable-language work (Dex, partial-JAX, Julia/Zygote, Enzyme) is the combination Chelis already commits to: dimension types from the start, effects as first-class, AD as IR transformation, verified properties. Each existing differentiable language has some of these; none has all. Chelis with this work shipped has all.

## Pre-locked decisions

Before any agent dispatches, the following are locked contracts:

**Decision 1: Forward-mode AD lives alongside reverse-mode.** Reverse-mode covers the dominant ML use case (gradients of scalar loss w.r.t. many parameters). Forward-mode covers cases reverse-mode is wrong for (Jacobians of vector-valued functions w.r.t. few inputs, sensitivity analysis, certain implicit-function applications). Both are first-class language transformations.

**Decision 2: AD is a transformation, not a runtime library.** The compiler emits forward and reverse passes at IR-transformation time. There is no `Tensor.grad()` accumulator pattern; gradients flow through explicit transformations applied to the program.

**Decision 3: Differentiability is type-level.** A function carries information about whether it is differentiable, partially differentiable, or non-differentiable. The type system catches AD applied to functions that don't admit it. Differentiability is a property the compiler can verify, not a convention.

**Decision 4: Effect handling is the gradient discipline for stochastic and stateful operations.** Different effects have different gradient interpretations. The compiler dispatches the AD transformation based on the effect, not on a global flag. Reparameterization, REINFORCE, and pathwise gradients become handler choices.

**Decision 5: Implicit differentiation is a marked construct.** Fixed-point computations and optimization-as-a-layer get explicit syntactic markers (`fix`, `argmin`, etc.) that tell the compiler "this is defined implicitly; use the implicit function theorem for differentiation." Iterative computations without markers get the standard reverse-mode treatment.

**Decision 6: Custom data structures admit AD via field-wise gradient rules.** ADT and record types get gradient versions automatically. A struct of tensors becomes a struct of gradient tensors. Pattern matching gradients dispatch the same way as the forward pattern match.

**Decision 7: Backward compatibility with existing AD.** Programs that work with today's AD continue to work. The expansion adds new capabilities; it doesn't change semantics of existing differentiable code.

## Phases

The work decomposes into seven phases, sequenced by dependency.

### Phase 0: Spec lock

Single agent. Lands the canonical spec document at `spec/design/differentiable_language.md`. Pins the contracts above plus the syntactic surface:

- The differentiability annotation on function types (the type-level marker that distinguishes differentiable, partially differentiable, and non-differentiable functions)
- The AD-mode markers in source (`grad`, `vjp`, `jvp`, `jacobian`, `hessian` as language-level operators rather than library functions)
- The effect-to-gradient mapping table (reparameterization for Normal/continuous, REINFORCE for Categorical/discrete, pathwise for transformations of base distributions)
- The implicit differentiation syntax (`fix`, `solve`, `argmin` markers and their gradient rules)
- The ADT/record gradient derivation rules (field-wise extension)
- The interaction with existing language features (effects, linearity, dimension types)

This document is the contract every subsequent phase implements against. It supersedes scattered AD documentation in the existing spec.

### Phase 1: Control flow AD

Closes the IR-SelectOp-F1 and IR-MatchLowering-F1 §5 entries from the broader compiler work, plus extends them with AD rules.

**Scope.**

- `if/else` differentiation: forward pass evaluates the chosen branch, backward pass propagates gradients through that branch only. The non-chosen branch contributes zero. This is mathematically correct for differentiable predicates only; for non-differentiable conditions (e.g., `if x > 0`), the gradient at the discontinuity is undefined — Chelis emits a clear diagnostic indicating the function is non-differentiable at boundary points.
- `match` differentiation: same shape as `if/else`. Each arm gets the gradient if selected. The pattern match itself is non-differentiable; gradient flow goes through the matched values to the matched expression.
- `while` and `for` differentiation: reverse-mode requires storing the trajectory (every iteration's intermediate values) for the backward pass. Compiler emits the trajectory storage automatically. Memory cost is real; programs with long loops have correspondingly large gradient memory. For forward-mode, no trajectory storage needed (forward AD is space-efficient).
- Recursive function differentiation: handled the same as loops via the trajectory pattern. Stack depth in the forward pass becomes trajectory length in the backward pass.

**Backend changes.**

The C and HIP backends gain support for `RiscOp::Select { cond, true_val, false_val }` (closes IR-SelectOp-F1). The match lowering closes IR-MatchLowering-F1 with the tag-compare-and-branch pattern plus structural destructuring primitives. The backends emit trajectory-storage code for loops in reverse-mode AD contexts.

**Tests.**

Numerical agreement between forward-pass-then-backward-pass and an analytical gradient computation for representative branching and looping functions. Specifically:

- A piecewise function (`if x > 0 { x*x } else { -x*x }`) and its analytical gradient
- A while loop computing a Taylor series and its analytical gradient
- A recursive function (factorial-like, Newton iteration) and its gradient
- A match on an ADT carrying gradient-relevant data

**Output.** AD composes through `if`, `match`, `while`, `for`, and recursive functions. Programs with arbitrary control flow admit forward and reverse mode differentiation. The compiler emits clear diagnostics when differentiability is structurally impossible (non-differentiable discontinuities, mutation across boundaries).

### Phase 2: ADT and record gradients

Closes the gradient story for user-defined data structures. Composes with IR-FirstClassFn-F1 from the existing §5 list.

**Scope.**

- For each ADT and record type defined in user code, the compiler synthesizes a corresponding gradient type. A struct `{ x: f32, y: tensor[3, f32] }` produces a gradient struct `{ x: f32, y: tensor[3, f32] }` (same shape; gradient values represent the gradient with respect to each field).
- Constructors and field accessors get gradient rules automatically. Forward pass: standard construction and access. Backward pass: gradient through the field comes from the corresponding field of the gradient struct.
- ADT pattern matching gets gradient rules. Pattern matches on a variant produce gradients that flow back to that variant's payload.
- Sum-type ADTs (where different variants have different payloads) get gradient struct shapes where the gradient corresponds to whichever variant was constructed. Other variants contribute zero gradients to inactive components.
- Higher-order functions (functions taking functions, returning functions) get AD rules. Composes with IR-FirstClassFn-F1.

**Tests.**

- A struct holding parameters of a small model. Computing gradient of a loss w.r.t. the struct produces the field-wise gradient struct.
- An ADT representing a computation tree. Pattern matching on tree nodes during evaluation produces gradients that flow back to the tree's leaves.
- A higher-order function (map, fold) applied to a list of tensors with a differentiable kernel. Gradients flow through the higher-order structure.

**Output.** User-defined data structures don't break gradient flow. Code that uses domain-natural representations (a struct for model parameters, an ADT for a computation graph, a record for a state vector) gets gradients without rewriting to flat tensor form.

### Phase 3: Effect-aware AD

Specifies how AD interacts with each effect class in Chelis. Composes with the existing effect system; doesn't add new effects, just adds gradient rules to existing ones.

**Scope.**

- Pure functions (no effects): standard reverse and forward mode as today.
- `state` effect: AD treats state mutations as part of the computation. Reverse mode replays state values from a trajectory store. Forward mode tracks state-derivative alongside state-value.
- `raises` effect: AD over a function that may raise. If the function raises in the forward pass, the backward pass propagates zero gradients (the raised path didn't contribute). If a handler converts the raise into a value, AD flows through the handler.
- `capability` effect: capability access is non-differentiable (it's a runtime authority check). AD treats capability boundaries as fixed; gradients flow through the values returned by capability-protected operations but not through the capability check itself.
- Stochastic effects (the new one): a `sample` effect represents drawing from a distribution. The gradient interpretation depends on the distribution and the handler. The compiler dispatches:
  - Reparameterizable distributions (Normal, Uniform, Beta with valid parameters): pathwise gradient. The sample is expressed as a transformation of a fixed base distribution; gradient flows through the transformation.
  - Non-reparameterizable distributions (Categorical, Bernoulli, discrete): REINFORCE-style score-function gradient. The handler emits the log-prob multiplier automatically.
  - Mixed cases (some parameters reparameterizable, others not): rao-blackwellized or partial-reparameterization, dispatched per-parameter.

**The probabilistic programming surface.** This phase unlocks Chelis as a PPL substrate. A user writes a probabilistic program (sample latents, condition on observations) and gets gradient-based inference (variational inference, HMC) by composing the standard AD operators with the sample handler. The framework-level PPLs (Pyro, NumPyro, Gen) become expressible as small Chelis libraries.

**Tests.**

- A function that samples from `Normal(mu, sigma)` and computes a loss. Gradient w.r.t. `mu` and `sigma` via reparameterization, verified against the analytical gradient.
- A function that samples from `Categorical(probs)` and computes a loss. Gradient via score function, verified against finite-difference baseline.
- A variational inference loop on a small Bayesian model. The loop converges and produces parameter estimates within tolerance of analytical posterior.
- A function with `raises` that AD flows through correctly when the raise happens and when it doesn't.

**Output.** Effects compose with AD. Stochastic programs get gradients via the right rule per distribution. State-carrying programs get gradients through the state-passing structure. The Chelis-as-PPL story becomes credible.

### Phase 4: Implicit differentiation

Adds language-level markers for fixed-point and optimization-as-a-layer patterns. Implements the implicit function theorem application that gives gradients through these constructs without unrolling the iteration.

**Scope.**

- A `fix f x` construct: compute the fixed point of `f` starting from `x`. Gradient through `fix` doesn't unroll the iteration; it solves the linear system from the implicit function theorem.
- An `argmin f x` (and `argmax`) construct: compute the minimizer (or maximizer) of `f` over `x` (or starting from `x` if `f` is a multi-arg function with one input being the optimization variable). Gradient through `argmin` uses the KKT conditions or the unconstrained-optimum implicit differentiation rule.
- A `solve f x` construct: solve `f(x) = 0` for `x`. Gradient via the implicit function theorem applied to `f`.
- The compiler emits the implicit-gradient computation automatically. For `fix` and `solve`, this is a linear solve against the Jacobian of `f` at the fixed point. For `argmin`, it's the KKT-derived implicit gradient.

**Examples enabled.**

- Equilibrium-finding in physics simulators: `fix` to find equilibrium state, gradient through the equilibrium for parameter inference.
- Optimization-as-a-layer: a model embeds `argmin` of an inner problem; outer training gets gradients through the inner solution.
- Differentiable algorithms: many algorithms (matching, shortest path with relaxation, certain optimization-based formulations) reduce to `argmin` or `solve` structures.

**Tests.**

- A simple fixed-point computation (`x = 0.5 * x + 0.5 * y` for varying `y`) and the gradient via implicit differentiation, verified against unrolled-gradient baseline.
- A quadratic `argmin` (solvable analytically) and the gradient through the optimum.
- A small constrained optimization with explicit KKT conditions and the verified implicit gradient.

**Output.** Programs that compute via fixed points, optimization, or implicit solves admit gradient flow without space-expensive unrolling. The differentiable-simulation use case (physics, equilibrium) becomes tractable.

### Phase 5: Differentiability typing

Adds the type-level expression of differentiability. Functions carry differentiability information; the compiler verifies it.

**Scope.**

- A function type extends with a differentiability marker: `Differentiable`, `PartiallyDifferentiable`, or `NonDifferentiable`. The marker is inferred from the function body for unannotated functions; explicit annotations are checked.
- A `Differentiable` function: all paths through the function admit gradient flow. AD applies and produces a well-defined gradient.
- A `PartiallyDifferentiable` function: gradient flow works on some inputs (a subset of the domain). The function carries a condition describing where gradients are defined. AD applied at a non-differentiable point produces a runtime error or zero gradient depending on configuration.
- A `NonDifferentiable` function: gradient flow doesn't work. AD applied to it is a type error.
- Type inference for differentiability composes through function calls. A function calling only `Differentiable` functions is `Differentiable`. A function calling a `NonDifferentiable` function is at most `PartiallyDifferentiable`.

**Verification connection.** Differentiability properties become attachable. A user can write `property f is Differentiable` or `property f satisfies Lipschitz(K)` and the harness verifies it by sampling. The differentiability type discipline plus the hull/c-earchin work compose: a function can carry properties about its gradient (monotonicity, sign preservation, magnitude bounds) that get verified.

**Tests.**

- A function annotated `Differentiable` whose body uses only differentiable operations type-checks.
- A function annotated `Differentiable` whose body uses a `NonDifferentiable` callee fails type-checking with a clear diagnostic.
- An unannotated function gets its differentiability inferred and reported via `chelis check --show-inferred`.
- A property `f satisfies Lipschitz(0.5)` gets verified by sampling input pairs and computing finite-difference Lipschitz estimates.

**Output.** Differentiability is a first-class property of functions, checked at compile time. Users can express "this code requires gradient flow," and the compiler enforces it. The verification layer extends to gradient-behavior properties.

### Phase 6: Documentation, examples, and ecosystem

The capability shipped in Phases 1-5 needs to be discoverable and usable.

**Scope.**

- A primer document targeting the differentiable-programming audience. Concrete examples per use case: differentiable physics, neural-symbolic, probabilistic programming, optimization-as-a-layer, inverse problems.
- Worked examples in `examples/differentiable/`: a small physics simulator with parameter inference, a small PPL example with VI, a small optimization-as-a-layer pattern.
- Shell library `chelis-diff` that bundles common patterns: standard distributions for the stochastic effect, common optimizer combinators (Adam, SGD, L-BFGS), implicit-differentiation helpers, finite-difference verification utilities.
- Integration documentation showing how Chelis fits relative to PyTorch and JAX for the target use cases.

**Output.** A researcher coming from the PyTorch/JAX world has a clear on-ramp. The capability is documented, the use cases are explicit, the shell library provides the standard tools.

## Dependencies and sequencing

```
Phase 0 (spec lock)
   ↓
Phase 1 (control flow AD)  ←──── depends on IR-SelectOp-F1, IR-MatchLowering-F1
   ↓
Phase 2 (ADT/record gradients)  ←──── depends on IR-FirstClassFn-F1
   ↓
Phase 3 (effect-aware AD)
   ↓
Phase 4 (implicit differentiation)
   ↓
Phase 5 (differentiability typing)
   ↓
Phase 6 (docs, examples, ecosystem)
```

Phases 1-5 are sequential because each builds on the prior. Phase 6 can develop in parallel with Phase 5 once Phase 4 lands.

The dependency on the existing §5 entries (IR-SelectOp-F1, IR-MatchLowering-F1, IR-FirstClassFn-F1) means those workstreams move from "deferred until customer shape forces" to "required for the differentiable-language direction." If this direction is committed to, those §5 entries get prioritized.

## Workstream sizing

Each phase decomposes into multiple agent dispatches with the same discipline as other Chelis workstreams: failing fixtures, diagnosis, fix, sibling sweep, red-team.

Phase 0 is a single spec PR.

Phase 1 is the largest single phase because it spans IR-level changes (Select op, match lowering), AD rule additions (per control-flow construct), and backend changes (trajectory storage). Plausibly 5-8 agent dispatches.

Phase 2 is medium. ADT and record gradients have a clear pattern (field-wise extension) but apply across multiple language constructs. Plausibly 3-5 dispatches.

Phase 3 is medium-to-large. Each effect class has its own gradient rule design. The stochastic-effect work is the most involved because the distribution-specific dispatch and the handler integration are both real design surfaces. Plausibly 5-7 dispatches.

Phase 4 is medium. Three constructs (`fix`, `argmin`, `solve`), each with its own implicit-gradient rule. Plausibly 3-4 dispatches.

Phase 5 is medium. Type system extension plus inference plus property integration. Plausibly 3-5 dispatches.

Phase 6 is variable depending on the scope of documentation and examples ambition.

## What's not in scope

**Distributed AD.** Computing gradients across multiple machines or accelerators is a separate workstream. This spec covers single-device AD that composes through arbitrary program structure. Multi-device gradients (data parallelism, model parallelism, ZeRO-style sharding) require additional effect machinery and runtime coordination that's a distinct project.

**JIT compilation of AD-transformed programs.** The AD transformations described here happen at IR-transformation time, producing static forward and backward functions. JIT recompilation per-input (in the manner of JAX's `jit`) is a separate concern about compilation strategy, not about AD capability.

**Mixed precision in AD.** The interaction of bf16/f16 forward with f32 backward is part of the broader dtype build-out, not this spec.

**Higher-order AD beyond Hessian.** Computing third, fourth, n-th order derivatives is an extension of the AD machinery but not a separate phase. Hessian-vector products via composition of forward and reverse mode work as a natural consequence of Phases 1-2.

**AD through native code or external libraries.** Calls into non-Chelis code can't be differentiated. The boundary is the language; AD stops at FFI calls. Users wanting gradients across a boundary must wrap external code in differentiable Chelis approximations.

## Strategic implications

This work, if committed to, repositions Chelis. Today's positioning is "tensor language with verified properties and trust stack." The differentiable-language direction adds "first-class differentiable programming for research and scientific computing."

The two positionings don't conflict; they share the substrate (dimension types, AD as transformation, effects, verification). They appeal to overlapping but distinct audiences: the AI/ML mainstream values the tensor language and the verifiability; the scientific-computing and research audience values the differentiable-language capability.

The strategic question is whether to commit. Arguments for:

- The substrate is mostly there. Chelis already has the structural commitments a differentiable language needs. The work is real but bounded; this spec scopes it concretely.
- The audience is underserved. Existing options (Dex, partial-JAX, Julia/Zygote, Enzyme) each have meaningful limitations. Chelis with this work shipped is meaningfully better.
- The competitive landscape is sparser than for tensor frameworks. PyTorch and JAX are entrenched in mainstream ML; differentiable-language territory is smaller and less defended.
- The work composes with the trust stack. Verified gradient-behavior properties, differentiability as a type-level claim, dimension types extending into gradient code — all of these reinforce the broader Chelis positioning.

Arguments against:

- The audience is smaller. Differentiable-language users number in the thousands, not the millions. Mainstream ML user count is orders of magnitude larger.
- The work is real. Even with the substrate in place, the phases above are a significant commitment.
- Some of the work depends on §5 entries that are currently deferred. Committing to this direction means those entries become required, not optional.

This spec doesn't make the strategic decision. It scopes the work concretely so the decision can be made with clear information about what committing entails.

## Net

Seven phases, sequenced by dependency. Phase 0 is the spec lock. Phases 1-5 are the implementation. Phase 6 is documentation and ecosystem. The work composes with existing Chelis commitments and unlocks a distinct audience the existing framework landscape doesn't serve well.

Total scope is substantial but bounded. Decision to commit is separate from this spec; if committed, this spec is the implementation roadmap.

## Host selectors in numeric lowering

The numbered spec/06 sections 2.1, 2.7, and 2.10.1 control this boundary.
`LoweredValue::HostConstant` retains exact strings beside numeric nodes through
helper inlining, lexical capture rebasing, and recursive aggregates. Equality
and inequality consume this carrier without putting strings in the RISC DAG.
Host-runtime Grad arguments and captures use the same carrier; structured
cotangents replace host leaves with unit. `examples/grad_host_selectors.ch`
exercises both branch outcomes in Eval and C.

This implements the exact-selector portion of [#2552](https://github.com/Chelis-Lang/chelis/issues/2552).
Runtime host-valued control still requires an executable host stage. Coral's
Hamt also computes string hashes using operations whose [05-OP-58] contract
structurally rejects differentiation; accepting that graph needs a contract
decision or an explicit `stop_gradient` boundary, beyond selector preservation.
