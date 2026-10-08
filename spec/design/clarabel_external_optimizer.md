# Clarabel as an external optimizer

## Purpose and authority

This document describes the implementation of a Chelis library call to the
Clarabel conic optimizer. The language rules for inbound native libraries
belong in `spec/11-ffi.md`, numeric operations in `spec/05-risc-primitives.md`,
and property assumptions in the numbered syntax and type chapters. This
document does not grant a proof assumption. The optional provider feature
routes evaluator and compiled-C calls through the checked package declaration
to the same Rust adapter. The ideal property contract has a separate opt-in
SMT lowering.

Chelis already uses Clarabel internally as an optional float proposer for
sum-of-squares certificates. That path checks an exact rational certificate
after solving. It is independent of the user-facing solver call described
here and must retain its separate trust model.

## Public shape

The external `chelis-clarabel` package exports a one-shot QP solve operation
over typed `f64` tensors: square `P`, vector `q`, matrix `A`, vector `b`, a
sequence of cone descriptions, and solver settings. The dimensions bind as
`P[n,n]`, `q[n]`, `A[m,n]`, and `b[m]`. Dense input is converted to compressed
sparse column (CSC) form inside the adapter. `P` supplies only its upper
triangle to Clarabel, after symmetric-input validation. The cone dimensions
sum to `m`. The solver's mathematical problem is

    minimize 1/2 x^T P x + q^T x
    subject to A x + s = b, s in K.

The package covers zero, nonnegative, second-order, exponential, power, and
generalized-power cones. The PSD-triangle cone is excluded from this package's
initial provider. This does **not** relax the convex-QP requirement that `P`
be positive semidefinite. Only finite numeric inputs and valid cone parameters
are admitted. Missing or malformed data is an error before invoking Clarabel.
An indefinite `P` is outside the solver contract: a floating-point PSD test
cannot establish the mathematical PSD premise needed by a proof.

The result keeps `Solved` separate from reduced-accuracy, infeasibility,
iteration/time-limit, and numerical-failure statuses. A solved result carries
primal `x`, dual `z`, slack `s`, and diagnostics. Other statuses never silently
become `Solved`; their candidate vectors, if exposed, confer no proof fact.
Settings explicitly select supported tolerances and the iteration limit.
The adapter uses one thread, disables verbose output, and has no wall-clock
termination limit in its reproducible default configuration.

The provider is an optional Linux/macOS package. Clarabel is pinned
independently of the optional BLAS-linked `chelis-prove` SoS feature. The
package's native binary identity and the source package lock identity are
checked before use. No default-toolchain dependency on Clarabel or BLAS is
introduced by an unreferenced package.

## Native provider boundary

The general boundary is a versioned provider ABI. Reef resolves the dependency
declaration, and the provider registration binds that exact declaration to a
provider artifact digest and symbol. The call carries checked tagged
scalar/tensor values, never a bare
`double` numeric channel or an unvalidated dtype selector. The provider
borrows its inputs, returns owned outputs using the invoking runtime's
allocation/accessor table, and has no independent runtime ownership ledger.
Evaluator calls and compiled-C linking use the same Rust adapter source. The
Clarabel provider is carried by the selected compiler build. Its runtime
archive is hash-checked while staging, and generated C records the archive
hash it links. A missing provider, target mismatch, ABI mismatch,
symbol mismatch, or changed artifact is a loud failure.

Linker-resolved dependency ownership, not the authored spelling of a call,
identifies a provider function. A root-package lookalike or alias to another
implementation cannot inherit a package contract. Generated C need only know
the general provider-call ABI; it does not carry Clarabel-specific lowering.
Device kernels cannot invoke a host provider. The call's effect and external
observation semantics must be stated in the numbered specifications before
shipping the user-facing API.

## Two proof views

The first proof view is an **opt-in idealization**. A property naming the
registered `clarabel.qp.ideal_optimality` contract may use it only for a
`Solved` branch of the exact resolved solve call and its own `P,q,A,b,K`.
The property must establish or assume the convex-QP domain, including
`P = P^T` and `P` positive semidefinite. In the property model, the `Solved`
primal projection denotes a fresh real vector `x*` satisfying primal feasibility and
`f(x*) <= f(y)` for each feasible comparison vector `y`. For an unconstrained
problem, `P x* + q = 0`; at a full-dimensional interior optimum with no
equality constraints, the same stationarity consequence is available. The
prover applies only consequences it can lower and discharge; it never treats
an unsupported goal as true.

The executable SMT lowering scalarizes fixed-size `f64` QP data and
zero/nonnegative cones. Its symbolic-input path is described below. It follows a typed Chelis helper that
returns one direct call to the same resolved `solve` declaration, substituting
its arguments before constructing the proof goal. Dynamic runtime dimensions
and the other runtime cone families are outside that deductive lowering and produce an
explicit `unsupported` result when the contract is requested. A contract
request cannot fall through to unqualified sampling or attach an assumption
to `AlmostSolved`. A successful dependent report has
`composite_verdict = proven_modulo_asserted_axiom`, retains
`real_arithmetic` in `qualifiers`, and records the contract and provider
identity. Independent properties retain their own proof strength.

This view idealizes the primal projection while proving the property; it does
not assert that the returned `f64` bits are an exact optimizer in execution.
For example, minimizing `1/2 * 3*x*x - x` has optimum `1/3`, which no `f64`
represents. The ideal real vector is a conditional model for downstream
mathematical proofs.

A later concrete view may assert or verify bounds on the returned floating
vector, such as feasibility residual or objective gap, and connect that
vector to the ideal optimizer when a sufficient error bound is available.
Solver diagnostics are preserved for that extension; no such bound follows
merely from a `Solved` status or a requested tolerance.

## Implementation and validation

The package source is `packages/chelis-clarabel/src/qp.ch`. It supplies the
checked `Clarabel.Qp.solve` signature, cone and status ADTs, and a loud
unavailable-provider body. With the `clarabel-provider` Cargo feature, the
evaluator intercepts the resolved package function and calls the Rust adapter.
The `packages/chelis-clarabel/tests/` programs exercise a solve, an
inequality, a stopped result, and an invalid cone partition. The separate
`examples/clarabel_qp` package exercises a path dependency.
The package README gives the exact commands.

The provider feature uses a pinned Rust dependency and dispatches only when
the Reef linker assigns the expected function to a package of the registered
version whose module source bytes match the provider registration. A different
package body, even one with the same linked name, executes as ordinary Chelis
code. The provider registration records the linked symbol, ABI version,
operation identity, C symbol, and carried archive digest together. The
evaluator invokes the adapter in process; the compiled-C host lane preserves
that checked call, boxes tagged arguments, and invokes the versioned entry
point in the carried runtime archive. The archive is checked when staged;
the staging receipt and generated C record its SHA-256. The property lowering
uses the same registration and records that digest in its axiom evidence.

The standalone Rust adapter lives in `crates/chelis-clarabel-provider/`.
`crates/chelis-compiler-api` owns the Reef source admission and evaluator
registration. `crates/chelis-ir` preserves the checked host call, and
`crates/chelis-runtime` implements its tagged C transport. `crates/chelis-prove`
owns the opt-in ideal QP lowering. The feature-gated CLI integration target
checks evaluator/C parity, status and validation failures, exact source
admission, feasible-baseline and stationarity proofs, wrapper substitution,
argument binding, and fail-closed unsupported cases.

The concrete bounded-result contract is a separate proof view. It requires an
explicit bound that applies to the floating-point result and cannot be inferred
from `Solved` or requested tolerances alone.

## Symbolic-input proof extension

The fixed-literal lowering demonstrates call attribution and conditional
reporting, but it cannot prove a property of an algorithm that constructs QP
data from its inputs. This extension keeps the same `solve` API and ideal-real
contract. It changes the proof representation and the set of programs the
prover can lower; it does not change runtime optimization or claim a bound on
the `f64` result. The controlling semantics are in `spec/11-ffi.md`.

### Acceptance theorem and supported program form

The first acceptance theorem has fixed, compile-time-known extents and
symbolic `f64` values. An algorithm receives a matrix `B`, vectors `q`, `b`,
`state`, and `baseline`, and a matrix `A`; it constructs `P = B^T B`, solves
the two-variable QP with a fixed zero/nonnegative cone partition, then passes
the `Solved` primal through pure `apply_step(state, primal)` and
`quality(updated, state, P, q)` helpers. The property assumes or proves that
`baseline` is feasible and
proves, for every valuation satisfying its `where` conditions, that

    quality(apply_step(state, primal), state, P, q)
      >= quality(apply_step(state, baseline), state, P, q)

where `quality(updated, state, P, q)` is a fixed scalar offset minus
`1/2 step^T P step + q^T step` for `step = updated - state`. The typed helper
reshapes `step` from `[2]` to `[2,1]` for `matmul(P, step_column)`, reshapes
that product back to `[2]`, and forms the two dot products by elementwise
multiplication and reduction. A companion claim uses primal feasibility to
bound the updated state. The QP data, baseline, and state vary at runtime;
the proof must establish the algebraic connection between the QP objective
and the downstream helpers. A claim that merely solves a literal one-variable
QP and restates its stationary point does not satisfy this acceptance oracle.

This lane specializes a property at concrete tensor extents and proves it for
all real-valued entries satisfying the stated premises at those extents. It
does not prove a theorem universally quantified over dimensions. Its initial
cone topology is a fixed list of `ZeroCone` and `NonnegativeCone` blocks,
while `P,q,A,b` entries may be symbolic. Settings remain an explicit,
well-formed fixed record. Pure straight-line helpers, named bindings,
conditionals, elementwise addition/subtraction/multiplication/negation,
`permute`, `reshape`, `matmul`, and reductions are lowered when their dimensions are
fixed. Division, effects, recursion, loops, symbolic extents, dynamic cone
selection, and other cone families produce `unsupported` for this lane.

### Typed call and value provenance

The prover starts from the checked, linker-resolved program representation,
not authored function names or a source-text search for `solve`. It assigns a
stable identity to each admitted dependency-owned call occurrence and records
the typed value expressions supplied as `P,q,A,b,cones,settings`. Bounded,
capture-avoiding expansion of pure helpers exposes the call and its downstream
uses while preserving source ownership and branch identity. If expansion
cannot account for a value, the goal is unsupported. A `Solved` pattern binds
one fresh real vector to that call's primal projection; every subsequent use
of that projection in the property refers to the same vector. A different
call, changed argument, lookalike declaration, or `Stopped` path receives no
assumption from this occurrence. The representation keys assumptions by call
identity even if the first lane admits only one reached solve call per goal.

For a fixed shape, the proof lowerer expands typed matrix/vector values into
real scalar terms and lowers the supported Chelis operations with their
checked extents. It carries a real-arithmetic interpretation of the *whole*
input construction and downstream computation. In particular, recognizing
`B^T B` establishes PSD for the ideal real expression computed by that
construction; it does not certify that the rounded runtime matrix is PSD.
The `real_arithmetic` qualifier discloses that distinction. No source-level
ghost optimizer type or new runtime ABI is needed.

### Convex-domain premise

The ideal contract requires symmetric PSD `P`. The lowerer first attempts a
checked derivation: retain the exact rational PSD check for literal matrices,
and accept a Gram construction only after verifying its typed scalar identity
`P[i,j] = sum_k B[k,i] * B[k,j]` and the sum-of-squares basis for
`v^T P v >= 0`. A syntax resemblance without that identity is insufficient.
The derivation is a fact of the real proof model and its evidence is attached
to the call. For other symbolic `P`, the author may add
`with contract = "clarabel.qp.assume_psd"` alongside
`clarabel.qp.ideal_optimality`. It asserts `P = P^T` and PSD for that same
call's `P`, with a distinct `method: "axiom"` assumption record. The named
premise cannot be used without the matching ideal-optimality call, and it
cannot be transferred to another matrix. With neither derivation nor explicit
premise the property is unsupported; `Solved` alone does not establish PSD.
At a fixed dimension, the explicit premise lowers to symmetry equalities and
nonnegative principal minors, an equivalent finite real-arithmetic PSD
condition. Resource exhaustion in that expansion or its SMT discharge is
`unsupported`; the contract name is not an uninterpreted predicate that may
be asserted without the matrix-specific condition.
Implementing this bound pair requires removing the executable lowering's
sole-contract restriction while retaining source-bound contract checks.

### Discharge and report

The ideal model grants primal feasibility and objective minimality over
feasible vectors for the call's own data. Instead of sending an unrestricted
quantified nonlinear QP axiom to SMT for every property, the lowerer finds the
comparison vectors that the goal actually uses. It proves each candidate's
feasibility under the property's preconditions, then instantiates the
optimality implication at that candidate. A comparison with unproved
feasibility cannot borrow the implication. The lowerer normalizes polynomial
identities introduced by pure helpers before dispatching the residual
property to the existing SMT engine. The existing sound stationarity rules
remain available for the unconstrained and admitted interior cases; this
extension does not treat stationarity as an arbitrary solver observation.

For a goal `A => G`, SMT checks the selected sound consequences `A` and
`not G` for a counterexample. The separate non-vacuity query uses the *full*
feasibility-and-global-optimality axiom, the convex-domain premise, and the
property preconditions. SAT establishes that at least one ideal-model input
valuation satisfies them; SAT of only the goal-directed instances would be
insufficient. This check does not establish a runtime `Solved` outcome for
every input. Timeout, unknown, unsupported lowering, an unestablished
convex-domain premise, or a vacuous assumption set cannot produce a green
proof or fall through to fuzz sampling. The existing
`proven_modulo_asserted_axiom` verdict and
`real_arithmetic` qualifier remain the report strength. Each dependent
assumption identifies its resolved provider symbol and archive digest, its
call occurrence, and a deterministic fingerprint of the typed symbolic input
expressions; the fingerprint is over expressions, not runtime values. The PSD
derivation or explicit PSD axiom appears separately in evidence. An
independent property retains its own verdict without a Clarabel dependency.
No new verdict token or top-level report kind is introduced.

### Implementation sequence and regression oracle

Before implementation, add spec-derived positive and negative CLI test stubs
for the symbolic acceptance theorem, an arbitrary-`P` variant with explicit
PSD assumption, and every failure condition below. Then replace the literal
argument parser and direct-wrapper substitution with the typed call/value
representation; add fixed-shape symbolic scalarization, PSD derivation and
explicit-premise binding, goal-directed optimality instantiation, and
downstream helper lowering. Preserve the existing literal proofs as regression
witnesses. The optional provider/SMT tests and feature-specific Clippy remain
owned by Clarabel paths, with broader coverage in the scheduled run; unrelated
PRs do not acquire a new serial Clarabel check.

The positive oracle runs `chelis prove --tier smt-only --json` on the
algorithm property and requires `status: "passed"`, zero samples, a
call-bound ideal-optimality assumption, the PSD evidence appropriate to the
variant, `proven_modulo_asserted_axiom`, and `real_arithmetic`. Negative twins
change the QP argument or call owner, use a `Stopped` arm, omit the PSD
premise, substitute an infeasible baseline, or reverse the downstream
inequality. Additional rejection tests cover contradictory premises, a
second solve result trying to inherit the first call's assumption, and a
symbolic shape or cone family outside this lane. Every negative must produce
`failed`, `invalid`, or `unsupported` as appropriate, with no
axiom-laundered pass. Runtime evaluator and compiled-C parity remain governed
by the existing provider tests; this extension adds no numerical-execution
guarantee.
