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

The SMT lowering scalarizes fixed literal `f64` QP data and zero/nonnegative
cones. It follows a typed Chelis helper that
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
