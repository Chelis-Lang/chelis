# Clarabel SoS certificate engine (WI-15)

`ClarabelSosEngine` proves univariate polynomial nonnegativity on a bounded interval through exact rational certificates.
The optional `clarabel` feature enables its native solver.
Linux uses OpenBLAS, and macOS uses Accelerate.
The shipped `smt` release does not enable Clarabel or link BLAS.

The host admits a certificate only after exact goal binding, arithmetic verification, and canonical interval-layout checks.
An admitted certificate carries `Qualifier::CertificateBearing` at `Soundness::Exact`.
A floating-point solver result alone is not a proof.

It covers the polynomial fragment cvc5's proof output cannot certify
(`spec/design/verification_stack_master_plan.md` WI-15): cvc5 may *decide* a
nonlinear-real-arithmetic goal but its Alethe proof output does not cover NRA,
so there is no independently checkable artifact. An exact SoS certificate is
that artifact.

## Portable arithmetic and host admission

`chelis-certificate-core` owns `poly`, `exact`, and `linalg`.
It uses unconditional `no_std`, `alloc`, and arbitrary-precision `BigInt` and `BigRational` arithmetic.
Public arithmetic signatures, constructor preconditions, deterministic repair, and error order remain unchanged.
Callers import the new crate directly without old modules or compatibility re-exports.

`SosCertificate::verify` checks each Gram's symmetry and PSD before it checks the combined polynomial identity.
It does not bind the certificate to a requested goal or establish arbitrary multiplier nonnegativity.
`RatSymMatrix::is_psd` needs a symmetric matrix.
Its caller must establish that precondition.

`chelis-prove` retains SMT extraction, canonical layout construction, floating-point conversion, Clarabel, BLAS, evidence codecs, and proof admission.
Admission preserves exact polynomial and endpoint agreement and the existing strict-goal rejection.
After arithmetic verification, it needs the canonical block count, matrix dimensions, and ordered interval multipliers.
Construction and admission share canonical block definitions without a second solver-layout algorithm.
Arithmetic error labels retain precedence over layout rejection.

The [acceptance command](../manual_gates.md#certificate-arithmetic-core-portability) checks independent target consumers and fixed-revision native parity.
Target compilation and host execution do not prove device execution, allocator capacity, or whole-pipeline portability.


## Goal-shape fit (Option (i), no GoalShape change)

`GoalShape` is one of the five Beacon-pinned frozen surfaces
(`docs/design/phase2_seam_contract.md`). We do **not** add a `PolyNonneg`
variant. Instead the engine fits `GoalShape::Smt(SmtProperty)` and internally
recognizes the subset of `SmtProperty` that *is* a univariate polynomial
inequality over the reals on a bounded box, extracting `(p, [a, b])` by a pure
structural match on the existing `SmtExpr` tree.

Rationale (grounded against the active docs):

- `GoalShape::BoxRange` is documented as Beacon's native *interval-engine* shape
  (input box -> output-range assertion). An SoS poly-nonneg goal is a different
  claim; reusing `BoxRange` would misrepresent it and collide with Beacon on the
  same shape, where the registry tie-break routes by registration order rather
  than by what each engine proves.
- The master plan (`verification_stack_master_plan.md` WI-9) routes "range to
  Beacon; polynomial with certificate to the SoS backend" as two distinct
  targets, and the dependency map pins WI-15 as depending on the WI-4
  *interface* (the discharge seam), not on WI-5 / `BoxRange`.

### The recognized subset

`SmtProperty -> Option<PolyOnInterval>` succeeds iff:

- `postcondition` is `Cmp(op, lhs, rhs)` with `op in {Ge, Gt, Le, Lt}`. It is
  normalized to `p(x) (>= | >) 0` by moving one side across (`Le`/`Lt` flip the
  sign). `Ge`/`Le` are the closed (`>= 0`) form Markov-Lukacs proves directly;
  `Gt`/`Lt` are accepted as the closed form too (a strict-on-a-compact-interval
  goal is conservatively discharged as the non-strict `>= 0`: a `>= 0`
  certificate is sound evidence for the strict claim only where the polynomial
  is nonzero, so the strict variant is NOT claimed proved unless `p` is bounded
  away from 0; see "Strict inequalities" below).
- `p` (the normalized difference) is a polynomial in a single real `Var`: a tree
  of `Arith(Add | Sub | Mul | Neg, ..)` over `Var(x)` and `RealLit`. Any `Div`,
  any `Apply` (a transcendental or uninterpreted function), an `IntLit`-only or
  `BoolLit` leaf in arithmetic position, or more than one distinct `Var` makes
  the property fall outside the subset.
- `preconditions` pin the interval: a lower bound `a <= x` (`Cmp(Le, a, x)` or
  `Cmp(Ge, x, a)` with `a` a `RealLit`) and an upper bound `x <= b`. Both bounds
  are required and `a <= b` (an inverted / empty interval is not a fit).

`fitness()` returns true **only** for this subset. Everything else (multivariate
polynomials, transcendental applications, a missing interval bound, a
non-polynomial body) returns false, so the goal stays with cvc5 / the no-fit
path. There is no false fit: a goal Clarabel cannot honestly attempt is never
claimed by Clarabel.

### Strict inequalities

The engine recognizes strict `Gt` and `Lt` goals.
It does not admit a certificate that proves only nonnegativity.
A certificate for a different margin-adjusted polynomial also fails exact goal binding.
Strict goals therefore remain `Unknown@Untrusted`.

## Markov-Lukacs SoS-to-SDP reduction (univariate-on-interval)

For a univariate polynomial `p` of degree `n` that is nonnegative on `[a, b]`,
the Markov-Lukacs representation gives an exact SoS decomposition over the
interval. The two standard forms (by parity of `n`):

- `n = 2m` (even degree): `p(x) = s(x) + (x - a)(b - x) * t(x)`, where `s` is SoS
  of degree `2m` and `t` is SoS of degree `2m - 2`.
- `n = 2m + 1` (odd degree): `p(x) = (x - a) * s(x) + (b - x) * t(x)`, with `s`,
  `t` SoS of degree `2m`.

A polynomial `s` is SoS iff `s(x) = z(x)^T Q z(x)` for a positive-semidefinite
Gram matrix `Q`, where `z(x) = [1, x, x^2, ..., x^d]^T` is the monomial vector up
to degree `d`. So "find an SoS decomposition" is "find PSD Gram matrices whose
quadratic forms, combined with the fixed interval multipliers, match `p`
coefficient by coefficient" -- a semidefinite feasibility problem.

### SDP encoding for Clarabel

Clarabel solves cone programs `min (1/2) x^T P x + q^T x s.t. A x + s = b, s in K`.
We use it as a feasibility problem (`P = 0`, `q = 0`):

- **Decision variables** are the entries of the Gram matrices `Q_i`
  (`s`'s Gram, `t`'s Gram), vectorized.
- **PSD cones**: each `Q_i` is constrained `Q_i >= 0` via a `PSDTriangleConeT(k)`
  cone (k = side length). Clarabel's PSD triangle cone consumes the symmetric
  matrix as its **upper triangle, column-major, with off-diagonal entries scaled
  by sqrt(2)** (svec convention, so the cone inner product equals `tr(Q_i Y)`).
  This scaling is load-bearing for reading the Gram back; it is pinned and
  round-trip-tested against an actual solve, not assumed blind.
- **Coefficient-matching equalities**: for each monomial degree `0..=n`, the sum
  of the contributions of the `Q_i` quadratic forms (each `Q_i` entry `Q[r][c]`
  contributes to monomial `r + c + (multiplier offset)`) must equal `p`'s
  coefficient. These are linear equalities in the Gram entries, encoded with a
  `ZeroConeT`.

Clarabel returns a FLOAT solution `Q_i ~ Q_i*`. This float solution is a
*heuristic proposal*, not a proof.

## Peyrl-Parrilo exact rational repair (the real work)

A float Gram matrix is not a certificate: rounding it back to rationals almost
never reproduces `p` exactly, and near-boundary instances produce small negative
PSD coordinates that naive rounding cannot reconstruct
(`verification_stack_master_plan.md` WI-15). The repair turns the float proposal
into an EXACT rational certificate or fails honestly.

1. **Round** the float Gram entries to rationals (bounded-denominator rational
   approximation of each float).
2. **Coefficient-matching projection**: the rounded Gram generally violates the
   exact coefficient-matching equalities. Project it onto the *affine subspace*
   of Gram matrices that reproduce `p` EXACTLY (an exact rational linear-algebra
   step over `BigRational`). After this step the polynomial identity
   `z^T Q z + (interval multipliers) == p` holds **exactly** as rational
   polynomials, by construction.
3. **Exact PSD repair / verification**: the projected `Q` reproduces `p` but may
   no longer be PSD. Test exact PSD-ness via a **rational LDL^T (symmetric)
   factorization**: `Q` is PSD iff the factorization completes with all pivots
   `>= 0` (and the appropriate handling of zero pivots / rank deficiency). This
   is exact rational arithmetic; no float tolerance. If `Q` is exactly PSD, the
   certificate is `(Q_i, the Markov-Lukacs form)` and it VERIFIES.
   - Boundary-degeneracy handling: when the float solution sits near the PSD
     boundary, the projected rational `Q` may have a tiny negative pivot. The
     repair attempts a bounded denominator-refinement / diagonal-shift search
     that stays inside the exact coefficient-matching subspace; if no exactly-PSD
     representative is found within the budget, the repair FAILS.

### Exact verification is independent of Clarabel precision

Exact rational arithmetic determines whether the certificate's polynomial identity and Gram PSD checks pass.
Clarabel precision affects whether the proposer can find such a certificate.
Arithmetic success alone does not establish a proof.
Host admission also needs exact goal binding and the canonical interval layout.

## Discharge mapping (honesty invariants -- soundness-critical)

`Qualifier::CertificateBearing` has `min_soundness == Soundness::Exact`
(`discharge.rs`). So:

- **Admitted certificate**: `Soundness::Exact`, `{CertificateBearing}`, and `TierBResult::Proved`.
  Evidence includes exact rational Gram matrices, interval multipliers, the polynomial, and endpoints.
  Admission needs goal binding, successful arithmetic verification, and the canonical layout.

- **Missing or rejected certificate**: `Soundness::Untrusted`, no qualifiers, and `TierBResult::Unknown`.
  Evidence records the rejection without a certificate.

- **Negative polynomial**: the engine does not fabricate a certificate or claim `Disproved`.
  A different engine can try the goal after `Unknown`.

A `CertificateBearing@Exact` result needs a certificate for the exact requested polynomial and endpoints.
Every Gram must be symmetric and exactly PSD.
The combined polynomial identity and canonical ordered interval layout must match.
No strict-goal nonnegativity certificate can pass as a strict proof.

## Dispatcher composition (with WS-5 fall-through)

Registration order is priority (`engine_registry.rs`). The approved composition
is (a) + the WS-5 fall-through:

- `ClarabelSosEngine` registers AHEAD of cvc5 with NARROW
  poly-nonneg-on-interval fitness, so only those goals hit Clarabel first;
  everything else routes to cvc5.
- A Clarabel `Unknown` (failed SDP / repair / boundary degeneracy) MUST fall
  through to cvc5 (its NRA may still prove a poly goal Clarabel could not
  certify). This depends on WS-5's try-until-discharge semantics, which are not
  on this branch yet. The engine is built standalone and registerable now; the
  fall-through integration and the `with_builtin_engines` registration order are
  sequenced at merge time with the team-lead.

Standalone (no WS-5), a Clarabel `Unknown` is returned as the honest Unknown
outcome. The engine never regresses a goal to a false result; the only behavioral
difference from the integrated path is that cvc5 does not get a second attempt
until WS-5 lands.

## Feature gating

The `clarabel` feature enables the host engine, native solver, and portable arithmetic dependency.
Default, `smt`, `z3`, and solver-free builds do not activate this optional path.
Independent arithmetic consumers depend directly on `chelis-certificate-core` without Clarabel or BLAS.

The PSD cone needs Clarabel's `sdp` feature, which needs a BLAS/LAPACK backend
(decision A). The per-OS backend is selected in target-specific dependency
tables that reference the SAME optional `clarabel` dep, so cargo unifies the
feature onto it:

```toml
[target.'cfg(target_os = "linux")'.dependencies]
clarabel = { version = "0.11", optional = true, features = ["sdp-openblas"] }

[target.'cfg(target_os = "macos")'.dependencies]
clarabel = { version = "0.11", optional = true, features = ["sdp-accelerate"] }
```

Linux builds and statically links the OpenBLAS source selected by
`openblas-src`; macOS uses the OS-bundled Accelerate framework. The Clarabel
float proposer (`propose.rs`) is itself gated to
`cfg(any(target_os = "linux", target_os = "macos"))` -- the targets with a
backend wired; on any other target the exact core and the `UnwiredProposer`
engine still build. Because `clarabel` is opt-in and the shipped release binary
ships `--features smt` only, the product and downstream shells link zero BLAS;
only chelis's own `clarabel` CI lane and dev builds pull OpenBLAS.

## Acceptance (spec-first negatives)

- known-nonnegative poly on an interval -> `CertificateBearing@Exact` with a
  certificate that exactly verifies (positive).
- poly NEGATIVE somewhere on the interval -> NOT proven, no certificate
  (soundness-critical negative).
- a certificate that fails exact verification -> `Unknown@Untrusted` + empty
  qualifier set, never `Exact` (the soundness-critical negative the RT hammers).
- narrow fitness: multivariate / transcendental / missing-interval / non-poly ->
  `fitness` false, no false fit (negative).
- feature lanes: `--features clarabel` green; default + `smt` + solver-free
  unaffected.
