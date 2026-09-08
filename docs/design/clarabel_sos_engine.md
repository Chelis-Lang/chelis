# Clarabel SoS certificate engine (WI-15)

Status: IMPLEMENTED (WS-6). Option (i): no frozen-seam change. BLAS decision
(A): `clarabel/sdp` with a cfg-gated backend (sdp-openblas on Linux,
sdp-accelerate on macOS), opt-in dev/CI feature only -- the shipped release
binary (`--features smt`) links zero BLAS.

This is the design for the sum-of-squares (SoS) certificate `DischargeEngine`
(`ClarabelSosEngine`), gated behind the `clarabel` cargo feature. It proves that
a univariate polynomial is nonnegative on a bounded interval, `p(x) >= 0` for
`x in [a, b]`, by producing an **exact rational** SoS certificate that an
independent checker re-verifies exactly. The discharge is tagged
`Qualifier::CertificateBearing` at `Soundness::Exact` -- but **only** when the
exact rational certificate verifies. A float SDP solution alone is never a
proof.

It covers the polynomial fragment cvc5's proof output cannot certify
(`spec/design/verification_stack_master_plan.md` WI-15): cvc5 may *decide* a
nonlinear-real-arithmetic goal but its Alethe proof output does not cover NRA,
so there is no independently checkable artifact. An exact SoS certificate is
that artifact.

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

`Gt`/`Lt` (strict) goals are recognized but discharged conservatively. The SoS
machinery proves `p >= 0`. A strict claim `p > 0 on [a,b]` is only entailed if
additionally `p` has no root in `[a,b]`. The exact certificate does not by
itself establish strictness, so a strict goal is reported proved ONLY when the
SoS certificate is for `p - eps` for some exact rational `eps > 0` that the
repair finds verifiable (i.e. `p >= eps > 0`). If only `p >= 0` verifies (the
boundary touches zero), a strict goal is NOT proved -- it returns Unknown, never
a false strict proof. This keeps the strict path fail-honest. (Initial
implementation may decline strict goals entirely via `fitness` returning false
for `Gt`/`Lt`; declining is sound. Pin which in the engine.)

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

The soundness of the certificate rests entirely on step 2 + step 3, which are
pure `BigRational` arithmetic. Clarabel's float precision affects only whether we
*find* a certificate, never whether a found certificate is *valid*. The exact
verifier (the polynomial identity check + the rational PSD test) is the trust
anchor; it would reject a bad Gram even if Clarabel claimed success.

## Discharge mapping (honesty invariants -- soundness-critical)

`Qualifier::CertificateBearing` has `min_soundness == Soundness::Exact`
(`discharge.rs`). So:

- **Exact-verified certificate** -> `Discharge::new(Soundness::Exact,
  {CertificateBearing}, TierBResult::Proved, evidence)`. The evidence carries the
  rational certificate (the Gram matrices and the Markov-Lukacs form) so a
  downstream auditor can re-check it. This is the ONLY path to `Exact`.
- **No certificate found** (Clarabel infeasible/unknown, repair fails, exact PSD
  test fails, coefficient projection cannot reproduce `p`, boundary degeneracy
  unrecoverable) -> `Discharge::new(Soundness::Untrusted, {}, TierBResult::Unknown,
  evidence)`. An empty qualifier set at the bottom of the lattice. NEVER `Exact`,
  NEVER a fabricated certificate. This is the result a failed/unverifiable cert
  produces -- fail honest. (At WS-5 merge time a Clarabel `Unknown` falls through
  to cvc5, whose NRA may still decide the goal; standalone, the engine returns
  the honest Unknown.)
- **`p` negative somewhere on `[a, b]`** -> no SoS certificate exists (the SDP is
  infeasible) -> Unknown, never a fabricated certificate. The engine does not
  attempt to disprove; it only certifies nonnegativity, so a negative polynomial
  is simply not-proved (not Disproved). cvc5 owns the disproof.

The invariant that makes the whole thing trustworthy: a `Discharge` carrying
`CertificateBearing` at `Soundness::Exact` exists **iff** the exact rational
certificate in its evidence re-multiplies to `p` exactly AND every Gram matrix in
it is exactly PSD over the rationals. The red team hammers this: "no false Exact
cert."

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

`clarabel` is a cargo feature: `clarabel = ["dep:clarabel", "num-rational",
"dep:num-bigint", "dep:num-traits"]`. The default, `smt`, `z3`, and solver-free
builds are unaffected: the engine, the Clarabel dependency, and the registration
are all behind `#[cfg(feature = "clarabel")]`. The solver-free gate
(`check_is_solver_free_on_the_corpus`) stays green because nothing on the default
path links Clarabel.

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
