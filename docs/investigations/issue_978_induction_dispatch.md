# Issue 978: induction dispatch investigation

Issue: [chelis#978](https://github.com/Chelis-Lang/chelis/issues/978)

## Result

The existing `tier_d` module is not a proof tier. It has no production call
site, does not parse or check the supplied property source, trusts an explicit
size-parameter classification, and previously returned `Proved` records whose
base and step evidence was `ASSUMED`. This change removes that unsound result:
until real obligations are dispatched, both cases fail closed and no CLI or
Tide surface may classify them as proof evidence.

This is a safety correction, not completion of structural induction. General-n
bond and lattice claims remain fixed-size downstream.

## Two-cycle evidence

### Cycle 1: remove the false proof

Tests were first changed to require `BaseFailed` for the undispatched lattice
and period paths. They failed against the scaffold's manufactured `Proved`
result, then passed after base and step verification were changed to explicit
fail-closed outcomes. The active Tide contract and CLI guide now state that
both obligations must be discharged through an existing sound tier.

Authoritative local evidence:

```text
cargo test -p chelis-prove tier_d::tests
6 passed, 0 failed

python3 scripts/gate.py --local
397 chelis-prove tests passed, 6 skipped
```

### Cycle 2: attempt the real recursive surface

The probe below checks as valid Surf on the current compiler:

```chelis
module Probe.Induction
def sum_to(n: int32) -> int32 = if (n <= 0) then 0 else (n + sum_to((n - 1)))
@property sum_to_nonnegative forall(n: int32) where (n >= 0), (n <= 8):
  (sum_to(n) >= 0)
```

The exact branch compiler reports:

```text
--tier smt-only: unsupported, "property does not lower to Tier B (smt-only)", exit 2
auto: no proof; terminal generator-exhaustion error, exit 3
```

That is the correct outcome. Sampling a finite subset would not establish a
general-n theorem, and the SMT lowerer deliberately stops when inlining reaches
the recursive call.

## Missing sound substrate

A real Tier D dispatch requires all of the following in one reviewed language
and artifact contract:

1. A source-visible way to identify the induction variable and the property,
   base, and step obligations. Naming conventions are not provenance.
2. Compiler-AST classification of a structurally decreasing recursive call;
   the caller-supplied `size_parameter` hint is not evidence.
3. One-step unfolding that preserves parameter substitution and introduces the
   induction hypothesis only for the exact recursive subproblem.
4. Separate solver discharges for the concrete base and symbolic step, with
   their goals, assumptions, soundness, and qualifiers retained in CLI/Tide
   artifacts.
5. A negative theorem whose step fails, plus a true general-n bond property.
6. CLI/Tide parity and a rule that any assumed, unsupported, sampled-only, or
   missing case prevents an induction proof.

Adding only a new enum value, trusting a named function, bounded fuzzing over
`n`, or unrolling to a fixed depth would recreate the false-completion bug.

## Deferral boundary

The fail-closed safety correction is ready independently. Full induction stays
open under chelis#978 until the proof-object and one-step-unfolding substrate
above is implemented. Shoals and C Note must continue to label their existing
`n=5` bond/CRR properties as fixed-size and may not promote them to general-n.
