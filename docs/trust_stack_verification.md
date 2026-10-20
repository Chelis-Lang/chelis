# Property verification architecture

`chelis prove` checks declared properties and invariant obligations, then
reports the method and assumptions behind each result. The
[user guide](book/src/proving.md) explains commands, result fields, and exit
codes. This page maps those results to the implementation.

## Inputs and verification methods

Requirements enter as Surf or Deep `@property` declarations with parameters,
preconditions, and a predicate. A translator can attach a sibling source
manifest. The consumer verifies the generated file, original source hashes,
byte ranges, and property mappings before attributing results to requirements.
That attribution establishes provenance; it does not strengthen a verdict.

Type checking rejects ill-formed declarations before verification. It does
not discharge an arbitrary property. The core verification methods are:

- **SMT solving:** cvc5 checks supported predicates over real arithmetic.
  Builds with the `z3` feature can use Z3 for goals cvc5 leaves unknown.
- **Induction:** supported recursive properties produce base and step goals.
  Both must be proved without vacuity; the arithmetic model remains real.
- **Sampling:** deterministic edge inputs precede seeded random draws. The
  edge schedule includes guard boundaries and their neighbours, both zeros,
  subnormals, finite extrema, small integers, and decade magnitudes. Every
  accepted input must satisfy the preconditions. A pass establishes only
  that the property held on the sampled inputs.

`--tier` selects a method or automatic dispatch. Availability depends on the
build; `chelis prove --capabilities` reports its configuration. The result's
`proof_tier` identifies its method, and `composite_verdict` and `qualifiers`
state what that method establishes.

## Machine arithmetic

An SMT proof over real arithmetic does not establish the same property for
floating-point execution. Automatic dispatch samples after an eligible solver
proof to look for a machine counterexample. A found counterexample makes the
record fail. A clean sample leaves the real-arithmetic qualification intact;
an inconclusive sample leaves the proof unchanged and explains why.

For a solver disproof, the verifier rounds the solver's witness to the
parameter widths and checks it concretely. A witness that satisfies all
preconditions and falsifies the body establishes a machine failure. An
unconfirmed witness leaves the machine question open. The `machine_arithmetic`
field records this distinction. Explicit solver selection also checks a
disproof's witness, although it does not add sampling after a proof.

## Assumptions and composed evidence

Composition combines a result with the producer and contract assumptions it
uses. JSON records disclose each assumption's `discharge` method and evidence:

- Sampling alone produces `fuzz_validated`, never a proof badge.
- An SMT proof with SMT-discharged assumptions is
  `proven_modulo_real_arithmetic`.
- A proof relying on a sampled contract carries
  `proven_modulo_fuzz_validated_contract`.
- A proof relying on an asserted axiom carries
  `proven_modulo_asserted_axiom`.

Assumptions must also pass a non-vacuity check: an inconsistent assumption
domain makes the verdict `invalid`; an inconclusive check makes it
`unsupported`. An idealizing asserted axiom can exclude a machine-arithmetic
cross-check, as with `clarabel.qp.ideal_optimality`. The result remains
conditional on that disclosed axiom.

## Rejections and solver limits

Timeouts and unknown answers establish neither a proof nor a counterexample.
Explicit solver requests report `unsupported`; automatic dispatch can try
sampling. A square root enters SMT only when a separate total-algebraic
obligation proves its argument nonnegative from independent preconditions.
An unresolved obligation cannot silently strengthen the query. Auxiliary and
main solver work share the request timeout.

## Compiler correctness

Property verification asks whether a program satisfies the author's stated
requirements. Differential compiler tests and the reference checker ask
whether the implementation follows the language specification. Both forms
of evidence matter, and neither replaces the other.
