# Opaque types with declared invariants

`@opaque` keeps a type's construction and representation inside its defining
module. Other modules can use exported values and functions, but cannot build
or inspect the representation themselves. An optional `@invariant` states a
condition that values produced for callers should satisfy. `chelis check`
checks that the declaration is well formed; it does not evaluate the condition.
`chelis prove` checks eligible exported functions that return the type.

## Declare the type and its producers

This package module defines a probability in the unit interval. It exports
the type so another module can name it, while keeping construction and field
access restricted to `Stats.Opaque`:

```chelis-surf
module Stats.Opaque
export (Probability, probability, scale, combine, prob_value)
@opaque
@invariant(p) ((p.value >= 0.0) && (p.value <= 1.0))
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] = if ((x >= 0.0) && (x <= 1.0)) then Some(Probability { value: x }) else None
def scale(p: Probability, factor: Probability) -> Probability = Probability { value: (p.value * factor.value) }
def combine(p: Probability, q: Probability) -> Probability = Probability { value: (p.value * q.value) }
def prob_value(p: Probability) -> f32 = p.value
@property prob_value_in_unit_interval forall(p: Probability):
  ((prob_value(p) >= 0.0) && (prob_value(p) <= 1.0))
```

The standalone runnable file, [`examples/opaque_invariants.ch`](../../../examples/opaque_invariants.ch),
contains these definitions but does not export `Probability`, since it has no
consumer module. Use the export list above when callers need to write the type
in their signatures.

The invariant's binder, `p`, has the representation type. A declared invariant
requires a named module and a single record-shaped variant. Its predicate can
use field projections, literals, arithmetic, comparisons, boolean operators
such as `&&`, `if`, selected math functions, `sum` over a fixed-shape tensor
field, and in-module zero-argument constant definitions. General function
calls, `match`, lambdas, and effects are not allowed in the predicate. The
[type-system specification](../../../spec/04-type-system.md) gives the full
value and predicate rules.

`probability` returns `Some` only for accepted inputs. `scale` and `combine`
return a new `Probability` from existing values. Their checks may assume that
opaque inputs received from callers satisfy the invariant, and must establish
it for their returned values. `prob_value` returns an `f32`, so it is not an
opaque-value producer. The `@property` is a separate check about values of
the type.

## Check and prove

```sh
chelis check examples/opaque_invariants.ch
chelis eval --file examples/opaque_invariants.ch --json
chelis prove --capabilities
chelis prove examples/opaque_invariants.ch --json
```

`chelis check` reports score `1` with no errors for this example. That means
the types and declaration form check; it does not certify the invariant. The
example has no value to evaluate at top level, so the JSON evaluation result
has `"roots":[]`. Building its C source likewise does not run the exported
producer functions or verify the invariant.

Before using a proof result, check `chelis prove --capabilities`: both
`"obligation_engine_available"` and `"smt_available"` should be `true` for
the SMT result shown below. The shipped release binary has SMT support. A
plain local `cargo build` omits it; its default prover can validate producer
obligations by sampling and emits a warning that they were not SMT-verified.
If the obligation engine is unavailable, a successful property run does not
verify producer obligations. See [Proving](proving.md) for proof options,
exit codes, and the complete output format.

For the SMT-enabled binary, the example has one property and three producer
obligations. These are **selected fields**, not complete output records or a
literal transcript:

```json
{"kind":"property","name":"prob_value_in_unit_interval","status":"passed","proof_tier":"fuzz"}
{"kind":"obligation","name":"invariant:Probability:probability","status":"passed","proof_tier":"smt","composite_verdict":"proven_modulo_real_arithmetic","arith_model":"real"}
{"kind":"obligation","name":"invariant:Probability:scale","status":"passed","proof_tier":"smt","composite_verdict":"proven_modulo_real_arithmetic","arith_model":"real"}
{"kind":"obligation","name":"invariant:Probability:combine","status":"passed","proof_tier":"smt","composite_verdict":"proven_modulo_real_arithmetic","arith_model":"real"}
{"kind":"summary","total":4,"passed":4,"failed":0,"unsupported":0,"errors":0,"obligations":3}
```

A `"fuzz"` result means sampled values passed; it is validation over those
samples, not a proof for all inputs. An `"smt"` result proves the stated goal
using real-number arithmetic. `"arith_model":"real"` and
`"composite_verdict":"proven_modulo_real_arithmetic"` disclose that it does
not certify IEEE floating-point execution. Read `"proof_tier"` and
`"composite_verdict"` alongside `"status"` before relying on a passed record.

If you replace the exported `probability` definition with one that omits the
upper guard, the producer obligation fails. For example, `x = 2.0` meets the
lower guard but returns a value above the declared bound:

```chelis-surf-fragment
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 then Some(Probability { value: x }) else None
```

With SMT enabled, the prover reports a counterexample and exits with a failure.
The exact value is solver output; use the record's `"counterexample"` field
rather than expect `2.0` specifically. Exported return types are read from
the checker, including inferred returns. A return container that the prover
cannot inspect produces an error instead of being silently skipped.

## Use the type from another module

Place the defining module and this consumer in separate Surf files of a
[Reef package](reef.md). Import both the public type and the functions:

```chelis-surf-fragment
module App.Pricing
import Stats.Opaque (Probability, probability, scale, prob_value)

def adjusted(x: f32, factor: Probability) -> f32 =
  match probability(x) with {
    | Some(p) => prob_value(scale(p, factor))
    | None    => 0.0
  }
```

Exporting `Probability` allows the annotation. It does not allow a caller to
construct `Probability { value: x }`, read `p.value`, update the record, or
match its constructor. `chelis check` reports `OpaqueTypeViolation` for those
operations outside `Stats.Opaque`; the diagnostic names the type, its defining
module, and exported producer signatures. For example, the following
consumer is rejected:

```chelis-surf-fragment
module App.Forge
import Stats.Opaque (Probability)

def forge(x: f32) -> Probability = Probability { value: x }
```

## Tensor fields and proof boundaries

[`examples/opaque_invariants_simplex.ch`](../../../examples/opaque_invariants_simplex.ch)
defines a fixed-size weight vector whose sum lies within a tolerance of one.
Its `make_simplex` producer passes by validated sampling even in an
SMT-enabled build: the record has `"proof_tier":"fuzz"` and no
`"arith_model"`. Its property checks that valid `Simplex` values can be
generated; it is not a proof that every possible input produces a valid sum.
A tolerance band is practical for floating sums, while exact equality can
starve sampled generation.

Opacity does not add bounds to downstream arithmetic. `chelis check` does not
infer from a `Probability` value that a later calculation lies in an interval.
Only return values produced across the module boundary receive the producer
obligations described here. If the defining module passes a raw opaque value,
or a function able to produce one, outward as a call argument, that path needs
module review; the `opaque-escape-site` lint identifies such calls.

Opacity also does not hide data in tooling output. When evaluation prints an
opaque value as a root, it shows the constructor and fields. Sampling
counterexamples can show them too.
