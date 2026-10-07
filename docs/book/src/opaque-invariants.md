# Opaque types with declared invariants

`@opaque` keeps a type's construction and representation inside its defining
module. Other modules can use exported values and functions, but cannot build
or inspect the representation themselves. An optional `@invariant` states a
condition that values produced for callers should satisfy. `chelis check`
checks that the declaration is well formed; it does not evaluate the condition.
`chelis prove` checks eligible exported functions that return the type.

## Declare the type and its producers

Save the following module as `opaque_invariants.ch` to check its declarations
and properties. It defines a probability in the unit interval and exports
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

Keep the type's constructors and invariant in its defining module. Include
`Probability` in the export list when other modules need to name it in their
signatures.

The invariant's binder, `p`, has the representation type. `chelis check`
rejects a declaration that breaks one of these rules, with an
`OpaqueTypeViolation` naming the field or expression:

- The type must be `@opaque`, declared in a named module, with exactly one
  record-shaped variant.
- Every field must be a numeric or `bool` scalar (`f32`, `f64`, a signed
  integer, `bool`), an `f32` or `f64` tensor whose dimensions are all
  literals, or a nested single-variant record of such fields. `string`,
  `List`, function, symbolic-dimension tensor, and multi-variant fields are
  rejected.
- The predicate must be boolean and may use only: literals; the binder and
  its field projections; `+`, `-`, `*`, `/`; comparisons; `&&`, `||`, `not`;
  `if`; `abs`, `min`, `max`, `sqrt`, `exp`, `log`, `sin`, `cos`; `sum` over a
  tensor field; and zero-argument constant definitions in the same module.
  Other function calls, `match`, lambdas, other tensor operations, and effects
  are rejected.
- Exact float equality (`==` on a field) is accepted with an
  `invariant-float-equality` lint warning, because sampling almost never
  generates a value that satisfies it. Use a tolerance band instead.

`probability` returns `Some` only for accepted inputs. `scale` and `combine`
return a new `Probability` from existing values. Their checks may assume that
opaque inputs received from callers satisfy the invariant, and must establish
it for their returned values. `prob_value` returns an `f32`, so it is not an
opaque-value producer. The `@property` is a separate check about values of
the type.

## Check and prove

```sh
chelis check opaque_invariants.ch
chelis eval --file opaque_invariants.ch --json
chelis prove --capabilities
chelis prove opaque_invariants.ch --json
```

`chelis check` verifies the types and declaration form; it does not certify
that producer functions preserve the invariant. Use `chelis prove` to check
the property and eligible producer obligations.

Before relying on an SMT result, check `chelis prove --capabilities` and
confirm both `obligation_engine_available` and `smt_available` are true. If
the obligation engine is unavailable, a successful property run does not
verify producer obligations. See [Checking Properties](proving.md)
for proof options and result qualifications.

For the SMT-enabled binary, the example has one property and three producer
obligations. These are **selected fields**, not complete output records or a
literal transcript:

```json
{"kind":"property","name":"prob_value_in_unit_interval","status":"passed","proof_tier":"fuzz"}
{"kind":"obligation","name":"invariant:Probability:combine","status":"passed","proof_tier":"smt","composite_verdict":"proven_modulo_real_arithmetic","arith_model":"real"}
{"kind":"obligation","name":"invariant:Probability:probability","status":"passed","proof_tier":"smt","composite_verdict":"proven_modulo_real_arithmetic","arith_model":"real"}
{"kind":"obligation","name":"invariant:Probability:scale","status":"passed","proof_tier":"smt","composite_verdict":"proven_modulo_real_arithmetic","arith_model":"real"}
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
def probability(x: f32) -> Option[Probability] = if (x >= 0.0) then Some(Probability { value: x }) else None
```

With SMT enabled, the prover reports a counterexample and exits with a failure.
The exact value is solver output; use the record's `"counterexample"` field
rather than expect `2.0` specifically. Exported return types are read from
the checker, including inferred returns. A return container that the prover
cannot inspect produces an error instead of being silently skipped.

## Use the type from another module

In a [Reef package](reef.md) with `module_prefix = "Stats"`, put the defining
module in `src/opaque.ch` and this consumer in `src/pricing.ch`. Import both
the public type and the functions:

```chelis-surf-fragment
module Stats.Pricing
import Stats.Opaque (Probability, probability, scale, prob_value)
def adjusted(x: f32, factor: Probability) -> f32 =
  match probability(x) with {
    | Some(p) => prob_value(scale(p, factor))
    | None => 0.0
  }
```

Exporting `Probability` allows the annotation. It does not allow a caller to
construct `Probability { value: x }`, read `p.value`, update the record, or
match its constructor. `chelis check` reports `OpaqueTypeViolation` for those
operations outside `Stats.Opaque`; the diagnostic names the type, its defining
module, and exported producer signatures. The following `src/forge.ch`
consumer is rejected:

```chelis-surf-fragment
module Stats.Forge
import Stats.Opaque (Probability)
def forge(x: f32) -> Probability = Probability { value: x }
```

## Tensor fields and proof boundaries

This module keeps three weights whose sum lies within `eps` of one:

```chelis-surf
module Stats.Simplex
export (make_simplex)
@opaque
@invariant(p) ((sum(p.weights) >= (1.0 - eps)) && (sum(p.weights) <= (1.0 + eps)))
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def eps() -> f32 = 0.0001
def make_simplex(a: f32, b: f32, c: f32) -> Simplex = {
  s = (((abs(a) + abs(b)) + abs(c)) + 0.001)
  Simplex { weights: to_tensor([(abs(a) / s), (abs(b) / s), ((abs(c) + 0.001) / s)]) }
}
@property simplex_binder_is_generated forall(p: Simplex):
  true
```

`chelis prove simplex.ch --json --seed 42` reports these fields (selected
from each record) even with SMT available:

```json
{"kind": "property", "name": "simplex_binder_is_generated", "status": "passed", "proof_tier": "fuzz", "composite_verdict": "fuzz_validated", "samples": 100}
{"kind": "obligation", "name": "invariant:Simplex:make_simplex", "status": "passed", "proof_tier": "fuzz", "composite_verdict": "fuzz_validated", "samples": 100}
{"kind": "summary", "total": 2, "passed": 2, "obligations": 1}
```

The `make_simplex` obligation passed on 100 sampled inputs, with no
`arith_model`: the solver did not prove it, so it is validated, not proved,
for inputs outside the sample. The property `true` checks only that the
sampler can generate valid `Simplex` values. A tolerance band is practical
for floating sums; exact equality to one would starve the sampler.

Opacity does not add bounds to downstream arithmetic. `chelis check` does not
infer from a `Probability` value that a later calculation lies in an interval.
Only return values produced across the module boundary receive the producer
obligations described here. If the defining module passes a raw opaque value,
or a function able to produce one, outward as a call argument, that path needs
module review; the `opaque-escape-site` lint identifies such calls.

Opacity also does not hide data in tooling output. When evaluation prints an
opaque value as a root, it shows the constructor and fields. Sampling
counterexamples can show them too.
