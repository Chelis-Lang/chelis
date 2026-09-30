# Opaque Types With Declared Invariants

A Chelis type can be declared **opaque**: it is constructible and
inspectable only inside its defining module, enforced by the type checker.
An opaque type can carry one **declared invariant** — a boolean predicate
over a single binder of its representation, written in Surf at the
declaration site. The everyday `chelis check` pipeline never evaluates the
invariant (it stays solver-free), but `chelis prove` turns the invariant
into machine-discharged proof obligations: for every way the module can
produce a value of the type, it proves the invariant holds.

This chapter explains the workflow through the executable example
[`examples/opaque_invariants.ch`](https://github.com/Chelis-Lang/chelis/blob/main/examples/opaque_invariants.ch).
The [type-system specification](https://github.com/Chelis-Lang/chelis/blob/main/spec/04-type-system.md)
defines the language rules.

## The workflow at a glance

1. **Declare the opaque type** with `@opaque` inside a named `module`.
2. **Declare the invariant** with `@invariant(binder) <predicate>`.
3. **Export constructors and updates** — the functions that produce values
   of the type.
4. **Run `chelis prove`** — it derives one obligation per exported producer
   and discharges it.
5. **Read the obligation records** in the `--json` output.

## Step 1 and 2: declare the type and its invariant

A `Probability` is a single `f32` whose value lies in the unit interval.
The invariant is a predicate over a binder `p` of the representation:

```chelis-surf-fragment
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
```

Rules to know:

- The enclosing `module` must be **named**. `@opaque` outside a named
  module is a declaration error (an unnamed top-level scope would make the
  enforcement boundary ambiguous across combined sources).
- The representation uses a single-record-variant ADT.
- `@invariant` is optional, takes exactly one binder, and must sit between
  `@opaque` and `type`. A bare `@invariant` without `@opaque` is an error:
  assumption injection would be unsound for a forgeable type.
- The predicate grammar is restricted: literals, the binder and its field
  projections, arithmetic (`+ - * /`), comparisons, `&&`/`||`/`not`, `if`,
  the whitelisted intrinsics `abs/min/max/sqrt/exp/log/sin/cos`, `sum` over
  a fixed-shape tensor field, and references to in-module zero-argument
  constant defs. Surf's boolean conjunction is `&&`, not the word `and`.
- Boolean conjunction in the predicate is `&&`. Anything outside the
  grammar (general calls, `match`, lambdas, effects) is a declaration
  error.

## Step 3: export the producers

A **producer** is an exported function whose result contains the opaque
type. The canonical shape is a guard-then-`Option` partial constructor: it
returns `Some` only on inputs that satisfy the invariant, and `None`
otherwise.

```chelis-surf-fragment
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
```

Update-shaped producers take a `Probability` and return one. They are the
inductive step of the soundness argument: they may **assume** the invariant
of their inputs (every input value reaching them from well-typed
out-of-module code originated from some producer), and must establish it of
their output.

```chelis-surf-fragment
def scale(p: Probability, factor: Probability) -> Probability =
  Probability { value: p.value * factor.value }
```

## The full example

Here is the complete executable module. `chelis check` scores it 1, and
`chelis prove` discharges every obligation:

```chelis-surf
module Stats.Opaque
export (probability, scale, combine, prob_value)
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

## Step 4 and 5: run `chelis prove` and read the obligations

The obligation surface is built into `chelis prove` under the `smt`
feature. Run it with `--json` for the machine-readable NDJSON stream:

```text
chelis prove examples/opaque_invariants.ch --json
```

The example is a library module — it declares types and exported producers
but has no top-level expression, so `chelis eval`/`chelis build` succeed
with nothing to run (`chelis eval` reports `{"roots":[]}`). `chelis prove`
is where its obligations are exercised.

Each exported producer yields one `{kind:"obligation"}` record, the user
`@property` yields a `{kind:"property"}` record, and a final
`{kind:"summary"}` record reports the totals (note the new `obligations`
count). The records below are shown with related fields grouped for
reading; the binary emits each object's keys in alphabetical order, so a
literal byte-diff against this page will differ in field order only:

```json
{"kind":"property","name":"prob_value_in_unit_interval","proof_tier":"fuzz","samples":100,"seed":0,"status":"passed"}
{"kind":"obligation","obligation_kind":"invariant_producer","source_type":"Probability","producer":"probability","name":"invariant:Probability:probability","proof_tier":"smt","arith_model":"real","samples":0,"seed":0,"status":"passed"}
{"kind":"obligation","obligation_kind":"invariant_producer","source_type":"Probability","producer":"scale","name":"invariant:Probability:scale","proof_tier":"smt","arith_model":"real","samples":0,"seed":0,"status":"passed"}
{"kind":"obligation","obligation_kind":"invariant_producer","source_type":"Probability","producer":"combine","name":"invariant:Probability:combine","proof_tier":"smt","arith_model":"real","samples":0,"seed":0,"status":"passed"}
{"kind":"summary","total":4,"passed":4,"failed":0,"unsupported":0,"errors":0,"obligations":3}
```

Reading the obligation record:

- `name` is `invariant:<Type>:<producer>`.
- `obligation_kind` is `"invariant_producer"`.
- `source_type` is the opaque type; `producer` is the exported def.
- `proof_tier` is `"smt"` (the obligation lowered to the SMT solver) or
  `"fuzz"` (validated sampling). The three `Probability` producers here all
  reach `"smt"`: the guard-then-`Option` constructor and the two
  update-shaped producers reduce to linear-arithmetic queries the solver
  closes.
- `arith_model:"real"` appears on every `proof_tier:"smt"` record. **The
  SMT proof is over the reals, not IEEE floats** — see the caveat below.
- `status` is `passed`, `failed`, `unsupported`, or `error`.

The exit code is the worst status across all properties and obligations:
`0` all passed, `1` something failed, `2` something unsupported, `3` a
setup/declaration error (a type-broken module is exit 3 — `prove` never
reports success on a module that does not type-check).

### A producer that does not establish the invariant

Drop the upper-bound guard and the obligation is disproved with a concrete
counterexample:

```chelis-surf-fragment
def bad_prob(x: f32) -> Option[Probability] =
  if x >= 0.0 then Some(Probability { value: x }) else None
```

```json
{"kind":"obligation","obligation_kind":"invariant_producer","source_type":"Probability","producer":"bad_prob","name":"invariant:Probability:bad_prob","proof_tier":"smt","arith_model":"real","samples":0,"seed":0,"status":"failed","counterexample":{"__arg0":"2.0"}}
```

The input `2.0` satisfies `x >= 0.0` but produces a `Probability` whose
value exceeds `1.0`. `prove` exits `1`.

### Every producer is checked

The producer set is computed from checker-**inferred** return types, so an
unannotated exported def cannot escape it. A producer that returns the type
through a container `prove` cannot decompose — a list, a record, a function
type, any generic other than `Option` — is a **declaration error** naming
the producer, never a silent skip. One uncovered producer would collapse
the soundness argument, so the obligation is reported as
`status:"error"` and `prove` exits `3`.

## Using an opaque type from another module

Opacity only matters across a module boundary, so here is the scenario it
defends. A *consumer* module imports the exported producers and works with
the type through them — it never names the representation:

```chelis-surf-fragment
module App.Pricing
import Stats.Opaque (probability, scale, prob_value)

// Legitimate: obtain and transform values only through exported producers.
def adjusted(x: f32, factor: Probability) -> f32 =
  match probability(x) with {
    | Some(p) => prob_value(scale(p, factor))
    | None    => 0.0
  }
```

Every attempt to *construct* or *inspect* the representation from a module
other than `Stats.Opaque` is a type error from `chelis check` (and blocks
`chelis build`), each naming the type, its defining module, and the
exported producers you should call instead:

```chelis-surf-fragment
module App.Forge
import Stats.Opaque (Probability)

def forge(x: f32) -> Probability = Probability { value: x }   // record construction: rejected
def peek(p: Probability) -> f32 = p.value                     // field access: rejected
def grab(x: f32) -> Probability = x : Probability             // ascription/cast: rejected
def unwrap(p: Probability) -> f32 =
  match p with { | Probability { value: v } => v }            // pattern match: rejected
```

Each line raises an `OpaqueTypeViolation`, for example:

```text
error[OpaqueTypeViolation]: record construction of opaque type `Probability`
  outside its defining module `Stats.Opaque`; obtain values through the
  exported producers of `Stats.Opaque`: probability, scale, combine
```

The same applies to functional record update, positional constructor
application, a bare reference to the constructor as a value, and a reference
to an *unexported* binding of the defining module whose signature mentions
the type. Inside `Stats.Opaque` itself, none of these are restricted — that
is where the proved constructors live.

A single Surf file holds one module, so the consumer and the defining module
live in separate files of a reef package (or separate `(module {} ...)`
wrappers in a hand-written `.dp`). Reusing the defining module's name to
"reopen" it is itself a `DuplicateModule` error, and the reef package
linker's internal name format is reserved (`ReservedLinkerName`) — neither
is an escape hatch.

## Invariants over tensor fields: the simplex

A `Simplex` is a fixed-shape weight vector that sums to one within a
tolerance band over a module constant `eps`:

```chelis-surf-fragment
@opaque
@invariant(p) sum(p.weights) >= 1.0 - eps && sum(p.weights) <= 1.0 + eps
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def eps() -> f32 = 0.0001
def make_simplex(a: f32, b: f32, c: f32) -> Simplex =
  {
    s = abs(a) + abs(b) + abs(c) + 0.001
    Simplex { weights: to_tensor([abs(a) / s, abs(b) / s, (abs(c) + 0.001) / s]) }
  }
```

The matching source file is
[`examples/opaque_invariants_simplex.ch`](https://github.com/Chelis-Lang/chelis/blob/main/examples/opaque_invariants_simplex.ch).
It is an executable example: the invariant
predicate (including the `sum`-over-a-tensor-field form) is declaration
metadata consumed only by `chelis prove`; it is never lowered to runtime IR,
so it does not affect runtime evaluation or compilation. Its top-level `eps`
definition supplies the tolerance used by the invariant; `chelis eval` and
`chelis build` process that definition normally.

The example shows two useful details:

- The `make_simplex` producer obligation discharges through validated sampling
  (`proof_tier:"fuzz"`): its body divides by a running sum, which does not
  reduce to a closed-form SMT query. Sampling records do not include an
  `arith_model` field.
- A `@property` quantifying over a `Simplex` binder is verified only over
  invariant-satisfying values. The tolerance band is measure-near-zero
  under independent component sampling. The generator uses
  **constructor-based generation** and evaluates `make_simplex` on sampled
  inputs. It checks every generated value against the invariant before
  accepting it, so the `forall(p: Simplex)` property receives valid samples.

Use a tolerance band over a module constant for a "sums to one" constraint.
An exact `sum(p.weights) == 1.0` does not work with sampled float values:
validation is strict, and a normalizing producer emits sums that are near the
target in floating point. An equality-shaped predicate needs a proof over the
real-number model when that property can be expressed for the SMT solver.

## What this feature does NOT do

This is the most important section. The composed guarantee is precise, and
it is easy to over-read.

**The invariant is invisible to `chelis check`.** `chelis check` stays
solver-free. It does not evaluate the predicate, it concludes nothing from
the invariant, and there is no solver in the check loop. The advisory
`opaque-domain-construction` lint gives fast editor and agent feedback, but
**the lint is fast feedback; the typing judgment is the guarantee.** The
mechanical guarantee comes from `chelis prove`'s discharged obligations,
not from `check`.

**This is not refinement typing.** Downstream arithmetic does not inherit
bounds. If you read `prob_value(p)` and add `0.5`, the checker does not know
the result is in `[0.5, 1.5]` — there are no predicates in the typing
judgment, no verification conditions at use sites, and no narrowed types
flowing out of an opaque value. Obligations about derived values remain
your own `@property` territory.

**SMT-tier proofs are over the reals, not floats.** A
`proof_tier:"smt"` obligation is discharged by the solver over the **reals**
while runtime arithmetic is IEEE floating-point. Such records carry
`arith_model:"real"`. This record describes a proof over real-number
arithmetic; it does not certify IEEE floating-point behavior. State this
limitation when relying on the proof.

**Argument egress is trusted, not obligated.** Producer obligations cover
every value of the type *returned* across the module boundary. But a value
the defining module passes *outward as a call argument* to an out-of-module
callee is not mechanically obligated. The rule is:

> Producer obligations mechanically cover every value of the type
> returned across the module boundary. Values of the type — and function
> values capable of producing it — that the defining module passes
> outward as call arguments are covered by module audit, not by
> machine-discharged obligations. The advisory `opaque-escape-site` lint
> enumerates every such site, labeled by local provenance; transitive
> flows within the module are the audit's responsibility.

The advisory `opaque-escape-site` lint enumerates every in-module
argument-egress site at two levels — a `note` for sites whose value traces
locally to a producer call or a type-`T` input, and a `warning` for
unattested sites that trace to a raw construction or representation update.
The enumeration is complete; the provenance labels direct a module audit,
which is what owns transitive flows.

**Tooling output discloses representation contents.** When `chelis eval`
actually reduces an opaque value to a result it prints the constructor and
its fields, and sampling counterexamples print representation values. (The
library examples in this chapter have no top-level expression, so `eval`
prints no roots; the disclosure applies to a program that evaluates an
opaque value to a root.) This is disclosure, not a secrecy break: none of
those outputs are re-importable as typed values, so the construction
guarantee is unaffected. Opacity is a construction-and-provenance
guarantee, not an encryption scheme.

## Cross-surface parity

The obligation collection, synthesis, assumption injection, and tiered
dispatch live in the `chelis-prove` crate and are shared by both the CLI
`chelis prove` path and the chelis-tide `chelis_prove` MCP tool. A prove run
through tide reports the same obligation records, the same proof tiers, and
the same `arith_model` caveat as the CLI on the same module.
