# Chelis Property Specification

## Purpose

`chelis prove` is the Level 2 executable-property verifier. It discovers
first-class `@property` declarations, generates deterministic samples from
their binder types, filters samples through `where` preconditions, evaluates
the property predicate, and reports pass/fail/unsupported/error records.

Properties desugar to ordinary callable `bool` definitions. Deep does not gain
a new tag.

## Surf Syntax

Canonical v1 form:

```chelis
@property call_price_non_negative forall(
    s: f32, k: f32, r: f32, sigma: f32, t: f32
) where s >= 0.0, k > 0.0, sigma > 0.0, t >= 0.0:
  call_price(s, k, r, sigma, t) >= 0.0
  with tolerance = 1e-6
  with seed = 0
  with samples = 100
```

Rules:

- Property names are required Surf value identifiers and are unique in the
  containing module's value namespace.
- Binder types are explicit-only in v1. Omitted binder types are parse errors.
- `where` clauses are harness filters. False-precondition samples are not
  counted and the predicate is not called for them.
- `with tolerance`, `with seed`, and `with samples` are optional.
- Numeric suffixes in property names have no harness semantics.

## Deep Representation

Desugaring emits a `defsig` plus an ordinary property-tagged `def`:

```deep
(defsig {} call_price_non_negative
  (t-fn {} (t-prim {} f32) (t-prim {} bool)))

(def {chelis_role: "property",
      property_source_kind: "user",
      property_quantifiers: (params {} (s {type: (t-prim {} f32)})),
      property_preconditions: (tuple {} ...)}
  call_price_non_negative
  (fn {} (params {} (s {type: (t-prim {} f32)})) ...))
```

Required metadata:

- `chelis_role: "property"`
- `property_source_kind: "user"` or `"bridge:c-earchin"`
- `property_quantifiers: (params {} ...)`
- `property_preconditions: (tuple {} ...)`

Optional metadata:

- `property_source_id`
- `property_tolerance`
- `property_seed`
- `property_samples`

The Deep `def` name is the property name. There is no `property_name` or
`property_predicate_body` metadata. Def parameters are the canonical binder
source; `property_quantifiers` is required as a serialization aid and must match
the parameter list.

c-earchin compatibility:

- Existing `c_earchin_role: "property_witness"` metadata is accepted as a
  synonym for `chelis_role: "property"`.
- c-earchin v0.2 and later emit both canonical metadata and existing
  `c_earchin_*` metadata indefinitely.
- The witness `def` name is the property name. `property_source_id` carries the
  EARS ID.

## Generation And Evaluation

V1 generators support:

- `bool`
- `int32`, `int64`
- `f32`, `f64`
- `string`
- fixed-shape numeric tensors with literal dimensions and `f32`/`f64` elements,
  e.g. `tensor[3, f32]` and `tensor[2, 3, f64]`

Unsupported in v1:

- symbolic tensor dimensions
- tensor rank greater than 2
- ADTs and records
- higher-order binders
- state, capability, or effect assertions

Unsupported selected properties exit `2`. Generator exhaustion exits `3`.
Default max attempts are `samples * 100`; `--max-attempts` overrides this.
Default samples are `100`; `with samples` and `--samples` override it. Default
seed is `0`; `with seed` and `--seed` override it. Runs with the same seed and
inputs must produce the same sample sequence.

V1 reports the first deterministic counterexample. Shrinking is deferred.

## CLI Contract

```sh
chelis prove
chelis prove properties/
chelis prove src/pricing.ch
chelis prove references/pricing_rules.dp --spans references/pricing_rules.spans.json
chelis prove --only call_price_*
chelis prove --samples 1000 --seed 42 --max-attempts 100000
chelis prove --json
```

Discovery:

- No path: scan current Reef package `properties/**/*.ch` and `src/**/*.ch`.
- Exclude `tests/`, hidden directories, `target/`, `dist/`, and dependencies.
- Explicit `.ch` files scan only those files for properties.
- Explicit `.dp` files validate Deep first, then scan property metadata.
- Imports are followed for name resolution, not property discovery.
- Dependency properties do not run unless explicitly targeted.

Span lookup:

- For each `.dp` input, auto-discover sibling `<stem>.spans.json`.
- `--spans <path>` overrides auto-discovery for a single `.dp` input.
- Multi-`.dp` custom layouts require conventional sibling spans in v1.

Exit codes:

- `0`: all selected runnable properties passed.
- `1`: at least one runnable property failed.
- `2`: at least one selected property is unsupported in v1.
- `3`: setup/config/input error.

Precedence is `3 > 2 > 1 > 0`.

A module that does not type-check is an `Error` (exit `3`): `chelis prove`
surfaces the check diagnostics and never reports success on a type-broken
module. Derived producer obligations cannot be meaningfully verified
without the checker-inferred signatures, so a rejectable producer must not
be hidden behind an unrelated type error. Strict downstream prove-compat
admission (FlukeBall) relies on this: a `prove` that exits `0` warrants
that the module type-checked and every obligation was discharged.

## JSON Output

`--json` emits NDJSON. Each selected property emits one record, followed by one
summary record.

```json
{"kind":"property","name":"call_price_non_negative","status":"passed","samples":100,"seed":0}
{"kind":"property","name":"req_PRC_001","status":"failed","samples":1,"seed":0,"source":{"kind":"bridge:c-earchin","spans":"references/pricing_rules.spans.json"}}
{"kind":"property","name":"tensor_symbolic_shape","status":"unsupported","reason":"symbolic tensor dimensions are not supported in L2 v1"}
{"kind":"summary","total":3,"passed":1,"failed":1,"unsupported":1,"errors":0}
```

### Derived obligation records (additive — `opaque_invariants_rfc.md` D-OBLIG)

In addition to `{kind:"property"}` records, `chelis prove` emits one
`{kind:"obligation"}` record per derived producer obligation of an
invariant-carrying opaque type, and the `{kind:"summary"}` record gains an
`obligations` field. These are **additive**: existing `kind:"property"` and
`kind:"summary"` records and fields are unchanged in meaning, and a strict
admission parser (FlukeBall) must add `kind:"obligation"` to its accepted
record set deliberately at pin time — it is a new record kind, not a change
to an existing one.

```json
{"kind":"obligation","obligation_kind":"invariant_producer","source_type":"Probability","producer":"probability","name":"invariant:Probability:probability","status":"passed","proof_tier":"smt","samples":0,"seed":0,"arith_model":"real"}
{"kind":"obligation","obligation_kind":"invariant_producer","source_type":"Probability","producer":"bad_prob","name":"invariant:Probability:bad_prob","status":"failed","proof_tier":"smt","samples":0,"seed":0,"arith_model":"real","counterexample":{"__arg0":"2.0"}}
{"kind":"obligation","obligation_kind":"invariant_producer","status":"error","reason":"opaque type `Probability`: exported producer `many` returns the type through an unsupported container (generic `List`); decompose-or-reject (RFC D-PRODUCER)"}
{"kind":"summary","total":1,"passed":1,"failed":0,"unsupported":0,"errors":0,"obligations":1}
```

Fields: `obligation_kind` is `"invariant_producer"` in V1; `source_type`
is the opaque type; `producer` is the exported def (or constant) under
obligation; `name` is `invariant:<Type>:<producer>`; `status` is one of
`passed`/`failed`/`unsupported`/`error`; `proof_tier` is `"smt"` (Tier B)
or `"fuzz"` (Tier C); `samples`/`seed` mirror the property records;
`counterexample` and `reason` are optional. A `counterexample` keys the
producer's inputs by **positional placeholder** (`__arg0`, `__arg1`, ...),
not by source parameter name. `arith_model:"real"` is
present on `proof_tier:"smt"` records (the SMT-over-reals caveat — see
the Tier B note below). A `status:"error"` record is a declaration-time
covered-or-rejected / signature-rejection failure and carries only
`obligation_kind`, `status`, and `reason`. Exit codes are unchanged in
meaning: a failed or errored obligation participates in the same
worst-status exit code as user properties (`Passed=0`, `Failed=1`,
`Unsupported=2`, `Error=3`).

### Invariant-binder generation and starvation (`--invariant-min-rate`)

A binder whose type is an invariant-carrying opaque type — in a derived
obligation or in a user `@property` — is generated only over
invariant-satisfying values: tiered rejection sampling then
constructor-based generation, with every accepted sample
predicate-validated (full design in `opaque_invariants_rfc.md` D-STARVE /
D-INJECT). When both tiers fall below the floor the binder starves and its
record is `status:"unsupported"` (exit `2`) with a `reason` naming the
type, per-method accepted/attempted counts, the rate, the floor, the
predicate-shape classification, and the recommended route. This is
distinct from user-precondition generator exhaustion, which stays an
`Error` (exit `3`): the two failure modes remain separable. The
`--invariant-min-rate <f64>` flag (default `0.01`) sets the floor for the
rejection tier; `--invariant-min-rate 0.0` disables the starvation
classification and preserves the legacy exhaustion-as-`Error` path.
Generation is deterministic under a fixed seed.

### Tier B SMT proofs are over the reals (caveat)

A `proof_tier:"smt"` obligation (or property) is discharged by the SMT
solver over the **reals**, while runtime arithmetic is IEEE
floating-point. Such artifacts carry `arith_model:"real"`. No
float-level soundness is claimed from a Tier B proof; admission policies
that quote the composed opaque-invariant guarantee must quote this gap
rather than rediscovering it. Tier C (`proof_tier:"fuzz"`) validates
concrete float samples and carries no `arith_model` field.

## Relationship To `chelis test`

`chelis test` remains the deterministic assertion runner for `Std.Test`.
`chelis prove` is the property runner. V1 does not run properties
automatically from `chelis test`.
