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

## JSON Output

`--json` emits NDJSON. Each selected property emits one record, followed by one
summary record.

```json
{"kind":"property","name":"call_price_non_negative","status":"passed","samples":100,"seed":0}
{"kind":"property","name":"req_PRC_001","status":"failed","samples":1,"seed":0,"source":{"kind":"bridge:c-earchin","spans":"references/pricing_rules.spans.json"}}
{"kind":"property","name":"tensor_symbolic_shape","status":"unsupported","reason":"symbolic tensor dimensions are not supported in L2 v1"}
{"kind":"summary","total":3,"passed":1,"failed":1,"unsupported":1,"errors":0}
```

## Relationship To `chelis test`

`chelis test` remains the deterministic assertion runner for `Std.Test`.
`chelis prove` is the property runner. V1 does not run properties
automatically from `chelis test`.
