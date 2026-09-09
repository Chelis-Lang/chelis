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
  with contract = "std.normal_cdf.reflection"
```

Rules:

- Property names are required Surf value identifiers and are unique in the
  containing module's value namespace.
- Binder types are explicit-only in v1. Omitted binder types are parse errors.
- `where` clauses are harness filters. False-precondition samples are not
  counted and the predicate is not called for them.
- `with tolerance`, `with seed`, and `with samples` are optional.
- `with contract = "..."` is a repeatable string-literal dependency on a
  standard contract invariant. Unknown contract IDs make the property
  unsupported.
- Numeric suffixes in property names have no harness semantics.

Contract options are part of Tier B lowering, not only report metadata. A
contract-bound call to the trusted implementation is replaced by a fresh SMT
symbol and the contract assumptions needed for that symbol. The prover must
check that the call resolves to the implementation named by the discharge
record before applying the abstraction. For `std.normal_cdf.reflection`, the
lowering recognizes syntactic `normal_cdf(x)` / `normal_cdf(-x)` pairs and
asserts the reflection coupling between their fresh symbols.

For `std.quantile.monotonicity`, the trusted implementation is the resolved
Reef dependency declaration for `Nautilus.Stats.quantile_vec`, whose linker
symbol is `pkg__nautilus__Nautilus__Stats__quantile_vec`. Trust comes from the
linker's dependency-owned declaration partition, not from parsing source names
or accepting a root-package lookalike. Lowering intercepts the call before the
generic scalar-argument pass: the tensor operand stays as compiler AST
identity, while the scalar quantile level lowers to SMT. Only two calls over
the same dataset identity receive the relational monotonicity assumption.
Missing trusted calls, different-dataset pairs, and the not-yet-bridged range
or boundary contracts return `unsupported` under `smt-only`.

The chelis#979 acceptance oracle is:

```sh
cargo test -p chelis-cli --features smt --test issue_979_nautilus_quantile
```

### General-n structural induction

`--tier induction-only` is a Surf-only, fail-closed deductive lane. The prover
selects an `int*` induction binder from the checked compiler AST, never from a
caller classification. The v1 accepted shape has an explicit `n >= 0` domain,
one scalar model call in the proposition, and one directly recursive model:
`if n <= 0 then base else step`, where `step` contains exactly one
`f(n - 1, unchanged_args...)` call. Contract abstractions, non-transparent
mutual or non-structural recursion, multiple model calls, uninterpreted
residual calls, and other shapes are `unsupported`; this lane never falls
through to sampling.

“Direct” is measured after the compiler's ordinary bounded helper inlining, so
a type-checked transparent alias may expose the same exact recurrence. This is
intentional: argument substitution and symbol ownership come from the compiler
AST, and the resulting base/step goals are identical to the unaliased form.
A syntactically exact decreasing call under a literal-dead branch is also
accepted. The classifier does not erase that branch: the full `if` remains in
the dispatched goal, so the solver proves its unreachability and no dead call
can manufacture an induction hypothesis or a green case.

The prover constructs a concrete `P(0)` obligation from a full one-step model
unfolding and a symbolic `P(k) => P(k + 1)` obligation whose induction
hypothesis replaces only the exact `f(k, unchanged_args...)` subproblem. Both
goals, including their branch conditions and non-vacuity checks, are dispatched
separately to the existing SMT engine. A green result requires both discharges.
Machine records use `proof_tier:"induction"`, `arith_model:"real"`, and
`induction:{variable,base:{status,arith_model},step:{status,arith_model}}`.
`ASSUMED`, missing, sampled, unknown, timed-out, or vacuous cases cannot produce
a pass. The executable acceptance oracle is:

```sh
cargo test -p chelis-cli --features smt --test issue_978_induction
```

The default `--tier auto` ordering is induction, then the existing Tier B SMT
and Tier C fuzz lanes. Auto enters induction only when the checked Surf AST
shows that the property reaches a recursive model. An accepted induction plan
is terminal whether its base/step proves, disproves, times out, or errors; an
unsupported recursive structure is likewise terminal and reports
`proof_tier:"induction"` with zero samples. This prevents recursive general-n
claims from reaching finite sampling or overflowing the evaluator stack.
Properties that do not reach a recursive model retain the existing Tier B then
Tier C behavior. No auto-induction result may contain `ASSUMED` evidence or a
sampling record.

### Scalar gradient goals in Tier B

Tier B lowers an applied scalar gradient into the same real-arithmetic
obligation language as an ordinary scalar property. The supported v1 shape is
`grad(f, wrt=x)(args...)`, with exactly one explicit `wrt`, where `f` is an
inline lambda or pure top-level function whose parameters and result are
`f32`/`f64`. The differentiated body may contain scalar literals and variables,
negation, `+`, `-`, `*`, `/`, named scalar block bindings, and recursively
inlined pure scalar helpers with `f32`/`f64` results within the normal Tier B
inlining-depth bound. Argument substitutions are resolved in the caller's
scope before differentiation.

This is a prover-owned symbolic dual lowering; it must agree with Chelis scalar
AD semantics but does not replace the compiler's `grad` transform. Its SMT
verdict retains the `real_arithmetic` qualifier. Multi-target or implicit
`wrt`, tensor/ADT gradients, non-floating results, conditionals, casts in
differentiated bodies, effects, recursion, nested transforms, helper-inlining
depth overflow, unsupported intrinsics, and malformed calls do not silently sample under
`smt-only`: they return
`status:"unsupported"` with a reason naming the scalar-gradient capability
boundary. Under `auto`, the same boundary may continue to Tier C fuzz
validation. In particular, conditionals remain outside this prover-owned
subset until the compiler's scalar AD transform can build the same programs.
Float casts in differentiated bodies follow the same executable-parity rule.

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
- `property_contracts: (tuple {} "contract.id" ...)`

The Deep `def` name is the property name. There is no `property_name` or
`property_predicate_body` metadata. Def parameters are the canonical binder
source; `property_quantifiers` is required as a serialization aid and must match
the parameter list.

c-earchin compatibility:

- Property discovery requires canonical `chelis_role: "property"` and its
  required schema. `c_earchin_role` is opaque producer data, not a discovery
  synonym (`spec/03-deep-syntax.md` [03-META-3]).
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

For a guarded property whose binders are all `f32`/`f64`, Tier C derives a
sampling domain from conjunctions of scalar interval and binder-order
comparisons (`<`, `<=`, `>`, `>=`). Constant bounds propagate through binder
orders before sampling, so narrow guards such as
`0.99 < alpha1 < alpha2 < 1.0` are generated in-domain rather than discovered
by rejection from `[-10, 10]`. Explicit bounds are not clipped to that legacy
uniform range; negative literals and reversed comparison spellings are
equivalent interval bounds. Strict spacing is computed with the binders'
actual IEEE `f32`/`f64` successor and predecessor values. Non-strict order
edges permit equality and reserve no strict spacing, while strict chains
reserve enough representable values for their remaining successors. The
construction and its random choices are seed-deterministic. A one-sided finite
interval chooses its missing endpoint within the binder dtype's finite range;
the synthesis clamps at `f32::MAX`/`f64::MAX` rather than overflowing near an
IEEE extremum. An empty interval,
an interval with too few representable values for its strict order chain,
cyclic ordering, disjunction, equality, arithmetic operand other than unary
literal negation, function predicate, or other unsupported guard shape is
`status:"unsupported"` (exit `2`) under `fuzz-only`; it cannot fall through to
a green empirical verdict. Non-scalar guarded properties retain their existing
typed generator and rejection behavior.

The shared Tier-C runner is enabled in every normal CLI build, including a
build without the `smt` feature, and is also the Tide implementation. Solver
availability may change `auto` dispatch into Tier B, but it never changes
`fuzz-only` generation or its machine record.

The authoritative chelis#977 acceptance oracle is:

```sh
cargo test -p chelis-cli --test issue_977_constraint_fuzz
cargo test -p chelis-tide --test mcp issue_977_tide_and_cli_match
cargo test -p chelis-cli --features smt --test issue_977_constraint_fuzz
```

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
{"kind":"property","name":"confidence_tail_order","status":"passed","composite_verdict":"fuzz_validated","qualifiers":["fuzz","fuzz_base"],"assumptions":[{"name":"preconditions:confidence_tail_order","discharge":{"method":"fuzz","evidence":{"status":"validated","sampling_method":"constraint_directed","accepted_samples":100,"attempted_samples":100,"rejected_samples":0}},"non_vacuity":{"status":"established","evidence":{"method":"fuzz","sampling_method":"constraint_directed","accepted_samples":100,"attempted_samples":100,"rejected_samples":0}}}],"proof_tier":"fuzz","sampling_method":"constraint_directed","accepted_samples":100,"attempted_samples":100,"rejected_samples":0,"samples":100,"seed":0}
{"kind":"property","name":"req_PRC_001","status":"failed","composite_verdict":"failed","assumptions":[],"samples":1,"seed":0,"source":{"kind":"bridge:c-earchin","spans":"references/pricing_rules.spans.json"}}
{"kind":"property","name":"tensor_symbolic_shape","status":"unsupported","composite_verdict":"unsupported","assumptions":[],"proof_tier":"none","reason":"symbolic tensor dimensions are not supported in L2 v1"}
{"kind":"summary","total":3,"passed":1,"failed":1,"unsupported":1,"errors":0,"dependency_graph":{"status":"complete","declarations":[{"id":"decl:…","name":"call_price","kind":"function","package":"pricing","module":"Pricing.BlackScholes","source":{"file":"src/black_scholes.ch","span":{"offset":42,"len":180}}}],"edges":[{"from":"decl:…property","to":"decl:…"}]}}
```

The summary's `dependency_graph` is the compiler-owned declaration ownership
wire (chelis#922):

- `status:"complete"` means linker analysis ran. Empty `declarations` and
  `edges` arrays are a complete empty result, not missing analysis.
- `status:"unavailable"` carries a `reason` and never carries a guessed partial
  graph. Bare Surf files have no stable Reef package/module identity, and Deep
  inputs do not carry compiler-owned source-file ownership, so both are
  unavailable.
- A declaration `id` is the deterministic
  `(package,module,kind,author-facing-name)` identity. Body and span edits keep
  the ID stable; a rename changes it. Every node also carries the package,
  module, declaration kind, and package-relative source file plus byte span.
- Edges are stable-ID `from`/`to` pairs derived from Reef's linker-resolved Surf
  AST. Consumers must not reconstruct ownership by parsing source. The graph
  covers functions, values, properties, types, constructors, aliases, macros,
  and each module-level dimension declaration; type/invariant/macro references
  participate alongside value references.
- Every root-package declaration is present, including unused declarations.
  Referenced dependency-package declarations are included transitively. Linker
  identity preserves same-name declarations, lexical shadowing, imports, and
  cycles without name guessing.
- A multi-input summary is `unavailable` if any selected input lacks complete
  attribution; Chelis does not present a partial union as complete.

Every Tier-C property record additively reports `sampling_method`,
`accepted_samples`, `attempted_samples`, and `rejected_samples`. Guarded
properties repeat those counts and the method in their precondition discharge
and non-vacuity evidence, so a consumer can distinguish empirical evidence
from an exhausted or unsupported generator without reconstructing it from
source. CLI and Tide user-property records rendered by the shared runner always
carry `proof_tier`; terminal outcomes report the explicit value `none` rather
than encoding it as field absence.

The legacy name-only `dependency_edges` array remains additive and deprecated
for at least one published release after `dependency_graph` is introduced.
New consumers use only `dependency_graph`.

Every `{kind:"property"}` and `{kind:"obligation"}` result record carries:

- `composite_verdict`: the single WEAKEST badge token, one of `proven`,
  `proven_modulo_real_arithmetic`, `proven_modulo_fuzz_validated_contract`,
  `proven_modulo_asserted_axiom`, `sound_approximate`, `fuzz_validated`,
  `invalid`, `unsupported`, or `failed`.
  `status:"passed"` remains the compatibility bucket; consumers that need
  proof strength must read `composite_verdict`. A fuzz-validated result must
  not render as `composite_verdict:"proven"` or any `proven_*` badge: a
  property whose BASE was established by fuzz sampling only (no SMT proof
  underneath -- a non-smt build, or a `--tier auto` fuzz fall-through, or
  `--tier fuzz-only`) renders `fuzz_validated`, a green-exit empirical pass
  that is not proven. A base discharged by a sound over-approximation (e.g. an
  interval engine) renders `sound_approximate`. An SMT proof is over the reals
  (see the Tier B caveat below), so under the current real-sorted lowering an
  SMT green renders `proven_modulo_real_arithmetic`, disclosing the
  machine-arithmetic gap; plain `proven` is reserved for a future
  exact-machine-arithmetic lowering. The
  `proven_modulo_fuzz_validated_contract` badge is reserved for an exact SMT
  base discharged modulo a fuzz-validated CONTRACT assumption -- the base
  itself is proven there, only a contract is fuzz-validated.
- `qualifiers`: an array of the FULL disclosed caveat set as snake_case
  strings (a green's union of every contributing qualifier), e.g.
  `["fuzz","real_arithmetic"]` for an over-reals proof modulo a fuzz contract.
  The `composite_verdict` token is the weakest single badge; `qualifiers`
  carries every caveat so a consumer sees them all. A non-green outcome carries
  an empty array.
- `assumptions`: an array of assumption records. Each record has `name`,
  optional `source_type` / `producer`, optional
  `discharge:{method:"smt"|"fuzz"|"axiom", evidence:{...}}`, and optional
  `non_vacuity`. A missing or failed discharge degrades the composite verdict.

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
{"kind":"obligation","obligation_kind":"invariant_producer","source_type":"Probability","producer":"probability","name":"invariant:Probability:probability","status":"passed","composite_verdict":"proven_modulo_real_arithmetic","qualifiers":["real_arithmetic"],"assumptions":[{"name":"invariant:Probability:probability","source_type":"Probability","producer":"probability","discharge":{"method":"smt","evidence":{"status":"proved","obligation":"invariant:Probability:probability","arith_model":"real"}},"non_vacuity":{"status":"established","evidence":{"solver":"cvc5","result":"sat","assumption_count":0,"trivial":true}}}],"proof_tier":"smt","samples":0,"seed":0,"arith_model":"real"}
{"kind":"obligation","obligation_kind":"invariant_producer","source_type":"Probability","producer":"bad_prob","name":"invariant:Probability:bad_prob","status":"failed","composite_verdict":"failed","assumptions":[{"name":"invariant:Probability:bad_prob","source_type":"Probability","producer":"bad_prob","discharge":{"method":"smt","evidence":{"status":"failed","obligation":"invariant:Probability:bad_prob","counterexample":{"__arg0":"2.0"}}}}],"proof_tier":"smt","samples":0,"seed":0,"arith_model":"real","counterexample":{"__arg0":"2.0"}}
{"kind":"obligation","obligation_kind":"invariant_producer","status":"error","composite_verdict":"unsupported","assumptions":[],"reason":"opaque type `Probability`: exported producer `many` returns the type through an unsupported container (generic `List`); decompose-or-reject (RFC D-PRODUCER)"}
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

### Composition and non-vacuity

COMPOSE folds each result with the assumption discharges it depends on. A
fuzz-only BASE (the property/obligation itself established by fuzz, with no
SMT proof underneath) composes to `composite_verdict:"fuzz_validated"` and can
never compose to any `proven_*` badge: a base that was not proved is not
proven-modulo-anything. A base discharged by a sound over-approximation
composes to `"sound_approximate"`. For an SMT-proved base: all-SMT discharges
compose to `"proven_modulo_real_arithmetic"` (the proof is over the reals --
see the Tier B caveat); any fuzz CONTRACT discharge composes to
`"proven_modulo_fuzz_validated_contract"` unless a weaker discharge is present;
any asserted axiom composes to `"proven_modulo_asserted_axiom"`.
Missing/unsupported discharges compose to `"unsupported"`;
counterexample-backed discharges compose to `"failed"`. The single
`composite_verdict` token is the weakest badge; the `qualifiers` array carries
the full disclosed set (e.g. an over-reals proof modulo a fuzz contract is
token `proven_modulo_fuzz_validated_contract` with
`qualifiers:["fuzz","real_arithmetic"]`).

For every claimed green result that has assumptions, `chelis prove` checks
the assumptions alone for satisfiability. SAT establishes non-vacuity. UNSAT
sets `composite_verdict:"invalid"` and the record does not render as a pure
pass. cvc5 unknown/timeout sets `composite_verdict:"unsupported"` with a
reason such as `non_vacuity_unestablished: smt unknown`; it is never reported
as `failed`. A `failed` verdict requires a counterexample.

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
floating-point (and integer widths lower to an unbounded integer sort, so
overflow is not modelled either). The gap is now disclosed ON THE VERDICT:
such artifacts carry the `real_arithmetic` qualifier and render the
`proven_modulo_real_arithmetic` token (chelis#422), so a consumer reading
`composite_verdict` alone sees the caveat rather than only the legacy
`arith_model:"real"` side field (which is retained as a mirror). No
float-level soundness is claimed from a Tier B proof; admission policies
that quote the composed opaque-invariant guarantee inherit the disclosure
from the verdict. Tier C (`proof_tier:"fuzz"`) validates concrete float
samples and renders `fuzz_validated` (or, for a fuzz contract under an SMT
base, contributes the `fuzz` qualifier); it carries no `arith_model` field.

The real-arithmetic model does not license cvc5's partial operations outside
their mathematical domains. In particular, Tier B lowers `sqrt(a)` only after
an auxiliary obligation proves every exact argument `a >= 0` from independent,
sqrt-free total-algebraic conjuncts in the user's preconditions. The derived
domain facts are redundant assertions in the main query, never added user
assumptions. If an argument is not proved non-negative, or the proof times out,
is unknown, or lies outside that conservative fragment, Tier B returns a loud
unsupported result and `auto` routes to Tier C (chelis#1475).

## Relationship To `chelis test`

`chelis test` remains the deterministic assertion runner for `Std.Test`.
`chelis prove` is the property runner. V1 does not run properties
automatically from `chelis test`.
