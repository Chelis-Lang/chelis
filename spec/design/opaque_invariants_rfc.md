# RFC: Opaque Types With Declared Invariants (Option 1.5)

Version: 1 (frozen). Survey evidence:
[`opaque_invariants_survey.md`](opaque_invariants_survey.md).
Decisions carry stable IDs (`D-*`) for citation in workstream briefs,
commits, and red-team reports. Changing a frozen decision requires a
version bump and an explicit note in the owning PR.

## 0. Feature statement

A type can be declared **opaque**: constructible and inspectable only
inside its defining module, enforced by the type checker. An opaque
type can carry one **declared invariant**: a boolean predicate over a
single binder of the representation, written in Surf at the
declaration site, recorded as metadata on the Deep `deftype`. The
everyday type checker never evaluates the invariant; `chelis check`
stays solver-free. `chelis prove` consumes the invariant two ways:
derived producer obligations and assumption injection. Codec paths
revalidate the invariant on materialization.

**Composed guarantee.** A signature returning the opaque type means
every value of it in any well-typed out-of-module program originated
inside the defining module (a typing judgment), and every producing
path establishes the declared invariant (derived obligations,
discharged through the existing three-tier dispatcher).

**What this feature deliberately does not do.** The invariant is
invisible to `chelis check`. Downstream arithmetic does not inherit
bounds; the checker concludes nothing from the invariant; obligations
about derived values remain user-property territory. This is not
refinement typing: no predicates in the typing judgment, no
verification conditions at use sites, no solver in the check loop.

## 1. D-SOUND: the soundness argument (provenance induction)

Assumption injection — assuming `inv(x)` for every binder `x` of an
invariant-carrying opaque type during verification — is justified by
induction over value provenance:

- **Base case.** Every base producer (exported function whose inputs
  contain no value of the type) takes raw inputs; its derived
  obligation proves the invariant of every produced value
  unconditionally over its input domain.
- **Inductive step.** Every update-shaped producer (inputs contain
  the type) may assume the invariant of its inputs — justified
  because, by opacity, any input value reaching it from well-typed
  out-of-module code originated from some producer, and every
  producer carries a discharged obligation. Its obligation then
  proves the invariant of its outputs under that assumption. Mutual
  recursion among producers is sound under the same induction (each
  call's arguments are themselves provenance-traced).
- **Side conditions.** (a) The producer set must be complete —
  see D-PRODUCER's covered-or-rejected rule; one uncovered producer
  collapses the induction. (b) No value may enter from outside the
  program without revalidation — see D-DECODE. (c) In-module code is
  unconstrained by construction; the guarantee quantifies over
  out-of-module provenance only.

Red teams attack this argument as stated, not a reconstruction.

## 2. D-SYNTAX: surface syntax

```
@opaque
@invariant(p) p.value >= 0.0 and p.value <= 1.0
type Probability = | Probability { value: f32 }
```

- `@opaque` (renamed from the baseline's `@chelis_opaque`; the
  baseline is unreleased, so no alias or deprecation period).
- `@invariant(<binder>) <expr>` is optional and must appear between
  `@opaque` and `type`. Exactly one binder; exactly one invariant;
  ADT form required (annotating a type alias is a parse error).
  A bare `@invariant` without `@opaque` is an error at declaration:
  assumption injection would be unsound for a forgeable type.
- The representation idiom is the existing single-record-variant ADT.
  No anonymous-record grammar (`type T = { ... }`) ships in V1; the
  earlier draft's shorthand is explicitly dropped.

## 3. D-META: Deep metadata schema v1

`deftype` metadata keys (additive; unprefixed because they are now
language semantics, like `type`/`eff`/`lin`):

```lisp
(deftype {opaque: true,
          invariant: (fn {} (params {} p) <desugared predicate>),
          invariant_amenability: "linear"}
  Probability ()
  (variant {} Probability (field {} value (t-prim {} f32))))
```

- The predicate is embedded as a Deep `fn` inside the metadata map
  (metadata values are full Deep expressions; the strict validator
  recurses into them; a deftype *child* node would break positional
  variant parsing and grow the closed tag vocabulary).
- `invariant_amenability` is one of `"linear" | "polynomial" |
  "transcendental" | "opaque"` (the existing `SmtAmenability`
  vocabulary).
- `invariant_amenability` is derived data: not decompiled back to
  Surf, recomputed on every desugar.
- Schema versioning: these keys are additive under the workspace's
  additive-metadata pattern (spec/10 §3); this document is the
  versioned schema record. Any change to key names, the predicate
  encoding, or the amenability vocabulary bumps the RFC version.

## 4. D-CHECK: checker enforcement design

- Enforcement runs **inside inference** (post-annotation passes are
  unsound: type stamps are dropped on Error-typed nodes and `var`s).
  Prerequisite: dispatch arms and inference for `record`, `access`,
  and `record-update` (currently silently untyped — survey §1), which
  also closes the latent bogus-field hole.
- `AdtDef` gains `opaque: bool` and `defining_module:
  Option<String>`; both compiled-context cache versions bump.
- Module identity: one `module_key` helper over both encodings —
  lexical `(module ...)` nesting (joined with `.`) and the reef
  internal-name stem (`Pkg__<pkg>__<Module>__<Name>`). Top-level
  decls outside any module key as `None`, which is its own module.
- A thread-local `OpacityContext` (install-guard pattern, precedent
  `DECLARED_SIG_PARAM_TYPES`) carries the opaque map, current
  module, current decl, and the producer enumeration for error text.
- The rejection set, each a `CheckErrorKind::OpaqueTypeViolation`
  returning the **true type** (no error cascades): record literal;
  positional constructor application; bare constructor reference
  (the constructor binding itself is hidden); `pat-record`;
  `pat-ctor`; field `access` (typed targets now, deferred-variable
  ledger for targets resolved later in the def, mirroring the
  deferred-borrow ledger); `record-update` (Deep-only form);
  cast-into (both the Surf `(t-prim {} Name)` desugar shape and the
  Deep `(t-adt ...)` shape) and cast-out; `lit {type: (t-adt ...)}`
  forging (Deep-only).
- Error contract (agent-grade): the message names the type, the
  defining module, and the exported producers of that module with
  signatures; location context is message-embedded (def name).
  True spans in `CheckError` are out of scope (schema change),
  explicitly flagged.
- Exhaustiveness: outside code may match an opaque scrutinee only
  with irrefutable patterns. The top-level irrefutable `pat-var` arm
  false positive is verified empirically (expected-fail test first)
  and fixed at the arm level only; nested `pat-var` keeps
  not-covering.
- Macro expansion attributes to the call-site module (survey §3):
  fail-closed, documented, tested.
- Aliases and imports resolve to the nominal registry entry; opacity
  survives them by construction (survey §2); locked by tests.

## 5. D-LINT: lint rule disposition

`opaque-domain-construction` is kept as defense-in-depth: per-file,
no type context, fast editor/agent feedback, and its fail-closed
untyped-`record-update` check complements the checker's deferral.
spec/01 §12.1 is rewritten to name the checker as the authoritative
gate. The lint is fast feedback; the typing judgment is the
guarantee.

## 6. D-WF: invariant well-formedness (declaration-time)

Checked by a checker pass over Deep (covers `.ch` and `.dp`), with
parser-level twins for best-effort Surf diagnostics:

- `invariant` requires `opaque: true`.
- Representation: exactly one record-shaped variant; every field in
  the V1 value class — scalar prims, fixed-shape numeric tensors
  (all dims literal), or nested single-variant records of those.
- Predicate grammar (pure and total by construction): literals, the
  binder and its field projections, arithmetic (`+ - * /`),
  comparisons, `and/or/not`, `if`, whitelisted
  `abs/min/max/sqrt/exp/log/sin/cos`, and references to in-module
  zero-argument constant defs whose bodies are themselves in-grammar.
  Anything else — general calls, `match`, lambdas, tensor ops,
  effects — is a declaration error. Free references outside
  {binder} ∪ {in-module constants} are declaration errors.
- The predicate must be boolean-shaped at the top.
- Equality (`==`) in predicates draws an advisory warning (tier
  epsilon mismatch; see D-STARVE for the generation consequence).
- Amenability is recorded at desugar; the checker recomputes and
  errors on mismatch (protects hand-written `.dp`).
- The language rejects nothing on amenability grounds; it records.

## 7. D-PRED: the `chelis-pred` crate

New leaf crate (`crates/chelis-pred`, depends only on `chelis-deep`)
holding `PredAmenability`, `classify_predicate`,
`predicate_free_vars`, and `predicate_in_grammar`. Consumed by
chelis-surf (desugar-time recording), chelis-types (well-formedness
re-verification), and chelis-prove (consumption + `.dp`
recomputation). Classification: comparisons/boolean connectives over
affine arithmetic ⇒ Linear; a product of two non-constant subterms ⇒
Polynomial; any whitelisted transcendental ⇒ Transcendental; any
out-of-grammar node ⇒ Opaque. Binder field projections classify as
variables. `From<PredAmenability> for SmtAmenability` lives in
chelis-prove (keeps the existing tide surface stable).

## 8. D-PRODUCER: the producer set (covered-or-rejected)

The obligation set for an invariant-carrying opaque type: every
**exported** function of the defining module whose result type
contains the type in a produced position.

- Produced positions are computed from **checker-inferred** return
  types (the check pipeline runs before obligation collection), not
  only declared signatures — an unannotated exported def cannot
  escape the set.
- Supported decomposition: Direct; inside `Option[...]` (the
  standard failure-carrying wrapper, a builtin ADT); tuple
  components (each position).
- **Covered-or-rejected:** a return type containing the type through
  any unsupported container (record, list, function type, nested
  generic other than Option) is an error naming the producer. Never
  a silent skip — one uncovered producer collapses D-SOUND.
- Update-shaped functions (type in params and result) fall out of
  the same rule; their input assumption is D-INJECT.
- Explicit module with no `export` decl ⇒ zero producers (legal;
  the advisory `unreachable-producer` lint flags an opaque type
  with no exported producers). Bare-script files (no module
  wrapper) ⇒ all top-level defs are producers-eligible.
- Internal functions carry no obligations; their outputs escape only
  through exported ones.

## 9. D-OBLIG: obligation synthesis and output

- Per producer, prove synthesizes a property: for all (typed) inputs,
  where the producer succeeds, the invariant holds of every produced
  value. Names: `invariant:<Type>:<producer>`. Option-wrapped:
  `match producer(args) { Some(v) => inv(v), _ => true }` (vacuous
  `None`-always passage is asserted in tests and documented). Tuple:
  conjunction over positions.
- Obligations flow through the same engine as user properties: tier
  resolution, samples, seed, timeout, counterexamples, artifact,
  JSON, exit codes. Run by default; selectable/excludable via the
  existing selection mechanism.
- JSON (additive to `chelis_property_spec.md`'s NDJSON contract):
  `{kind:"obligation", obligation_kind:"invariant_producer",
  name, source_type, producer, status, proof_tier, samples, seed,
  counterexample?, reason?}`; summary gains `obligations: N`;
  exit-code meanings unchanged. `ProofArtifact` gains
  `obligation: Option<ObligationMeta{obligation_kind, source_type,
  producer}>` (serde-additive). Release notes call out the new
  record kind for strict downstream admission parsers (FlukeBall).

## 10. D-PARITY: one verifier across surfaces

Obligation collection/synthesis and assumption injection live in
`chelis-prove` (which already depends on chelis-surf and
chelis-deep), consumed by both the CLI prove path and the chelis-tide
MCP `chelis_prove` tool. A prove invoked through tide runs the same
derived obligations and injection as the CLI on the same module —
locked by a cross-surface parity test. Schema snapshots alone do not
satisfy this decision.

## 11. D-TIERB: SMT-tier coverage of the canonical producer

The flagship producer shape is guard-then-Option:

```
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 and x <= 1.0 then Some(Probability { value: x }) else None
```

Tier B lowering gains, in addition to the existing function-call
inlining: record beta-reduction (`Access` over a reduced `Record`
substitutes the field expression), `if` ⇒ SMT ite, and
case-of-known-constructor reduction (the obligation's `match` over an
inlined `if guard then Some(...) else None` reduces per branch). The
acceptance bar: a guarded-Option constructor's derived obligation
proves with `proof_tier:"smt"` for linear-arithmetic invariants.
Residual irreducible `match` bodies fall to Tier C (documented).
Clamping constructors proving at SMT tier is necessary but not
sufficient — the consuming use case bans them.

SMT proofs are over the reals; runtime arithmetic is floating-point.
smt-tier artifacts carry `arith_model: "real"` (additive), and
spec/09 carries a verbatim-citable caveat so admission policies quote
the gap instead of rediscovering it. No float-level soundness is
claimed from a Tier B proof.

## 12. D-INJECT: assumption injection

During verification of any property (user or derived), every binder
whose type is an invariant-carrying opaque type — including such
types nested in record binders, recursively — gets the invariant
injected as a precondition, composed with (never replacing) existing
preconditions.

- Tier B: opaque binders are flattened to per-field solver variables
  (`p.value`, dotted paths for nesting); the rewritten invariant is
  asserted alongside user preconditions.
- Tier C: binder values are generated per D-STARVE; user
  preconditions keep their existing rejection-sampling semantics and
  their existing exhaustion behavior (`Error`).
- Injection applies only to invariant-carrying opaque types; never
  to non-opaque or invariant-free types (test-locked).

## 13. D-STARVE: generator strategy per invariant shape

Classified at design time, not discovered empirically:

- **Inequality-shaped predicates** (no equality atoms over the
  binder): per-binder rejection sampling. Budget:
  `--invariant-min-rate <f64>` (default `0.01`); when the observed
  satisfaction rate falls below the floor, the property aborts as
  `Unsupported` (exit 2) with a generator-starvation diagnostic
  naming the type, accepted/attempted counts, rate, floor, the
  predicate's shape classification, and the recommended route.
  `--invariant-min-rate 0.0` disables the classification (legacy
  exhaustion ⇒ `Error` path preserved). Deterministic under a fixed
  seed.
- **Equality-constrained / measure-near-zero predicates** (any `==`
  atom over binder fields, e.g. a simplex's sum-to-one): rejection
  sampling is structurally starved (acceptance ~ε). These route to
  Tier B where the property lowers; for Tier C,
  **constructor-based generation** is in scope: binder samples are
  produced by evaluating the module's exported producers on sampled
  raw inputs (Option-unwrapping failures). Sound under D-SOUND —
  producers are obligation-checked — and that dependency is stated
  here for red-team attack.
- The flagship simplex example (W6) must verify under injection
  without starving; it is the acceptance probe for this decision.

## 14. D-DECODE: decode revalidation

No codec or deserialization path may materialize a value of an
invariant-carrying opaque type without checking the invariant at
decode time. Decode of a violating payload is a failure, never a
repair. NaN comparisons are false, so NaN payloads fail closed.

V1 reality (survey §7): no external ADT payload codec exists yet.
V1 therefore ships: `revalidate_adt_value` in the evaluator
(predicate evaluated via the interpreter's own `eval_expr`;
compiler-api does not depend on chelis-prove), a public
`decode_adt_value` chokepoint (the contract point future codecs MUST
call — normative rule added to spec/10), and a conformance suite over
valid/boundary/violating/NaN/structurally-corrupt payloads. The
chokepoint is documented experimental until a real codec consumes it.

## 15. D-CORPUS: differential and regression coverage

In-tree Python corpus generator emitting opaque/invariant
declarations and out-of-module usage; coverage measured (every new
rejection and injection diagnostic hit by ≥1 generated program), not
assumed. An explicit regression test asserts `chelis check` remains
solver-free on the corpus (the non-smt build checks it identically).
Prove performance sanity on a many-producer module. Hull-side
reference support is an out-of-repo follow-up, not claimed.

## 16. Compatibility

- Deep metadata: additive keys only; no existing metadata changes
  meaning.
- prove JSON: additive records and fields only; `proof_tier`,
  `smt_status`, counterexamples, seeds, exit codes unchanged in
  meaning.
- CLI: existing invocations behave identically on code using none of
  the new forms (locked by the existing suites plus D-CORPUS).
- The feature ships in a minor release; downstream consumers pin to
  it explicitly.
