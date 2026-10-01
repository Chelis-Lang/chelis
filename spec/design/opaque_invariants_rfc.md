# RFC: Opaque Types With Declared Invariants (Option 1.5)

Version: 7 (frozen). Survey evidence:
[`opaque_invariants_survey.md`](archive/opaque_invariants_survey.md).
Decisions carry stable IDs (`D-*`) for citation in workstream briefs,
commits, and red-team reports. Changing a frozen decision requires a
version bump and an explicit note in the owning PR.

Version history:
- v7: editorial only, no decision change (surfaced during W2). The
  predicate examples used the word `and`; Surf's boolean conjunction
  operator is `&&` (the word `and` does not parse as a conjunction).
  Corrected the three example predicates and the simplex tolerance
  band to `&&` and to `sum(p.weights)` so the examples are
  executable. No contract, grammar, or amenability decision changed.
- v6: RT-1 final-pass bypass of the v5 fix. The v5 reserved-name
  suppression (flag TRUE in linked contexts) is sound ONLY when
  paired with guaranteed re-mangling of every decl name in that
  context. The `chelis test` entry path (`prepare_eval_in_context`
  on formatted synth decls) re-asserts the linked flag but does NOT
  re-mangle the user-authored test/entry decls — so a hand-authored
  `pkg__victim__forge`-format def name in a reef package test file
  self-keys to the victim module and forges the real opaque type;
  `chelis test` accepts it (eval/check/build/validate all reject,
  because they re-mangle entry decls or apply the rule
  unconditionally). Decision — make the pairing structural, two
  layers: (a) the test/entry path re-mangles user-authored
  entry/test decl names exactly as the eval entry path
  (`rewrite_entry_decls_with_reef_graph`) does, so a user decl can
  never self-key to a victim module; (b) belt-and-suspenders: the
  `ReservedLinkerName` check stays ACTIVE for user-authored
  entry/test decls regardless of the surrounding linked flag — the
  flag answers "did the LINKER produce this decl," not "is there
  linked content in this check unit," and user entry decls are never
  linker output. Sibling-sweep requirement: every code path that
  sets the linked flag TRUE must be paired, at the same boundary,
  with re-mangling of any user-authored decls it admits; the sweep
  enumerates all such paths and proves the pairing, so a future
  flag-TRUE path cannot silently reintroduce this class.
- v5: RT-1 verification bypass. The F2 fix guarded only lexical
  `(module ...)` wrappers, but module identity also derives from
  reef-stem mangled names (`Pkg__pkg__Module__Name`), so a
  hand-authored program using that name format forges module
  identity through the stem channel and constructs/inspects the
  opaque type clean on check + validate + build (pure-flat
  `stem_only.dp`, no wrappers, scores 1). Decision: the reef-internal
  mangled-name format is the linker's PRIVATE output and is rejected
  as a declaration error in any program not produced by the reef
  linker — threaded by an in-process provenance flag set at the link
  boundary (`prepare_program_for_file`), NOT by filesystem or
  heuristic detection. This restores the invariant that
  hand-authored module identity (Surf or raw `.dp`) comes only from
  lexical wrappers, which the F2 fix guards. Verified safe: no
  tracked `.dp`/fixture/corpus file uses mangled names, and the
  linker feeds linked Deep to the checker in-process without ever
  serializing to `.dp` for re-ingest, so the flag is never lost.
  Belt-and-suspenders: even in linked input, a stem-derived
  defining-module colliding with a lexical wrapper key in one check
  unit is a `DuplicateModule` error (genuine linker output has no
  lexical wrappers, so this never fires legitimately).
- v4: RT-1 (forgery suite) fixes. (a) The sixth rejection must be
  enforced on the reef package surface: the reef link path must
  attribute exports and T-mentioning bindings into the opacity
  metadata (fail-open there was a W1 residual; RT-1 F1 proved the
  unexported-producer attack passes `check` AND `build` in a real
  package, collapsing D-SOUND side condition (a) on the primary
  packaging surface). (b) New rejection: a named module may be
  opened by at most one `(module ...)` wrapper per check unit;
  re-opening a module name is an error (RT-1 F2 — module identity
  is otherwise a forgeable string; follows the same ambiguity
  rationale as the named-module requirement in M6). (c) The
  violation-message contract requires user-facing (de-mangled)
  type/module/producer names on the reef surface, with
  exact-message test assertions (RT-1 F3). (d) Decompiler
  round-trip is a mandatory invariant on the opaque path:
  multi-segment module names and record variants must re-parse
  (RT-1 F4; repo contract invariant).
- v3: RT-0 fix-verification residuals. Sixth rejection broadened to
  signature-mentions-T (closes the unexported caller-receives HOF
  residual and makes "fully sealed" accurate); escape-site lint made
  two-level (complete egress enumeration with provenance labels) and
  extended to T-producing function values, with the indirect-callee
  fail-closed rule and the one-hop laundering shape added to RT-3's
  required attacks; signature rejection re-scoped by polarity
  (caller-receives only — consumption positions are legal, restoring
  D-INJECT's record-binder support); the invariant-only scoping
  cliff reconciled explicitly; W6 flagship invariant pinned as a
  tolerance band with exact float `==` documented as
  Tier-C-starves-by-design; Inf-narrowing of the decode pre-check
  documented; D-SYNTAX example wrapped in a named module.
- v2: RT-0 fixes. Closes the two CRITICAL soundness holes (C2:
  sixth rejection for unexported-producer references; C1: exported
  signatures with caller-receives occurrences of the type are
  declaration errors; argument egress handled by the explicit-TCB
  model with a provenance-aware escape-site lint). Repairs the NaN
  fail-closed claim (representation sanity pre-check), corrects the
  lit-forge surface labeling (reachable from Surf ascription),
  whitelists `sum` over fixed-shape tensor fields, makes Tier C
  generation tiered-and-validated (dissolving the constructor-trust
  circularity), relocates the Tier B lowering home, widens the
  exhaustiveness fix to `pat-as`-wrapped irrefutable arms, requires
  a named module for `@opaque`, and pins the smaller RT-0 items
  (sig-only defs, exported constants, recursive container
  decomposition, whitelist consolidation, duplicate-deftype lock).
- v1: initial frozen contract.

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
  collapses the induction. Completeness requires both that
  unexported producers are unreachable from outside (the sixth
  rejection, D-CHECK) and that exported signatures cannot hand
  caller-supplied code unobligated values (the signature rejection,
  D-PRODUCER). (b) No value may enter from outside the program
  without revalidation — see D-DECODE. (c) In-module code is
  unconstrained by construction; the guarantee quantifies over
  out-of-module provenance only.

**The explicit trust caveat (argument egress).** Return-egress is
mechanically obligated; argument egress is not: in-module code that
passes a value of the type as an argument to an out-of-module callee
is trusted not to leak an invariant-violating value through that
channel. This is the residual of the classic abstract-type trust
model after derived obligations mechanize the output boundary. The
quotable form, for downstream admission manifests that cite the
composed guarantee:

> Producer obligations mechanically cover every value of the type
> returned across the module boundary. Values of the type — and
> function values capable of producing it — that the defining module
> passes outward as call arguments are covered by module audit, not
> by machine-discharged obligations. The advisory
> `opaque-escape-site` lint enumerates every such site, labeled by
> local provenance; transitive flows within the module are the
> audit's responsibility.

The `opaque-escape-site` lint (D-LINT) enumerates **every**
in-module argument-egress site — a call passing a value of the type,
or a function value capable of producing it (the bare constructor, a
closure whose return type contains it), to an out-of-module callee —
at two levels: **note** for sites locally attested (the value traces
to a producer call or a type-T input of the enclosing function) and
**warning** for unattested sites (the value traces to a raw
construction or representation update). The enumeration is complete;
the provenance labels direct the audit. A call through a
function-typed value of unknown provenance (a function parameter, a
stored closure) counts as out-of-module — fail-closed. Local
attestation is one-hop by design: a helper `g(p: T) = outside.f(p)`
attests its own egress, and the audit follows callers of `g` —
transitive flows are precisely what the module audit owns. RT-3's
soundness suite must include adversarial leaks through argument
egress in BOTH shapes — the naive direct leak and the one-hop
laundered leak through an attested helper — and confirm the lint's
enumeration surfaces the audit path for each. The designed follow-up
(out of V1 scope, queued): provenance-triggered escape obligations —
only raw-construction-traced arguments carry them — bundled with the
HOF extension, since both are the same escape-analysis family.

Red teams attack this argument as stated, not a reconstruction.

## 2. D-SYNTAX: surface syntax

```
module Stats.Prob

@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability = | Probability { value: f32 }
```

(The enclosing named module is required — see D-CHECK; `@opaque`
outside a named module is a declaration error.)

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
- A thread-local `OpacityContext` install guard carries the opaque map,
  current module, current decl, and the producer enumeration for error text.
  This opacity-only mechanism is not a precedent for type binders: declaration
  type/dimension/rank binders live in the explicit, serde-skipped
  `TypeResolutionScope` field on `Env` and follow lexical `Env` clones; there
  is no ambient `DECLARED_SIG_PARAM_TYPES` state.
- The rejection set, each a `CheckErrorKind::OpaqueTypeViolation`
  returning the **true type** (no error cascades): record literal;
  positional constructor application; bare constructor reference
  (the constructor binding itself is hidden); `pat-record`;
  `pat-ctor`; field `access` (typed targets now, deferred-variable
  ledger for targets resolved later in the def, mirroring the
  deferred-borrow ledger); `record-update` (Deep-only form);
  cast-into and cast-out (Deep `cast` with `t-prim` or `t-adt`
  targets — Surf has no cast-into-ADT surface form today, but the
  Deep gate is required regardless); `lit {type: (t-adt ...)}`
  forging — reachable from BOTH surfaces: Surf expression ascription
  (`0.5 : Probability`) and block-binding ascription desugar to
  exactly this lit metadata, so the gate must not be scoped to `.dp`
  ingestion; and the **sixth rejection** (RT-0 C2, broadened in v3):
  an out-of-module reference to an *unexported* binding of an
  opaque-defining module whose **signature mentions the opaque
  type** — in the result, in any parameter, in function-typed
  parameter domains, or as a non-function binding's type. The
  broad scope (signature-mentions-T, not just result-contains-T)
  closes the unexported caller-receives HOF channel: an unexported
  `with_t(f: T -> f32) -> f32` must not be callable from outside, or
  it hands caller code unobligated values while being held to a
  weaker standard than its exported twin. "Mentions" means
  containment chased through named type definitions, not a syntactic
  scan: an unexported `helper() -> MyRecord` where the non-opaque
  `MyRecord` carries a T field mentions T. The sixth rejection
  applies to all opaque types (not only invariant-carrying ones) so
  that adding an invariant later does not change which references
  are legal for *other modules'* code; a module with no `export`
  decl is fully sealed for every T-mentioning binding (the
  `unreachable-producer` advisory flags an opaque type left with
  zero exported producers).
- `@opaque` requires a named enclosing module (declaration error
  otherwise; RT-0 M6). Top-level code outside any module shares one
  anonymous module key per check unit, which would make the
  enforcement boundary collide across combined sources; requiring a
  named module removes the ambiguity. Duplicate same-name `deftype`
  across modules is already rejected (`DuplicateDefinition`); W1
  locks that rejection with a test as an opacity invariant, since
  the registry insert is otherwise last-write-wins.
- Error contract (agent-grade): the message names the type, the
  defining module, and the exported producers of that module with
  signatures; location context is message-embedded (def name).
  True spans in `CheckError` are out of scope (schema change),
  explicitly flagged.
- Exhaustiveness: outside code may match an opaque scrutinee only
  with irrefutable patterns. The top-level irrefutable arm false
  positive is now verified (RT-0 probes): a bare `| x =>` arm AND a
  `pat-as`-wrapped `| q @ x =>` arm both false-positive
  `NonExhaustiveMatch`, while `| q @ _ =>` covers. The fix covers
  both irrefutable shapes — top-level `pat-var` and `pat-as` whose
  inner pattern is irrefutable — at the arm level only; nested
  `pat-var` keeps not-covering.
- Macro expansion attributes to the call-site module (survey §3):
  fail-closed, documented, tested.
- Aliases and imports resolve to the nominal registry entry; opacity
  survives them by construction (survey §2); locked by tests.

## 5. D-LINT: lint rule disposition

`opaque-domain-construction` is kept as defense-in-depth: per-file,
no type context, fast editor/agent feedback, and its fail-closed
untyped-`record-update` check complements the checker's deferral.
The W1 rename rekeys the rule from `chelis_opaque`/`@chelis_opaque`
to `opaque`/`@opaque` in the same change set (unmigrated it would
silently never fire — RT-0 M7). spec/01 §12.1 is rewritten to name
the checker as the authoritative gate. The lint is fast feedback;
the typing judgment is the guarantee.

New advisory lints: `opaque-without-invariant` (note),
`unreachable-producer` (opaque type with no exported producers), and
`opaque-escape-site` (RT-0 C1 shape 2). The escape-site lint's
contract is defined once, in D-SOUND §1, and that definition is
authoritative: complete enumeration of every in-module
argument-egress site — values of the type AND function values
capable of producing it — at two levels (note for locally-attested
provenance, warning for unattested), with indirect callees treated
as out-of-module (fail-closed). Local dataflow only; no prove
machinery; the audit surface for the explicit trust caveat.

## 6. D-WF: invariant well-formedness (declaration-time)

Checked by a checker pass over Deep (covers `.ch` and `.dp`), with
parser-level twins for best-effort Surf diagnostics:

- `invariant` requires `opaque: true`.
- Representation: exactly one record-shaped variant; every field in
  the V1 value class — NUMERIC/BOOLEAN scalar prims (`f32`, `f64`,
  the signed integer widths, `bool`; NOT `string`/`f8e4m3`),
  fixed-shape `f32`/`f64` tensors (all dims literal), or nested
  single-variant records of those. The class is exactly the
  representations the prover can mechanically verify; the D-WF check
  and the prover's field model share one definition
  (`chelis_types::invariants::invariant_value_class_prim`), so a
  checker-admitted representation always reaches obligation collection
  and is NEVER silently dropped (covered-or-rejected). An
  invariant-carrying opaque type the prover cannot model is a
  collection-time `Error` outcome, not zero obligations.
- Predicate grammar: literals, the binder and its field projections,
  arithmetic (`+ - * /`), comparisons, `and/or/not`, `if`,
  whitelisted `abs/min/max/sqrt/exp/log/sin/cos`, `sum` over a
  tensor-typed binder field whose shape is fully literal (RT-0 M1 —
  required by the simplex flagship; the classifier and Tier B
  lowering expand it to finitely many scalar terms), and references
  to in-module zero-argument constant defs whose bodies are
  themselves in-grammar. Anything else — general calls, `match`,
  lambdas, other tensor ops, effects — is a declaration error. Free
  references outside {binder} ∪ {in-module constants} are
  declaration errors.
- "Pure" is by construction; "total" is not (RT-0 M3): `/`,
  `log`, and `sqrt` are partial or non-finite on parts of their
  domain. Pinned semantics for predicate evaluation error or
  non-finite results: at decode, the decode fails; in Tier C
  sampling, the candidate sample is rejected; in Tier B, real
  semantics apply and the `arith_model` caveat covers the gap.
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
variables; `sum` over a literal-shape tensor field expands to its
scalar terms before classification. `From<PredAmenability> for
SmtAmenability` lives in chelis-prove (keeps the existing tide
surface stable). chelis-pred exports the canonical
transcendental/intrinsic whitelist; `chelis-prove`'s inlineability
module consumes it rather than keeping a second copy (RT-0 L5).

## 8. D-PRODUCER: the producer set (covered-or-rejected)

The obligation set for an invariant-carrying opaque type: every
**exported** function of the defining module whose result type
contains the type in a produced position.

- Produced positions are computed from **checker-inferred** return
  types (the check pipeline runs before obligation collection), not
  only declared signatures — an unannotated exported def cannot
  escape the set.
- Supported decomposition, applied **recursively**: Direct; inside
  `Option[...]` (the standard failure-carrying wrapper, a builtin
  ADT); tuple components. Compositions like `Option[(T, f32)]` and
  `Option[Option[T]]` decompose; anything else rejects.
- **Covered-or-rejected:** a return type containing the type through
  any unsupported container (record, list, function type, generics
  other than Option) is an error naming the producer. Never a
  silent skip — one uncovered producer collapses D-SOUND.
- **Signature rejection (RT-0 C1 shape 1; polarity-scoped in v3):**
  for invariant-carrying opaque types, exported signature positions
  are judged by which side receives the value:
  - *Produced positions* (the return side): decompose-or-reject per
    the rules above.
  - *Caller-receives positions* (the domain of a function-typed
    parameter, transitively): declaration error naming the function
    and the channel — the module would hand caller-supplied code
    unobligated values.
  - *Module-receives positions* (plain type-T parameters, record
    parameters with T fields, caller-implemented function parameters
    whose RETURN contains T): **legal**. Outside code can only
    assemble these from legally obtained values, so the induction
    covers them — and D-INJECT's support for T nested in record
    binders depends on record parameters staying legal.
  Scoped to invariant-carrying types: the channel only breaks
  assumption injection, and plain opacity is unaffected by it. The
  scoping asymmetry with the sixth rejection is deliberate and
  reconciled as follows: the sixth rejection's all-opaque scope
  protects *other modules'* code from a semantics cliff when an
  invariant is added; the signature rejection's errors land in the
  defining module's own declarations, where adding `@invariant` is
  the author's own act of strengthening the type's contract and is
  expected to impose new conditions on the module's exported API.
- Exported **non-function bindings** whose type contains the type
  are producers too (RT-0 L4): the obligation is that the invariant
  holds of the constant value.
- A **sig-only def** (defsig with no body) in the defining module
  whose result contains the type is covered-or-rejected: error at
  obligation collection (no body to prove). An out-of-module
  sig-only declaring the type's return remains legal — that is what
  imports look like, and it can materialize no values.
- Update-shaped functions (type in params and result) fall out of
  the same rule; their input assumption is D-INJECT.
- Explicit module with no `export` decl ⇒ zero producers, and the
  sixth rejection (D-CHECK) makes the type fully sealed — internal
  producers are not callable from outside, keeping the premise of
  the next bullet true. The advisory `unreachable-producer` lint
  flags an opaque type with no exported producers. `@opaque`
  requires a named module (D-CHECK), so there is no bare-script
  producer rule.
- Internal functions carry no obligations; their outputs escape only
  through exported ones — a premise the sixth rejection enforces
  rather than assumes (RT-0 C2).

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
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
```

Tier B lowering gains, in addition to the existing function-call
inlining and `if` ⇒ ite (which already exists — RT-0 L2): record
beta-reduction (`Access` over a reduced `Record` substitutes the
field expression) and case-of-known-constructor reduction (the
obligation's `match` over an inlined `if guard then Some(...) else
None` reduces per branch). The acceptance bar: a guarded-Option
constructor's derived obligation proves with `proof_tier:"smt"` for
linear-arithmetic invariants. Residual irreducible `match` bodies
fall to Tier C (documented). Clamping constructors proving at SMT
tier is necessary but not sufficient — the consuming use case bans
them.

**Lowering home (RT-0 M4):** the Surf→SMT lowering machinery —
including the existing inlining and the new reductions — relocates
into `chelis-prove` as part of the D-PARITY consolidation, so the
CLI and tide paths share one lowering. The parity test asserts not
just the same obligation set but the same `proof_tier` per
obligation across surfaces.

For invariant-carrying types whose representation includes
fixed-shape tensor fields, Tier B flattening expands tensor fields
to per-element solver variables, capped (total scalar count ≤ 64)
— beyond the cap the property falls to Tier C.

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

## 13. D-STARVE: tiered, validated generation

Binder generation for invariant-carrying opaque types is tiered, and
every accepted sample is validated against the predicate regardless
of how it was proposed — generation methods are proposal
distributions, never trust sources:

1. **Rejection sampling** of the representation against the
   predicate, within budget.
2. **On starvation, constructor-based generation**: candidate binder
   values are produced by evaluating the module's exported producers
   on sampled raw inputs (Option-unwrapping failures) — a smarter
   proposal distribution for predicates whose satisfying set is
   measure-near-zero under independent component sampling (equality
   atoms, near-zero-width bands; the simplex's sum-to-one is the
   flagship case). Each generated sample is still checked against
   the predicate before acceptance, so a buggy producer costs
   sampling efficiency, never soundness — this removes any ordering
   dependency between producer obligations and generation (RT-0 M2).
   A degenerate producer (constant output) yields full acceptance
   with zero domain coverage; the diagnostic reports distinct-sample
   counts so this is visible.
3. **Both starved ⇒ `Unsupported`** (exit 2) with a
   generator-starvation diagnostic naming the type,
   accepted/attempted counts per method, rate, floor, the
   predicate's syntactic shape classification (equality atoms /
   band widths — recorded to explain the starvation, and to skip
   straight to step 2 when obvious), and the recommended route
   (Tier B where the property lowers; a richer producer set
   otherwise).

Budget: `--invariant-min-rate <f64>` (default `0.01`) floors step 1;
`--invariant-min-rate 0.0` disables the starvation classification
(legacy exhaustion ⇒ `Error` path preserved). Deterministic under a
fixed seed. User-precondition rejection sampling keeps its existing
semantics and exhaustion behavior (`Error`) — the two failure modes
stay distinguishable.

**Exact float equality starves by design (RT-0 NEW-1).** Validation
is strict — no epsilon semantics anywhere, because epsilon-validated
samples would weaken exactly the soundness that validation provides.
Consequently an invariant using exact `==` over float fields is
nearly unsatisfiable as a float predicate: step 1 starves
geometrically and step 2 starves too, since even a correct
normalize-style producer emits sums merely *near* the target. This
is treated as a defect of the invariant, not the generator: the
documented idiom for float aggregates is a tolerance band over a
module constant (`sum(p.weights) >= 1.0 - eps && sum(p.weights) <= 1.0 + eps`), which a
correct producer satisfies at high rate under constructor-based
proposals. The `==` advisory (D-WF) points at this section; the
starvation diagnostic for equality-shaped predicates names Tier B
(real semantics) as the only verification route.

The flagship simplex example (W6) uses the tolerance-band invariant
and must verify under injection without starving; it is the
acceptance probe for this decision.

## 14. D-DECODE: decode revalidation

No codec or deserialization path may materialize a value of an
invariant-carrying opaque type without checking the invariant at
decode time. Decode of a violating payload is a failure, never a
repair.

**Representation sanity pre-check (RT-0 H1).** "NaN comparisons are
false" does NOT make NaN payloads fail closed for all in-grammar
predicates: `not (p.value > 1.0)` is *true* on NaN, and `p.value !=
5.0` is true on NaN under IEEE semantics. Decode therefore rejects
any payload containing a NaN or infinite value in a numeric
representation field BEFORE predicate evaluation, restoring
unconditional fail-closed behavior. Predicate evaluation error
(division by zero, domain errors) also fails the decode (D-WF).
The pre-check deliberately narrows types whose invariants would
legitimately admit infinities (e.g. a log-probability field wanting
`-Inf`): such representations are outside V1's decodable class. Any
future relaxation must re-derive fail-closed semantics for the
admitted non-finite values rather than silently reopening H1.

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
- Tooling output discloses representation contents (`chelis eval`
  prints constructor and fields; Tier C counterexamples print
  representation values). Disclosure-only: none of these outputs are
  re-importable as typed values (eval bindings are tensors-only), so
  opacity's construction guarantee is unaffected. Documented so the
  guarantee is not over-read as secrecy.
