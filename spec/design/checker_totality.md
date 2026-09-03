# Checker Totality: every construct is checked or loudly rejected

**Status:** Phases 0-3 and PP1-PP4 are delivered. PP5 is partial: it delivers
the four results listed in its own section, three under [#668] and the C
host-value lane's operand guard under [#1484], and makes no claim beyond
them. PP6 is decided in its own section and not delivered; its three
issues remain open until its oracle is green. PR [#1406] delivered the
separately owned [#1247] kinded nominal-application residue; the bounded
[#1125] nominal-rank ingress repair and [#1134] forward-reference parity are
also delivered residue rather than additional phases. Phase 3 first shipped
`DeepTag` as derive-on-demand dispatch and was red-teamed in that form
(round-1 QUALIFIED PASS with findings folded, round-2 PASS); a maintainer
directive then superseded that record with the decode-once rework now in the
tree (the parser stamps `Atom::Tag`, the tag string does not exist in memory,
and the [#858]/[#859] discoveries are fixed rather than deferred). Fresh
local-subagent reviews of the rework ran on PR #855 and their findings were
folded before merge; the former pending-review status was stale. [#908] is
now replacing that physical carrier with private validated `Node`; the
replacement is a successor only when it preserves the decode-once,
exhaustive-disposition, and continuous-oracle guarantees below. Tracking
issue: [#731].
**Owning specs:** `spec/03-deep-syntax.md` (the 62-tag closed vocabulary),
`spec/04-type-system.md` (what "checked" means per construct; its §10
carries this plan's decided contract as current blockquote authorities
[04-TOT-1..3], whose semantics this plan ratifies independently of their
later chelis#733 migration through the pinned Buoy shell-side integration),
the repo Contract Invariants ("if a command reports perfect success, its error list
must be empty"), and the audit record in
`docs/investigations/numeric_audit_next_sweeps.md` (sweep 6) /
`docs/investigations/numeric_audit_structural_prevention.md` (item 5).
**Class fixed:** [#709] (a construct with no `infer.rs` case silently
disables type checking for its subtree) and the silent half of [#710]
(`.dp` guard paths that return `Type::Error` without a diagnostic).
**Sibling plans:** `spec/design/loud_unsupported.md` ([#730]) is the
same disease in the lowering/codegen organ - there the default substitutes
a *value*; here it substitutes a *verdict*. The `EffectKind` enum is
delivered by [#730] Phase 2 and consumed here (§I1).

## Summary

Before Phase 1, `chelis check` could report a perfect score of 1.0 on an
ill-typed program. The mechanism was an unknown-tag wildcard and malformed
guards returning bare `Type::Error` without pushing a `CheckError`; because
that sentinel unified with anything, a construct and its subtree could be
silently exempted. The executed blast radius included `with seed` / `with
device`, a runnable host binary with an invalid declared return type, and a
tensor case that reached clang instead of a checker diagnostic.

Phase 1 closed the known holes. Phase 2 makes the class structurally hard to
recreate: a fresh error type requires a witnessed, append-only diagnostic;
checked-result construction is fallible; annotation consumes the types from
the owning inference epoch; and one finalizer validates the annotated tree
before the session boundary may return success. Phase 3 remains the separate
compile-time exhaustiveness ratchet for the next Deep tag.

## Phase boundaries

Phase 1 provides the loud checker cases for the known holes. Phase 2 makes a
silent `Type::Error` unconstructible by coupling fresh error types to the
authoritative diagnostic session, preserving inferred type provenance through
annotation, and validating every successful checked result. Phase 3 adds the
`DeepTag` enum at dispatch chokepoints so that a new tag makes every undecided
consumer fail to compile. The Deep AST representation remains unchanged: the
canonical serialized form (the 3-tuple shape and spec/03's closed string-tag
vocabulary) is frozen, and `DeepTag` is an in-memory artifact of parsing.
Whether the parsed node carries the enum alongside its validated string
(§C4.2's reading) or the dispatch chokepoints derive it on demand is Phase 3's
implementation choice; the typed-vocabulary pattern from [#730] Phase 2
(PR #799), a single declaration whose exhaustive typed consumers turn a new
variant into a compile-time work-list, is the same shape proven for
`EffectKind`. Initially executed as derive-on-demand;
SUPERSEDED at the 2026-07-24 maintainer rework by the stronger decode-once
representation §C4.2 now mandates: the parser stamps `Atom::Tag(DeepTag)` at
element 0, the tag string does not exist in the parsed tree, and every
producer and consumer crosses the typed constructors/accessors. The enum
lives in `chelis_deep::tag` (every chokepoint crate already depends on
chelis-deep for the AST, and the parser that enforces the closed vocabulary
lives there); `chelis_deep::validate::VALID_TAGS` is derived from
`DeepTag::ALL` and the duplicated `chelis-validate` string copy is retired.
The serialized `.dp` form is unchanged; the in-memory `Atom` gained the
`Tag` variant, and the typecheck cache's envelope identity check absorbs the
representation change by invalidation.

**2026-08-01 successor interlock.** The frozen Phase-3 result is the logical
contract, not the spelling `Atom::Tag` itself: one boundary decodes the closed
vocabulary to `DeepTag`, every semantic consumer chooses an exhaustive typed
disposition, raw vocabulary strings are confined to pre-decode/serialization
boundaries, and the standing oracle traverses every enforcement-relevant
child. [#908] may replace `List + Atom::Tag` with `Node { tag: DeepTag, ... }`
only as an at-least-as-strong successor. During the migration both carriers
are transitional; construction of a private Node followed by normalization
back to public List does not by itself supersede this Phase-3 contract.

## Phase 2 architecture

The current implementation has one explicit ownership chain:

1. `session::DiagnosticSink` owns the check's one canonical error vector. Its
   storage and constructor are private to `session.rs`; lower layers can only
   append. There is no production clone/default/conversion, mutable deref,
   extraction, `retain`, clear, truncate, drain, or post-report deletion seam.
   `session::run_result` is the central success boundary: any non-empty sink
   vetoes `Ok` and returns the exact diagnostics.
2. `errors::report` appends through that sink and mints a fresh
   `ErrorWitness`; `propagate` copies an existing witness without reporting a
   duplicate. Production resolver/validation paths cannot use an arbitrary
   `Vec<CheckError>` as a diagnostic output.
3. Deep type syntax crosses one located `DeepTypeResolver`. The resolver owns
   its use site, binder mode, nominal-header environment, variable generator,
   and source/owner location, reports a failure once, and returns
   `Result<ResolvedDeepType, ErrorWitness>`.
4. Source binder state is lexical, not ambient. A serde-skipped
   `TypeResolutionScope` field on `Env` is installed on the cloned environment
   for one declaration and inherited only by its nested lexical clones; it
   cannot leak into a sibling declaration, a stacked check, or a cached
   `TypeEnv`.
5. Each primary inference root opens an `InferenceProduct` owner epoch.
   Registration and finalization use one exhaustive child-role table:
   `RuntimeExpr`, `Syntax`, `Selector`, `EffectHandler`, `Binder`, `Type`, and
   `ExplicitInferenceBypass`. `EffectHandler` is deliberately effects-owned;
   the handled body remains a type-owned runtime child. During `finish_root`,
   the final substitution is applied to every recorded owner write,
   conflicting writes are rejected, and canonical types for stamp-required
   owners remain available through annotation.
6. Annotation consumes those finalized canonical owner types. It does not
   semantically re-infer expressions with a fresh `VarGen`/`Subst`; a missing
   or conflicting owner write is a diagnostic, never a default type.
7. `finalize_checked_program` checks fresh inference results: the annotated
   runtime tree and pattern/function stamps, with input and output signature
   metadata as structural backstops. There is no public raw `from_parts` or
   `try_from_parts` API. The effects pass can call only
   `CheckedProgram::try_with_effect_annotations`: it proves every root, span,
   atom, child, and metadata entry is identical to the checked input except
   the effects-owned `effects` entry, preserves the original type environment,
   signature inference, and linearity, and reruns totality validation. A
   violation maps to `EffectErrorKind::TypeTotality` exactly once. The other
   checked-result operations have disjoint ownership:
   `CheckedProgram::with_linearity` changes only linearity metadata, while
   `CheckedProgram::compose` combines two already-successful, context-stacked
   checked halves. Neither operation rewrites type-owned tree structure.
8. Recursive callable availability is planned by the canonical function SCC
   schedule. Only a genuinely recursive SCC receives provisional monomorphic
   bindings; its members infer, unify, remove the provisional entries, and
   generalize as a unit. Acyclic generic helpers stay polymorphic, bare
   acyclic forward calls retain textual semantics, and no diagnostic is
   erased after it has been reported. Top-level eager values are inferred in
   source order at both checker ingresses; serialized body-type metadata does
   not make a later value visible. An explicitly typed self-reference receives
   its external-input type only while its own declaration is checked; bare
   self-reference remains an eager cycle.

## Why the default is the bug, not the instance

The three-sighting pattern, checker edition:

| site | on unrecognized/malformed input | diagnostic pushed? |
|---|---|---|
| `infer_expr` unknown-tag wildcard | returns `Type::Error` | **no** ([#709]) |
| `.dp` arity guards (`(def {} orphan)`, `(cast {} ...)`) | return `Type::Error` | **no** ([#710]) |
| the `let`/`fn`/`app`/`if` malformed forms | rejected BEFORE the checker (Deep parser arity validation) | n/a - CORRECTED 2026-07 (PR #757): parser-screened, not checker templates; the checker guards at those sites are equally silent |

Correction (Phase 0 execution, PR #757): the original table held the
`let`/`fn`/`app`/`if` guards up as the correct in-checker pattern; they
are actually screened by the Deep PARSER's arity validation, and the
checker-side guards there are silent like the rest. Phase 1's
`MalformedForm` sweep therefore has no in-checker template at those
sites - the template is the push-then-return idiom of the reporting
arms ([#710]'s extension comment carries the full 15-form list).

Exactly the [#703] shape: the correct response exists in the same file and
the neighboring arm does not use it. A fix that adds the `handle-effect`
case without changing the default fixes instance one of an open-ended
series; [#709]'s own point 2 says this, and this document is its execution
plan.

## Non-goals

- **Not** the lowering-side effect catch-all (`lower_handle_effect`'s
  `_ => lower body, drop handler`) - that is a value substitution, census
  row 9 of `loud_unsupported.md` ([#730] Phase 1/2 delivers `EffectKind`;
  §I1 here consumes it for the checker case).
- **Not** [#721] (eval cannot ingest the canonical Deep of a nullary
  fn) - an eval-lane ingestion bug found while probing this class's
  controls; it involves no checker code. Independent fix.
- **Not** a semantics change for `with seed` / `with device` beyond
  checking them: the effect system's design (what `random`/`resource`
  MEAN, whether seeding reproduces cross-lane) stays as specified today;
  this plan only makes the checker see the bodies.
- **Not** score-model reform. How the 0-to-1 score is computed is
  untouched; this plan guarantees only that a silent exemption can never
  again masquerade as a 1.0.

## Vocabulary

- **Silent `Type::Error`** - a `Type::Error` produced without a
  corresponding `CheckError` in the result vector. The bug, as a noun.
- **Cascade suppression** - the legitimate use of `Type::Error`:
  descendants of an already-reported error unify freely so one mistake
  does not spray dozens of diagnostics. Preserved by this plan; the
  witness token (§C3) distinguishes it from the silent kind by
  construction.
- **Dispatch chokepoint** - a function that branches on a Deep tag to
  decide behavior (`infer_expr`, `lower_expr`, the `.dp` structural
  validators, the printers).
- **Canary** - the wrapper-battery test
  (`issue_709_handle_effect_and_dp_roundtrip.rs::wrapper_constructs_catch_the_masked_error`):
  eleven constructs wrapping one ill-typed body, each required to score
  below 1. If a new hole opens, this goes red.

---

# Part I - the normative contracts (§C1-§C4)

## C1. The totality contract

Normative, for every node the checker visits:

1. **Every Deep tag in the 62-tag vocabulary has an explicit checker
   disposition**: a real inference case, or an explicit rejection with a
   diagnostic. "Fell through the wildcard" stops being a disposition.
2. **Unknown-at-dispatch is a loud error.** The parser already enforces
   the closed vocabulary, so an unknown tag reaching `infer_expr` means a
   version skew or a bug - the response is a pushed
   `CheckError { kind: UnknownForm, ... }` naming the tag, never a silent
   exemption. (After Phase 3 this arm becomes unreachable-by-types for
   in-vocabulary consumers and remains as the guard for raw string entry
   points.)
3. **`Type::Error` implies a reported error.** At the end of any check,
   if the error vector is empty, NO expression in the typed result carries
   `Type::Error`. This is the executable invariant (§C4.1) that both [#709]
   and [#710] violate today, and it is the class-level acceptance test:
   whatever future code does, it cannot hold a silent exemption without
   tripping this.
4. **Structural malformation is a checker error too**: the `.dp` arity
   guards that today return silent `Type::Error` (the [#710] half) push
   diagnostics like their correctly-rejecting siblings (`let`/`fn`/`app`/
   `if` guards). Runtime-catches-it-later (today's saving grace) is not a
   disposition.
5. **`handle-effect` gets a real case** (the instance fix): check the
   handler expression against its effect kind's signature (`random`: an
   int64-SUFFIXED integer literal seed - `42i64` per spec/02 §P10a; an
   unsuffixed literal is a type error naming the required suffix -
   explicit over implicit, the width is visible in the source;
   `resource`: a string-LITERAL device, literal-ness checked, name
   vocabulary not validated - the kinds come from `EffectKind`, §I1;
   non-literal arguments are rejected citing §P5's shipped constraint
   and [#735] - Phase 1 checks FORM, [#735] authors meaning), check the
   body in the enclosing context, and
   return the BODY's type so the enclosing `def` signature is enforced.
   The case's typing SHAPE has a formal target (Jeff's 2026-07 note on
   [#709]): mirror LaCaDiLE's T-Handle rule (mechanized; the basis of
   its Theorem 3) - type the body under the handled effect, discharge
   the handled label from the residual effect row, enforce the declared
   type, never a silent `Type::Error`. T-Handle governs the typing
   shape; the literal-form rules above are ours (LaCaDiLE does not model
   seed values).
   The three executed escalations become impossible: an int64 body in an
   `-> f32` def is a type error; the tensor variant is a type error; both
   are caught before any backend sees them.

## C2. The diagnostic shape

Follows `loud_unsupported.md` §C2's format family for consistency, with
checker-appropriate kinds:

- `CheckError { kind: UnknownForm }` - a tag with no disposition
  (§C1.2). Message names the tag and the 62-tag vocabulary source.
- `CheckError { kind: MalformedForm }` - arity/shape guards (§C1.4).
  Message names the tag, expected shape, and found shape (the parser's
  existing `expected valid Deep tag, found unknown tag ...` message is the
  calibration example - it already names the vocabulary and the byte
  offset).
- `handle-effect` type errors reuse the existing `TypeMismatch` kinds -
  no new vocabulary needed; the novelty is that they fire at all.

All surface through the existing `chelis check` JSON error list (score
drops below 1 per the existing scoring), and through `build`/`eval` as
`error:` + nonzero exit, since both run the checker.

## C3. The witness token: silent `Type::Error` becomes unconstructible

The type-state ratchet, this plan's structural core:

```rust
/// Zero-sized witness that an error REACHED THE ERROR VECTOR. The only
/// constructors live in the diagnostics module:
///   report(sink, check_error)   -> Type   // appends, then mints
///   propagate(&ErrorWitness)    -> Type   // copies an existing witness
/// Field is private: no other module can mint one.
pub struct ErrorWitness(());

pub enum Type {
    // ... existing variants ...
    Error(ErrorWitness),   // was: Error
}
```

- `report` is the ONLY path that turns a fresh problem into `Type::Error`,
  and it pushes the diagnostic in the same expression; the two operations
  cannot diverge.
- `propagate` preserves cascade suppression exactly as today: a node whose
  child is `Type::Error(w)` may type itself `Type::Error(propagate(w))`
  without re-reporting.
- The compiler enumerates every current construction site during the
  migration (that is the point); each becomes `report(...)` (gaining its
  missing diagnostic - the [#710] sweep happens *here*, mechanically) or
  `propagate(...)` (documented cascade).
- Unification and equality treat `Type::Error(_)` exactly as before; the
  token carries no data and costs nothing.

The sink is part of the witness contract, not incidental plumbing. Fresh
witness minting accepts only the checker-owned `DiagnosticSink`; the
`DiagnosticOutput for Vec<CheckError>` compatibility used by internal probes
is `#[cfg(test)]`. `session::run_result` observes the same vector and rejects
an otherwise-`Ok` value whenever a diagnostic was appended. This closes both
ways the old invariant could be bypassed: minting into a throwaway vector and
returning success while authoritative errors existed.

Serialization note: the typecheck cache serializes `Type`, so serde is the one
accepted non-constructor witness mint. Production writers only receive
successful `TypeEnv`/`CheckedProgram` values: non-empty checker errors prevent
context construction, and the totality invariant forbids `Type::Error` in a
successful result. Cache envelopes verify format/build identity and byte
integrity, but deserialization does not rerun semantic checking; cache bytes
are a trusted internal artifact. This boundary is documented in the witness,
type-context, and compiler-api cache module docs.

### C3.1 Deep type/dimension resolution boundary (chelis#756)

`ErrorWitness` also governs conversion from Deep `t-*`/`d-*` syntax into the
internal `Type`/`Dim` representation. There is one recursive resolver, shared
by inference and ADT/alias declaration collection. Its private successful
value (`ResolvedDeepType`) cannot be forged by callers, and its public-to-the-
crate boundary is `Result<ResolvedDeepType, ErrorWitness>`:

- a fresh resolution failure pushes exactly one located diagnostic and returns
  its witness;
- recursive parents compose with `?`, never substitute a fresh variable,
  wildcard, dropped dimension, or unchecked nominal type;
- callers that require poison for ordinary checker cascade suppression convert
  the returned witness explicitly with `propagate`;
- cast targets use the same boundary before semantic classification. The bare
  primitive compatibility spelling and canonical `t-prim` have parity for all
  nine active scalar targets, while canonical metadata/child arity is still
  validated exactly and no extra child can be ignored;
- an explicit resolution context carries the use site, binder mode, known
  nominal headers/arity, and variable generator. Its `TypeResolutionEnv` is a
  serde-skipped runtime field in the per-check ADT-registry clone, separate
  from the validated definition/alias maps;
- binder modes are closed input, explicit `deftype`/`typealias` parameters,
  implicit-generic `defsig` parameters, and trusted compiler-generated
  metadata. Only actual binders or explicitly legal inference holes mint
  type/dimension/rank variables;
- declaration headers are precollected before bodies, preserving legal self
  and forward ADT/alias references while rejecting unknown names and wrong
  arities before a context can be cached. That explicit header environment is
  carried for the whole check unit (including body annotations), not rebuilt
  mid-check from the subset of bodies that registered successfully. A failed
  declaration therefore owns its one resolution diagnostic without downstream
  unknown-nominal spray. It never enters the validated maps, the non-empty
  error vector prevents context construction, and serde skips the provisional
  environment. A later stacked or decoded check reconstructs visibility from
  validated definitions plus its own precollected headers.

The exact Deep grammar and binder rules are normative in spec/03 §2.5.1/§2.6.

## C4. The invariants and the enum

1. **The totality invariant** (§C1.3), executable: the shared
   `finalize_checked_program` boundary validates every fresh inference result.
   The narrow effects-only transformation reruns the same annotated/signature
   totality validation after proving that no type-owned structure changed;
   arbitrary public reconstruction does not exist. The finalizer inspects the
   authoritative annotated runtime tree and its required owner stamps, then
   uses both incoming and inferred signature metadata as a structural
   backstop. (`CheckedProgram::compose` combines already-successful checked
   halves.) Combined with the session veto, `errors.is_empty() => the checked
   result contains no silent error or missing owner stamp`.
   Violation is itself a pushed internal error (never a panic - the
   checker is reachable-input territory, `loud_unsupported.md` §C1.3
   applies). This invariant is the tripwire that outlives everyone's
   memory of this document; with §C3 in place it should be structurally
   impossible to trip, and it stays on precisely to verify that claim.

   **Scope note (2026-07-24): both halves of this invariant are tag-keyed
   and therefore quantify over the checked result, not over the source
   program.** A node with no recognized tag is exempt from the stamp
   requirement (`infer.rs`, `requires_stamp = tag.is_some_and(...)`), from
   child ownership classification (the untagged arm of the
   `child_stamp_role` match, which recurses without registering), and from
   owner registration (`register_annotation_owners`'s `if let Some(tag)`) -
   so a node the checker never visits satisfies the invariant vacuously.
   [#858] is the instance that made this concrete; [#874] tracks the class.
   Phase 3 closes the known *top-level* path to the exemption - a loud
   `UnknownForm` where `infer_top_level` used to skip silently, with
   rejection parity for that input class on both `.dp` validator surfaces -
   without removing the exemption itself; nested bare lists stay legal by
   design (empty guards, `loc` metadata values), so the untagged arm
   remains reachable below the top level. The complementary obligation -
   every runtime node in the *parsed program* is stamped, dispositioned, or
   diagnosed - is not part of §C4.1. `DeepTag` exhaustiveness (Phase 3)
   does not close it either: an untagged list has no tag to be exhaustive
   over.
2. **`DeepTag` at the chokepoints is decode-once** (Phase 3, MANDATED at
   the 2026-07-24 rework per [#730] §C4.2's doctrine: raw strings exist
   only at serialization boundaries; decode once, then exhaust). The
   parser stamps every vocabulary tag as `Atom::Tag(DeepTag)` at element
   0 (including inside metadata map values, where canonical type syntax
   lives), so after parsing the tag string DOES NOT EXIST in the
   in-memory tree and no consumer can dispatch on it. Programmatic
   producers (desugar, macros, the checker's own type-metadata builder,
   the runtime transform synthesizers) construct `Atom::Tag` through
   typed constructors. `List::tag()` is the only dispatch accessor;
   `List::unknown_tag_symbol()` names non-vocabulary heads for
   diagnostics and returns `None` for stamped nodes by construction.
   `infer_expr`, `lower_expr`, the `.dp` structural validators, and every
   migrated consumer match the enum **exhaustively - no `_` arm over the
   vocabulary**; tag 63 stops the build at every consumer that has not
   chosen a disposition. Printers and serializers regenerate the string
   via `as_str()` at the boundary only; the serialized `.dp` form and the
   wire schemas are unchanged. The permanent decode-once invariant
   (`chelis_deep::validate::find_raw_vocabulary_tag`: no parsed or
   desugared tree carries a vocabulary string at element 0) is a standing
   suite member on both the parser and desugar sides. The recorded
   raw-string entry points that legitimately keep symbol heads:
   lenient-parsed unknown tags (spec/03 §8.3 fitness mode), the
   compiler-internal pre-expansion tags (`defmacro`/`macro-invoke`), the
   host-lane internal spellings (`vmap-grad`, the list literal, legacy
   `ascribe`/`:`/`drop`), and the parser's own pre-stamp internals - each
   guarded by a §C1.2 loud arm or an explicit recorded check.

   **Successor acceptance under [#908]/[#1023].** The physical carrier may
   change from `Atom::Tag` to `Node::tag`, but the word "decode-once" keeps
   its full force. Every public/compiler `.dp` ingress must return and consume
   the stamped representation rather than validate and then reparse or
   normalize it away; validators and the raw-tag oracle must recurse through
   Node metadata and all enforcement-relevant children; `BareList` and
   `UnknownForm` require explicit positive/negative-tested dispositions; and
   `Expr::List`, `List`, `Atom::Tag`, and the bridge may be deleted only in the
   change set that proves no active path depends on them. Until all four are
   executable and green, the new representation is an in-progress migration,
   not a weaker replacement for Phase 3.
3. **The canary stays forever**: the wrapper battery is cheap, runs in the
   default suite, and is the behavioral proof the structural claims cash
   out. Every new wrapper construct SHALL add a positive and negative row in
   the same change set. Six legacy wrapper strings are
   parse-rejected Surf as written, so those rows score below 1 via the
   parser, not the checker; re-probed with corrected syntax, the
   ill-typed variants ARE checker-caught, so the coverage claim
   survives. Rows may be corrected to test what they claim, but never removed
   or weakened.
4. **The fitness-honesty corpus**: the permanent CI suite contains
   known-ill-typed programs - the wrapper battery plus every
   census-verified silent-hole repro ([#709]/[#710]/[#755]/[#756],
   [#833]'s declaration-only `total_nodes == 0` hole, and future
   finds) - asserting every member scores strictly below 1.0;
   any member scoring 1.0 fails the build. This is the continuous,
   corpus-level enforcement of §C1.3, standing even after §C3 makes
   violations unconstructible. Margin thresholds (e.g. < 0.9) are
   calibration, decided with open question 4's severity weights.
   Delivered at Phase 1; permanent thereafter.

---

# Part II - process rules at every boundary

## B1. Freeze points

| contract | frozen at end of | may change after only by |
|---|---|---|
| §C2 diagnostic kinds + message shapes | Phase 1 | this doc + updated canary/corpus in the same PR |
| §C3 `ErrorWitness` API (`report`/`propagate` signatures) | Phase 2 | this doc |
| §C4.1 invariant (on-by-default) | Phase 2 | never off; scope changes via this doc |
| `DeepTag` variant set = the 62-tag vocabulary | Phase 3 | spec/03 change + enum + all consumers, one change set |

## B2. Invariants that hold across every boundary

1. **The canary never weakens.** No phase may remove or loosen a
   wrapper-battery row; language additions extend it in the same PR.
2. **Red-to-green only by un-ignoring** (the shared audit rule): the
   `#[ignore]`d [#709]/[#710] tests flip by deleting the attribute, never by
   editing assertions.
3. **Cascade behavior is a control.** Existing multi-error programs must
   not spray new duplicate diagnostics after §C3; a before/after
   diagnostic-count corpus lands with Phase 2 (negative parity for the
   witness migration).
4. **No new checker case lands without both polarities**: the positive
   test (well-typed use accepted) and the negative (ill-typed use caught,
   with the right kind) - the repo's negative-test-parity rule applied to
   `handle-effect` and every future tag.
5. **Discoveries fork.** New silent-`Type::Error` sites found during the
   §C3 migration are diagnostic-gaining conversions IN the migration
   (that is its purpose); new *classes* of silent exemption (if any) are
   filed and linked to the tracking issue.

## B3. How to pick up a phase

1. Read Part I, your phase, and the previous phase's frozen-at-exit list.
2. Run the canary and your phase's oracle suite first; record the red set
   in the PR.
3. The probe corpus (`docs/investigations/probes/`, `checker_holes.py`)
   regenerates the wrapper battery's evidence from scratch if you need to
   re-derive current behavior.
4. Gate with `scripts/gate.py --local`; the workspace oracle is CI's
   macOS Smoke.

---

# Part III - the phases

## Phase 0 - census + red invariant (small, land-first)

**You inherit:** the wrapper battery canary and the [#709]/[#710]
`#[ignore]`d tests (already committed on the PR [#696] branch).

**You deliver:**

1. **The `Type::Error` construction census**: every site in
   `chelis-types` that produces `Type::Error`, classified
   report-adjacent / cascade / silent, committed as a table in the
   tracking issue. (Mechanical: grep + read; the audit pattern says
   verify the "silent" classification by driving each through `chelis
   check` where reachable.)
2. **The totality invariant as a red harness**: §C4.1 implemented as a
   test-only validation over check results, `#[ignore]`d red on the
   known holes (`with seed`, `with device`, the two [#710] forms), green
   on a clean-program corpus.

**Frozen at your exit:** the census baseline; the invariant's definition.

**Explicitly not yours:** fixing anything; touching `Type`.

**Oracle:** the invariant harness red on exactly the four known forms and
green on the control corpus; every census row's classification named to a
probe or test.

## Phase 1 - the loud default and the known holes (the instance fix; small)

**You inherit:** the census and the red invariant harness.

**You deliver:**

1. `infer_expr`'s wildcard pushes `UnknownForm` (§C1.2).
2. **The `handle-effect` case** (§C1.5), consuming `EffectKind` if [#730]
   Phase 2 has landed, else matching the two known kind strings with a
   loud else (and a note to migrate; §I1).
3. **The [#710] guard sweep**: the silent arity guards push
   `MalformedForm`; the template is the push-then-return reporting idiom
   (the formerly-cited `let`/`fn`/`app`/`if` "siblings" are
   parser-screened, not templates - see the corrected table above).
4. Diagnostic-shape conformance per §C2.
5. **The fitness-honesty corpus** (§C4.4), seeded from the wrapper
   battery (fixtures corrected per §C4.3's caveat), the four holes, and
   the census-verified silent sites ([#755]/[#756] and the [#710]
   extension).

**Frozen at your exit:** §C2 kinds and shapes.

**Explicitly not yours:** the `Type` variant change (Phase 2); `DeepTag`
(Phase 3); any lowering-side behavior.

**Oracle:** `issue_709_handle_effect_and_dp_roundtrip.rs`'s five
`#[ignore]`d tests green and un-ignored (with-seed/with-device bodies
checked; the masked-return-type program rejected at build with a type
diagnostic; the bogus-effect `.dp` rejected loudly - the last one via the
checker OR via [#730]'s lowering raise, whichever lands first, per §I1);
the [#710] false-green `.dp` tests green; the canary green; the Phase 0
invariant harness green on the four former holes.

## Phase 2 - the witness token (silent exemption becomes unconstructible)

**You inherit:** a checker with no KNOWN silent sites (Phase 1) and the
census enumerating every construction site.

**You deliver:**

1. `Type::Error(ErrorWitness)` and the `report`/`propagate` constructors
   (§C3); every construction site migrated. The source census recomputed
   2026-07-22 is 269 `report(...)` call sites in `infer.rs`, 2
   resolver-boundary `report_witness(...)` call sites in `deep_type.rs`, 14
   direct production `propagate(...)` call sites, and 8 aggregate
   `propagate_if_error(...)` call sites. The direct-propagate number excludes
   the function definition and one rustdoc mention. The reproducible census
   is `rg -n '\breport\(' crates/chelis-types/src/infer.rs`,
   `rg -n '\breport_witness\(' crates/chelis-types/src/deep_type.rs`, and
   `rg -n '\bpropagate(_if_error)?\(' crates/chelis-types/src`.
2. **The totality invariant promoted** from test harness to an
   on-by-default post-check validation (§C4.1).
3. The cascade-behavior corpus (B2.3): diagnostic counts before/after on
   a multi-error program set, asserting no spray regression.
4. Typecheck-cache boundary note executed: cache (de)serialization keeps
   working; the entry point documented as the one non-constructor mint
   (open question 2 resolved in this PR).
5. The chelis#756 converter family retired in favor of §C3.1's centralized
   witnessed resolver; malformed/unknown types and dimensions are rejected
   exactly once before declarations or cached contexts become successful.
6. The adversarial Phase 2 pass exposed chelis#813's pre-existing one-argument
   `conv2d` validator panic. The fix and complete arity 0-through-6 regression
   matrix land here, so the Phase 2 PR may truthfully close #813.
7. Diagnostic ownership hardened around the one append-only session sink,
   the verified effects-only checked-program transformation, explicit lexical
   binder scope, authoritative inference epochs, annotated-tree finalization,
   and SCC-scoped recursive prebinding. The structural source tests lock the
   absence of ambient binders, throwaway production sinks, public raw
   reconstruction, error deletion, annotation re-inference, and post-report
   recursive-cycle suppression.
   The 2026-07-22 sink-construction census finds five textual constructions in
   `session.rs`: three production owners (`infer_program`, `run_result`, and
   `infer_ir_program`) that return their error vector, plus two `#[cfg(test)]`
   harness owners. The source-contract test parses this ownership boundary;
   it does not freeze a stale raw count as the invariant.

**Frozen at your exit:** §C3 API; §C4.1 always-on.

**Explicitly not yours:** `DeepTag`; new checker cases beyond what the
migration forces.

**Authoritative Phase 2 oracle:**

```sh
cargo nextest run --profile ci --no-fail-fast \
  -p chelis-types -p chelis-effects -p chelis-surf -p chelis-cli \
  -p chelis-ir -p chelis-compiler-api -p chelis-e2e -p chelis-prove
```

The invariant and source contracts run inside that command, including the
fitness/cascade/handler/owner-stamp controls. The compile-fail witness doctests
and `scripts/gate.py --local` are required supporting evidence, but neither
replaces this oracle.

The witness doctests are `crates/chelis-types/src/errors.rs`'s eight
` ```compile_fail ` blocks. `cargo nextest` does not execute doctests, so
until chelis#875 they ran in no continuous job and this paragraph claimed
supporting evidence the repo was not producing. They are now driven by the
`cargo test -p chelis-types -p chelis-compiler-api --doc` stage in
`scripts/gate.py`, which is in
both the `--local` subset and CI's `lint-rust` worker.

Scope of the guarantee, so this section does not read stronger than the
mechanism: the witness makes a `Type::Error` **without a diagnostic**
unconstructible. It says nothing about an ordinary type standing in as a
verdict for an unrecognized construct, which is the same class defect one
substitution away (chelis#873: `infer_atom` returned `Type::Unit` for a form
it could not type, and `chelis check` scored those programs 1.0). No
constructor gate can close that variant, because the types it borrows are
legitimately constructible. It is held behaviorally, by the score-surface
corpus in `crates/chelis-cli/tests/issue_731_fitness_honesty_corpus.rs`.

## Phase 3 - `DeepTag` at the chokepoints

**You inherit:** a checker that cannot silently exempt (Phase 2) - this
phase is about the NEXT tag, not the current ones.

**Why this phase is not only about tag 63.** Two mechanisms guard two
populations, at two different times. The parser enforces the closed
vocabulary at *parse time, over user input*
(`chelis-deep/src/validate.rs`, promoted to a hard error by
`parse_str_strict`), so a tag that is *unknown* never reaches a consumer -
[#710]'s probe confirmed the `infer_expr` catch-all is dead for parsed
input. `DeepTag` exhaustiveness guards the complement at *compile time,
over this compiler's own source*: a tag that IS in the vocabulary but has
no decided disposition at a given consumer. No input check can find that,
because nothing about the input is wrong - the hole is in our dispatch,
and a validator cannot validate its own consumers. That population is not
hypothetical: Phase 3 found **31 of the 62 tags with no expression-position
case in `infer_expr`**, every one previously falling through the
unknown-tag wildcard and being reported with a message claiming it was
outside the vocabulary it is listed in. `block` is the clearest case -
spec/03 §2.3 presents it as an ordinary expression, and the checker had no
case for it ([#859]). So the phase closes two things: the future supply
(tag 63) and the present backlog (known tags, no disposition), and the
second is what the wildcard was actively mis-describing.

**You deliver:**

1. `enum DeepTag` (62 variants, `parse`/`as_str`, produced by the Deep
   parser alongside the validated string).
2. `infer_expr`, `lower_expr`, and the `.dp` structural validators
   dispatch on it **exhaustively**; the string-keyed `tag(list) ==
   Some("...")` chains at those three chokepoints retire. (Printers and
   producers may migrate opportunistically; they are not chokepoints -
   they cannot exempt or substitute.)
3. The §C1.2 loud arm is retained only at raw-string entry boundaries.
   `DeepTag` exhaustiveness is proved by its typed consumers and mutation
   oracle; it does not depend on [#730]'s extracted source lint.

**Delivered (2026-07-23).** `chelis_deep::tag::DeepTag` (62 variants,
`parse`/`as_str`/`ALL`, unit tests pinning the set to spec/03 §2.10 by an
independent in-test spelling); `chelis_deep::validate::VALID_TAGS` derived
from `DeepTag::ALL`; exhaustive no-`_` dispatch at `infer_expr`
(chelis-types), `lower_list` (chelis-ir, `lower_expr`'s tag dispatch), and
both `.dp` structural validators (`chelis_deep::validate::validate_tag_shape`
and `chelis-validate`'s pest-side `validate_tag_shape`, whose duplicated
`VALID_TAGS` copy and drift test retire). Execution notes:

- The 31 in-vocabulary tags with no expression-position inference case get
  an explicit loud `UnknownForm` disposition in `infer_expr` whose message
  names the tag and the real reason: "no expression-position checker
  disposition (helper/pattern/type syntax outside its owning form, or an
  expression form with no implemented case; ...)". All but one of the 31
  are declaration internals, patterns, type/dimension syntax,
  metaprogramming forms, or structural helpers whose checking belongs to an
  owning enclosing form; the exception is `block`, a spec/03 §2.3
  expression form with no implemented checker case (the spec-vs-checker
  decision is [#859], surfaced by the PR #855 round-1 red team). Before
  Phase 3 all 31 fell through the unknown-tag wildcard, whose message
  wrongly claimed they were outside the vocabulary; this message-shape
  correction is the one §C2 delta of the phase, recorded here per B1 with
  the exact wording above (kind, severity, and the raw-string arm's
  message are unchanged).
  `crates/chelis-types/tests/issue_731_deeptag_dispatch.rs` is the
  executable record of the 31/31 dispatch split.
- `lower_list` is behavior-preserving: the formerly-fallthrough tags keep
  the census-row-16 sequence-lowering disposition ([#730] §C1.4
  keep-with-comment), now spelled per-tag, and the raw-string `None` arm
  shares the same helper.
- The §C1.2 loud arms survive exactly at the two raw-string boundaries:
  `DeepTag::parse` returning `None` at a chokepoint (programmatic Deep
  that never crossed the parser) and the validators' unknown-tag paths.

**Reworked (2026-07-24, maintainer directive).** The derive-on-demand
delivery above is superseded by decode-once (§C4.2's mandated form): the
parser's `stamp_tags` pass converts every vocabulary tag to `Atom::Tag`
(metadata map values included), typed constructors cover the desugarer,
macro expander, checker type-metadata builder, and runtime transform
synthesizers, and every tag-reading consumer across chelis-deep, -surf,
-macros, -types, -pred, -ir, -effects, -compiler-api, -prove, -cli,
-lint, -validate, and -e2e dispatches on the decoded enum (printers and
the decompiler included; open question 3's deferral is REVERSED). The
permanent decode-once invariant tests stand on the parser and desugar
sides. In the same rework, [#858] is fixed (the top-level untagged-list
silent skip is loud, the fitness clean path reports the checked
counters, both `.dp` validator surfaces reject the input class, and the
repro joined the §C4.4 corpus with both polarities) and [#859] is fixed
(`block` is implemented end-to-end - checker, DAG lowering, host eval,
dual eval, and C host emission - with both polarities and a
check/eval/build pipeline test; the expression dispatch split is 32/30).

**Carrier transition (2026-08-01).** [#908] has landed `RawExpr`, validated
`Node`, role-directed stamping, a consumer bridge, and several stamped CLI
entry paths. That is useful progress, but the successor acceptance in §C4.2
is not yet discharged: legacy List normalization and public carriers remain,
validator/oracle coverage is incomplete, and some paths validate then consume
a reparsed legacy tree. This paragraph records state; [#908] owns completing
the structural cut.

**Construction gate (2026-08-03).** A partial [#731] Phase 3 hardening now
closes the *construction* half of successor acceptance. Public `Node`
construction rejects a raw closed-vocabulary tag recursively, in metadata
values as well as children, so a legacy form cannot hide below a newly
stamped carrier. RuntimeExpr slots reject the transitional `Atom::Tag`
carrier alongside bare names. Metadata and child replacement are
validate-before-commit and transactional: a rejected candidate leaves the
original node untouched. The raw mutable child and metadata borrows
(`children_slice_mut`, `children_vec_mut`, `meta_mut`) are gone, and the
path/module rewrite helpers rebuild stamped Nodes inside-out, so no caller
can reopen the raw-vocabulary or arity domain after construction. Mutable
path resolution into a stamped Node is now a structured error rather than a
borrow. Checker-totality mutations that deliberately bypass the constructor
do so through an explicitly test-only raw-to-legacy adapter; ordinary
positive coverage continues through the production stamped parser. The
WI-1 depth-oracle harness tears down its owned deep fixture on a fresh
production-sized grown segment, so a green test proves the typed
stack-budget diagnostic rather than accepting an abort.

**Authoring scope (2026-08-03).** The read-only authoring surfaces
(outline, references, call graph) no longer deep-copy their input into
`Expr::List`. They read either carrier in place through a borrowed node
view, and their scope model is repaired: a `let` binds pair by pair, so a
later right-hand side sees an earlier binder while an earlier one still
sees the top level, and an arm's pattern binders (including the recursive
`pat-as`/`pat-tuple` shapes) cover that arm's guard and body. A binder that
shadows a top-level name removes its references, its call-graph edges, and
its renames. `crates/chelis-compiler-api/tests/authoring_scope.rs` locks
each direction, with a positive control that an unshadowed reference is
still found and a tripwire that the read-only paths call no normalization
helper.

**Ingress (2026-08-24, [#1088]).** The *ingress* half is discharged for
the compiler API. Every public Deep text boundary in `chelis-compiler-api`
consumes the role-stamped carrier, and each boundary names the role its
field actually occupies: `parse_and_stamp_file` for a `.dp` program (the
generic `parse`/`check`/`decompile`/pipeline door and every authoring
`module` field), `parse_and_stamp` for a declaration bundle,
`parse_and_stamp_runtime_exprs` for a replacement body, and
`parse_and_stamp_tagged` for a field whose contract names one tag. The
weaker `parse_str`/`parse_str_strict` doors, which stamped every top-level
form as a bare/syntax position with no declaration requirement, are gone
from the crate, and the named non-compiler-API stragglers
(`chelis-validate::validate_deep`, the `opaque-domain-construction` lint
rule, `chelis-e2e`'s snippet checker, and the `chelis-cli` style-gate
fallback) moved with them. Two consequences are worth recording. A stamp
rejection now carries the offending form's span instead of the
whole-input offset a re-wrapped parse error reported. And
`validate --deep` and `check` accept one Deep language: the AST leg of
`validate_deep` used to skip its module-identity forgery checks silently
whenever its own parse failed.

The top-level rule this enforces is decided by the numbered spec:
`spec/03-deep-syntax.md` §7.1 [03-PROG-1] enumerates the admissible
top-level forms and [03-PROG-2] states the rejection contract.

The authoritative oracle is [#908]'s
`scripts/unrepresentable_domain_oracle.py`, run by `scripts/gate.py`'s
`integration` stage (hosted CI's `workspace-tests-shard` matrix, on every
non-docs-only pull request) and by its `--local` pre-push subset. It executes
`crates/chelis-compiler-api/tests/phase3_stamped_ingress.rs` as one of its
obligations; that suite is evidence, not a second oracle. The suite's
parity table drives every module-text door over one shared accept/reject
corpus, so the two strengths cannot silently reappear, and a structural
guard over the workspace's production sources fails the build if any of
them reaches a weaker Deep ingress again. The guard resolves `use`
imports, renames, module aliases, and glob imports rather than matching
source lines, because a line-substring guard is blind to exactly the
alias a regression would introduce.

Still open at §C4.2: mutating authoring normalization remains active, and
the legacy `List` variant and consumer bridge remain active ([#1029], in
turn blocked on [#1082]). Phase 3 MUST NOT be called successor-accepted
until those remaining deletion clauses are executable and green.

**Frozen at your exit:** the variant set = the vocabulary, changing only
per B1's one-change-set rule.

**Explicitly not yours:** adding tag 63 or any vocabulary change. (Giving
an already-in-vocabulary tag a real checker case - as opposed to an
explicit loud disposition - was scoped out here and folded into the phase's
own change set instead; see [#859].)

**Oracle:** the build itself - the mutation test: adding a scratch
variant to `DeepTag` must produce compile errors in `infer.rs` AND
`lower.rs` AND the validators (verified once in the PR, recorded, then
the scratch variant deleted); the canary and full matrix stay green. A [#908]
successor reruns the same mutation against stamped ingress, the Node validator,
the raw-tag walker, and every new typed disposition table. Moving the carrier
may change the compile-error site list, but may not reduce the class of
undecided consumers that fail closed.

## Post-Phase-3 work items

Named deliverables opened after Phase 3 shipped. They are not a phase: they
add no freeze point, inherit no phase's exit state, and change no contract in
Part I. Each one exists because an instance was found whose local fix would
leave the mechanism that produced it in place, and the class intent (§C1:
every construct covered or loudly rejected) is a statement about mechanisms,
not about the instances anyone has happened to trip over. Each therefore
names the mechanism that holds the class, and carries its own oracle in the
phase idiom.

They are numbered PP-N rather than W-N to keep them distinct from the
verification stack's WI-N work items (`verification_stack_dependency_map.md`),
one of which - WI-1's depth-oracle harness - this document already cites
above.

### PP1. The function-parameter typing channel ([#780], [#847], with [#783])

**Opened 2026-08-04; decided below.** Two filed defects were treated as the
two faces of one semantics question: how the type of a lambda-bound or
function-valued parameter binds, and which checks re-run once it is bound.

- [#780] is the silent-acceptance face. An operand flows through an
  UNANNOTATED lambda parameter, that parameter stays a bare type var, the
  shape-computed builtin's Var arm returns the unresolved result var instead
  of deferring and re-checking once the application binds it, and the
  `[4,4]`-vs-`[9,9]` conflict is never revisited. `chelis check` is clean on
  a genuinely ill-shaped program.
- [#847] is the over-unification face on the same channel. Differentiating a
  scalar projection through an arbitrary FUNCTION-VALUED model parameter
  unifies `n` and `m` - two independently declared, rigid dim parameters -
  so a legitimate generic wrapper is rejected. The concrete-dim variant
  checks; the same objective with the function parameter removed checks,
  evaluates, and C-builds. The trigger is the function parameter, not `grad`.

Both were sequenced behind Phases 1-2, and Phases 1, 2, and 3 all shipped
without deciding it. That is the reason this is one item and not two: two
local patches would encode two ad-hoc answers to one question - a
bind-later rule at the shape-checker site and a keep-rigid rule at the
differentiation site, neither written down anywhere a third site could read -
and each would be free to drift back into the other's failure. §C1 is not
satisfied by teaching one builtin to notice one unbound operand. The channel
needs a decided rule, and the fixes are then derived from it.

[#783]'s durable invariant belongs to the same channel and MAY ride this
item: the post-inference annotation writeback never replaces a concrete type
annotation with a degraded one (a Var-derived rank-0 `f32` default or an
Error-derived type). When re-inference is unresolved, the existing annotation
survives or the pass fails loudly - covered-or-rejected on the type-metadata
channel. It is the same unresolved-Var value one layer down, where the
consumers are the IR lowering's type readers, the host pipeline's `expr_type`,
and eval-root expansion; the conv2d ICE was that value arriving as plausible
metadata rather than as a diagnostic.

**Decision (2026-08-04).** `spec/04-type-system.md` [04-INF-1] is the
controlling rule. A shape-constrained lambda with an unknown parameter
constructor retains an obligation over the exact inference variables seen by
the ordinary operation checker, stays monomorphic, and binds at its first
application within the enclosing declaration. That application replays the
same checker function. A still-unbound obligation at that declaration's own
boundary is a type error requiring an outer-constructor parameter annotation;
a result annotation and a later top-level caller are not binding sites. An
ordinary lambda with no such obligation still generalizes. Readiness is
deliberately about an unknown outer type constructor (`Type::Var`), including
a wildcard/bare-variable signature slot and a projection-derived descendant,
not every free variable: a declared tensor with symbolic dimensions or
precision is already shape-checkable and remains polymorphic.

The implementation ledger is owned by `InferenceProduct`, not by `matmul`.
It covers the eleven existing shape-computed overrides (`matmul`; the seven
reductions; `expand`; `layer_norm`; `conv2d`) plus PP2's
`scatter_elements`, and stores the ordinary rule plus its original argument
expressions and types. Construction records unresolved type variables owned by
lambda parameters regardless of whether the surface omitted an annotation or
a signature supplied an informationless hole. Ownership follows the resolved
parameter structure, so tuple/record/function projections cannot sever it,
while an unrelated unresolved value governed by contextual inference is not
mistaken for a lambda awaiting its first application. This is the single replay path
for the class. `let` consults that ledger before generalization, applications
replay newly ready rows, and both program drivers reject residual rows before
successful finalization. The `ShapeComputed` disposition is also a fail-loud
construction guard: if any present or future shape-computed builtin reaches an
unbound operand without recording its ordinary rule in this ledger, the
application is rejected as an internal coverage failure rather than returning
an unchecked result variable.

[#783] rides PP1. Annotation writeback now uses one information-ordering
gate: a Var-, Error-, or partial-dimension-derived candidate cannot replace a
more informative existing type expression. Resolved owner metadata may still
refresh derived metadata, and a symbolic type may still be written when there
is no existing annotation to degrade. That distinction preserves legitimate
generic metadata while making the clobber class impossible at the write site.

[#847]'s exact `jacobian_row` reproducer already checked clean at this change
set's `origin/main` baseline, so no one-off `grad` or rigidity patch was
justified. It remains in PP1 as a permanent positive regression for
[04-INF-1]'s declared-symbolic side, paired with a genuinely equal-dim body
that still rejects. This is a verified migration miss/stale instance, not
evidence for a second typing policy.

**You deliver:**

1. **The decision, recorded once.** At minimum it covers: when an unannotated
   lambda or function-valued parameter's type is bound (at the abstraction, at
   the application, or by explicit deferral); what happens when nothing ever
   binds it - accepted as polymorphic, or rejected, and with which
   diagnostic; which checks re-run at binding, naming the shape-computed
   builtin overrides whose Var arm today returns an unresolved result var;
   and how declared rigidity survives a function-valued parameter, so that
   two independently declared rigid dim parameters reaching a differentiating
   combinator stay distinct while genuinely required equalities still unify.
2. **The authority for it.** If the rule is a language decision it amends
   `spec/04-type-system.md` first and this item cites the amended text
   (Numbered Specs Decide); if it is an inference-implementation rule under an
   existing spec/04 authority, it names that authority. A rule that lives only
   in this design doc is the drift shape the repo contract calls out.
3. **Both fixes, derived from the decision, in one change set** - not a patch
   at each site.
4. **An explicit in-or-out record for [#783].** Whoever lands this states
   whether [#783]'s writeback invariant rides here or stays standalone, and
   why. The condition is real - if the decision changes what an unresolved
   parameter type may BECOME, the writeback is downstream of the same value
   and landing it separately risks a second, contradicting rule - but a
   conditional left implicit in a plan is how a half gets dropped. Recording
   the choice costs a sentence; leaving it to be inferred is how [#780] and
   [#847] spent three phases attached to a note that scheduled nothing.

This entry's former open semantics question is answered by [04-INF-1]. The
delivery remains one class change: a site-only `matmul` retry or a separate
rigidity exception does not satisfy it.

**Oracle:**

- [#780]'s reproducer rejects with a shape mismatch naming `[4,4]` and
  `[9,9]`; its annotated-lambda variant (today's control) and a
  consistent-shape variant stay green.
- [#847]'s `jacobian_row` reproducer checks clean with `n` and `m` distinct;
  its concrete-dim variant stays green; and a negative control - a wrapper
  whose body genuinely does require the two dims equal - still rejects, so
  the fix is a correct binding rule and not a disabled rigidity check.
- Both reproducers become named regression tests, and [#780]'s ill-typed
  program joins the §C4.4 fitness-honesty corpus scoring strictly below 1.0.
- [#783] rides: a writeback whose owner type yields Var or Error over a
  node carrying a concrete annotation preserves that annotation byte for
  byte or fails loudly, plus the sweep of the writeback's other degrade paths
  (Error-derived, partial-dim) against the same invariant.
- §C4.3 applies unchanged: the unannotated-lambda and function-valued-parameter
  wrappers are wrapper constructs in the canary's sense, so each adds its
  positive and negative row in the same change set.

### PP2. The registered-builtin arm tripwire ([#1147])

**Opened 2026-08-04.** `scatter_elements` shipped as a registered builtin -
`generic_quadop("scatter_elements", ...)` in
`crates/chelis-types/src/builtins.rs` - with no inference arm in
`crates/chelis-types/src/infer/app_post.rs`. A string axis, an out-of-bounds
axis on a rank-2 operand, an f32 tensor as indices, and a string operand all
check clean at score 1.0, while the sibling `scatter` rejects every one. Found
by the PR #1145 review sweep, after Phase 3.

This is [#709]'s mechanism on a second keyed lookup, and it is why the item is
a tripwire rather than an arm. The generic-arity registration hands the call a
signature that unifies with whatever the caller supplies, so the checker
reports success on operands it never examined: the registration IS the
wildcard, and it is a wildcard nothing in Phases 1-3 quantifies over. Phase 3
made a Deep TAG with no disposition uncompilable; a registered BUILTIN with no
arm is the same silent exemption keyed by callable name, and the compiler has
no reason to complain about it.

**Relation to §C4.1's scope note and [#874].** §C4.1's coverage half is
tag-keyed, so it quantifies over the checked result and is satisfied
vacuously by a node with no tag to key on; [#874] tracks that face. PP2 covers
the complementary face, and neither subsumes the other. Here the node IS
tagged (`app`), IS visited, and IS stamped - §C4.1 holds - and the operands
are still unchecked, because the disposition the `app` arm reaches is a
generic signature. [#874] is about a node the checker never visits; PP2 is
about a node it visits and learns nothing from.

**You deliver:**

1. **The `scatter_elements` arm** on the `scatter`/`scatter_replace` model -
   operand, indices, updates, and axis checking - with both polarities. This
   is the one-line-shaped prerequisite, not the deliverable; per [#1147], an
   axis-dtype-only screen would misrepresent the builtin as checked and does
   not satisfy it.
2. **The tripwire.** Every name in the builtin registry (`BUILTIN_NAMES` and
   the consolidated `BUILTINS` table in `builtins.rs`) either reaches a
   name-keyed inference arm or carries an explicit, recorded generic-acceptance
   disposition: a named entry stating that the generic signature IS the
   intended check for that callable, with its reason. A registered builtin
   that is neither armed nor dispositioned is a red check, never a silent 1.0.
3. **The disposition lives beside the registration**, so the omission fails
   where the builtin is added rather than being discovered months later by a
   review sweep. `BuiltinDecl` is the existing single source of truth for
   per-builtin metadata and already carries the doctrine this needs -
   "Omitting any field is a compile error" - which is the [#730] Phase 2
   typed-vocabulary shape applied to the callable registry.

The adjacent cosmetic defect [#1147] records rides this change set: the shared
indices helper hardcodes `gather` in its message, so `scatter` and
`scatter_replace` reject with a diagnostic naming an op the program does not
call. A rejection that names the wrong op is loud but not actionable.

**Implementation.** `BuiltinDecl.inference` is a required field with no
default. A declaration is either `Checked(ShapeComputed|Specialized)` or
`GenericAccepted { reason }`. The checked route sets are bidirectionally
pinned: every checked declaration must name an exact route, and every route
must be owned by the exact checked declaration. Runtime dispatch additionally
requires an execution witness set inside the actual semantic checker family;
a checked application that falls through because an arm was removed is an
internal error rather than a generic-signature acceptance. `scatter_elements` is a
shape-computed route in its own `app_scatter` module and checks all four operands,
rank, indices dtype, update shape/precision, axis dtype/bounds, and non-axis
extent containment. The shared gather/scatter helper now receives the actual
operation name, removing the misleading `gather` diagnostic.

**Oracle:** the required `BuiltinDecl.inference` field is compile-fail tested,
the route tripwire must turn red when a checked declaration's exact route is
deleted, and the runtime observation guard must reject a checked application
that reaches no real semantic family. On 2026-08-04, deleting `scatter_elements` from
`SHAPE_COMPUTED_INFERENCE_BUILTINS` and running
`cargo nextest run -p chelis-types
every_checked_builtin_has_an_exact_inference_route` failed with
`builtin 'scatter_elements' declares ShapeComputed but no exact route owns it`;
restoring the route made the same command green. The four [#1147] programs
reject with the diagnostics their `scatter` equivalents produce, each joining
the §C4.4 fitness-honesty corpus; a well-typed `scatter_elements` call is the
positive control. The synthetic `probe_unarmed_op` unit control pins the
fail-loud observation diagnostic independently of the two route manifests.

### PP3. The name-keyed binding-identity channel ([#1209], [#1211], with [#1212])

**Opened 2026-08-21; decided below.** PR [#1208]'s adversarial review filed
three issues that read as separate defects and are one question: what is the
identity of a binding? The linearity checker kept every fact it knows —
alias links, consumption marks, destructured-component identity — in maps
keyed by the variable's name string, and `resolve_alias_chain` walked names
through whatever binding currently owned each one. [#1209] is the direct
consequence (an alias crossing a binding generation misroutes in both
directions: a component double-consume escapes with blame on an unrelated
fresh binding, and an ordinary alias inherits a later destructure's
component mark); [#1212] is the same defect entering through a synthesized
name (an authored `__chelis_tmp0` in an enclosing block shares a spelling
with a desugarer-minted carrier, and the name-keyed checker cannot tell them
apart, so a genuine use-after-consume vanishes); [#1211] is the review
bucket whose still-open remainder is re-verified against this delivery.
Three local patches would have encoded three ad-hoc answers to the identity
question, each free to drift back into the others' failure. §C1's intent
read at the linearity stage — the checker must not return a verdict a
different binding earned — needs the decided rule, with the fixes derived
from it.

**Decision (2026-08-21).** `spec/04-type-system.md` §8.3 [04-LIN-1] is the
controlling rule: every binding introduction creates a distinct binding,
a name at a use site denotes the innermost enclosing one, and every
ownership fact attaches to the binding it was resolved against — never to
the name. [04-LIN-2] pins the capture verdict that rides along: a closure
capture consumes the binding it names, distinct user-visible bindings of
one value are distinct for capture (the nautilus `lu_solve` spelling), and
only a destructured component or an alias of one forwards to its carrier.
The two-closure tightening [#1209] flagged is decided as **preserved** —
recorded normatively rather than left as an implementation accident. The
implementation ledger owner is `spec/design/implicit_linearity.md`
§"Aliases".

**You deliver:**

1. **The identity mechanism, not instance patches.** `LinearScope` keys
   all checker state on per-binding generation ids (`BindingId`, minted at
   `declare`); `BindingOrigin.alias` stores the id resolved when the alias
   bind is recorded and never re-resolves. Both [#1209] misroute
   directions and [#1212]'s collision must become unrepresentable through
   the one mechanism.
2. **The authority for it.** [04-LIN-1] and [04-LIN-2] amend
   `spec/04-type-system.md` §8.3 in the same change set (Numbered Specs
   Decide); `implicit_linearity.md` §"Aliases" carries the mechanism
   description and the reasoning.
3. **An explicit in-or-out record for [#1211].** Each of its four items is
   re-verified by execution against merged main and dispositioned on its
   thread: items 1 and 3 landed inside [#1208]'s final review rounds, item
   2's verdict is recorded (ordinary-alias join behavior stays, per
   [04-LIN-2]'s companion text in `implicit_linearity.md`), item 4 is
   [#1212] and closes here.
4. **Verdict neutrality outside the class.** The only verdict changes are
   the three acceptance cells and the recorded `x = x` component
   self-rebind delta; every other cell in the #1200/#1208 contract suites,
   the lane-parity suite, and coral's vendored corpus keeps its verdict.

**Oracle:** the three formerly-`#[ignore]`d cells in
`crates/chelis-types/tests/issue_1200_destructure_component_scope.rs`
(`pass_b_alias_survives_shadowing_of_its_source`,
`pass_b_alias_is_not_captured_by_a_later_destructure_of_its_source_name`,
`authored_destructure_temp_name_does_not_hide_an_outer_double_consume`) run
un-ignored and green; every sibling cell in that file is a negative
control proving the fix is an identity rule and not a loosened or
disabled check, with
`two_closures_capturing_one_value_through_an_alias_still_compile` pinning
[04-LIN-2]'s preserved verdict and the chelis-cli
`issue_1200_destructure_scope_lane_parity` determinism cell pinning the
sorted-capture rejection. Executed mutation receipt, 2026-08-21: making
`resolve_alias_chain`'s hop re-resolve each source through its *name*
(`.and_then(|source| self.record(source)).and_then(|source_record|
self.top_id(&source_record.name))` in place of following the recorded id)
and running `cargo nextest run -p chelis-types --test
issue_1200_destructure_component_scope` failed exactly the three
acceptance cells and no sibling (at the file's then-27-cell state: 24
passed, 3 failed); reverting made all 27 green. Cells added afterwards
that also ride generation identity (the component self-rebind pin) would
join the failing set under the same mutation. The delivery remains one
class change: a shadowing special-case in the chain walk or a
reserved-name screen in the desugarer does not satisfy it.

### PP4. Exact module scope at checker and test-batch boundaries ([#1264], residue of [#1261])

**Delivered 2026-08-31 by PR [#1402].** [#1264] and the residue explicitly left
by PR [#1273] were two entry paths into one scope-identity defect.

- Before PP4, package checking already rewrote every value that was genuinely in
  scope to its exact internal identity. A bare unimported value remains bare,
  but `infer_var` followed a failed exact lookup with
  `lookup_terminal_unique`, so one foreign export with the same terminal name
  became an accidental binding. `chelis check` and `eval` therefore accepted a
  program whose build-side check rejected.
- Before PP4, suite batching flattened the raw declarations from every
  admitted file into one `combined_decls` unit before name resolution. PR
  [#1273] made explicit declared/imported collisions demote to per-file
  execution and made fallback visible, closing [#1261]. It deliberately could
  not see the opposite sign: file A has an unimported bare use and file B
  declares the same terminal name, so there is no declared/imported collision
  to classify and B silently confers scope on A.

Both paths violate the same rule in `spec/02-surf-syntax.md` P2: scope comes
from lexical/module declarations, builtins, and imports, never from a unique
terminal match elsewhere in the linked program. A checker-only removal of the
fuzzy fallback would close the package reproducer while leaving the batch
construction able to manufacture a new exact binding before the checker sees
the use. Another `BatchScope` census arm would close only known declaration
shapes and leave the next unenumerated scope producer open. PP4 closes the
identity mechanism at both boundaries.

**Decision (2026-08-31).** `spec/02-surf-syntax.md` P2 is the controlling
scope rule. `spec/04-type-system.md` [04-FIT-2] controls the existing fitness
wire: `UnboundVariable` and `UnknownConstructor` both lower `names`, populate
`unresolved_names`, keep `errors` non-empty, and force `score < 1`.
`spec/design/chelis_native_testing_plan.md` controls the execution mechanism:
`--batch-mode auto` may share a prepared package and one evaluator handle, but
each test file retains the same source scope it has under `--batch-mode file`.
No new CLI option or JSON field is introduced.

**You deliver:**

1. **Exact value resolution.** Value-position inference resolves through the
   exact environment only. Terminal-segment lookup remains available for
   spelling suggestions and out-of-scope diagnostics, but it is never a
   binding path. The existing constructor rule is the positive architectural
   precedent: reef-rewritten exact constructors resolve, while fuzzy
   constructor lookup is diagnostic-only.
2. **One isolated batch-link operation.** The prepared-graph boundary accepts
   a list of test entry modules, each carrying its manifest file index, parsed
   declarations, and selected test roots. For each entry it inserts the
   synthetic roots and runs the ordinary module rewriter under a deterministic
   reserved module identity derived from that index. It returns rewritten
   declarations and an original-root-to-exact-root map. The manifest index,
   not a path spelling or user declaration, is the namespace key, so two test
   files cannot collide with each other or a package module. The operation
   derives the synthetic module's complete local symbol map from that file
   alone (including signatures, values, types, constructors, dimensions,
   macros, and properties) and resolves only its declared imports through the
   prepared package graph; it never publishes one test module's symbols into
   another's resolver.
3. **Rewrite before combine, exactly once.** `run_test_batch` combines only
   the independently rewritten declarations and evaluates only the returned
   exact roots. Both the cached `CompiledContext` path and the legacy
   `PreparedReefGraph` worker consume the same isolated rewrite product. That
   product is appended to the prepared library snapshot without passing
   through the eval-entry rewriter a second time. Raw declarations from two
   files never coexist in one source scope.
4. **Admission is not authority.** `BatchScope` remains a conservative
   performance/admission rule and continues to explain demotions, but no
   successful verdict depends on its name inventory being complete. A batch
   rejection that requires per-file attribution uses the already-reported
   fallback and produces the same file/test rows and exit verdict as
   `--batch-mode file`; it cannot disappear behind the fallback.
5. **Structured fitness accounting.** The unresolved identifier is structured
   checker-diagnostic data, populated when an `UnboundVariable` or
   `UnknownConstructor` is constructed. One shared kind-and-identifier
   accessor drives both the `names` numerator and `unresolved_names`; fitness
   does not recover identifiers by parsing rendered messages. The public
   report keeps its existing keys and [04-FIT-2]'s one-entry-per-diagnostic
   ordering.
6. **Spec-first paired coverage.** The implementation change begins with the
   PP4 oracle's failing negative and positive stubs. It does not absorb
   builtin/prelude shadowing ([#1076]/[#672]), integer type applications
   ([#1247]), stamped-ingress parity ([#1125]/[#1134]), or the tag-keyed
   coverage exemption ([#874]/[#887]).

**Oracle:**

```sh
cargo nextest run -p chelis-cli --test issue_1264_module_scope_honesty
```

The test is the authoritative PP4 completion oracle and drives the public CLI
against temporary reef packages. It contains both polarities for every rule:

- a unique unimported foreign value, and the same bare name exported by two
  foreign modules, reject as `UnboundVariable` in `check`, `eval`, and build;
  an explicit selective import and a qualified reference pass;
- an unimported record constructor rejects as `UnknownConstructor`; both name
  error kinds make `check` exit 2 with `names < 1`, a non-empty `errors` array,
  and one matching `unresolved_names` entry, while their imported controls
  keep `names == 1` and an empty list;
- adding, removing, or renaming an unrelated exporting module cannot change a
  selected module's verdict;
- in a two-file suite, file A's unimported bare use cannot resolve to file B's
  declaration under `--batch-mode auto`; `auto` and `file` emit the same
  pass/fail rows and nonzero test verdict. An explicit import and an unrelated
  sibling declaration remain green controls;
- exact linker identities, lexical locals, and builtins remain positive
  controls, proving that the change removes fuzzy scope rather than name
  resolution generally.

Supporting unit tests pin the exact-vs-terminal environment edge and
[04-FIT-2]'s two diagnostic kinds. The mutation receipt adds a foreign export
to the negative package and independently makes the batch sibling declare the
bare name; neither mutation may turn a red source module green.

PR [#1402] carries the implementation and exact-head validation record. The
command above remains the standing completion oracle after delivery;
supporting unit coverage pins every isolated-entry declaration namespace and
fail-closed synthetic identity in addition to the two checker edges above.

### PP5. Elementwise rank honesty ([#668], partial)

`spec/04-type-system.md` §4.1-§4.3 and
`spec/05-risc-primitives.md` §1.2/§2.1 already decide the language
rule: elementwise tensor operands have identical dimension lists, and a rank
change must be explicit. PP5 adds no semantic rule.

This section states only results that a named test or a reproducible probe
demonstrates. Each one carries its evidence. Two earlier attempts to describe
PP5 wrote a broad claim and then subtracted exceptions from it; both times a
review found the subtraction incomplete, because a sentence of the form
"closed except X" is only as true as the enumeration behind it. Nothing here
is written in that shape. What is not listed is absent, not qualified.

#### The four results

**R1. Where the checker has derived a rank fact for BOTH operands of a
`ShapeClass::Identity` call, a positive-rank disagreement is rejected.**

Two roles are distinct here. Facts are *derived* for `ShapeClass::Identity`
plus the four operations named in `derive_ir_builtin_output_type`: `conv2d`,
`stride`, `expand`, `softmax`. Facts are *acted on* only at a
`ShapeClass::Identity` call site, gated in
`validate_ir_builtin_symbolic_requirements`; the four named operations are
derivation sources, not rejection sites. The check also needs facts for two
operands: an argument with no fact is skipped, so one fact beside one factless
operand raises nothing. Where both are present and the positive ranks differ,
the result is a located `DimensionMismatch`, `check` exits nonzero, and
`score < 1`. Rank-zero
arguments remain subject to the builtin's ordinary scheme (including `clamp`'s
explicit scalar-bound form); PP5 creates no scalar-broadcast permission.

*Evidence:* `cargo nextest run -p chelis-cli --test
issue_668_elementwise_rank_honesty`, 17 tests, covering operand order, nested
inline identities, `floor_div`, discarded comparisons, rank-zero, matching-rank
controls, and lexical shadowing. The negative also joins the permanent #731
fitness-honesty corpus.

**R2. The rank facts the checker carries cannot be forged from source, and are
not inherited across a rebinding.**

Rank-only facts live in a private Rust `ShapeTypeFact::RankOnly` carrier with
no Deep-syntax representation, so an authored dimension name cannot alias
validator state. Exact-shape validators consume only the `Exact` variant, so
the checker neither copies nor invents runtime extents. The carrier accepts
exact facts only from checker-owned lexical bindings: raw `type` metadata on a
runtime expression is source syntax, not validator evidence. Binding a name to
a value whose shape is not derivable clears that name's prior fact rather than
inheriting it.

*Evidence:* within the same 17 tests, `authored_rank_only_prefix_dimension_is_
not_internal_validator_state`, `authored_deep_type_metadata_cannot_override_
checker_owned_rank_facts`, and the rebinding pair. The rebinding positive was
confirmed to fail against the pre-repair tree and pass after it; its negative
companion pins the opposite direction, so the repair cannot become "stop
checking rebound names".

**R3. The tensor-DAG C emitter compares every positive-rank input pair and
aborts before allocation or indexing.**

This covers operations lowered through the tensor-DAG emitter
(`chelis-backend-c/src/emit.rs`, `emit_elementwise_operand_guard`).

*Evidence:* `chelis-backend-c`'s
`direct_positive_rank_mismatch_traps_before_indexing`, which constructs a
rank-divergent `Dag` directly and asserts both that the guard appears in the
emitted C and that the compiled binary aborts.

**R4. The host-value C emitter compares the operands of its six binary
elementwise builtins and aborts before the result allocation ([#1484]).**

`chelis-backend-c/src/host_emit.rs` is a different lane from R3's, and an
elementwise call one of whose operands carries an IO effect is lowered there.
It allocated the result at the LHS rank and then computed `idx_lhs` and
`idx_rhs` by feeding the target's indices into each operand's strides, with no
comparison of any kind: `add(debug(e), s)` for a rank-2 `e` and a rank-1 `s`
compiled, exited 0, and printed a `[2, 6]` result invented from a `[2, 6]` and
a `[3]` operand, deterministically, while `chelis eval` rejected it.

The guard now runs at the top of both of that file's two-tensor elementwise
emitters, which between them carry six builtins: `add`, `sub`, `mul`, and
`div` through `assign_tensor_binary_elementwise`, and `max_elem` and
`min_elem` through `assign_tensor_binary_func_elementwise`. Two positive
operand ranks that differ abort; at equal positive rank the shapes are
compared axis by axis, which is what `spec/05-risc-primitives.md` §2.4 already
requires of R3's lane. A rank-0 operand keeps the backend's scalar-input
meaning and is not reinterpreted as source-level broadcasting. Nothing else in
the file is claimed: the unary, movement, gather/scatter, and summary emitters
are untouched.

R4 is a loud-failure backstop and changes no checker verdict. `add(debug(e),
s)` still scores 1 with an empty error list; the checker half of that program
is R1's subject and is unchanged.

*Evidence:* four `chelis-backend-c` tests that drive `emit_host_program`
directly, `host_lane_positive_rank_mismatch_traps_before_indexing`,
`host_lane_max_elem_positive_rank_mismatch_traps_before_indexing`,
`host_lane_equal_rank_shape_mismatch_traps_before_indexing`, and the
`host_lane_matching_shapes_still_compute` control; and four `chelis-cli` tests
in `issue_1484_host_lane_rank_guard`, which run the three programs recorded in
[#1484] through `chelis build --target c` and the host linker and assert that
each aborts with the guard text where `chelis eval` reports a shape mismatch,
plus the `issue_1484_matching_rank_host_lane_still_runs` control. Each of the
six rejection tests was confirmed red on the pre-fix tree, three of them by the
compiled binary printing values instead of aborting. `grep -c 'rank mismatch'
crates/chelis-backend-c/src/host_emit.rs` now returns non-zero, locked by
`sibling_sweep_host_emit_carries_the_elementwise_rank_guard`.

#### What PP5 does not establish

No test asserts that PP5's rank validator rejects a disagreement arising from
an operation outside R1's derived set, because it does not reject one.

This is a statement about PP5's validator, not about the checker as a whole.
Ordinary inference still rejects what it can see: a declared-rank mismatch
between a `sig` and its argument reports `tensor rank mismatch: 2 dims vs 1
dims` at score 0.92, and mixing dtypes reports a precision mismatch. The
escapes below are programs where inference unifies through a movement
operation's wildcard, so no rank conflict reaches it, and PP5's validator has
no fact to supply one.

Constructing one therefore takes a movement-op result laundered through an
operation the resolver does not classify. The direct form is caught:
`add(x, cast(y, f32))` on a rank-1 and a rank-2 parameter reports a rank
mismatch at 0.9586. `cast_trunc` needs both operands cast, because casting one
leaves a precision mismatch to report at 0.9385. Confirmed escapes, each
scoring 1 with an empty error list: `ShapeClass::Rewriting` operations such as
`cast`, `cast_trunc`, `normalize`, and `realize`; rank-changing
`ShapeClass::NameTracked` reductions such as `add(e, sum(e, 0i32))` and
`add(e, mean(e, 0i32))`; and a rank reaching the operation through a user
`def`. These are examples of the gap, not a partition of it.

The gap is therefore not "shape-preserving operations are missing". The
resolver derives rank for one central class plus four separately named
operations, and everything else escapes, whether it preserves rank or changes
it. Closing that requires deciding which property the resolver should be driven
by and driving it totally from a registry, with no separately named
operations. That is follow-up work owned by [#668]. Adding further names would
rebuild the enumeration this slice removed, and would not change the shape of
the claim.

#### Acceptance

```sh
cargo nextest run -p chelis-cli --test issue_668_elementwise_rank_honesty --no-fail-fast
cargo nextest run -p chelis-backend-c --test exec_compile direct_positive_rank_mismatch_traps_before_indexing
cargo nextest run -p chelis-backend-c --test exec_compile host_lane --no-fail-fast
cargo nextest run -p chelis-cli --test issue_1484_host_lane_rank_guard --no-fail-fast
```

The first is the oracle for R1 and R2, the second for R3, and the last two for
R4. None of them covers the operations named above, and the last two are about
generated C, not about a checker verdict.

### PP6. Schedule and header honesty ([#1486], [#1487], [#1485]; closes [#1134])

**Opened 2026-09-03; decided below, not delivered.** PR [#1457] delivered
[04-INF-4] and recorded three defects as ratcheted residue. They share one
mechanism: the checker lets a top-level reference observe a declaration
before that declaration's own body has been checked, or fails to see that a
reference will run while a value is being initialized. Each was confirmed
from the code and, where the prebuilt binary predating [#1457] could reach
it, by execution; the PP6 pull request body carries the probe transcript.

- [#1486], a compiled wrong answer. `def f(n: int32) = add(1, n)` desugars
  to a `defsig` whose result slot is `(t-var {} _)`
  (`crates/chelis-surf/src/desugar.rs`, the `None => node(DeepTag::TVar,
  vec![sym("_")])` arms of the synthesized signature). At
  `TypeUseSite::Defsig` the resolver mints a fresh variable for `_`
  (`crates/chelis-types/src/deep_type.rs`, `resolve_type_var`: `if name ==
  "_" { ... self.vg.fresh_tvar() }`), and `collect_declarations`
  (`crates/chelis-types/src/infer/common.rs`, the `DeepTag::Defsig` arm)
  resolves the signature inside `subst.enter_level`, leaves the level, and
  calls `env.generalize(&ty, subst)`, which quantifies every variable minted
  above the current level, the hole included: the header becomes
  `forall a. (int32) -> a`. `infer_top_level` later instantiates that scheme
  for the body, unifies the body against the instance, and rebinds the name
  to the narrowed, regeneralized result, so the environment scheme is
  `(int32) -> int32` only after the body is inferred. A reader scheduled
  before the body (`infer_var`, `crates/chelis-types/src/infer/expr.rs`,
  `env.instantiate(&scheme, vg, subst)`) instantiates the quantified hole at
  a fresh variable, accepts `r: f32 = f(2)`, and is never revisited. The
  schedule places a value below the hoist floor, and every value in a bare
  unit, before every function body, so the verdict depends on layout:
  measured on the prebuilt binary, `chelis check` scores 1 with the reader
  first, `chelis eval` prints `r = 3`, and the compiled C prints `r = 3.0`,
  while the same reader after an anchor function rejects with
  `TypeMismatch`. The issue's diagnosis is exact. Its second half is the
  same mechanism through an authored binder: `def f[a](x: a) -> a =
  add(x, 1)` and the implicit `def f(x: a) -> a = add(x, v)` both check
  clean, the body instance of `a` is bound to `int32`, and the registered
  scheme is the narrowed `(int32) -> int32`; the only rigidity check today is
  `check_declared_dvars_rigid`, for dimensions. Measured: a reader after the
  function sees the narrowed scheme (`s = f(1.5f64)` rejects with
  `PrecisionMismatch`), a reader before it is accepted and evaluates to a
  numeric-op error.
- [#1487], a compiled wrong answer of the [#1339] class.
  `detect_top_level_binding_cycles` (`crates/chelis-types/src/infer/
  declarations.rs`) collects each value's eager references with
  `collect_eager_refs`, whose `Some(DeepTag::Fn) => {}` arm skips every
  lambda body ("only its application at this site (if any) is eager; the
  body itself is deferred"). The schedule's own walker,
  `collect_top_level_calls`, does descend into lambdas, so the two walkers
  disagree about the same reference. The detector is also asymmetric in a
  second way, measured on the prebuilt binary: `carried = pick(f)` with
  `f` reading `carried` is `CycleDetected` (a bare function reference is
  followed into the function's body), while `carried = pick(fn (x: int32)
  -> f(x))` with the same `f` is not. A third blind spot is in the DFS: a
  `call_edges` step onto a value that is on the value stack is skipped as
  recursion rather than reported, so a value that applies a closure held
  by an enclosing value's initializer is never a cycle. The issue's
  diagnosis of the lambda skip is exact; the accidental rejection it
  describes is the [#1485] stall, and the annotated spelling is accepted
  with score 1, fails under `chelis eval` with `cyclic top-level runtime
  definition carried`, and prints `carried = [1, 2]` from the compiled C.
  The issue's candidate fix (eager only in argument position of an eager
  application) is not adopted, for the reason given under the decision.
- [#1485], an over-rejection and an ingress split. `primary_inference_
  schedule` (`crates/chelis-types/src/infer/program.rs`) builds a mirror
  edge from a module function to every item at or after the floor that
  references it and a read edge from a value to every later item that reads
  it. `carried = wrap(f)` with `f` reading `carried` closes a two-cycle;
  Kahn's algorithm stalls and the release `ready.first().or_else(||
  pending.first())` emits the hoist-order-least remaining vertex, the
  function, whose body then reads a value that has no binding yet at the
  typed ingress and a body-stamp binding at the serialized-IR ingress. The
  issue's diagnosis of the stall is exact. Its claim that the shape is legal
  rests on [04-INF-4] alone; under [04-INF-7] below, every one of its three
  spellings is an eager value cycle, so the disposition is a different
  rejection reported identically, not an acceptance.

**Decision (2026-09-03).** `spec/04-type-system.md` §3.1.3 and §3.1.4 now
carry the language rules; this section only implements them.

- [04-INF-5]: a wildcard slot is an inference hole, not a binder. It is
  never quantified, and every reference to the declaration is typed at the
  body-determined signature wherever the reference sits. The alternative
  considered, one monomorphic variable shared by header, body, and readers,
  was rejected because a hole that the body resolves to a type mentioning
  the declaration's own binders (`def f(x: tensor[n, f32]) = x`) would tie
  the body instance of `n` to an outer-level variable and make the function
  monomorphic in `n` for the whole unit, a regression on a common partial
  header. [04-INF-5] instead keeps today's post-body scheme and forbids
  observing the hole early.
- [04-INF-6]: an authored type binder, explicit or implicit, is rigid in the
  body. This is the type-variable form of §4.4's dimension rule, follows
  §5.8.1's `forall` quantification and [04-INF-2]'s strict treatment of
  authored binders, and closes [#1486]'s second half at the declaration
  rather than by scheduling: a reader that instantiates an honest header
  early is sound. The alternative, ordering readers after the body and
  letting the body narrow the binder, would make an authored `[p: Float]`
  silently mean `f32`, which contradicts the promise the author wrote.
- [04-INF-7]: the eager reference set follows into lambda bodies and treats
  applying a value as requiring it. The rule is deliberately the syntactic
  over-approximation. The finer rule the issue proposed (eager only for a
  lambda in argument position of an eager application, or only for callees
  that apply their argument) is not sound: a closure stored in a list or
  returned from a helper and applied by a later value's initializer
  (measured: `def mk() = fn (x: int32) -> f(x)`, `b: int32 = (mk())(1)`,
  `f` reading `b` checks clean and fails under `eval` with a runtime cycle)
  escapes it, and deciding whether a user-defined callee applies its
  parameter is a higher-order flow question the checker cannot answer at
  the cycle detector. The over-rejection is accepted under the repository's
  explicit-over-inference bias, and it adds no new class: the detector
  already treats a bare function reference this way, so the rule removes an
  asymmetry rather than introducing a policy. The escape hatch is to pass
  the value as an argument.
- [#1485]'s three spellings under these rules: `carried = wrap(f)` with a
  signed `f` that applies `carried`, `carried = pick(fn (x: int32) -> f(x))`
  with `f` reading `carried`, and `carried = wrap(g)` with a `defsig`-less
  `g` that applies `carried` are each an eager value cycle under [04-INF-7]
  (the value's initializer names the function, the function's body reads or
  applies the value). Expected verdict at both ingresses, Surf and stamped
  IR alike: `CycleDetected` naming the path, no `UnboundVariable`, no
  split. With the mirror edge narrowed, the two signed spellings no longer
  cycle in the schedule; the `defsig`-less spelling still does and stays
  total through the mixed group below. The design does not depend on
  their acceptance.

**Mechanism.** The [#1134] invariant is unchanged: visibility is a function
of source position and body-inference order is a function of dependency.
PP6 adds a third clause: **a header is available to a reference only when
it is honest**, where a complete or authored-binder header is honest by
[04-INF-6] and a header with a hole is honest only after its body.

1. *Rigid authored binders.* After the post-body signature unification in
   `infer_top_level`, every authored type binder's body instance must still
   be an unbound variable, and the instances must be pairwise distinct;
   otherwise a `TypeMismatch` at the declaration names the binder and the
   type it was narrowed to, in the shape of `check_declared_dvars_rigid`.
   The resolver already keys binder names to variables for dimensions
   (`record_declared_dim_names`); the same recording is added for type
   binders so implicit binders (`def f(x: a) -> a`, no binder list) are
   covered as well as `DeclaredSigMetadata.binders`. The registered scheme
   is unchanged when the check passes, since an unnarrowed instance
   generalizes back to the declared signature.
2. *Hole edge.* The schedule gains one edge kind: an item is inferred after
   every function it references whose signature contains a wildcard slot,
   in every region, below the hoist floor and in bare units included. The
   set of such functions is a syntactic scan of the unit's `defsig`
   expressions for `_` in a type, dimension, or rank position. Kahn's
   priority keeps the displacement minimal: the function keeps its hoist
   position, and only its readers slide after it. The mirror edge narrows
   to `defsig`-less module functions, [#1457] round 10's mechanism: a
   complete or authored-binder header is honest before its body under
   [04-INF-6], a hole header is covered by the hole edge, and only a
   `defsig`-less function has no header for a reader to use, so the edge
   keeps exactly the availability role round 7's class needs. Round 11's
   counterexample to round 10 was a quantified hole, which [04-INF-5]
   removes. The `defsig`-less floor bound (a value below the floor reading a
   `defsig`-less later function is unbound) is function visibility, owned
   by [04-INF-2]/[04-INF-3], and PP6 does not move it.
3. *Eager references.* `collect_eager_refs` descends into `fn` bodies with
   the lambda's parameters bound, in the initializer walk and in the
   function-body walk that `detect_top_level_binding_cycles` chains through
   `fn_body_refs`. In the DFS, a `call_edges` step onto a member of the
   value stack reports the cycle exactly as a `value_edges` step does. The
   external-input exemption and the function-application chaining are
   unchanged, and so is the value/function line: [04-INF-7] now states
   what `is_value` in `detect_top_level_binding_cycles` already decides, a
   `def` whose initializer is a lambda is a function.
4. *Mixed groups.* `primary_inference_schedule` contracts every strongly
   connected component of the full reference graph (call, read, mirror, and
   hole edges), not only the planner's recursive function components, and
   `primary_inference_groups_for_schedule` emits each component as one
   group. `prebind_recursive_function_schemes` additionally prebinds a
   value member with a monomorphic fresh variable when the value has no
   declared signature and no metadata prebind; `infer_top_level`'s
   provisional path already unifies a member's body with its provisional
   type and defers generalization to the group, and needs no change for a
   value. The stall release (`or_else(|| pending.first())`) is deleted: a
   DAG of components never stalls, and the schedule is total by
   construction. Under [04-INF-7] a mixed group is always part of a
   rejected program; the group exists so that inference on that program is
   total and reports the same diagnostics at both ingresses, and so that a
   future refinement of [04-INF-7] would not reopen the stall.

**You deliver:**

1. **Slice A ([#1486]).** Items 1 and 2 above. Ratchets that invert:
   `a_partial_or_generic_header_is_not_instantiated_before_its_body_narrows_it`
   (`crates/chelis-types/tests/issue_1134_forward_reference_parity.rs`)
   keeps its two programs and gains their below-floor and bare-unit
   layouts, each rejecting identically; the generic program's expected kind
   moves from the reader's `PrecisionMismatch` to the declaration's
   `TypeMismatch`, because [04-INF-6] rejects `f` itself.
   `check_rejects_a_mismatched_read_of_a_partial_header_deferred_by_a_barrier`
   (`crates/chelis-cli/tests/issue_1134_forward_reference_cli.rs`) gains the
   reader-first layout at `check`, `eval`, and `build`. New regressions:
   `def f[a](x: a) -> a = add(x, 1)` and `def f(x: a) -> a = add(x, 1)`
   reject at the declaration with no reader present, a binder collapse
   `def g[a, b](x: a, y: b) -> a = y` rejects, and `def id(x: a) -> a = x`,
   `def k(x: a) = x`, and a bounded `[p: Float]` body written with
   `cast(0.0, p)` stay accepted. `schedule_invariants.rs`'s generator
   declares whether a function's signature has a hole, `reference()` emits
   the hole edge for every reader position, and a named regression pins
   that a below-floor reader of a hole-signature function is scheduled
   after it while a complete-header function's reader is not moved;
   `reference()`'s mirror rule becomes `defsig`-less-only, and the signed
   spelling leaves `a_value_naming_a_function_that_reads_it_back_is_a_
   recorded_stall`, whose graph must stay cyclic only for the `defsig`-less
   one.
2. **Slice B ([#1487]).** Item 3. `an_initialization_cycle_leaves_the_
   schedule_total_at_both_ingresses` gains the lambda-mediated spellings
   from the issue, annotated and unannotated, each `CycleDetected`
   identically; a CLI test rejects them at `check`, `eval`, and `build`; the
   returned-lambda shape and a closure applied by a later value are
   negatives; the stored lambda `carried = fn (x: int32) -> f(x)` (a
   function `def` to the planner), a lambda reading an earlier value
   through a callee, and every program in `examples/iter_foundation.ch` are
   positive controls.
3. **Slice C ([#1485]).** Item 4. The three ratchets invert:
   `a_value_naming_a_function_that_reads_it_back_is_a_recorded_stall` in the
   parity suite asserts `CycleDetected` and the absence of
   `UnboundVariable` for all three Surf spellings and asserts that the
   stamped spelling rejects identically at both ingresses; the CLI ratchet
   `a_value_that_names_a_function_reading_it_back_is_a_recorded_stall`
   asserts `CycleDetected`; the schedule-oracle ratchet
   asserts that the reference graph is cyclic, that the schedule emits the
   component contiguously, and that no `UnboundVariable` reaches the
   verdict. `a_genuine_binding_cycle_stays_total_with_callees_first` is
   restated over component contiguity. The `--lib` oracle's stall
   mutations (release by lowest ordinal) become inexpressible and are
   deleted from the receipt table; a new receipt restores the stall
   release and shows the ratchet reddening.
4. **Closing [#1134].** After Slice C the parity suite carries no residual
   predicate, `[04-INF-4]`'s parenthetical is removed together with the
   three new atoms' parentheticals, the `[#1134]` residue section above is
   updated to name PP6 as the owner of the mirror edge's remaining role, and
   the issue map row below turns green. "Dispositioned" means, per issue:
   [#1486] rejects at both ingresses and at `check`/`eval`/`build` for the
   reader-first, at-floor, and bare layouts of both programs, and the
   rigid-binder negatives above reject; [#1487] rejects its annotated and
   unannotated programs as `CycleDetected` at both ingresses and at the CLI,
   with the stored-lambda control accepted; [#1485] reports identical
   diagnostics at both ingresses for all three spellings and the stamped
   one, with no `UnboundVariable`.
5. **Stdlib migration under [04-INF-6].** Ten stdlib declarations narrow a
   bounded binder with an unsuffixed float literal and are rejected once
   the binder is rigid: `abs_float`, `erf_approx`, and `normal_cdf` in
   `packages/chelis-std/src/contracts.ch`; `validate_fan_in` and
   `finite_float` in `init/kaiming.ch`; `validate_normal_params` and
   `finite_float` in `init/random.ch`; `validate_xavier_params`,
   `validate_trunc_params`, and `finite_float` in `init/xavierext.ch`. Each
   repair is the §P10 `cast(<literal>, p)` override; the only in-tree
   precedent is the Int form `cast(1, p)` in `arange_values`, and no float
   spelling exists in the tree yet. The count was taken by inspection of every
   binder-list declaration in `packages/chelis-std/` and `examples/`; the
   implementer's first Slice A step is to run the rigid check over the
   stdlib and confirm exactly that list reddens. A stdlib source change
   regenerates the bundle and commits `reef.lock` and the tracked `dist/`
   artifacts. No stdlib or example source declares a partial header (a
   `def` with an annotated parameter and no result type): zero in both
   trees, so item 2 changes no shipped schedule, and no top-level value in
   either tree nests a lambda that names a top-level `def`, so item 3
   changes no shipped verdict.
6. **Slices.** Hand-written estimate: Slice A 300-400 lines including
   tests and the stdlib repair, Slice B 120-180, Slice C 200-300. Land as
   one pull request with one commit per slice in the order A, B, C. A
   shrinks C's cyclic class to two-cycles through a `defsig`-less mirror
   edge or through a hole edge (`r = f(2)` with `def f(n: int32) =
   add(r, n)`), and C is still needed for those; C is only sound after A,
   because a mixed group lets a value instantiate a partial header before
   the function's body. If A lands alone, the two signed [#1485] spellings
   stop stalling at A and their ratchets invert to acceptance there, then
   to `CycleDetected` at B; in one pull request they invert once, at B. If
   the total exceeds about 800 hand-written lines, split as A alone, then
   B and C together.

**Prove-fails-first.** Every new rejection is shown red against the base
tree by reverting the owning source paths to the base commit, rebuilding the
one test target, and watching the assertion fail, then restoring and
watching it pass; every inverted ratchet is shown red against the base tree
by the same procedure. Each new assertion's doc comment is labeled
"regression test" or "disposition lock". The mutation receipts are: quantify
the hole again (drop the hole edge) and watch the reader-first [#1486]
programs accept; skip the rigid check and watch the binder negatives accept;
restore the `Fn` skip in `collect_eager_refs` and watch the [#1487] programs
accept; skip the value-stack test on `call_edges` and watch the signed
[#1485] spelling lose its `CycleDetected`; restore the stall release and
watch the stamped [#1485] spelling split again.

**Oracle:**

```sh
cargo nextest run -p chelis-types --test issue_1134_forward_reference_parity --no-fail-fast
cargo nextest run -p chelis-types --lib infer::tests::schedule_invariants
cargo nextest run -p chelis-cli --test issue_1134_forward_reference_cli --no-fail-fast
```

The first is the verdict oracle for all three issues at both ingresses, the
second the order oracle, the third the public-surface oracle. Before
pushing, the implementer also runs the complete `chelis-types` corpus,
`issue_1124_ir_defsig_unification_parity`,
`rt800_append_only_cycle_resolution`, `recursive_generic_monomorphization`,
the `issue_1339_*` CLI tests, `scripts/compiled_value_ownership_oracle.py
--phase 0`, `scripts/unrepresentable_domain_oracle.py`,
`scripts/dtype_phase4b_oracle.py`, and a differential `chelis check` over
every tracked `.ch` and `.dp` file against a control binary built from the
base, expecting the stdlib list above and no other score change.

**Exclusions.** PP6 does not absorb [#1512] (an early return on an
unresolved `expand` operand skipping validation, owned by the [#1277]
stream; PP6 touches header-versus-body typing in the schedule, not deferral
settlement); [#1339]'s indirect shape (an earlier initializer that calls a
function reading a later-assigned value obeys [04-INF-4] and [04-INF-7]
alike and stays with the compiled-value ownership plan); [#874]/[#887]'s
tag-keyed vacuity; [#1125]'s reader audit; the `defsig`-less floor bound on
function visibility; and any change to `spec/02` §P10's literal rule.

**Risks.**

- *PP1 obligations.* A mixed group runs `finish_deferred_shape_checks` per
  member exactly as a recursive function component does; [04-INF-1]'s
  bind-on-first-use lambdas are unaffected because the group boundary is
  the declaration boundary. Verify with the PP1 suites in the corpus run.
- *Typecheck cache.* `CACHE_FORMAT_VERSION` (`crates/chelis-compiler-api/
  src/context.rs`) need not move: `Scheme` gains no field and the cache key
  already includes the compiler identity. A cached context written by an
  older compiler cannot hold a stdlib scheme the new compiler rejects,
  because the stdlib is repaired in the same change set.
- *Compiled-lane prerequisite.* The C emitter assigns statics in source
  order and has no runtime cycle check; the detector is the only guard
  between an accepted program and a wrong answer of the [#1339] class,
  which is why [04-INF-7] over-approximates rather than refines.
- *Diagnostic order.* Readers of hole-signature functions move after the
  function, so their diagnostics move with them; the parity suite compares
  ordered diagnostics between ingresses, never against source order.
- *Stdlib exposure* is item 5; the census is by inspection until the rigid
  check runs.

### Later residue: kinded nominal applications ([#1247], with [#1258])

**Delivered by PR [#1406].** This is a separately landable #731 residue
mechanism, not PP4 or a fifth historical phase. [#1247] owns checker honesty;
[#1258] is the same boundary viewed from #1024's Surf/Deep round-trip
contract. The PR merged after its acceptance oracle passed; the command below
remains the standing regression oracle.

Before this slice, Surf correctly admitted an integer only inside a nominal
argument such as `Column[3]`, but desugared it as `(t-var {} 3)`. Nominal
headers recorded arity without parameter kind, and the checker represented
every application argument as an ordinary type. The integer therefore became
a fresh type wildcard at `Option[3]` and carried no extent at `Column[3]`.
`chelis check` and `chelis test` could report success on an unenforced
application, while `chelis surf` and `migrate surf` rejected the malformed
numeric `t-var` produced by Chelis itself.

`spec/04-type-system.md` [04-ADT-3]/[04-ADT-4] now control the language rule.
Every `deftype` and `typealias` parameter receives one checker-owned `Type` or
`Dimension` kind before declaration bodies resolve. Direct tensor-axis uses
and transitive nominal-argument uses contribute to a deterministic least
fixed point across forward references, aliases, and recursive headers. A
parameter is dimension-kinded only with dimension evidence and no type
evidence; mixed use rejects and unused parameters retain the ordinary-type
default.

The implementation closes the mechanism at every representation boundary:

1. Surf carries an integer nominal argument as a dedicated dimension-literal
   node and desugars it to `d-lit`; bare integer type positions remain parse
   errors. Deep's typed child-role table admits type or non-rank dimension
   syntax only at a `t-adt` argument.
2. The checker stores nominal type and dimension arguments in distinct enum
   variants. A dimension cannot enter the ordinary `Type::Adt` argument
   vector, and kind mismatches fail before unification instead of allocating a
   wildcard.
3. Dimension arguments participate in substitution and `unify_dim` through
   aliases, constructors, records, matches, signatures, opacity checks,
   evaluation, and native lowering. Unequal concrete extents produce
   `DimensionMismatch`; a dimension supplied to a type slot or a type supplied
   to a dimension slot produces `TypeMismatch`.
4. `chelis check --show-inferred` preserves the dimension structurally under
   the existing ADT wire node, and Hull consumes the new closed dimension
   wrapper. Every bincode cache that can persist the checker/registry shape is
   version-bumped and rejects its predecessor.
5. `chelis check` and `chelis test` share the same rejection path. A failed
   check exits nonzero with `score < 1` and a non-empty error list; a test file
   with the same program reports a compile failure and cannot count as passed.

The standing completion oracle is:

```sh
cargo nextest run -p chelis-cli --test issue_1247_integer_type_application --no-fail-fast
```

It pins both polarities for ordinary, dimension-only, mixed, alias-mediated,
and forward headers; exact error kinds and fitness; structured inferred JSON;
Deep/Surf/migration round trips; record and match behavior; eval and generated
C; and test-runner failure. The existing `issue_935_nullary_generic_adt`
suite is the positive regression oracle for representation-erased nominal
dimensions. The five `hello-chelis` Coral files named by [#1258] are an
ecosystem migration receipt, not a substitute for the repository oracle.

This slice does not absorb the remaining tag-keyed vacuity
([#874]/[#887] Tier 1), [#1125]'s carrier-reader audit, or
compiler-provided-name precedence ([#1076]/[#672]). Their contracts and tests
remain independently owned.

### Later residue: nominal-argument file-ingress parity ([#1125])

The kinded nominal-application delivery exposed one bounded stamped-ingress
instance: the serialized-type boundary rejected a `d-rank` rank spread under
`t-adt`, while ordinary `.dp` program ingress stamped the same form through
the broader `Type` role. Consequently both `chelis surf` and
`chelis validate --deep` accepted and rendered a nominal application such as
`Rows[..r]`, contrary to `spec/03-deep-syntax.md` §2.5.1 and [04-ADT-4].

Program ingress now routes every decoded `t-adt` through the existing
recursive type grammar. Its nominal argument children therefore accept type
or non-rank dimension syntax and reject rank spreads at the boundary. This is
an ingress repair only: `d-rank` remains legal in tensor-axis slots, and
concrete or symbolic dimensions remain legal nominal arguments.

The authoritative oracle for this bounded residual is:

```sh
cargo nextest run -p chelis-cli --test issue_1125_nominal_rank_ingress --no-fail-fast
```

It drives both `surf` and `validate --deep`, with the invalid nominal-rank
case and positive nominal-dimension and tensor-rank controls. Supporting
`chelis-deep` integration tests pin the same role decision directly at
`parse_and_stamp_file`.

This slice is only **part of [#1125]**. It does not complete that issue's
whole-module carrier-reader audit or its carrier-complete structural lint and
justified escape hatch. It also does not absorb [#1134]'s independently
delivered pass-set/forward-reference decision, [#874]/[#887]'s tag-keyed
vacuity, or [#1076]/[#672]'s compiler-provided-name precedence work.

### Later residue: top-level forward-reference parity ([#1134])

[04-INF-4] selects sequential scope for eager top-level values. Both the
stamped typed and serialized-IR checker ingresses reject a reference to a
later top-level value as `UnboundVariable`; body-type metadata is evidence
about a declaration, not permission to manufacture earlier scope. Backward
value references resolve, from a value initializer and from a function body
alike, and an explicitly typed self-reference used for an external input
receives declaration-local type availability before its binding becomes
ordinary scope for later declarations. Bare `x = x` remains an eager cycle.
Existing function inference and recursive SCC behavior remains owned by
[04-INF-2]/[04-INF-3]. Local `let` scope is sequential per spec/03 §6.2.

This narrower decision is required by the executable compiler boundary. A
general value dependency schedule would make [#1339]'s release-blocking
compiled wrong-answer path newly reachable without its formerly required
annotation. Sequential eager scope instead closes the direct form of that
prerequisite at the checker: an annotated or unannotated forward capture,
where a compiled function names the later value itself, rejects before any
backend artifact exists. It does not close the indirect form, in which an
earlier value's initializer calls a function that reads a later-assigned
value; every reference there obeys [04-INF-4] and the emitted `main` still
assigns in source order. That remainder stays with [#1339], and the ownership
oracle freezes only the two direct rows to the rejection shape so the
indirect shape stays representable as a positive row. This residual adds no
dependency-ordered global initialization and does not absorb the
compiled-value ownership class.

#### The invariant this rule rests on

> **Visibility is a function of source position. Body-inference order is a
> function of dependency. Neither may be derived from the other.**

Both halves are needed, and the failure mode is symmetric. Deriving visibility
from inference order means encoding [04-INF-4] as a binding timeline advanced
by the schedule, and every place the timeline and the source order disagree
becomes a scope leak. Deriving inference order from visibility means refusing
to reorder function bodies at all, which breaks [04-INF-2]/[04-INF-3] forward
calls and SCC inference.

The implementation keeps them apart. `Env::top_level_value_visibility` answers
the scope question from recorded declaration positions alone, so no schedule
can widen or narrow it, and every top-level signature stays in the global
header environment where the ingresses already agree about it.
`primary_inference_schedule` answers the availability question: it may reorder
module function bodies, but it must infer an eager value before any function
that legally reads it, because an unannotated value has no header anywhere and
its type exists only once its own `def` has been inferred. Without that, a
reordered reader reports a legal backward reference as unbound, and the two
ingresses diverge, since only the serialized-IR ingress prebinds body-type
stamps.

#### The schedule

The schedule is the hoist-order-least linear extension of a precedence graph
whose every edge is a real reference in the program. The hoist order is the
total order the schedule used before this residual: every module function
spliced at the earliest module-function ordinal in the planner's callee-first
order, everything else textual. The graph's vertices are the items, with each
recursive component contracted to one vertex because
`primary_inference_groups` infers a component as one unit at the first member
the schedule reaches, so an edge one member earns has to constrain them all.
The edges, all "referenced before referencer", are:

- **call**: a module function after every module function it calls, the
  planner's own callee-first order;
- **read**: an item after every eager value it reads that is declared before
  it. For a module function this is the barrier that keeps the hoist from
  carrying it across the value; for a value it is the ordinary value
  dependency. A forward value reference produces no edge, because the scope
  rule leaves it unbound;
- **mirror**: an item at or after the earliest module-function ordinal after
  every module function it reads, signed or not. A declared signature is in
  the global header environment, but it is not yet the function's scheme: a
  synthesized `defsig` with a wildcard slot or an implicit binder is
  generalized over fresh variables, and a reader that instantiates it before
  the body narrows it accepts programs the checker rejects when the body is
  inferred first ([#1486]; round 11 measured the compiled wrong answer that
  dropping the edge for signed functions produced). Bounded to the hoist's
  region, the mirror reproduces exactly what the blunt hoist supplied
  implicitly, so function visibility, which [04-INF-2]/[04-INF-3] own, is
  neither narrowed nor widened. Bare functions keep textual availability and
  earn no call or mirror edge.

Nothing else orders the graph. In particular there is no textual chain over
the non-function items and no chain over the planner order. Four earlier
repairs of this residual carried both chains, and every one of them was found
by a reviewer constructing a layout the verdict matrices could not express:
the chains are not dependencies, so they closed cycles on legal programs
(round 8's `C2_stall_no_cycle`), which made a cycle break a live path, and the
break then released a function out of planner order (round 8's `D4_min`).
With reference edges only, a cycle in the graph is a reference cycle through
an eager value. Most such cycles are runtime initialization cycles that
`detect_top_level_binding_cycles` reports as `CycleDetected` at every ingress.
One is not, and it is the recorded residual of this slice ([#1485]): a value
that names a function reading the value back (`carried = wrap(f)` with `f`
reading `carried`) is legal under [04-INF-4] and has no runtime cycle, but the
mirror and the read edge point both ways. The stall releases the function
first, its backward read reports unbound in Surf, and on stamped IR the
serialized-IR ingress accepts from the body stamp while the typed ingress
rejects. For a `defsig`-less function this is a genuine two-way inference
dependency that no edge choice can order; for a signed one the edge cannot be
dropped while a hole is quantified ([#1486]). PP6 decides all three: the
hole is never quantified and readers wait for its body ([04-INF-5]), so the
mirror edge returns to round 10's `defsig`-less form; the cycle is an eager
value cycle under [04-INF-7]; and a component the reference graph still
closes is inferred as one group so the rejection is identical at both
ingresses.
Until PP6 lands, the schedule stays total by releasing the hoist-order-least
remaining vertex, so a callee is still inferred before its caller.

Two properties follow and are the reason this design is trusted where the
chained ones were not. First, the hoist order is itself a linear extension of
every edge except a barrier into a hoisted function, so Kahn's algorithm with
the hoist order as its priority returns the hoist order on every program that
carries no such barrier, up to the contiguity of a contracted recursive
component, which is invisible past `primary_inference_groups`: a program the
blunt hoist scheduled correctly is grouped identically, and only programs it
mis-scheduled change.
Second, because every edge is a reference, the stall path is reachable only
from a reference cycle through an eager value: a runtime initialization cycle
the detector rejects, a runtime cycle through a lambda applied during the
value's initialization that the detector does not yet see ([#1487]), or the
[#1485] shape above.

The authoritative oracle for the schedule asserts these invariants directly on
the returned order, not on accept/reject verdicts:

```sh
cargo nextest run -p chelis-types --lib infer::tests::schedule_invariants
```

For generated programs whose reference structure is declared by the generator
rather than recovered by the collectors, it checks that the schedule is a
permutation with every component contiguous; that every declared edge is
respected when the declared graph is acyclic; that the schedule equals the
hoist order whenever the hoist order already respects every edge; and that a
cyclic program still schedules every item once with callees before callers.
Named regressions carry the shapes rounds 5 through 8 found: the hoisted
reader, the straddled recursive component wrapped, bare and mixed, the value
that reads a `defsig`-less caller, the planner-order break, and the program a
textual chain would have stalled; the [#1485] shape is pinned as a stall whose
reference graph must stay cyclic, while the parity suite and the CLI oracle
pin its verdict and are the ratchets that redden when it closes. Each of the
mutations that reproduces
one of the abandoned repairs reddens the oracle: restoring the blunt hoist,
dropping the contraction, dropping the read edge, dropping the mirror,
reinstating either chain, releasing a stall by lowest ordinal, and restoring
the identity short-circuit for bare units.

The verdict oracle for the atom is:

```sh
cargo nextest run -p chelis-types --test issue_1134_forward_reference_parity --no-fail-fast
```

Its core is a set of generated ordering matrices rather than a list of
examples. A hand-written case fixes one declaration layout, and layout is
precisely what this rule interacts with, so the matrices enumerate layouts
instead. The first enumerates every ordering of an anchor function, the eager
value, and a reader; a function reader and a value reader; the annotated,
unannotated, adjacent-`defsig` and separated-`defsig` spellings; and both
module-wrapped and bare declarations. The second enumerates the
recursive-component interaction the first cannot reach: it orders a mutually
recursive pair around the value they straddle, in a type-stamped spelling and
a header-less one, wrapped and bare. The third pins function visibility in
both directions against the blunt hoist's region. Each row's expected verdict
is derived from [04-INF-4], every row must pass, and both ingresses must
produce identical ordered diagnostics. The named regressions beside them pin
the external-input spellings, a malformed external-input type, bare-self
rejection, sequential-`let` shadowing, source-ordered value-only diagnostics,
missing names, local forward references, and eager value cycles.
`issue_1134_forward_reference_cli` carries the interleaved and straddled cases
at the public `chelis check`, `eval`, and `build` surfaces, and the existing
signature-inference and chelis#1124 suites are supporting regressions. The
[#1485] shape is recorded as a ratchet in the parity suite and the CLI oracle:
its Surf spellings must still reject identically and its stamped spelling must
still split, so the residual cannot grow silently and both tests redden when
it closes. The reason the mirror edge stays unconditional has its own direct
regressions beside the ratchet: a partial and a generic header read by a value
the schedule defers the function past must still reject at both ingresses and
at `chelis check`. Two adjacent pre-existing defects the rounds surfaced are
tracked beside it and not absorbed here: a partial or generic header
instantiated by a reader below the hoist floor ([#1486]), and the eager-cycle
detector's blind spot for a lambda applied during a value's initialization
([#1487]). This
slice does not absorb [#1125]'s reader audit, [#874]/[#887]'s tag-keyed
vacuity, or [#1076]/[#672]'s independently owned name-precedence work.

### Adjacent ledger rows delivered with the class change

- **[#850], checker half.** `defsig` is now a same-unit annotation for a
  same-name `def`, as required by `spec/03-deep-syntax.md` §2.2. Declaration
  collection rejects an orphan before persistent library context can make it
  look backed. Reef applies the same pair check to every authored source
  module and synthetic entry before linking; only a provenance-marked
  dependency-shell interface may then carry signature-only internal rows.
  The §C4.4 corpus contains the declaration-only negative row and a paired
  positive. A repository sweep found three stdlib files relying on
  signature-only placeholders (Parquet, SafeTensors, and Xavier); each now has
  an explicit fail-loud Chelis body. The native-emission half remains [#730]
  work and is not claimed here.
- **[#851].** Match-arm pattern binders are removed from eager-reference
  collection for both the guard and body. The exhaustive binder walk has one
  shared home in `chelis-deep` and is consumed by authoring, macros, and the
  checker. Reef's Surf `Pattern` walk remains separate because it operates on
  a different typed AST, and the source records that boundary. This is the
  missed-migration case the roadmap predicted, repaired at the policy owner
  rather than with a fourth walk.
- **[#1131].** `spec/03` §6.4 and [04-LIT-1] define the closed atom/primitive
  matrix. The checker enforces every cross-family scalar pairing after alias
  resolution. Deep and Surf producers use the one typed exception for an
  integer-spelled float: an exact Int payload plus `literal_source: integer`.
  That form preserves [04-NUM-14]'s direct target-width rounding instead of
  manufacturing a double-rounding route through f64 in both the tensor/DAG
  and scalar host-eval constructors. The full positive,
  malformed-marker, cross-family, and producer paths are regression-tested;
  the former score-1 input is also in §C4.4.
- **[#1355].** `diagonal` now declares [05-OP-33]'s smaller selected extent
  wherever that minimum is statically known, instead of the wildcard that
  unified with any declared return type and let `def f(x: tensor[3, 4, f32])
  -> tensor[4, f32] = diagonal(x, 0, 1)` score 1.0. Two literal extents give the
  smaller value; two occurrences of one named extent give that name, because
  spec/04 §4.1 makes two `d-name` unify only when equal, so both axes denote a
  single runtime value. The wildcard survives for distinct names, a mixed
  literal/symbolic pair, a dimension variable, and a rank spread, where the
  minimum genuinely is not known at check time. The evidence is the non-square
  rank-2 and rank-3 f32 forms plus the identical-name form in
  `crates/chelis-types/tests/issue_1355_diagonal_extent.rs` and
  `crates/chelis-cli/tests/issue_1355_diagonal_extent_cli.rs`, whose five
  rejections were proved red on the pre-fix tree. Two boundaries are recorded
  there rather than moved: `trace` routes through `infer_trace_result_type`,
  which removes both selected axes and never reads their extents; and a declared
  literal extent is still admitted against a named one, because `unify_dim`
  accepts `Name` against `Lit` under [#219]'s Option A, which is
  dimension-unification policy rather than diagonal's extent rule.
- **[#1494].** A literal pattern is now a typing constraint on the scrutinee.
  `pattern_bindings` did nothing at `pat-lit`, so an `f32` pattern against an
  `int32` scrutinee scored 1.0 and `chelis eval` printed a result from an arm
  that can never match. The numbered spec had not decided the rule: §3's Match
  rule never defined `bindings` for `pat-lit`, and [04-LIT-1]'s closed
  atom-to-primitive matrix is scoped to a literal's declared `lit` metadata,
  which a `pat-lit` structurally cannot carry. [04-PAT-1] and a `bindings`
  definition at the Match rule were authored first, then implemented. The rule
  is family agreement rather than unification, because a `pat-lit` admits no
  suffix: unifying with §5.3's `int32` default would reject a `| 1 =>` arm over
  an `int64` scrutinee and leave no spelling for an `int64` literal pattern.
  Two sub-clauses are errors because each arm is provably dead: a non-primitive
  scrutinee admits no literal pattern, and an integer pattern outside the
  scrutinee width's range is rejected under §5.3's and §5.6's range rule. The
  check sits in the one recursive walk both ingresses share, so the tuple,
  record, and constructor nestings are the same finding at depth, and the tests
  assert ingress parity rather than assuming it. The evidence is
  `crates/chelis-types/tests/issue_1494_literal_pattern_scrutinee.rs` and
  `crates/chelis-cli/tests/issue_1494_literal_pattern_cli.rs`, whose rejections
  were proved red on the pre-fix tree, with the score-1 inputs also in §C4.4 as
  the float-versus-`int32` and out-of-range-`int8` members beside a
  matching-family positive control. One boundary is recorded rather than moved:
  a `pat-lit` whose child is not a scalar atom still scores 1.0, because that is
  Deep well-formedness rather than typing, and it is tracked as [#1525].

---

# Part IV - bookkeeping

## I1. Interlocks

- **With [#730] (`loud_unsupported.md`)**: `EffectKind` is delivered there
  (Phase 2) and consumed here (§C1.5) - if this plan's Phase 1 lands
  first, it matches the two kind strings with a loud else and migrates to
  the enum when available; the bogus-effect `.dp` repro must be rejected
  by whichever side lands first, and BOTH once both land (checker: unknown
  kind is `MalformedForm`; lowering: its catch-all raises). `DeepTag` remains
  owned by this plan and its added-variant compile oracle.
- **With [#729] (`dtype_semantics.md`)**: none structural. Phase 4's
  capability table derives op x dtype acceptance; this plan governs
  CONSTRUCT-level totality. The two meet only in that both make `chelis
  check`'s score-1 claim honest.
- **With [#908] (`unrepresentable_ast_domain.md`)**: this plan owns the
  decode-once and exhaustive-consumer strength; [#908] owns making invalid AST
  states unrepresentable by changing the carrier. `Node` is the accepted
  successor only at §C4.2's four-part ingress/validator/disposition/deletion
  boundary. Neither plan may declare the other complete from a bridge state.
- **With [#1261]/[#730] (`chelis_native_testing_plan.md`)**: PR [#1273]
  delivered the declared/imported collision guard and visible fallback. PP4
  preserves both and closes only the raw-flat-scope residue that PR explicitly
  left to [#1264]/[#731]. The already-closed issue is context, not a second
  closing claim.
- **With [#1024]**: [#1258] is the resugaring half of [#1247]. PR [#1406]
  closes both with one structural dimension-literal representation; it does
  not absorb unrelated canonical-Surf or total-resugaring instances.
- **With [#721]**: none (eval ingestion, no checker code); listed so nobody
  searches for it here.

## Issue map

| delivery | goes green / becomes unwritable |
|---|---|
| 0 | detection; the invariant exists |
| 1 | [#709] (all three escalations), [#710]'s silent half |
| 2 | the future supply of silent exemptions (type-state) |
| 3 | the future supply of undecided TAGS (compile-time totality) |
| PP1 | [#780]/[#783]'s deferred parameter checks; [#847]'s rigid control |
| PP2 | [#1147] and the future supply of registered builtins with no inference disposition |
| PP3 | [#1209]/[#1211]/[#1212]'s name-keyed binding-identity channel |
| PP4 | [#1264] and [#1261]'s raw-flat-test-scope residue; exact module scope in every checker/test entry |
| PP5 (partial) | [#668]; the checker derives a rank fact for `ShapeClass::Identity` plus `conv2d`/`stride`/`expand`/`softmax` and rejects a positive-rank disagreement where it has one, on unforgeable rank-only facts; the tensor-DAG C emitter aborts on a positive-rank operand disagreement, and under [#1484] so does the host-value emitter for its six binary elementwise builtins. No claim is made for operations outside that set, nor for any checker verdict on the host-lane programs; see PP5 |
| [#1247] residue | integer nominal arguments are kind-checked and concrete dimensions constrain every checker/test/compiler lane; [#1258] round trips the same representation |
| [#1125] nominal-rank ingress residual | ordinary `.dp` ingress, `surf`, and `validate --deep` reject `d-rank` in nominal argument slots while preserving legal dimension arguments and tensor rank spreads; the broader reader-audit/lint issue remains open |
| PP6 (decided, not delivered) | [#1486] (a hole is never quantified and no reference observes it before the body; an authored binder is rigid), [#1487] (lambda bodies and applied values are eager references), [#1485] (every reference-graph component is inferred as one group; the three spellings reject as `CycleDetected` identically at both ingresses); [#1134] closes when all three are dispositioned as PP6 states |
| [#1134] forward-reference residual | both checker ingresses reject eager forward values, accept backward values from value initializers and from function bodies wherever the schedule places them except the [#1485] shape, accept declaration-local explicitly typed external inputs, retain sequential local scope, and reject bare self-reference/eager value cycles identically; the schedule's order invariants are asserted directly |

## Decisions and remaining questions

| # | question | decided in | recorded where |
|---|---|---|---|
| 1 | `handle-effect`'s checked signature details | DECIDED 2026-07-17 (revised same day, explicit over implicit: this code is agent-written, so there is no ergonomic case for contextual binding). Phase 1 checks FORM, [#735] authors meaning. Seed = an EXPLICITLY int64-suffixed integer literal (`42i64`, spec/02 §P10a); an unsuffixed literal is a type error whose diagnostic names the requirement and the suffix spelling; non-literal seed expressions are rejected, diagnostic citing §P5's shipped constraint and [#735]. Device = a string literal; the checker validates literal-ness only, never the device-name vocabulary (target knowledge, [#735]'s territory). No spec/02 §P10 change needed - the width is visible in the source itself. Existing `with seed(n)` fixtures/examples migrate to the suffixed form in P1's change set (Public-Surface Change Rule) | §C1.5 + spec/04 effect section |
| 2 | typecheck-cache deserialization as a witness mint (accepted, or cache entries re-validated?) | Phase 2 | §C3 note + the cache module doc |
| 3 | whether printers/desugar also migrate to `DeepTag` (nice-to-have; they are not chokepoints) | DECIDED 2026-07-23: deferred; REVERSED 2026-07-24 by the decode-once rework directive - printers, desugar, and every other producer/consumer migrated; no string-keyed tag idiom survives outside the parse/serialize boundary | this doc |
| 4 | score semantics for `UnknownForm`/`MalformedForm` | DECIDED 2026-07-17: severity parity with `TypeMismatch` (the existing 0.5-class precedent), no new weight class. The invariant that matters - any pushed error forces score < 1.0 - is locked by §C4.4's corpus independently of the weights, so calibration can move later without touching it | scoring code + this doc |
| 5 | how a lambda-bound or function-valued parameter's type binds, and which checks re-run once it is bound | DECIDED 2026-08-04: shape-constrained lambdas with an unknown outer parameter constructor are monomorphic bind-on-first-use within their enclosing declaration, replay the ordinary semantic rule, and reject unresolved at that declaration's own boundary; a result annotation or later top-level caller does not bind them; symbolic declared tensors remain polymorphic and rigid dimensions remain distinct absent a real equality constraint | [04-INF-1] + PP1 |
| 6 | whether two closures may each consume one underlying value through two user-visible names (`y = x`, one capture per name), or capture forwards through the alias chain generally | DECIDED 2026-08-21: preserved and made normative. A capture consumes the binding it names; distinct user-visible bindings of one value are distinct for capture; only a destructured component (or an alias of one) forwards to its carrier. Nautilus `lu_solve` and coral depend on the spelling; the reviewer guidance on [#1209] was to specify the choice explicitly and keep any tightening separate | [04-LIN-2] + PP3 |
| 7 | whether one linked module's uniquely matching terminal name or one batched test file's declaration can confer unimported scope on another file | DECIDED 2026-08-31: no. Value lookup is exact-only after reef rewriting; batch entries are independently module-rewritten before combination; terminal matching is diagnostic-only | spec/02 P2 + [04-FIT-2] + PP4 |
| 8 | whether an unannotated nominal parameter is a type, a dimension, or contextually reinterpreted per application | DECIDED 2026-08-31: one checker-owned header kind is fixed before body resolution. Dimension-only evidence selects `Dimension`; mixed use rejects; unused defaults to `Type`; transitive nominal uses propagate by least fixed point | [04-ADT-3]/[04-ADT-4] + [#1247] residue |
| 9 | whether a top-level eager value may refer to a later value, and whether the two checker ingresses may differ | DECIDED 2026-09-01: no. Both ingresses reject a later eager value as unbound; serialized body metadata cannot create scope. Scope is read from declaration position, never from a binding timeline the inference schedule advances, and the schedule infers an eager value before any function that legally reads it, using only reference edges over the hoist order so a program without such a read keeps its previous grouped order. Only an explicitly typed self-reference receives a declaration-local external-input type; bare self-reference remains an eager cycle. Function inference groups remain separately governed by [04-INF-2]/[04-INF-3] | [04-INF-4] + [#1134] residue |
| 10 | whether a wildcard slot in a signature is a polymorphic binder, and what a reference sees before the declaration's body is inferred | DECIDED 2026-09-03: a hole, never quantified; every reference is typed at the body-determined signature wherever it sits, so readers of a hole-signature function are scheduled after its body in every region. A shared monomorphic hole was rejected because it makes a partial header monomorphic in its own dimension binders | [04-INF-5] + PP6 |
| 11 | whether a body may narrow an authored type binder | DECIDED 2026-09-03: no. Explicit and implicit binders are rigid in the body, as dimension parameters already are under §4.4; the scheme is the declared signature. Ten stdlib declarations that narrow a bounded binder with an unsuffixed literal migrate to `cast(literal, p)` | [04-INF-6] + PP6 |
| 12 | which references inside a top-level value's initializer are eager for cycle detection | DECIDED 2026-09-03: all of them, lambda bodies included, transitively through every referenced top-level declaration, with an applied value required like a read one. The argument-position refinement was rejected as unsound for stored and returned closures; the over-rejection is accepted and is already the detector's treatment of a bare function reference | [04-INF-7] + PP6 |

## Contract summary

The checker rejects unsupported or malformed constructs loudly.
`Type::Error` without an authoritative pushed diagnostic is unconstructible,
and the always-on finalizer enforces that successful checked output contains
neither an error type nor a missing authoritative owner stamp. Phase 3 makes a
Deep tag without a checker disposition uncompilable through exhaustive
`DeepTag` matching. [#908] may replace Phase 3's physical tag carrier, but its
successor must retain the same decode-once, exhaustive-disposition, and
continuous-oracle guarantees. PP4 additionally makes module scope exact at
both package and batched-test boundaries: a foreign terminal-name match is
never a binding, and the fitness report cannot describe an unresolved value or
constructor as fully resolved. PP5 preserves declared parameter ranks and
rank-only movement facts through the validator, on a carrier source cannot
forge, so an elementwise rank mismatch the resolver derives a fact for does not
receive a perfect checker verdict. Where the resolver derives no fact the
checker raises nothing; the generated C aborts in both emitter lanes, the
tensor-DAG one under [#668] and the host-value one under [#1484], but neither
abort makes such a program checkable. PP5 states which results are
demonstrated and which are not.
The separately owned [#1247] residue applies the same honesty rule to
nominal arguments: integer syntax is either an exact checked dimension or a
kind error, never an inference wildcard, and every downstream checker/test
signal preserves that decision. The bounded [#1125] ingress residual applies
the corresponding structural grammar at ordinary `.dp` doors: a nominal
argument is a type or non-rank dimension, never a rank spread.
The [#1134] residual applies the same ingress-honesty rule to top-level eager
value scope: textual order, never compiler metadata availability and never the
inference schedule's position, decides which other values are visible.
Explicitly typed self-reference retains its declaration-local external-input
meaning, while bare self-reference, local forward bindings, and eager value
cycles remain errors.

[#696]: https://github.com/Chelis-Lang/chelis/pull/696
[#703]: https://github.com/Chelis-Lang/chelis/issues/703
[#668]: https://github.com/Chelis-Lang/chelis/issues/668
[#709]: https://github.com/Chelis-Lang/chelis/issues/709
[#710]: https://github.com/Chelis-Lang/chelis/issues/710
[#721]: https://github.com/Chelis-Lang/chelis/issues/721
[#735]: https://github.com/Chelis-Lang/chelis/issues/735
[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#731]: https://github.com/Chelis-Lang/chelis/issues/731
[#755]: https://github.com/Chelis-Lang/chelis/issues/755
[#756]: https://github.com/Chelis-Lang/chelis/issues/756
[#780]: https://github.com/Chelis-Lang/chelis/issues/780
[#783]: https://github.com/Chelis-Lang/chelis/issues/783
[#833]: https://github.com/Chelis-Lang/chelis/issues/833
[#847]: https://github.com/Chelis-Lang/chelis/issues/847
[#850]: https://github.com/Chelis-Lang/chelis/issues/850
[#851]: https://github.com/Chelis-Lang/chelis/issues/851
[#858]: https://github.com/Chelis-Lang/chelis/issues/858
[#859]: https://github.com/Chelis-Lang/chelis/issues/859
[#874]: https://github.com/Chelis-Lang/chelis/issues/874
[#908]: https://github.com/Chelis-Lang/chelis/issues/908
[#1023]: https://github.com/Chelis-Lang/chelis/issues/1023
[#1029]: https://github.com/Chelis-Lang/chelis/issues/1029
[#1082]: https://github.com/Chelis-Lang/chelis/issues/1082
[#1088]: https://github.com/Chelis-Lang/chelis/issues/1088
[#1131]: https://github.com/Chelis-Lang/chelis/issues/1131
[#1147]: https://github.com/Chelis-Lang/chelis/issues/1147
[#1208]: https://github.com/Chelis-Lang/chelis/pull/1208
[#1209]: https://github.com/Chelis-Lang/chelis/issues/1209
[#1211]: https://github.com/Chelis-Lang/chelis/issues/1211
[#1212]: https://github.com/Chelis-Lang/chelis/issues/1212
[#1261]: https://github.com/Chelis-Lang/chelis/issues/1261
[#1264]: https://github.com/Chelis-Lang/chelis/issues/1264
[#1273]: https://github.com/Chelis-Lang/chelis/pull/1273
[#1402]: https://github.com/Chelis-Lang/chelis/pull/1402
[#1258]: https://github.com/Chelis-Lang/chelis/issues/1258
[#1406]: https://github.com/Chelis-Lang/chelis/pull/1406
[#1076]: https://github.com/Chelis-Lang/chelis/issues/1076
[#672]: https://github.com/Chelis-Lang/chelis/issues/672
[#1247]: https://github.com/Chelis-Lang/chelis/issues/1247
[#1125]: https://github.com/Chelis-Lang/chelis/issues/1125
[#1134]: https://github.com/Chelis-Lang/chelis/issues/1134
[#887]: https://github.com/Chelis-Lang/chelis/issues/887
[#1485]: https://github.com/Chelis-Lang/chelis/issues/1485
[#1486]: https://github.com/Chelis-Lang/chelis/issues/1486
[#1487]: https://github.com/Chelis-Lang/chelis/issues/1487
[#1355]: https://github.com/Chelis-Lang/chelis/issues/1355
[#219]: https://github.com/Chelis-Lang/chelis/issues/219
[#1484]: https://github.com/Chelis-Lang/chelis/issues/1484
[#1494]: https://github.com/Chelis-Lang/chelis/issues/1494
[#1525]: https://github.com/Chelis-Lang/chelis/issues/1525
[#1457]: https://github.com/Chelis-Lang/chelis/pull/1457
[#1512]: https://github.com/Chelis-Lang/chelis/issues/1512
[#1339]: https://github.com/Chelis-Lang/chelis/issues/1339
[#1277]: https://github.com/Chelis-Lang/chelis/issues/1277
