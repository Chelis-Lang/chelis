# Checker Totality: every construct is checked or loudly rejected

**Status:** Phases 0-3 and PP1-PP4 are delivered. PP5 is partial. Its
completion design (2026-09-03, in that section) retires the rank side channel
in favour of unification and routes the residue to [#597], [#1512], and
[#1506] under the single-meaning `expand` rule of [#1532]. D8 PR A has landed
the checker half: the side channel is deleted, unification is the one rank
authority, and both ingresses agree on every program the RANK oracle
(`issue_668_rank_agreement_is_unification`) covers. PP9 extends that agreement
to the decided semantic pass set: every public entry now runs one ordered
semantic validation protocol, and the PP9 oracle checks exact ordered
diagnostics across both admitted carriers. Backend-only refusals removed from
that protocol remain owned by [#730].
D8 PR B has landed the comparison surface: the seven identities refuse a
scalar beside a tensor under `[05-OP-36]`, and the diagnostic names the
explicit replacement. The C emitter guards of R3 and R4 stand. PP5 stays
partial until [#597] and [#1512] close with their owners, and until [#1619]
lets the symbolic-size replacement execute on the C lane. PP6 is delivered by Slice A and Slices B/C; its shared reference graph,
schedule, paired-ingress, and public CLI oracles are green. PR [#1406] delivered the
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
   `Result<ResolvedDeepType, ErrorWitness>`. Known constructors check every
   sibling type component before failure. A rejected signature retains a
   private frame with its valid constraints and binder metadata. Error slots
   carry existing witnesses, not invented variables or dimensions. The body
   still receives its checks, but public name lookup retains the failure.

   Surf §P4 and §5.2 give each ordinary inline parameter type one owner in
   its generated signature. The `fn` parameter carries an inference hole,
   which retains annotation presence without a duplicate type constraint.
   The parameter resolver returns either a resolved constraint or a hole.
   Signature-directed inference supplies each hole's declared slot before
   the body receives its checks. A partial rejected frame supplies its valid
   constraints and existing error witnesses through the same path.

   Standalone signatures and independently authored parameter annotations
   remain independent, even when their source ranges or display labels
   coincide (chelis#1527). No diagnostic strings, spans, private source tokens,
   or post-inference deduplication decide ownership. Property quantifiers retain
   the typed copies required by the Deep metadata contract. The checker treats
   those parameter annotations as generated copies only after their complete
   Deep type syntax matches the adjacent `defsig` slot at the same canonical
   parameter position through the Deep metadata layer's canonical semantic
   view. Ownership is position-local: a mismatch, missing parameter, malformed
   annotation, or extra parameter remains independently checked without
   revoking a verified neighboring copy. A malformed outer parameter carrier
   remains wholly unverified and fails closed. The semantic comparison recursively
   erases AST and metadata-token spans plus the source-only `span`, `span_*`,
   `loc`, and `source` metadata namespaces at enclosing metadata maps; the same
   spellings inside extension-data maps or preserved payloads remain opaque
   data. It retains every other metadata key and payload, accepts the stamped
   `Node` and exact transitional `List` carriers, and rejects malformed or
   non-type carriers. A disagreement remains an independently checked
   annotation constraint. For each verified slot, the `defsig` is the one
   semantic diagnostic owner. The canonical form survives
   Deep print/parse and direct `chelis_surf::TypeExpr` Serde.
   The compiler API's separate structured `surf_ast` wire continues to carry a
   string precision field. Surf §0.1 and Deep §6.3.2 own the corresponding
   resugar and normalization contracts.

   Recovery never turns an invalid type into a successful `ResolvedDeepType`.
   The private frames do not enter serialized environments. No reported
   diagnostic is deleted or deduplicated.
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
   i64-SUFFIXED signed integer literal seed - `42i64` or `-1i64` per spec/02 §P5/§P10a; an
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
   Signed seed admission follows [05-RNG-1]'s two's-complement bits,
   including `i64::MIN`. Surf's single unary-minus literal encoding and
   equivalent Deep literals use the same typed static evaluator as lowering.
   The source form check still rejects casts, nested arithmetic, runtime
   variables, and a lexically shadowed `neg` callable. The executable regression
   is `cargo nextest run -p chelis-cli --test issue_1803_constant_signed_seed`:
   formatted Surf/Deep checks, exact first/next uniform draw bits in eval and
   compiled C, and rejection controls. This is source admission, not expanded
   native Dropout or runtime-seed support. Uniform retains its legacy source-word
   algorithm; its exact-bit controls prove seed transport, not adoption of the
   full [05-RNG-1] algorithm. The same suite checks canonical Dropout evaluator
   masks and the next draw for signed seeds.
   The three executed escalations become impossible: an i64 body in an
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
successful result. Every cache envelope verifies format and build identity and
its own byte integrity before decoding. What happens after that decode differs
by route, and the difference is not a preference (chelis#2211).

Every **on-disk cache entry** reruns the effect and linearity checkers over the
decoded program, because the entry outlives the process that wrote it and there
is no second channel on which its producer could have said what the bytes ought
to be. How much more each route does varies with what its wire carries, and the
three are not interchangeable:

- `CompiledContext::load_if_fresh` also re-lowers and compares the result
  against the transmitted lowered payload, since its wire always carries one.
- `StdLibContext` carries an optional lowered payload and performs that
  comparison only when one is present.
- `LibraryContext` carries none (`LibraryContextWire`, `library_cache.rs`), so
  the checker rerun and the type-environment agreement are its whole
  post-decode check.

Where the comparison does run, it catches a payload whose parts stopped
agreeing with each other, and the envelope's embedded digest cannot stand in
for it, because whoever rewrote the payload recomputed that digest;
`cache_wire_compatibility.rs`'s
`cache_reconstruction_rejects_changed_numeric_bits_after_checksum_recomputed`
is the control that demonstrates exactly that. What the rerun and the
comparison cannot do is establish provenance. A *substituted* payload -- a
different library, compiled by the same build, carrying the victim's
`source_hash` and `identity` -- is internally consistent by construction, so
it passes every one of these checks (chelis#2257).

A **parent-to-worker handoff** -- `CompiledContext::encode_for_handoff` and
`decode_authenticated`, which `chelis test` uses -- does have a second channel.
The parent writes the bytes to a tempfile and passes their digest in the
worker's environment, so the worker can establish that the bytes on disk are
the ones the parent wrote, and does not rerun the checkers. That is a different
question from the one the comparison answers, and a strictly harder one to
defeat: the digest rejects every payload but the parent's, the substituted
well-formed one included, which is the case the re-derivation accepts. It was
executed rather than argued, in the chelis#2258 review: a library compiled from
rewritten sources at the same package root was accepted by `decode` and refused
by `decode_authenticated`.

Note what the channel does not claim. A same-user process can read another's
environment, so this is not same-user isolation and is not attempting it;
anyone who can set the worker's environment or replace the `chelis` binary
already runs chosen code as this user. What the digest defends is the gap the
file's `0600` mode leaves open, which is a `TMPDIR` other users can write to,
the ordinary shape of a shared build machine.

Skipping the reruns is sound only while rerunning them reproduces the program
they were handed. `chelis-compiler-api`'s
`compiled_context_authenticated_handoff.rs` reconstructs one payload through
both routes and requires byte-identical results. Other tests assert that fixed
point for the *untrusted* route -- `cache_wire_compatibility.rs`'s
`current_compiled_disk_and_worker_preserve_scalar_storage_bits_and_reconstruct`
compares `decode`'s output against the producer's -- so a normalizing pass in
either checker would be caught somewhere regardless. What is specific to this
test is the authenticated route: every other use of `decode_authenticated` in
the tree is a negative one, so this is the only place that compares what that
route produces against anything.

Until chelis#2211 this section read "deserialization does not rerun semantic
checking; cache bytes are a trusted internal artifact", while both routes were
in fact rerunning the checkers. The doc was wrong in the permissive direction:
read as permission, it would have licensed removing the disk route's
re-derivation on the strength of a sentence, which is the escalation
`AGENTS.md` warns about under "permission-to-mandate". The disk route keeps its
re-derivation for the reason above, not because no document said to remove it.

This boundary is documented in the witness, type-context, and compiler-api
cache module docs.

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
  complete explicit `defsig` binder lists, and trusted compiler-generated
  metadata. Only actual binders or explicitly legal inference holes mint
  type/dimension/rank variables;
- declaration headers are precollected before bodies, preserving legal self
  and forward ADT/alias references while rejecting unknown names and wrong
  arities before a context can be cached. A wrong nominal arity retains one
  arity witness but does not return before recursively resolving every supplied
  argument whose position has a header-defined type or dimension kind. Missing
  positions and surplus positions beyond the header invent no child role. That
  explicit header environment is
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

   **Scope note (2026-07-24, corrected 2026-09-03): this invariant
   quantifies over the checked result, not over the submitted program, so a
   node inference never visited satisfies it vacuously.** The original note
   attributed that gap entirely to tag-keying. Tag-keying is one route and
   it is real: a node with no recognized tag is exempt from the stamp
   requirement (`requires_stamp = tag.is_some_and(...)`,
   `infer/validate.rs`), from child ownership classification (the `None`
   arm of the `child_stamp_role` match, which recurses without
   registering), and from owner registration
   (`register_annotation_owners`'s `if let Some(tag)`). Phase 3 closed the
   known *top-level* path - a loud `UnknownForm` where `infer_top_level`
   used to skip silently, with rejection parity on both `.dp` validator
   surfaces - without removing the exemption itself; nested bare lists stay
   legal by design (empty guards, `loc` metadata values), so the untagged
   arm remains reachable below the top level.

   Tag-keying is not the only route, and PP8 records the ones that are
   live. **The walk also skips whole child slots by ROLE.** Both halves of
   the walk recurse only into `RuntimeExpr`, `ExplicitInferenceBypass`, and
   untagged children; the `Syntax`, `Selector`, `EffectHandler`, `Binder`,
   and `Type` arms return without descending. A node in one of those slots
   is therefore exempt whether or not it carries a tag, and a `var` node is
   doubly exempt because `should_attach_type_metadata(Var)` is `false`, so
   even a walk that reached it would require nothing of it. The
   complementary obligation - every node the submitted program's forms
   semantically read is consumed or diagnosed - is [04-TOT-4] and is not
   part of §C4.1. `DeepTag` exhaustiveness (Phase 3) does not close it
   either: exhausting the tag of the PARENT says nothing about whether the
   parent's disposition read the child.
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
and applicable CI checks on the candidate head are required supporting evidence,
but neither replaces this oracle. `scripts/gate.py --local` is an optional local
reproduction of supporting checks.

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
`integration` support stage (`heavy-e2e.yml` nightly/manual
`integration-support` worker) and by its `--local` pre-push subset. It executes
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

#### General operation-admission contracts

The language decision is owned by
[`spec/04-type-system.md` §3.1.5, [04-INF-9]](../04-type-system.md#315-explicit-generic-operation-contracts),
with dtype-family syntax and transport in §5.9. It is a general checking
contract, not a runtime-extents exception. PP1's local monomorphic replay
remains useful for inference holes; it does not authorize publishing inferred
generic admission requirements.

The choice favors a contract whose admitted inputs can be read independently
of its implementation. Inferring requirements for an omitted signature would
accept more concise generic wrappers, but would make editing a body change
its callable contract and give annotated and unannotated definitions different
admission policies. The selected rule keeps ordinary local inference,
unconstrained polymorphism, and transport of already-checked values without
that second policy.

This decision revised the inferred-collection exception proposed in
[PR #2038](https://github.com/Chelis-Lang/chelis/pull/2038) for
[#1654](https://github.com/Chelis-Lang/chelis/issues/1654). Its dtype-family
half was implemented by
[PR #2071](https://github.com/Chelis-Lang/chelis/pull/2071) for
[#1942](https://github.com/Chelis-Lang/chelis/issues/1942), including omitted
signatures and escaping anonymous functions rather than only authored named
dtype binders. An explicit `List[a]` parameter already supplies a collection
contract for a generic list wrapper. Syntax for one authored generic
abstraction spanning several collection constructors is not decided here:
that needs a separate normative decision, not an inferred-contract exception
or an invented bound spelling.

Implementation follows the declaration contract:

1. Preserve authored binders and their declared restrictions as givens before
   checking the body. Each operation checks its requirements against those
   givens during ordinary checking; failure reports at the declaration.
2. Keep genuinely local inference obligations monomorphic and replay the
   ordinary operation checker when their operands bind. At the enclosing
   declaration boundary, reject unresolved admission obligations rather than
   generalizing them into new qualified schemes. Apply this check to escaping
   anonymous functions as well as named definitions.
3. Transport checked restrictions through the shared scheme-instantiation,
   unification, generalization, function-value and checked-context paths.
   Distinguish transporting a builtin or alias from synthesizing a new
   contract for a wrapper. Unary dtype-family membership and collection
   operand/result relations need not use identical payload representations.
4. Retire call-site callee-body walking as an admission validator. Direct,
   indirect, and transitive calls consume the same checked function contract;
   none depends on the callee body being inspectable.

Serialized checked contexts preserve every checked restriction they support.
#2071 coordinated the `TypeEnv` and CHB format identities, payloads and
regenerated artifacts for its delivered dtype-family layout. #2038 owns the
collection-constraint payload and corresponding round trips. Any later layout
change must allocate its own coordinated format identities and regenerate the
artifacts; reusing one version for different layouts is not integration.

The acceptance matrix below is the combined permanent coverage obligation.
#2071 delivered the dtype-family and shared declaration-boundary rows through
the ordinary checker and public checking entry points. The collection-specific
rows remain with #2038/#1654. None establishes a runtime-extent execution
claim.

| Contract boundary | Accepted control | Required rejection |
|---|---|---|
| Authored dtype binder | Float-bound wrapper around a float-only operation | Unbounded or Numeric-bound wrapper, even with only float callers |
| Omitted signature | Unconstrained identity | Newly inferred constrained generic `def size(x) = len(x)` |
| Collection constructor | `List[a]` length wrapper at independent element types | Arbitrary `a -> i64` length signature, or scalar argument to the list wrapper |
| Local inference | Unannotated lambda bound to a valid concrete operand within its enclosing declaration | Invalid first binding, or unresolved obligation at that boundary |
| Function values | Alias or aggregate field retains an existing checked contract | Newly authored generic wrapper or escaping lambda needs undeclared requirements |
| Higher-order and transitive calls | Valid instantiation through a checked function parameter or wrapper chain | Inadmissible instantiation on each same route without a callee-body lookup |
| Recursion | Declared requirements preserved under permitted recursive instantiation | Missing requirements in a recursive member; incompatible family intersection |
| Checked contexts | Both restriction kinds survive supported round trips | Unsupported format rejected; restored function cannot admit an invalid operand |

Runtime-extents delivery retains its exact eval/C witnesses and class oracle;
those receipts supplement, rather than replace, declaration and contract-
transport tests. The delivered policy completes no runtime-extents phase or
implementation issue by itself.

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

**R1 is retired.** It described the shipped behaviour at `f5ec5ca63`, not the
contract, and D5 records why: the rank it derived for `expand`, the input rank
plus one, is a rank the language never assigns to `expand` under the
single-meaning rule of [#1532], and one the checker did not stamp at those
call sites either, so the rejection disagreed with the checker's own stamped
types in the same run. D8 PR A deleted the carrier, the derivation, and the
gate. What remains at every call site above is unification, which rejected the
same programs already; D8's measured outcome below records that the deletion
changed no verdict.

*Evidence:* `cargo nextest run -p chelis-cli --test
issue_668_elementwise_rank_honesty`, 18 tests after PR A, covering operand
order, nested inline identities, `floor_div`, discarded comparisons,
rank-zero, matching-rank controls, lexical shadowing, and the run-time
loudness of the `expand`-built reproducer. The negative also joins the
permanent #731 fitness-honesty corpus, whose
`issue_668_rank_divergent_elementwise` row spells the rank raise with `insert`
and is unchanged by PR A.

**R2. The shape facts the checker carries cannot be forged from source, and
are not inherited across a rebinding.**

R2's forgeability half is retired with R1. It held because rank-only facts
lived in a private Rust `ShapeTypeFact::RankOnly` carrier with no Deep-syntax
representation, so an authored dimension name could not alias validator state.
PR A deleted the carrier, so there is nothing left to forge, and what the row
now locks is the weaker property that an unusual `d-name` does not perturb the
exact-shape `conv2d` derivation.

The rest stands and is unchanged by PR A. The environment accepts exact facts
only from checker-owned lexical bindings, so raw `type` metadata on a runtime
expression is source syntax rather than validator evidence; and binding a name
to a value whose shape is not derivable clears that name's prior entry rather
than inheriting it.

*Evidence:* within the same file,
`an_authored_rank_only_prefix_dimension_is_an_ordinary_name`,
`authored_deep_type_metadata_cannot_override_the_inferred_rank`, and the
rebinding pair. The rebinding positive was confirmed to fail against the
pre-repair tree and pass after it; its negative companion pins the opposite
direction, so the repair cannot become "stop checking rebound names". Both
rebinding rows are disposition locks after PR A: their original consumer was
the rank comparison it deletes, and the environment they exercise now feeds
only the exact-shape validators.

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

The rank claim is now stated at the granularity unification proves, and the
"escapes" of the pre-PR-A text are not checker negatives.

Under the single-meaning rule an `expand` result is rank 1, which is the rank
both the checker stamps and the language assigns, so `add(s, e)` and every
program that launders `e` through `cast`, `cast_trunc`, `normalize`,
`realize`, or a user `def` is rank 1 beside rank 1 and WELL TYPED. Those
programs are positive controls, not gaps. Their loud outcome when the
operand's extent at the axis is not 1 is the `spec/05-risc-primitives.md`
section 2.4.1 `Domain` trap, which [#1277] S2b landed on the evaluator and C
lanes; `issue_668_elementwise_rank_honesty`'s
`the_expand_built_reproducer_is_loud_at_run_time` runs one such program on
both lanes and asserts the trap text. What closes for them is the rest of
[#597], not a checker rule. The `add(e, sum(e, 0i32))` and `add(e, mean(e,
0i32))` rows are [#1512]'s, for the same reason: their operand is a genuinely
unresolved reduction result, not an `expand` one.

The comparison rule is decided, and PR B delivers its concrete-dtype cases:
the scalar rewrite is gone, and `issue5_cmp_broadcast_both_forms` proves that
all seven identities reject the concrete scalar/tensor pairs its fixtures
spell, in both operand orders at both ingresses. Those rejections name
`[05-OP-36]` and the explicit replacement. This is partial enforcement of
[#1506]; the bounded-binder extension below is owned by [#1621].
`issue_668_rank_agreement_is_unification` asserts nothing about comparisons.

The execution-lane gap is separate from that checker residue. The replacement the
diagnostic names, `expand(to_tensor([1.5f32]), 0i32, shape(xs, 0i32))`,
executes on the evaluator and traps on C, because the unit-extent claim
attributes itself to the axis the `size` expression reads rather than to the
`expand` operand's. That is [#1619], owned by the [#1277] stream and
reproducible with PR B's own source files reverted. The literal-size spelling
executes on both lanes today. `issue_1506_replacement_spelling_on_the_lanes`
holds all three facts and flips when [#1619] closes.

The [#1621] extension preserves each declaration's instantiated type-binder
identity in scalar annotations and casts. Its existing dtype-family restriction
therefore continues to identify a scalar before operand unification. The mixed
surface diagnostic covers the seven comparison identities and the eight binary
arithmetic identities enumerated in `issue_1621_scalar_surface`, both operand
orders and both checker ingresses. Explicit scalar/tensor conversions preserve
the dtype variable; callers construct matching shapes with `scalar_to_tensor`
and `insert`. The original binder-literal precision assertions remain in the
CLI suite after its tensor fixture adopts that spelling.

The authoritative acceptance command for this extension is:

```sh
cargo nextest run -p chelis-types -p chelis-cli \
  --test issue_1621_scalar_surface \
  --test issue_1621_scalar_surface_cli \
  --test issue_1544_binder_cast_precision --no-fail-fast
```

The oracle bounds the claim to its generic operand, scope, diagnostic and
execution fixtures. Hosted CI and executable adversarial review supplement it;
it is not PP5 completion. [#1619]'s same-rank `expand` C defect and [cast-source issue #1489](https://github.com/Chelis-Lang/chelis/issues/1489)'s
unresolved cast-source obligations remain separate. The DAG arithmetic helper
retains compiler-generated rank-zero operands permitted by spec/05 §2.4.1;
that IR convention grants no source-level broadcasting permission.

What PR A does not claim: nothing here is an exhaustive statement about
elementwise operations. The oracle covers `add`, `mul`, `eq`, `max_elem`,
`where`, and `floor_div` over the operand shapes its fixtures spell. No
enumerator drives it, so an operation absent from it is unclaimed rather than
excluded. The general property is `spec/04-type-system.md` section 4.7.2's:
every `ShapeClass::Identity` builtin is registered with one dimension row for
both operands, so unification is the only route to a result, but no test in
this repository enumerates that registry against the oracle.

#### Completion design (2026-09-03)

This subsection decides how PP5 completes. It was written against the code at
`f5ec5ca63`. Every probe below was run twice: on a `chelis 0.18.6` binary
built from `main` on 2026-09-01 09:10, which predates [#1463], [#1508], and
[#1511] ("exec on prebuilt 09-01 binary"), and on a debug build of the
`f5ec5ca63` source tree ("exec on f5ec5ca63 build"). The two runs agree on
every row except the ones the [#1463] validator decides, which are marked. The
full transcript is in the pull request that landed this subsection.

The section keeps the doc's rule: each claim carries its evidence, and what
is not listed is absent, not qualified.

Revised 2026-09-03 for the user's decision that positional `expand` has one
meaning (row 16; recorded in PR [#1532], merged as `b8e08e5b9`, with [#1523]
merged the day before):
`expand` keeps only the same-rank singleton broadcast, a new primitive
`insert` adds an axis, and `spec/04` §4.7.2's two-candidate deferred model is
deleted: `spec/05` §2.4 carries rows and adjoint rows for both primitives under
`[05-AXIS-1]` and `[05-MOV-1]`, §4.7.2 states one result shape per operation,
and §4.5.3's named-axis forms are `insert`'s. The measurements below were taken
on the two-candidate code at
`f5ec5ca63` and are kept as the record of what that code does; each
conclusion is stated under the single-meaning rule, and where the two differ
the text says which is which. That code no longer exists: chelis#1277 S2b
flipped the checker, the lowering and the host interpreter onto the single
meaning, and S2c deleted the deferral ledger, the settlement executor, the
`TensorSettlement` registry and the source-ordinal index. Every symbol the
measurements name below (`DeferredExpandConstraint`, `candidate_types`,
`DeferralAction`, `ShapeEvidence`) is gone from the tree, so the line
references are archaeology rather than navigation.

**D1. Rank agreement between elementwise operands is decided by unification,
totally and at both ingresses.** Every `ShapeClass::Identity` binary builtin
is registered by `tensor_binop` (`crates/chelis-types/src/builtins.rs:1648`)
with one whole-tensor variable for both operands and the result, `(&tv, &tv)
-> tv`; the comparison family uses `cmplt_sig` (`:1687`), `(&tv, &tv) ->
output`. `unify`'s tensor arm (`crates/chelis-types/src/unify.rs:2335-2377`)
resolves both dimension rows and, when neither carries a rank spread, rejects
unequal lengths with `tensor rank mismatch: {} dims vs {} dims` before
unifying dimensions pairwise; a row with spreads goes through
`unify_row_against_ground` or `unify_row_against_row`. `Dim::Wildcard`
unifies with any single dimension (`unify_dim`, `:2534`) and never with a run:
a wildcard is a dimension of unknown extent, not an unknown rank, and
`stride`'s runtime-step axes are wildcards at the input rank
(`infer/app_shape.rs:849-884`). This runs inside `infer_app`, so it runs for
the `check_ir_program` ingress and for `check_typed_program`, which `chelis
prove` uses and which never calls `validate_ir_program`
(`infer/program.rs:1056-1100`).

*Evidence* (exec on prebuilt 09-01 binary and exec on f5ec5ca63 build, same
verdicts, `chelis check`):

| program | verdict |
|---|---|
| `add(s, g(e))` with `def g[a, b](y: tensor[a, b, f32]) -> tensor[a, b, f32] = y` | `DimensionMismatch: tensor rank mismatch: 1 dims vs 2 dims`, score 0.9415 |
| `add(sum(x, 0i32), x)` for `x: tensor[n, f32]` | `DimensionMismatch: tensor rank mismatch: 0 dims vs 1 dims`, score 0.9429 |
| `add(1.5f32, to_tensor([1.0f32, 2.0f32, 3.0f32]))`; likewise `max_elem` | `TypeMismatch: type mismatch: f32 vs tensor[3, f32]`, score 0.96 |
| `where(gt(x, 0.0f32), x, y)` with `y` rank 2 | `TypeMismatch` from `where`'s own procedural rule, score 0.9765; on the f5ec5ca63 build the PP5 validator adds a second `DimensionMismatch` for the same call |

Whenever both operands carry definite tensor types, a rank disagreement is
rejected by inference. No side channel takes part.

**D2. The listed escapes have equal checker ranks. `expand` is
rank-preserving, and the disagreement is between the checker's rank-1 `expand`
and the runtime's insertion.** Under the decided rule, `expand(x, 0i32, 2i64)`
on `x: tensor[n, f32]` is unconditionally `tensor[2, f32]`, and the operation
is a claim that `n` is 1: `spec/05` §2.4's row ("Set the size-1 dimension at
position `axis` to width `size`. Rank is unchanged and the operand's extent at
`axis` is 1"), §2.4.1 as [#1523] merged it ("A literal operand extent at
`axis` other than 1 is a type error. A symbolic or runtime operand extent at
`axis` other than 1 fails that claim's runtime extent guard and traps
`Domain`"), and §4.7.2 as [#1532] merged it ("Each operation has exactly one
result shape ... No result is deferred, no consumer selects between shapes,
and no context supplies a default"). No consumer chooses a form, because there
is one form; a rank-2 result is `insert`'s, `insert(x, 0i32, 2i64) : tensor[2,
n, f32]`.

The checker at `f5ec5ca63` reaches the same rank 1 by a route [#1532]
deletes, recorded here because the measurements below were taken on it.
`check_expand_signature` (`infer/app_tensor.rs:913`) records a
`DeferredExpandConstraint` on the call's result variable (`:1172`) carrying two
candidate shapes (`unify.rs:2272`, `candidate_types`), the same-rank
`tensor[2, f32]` first and the insertion `tensor[2, n, f32]` second, and the
pre-[#1532] §4.7.2 let the first shape-bearing consumer or a shape-neutral
`cast` select between them: `bind_tvar` routes a concrete partner through
`DeferralAction::Constrain(ShapeEvidence::Type)` (`unify.rs:2592`), which
tries the same-rank candidate first; `infer_cast` routes through
`DeferralAction::Freeze` (`infer/expr_record.rs:801`), which materializes the
same default (`materialize_deferred_expand_default`, `unify.rs:1197`); a
`realize` wrapper returns its operand's type unchanged
(`infer/expr.rs:210,480`), an unannotated user `def` instantiates a fresh
variable that `bind_tvar`'s variable-to-variable arm merges, and the program
freeze loop selects the same default for anything still open. Every call site
in the tables below therefore stamps `e` at `tensor[2, f32]`, which is the
single-meaning rule's answer; the one place the two disagree is a declared
rank-2 result, which the `f5ec5ca63` checker satisfies by selecting the
insertion candidate and which [#1532] makes a type error for `expand`.

So at `add(s, e)` both operands carry rank 1 in the checker's stamped program,
and at `add(s, cast(e, f32))` likewise. The runtime does the opposite,
through two mechanisms with one symptom and two owners, both [#597], and under
the single-meaning rule plainly so: an inserted axis is `insert`'s result, not
`expand`'s. On the C lanes, lowering's `fallback_expand_type`
(`crates/chelis-ir/src/lower.rs:5588-5606`) discards the stamped result type
and always inserts an axis; `spec/design/runtime_extents.md` C2.7 and Slice B
own its removal ("positional same-rank replacement never executes on any lane
today"), as item b2.5 of the [#1277] owner's Slice B2a, which carries the
§2.4.1 unit-extent guard in the same commit. On `chelis eval`, a
`def main() = f(...)` program routes through the host interpreter, which never
lowers `f`: `runtime/eval.rs:2733` evaluates `expand` through
`tensor_expand_host` (`crates/chelis-compiler-api/src/runtime/host_ops.rs:1425`),
whose comment says the typer "also accepts a same-rank 'replace-singleton'
interpretation ... but the host runtime has no access to user annotations, so
it always picks the canonical INSERT branch". The [#1277] owner's Slice B2h
routes host-lane application of a lowerable def through the DAG evaluator, and
[#597] closes only after B2h. (The B2a/B2h labels are the owner's, named on
2026-09-03; `runtime_extents.md` at `f5ec5ca63` still names one Slice B.) The
`[3]` versus `[2, 6]` that eval reports is the checker's rank-1 `e` meeting
the host interpreter's rank-2 `e`, and the invented values the C lanes return
are the same `e` meeting lowering's; neither is two operands the checker failed
to compare.

*Evidence* (both binaries agree unless a cell says otherwise; `sig` is the R1
reproducer's spelling, all others use `def f[n](x: tensor[n, f32])`; `eval`
verdicts are identical on both):

| program | `chelis check --show-inferred` | `chelis eval` |
|---|---|---|
| `add(s, e)` (inline params) | 09-01 binary: score 1, `f: (tensor[d0, f32]) -> tensor[*, f32]`; f5ec5ca63 build: rejected by the PP5 validator, `IR elementwise builtin add requires matching positive-rank tensor operands, got ranks 1 and 2`, score 0.9842, no other error | `got [3] vs [2, 6]` |
| `add(s, e)` under `sig f: tensor[n, f32] -> tensor[u, f32]` | 09-01 binary: score 1, `-> tensor[*, f32]`; f5ec5ca63 build: the same single validator error, score 0.9842 | `got [3] vs [2, 6]` |
| `add(s, e)` under a declared `-> tensor[2, n, f32]` | both: inference reports `def 'f' body doesn't match declared signature: body has type (tensor[d43, f32]) -> tensor[*, f32], declared type is (tensor[d43, f32]) -> tensor[2, d43, f32]`; the f5ec5ca63 build adds the validator's `got ranks 1 and 2` beside it, score 0.9684 | (type error) |
| `add(s, e)` under a declared `-> tensor[2, f32]` | 09-01 binary: score 1, `-> tensor[*, f32]`; f5ec5ca63 build: the single validator error, score 0.9842 | (f5ec5ca63: type error) |
| `add(s, cast(e, f32))` | score 1, `-> tensor[*, f32]` | `got [3] vs [2, 6]` |
| `add(cast(e, f32), s)` | score 1, `-> tensor[2, f32]` | `got [2, 6] vs [3]` |
| `add(s, realize(e))`, `add(realize(e), s)` | score 1, `-> tensor[*, f32]` | rejected, both orders |
| `add(s, normalize(e))`, `add(normalize(e), s)` | score 1, `-> tensor[*, f32]` | `unsupported builtin normalize in host runtime` |
| `add(s, g(e))`, `add(g(e), s)` with `def g(y) = y` | score 1, `-> tensor[*, f32]`, `g: (?0) -> ?0` | rejected, both orders |
| `add(cast(s, f32), cast(e, f32))` | score 1, `-> tensor[*, f32]` | `got [3] vs [2, 6]` |
| `cast(e, f32)` alone | score 1, `-> tensor[2, f32]` | `shape=[2, 6]` |
| `e` under a declared `-> tensor[2, f32]` | score 1 | `shape=[2, 6]` |
| `e` under a declared `-> tensor[2, n, f32]` | score 1, `-> tensor[2, d0, f32]` | `shape=[2, 6]` |

The third row is the direct measurement of the two rank models: in one
`chelis check` run on the f5ec5ca63 build, inference reports the body of
`add(s, e)` at `tensor[*, f32]`, rank 1, and the PP5 validator reports the
same `add` at "ranks 1 and 2". The `cast`, declared-rank-1, and
declared-rank-2 rows at the end show the disagreement without any elementwise
operation: the checker stamps rank 1, and eval prints rank 2 in every case;
those three eval cells are [#597] on the eval lane and belong to Slice B2h.
The declared-rank-2 row is the one program in the table whose checker verdict
[#1532] changes: the `f5ec5ca63` checker accepts it by selecting the insertion
candidate, and under the single-meaning rule it is a type error for `expand`
(the program wants `insert`). `add(s, cast(e, f32))` also compiled through the
prebuilt binary's C emitter (which predates R3's guard): `clang` exit 0, the
binary printed `tensor(shape=[3], data=[4.0, 6.0, 5.0])` and exited 0, with
0 `rank mismatch` guards beside 4 `chelis_indices_to_flat` calls in the
emitted C.

**D3. A second class: consumers that return early on a pending operand and
publish a result unrelated to it.** `check_reduction_signature`
(`infer/app_tensor.rs:576`) returns its fresh result variable untouched when
the operand is still a variable (`Type::Var(_) | Type::Error(_) => return
subst.apply(result_ty)`, `:591`). `add(e, sum(e, 0i32))` therefore unifies
`e`'s deferred variable with `sum`'s unrelated result variable, the freeze
default settles both to `tensor[2, f32]`, and the reduction is stamped at the
rank of its own input. This is [#1512] (filed from the `sum` case; the same
early return in `matmul` was [#1380], repaired by [#1511]) and it is tracked
by [#1277] Slice C, whose part 3 ([#1513]) records the disposition of every
builtin and names [#1512] as the owner of the skipped validation. Under the
single-meaning rule `e` is never a pending variable: it is `tensor[2, f32]`,
`sum(e, 0i32)` is `tensor[f32]`, and `add(e, sum(e, 0i32))` is rejected by
unification as rank 1 beside rank 0, exactly as D1's second row already
rejects `add(sum(x, 0i32), x)`, with no side channel. The early-return class
survives only for operands left unresolved for other reasons, which is why
[#1532] narrows [#1512] to its non-`expand` sources.

*Evidence* (exec on prebuilt 09-01 binary and exec on f5ec5ca63 build, which
is after [#1511]; every cell identical on both; all measured on the
two-candidate code):

| program | `chelis check --show-inferred` | `chelis eval` |
|---|---|---|
| `add(e, sum(e, 0i32))`, `add(sum(e, 0i32), e)` | score 1, `-> tensor[2, f32]` | `got [2, 6] vs [6]` / `got [6] vs [2, 6]` |
| `add(e, mean(e, 0i32))`, `add(mean(e, 0i32), e)` | score 1, `-> tensor[2, f32]` | same pair |
| `sum(e, 0i32)` alone | score 1, `-> ?0` (an inference variable is published) | `shape=[6]` |
| `add(e, stride(e, 1i64))` | score 1, `-> tensor[2, f32]` | `stride expects 2 strides for rank-2 tensor` |

The `stride` row is a different disposition with the same shape:
`infer_stride_app` returns the input's own variable for a pending operand
(`infer/app_shape.rs:800`), which propagates the choice but cannot express
`stride`'s extent change. It preserves rank, so it is not a rank escape; it is
listed because it is the same early-return pattern.

**D4. A third class: the comparison family's scalar rewrite ([#1506]).**
`infer_app` (`infer/app.rs:457-506`) rewrites a scalar operand's type to the
partner tensor's type "for unification purposes only" whenever one operand of
`cmplt`/`lt`/`gt`/`gte`/`lte`/`eq`/`neq` is a tensor and the other a scalar of
matching dtype, and `finish_unified_app` (`infer/app_post.rs:320-330`) builds
the `tensor[D, bool]` result from "whichever argument is the tensor". [#1508]
left both blocks in place by instruction. `[05-OP-36]` says the opposite:
"Mixed surfaces, numeric dtypes, static structured types, or tensor dimensions
are type errors"; `spec/05` §1.2 and `spec/04` §4.2 call broadcasting a hard
rule with no exception; `spec/04` §4.3 lists the comparison rows as "tensor,
tensor" only; and the `spec/05` §2.1 prose immediately above `[05-OP-40]`
spells out for `max_elem`/`min_elem` that the scalar form "is the rank-zero
instance of the tensor rule, not scalar/tensor broadcasting". `add(1.5f32, t)`
and `max_elem(1.5f32, t)` are rejected today
(D1). `gt(1.5f32, t)` and `gt(t, 1.5f32)` score 1 and eval broadcasts
(`[true, false, false]` and `[false, true, true]`; exec on prebuilt 09-01
binary; the same verdict is recorded on `989cdb7d2` in [#1506]). The decision
is recorded in row 14 of "Decisions and remaining questions": the checker
rejects, the spec wins, and the alternative is written there with its cost
because taking it is the user's explicit choice.

Three artifacts pin the accepting behaviour and flip with the decision: the
three accepting rows of
`crates/chelis-types/tests/issue5_cmp_broadcast_both_forms.rs`; the ignored
manual gate `coral_comparison_ops_broadcast_tensor_scalar` in
`crates/chelis-cli/tests/coral_prerequisites.rs:315`, whose doc comment
describes the rewrite as a Coral upstream fix, so the downstream shell has
source that depends on it; and the evaluator's broadcast in
`crates/chelis-compiler-api/src/runtime/host_ops.rs`, which becomes
unreachable from a checked program. The stdlib and `examples/` corpora were
searched for a tensor beside a scalar under the seven identities and none was
found; every comparison there is scalar-scalar or tensor-tensor.

**D5. What the shipped PP5 validator models.** `derive_ir_builtin_output_type`
(`infer/validate.rs:2459-2492`) derives `expand`'s rank as
`derive_movement_rank_output_type(list, type_env, static_env, 1)`
(`infer/shape_honesty.rs:129-143`): the input rank plus one, unconditionally.
That is a rank the language never assigns to `expand` under [#1532] (an added
axis is `insert`'s result), and one the `f5ec5ca63` checker did not stamp at
any call site in D2. In R1's own reproducer the checker therefore holds two
ranks for `e` in one run:
inference stamps `tensor[2, f32]` and the validator derives `RankOnly { rank:
2 }`, and the `DimensionMismatch` R1 reports ("got ranks 1 and 2") compares
`s`'s rank against the second. This is measured, not inferred: D2's
declared-rank-2 row shows one `chelis check` run on the f5ec5ca63 build
reporting the body of `add(s, e)` at rank 1 from inference and the same `add`
at "ranks 1 and 2" from the validator. R1 thus rejects a program whose stamped
types agree with each other and with the language's one meaning of `expand`,
and it agrees with eval only because eval's `tensor_expand_host` carries
[#597]. The same validator also duplicates a verdict
`where`'s own rule already reports (D1's fourth row), a cascade the deletion
removes. The rebinding false positive repaired in round 6 was an
instance of the same thing: a second rank model, keyed by name, drifting from
the first. R2's unforgeability property is real but protects a channel the
contract does not need; the exact-shape facts that predate PP5 (the `conv2d`
chaining passthroughs from RT-205) are separate and are not part of this
finding.

**D6. Mechanism evaluation.** The three candidates the brief named, then the
recommendation.

(a) *Rank agreement inside type inference for `Identity` applications, read
from each operand's inferred type.* This is what D1 shows the unifier already
does: both operands are one variable, and unequal rank is a `DimensionMismatch`
at both ingresses. Adding a second comparison at the application changes no
verdict for definite ranks, and under [#1532] an `expand` result is always
definite (on the `f5ec5ca63` code a pending one could only re-derive the
selection `bind_tvar` already performed). It is redundant by construction.
Failure modes if it were nevertheless added as a rank test rather
than a unification: a rank spread beside a ground row would need the row
unifier's splitting rules re-implemented or would false-positive on every
rank-polymorphic body; a `vmap` body sees its operands at the unbatched rank
and would be unaffected; rank-0 beside rank-N is already rejected by
unification (D1) except where a builtin's own scheme admits it, and a
duplicate test would have to encode that exception list again.

(b) *Keep the IR-validator channel but read the checker's stamped `type:`
metadata for any expression.* The stamps agree with inference by construction,
so on every escape in D2 the validator would read rank 1 beside rank 1 and
raise nothing; the only program it changes is R1's reproducer, which it would
now accept, exactly as inference does. It would still run only on the
`check_ir_program` ingress. A second reader of the stamps adds an ingress
dependence and no verdict.

(c) *A `ShapePreserving` registry predicate plus special-form and user-def
handling.* The predicate would drive the same derivation the validator runs
today, so it either keeps `expand` at input rank plus one and stays in
contradiction with the single-meaning rule, or reads the stamps and collapses
into (b). Its
user-def and special-form arms would re-implement instantiation and
`Freeze`/`Propagate`, which the unifier already owns. It is the enumeration the
round-6 repair removed, with a broader key.

(d) **Recommended: retire the rank side channel and let unification be the one
rank authority; route the residue to its three owners.** Concretely:

1. Delete `ShapeTypeFact::RankOnly`, `validate_identity_builtin_rank_
   requirements`, `derive_movement_rank_output_type`, `arg_tensor_rank`, the
   `stride` and `expand` arms of `derive_ir_builtin_output_type`, and the
   `ShapeClass::Identity` gate in `validate_ir_builtin_symbolic_requirements`
   (`validate.rs:2063`). `ShapeTypeFact` collapses to the exact-shape carrier
   the RT-205 conv2d chaining needs. Nothing is added: no spec sentence
   authorizes a second rank model, so the trace for this item is a deletion.
   One coupling this list did not anticipate, found by red-team round 1 and
   part of the same item: the failed-derivation marker of
   `record_let_binding_shape_fact` was keyed on `is_ir_shape_sensitive_builtin`
   membership, which still holds `stride`, `expand`, and `insert`. Deleting
   their arms without re-keying it turned all three into PERMANENT failed
   derivations, and a marked binding suppresses the whole `conv2d` validator
   downstream. The marker is therefore re-keyed onto
   `ir_builtin_has_output_type_derivation`, the table that owes the type. That
   is a repair to the deletion, not an addition: a marker meaning "this
   validator owed a type here and could not produce one" cannot be keyed on a
   list that answers a different question.
2. The `cast`/`realize`/`normalize`/user-def family and R1's own reproducer
   close through [#597], not through the checker: on the C lanes when Slice
   B2a item b2.5 deletes `fallback_expand_type` (C2.7 of
   `runtime_extents.md`), and on eval when Slice B2h routes the host-lane
   application of a lowerable def through the DAG evaluator instead of
   `tensor_expand_host`. Once every lane follows the stamp, `add(s, e)`
   executes `e` as the unit-extent broadcast that is `expand`'s one meaning,
   and the loud outcome for `n != 1` is the `Domain` trap `spec/05` §2.4.1
   states (merged in [#1523] on 2026-09-03, narrowed to the single meaning by
   [#1532]): a literal operand extent other than 1 is a check-time type
   error, a symbolic or runtime one fails the claim's §4.7 runtime extent
   guard, placed per `runtime_extents.md` C1.3 at the `expand` site, which
   Slice B2a item b2.5 carries in the same commit that deletes
   `fallback_expand_type`. At `f5ec5ca63` that sentence did not exist
   (§4.7.2's guard compared only the claimed result extent against `size`,
   and `runtime_extents.md:744` recorded the rows `silent_unguarded`), which
   is why D8 sequences PR A after the guard rather than before it.
3. The `sum`/`mean` family closes through [#1512] under [#1277] Slice C.
4. The comparison surface closes through [#1506] (row 14).
5. The [05-UNS] backstop stays: R3's DAG-lane guard and [#1484]'s host-lane
   guard (D9).

Why (d) satisfies the repo's bias: it is structural (one rank model, the
type), total by construction (unification is the only route to an
`Identity` result, so there is no operation to enumerate), ingress-independent
(both entry points run `infer_app`), and it deletes a hand-maintained
derivation rather than adding one. It also removes a class of false positives
(D5's second model) instead of narrowing them.

**D7. Rank-zero, symbolic ranks, spreads, and `vmap` under (d).** The checker
admits no rank-0 beside positive-rank pair for an `Identity` builtin except
where that builtin's own scheme does: `add(sum(x, 0i32), x)` is `0 dims vs 1
dims`, `add(1.5f32, t)` is a type mismatch, and `clamp(x, 0.0f32, 1.0f32)` is
rejected today with `clamp expects tensor input and tensor bounds` (exec on
prebuilt 09-01 binary), so the rank-zero bound that `spec/04` §4.3's `clamp`
row admits is not implemented; that gap is outside this section and is listed
under "Not established" rather than folded in. The tensor-DAG emitter's guard
condition `rank > 0 && rank > 0 && rank != rank` (`emit.rs:485`) preserves the
IR-internal rank-0 operand idiom of Tier-2 lowerings; it is not a source-level
permission, and (d) creates none. Symbolic dimensions, wildcards, and rank
spreads keep their `unify` semantics unchanged, because nothing is added to
them; a rank-polymorphic body remains governed by the §4.2 body-discipline
check that `ShapeClass` exists for, and a `vmap` body is inferred at the
unbatched rank exactly as today.

**D8. Implementation plan.** Two pull requests, because the slices can ship
apart and the second has a sequencing dependency.

*PR A, checker side (`crates/chelis-types`, `crates/chelis-cli` tests, this
document).* Files: `crates/chelis-types/src/infer/shape_honesty.rs` (delete the
`RankOnly` variant and the four functions named in (d)1),
`crates/chelis-types/src/infer/validate.rs` (the `expand`/`stride` arms at
`:2474-2475` and the `Identity` gate at `:2063`),
`crates/chelis-cli/tests/issue_668_elementwise_rank_honesty.rs`,
`crates/chelis-cli/tests/issue_731_fitness_honesty_corpus.rs` (the
`issue_668_rank_divergent_elementwise` row at `:103`), and a new
`crates/chelis-types/tests/issue_668_rank_agreement_is_unification.rs`. The
sibling [#1277] stream owns `unify.rs`, `app.rs`, `app_post.rs`,
`expr_record.rs`, and `builtins.rs`; PR A touches none of them. PR A still
changes what `chelis check` reports on deferred-expand programs (the
validator's rejections in the D2 table disappear), and the [#1277] owner's
Slice B2a/B2h fixtures assert checker verdicts on exactly those shapes, so the
owner must be told before PR A pushes, not only before PR B. Sequencing rule
from D6 (d)2 and the single-meaning rule: [#1532] is merged (`b8e08e5b9`), and
PR A lands after S2a, the [#1277] stream's `insert` builtin plus mechanical
rename on branch `agent/1277-s2a-insert-rename` (in progress; it touches
`check_expand_signature`, the `infer/app.rs` and `app_post.rs` dispatch arms,
the builtin registries, and the corpus call sites), because PR A's rank
negatives are built from `insert` (below) and because the `Domain` trap of
`spec/05` §2.4.1 is then the named loud outcome for the programs PR A stops
rejecting; and the
orchestrator is told before PR A changes any `chelis check` verdict on a
deferred-expand fixture, since the [#1277] stream's B2a/B2h rows assert those
verdicts.

Under the single-meaning rule every [#668] reproducer built from `expand` is
rank 1 beside rank 1 and well typed, so PR A constructs its rank-2 operand
with `insert` once it exists: `add(s, insert(x, 0i32, 2i64))` and `add(insert(x,
0i32, 2i64), s)` are rejected by ordinary unification as `tensor rank
mismatch: 1 dims vs 2 dims`, with no side channel, which is the cleanest
demonstration of D1; the `expand`-built rows become positive controls whose
runtime outcome for `n != 1` is §2.4.1's `Domain` trap once b2.5 lands.

**What PR A measured, against what this plan predicted.** The plan above was
written against `f5ec5ca63` and predicted that ten [#1463] rows, the control
half of an eleventh, and the corpus row would flip from rejection to
acceptance. None of them did, and the reason is [#1277] S2b, which landed
between the plan and the implementation. S2b did two things this plan did not
anticipate. It gave the validator rank delta 0 for `expand`, so the validator
agreed with the stamp and stopped rejecting the `expand`-built reproducers on
its own; and it re-vehicled every row of
`issue_668_elementwise_rank_honesty` onto `insert`, whose result genuinely is
rank 2, so those rows became ordinary unification rejections. The verdict flip
the user confirmed in row 15 therefore shipped in S2b. PR A deletes a
mechanism that by then changed no verdict at all.

Measured on `12c04c66a`, the rebased base, and on the repaired deletion, at
both ingresses: for
the elementwise family every rejection stays a rejection with the same `unify`
text, and every acceptance stays an acceptance. What the deletion removes there
is a DUPLICATE. On the `check_ir_program` ingress the validator reported a
second `DimensionMismatch` for a call unification had already refused, and
`check_typed_program`, which never calls `validate_ir_program`, reported one.
Same program, two diagnostic sets: the [#1107] class. It also removed the
`where` cascade D5 names, where the duplicate sat beside `where`'s own rule.

**That claim is stated for the elementwise family because that is where it was
measured, and the first attempt to state it unqualified was false.** Red-team
round 1 found the reason: the same `ShapeTypeEnv` has a second consumer, the
`conv2d` validator, reached through the failed-derivation marker rather than
through a rank comparison. Before the marker was re-keyed ((d)1 above), a
`conv2d` whose input came from a let-bound `stride`, `expand`, or `insert`
had its whole validator suppressed, so an invalid `stride = 0` scored 1 and
then panicked in codegen. With the marker re-keyed, the repaired tree and
`12c04c66a` return byte-identical verdicts, messages, and scores on all seven
of that round's probes, so the no-verdict-change result holds for both
consumers of the environment. The lesson generalises and is recorded because
it will recur: **a claim about deleting a shared data structure has to be
measured at every consumer of that structure, not at the one the deletion was
about.**

Test dispositions, each stated for the `check` ingress and, in the new
`chelis-types` file, for `check_typed_program` as well:

- The REGRESSION assertion is ingress agreement, in
  `issue_668_rank_agreement_is_unification`'s `agreed_diagnostics`: the two
  ingresses must return the same diagnostic set. Six of that file's fifteen
  rows are red on `12c04c66a` and green after the deletion, each failure
  printing the validator's surplus diagnostic. Its `where` row asserts a
  diagnostic COUNT of one and is red on the base for the same reason. The
  prove-red step is `git checkout 12c04c66a --
  crates/chelis-types/src/infer/shape_honesty.rs
  crates/chelis-types/src/infer/validate.rs`, rebuild the one target, watch
  those rows fail, restore, watch them pass. The receipt was first taken
  against `c8a5f1a75` and re-run against `12c04c66a` after the rebase; both
  files are byte-identical between those two commits, so it is the same
  experiment under the current base's name rather than a relabelled one.
- Every verdict assertion is a DISPOSITION LOCK, green in both states, and
  labelled as one. That covers the rejections in both operand orders (a
  declared rank-2 user def beside a rank-1 operand, D1 row 1; a rank-0
  reduction result beside its rank-1 input, D1 row 2; a scalar beside a tensor
  under `add` and `max_elem`, D1 row 3; a rank-2 `where` branch, D1 row 4; and
  the `insert` rows `add(s, insert(x, 0i32, 2i64))` and its reverse), the
  acceptances, and every row of the CLI file.
- Mutation receipt, run and recorded in the pull request: with the side
  channel gone, `unify`'s `(0, 0)` arm was mutated to skip the length check
  and the new file rerun. All seven `tensor rank mismatch` rows went red at
  BOTH ingresses, reporting no diagnostic at all. The four
  scalar-beside-tensor rows stayed green and are not claimed to move: they are
  rejected by the type constructor, not by the rank comparison. The `where`
  row stayed green, which confirms its diagnostic is `where`'s own rule's.
- The forged half of `authored_deep_type_metadata_cannot_override_the_
  inferred_rank` is now measured rather than open: authored `type:` metadata
  on a `var` that disagrees with its binding does not displace the inferred
  type, so the forged program is refused for the same reason as the control.
- At PP5 PR A delivery, the round-1 file
  `crates/chelis-types/tests/issue_668_deleted_derivation_does_not_suppress_conv2d.rs`
  covered the environment's other consumer. Four regression rows were red at the
  first pushed head `948ed5736`, where each scored 1 with an empty error list:
  `stride = 0` behind a let-bound `expand`, `insert`, and `stride`, and a
  symbolic spatial dimension behind a let-bound `expand`. Two disposition
  locks: the direct call, which reaches the stride check itself and names it,
  and a genuinely failed derivation, whose cascade had to stay suppressed.
  PP9 supersedes those capability-era dispositions: symbolic metadata is
  checker-legal, the cascade side channel is deleted, the well-typed let-bound
  form is accepted, and static convolution-domain checks run at both
  ingresses. The file now locks those PP9 outcomes while preserving the
  original zero-stride regression.
- New: `the_expand_built_reproducer_is_loud_at_run_time` builds and runs the
  `expand`-built reproducer with a refuted unit-extent claim on the evaluator
  and C lanes, and asserts section 2.4.1's `Domain` trap, `numeric trap:
  domain in load at i64` with `claimed = 1` and `axis 0 = 2`, beside a
  satisfied-claim control that executes exactly. Both lanes trap.

The six D2/D3 escapes in both operand orders are *not* checker negatives in
PR A: the `cast`/`realize`/`normalize`/user-def rows are spec-accepted
programs whose loud outcome belongs to [#597]'s runtime guard, and the
`sum`/`mean` rows are [#1512]'s. PR A records both facts in this section
rather than asserting a verdict it does not own.

*PR B, the comparison surface ([#1506]).* Files:
`crates/chelis-types/src/infer/app.rs:457-506` (delete the rewrite block so
the `(&tv, &tv)` scheme reports the mismatch),
`crates/chelis-types/src/infer/app_post.rs:320-330` (the comment and the
"whichever argument is the tensor" search become dead for mixed pairs and are
simplified), `crates/chelis-types/tests/issue5_cmp_broadcast_both_forms.rs`
(the three accepting rows become negative controls asserting the
`[05-OP-36]` rejection; the Slice C rows are untouched),
`crates/chelis-cli/tests/coral_prerequisites.rs:315` (the broadcast gate
becomes a negative control), and optionally
`crates/chelis-compiler-api/src/runtime/host_ops.rs` (the broadcast path is
unreachable from a checked program and may be removed or kept as a `[05-UNS-1]`
rejection; the implementer chooses and says which). The diagnostic should name
`[05-OP-36]` and the explicit spelling, `gt(xs, expand(to_tensor([1.5f32]),
0i32, shape(xs, 0i32)))`, following the `div`/`floor_div` precedent of a
rejection that names its replacement. Positive controls: scalar beside scalar
(`gt(1.5f32, 2.0f32)` is `bool`) and tensor beside tensor of one shape, both
orders. Negatives: the seven identities with a scalar on each side, and
`eq(mk_bools(), true)`. Prove-red: run the negatives against the tree with the
rewrite present and watch them accept; delete; watch them reject. Since PR B
touches `app.rs` and `app_post.rs`, the [#1277] stream must be told before it
opens. Sequencing: the explicit spelling is well-typed today but executes as
an insertion on every lane until [#597] closes (Slice B2a item b2.5 for C,
Slice B2h for eval), so Coral has no executable replacement until then; PR B
should land after both or state that gap in its body and in the Coral pin
bump. Estimated hand-written size: under 250 lines.

**D9. The C guards remain the `[05-UNS]` backstop.** R3's
`emit_elementwise_operand_guard` (`crates/chelis-backend-c/src/emit.rs:462`)
and [#1484]'s host-lane guard (landed as PR [#1519]) are not made dead by (d).
`[05-UNS-4]` says a pre-codegen gate "SHALL NOT be the sole defense against an
unsupported case reaching emission", and `[05-UNS-1]` forbids substituting a
value. Today the IR they guard is produced by the checker/lowering
disagreement of D2; after Slice B2a item b2.5 it can arise only from malformed
or dynamically bound IR, which is the case [#668]'s first comment already
assigns to them. They do not cover the unit-extent claim of D6 (d)2: once a
lane executes the stamped `expand`, both `add` operands are rank 1 with extent
2, and the guard's rank and per-axis comparisons pass; that claim's guard sits
at the `expand` site (b2.5). Their oracles stay in the acceptance list below.

**Necessity trace.** Every mechanism above maps to a numbered-spec sentence
or a named instance; nothing without one is kept.

| mechanism or claim | authority |
|---|---|
| operand rank agreement for `Identity` builtins is unification (D1, (d)) | `spec/05` §1.2, §2.1 "Dimension rule"; `spec/04` §4.1, §4.2, §4.3; instances D1 rows 1-4 |
| positional `expand` has one result shape, rank-preserving, and is a unit-extent claim on the operand (D2) | the user's decision (row 16); `spec/05` §2.4 row and §2.4.1 ([#1523], merged); `spec/04` §4.7.2 as merged by [#1532] (`b8e08e5b9`) |
| every runtime lane must execute `expand` as that broadcast: C via `fallback_expand_type` (Slice B2a item b2.5, which also carries the §2.4.1 guard), eval via `tensor_expand_host` (Slice B2h) (D2, (d)2) | `spec/05` §2.4, §2.4.1; `runtime_extents.md` C2.7, Slice B; instance [#597] |
| a reduction over a pending operand must eliminate or reject, never publish an unrelated result (D3, (d)3) | `spec/04` §4.7.2 elimination sentence; instances [#1512], [#1380] |
| a scalar beside a tensor under the seven comparison identities is a type error (D4, (d)4) | `[05-OP-36]`; `spec/05` §1.2; `spec/04` §4.3; instance [#1506] |
| the emitter guards stay (D9) | `[05-UNS-1]`, `[05-UNS-4]`; instances [#664], [#668], [#1484] |
| retire `ShapeTypeFact::RankOnly` and the identity-rank validator ((d)1) | a deletion; no sentence authorizes a second rank model |
| an `expand` whose operand extent at `axis` is not one has a stated outcome ((d)2, row 15) | `spec/05` §2.4.1 since [#1523] merged (literal: type error; symbolic or runtime: §4.7 `Domain` guard), narrowed to the single meaning by [#1532]; not stated at `f5ec5ca63`, where §4.7.2's guard compared only the result extent against `size` and `runtime_extents.md:744` recorded the rows `silent_unguarded`; not authored here |

This PR authors no numbered-spec text. The unit-extent claim above is stated
by [#1523] and narrowed by [#1532]; the one sentence the residue still needs
and the spec does not state, a reduction's result type while its operand is
genuinely unresolved, belongs to [#1512]'s repair under [#1277], narrowed by
[#1532] to non-`expand` sources.

**Not established.**

- The stamped `type:` metadata of `e` itself under the validator's rejection
  was not read directly, because `chelis check` publishes no inferred
  signatures when an error is present; D2's declared-rank-2 row reads the
  rank off the body-versus-signature diagnostic instead, which is the same
  inference result one step later.
- What inference does with authored `type:` metadata on a `var` that
  disagrees with its binding (the forged half of the R2 test).
- The result type of a reduction whose operand is genuinely unresolved (not
  an `expand` result, which [#1532] makes definite). Whether such a reduction
  rejects, as [#1512] argues from §4.7.2's elimination sentence, or carries a
  derived obligation, is [#1512]'s to decide and needs a numbered-spec
  sentence there.
- The implementation of `spec/05` §2.4.1's unit-extent claim: the literal
  case (`expand(a, 0, 3i64)` over `tensor[2, f32]`, accepted at `tensor[3,
  f32]` by the [#1508] rows on the two-candidate code) becomes a check-time
  type error and the symbolic case (R1's reproducer, `n != 1`) a runtime
  `Domain` trap, but neither is implemented at `f5ec5ca63`; the checker half
  is S2a (`agent/1277-s2a-insert-rename`) and the guard is Slice B2a item
  b2.5, and D8 sequences PR A after both. Nothing here measures them.
- `clamp`'s rank-zero bounds (`spec/04` §4.3) are rejected by the checker
  today; not PP5's, recorded so no reader takes D7 for a claim that the
  permission is implemented.

#### Acceptance

For the shipped R1-R3:

```sh
cargo nextest run -p chelis-cli --test issue_668_elementwise_rank_honesty --no-fail-fast
cargo nextest run -p chelis-backend-c --test exec_compile direct_positive_rank_mismatch_traps_before_indexing
cargo nextest run -p chelis-backend-c --test exec_compile host_lane --no-fail-fast
cargo nextest run -p chelis-cli --test issue_1484_host_lane_rank_guard --no-fail-fast
```

The first is the oracle for R1 and R2, the second for R3, and the last two for
R4. None of them covers the operations named above, and the last two are about
generated C, not about a checker verdict.

For the completion design, PP5 is complete when all of the following hold,
and the ledger row stays "partial" until they do:

```sh
cargo nextest run -p chelis-types --test issue_668_rank_agreement_is_unification --no-fail-fast
cargo nextest run -p chelis-cli --test issue_668_elementwise_rank_honesty --no-fail-fast
cargo nextest run -p chelis-backend-c --test exec_compile direct_positive_rank_mismatch_traps_before_indexing
```

with the first file's rows green at both ingresses and its rank rows red
under the D8 mutation; [#1484]'s host-lane oracle green; [#1506]'s negative
controls green (PR B); and [#597] and [#1512] closed by their owners, each
with the D2 or D3 rows re-measured on the closing head. PR A has delivered the
first two commands and the mutation receipt; the remainder is PR B's and the
two owners'. The checker half of [#668] is then a statement about unification,
and the runtime half about the guards.

### PP6. Schedule and header honesty ([#1486], [#1487], [#1485]; closes [#1134], follow-up [#1854])

**Opened 2026-09-03; delivered by PRs [#1542] and [#1551].** PR [#1457] delivered
[04-INF-4] and recorded three defects as ratcheted residue. They share one
mechanism: the checker lets a top-level reference observe a declaration
before that declaration's own body has been checked, or fails to see that a
reference will run while a value is being initialized. Each was confirmed
from the code and, where the prebuilt binary predating [#1457] could reach
it, by execution; the PP6 pull request body carries the probe transcript.

**[#1854] explicit-binder follow-up (2026-09-17).** The original PP6 design
preserved occurrence-based `defsig` binders because [04-INF-6] needed to make
both explicit and implicit variables rigid. That choice is superseded by the
numbered-spec decision in `spec/02` P4b, `spec/03` §2.2/§2.5.1, and `spec/04`
§3.1.3/§5.8.1: a declaration's binder list is complete. Surf carries that
complete list on its `defsig`; the resolver admits only listed `t-var`,
`d-var`, and `d-rank` uses. Unlisted scalar/precision names remain `t-prim`,
and unlisted variable nodes reject. Canonical Deep adds a structural
binder-list child only for a polymorphic `defsig`; the two-child form is
monomorphic. The implementation
slice removes both implicit Surf collectors and the checker scan that rebuilt
binders from type occurrences, makes resugaring preserve and validate the
explicit list, migrates every accepted source/Deep fixture, and adds paired
ingress plus CLI negatives for unknown dtype aliases and undeclared
type/dimension/rank variables. No typo regex or compatibility admission path
is part of the design.

- [#1486], a compiled wrong answer. `def f(n: i32) = add(1, n)` desugars
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
  `forall a. (i32) -> a`. `infer_top_level` later instantiates that scheme
  for the body, unifies the body against the instance, and rebinds the name
  to the narrowed, regeneralized result, so the environment scheme is
  `(i32) -> i32` only after the body is inferred. A reader scheduled
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
  add(x, 1)` and the formerly implicit, now explicit
  `def f[a](x: a) -> a = add(x, v)` both check
  clean, the body instance of `a` is bound to `i32`, and the registered
  scheme is the narrowed `(i32) -> i32`; the only rigidity check today is
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
  followed into the function's body), while `carried = pick(fn (x: i32)
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
  the declaration's own binders (`def f[n](x: tensor[n, f32]) = x`) would tie
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
  (measured: `def mk() = fn (x: i32) -> f(x)`, `b: i32 = (mk())(1)`,
  `f` reading `b` checks clean and fails under `eval` with a runtime cycle)
  escapes it, and deciding whether a user-defined callee applies its
  parameter is a higher-order flow question the checker cannot answer at
  the cycle detector. The over-rejection is accepted under the repository's
  explicit-over-inference bias, and it adds no new class: the detector
  already treats a bare function reference this way, so the rule removes an
  asymmetry rather than introducing a policy. The escape hatch is to pass
  the value as an argument.
- [#1485]'s three spellings under these rules: `carried = wrap(f)` with a
  signed `f` that applies `carried`, `carried = pick(fn (x: i32) -> f(x))`
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
   binders listed by `DeclaredSigMetadata.binders`. The registered scheme
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
3. *Canonical eager-reference graph.* `TopLevelReferenceGraph::build`
   indexes the unit's top-level definitions and runs
   `collect_top_level_references` once for each flattened item. That one
   lexical collector records read and direct-application edges, descends into
   lambda bodies with their parameters bound, and preserves sequential local
   and match-arm scope. It removes only [04-INF-4]'s explicitly typed literal
   external-input self edge. A lambda initializer classifies its top-level
   `def` as a function while the references in its body remain graph edges,
   which is [04-INF-7]'s stored-closure rule. The resulting graph supplies the
   function plan, mixed-component schedule, and `report_initialization_errors`;
   both production drivers retain that same instance through validation, so
   cycle reporting performs no second reference walk.
4. *Mixed groups.* `TopLevelReferenceGraph::inference_components` projects
   strongly connected components of the complete reference graph in
   dependency-first order with each component's members in source order.
   `primary_inference_schedule_with_reference_graph` contracts those
   components before adding the backward-value, `defsig`-less mirror, and
   hole-header precedence edges, so Kahn's algorithm operates on a DAG and
   has no stall-release path. `primary_inference_groups_for_schedule` emits
   each component once and separately records its recursive-function members;
   a mixed cycle alone therefore does not activate recursive-instantiation
   validation. In both production drivers a cyclic group enters the scoped
   component level, and `prebind_cyclic_component_schemes` installs
   monomorphic provisional types for members without a declared or metadata
   prebind: arity-shaped types for functions and one fresh type for eager
   values. Each body unifies with its provisional type before the component
   is generalized as a unit, and scoped restoration handles cancellation and
   failure. The graph then appends the ingress-independent `CycleDetected`
   diagnostic for a cyclic component containing an eager value; no earlier
   `UnboundVariable` is erased.

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
   `def f[a](x: a) -> a = add(x, 1)` rejects at the declaration with no
   reader present, while the unlisted spelling rejects earlier as an unknown
   type name; a binder collapse
   `def g[a, b](x: a, y: b) -> a = y` rejects, and
   `def id[a](x: a) -> a = x`, `def k[a](x: a) = x`, and a bounded
   `[p: Float]` body written with
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
   negatives; the stored lambda `carried = fn (x: i32) -> f(x)` (a
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
   restated over component contiguity. A paired control checks the same later
   value outside the active component and must still report
   `UnboundVariable`, proving provisional visibility cannot leak past the one
   co-inference scope. The `--lib` oracle's stall
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
   edge or through a hole edge (`r = f(2)` with `def f(n: i32) =
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
the `issue_1339_top_level_initialization` CLI oracle,
`scripts/compiled_value_ownership_oracle.py
--phase 0`, `scripts/unrepresentable_domain_oracle.py`,
`scripts/dtype_phase4b_oracle.py`, and a differential `chelis check` over
every tracked `.ch` and `.dp` file against a control binary built from the
base, expecting the stdlib list above and no other score change.

**Residue: the #256 deferred-borrow classification ([#1589]).** [04-INF-6] made
`def use_it[a](seed: a) -> bool = consume_bnp(&seed)` a rigidity rejection, which
was the header both of the classification's acceptance tests used; PR [#1542]
migrated them to a concrete carrier and relabelled them as disposition locks.
That relabel was correct and the conclusion drawn from it was not. Measured, the
mechanism is live: its reject branch is exercised by three tests in
`issue_256_polymorphic_return_borrow`, and its accept branch is reached by an
inferred parameter, an unannotated def, a lambda parameter and a `let` alias of
any of them, resolving `sound=true` in each case. What no longer reaches the
accept branch is any tracked source: zero of 171 tracked `.ch` files, and zero of
the 85 shell definitions that borrow an unannotated parameter, because every one
of those takes a concrete type from a `sig`. The residue is therefore not a dead
mechanism but a missing corpus row plus one spec-compliance defect outside PP6's
schedule subject: `check_borrow_arg` failed closed when `expr_type` returns
`None`, rejecting a borrow whose inner `spec/04` §8.2 says "must ultimately
resolve to" a carrier and which inference has already resolved to one. PP6 does
not absorb it; [#1589] owns it, and decision row 21 records that §8.2 already
decides the rule.

**Exclusions.** PP6 does not absorb [#1512] (an early return on an
unresolved `expand` operand skipping validation, owned by the [#1277]
stream; PP6 touches header-versus-body typing in the schedule, not deferral
settlement). PP6 supplies [#1339]'s required complete eager-reference graph,
but does not decide that issue's distinct acyclic later-value verdict;
[04-INF-8] and the [#1339] implementation below own it. It is not part of the
compiled-value ownership plan. PP6 also does not absorb [#874]/[#887]'s
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

### PP7. Stamped-ingress reader parity ([#1125]; the carrier axis, with [#1537]'s pass-set residue)

**Opened 2026-09-03; decided below.** [#1107] swept one reader shape - `let
deep::Expr::List(..) = x else { continue | return }` - across
`chelis-types/src/infer/`, found 31 confirmed typed-vs-IR divergences, and
routed every one through the carrier-preserving `stamped_parts`. [#1125]
audited the `match`-arm and `if let` readers of the same class and recorded 20
divergent sites over 12 defects inside `chelis-types`, plus verified clusters
in `chelis-compiler-api` and `chelis-prove`. Nothing prevents the next one: no
lint, no tripwire, and no test quantifies over readers.

#### What execution shows

Six hand-authored programs at `6fd95fd5`, each run through both shipped
surfaces of the same file - `chelis check` (the serialized-IR ingress, which
normalizes) and `chelis prove` (the stamped typed ingress, which does not).
The debug binary carries default features, so `chelis-prove` is on and `smt`
is off.

| program | `check` | `prove` | direction |
|---|---|---|---|
| `(lit {type: (t-prim {} i8)} 200)` | rejects, exit 2 | accepts, exit 0 | fail-open |
| `(lit {type: (t-prim {} i8)} 100)` (control) | exit 0 | exit 0 | agree |
| `(var {} nope)` (control) | exit 2 | exit 3 | agree |
| `deftype` with `invariant:` and no `opaque:` | rejects, exit 2 | accepts, exit 0 | fail-open |
| `(t-tensor {} (d-lit {} 2) (t-prim {} f8e4m3))` | rejects, exit 2 | accepts, exit 0 | fail-open |
| well-formed `(tuple-get {} (tuple {} ...) (lit ...))` | accepts, exit 0 | rejects, exit 3 | fail-closed |

The two controls carry the argument: `chelis prove` does type-check the
module and does surface an unbound variable, and it does accept the in-range
literal, so the three fail-open rows are missed checks rather than an absent
checker. The fail-closed row rejects a program `chelis check` accepts at score
1.0, with `invalid tuple index: expected a non-negative integer literal, found
a non-literal expression`.

Three further probes drive the same divergence outside the checker entirely.
Through tide's `/lower`, the identical two-def program with `entry: "target"`
lowers cleanly from Surf and fails the check stage from Deep with the
unrelated def's `unbound variable: does_not_exist`, because
`prune::prune_to_entry` recognizes no `def` in the stamped carrier and returns
the program unpruned. `deep_referenced_vars` in the same file was migrated to
`Node`; `deep_def_name` and `deep_named_decl_name` beside it were not.

The other reaches further. `chelis prove` warns when a module declares
invariant-carrying opaque types that the non-`smt` build could not verify.
The same module warns as `.ch` and does not warn as `.dp`, though the `.dp`
declares exactly one such type. `count_invariant_opaque_deep`
(`chelis-cli/src/prove/mod.rs`) is List-only **and** reads its tag as
`Atom::Name("deftype")`, which decode-once (§C4.2) already forbids: the parser
stamps every vocabulary tag, so element 0 is `Atom::Tag`. The function
therefore returns zero for every input, in both carriers, so the warning cannot
fire on the `.dp` path at all. It is the class's worst shape - a reader dead
twice over, on a release-blocking surface, silently.

The seventh divergence is the proof tier itself, and it needed a solver build to
see. With `--features smt` against the local cvc5 store, one opaque-invariant
module with a guarded producer verifies at two different tiers depending only on
which spelling of the same module is submitted:

| surface | `proof_tier` | `composite_verdict` | `qualifiers` |
|---|---|---|---|
| `chelis prove mod.ch` | `smt` | `proven_modulo_real_arithmetic` | `["real_arithmetic"]` |
| `chelis prove mod.dp` | `fuzz` | `fuzz_validated` | `["fuzz", "fuzz_base"]` |

Both report `passed`, both exit 0, both summarize `obligations: 1, passed: 1`.
The `.dp` module was produced from the `.ch` one by `chelis deep`, so the two
are the same program. Nothing in the summary distinguishes a proof from a
hundred samples; only the per-obligation `proof_tier` does, and a reader who
does not compare surfaces has no reason to look. `prove_deep_file` hands
`run_module_obligations` the exprs from `parse_and_stamp_file`, and
`tier_b_lower`'s `tag`, `children`, and `lookup_producer` are List-only, so the
obligation cannot lower and `run_one` falls through to Tier C. This is the row
[#1362] §1.I carries; it was reported from the code and is now measured.

**Why this survived.** `crates/chelis-cli/tests/prove_deep_obligations.rs`
already asserts `proof_tier == "smt"` for exactly this fixture on exactly this
surface. The file is `#![cfg(feature = "smt")]`, and no runner exists: the gate
never passes `--features smt`, `ci.yml`'s smt job builds `chelis-cli` and then
tests only `chelis-prove`, and `smt-full-prove.yml` runs `chelis-prove` lanes
nightly and never names `chelis-cli`. A merged test that pins the correct
behavior has no invocation anywhere, so the regression it was written to catch
landed green. The lint E5d delivers is worth nothing on the same terms. Three of
the PP7 parity set's four commands therefore run in the default suite; the
fourth is the smt-gated one, and the CI step E5c owes is what makes it run at
all.

#### Two axes, and the audit conflates them

**Axis A, carrier divergence.** A reader destructures `Expr::List`, receives
an `Expr::Node`, `Expr::BareList`, or `Expr::UnknownForm`, and observes
nothing. Nothing distinguishes "this subtree is empty" from "I cannot read
this carrier", so the check does not run and no diagnostic says so. Every row
in the table above is this axis.

**Axis B, pass-set asymmetry.** The two entries do not run the same passes.
Measured at `6fd95fd5`: `check_ir_with_signature_context_in_session` runs
`validate_ir_program`, `validate_tensor_precisions_in_program`,
`validate_type_invariants_in_program_with_sink`, and
`validate_polymorphic_op_constraints`. `check_typed_program_in_session`
reaches the last three through `infer_program_with_product_in_session` and
does not run `validate_ir_program` at all.

The audit reports the opaque-invariant pass as inert on the typed lane and
reads that as axis B. It is axis A: the pass **is** invoked from the typed
entry, and its private List-only `tag()` and `children()` return nothing for a
stamped `Node`. Axis B's only measured instance is `validate_ir_program`.
Sorting the two matters because they have different owners and different
fixes, and because a mechanism chosen for the wrong axis closes neither.

#### Why normalizing at the sixth entry is not the mechanism

`check_typed_program_in_session` is the one check entry that does not run its
inference over `normalize_nodes_to_lists` output. It already calls that
function for the [04-INF-4] cycle detector alone, so the cheap repair is to
move the normalization above inference and let the other five entries' shape
apply to the sixth.

Reject it as the mechanism, on measurement rather than doctrine. Of the seven
executed divergences, normalizing inside `program.rs` reaches at most the four
that live inside the checker. It does not reach `prune.rs`, which reads
stamped Deep in `chelis-compiler-api::pipeline` **before** any checker entry
runs; it does not reach `tier_b_lower.rs`, which reads the stamped exprs
`prove_deep_file` hands `run_module_obligations` **after** the check
completes; and it does not reach `count_invariant_opaque_deep`, which reads
the same exprs later still, to decide whether to warn.
The defect is not that one checker entry forgot to normalize. It is
that every consumer of `parse_and_stamp_file` output is a potential silent
reader, and consumers exist on both sides of the checker.

The doctrinal objection stands beside the measured one and does not carry it
alone. §C4.2 requires every public and compiler `.dp` ingress to consume the
stamped representation rather than normalize it away; a sixth normalization
moves further from that target and deepens the dependency on the `Expr::List`
carrier [#1029] plans to delete.

#### Three findings that change what the fix must be

1. **`stamped_parts` is not carrier-complete.** It reads `Node` and `List` and
   returns `None` for `BareList` and `UnknownForm`. `build_def_param_scope`
   was migrated to it by [#1126] and is still inert for inline-annotated
   params, because a `(x {type: T})` param stamps to `BareList` and the
   migrated caller's last step, `param_name_and_inline_type`, is List-only.
   "Route it through `stamped_parts`" is therefore not a sufficient
   instruction, and a lint that mandates `stamped_parts` would certify that
   site as fixed.

2. **Seven private copies exist, none shared.** `chelis-types/src/infer/common.rs`,
   `chelis-types/src/linearity.rs`, `chelis-types/src/adt.rs`,
   `chelis-effects/src/lib.rs`, `chelis-ir/src/lower.rs`,
   `chelis-ir/src/host.rs`, and `chelis-ir/src/host_type_state.rs` each define
   a function named `stamped_parts`, with four different signatures: one takes
   an expected tag, one returns `Result`, one drops the metadata, four return
   `Option`. All seven handle exactly `Node` and `List`. A rule expressed over
   a private helper cannot be enforced across seven definitions that a
   reviewer must recognize by name.

3. **A shallow bridge reads as migrated and is not.** `walk_for_tensor_precision`
   has an `Expr::Node` arm that rebuilds the node through `Node::to_list` and
   recurses, so it looks carrier-complete to any grep for `Expr::Node`
   coverage. `to_list` copies children verbatim, so the node becomes a `List`
   whose children are still `Node`s, and the arm's own
   `let deep::Expr::List(prec_list, _) = last` then fails on the trailing
   `t-prim`. That is the `f8e4m3` row: the entire tensor-precision check is
   skipped on the stamped ingress by a walker that has a `Node` arm.

#### The mechanism

Make "I cannot read this carrier" unrepresentable as a silent outcome, rather
than banning a spelling. This is [#729]'s optionality removal and [#908]'s
unrepresentable-domain shape applied to the reader side, and `ChildRef` in
`chelis-deep/src/node.rs` is the existing precedent for its form: a role-tagged
enum a traversal must exhaust, not an `Option` it may drop.

1. **One shared total accessor in `chelis-deep`.** It returns an enum over
   every admitted carrier - a decoded vocabulary node, a structural bare list,
   an undecodable head, an atom, a metadata map - not an `Option`. A caller
   that cares only about `Def` still has to write, or explicitly delegate, the
   arm for the carrier it cannot use. The seven private helpers collapse into
   it. The accessor is the only sanctioned reader, and it is public and
   testable rather than private and seven times duplicated.

2. **Delete `Node::to_list` from reader paths.** Finding 3 shows the shallow
   bridge is not a partial migration but a defect that hides itself. §C4.2
   already names the bridge for deletion; PP7 deletes it where readers use it,
   which does not wait on [#1029]'s enum deletion and reduces its blast radius.

3. **The lint is the ratchet, not the mechanism.** With (1) landed, the rule
   the lint states is "no `Expr::List` pattern outside `chelis-deep` and the
   recorded legacy producers", which is decidable by inspection and catches a
   List-only helper **definition** as directly as a use. That answers the
   audit's three lessons: it scopes past `infer/`, it fires on the helper
   definition rather than only on call sites, and its planted-violation corpus
   must include a guarded arm, since the audit's classifier blind spot was
   exactly a `match` arm whose guard contains `==`.

#### You deliver

- **E5a, the reproducers (lands first). Delivered by PR [#1543].** The six programs above - the four
  in-checker divergences and their two controls - as rows in
  `crates/chelis-types/tests/issue_1107_stamped_node_ingress_parity.rs`, whose
  `agreed_diagnostics` helper already asserts the exact invariant. The
  delivered slice carries thirteen rows rather than six: the `deftype` and
  `t-tensor` programs each took an over-rejection control, because their
  repairs newly enable a rejection; `describe_tuple_index` took the negative
  twin that proves it, its accepted control being the well-formed `tuple-get`
  program already in the set; `param_name_and_inline_type` took its own
  rejection row and control, none of the six programs reaching it; and the
  seed-literal `type:` reader in `infer/expr.rs` was found during the slice to
  be an eighth in-checker divergence and took the same pair. Each row proved
  red on the pre-fix tree before its site is touched. The three
  divergences outside the checker are E5c's rows, in E5c's crates. Then the
  sites the rows cover: the `type:` metadata reader in `infer/expr.rs`, the `t-prim`
  read and `param_name_and_inline_type` in `infer/validate.rs`, the four
  private helpers in `invariants.rs`, and `tuple_get_index` in
  `infer/expr_record.rs`. Roughly 300 hand-written lines.
- **E5b, the accessor.** The `chelis-deep` accessor, the seven-copy collapse,
  and the reader-path `to_list` removals. Roughly 320 lines if the collapse is
  mechanical, materially more if the `Option`-to-enum change ripples through
  callers; slice again on that evidence rather than growing one pull request.
- **E5c, outside `chelis-types`. Delivered by PR [#1546].** `prune::deep_def_name`
  and
  `deep_named_decl_name`, `prune_to_entry`'s module descent,
  `tier_b_lower`'s `tag`/`children`/`lookup_producer`, and
  `count_invariant_opaque_deep`, whose raw-tag read also owes a decode-once
  regression row. It also owes `prove_deep_obligations.rs` a runner: that file
  already asserts the correct tier and nothing invokes it, so E5c's own fix
  would land unverified on the same terms. E5c therefore carries all three
  outside-the-checker rows, each in the crate that can reach it. Roughly 180
  lines plus the CI step. Delivery found four more readers this list does not
  name, all in `tier_b_lower.rs` and all required before the obligation would
  lower: `lookup_producer`'s module descent and its `BareList` parameter read,
  `rewrite_opaque_field_access`'s recursion, and `rebuild`'s tag copy. Making
  `tag` and `children` carrier-complete was measured, not assumed, to be
  insufficient on its own.
- **E5d, the lint and its corpus.** Roughly 250 lines.
- **E5e, the remaining sites.** The 59 unadjudicated guarded-arm sites and the
  19 never-adjudicated ones the audit inventories, swept behind E5b so the
  sweep has one accessor to route to. Unbounded until E5b lands; do not
  estimate it before then.

The five slices exceed one pull request's hand-written budget together. E5a is
one pull request; E5b with E5c is a second; E5d is a third. E5e is its own,
sliced on the evidence E5b produces, because a slice with no estimate cannot
be budgeted against the size cap alongside one that has an estimate.

#### The oracle

PP7's authoritative completion oracle is one named set, the **PP7 parity set**:
four commands, one per group of rows. A single command cannot span it, for the
same reason that decided the mechanism: three of the seven divergences sit
outside every checker entry, and no `-p chelis-types` run reaches them.

| rows | the command that owns them |
|---|---|
| the four in-checker divergences and their two controls (E5a) | `cargo nextest run -p chelis-types --test issue_1107_stamped_node_ingress_parity --no-fail-fast` |
| the tide `/lower` entry-pruning row (E5c) | `cargo nextest run -p chelis-compiler-api --lib prune --no-fail-fast` |
| the Tier-B downgrade row (E5c) | `CVC5_DIR=<cvc5 store> cargo nextest run -p chelis-cli --features smt --test prove_deep_obligations --no-fail-fast` |
| the `count_invariant_opaque_deep` decode-once row (E5c) | `cargo nextest run -p chelis-cli --test prove_type_check_gate --no-fail-fast` |

Acceptance is every row green under the command that owns it. The tide row is a
`chelis-compiler-api` row because it exercises a pre-checker consumer, and
`chelis-compiler-api` already depends on `chelis-types`, so it cannot be a
`chelis-types` row. The last row runs in the default build by construction: the
warning it exercises fires only when `smt` is off. The third is the CI step E5c
owes, and today nothing runs it.

E5d's ratchet is proved by a different command again, because a lint is not a
test: plant a bare `Expr::List` destructure in a guarded match arm inside
`infer/`, and `chelis lint --check .` must reject it; removing the plant must
make that command green.

#### What PP7 does not establish

- **At PP7 delivery, axis B was not closed; [#1537] owned it.** `validate_ir_program` ran on the
  serialized-IR entry only, and the two entries drive different inference
  functions (`infer_ir_program_with_state` against
  `infer_program_with_product_in_session`). Unifying them is a driver merge,
  not a carrier repair, and PP7 does not absorb it: it collides with PP6's
  schedule work in `program.rs`, and the repair's shape depends on which
  entry's pass set is the correct one, which is a question no probe here
  answers. PP6 closes [#1134] on the forward-reference and schedule questions
  and names neither `validate_ir_program` nor the pass set, so the asymmetry
  outlived its former tracker and received its own. PP9 below dispositions
  each pass separately, records the choice as decision row 20, and closes that
  axis.
- **No universal reader claim.** The oracle proves the listed rows and
  whatever the lint's corpus plants. It does not prove that no reader remains
  carrier-incomplete; E5e's inventory is the honest statement of what is
  unswept.
- **The tide MCP prove route is not affected, and the source says otherwise.**
  `run_deep_source_obligations` normalizes through
  `deep_compat::parse_file_to_lists`; `prove_deep_file` does not. The comment
  at `obligation_engine.rs` asserting that a prove through tide is identical to
  the CLI `chelis prove foo.dp` path is therefore false on this head, in the
  CLI's disfavour. PP7 records it; correcting it belongs with E5c.
- **One recorded site did not reproduce.** The `pipe_stage.rs` auto-borrow
  misclassification did not diverge on
  `add(x |> shape(0), x |> shape(0))` at this head; both surfaces accept.
  The site is still List-only and stays in E5e's inventory, unconfirmed.

#### Risks and overlaps

The [#1277] stream owns `unify.rs`, `infer/app.rs`, `app_post.rs`,
`expr_record.rs`, and `builtins.rs`; E5a touches `expr_record.rs` and must
coordinate before editing it. PP6's [04-INF-4] schedule work owns `program.rs`;
PP7 touches no entry function there, which is also why axis B is excluded
rather than deferred. [#1029]'s deletion is helped, not blocked: E5b shrinks
the `Expr::List` reader surface that deletion needs empty, and it changes no
producer, so [#1320]'s macro-hygiene precondition is untouched.

PP7's normative authority is [04-TOT-5]: a program's verdict does not depend on
which entry receives it or which admitted representation carries it, and a
representation a check cannot read is a silent exemption under [04-TOT-1]
rather than an absent subtree. Before that atom, no numbered spec required
the two entries to agree except [04-INF-4]'s clause for top-level value scope,
so PP7 could not have decided the rule for itself.

### PP8. Source-coverage totality ([#874], [#887])

**Opened 2026-09-03; Slice 1 delivered by PR [#1602].** §C4.1's invariant
quantifies over the checked result.
[#874] proposed the complementary obligation over the submitted program and
recorded three tag-keyed routes to the vacuity. This item establishes three
things by execution. The vacuity is **live on `main` today**. The live routes
are **role-keyed, not tag-keyed**: every one sits in a child slot the walk
declines to enter, and the tags involved are ordinary vocabulary tags with
ordinary dispositions. And the mechanism the class needs is a typed read at
the slot, **not** a further tightening of the stamp walk, because no stamp
walk can reach a node that legitimately carries no stamp in either the
accepted or the rejected case.

#### The live instances

Executed 2026-09-03 against `6fd95fd52` on the `.dp` CLI ingress
(`chelis check --allow-style-violations <file>`). Each row is a complete
program; "vacuous" means score 1.0 with an empty error vector on a program the
spec says is ill-formed. The paired loud row is the same slot with a readable
child, so each pair isolates the extraction failure rather than the form.

| # | form and slot | probe child | verdict |
|---|---|---|---|
| R1 | `vmap` axis, `child_stamp_role(Vmap, 1) = Selector` | `(var {} nonexistent_name_zzz)` | **vacuous**, score 1.0 |
| R1 | same | `(lit {type: (t-prim {} f32)} 1.5)` | **vacuous**, score 1.0 |
| R1 | same | `(t-prim {} f32)`, a type node in a value slot | **vacuous**, score 1.0 |
| R1 | same | `(app {} (var {} missing_fn_qqq) (var {} missing_arg_www))` | **vacuous**, score 1.0 |
| R1 control | same | `(lit {type: (t-prim {} i32)} 2)`, rank-1 operand | loud: `vmap axis 2 is out of bounds for rank 1 tensor` |
| R1 control | same | `(lit {type: (t-prim {} i32)} -7)` | loud: `vmap axis must be non-negative, got -7` |
| R2 | `pat-ctor` head, `child_stamp_role(PatCtor, 0) = Selector` | `(app {} (var {} missing_fn_qqq) (var {} missing_arg_www))` | **vacuous**, score 1.0 |
| R2 control | same | `NoSuchCtorZZZ`, a bare symbol | loud: `unknown constructor: NoSuchCtorZZZ` |
| R3 | `pat-record` head, `child_stamp_role(PatRecord, 0) = Selector` | `(var {} nonexistent_name_zzz)` | **vacuous**, score 1.0 |
| R4 | `grad` operand, `child_stamp_role(Grad, 0) = RuntimeExpr` | an `f32`-typed non-function | **vacuous**, score 1.0 |
| R4 control | `vmap` operand, same role | an `f32`-typed non-function | loud: `vmap expects a function, got f32` |
| R5 | `kv` key, `child_stamp_role(Kv, 0) = Selector` | `(var {} nonexistent_name_zzz)` | loud, but misattributed: `internal: annotation owner-stamp invariant violated: missing authoritative type stamp for metadata-eligible expression `lit`` |
| R5 control | same | `f`, the declared field name | clean pass |
| R5 control | same | `nosuchfield` | loud: `unknown record field 'nosuchfield' in construction of R` |

The implementation pass re-ran this table on `3b701e54b` and found four further
live instances of the same class, in the same four inference functions, that
the probe pass above missed; they are recorded here rather than left to the
reader to rediscover.

| # | form and slot | probe child | verdict |
|---|---|---|---|
| S1 | `pat-var` name, `child_stamp_role(PatVar, 0) = Binder` | `(var {} zz)` | loud, but misattributed: `internal: ... missing authoritative type stamp for pattern binding `pat-var`` |
| S2 | `pat-as` name, same role | `(var {} zz)` | loud, misattributed the same way, naming `pat-as` |
| S3 | `kv` key inside a `pat-record`, `child_stamp_role(Kv, 0) = Selector` | `(var {} zz)` | **vacuous**, score 1.0 |
| S4 | `kv` key in `record-update`, same slot | `(var {} zz)` | loud, misattributed exactly as R5 |

S3 is the third `kv` key read in the checker, beside `infer_record`'s (R5) and
`infer_record_update`'s (S4); it defaulted to a fresh type variable rather than
skipping, which is why it is fully silent where the other two are misattributed.
S1 and S2 sit in `Binder` slots, so they are named instances and not a
statement about that role, which stays unenumerated below.

The comparison rows matter more than the vacuous ones, and the `Selector`
role is small enough to enumerate exhaustively. `child_stamp_role` is total
over `DeepTag`, and six of its match arms yield `Selector` at some child
index. Because `PatCtor | PatRecord` and `Grad | Vmap` each cover two tags,
those six arms are eight tag-and-index slots. All eight were probed:

| slot | unreadable child | verdict |
|---|---|---|
| `access` index >= 1 | a two-child expression | loud: `malformed 'access': expected a symbol field name as its second child` |
| `tuple-get` index >= 1 | a `var` | loud: `invalid tuple index: expected a non-negative integer literal, found a non-literal expression` |
| `cast` index >= 2 | a `var` | loud: `(var {} nonexistent_name_zzz)` is not a recognized cast mode selector |
| `grad` index >= 1 | a `var`, operand a function | loud: ``grad `wrt` must be an integer parameter index or tuple of indices`` |
| `kv` index 0 | a `var` | loud, misattributed (R5) |
| `vmap` index >= 1 | a `var`, an `f32` literal, a type node, an application | **silent** |
| `pat-ctor` index 0 | an application | **silent** |
| `pat-record` index 0 | a `var` | **silent** |

Five of the eight reject, three accept. The three that reject with a good
diagnostic all spell the read as a total decision over the child; the three
that accept all spell it as a partial extraction whose failure branch is a
default. `record`'s constructor head is a `Type` slot rather than a
`Selector` one and rejects with the same shape of message, which is the
point: the obligation is met under two different roles and unmet under a
third, so the class is not the role. It is the spelling of the read.

R5 is the instructive middle case. The `kv` key is unreadable, the field is
never resolved, and the program IS rejected - but by §C4.1's own tripwire
firing on the unstamped `lit` in the value slot, reported as an `internal:`
owner-stamp violation naming `lit`. The invariant caught a coverage gap only
because the collateral node happened to require a stamp, and the diagnostic
it produced blames a node the author did not write wrongly. That is the
invariant working at the edge of its reach and still not being the right
instrument: it detects, but it cannot say what went wrong.

#### The four defect sites

1. `crates/chelis-types/src/infer/expr_transform.rs:308` -
   `let axis = kids.get(1).and_then(extract_int_for_dim).unwrap_or(0);`.
   One `unwrap_or` serves two different inputs: `kids.get(1) == None`, where
   the default is correct because spec/02 §0.1 writes the zero axis as bare
   `vmap(f)`, and `Some(child)` that `extract_int_for_dim` cannot
   read, where the default silently discards a node the program submitted.
   The comment above it claims defense-in-depth for Deep-direct callers; it
   peels casts and screens negatives, and passes everything else through as
   axis 0.
2. `crates/chelis-types/src/infer/expr_pattern.rs:190` (and its three
   siblings at `:176`, `:259`, `:282`) -
   `if let Some(ctor_name) = kids.first().and_then(|e| symbol_name(e))` with
   no `else`. A non-symbol head falls off the binding and the whole pattern
   is skipped, so the arm neither binds nor rejects.
3. `crates/chelis-types/src/infer/expr_transform.rs`, `infer_grad`'s
   `_ => { vg.fresh_type() }` arm, commented "Can't determine function
   structure, return fresh var". R4's primary class is therefore the silent
   fallback of [04-TOT-1], not coverage; its coverage consequence is
   secondary, since `kids[1]` is never reached on that path. It is listed
   here because the sibling `vmap` rejects the identical input and the two
   were written to the same template.
4. `crates/chelis-types/src/infer/expr_record.rs:335` (and its sibling at
   `:676`) -
   `let (Some(field_name), Some(value)) = (kv_kids.first().and_then(symbol_name), kv_kids.get(1)) else { continue; };`.
   R5's site. The `continue` skips the `infer_expr(value, ..)` two lines
   below, which is why the value goes unstamped and §C4.1 fires on it.

Three spellings, one defect: `unwrap_or(default)`, `if let Some(..)` with no
`else`, and `else { continue }`. Each converts "I could not read this child"
into "there was no child", and that conversion is what [04-TOT-4] forbids.
Naming the three spellings is what makes the class greppable, and it is why
the fix belongs at the read rather than in a walk.

#### Why §C4.1 cannot see any of them, and why Phase 3 does not help

Both halves of the walk - `annotated_totality_invariant_traces`
(`infer/validate.rs:~2816`) and `register_annotation_owners`
(`infer/checked.rs:~977`) - recurse only into `RuntimeExpr`,
`ExplicitInferenceBypass`, and untagged children. The `Syntax`, `Selector`,
`EffectHandler`, `Binder`, and `Type` arms return without descending. R1
through R3 sit in `Selector` slots, so the walk never arrives. Even if it
did, it would require nothing: `should_attach_type_metadata(DeepTag::Var)` is
`false` (`infer/annotate.rs:~556`), so a `var` node carries no stamp
obligation in any slot, and a hand-written `.dp` supplies its own `type:`
metadata for the nodes that do.

Phase 3's `DeepTag` exhaustiveness does not close it either, and the reason
generalizes past [#874]'s original wording. That wording says an untagged
list has no tag to be exhaustive over. The stronger statement is that
exhausting the tag of the **parent** says nothing about whether the parent's
disposition **read the child**. `vmap` has a disposition, that disposition is
exhaustive over `DeepTag`, and it still discards its axis.

**Relation to PP2.** PP2 is a node the checker visits and learns nothing
from, because the disposition it reaches is a generic signature. PP8 is a
node the checker never reads at all, because the disposition it reaches
defaults instead. Neither subsumes the other, and R4 sits on the seam: a
visited node whose silent disposition causes a second node to go unread.

#### The fitness ratio is measured over the walk's own image

`CheckResult.total_nodes` is `InferStats::total_nodes`, incremented once per
`infer_expr` entry (`infer/expr.rs:90`) and surfaced by
`clean_fitness_from_stats` (`fitness.rs:~202`). It counts nodes inference
**visited**, not nodes the program **contains**. `typed_nodes/total_nodes` is
therefore a tautology over the visited set - "every node I looked at, I
typed" - and cannot fall when a node goes unlooked-at. Executed: the valid
`axis=0` program and the R1 unbound-variable program are byte-identical in
the report, both `typed_nodes: 4, untyped_nodes: 0, total_nodes: 4`. The
`fitness.rs:166-190` comment promising "its counts are now the checked truth"
holds only for the nodes the walk reached.

A parsed count does exist in the same function. `analyze_ir_program`
(`fitness.rs:~172`) computes `structural_stats(exprs)`, whose `total_nodes`
is `count_nodes` over the parsed tree, and hands it to
`clean_fitness_from_stats` (`fitness.rs:190`) alongside the inference stats,
which uses it only for the validator-warning structure score and never
compares the two. That comparison is not the repair, and this item does not
propose it: the two populations are not directly comparable, since the
structural count includes atoms, maps, and metadata values inference never
visits by design. The paragraph is here to explain why the ratio is silent,
not to argue for a second counter.

#### Relation to [04-TOT-3], and what [04-TOT-4] adds

[04-TOT-3] governs this class already, and the implementation agrees. The
`access` and `record` rejections are abbreviated in the tables above; in full
each ends `(spec/03-deep-syntax.md; chelis#731 [04-TOT-3])`, so an unreadable
child in a selector-shaped slot is an [04-TOT-3] rejection wherever it is
implemented. R1 through R3 are **unimplemented [04-TOT-3] cases, not a hole
in [04-TOT-3]'s wording**, and an implementer fixing them cites [04-TOT-3],
the same as the adjacent shipped code.

[04-TOT-4] therefore extends rather than replaces. It carries [04-TOT-3]'s
obligation from the form down to each of the form's slots, and it adds two
sentences no earlier atom states:

1. **An omitted optional child and a present unreadable one are distinct
   inputs, and only the omission may default.** Nothing before this says so,
   and R1 is exactly the conflation: one `unwrap_or(0)` serving both.
2. **Coverage quantifies over the submitted program, not the checked
   result.** This is the one [04-TOT-2] structurally cannot express, and R1
   is the proof: R1 satisfies [04-TOT-2] completely - empty error vector, no
   `Type::Error` anywhere - and is still wrong. A missing quantifier, not a
   missing rule.

#### What we deliver

**The primary mechanism is a typed selector read, not another walk.** No
stamp-walk tightening can reach R1 through R3: the walk inspects stamps, and
the discarded nodes legitimately carry none in either the accepted or the
rejected case. What distinguishes the four correctly-rejecting slots
(`access`, `tuple-get`, `cast`, `grad`) from the three silent ones is the
spelling of the read. So the structural fix belongs at the read:

1. **Every role-slot read returns a result, never an `Option` with a
   default.** A `Selector`, `Binder`, `EffectHandler`, or `Type` child is
   read through one seam that takes the parent tag, the child index, and the
   expected shape, and returns either the extracted value or a pushed
   `MalformedForm` naming the form and the shape. The three defect
   spellings named above do not survive it: there is no `Option` at the
   call site to default, skip, or `continue` past.

   *Implementation note.* The seam owns the decision, the guarantee, and the
   message; the caller supplies no text. It returns a `Result` whose error arm
   is an `ErrorWitness`, which only `report_witness` can mint, so no caller can
   produce a value from the unreadable branch without visibly laundering a
   witness of a diagnostic that has already been pushed. It takes the parent
   `DeepTag` and a `SlotShape` -- an enum with a `Display`, not free text -- so
   the message always names both and the kind is always `MalformedForm`.

   Slice 2 is where that becomes a live tension rather than a design
   statement. The four slots it migrates already reject correctly, and two of
   them say more than a shared template can: `tuple-get`'s diagnostic peels a
   `lit` wrapper to name the found atom's family and value. Slice 2 therefore
   adds the affordance for a caller to append its own found-shape detail, with
   `tuple-get` as its first consumer, and changes those four slots' diagnostic
   text and kind rather than preserving them. Slice 1 has no such caller and
   carries no such affordance: a seam that let each caller supply its own
   message would guarantee nothing, and one that carried the parameter before
   any caller used it would be a mechanism ahead of its use.
2. **Absence and unreadability become different types at the seam.** The
   `vmap(f)` default axis is legitimate and stays; it is expressed as
   `kids.get(1)` being `None`, which the seam distinguishes from a present
   child it could not read. This is the whole of R1, and it is why the repair
   is not "delete the default".
3. **Both Deep carriers reach the same read.** `infer_expr` dispatches
   `Expr::List` and `Expr::Node` to the same `infer_vmap` and `infer_grad`
   (`infer/expr.rs:143-144` and `:442-443`), so a typed read placed at the
   `Selector` slot inside those functions is reached from both carriers
   through that dispatch. That is the whole of the claim: PP8's oracle runs
   its rows through `chelis check` only and claims nothing about the two
   ingresses agreeing. Ingress agreement is PP7's axis under [04-TOT-5], and
   an ingress-parity row for a PP8 program belongs in PP7's parity set
   rather than here.

**Explicitly not delivered here.** Two things. Any change to which slots are
legal bare lists: the sanctioned set stays exactly as `desugar.rs`'s
`bare_list()` builds it, enumerated below. And the parsed-vs-checked census,
which closes none of R1 through R5 and is recorded as an open question rather
than a deliverable - see decision row 18.

#### Oracle and ratchet

**Oracle:** `cargo nextest run -p chelis-types --test
issue_874_source_coverage_totality`, a new file whose rows are the probe
table above driven through the programmatic ingress, plus the positive
controls.

**Negative rows (regression tests, red on `6fd95fd52` before any fix).** R1's
four unreadable axes, R2's non-symbol constructor head, R3's non-symbol
record-pattern head, and R4's non-function `grad` operand each assert a
pushed diagnostic naming the form and the expected shape. All seven are red
on the current head by construction: the probes above show them accepting at
score 1.0 today, so the assertion cannot pass until the seam lands. They need
no revert step to be proven red, and this design PR does not run them - it
specifies them.

R5 needs a different assertion, because it is already rejected. Its row
asserts the diagnostic's *identity*: an unreadable `kv` key must produce a
`MalformedForm` naming `kv` and the expected symbol key, not the `internal:`
owner-stamp violation naming `lit` that it produces today. That row is red
now for the reason that matters - the message is wrong - and it is the row
that would otherwise be skipped, because the program already fails and a
coarser test would call that good enough.

**Positive controls (disposition locks, green in both states).** The
sanctioned bare-list slots must stay accepted, enumerated from
`crates/chelis-surf/src/desugar.rs`'s `bare_list()` (defined `:481`):
`fn` parameter lists (`:1200`, `:1545`), an import name list (`:1287`),
a qualified import's empty name list (`:1297`), the empty match guard (`:1804`), and
`:1836`'s empty list, which is the unit literal's. Add `span` metadata values,
ADT type-parameter lists, `vmap(f)` with no axis child at all, `vmap(f, axis=0)`,
and a `pat-ctor` whose head is a legitimate in-scope constructor. Their job is to prove the
seam rejects unreadability rather than non-tagged-ness; without them the
cheapest wrong fix - rejecting every bare list - passes the negatives.

**Mutation receipt.** Four staged mutations, each run before and after.
Restoring `unwrap_or(0)` at `expr_transform.rs:308` must redden exactly R1's
four rows and leave the positives green. Reverting the read at
`expr_pattern.rs:190`, the `DeepTag::PatCtor` arm, must redden R2 and only
R2. Reverting `expr_pattern.rs:282`, the `DeepTag::PatRecord` arm, must
redden R3 and only R3; the two arms are separate and a mutation at one leaves
the other's row green. Reverting the `continue` at `expr_record.rs:335` must
redden R5's attribution row while leaving the program rejected, which is the
mutation most likely to be mis-run, because a coarser assertion passes in
both states.

**Prove-red-first is free here and must not be skipped anyway.** Record each
negative's exact failing command and output on the pre-fix tree in the
implementation PR; the probe transcript in this PR's body is the design-time
evidence, not that record.

#### Scope, slices, and overlap

Roughly 150 to 250 hand-written lines across two slices, each shippable
alone:

- **Slice 1 - the four instances.** `expr_transform.rs:308` and
  `infer_grad`'s fallback arm; `expr_pattern.rs:176/190/259/282`;
  `expr_record.rs:335/676`. The negative and positive rows above. Smallest
  honest unit, and it closes the named live instances without any new
  mechanism.
- **Slice 2 - the selector-read seam.** One helper, then migrate the four
  slots that already reject correctly (`access`, `tuple-get`, `cast`,
  `grad`) onto it, so the seam is proven against known-good behavior before
  it is trusted for new rejections. `kv` is not in that set: it rejects, but
  by the wrong instrument, so it migrates with Slice 1's repair. `chelis-deep/src/role.rs`
  gains the expected-shape half beside `bypass_child_expectation`
  (`role.rs:~232`), which already names exactly this concept for
  `ExplicitInferenceBypass` and needs extending to `Selector`.

**Overlap.** `chelis-deep/src/role.rs` is shared with [#908]'s carrier work;
Slice 2 extends `bypass_child_expectation`'s neighbourhood rather than
`child_stamp_role` itself, so the role table's totality test is untouched.
`normalize_nodes_to_lists` (`infer/program.rs:~1931`) rewrites `BareList` to
a tagless `Expr::List` at the serialized-IR ingress while the typed ingress
keeps `BareList`; [#1125]'s ingress-parity work may unify those, and PP7
owns that axis. Neither slice here depends on the outcome. Every Slice 1 site
sits under a function `infer_expr` dispatches from both carrier arms -
`infer_vmap` and `infer_grad` (`expr.rs:143-144`, `:442-443`), `infer_record`
(`:134`, `:436`), and `infer_match` (`:127`, `:430`), which reaches the two
pattern reads through `pattern_bindings`, itself keyed on the carrier-total
`stamped_parts`. Nothing here depends on the [#1085] BareList
disposition work,
which governs expression position where the checker is already loud.

#### What this does not establish

The `Selector` role IS enumerated: `child_stamp_role` is total over
`DeepTag`, its eight `Selector` arms were read off that table, and each was
probed, so "three of the eight accept an unreadable child" is a claim about
the whole role and not a sample. Nothing else here is.

No claim is made about the `Binder`, `EffectHandler`, `Syntax`, or `Type`
slots, which the walk skips on the same footing and which were spot-checked
at four points, not enumerated: `deftype`'s type-parameter list, `params`
entries, `handle-effect`'s kind child, and `record`'s constructor head all
rejected, which is evidence of health and not proof of it. Nor is any claim
made about `ExplicitInferenceBypass` slots, where a dedicated owner is
supposed to visit and this design did not verify that one does. Converting
the remaining roles from spot checks into a statement needs an enumerator
this item does not deliver (decision row 18); until one exists, the honest
scope is the eight `Selector` slots.

One ingress was exercised: the `.dp` CLI path, which is
`chelis-compiler-api`'s `check` beneath. `chelis prove`, the typed
`check_typed_program` ingress that preserves `BareList` where the serialized
ingress normalizes it away, and the `check_in_context` surface - which
hardcodes `score: 1.0` and an empty error vector for any compile that
returns `Ok` (`compiler.rs:2398`) - were not probed. That last one is worth
a look under this class's lens and is not claimed either way here.

The cancellation route [#874] records as its fourth, coverage-keyed door was
checked and is **closed at the surface it named**. PR #934 never merged. The
cancellation work that did land came in under [#930] and carries an explicit
`bail_if_cancelled("check")` after fitness is computed
(`crates/chelis-compiler-api/src/compiler.rs:829`) whose comment - cited to
chelis#930, not #934 - names this exact failure: "a cancelled walk must not
become a CheckResult". Whether
every other `Ok`-returning front-end entry has the same guard was not
audited. That audit is not this item's, and it is not needed for the four
named instances, none of which involves cancellation.

### PP9. Ingress pass-set parity ([#1537]; axis B of the ingress-parity family)

**Implemented 2026-09-16.** PP7 closed the carrier axis and named this one
residue: at PP9's opening, `validate_ir_program` ran on the serialized-IR entry
only, the two entries drove different inference functions, and which pass set
was correct was a language question no carrier probe answered. [04-TOT-5]
already made the asymmetry a violation. This item records and implements the
surviving set.

The delivered driver is `validate_semantic_program`. It reports eager
initialization defects, enforces the launch-core transform fence, checks
`vmap` extent dependencies, and applies the spec-owned `conv` static domain
rules at all four public entries. It preserves stamped input. The implementation
uses the same two bounded prebinds at both entries: declaration-local
external-input ascriptions under [04-INF-4], and compiler-authored body stamps
only as defsig-less callable headers under [04-INF-2]/[04-INF-3]. Eager values
are never prebound from body metadata. The PP6 schedule remains the ordering
authority for legal function recursion and forward references.

Everything below was measured on `3b701e54b` by driving a 32-row corpus
through every entry and by applying, running, and reverting three candidate
repairs. The transcripts are the probe reports named in the PR.

#### There are four entries, not two

| entry | driver | reached by |
|---|---|---|
| `check_ir_program` | `check_ir_with_signature_context_in_session` | `chelis build`, `chelis check`, and through `chelis_pipeline_core::semantic` every CLI, Python-binding, tide, and reef surface |
| `check_typed_program` | `check_typed_program_in_session` | `chelis prove`, `chelis-backend-c`, `chelis-effects` |
| `infer_ir_program` | `infer_ir_program_in_session` | `chelis_types::check_ir_fitness` / `analyze_ir_program` |
| `infer_program` | `infer_program_in_session` | `chelis_types::check_program`; no other in-tree production caller |

`check_ir_program(exprs)` is `check_ir_with_context(&TypeEnv::empty(), exprs)`,
and `TypeEnv::empty()` installs the builtins and the prelude ADTs, so the typed
driver's fresh builtin env is not a difference.

#### What execution shows

Eighteen of the 32 rows give different verdicts from `check_ir_program` and
`check_typed_program`. Every one runs in the same direction: the IR-driven
entry rejects or reports more. No fail-closed row appeared; [#1124]'s mirror
image is closed and its row agrees.

| row | program | `check_ir_program` | `check_typed_program` |
|---|---|---|---|
| trivial non-termination, five spellings | `def a(x: i32) -> i32 = a(x)` and its mutual, `sig`-carrying, module-wrapped, stamped-`.dp` and round-tripped forms | REJECT `CycleDetected` | **ACCEPT** |
| conv2d zero stride | `conv2d(&x, &k, cast(0, i32), 0)` | REJECT "requires a positive stride, got 0" | **ACCEPT** |
| conv2d negative padding | `conv2d(&x, &k, 1, cast(-1, i32))` | REJECT "requires non-negative padding, got -1" | **ACCEPT** |
| conv2d non-literal stride or padding, each in both carriers | `conv2d(&x, &k, s, 0)` with `s: i32` a parameter | REJECT "requires a literal integer stride" (respectively "padding") | **ACCEPT** |
| vmap batch-varying extent | a `shrink` bound read from batched elements | REJECT `batch_varying_extent` | **ACCEPT** |
| `mean` over a symbolic axis | `mean(x, 0)` on `tensor[n, f32]` | REJECT, 2 diagnostics | REJECT, 1 |
| `layer_norm` symbolic final axis | | REJECT, 2 | REJECT, 1 |
| elementwise rank mismatch | `add(tensor[2,3,f32], tensor[3,f32])` | REJECT, 2 | REJECT, 1 |
| unknown tag | `(bogus-tag {} ...)` | REJECT, 2 (the second a duplicate) | REJECT, 1 |

Controls that agree, so the corpus cannot pass vacuously: two well-typed
programs, a Deep arity mismatch, an out-of-bounds reduction axis, a
[#1124] `defsig`/body mismatch, an ascribed external-input self-reference, and
a top-level binding cycle.

**A fifth divergence the issue does not list, and a live defect.** [#1457] put
`report_initialization_errors` in `check_typed_program_in_session`, the
wrapper, rather than in `infer_program_with_product_in_session`, the driver
that wrapper shares with `infer_program`. So `infer_program` accepts
`a = b; b = a`, and the public `chelis_types::check_program` returns
**score 1.000 with an empty error vector** for it, against 0.700 and one error
from `check_ir_fitness` on the same program. For a trivially non-terminating
def the same pair reads 1.000/0 against 0.800/1. That is §C4.4's honesty
invariant failing on a public entry. It has no in-tree production caller, but
it is library API and the `chelis-e2e` spec-conformance suite type-checks
through it.

#### Three candidate repairs, measured

**Union alone** — add `validate_ir_program` and the `chelis_deep::validate`
loop to the typed driver, and let the former own the initialization report.
Closes 14 of the 18 check-level rows and all of the inference-level ones. It
does not close the stamped-`.dp` rows, and it is worse than that: run every
Surf row back through `chelis deep` (`print_canonical` then
`parse_and_stamp_file`) and seven of nine round trips still disagree. The
conv2d rows disagree in a new way they did not before — the typed entry now
rejects, but with "requires concrete tensor argument metadata" where the IR
entry says "requires a positive stride, got 0" (and, on the two literal-operand
round trips, where it says "requires a literal integer stride" or "padding").
The pass runs, misreads the tensor metadata off the stamped carrier, and bails
at the concreteness gate before reaching either the range check or the literal
check. The two literal-operand rows in the Surf carrier do close under this
candidate, and that is the point against it: they close by unioning a rejection
that carries no spec sentence and that [05-RWIN-1] forbids as a rejection
shape. Adding the passes without repairing their
readers manufactures a wrong-diagnostic divergence in place of a missing one.
This is PP7's own warning: a mechanism chosen for the wrong axis closes
neither.

**Union plus normalization at the typed entry** — closes 26 of the 27 rows
the corpus held when this candidate was run, and the twenty-seventh is the
`chelis_deep::validate` duplicate. Deleting that pass too closed all 27, across
all four entries. The four conv2d literal-operand rows were added afterwards
and were measured at the base and under the union only, so no claim is made
that this candidate closes them. It is the only measured mechanism that closes
the carrier residual. It is also **not available**: on the full
`-p chelis-types` suite it fails ten tests, and seven of those are
`issue_1023_stamped_checker_boundary` and
`issue_1085_barelist_expression_disposition`, which pin that the typed entry
consumes and returns the stamped representation. §C4.2 requires exactly that,
and PP7 rejected a sixth normalization on the same ground. The doctrinal
objection turns out to have executable tests behind it.

**Union plus dispositions** — selected by the repository owner on 2026-09-08
(decision row 20). The union is right for the
passes that survive their necessity trace; the residual rows belong to passes
that do not survive it, and to two shape readers that PP7's sweep owns. The
three remaining full-suite failures under the union alone are each
informative rather than costly, and none is a reason to keep the asymmetry:

| failing test | what it shows |
|---|---|
| `infer::tests::builtin_conv2d_accepts_int_stride_padding` | the typed lane's own test blesses symbolic conv2d tensor metadata that the IR validator rejects; the two lanes disagree about whether that is legal, and no spec sentence settles it |
| `infer::tests::builtin_layer_norm` | the same for a symbolic normalized axis |
| `slice_c_constrain_contexts::builtin_relation_with_a_resolved_operand_constrains` | the identity-rank validator rejecting a rank pair PP5 Slice C asserts must be accepted. Decision row 15 has already decided that validator goes; this failure is that deletion's own evidence, not PP9's cost |

#### Disposition for every pass in either set

| pass | in | disposition |
|---|---|---|
| module-reopen and forged-linker-name guards, `collect_all_declarations`, opacity, `validate_binder_literal_adoption_in_program`, the PP6 schedule, the deferred borrow and opaque ledgers, `validate_tensor_precisions_in_program`, `validate_type_invariants_in_program_with_sink`, `validate_polymorphic_op_constraints` | all four | shared already; no action |
| `report_initialization_errors` | IR, typed (wrapper), `infer_ir` | **[04-INF-4]/[04-INF-7]/[04-INF-8].** Move into the shared driver so the fourth entry gets it |
| `validate_vmap_extent_dependencies` | IR, `infer_ir` | **spec/06 §3.7: "Such a program is a type error, `batch_varying_extent` (§8.6)".** The spec makes it a *type* error, so it runs at every checker entry. Add, and repair its stamped-carrier read |
| `conv` per-axis stride > 0 and both padding bounds >= 0 | IR, `infer_ir` | **apply [05-OP-51] and spec/05 §4.5 at every entry.** The governing atom now exists; its exact per-axis shape and static-versus-runtime rules replace the former scalar-metadata proposal. PP9 owns consistent enforcement, not a second semantic definition |
| conv2d "requires a literal integer stride / padding" (`extract_typed_scalar_literal`) | IR, `infer_ir` | **delete as a checker rejection; the lowering restriction relocates to [#730].** [05-RWIN-1] decides this by category: for the analogous windowed primitives the stride list "may be runtime values", and the violation "never becomes an empty-list default, truncated rank, **static-parameter signature**, or backend assertion". A demand that the operand be a compile-time literal is exactly a static-parameter signature. A non-literal stride is therefore legal, and the range rule reaches it through the runtime half of the split. That the Phase 0 im2col lowering cannot emit a runtime-strided conv2d today is a capability restriction, and it moves with the concreteness gate below rather than separately |
| conv2d "concrete tensor argument metadata", `mean` concrete reduced axis, `layer_norm` concrete normalized axis | IR, `infer_ir` | **relocate to [#730].** No spec sentence; the rationale is that the IR lowering needs the extent statically. That is a backend capability, and the typed lane's own tests accept the symbolic forms. `loud_unsupported.md`'s class owns it; `chelis check` does not |
| `detect_trivial_non_terminating_fns` | IR, `infer_ir` | **relocate to [#730].** No spec sentence. [04-INF-2]/[04-INF-3] admit recursion at the type level; nothing makes a base-case-less function a type error. Its own comment says the Phase 0 DAG lowering cannot represent recursion and would silently elide it to an identity, which is precisely the loud-unsupported class |
| `validate_identity_builtin_rank_requirements` | IR, `infer_ir` | **delete; already decided.** Decision row 15, PP5 D6 (d). PP9 must land after PP5 D8 PR A or exclude this leg |
| `chelis_deep::validate::validate` | IR, `infer_ir` | **delete.** spec/03 §8.1 and §8.2 name the categories, but the checker already reports the same defects: measured, its only distinguishable output on the corpus is a duplicate unknown-tag diagnostic, and its top-level structural leg is unreachable because `parse_and_stamp_file` rejects an untagged top-level list at the stamp boundary |
| body-stamp prebind (`collect_ir_types_with_origins`) against `collect_literal_external_input_types` | IR/`infer_ir` against typed/`infer` | **implemented as the same two bounded capabilities at all entries.** [04-INF-4] grants an ascribed external-input self-reference its type only within its declaration. [04-INF-2]/[04-INF-3] let a defsig-less function body stamp supply its callable header. No eager value is prebound from body metadata, and an explicit `defsig` remains authoritative. The [#1124] and [#1134] locks cover both boundaries |
| `resolve_owner_types` | IR, typed, `infer`; **not** `infer_ir` | **added to `infer_ir_program`.** The previously named `materialize_deferred_expand_defaults` function no longer exists on the implementation base and therefore supplies no live missing pass |
| `normalize_nodes_to_lists` | IR, `infer_ir`, `infer`; **not** typed | not a checker pass, and not PP9's to add. §C4.2 and `issue_1023_stamped_checker_boundary` forbid a sixth normalization; [#1029] deletes the carrier it bridges |

#### The spec/04 amendment and spec/05 authority

[04-TOT-5] already carries the rule; no amendment is needed for the parity
obligation itself. What it does not say is which side of the divergence must
move, and one added sentence would close the recurring argument:

> *Appended to spec/04 §10 [04-TOT-5]:* A check applied at one
> entry and not another SHALL be resolved by deciding the check, never by
> narrowing the entry that applies it: either every entry applies it, or no
> entry does and the rejection it performed moves to the stage whose
> capability it describes.

The convolution constraint is now owned by [05-OP-51] and spec/05 §4.5.
The canonical `conv` takes explicit per-axis i64 strides and `(low,high)`
padding pairs at every positive spatial rank. That numbered contract owns
the shape relation, domain checks, and static-versus-runtime split; PP9
must consume it rather than author another operation rule. The historical
`conv2d` probes below retain their original evidence spelling. Their
successors use `conv` with per-axis metadata.

That is [05-RWIN-1]'s shape, deliberately. The reviewer's hypothesis was that
the literal-operand requirement is not a checker rule at all, and the spec text
supports it twice over: [05-RWIN-1] says the analogous lists "may be runtime
values", and it names "static-parameter signature" among the forms the
rejection may never take. It is also the shape decision row 15 already took for
`expand` — a literal violation is a check-time type error, a symbolic operand is
a runtime `Domain` trap — so PP9 adopts an existing decision rather than
inventing a second model for the same question.

Nothing about `mean`, `layer_norm`, symbolic conv2d metadata, or termination is
proposed as normative text. Those rejections move to [#730] because they
describe what a backend can lower, not what the language admits.

#### The oracle

PP9's authoritative completion oracle is one command over one file:

```
cargo nextest run -p chelis-types --test issue_1537_ingress_pass_set_parity --no-fail-fast
```

The delivered file drives every row through all four entries and asserts **identical
ordered diagnostics**, not a sorted set: the existing
`issue_1107_stamped_node_ingress_parity` sorts, which cannot see a divergence
in report order, and [04-TOT-5]'s "SHALL report the same defects" is an
ordered claim once two entries push from different passes.

The delivered corpus has 23 logical rows. Every admitted row appears once as
ordinary unstamped Deep and once after canonical print/stamp admission. The
legacy malformed-arity row drives all four checker entries directly and
separately proves that the stamped boundary rejects the malformed carrier.
Regression rows cover recursion, a defsig-less stamped forward function,
static and runtime convolution metadata, `vmap`, symbolic
`mean`/`layer_norm`, duplicate unknown-tag ownership, and the previously
perfect wrapperless eager cycle. Disposition locks retain elementwise-rank,
reduction-axis, [#1124], [#1134], literal-convolution, scalar, and tensor
behavior.

Acceptance is that command green with no row skipped.

#### Sequencing and overlap

`crates/chelis-types/src/infer/program.rs` is the file every stream wants.
PP6's schedule work is merged ([#1542], [#1551], [#1552]), which is why this
item could start; the schedule loop is now character-identical in the two
drivers, so the merge is the prebind reconciliation and one flag, not a
rewrite. The runtime-extents stream's S2b ([#1590]) touches
`infer/validate.rs`, `checked.rs`, and `program.rs`; PP9 deletes one leg of
`validate.rs` and relocates three more, so the two must agree on order before
either edits it. PP5 D8 PR A deletes `shape_honesty.rs`'s identity-rank
validator under decision row 15; PP9 lands after it or excludes that leg
explicitly. PP7's E5 reader sweep owns the stamped-carrier reads inside
`validate_ir_expr` that leave four `.dp` rows open under the union; PP9 cannot
close its own oracle without them, so PP7 E5 leads.

In the [#731] order this sits after PP8 and is the last of the ingress-parity
family. It closes [#1537] and takes PP7's axis-B residue off that item's books.

#### What PP9 does not establish

- **No universal pass claim.** The oracle proves 23 logical rows across every
  admitted carrier. It does not prove that future pass sets are otherwise
  identical; the inventory table is a reading of the two drivers, and a pass
  added to one of them tomorrow is caught only if a row exercises it.
- **Prebinding is deliberately split by declaration kind.** External-input
  ascriptions are declaration-local under [04-INF-4]. A defsig-less function
  body stamp supplies only its callable header under [04-INF-2]/[04-INF-3].
  Body stamps never make eager values module-visible. The [#1124] and [#1134]
  locks cover the known separating cases, not every possible scope interaction.
- **Carrier repairs are bounded to the exercised semantic readers.** PP9 makes
  parameter names, tensor type metadata, let bindings, and `vmap` declarations
  carrier-preserving where the oracle reaches them. PP7 retains ownership of
  the general reader-totality contract.
- **The relocation targets are named, not designed.** Moving termination,
  symbolic conv2d metadata, the two axis-concreteness rules, and the
  literal-operand demand to [#730] says where they belong. What their
  loud-unsupported diagnostics say, and at which stage, is that item's to
  design.
- **Five program classes get looser before they get louder.** Under the
  recommendation, a base-case-less def, a conv2d with symbolic tensor metadata,
  a `mean` or `layer_norm` over a symbolic axis, and a conv2d with a runtime
  stride or padding are all accepted by every checker entry from the moment
  PP9's change lands until [#730] lands their loud form. That is the price of
  parity-by-subtraction and it is deliberate, but it is a real regression in
  loudness for the window between the two, and the change that lands it says so
  in its own body. The two conv2d relocations in particular must move together:
  the metadata-concreteness gate returns early ahead of the literal-operand
  check today, so relocating concreteness alone would make the literal demand
  fire on strictly more programs.

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

### Delivered residue: top-level forward-reference parity ([#1134])

The earlier pre-PP6 schedule narrative in this location is retired. The
delivered PP6 section above is the current implementation record: source
position controls eager-value visibility, one complete top-level reference
graph drives both mixed-component inference and eager-cycle diagnostics, and
both checker ingresses report the same result. There is no recursive-only
schedule, stall release, separate cycle-detector graph, or pending
[#1485]/[#1486]/[#1487] work in this residue. PP6's paired-ingress, schedule,
and public CLI commands are its standing acceptance oracles.

[04-INF-8] closes [#1339]'s indirect form at the same checker boundary: an
earlier eager value may not call through a function or nested lambda whose
eager closure reaches a later non-function value. The dedicated [#1339]
oracle owns this indirect frontier; the compiled-value ownership oracle keeps
its direct rows, and [#1362] invokes both. No dependency-ordered global
initialization is introduced, and the compiled-value ownership class is not
enlarged.

#### Top-level initialization frontier ([#1339])

**Decision.** [04-INF-8] rejects an acyclic eager reference set that contains a
later non-function value as `UnboundVariable`. Reordering compiled globals was
rejected: top-level expressions may perform `IO` or trap, eval forces a missing
dependency from inside the earlier initializer, and moving the later whole
initializer ahead of it would choose a different observable order. Matching
eval instead would require per-global state, forcing accessors, memoization,
runtime cycle handling, and owned cached heap values. That is a new runtime
model, not an emitter ordering repair.

**Dependency order.** PR [#1457] supplied the direct rejection and PR [#1516]
decided [04-INF-7]. PP6 Slice A ([#1486]) lands first; Slices B/C ([#1487] and
[#1485]) then provide the lambda-complete graph and ingress-identical cycle
precedence. The [#1339] implementation reuses that graph and checks every
eager root's acyclic closure against the graph's declaration ordinals.
Imported library values are already available and are excluded. A cycle is
diagnosed first as `CycleDetected`; the later-value diagnostic never replaces
it. For a cyclic component that also references a later eager value, the same
graph adds a checker-only availability edge that infers the later value first;
the component's transactional visibility capability then exposes that exact
binding only while the already-rejected component is co-inferred. This is not
runtime initialization reordering, and an unrelated unknown name remains an
`UnboundVariable` beside the cycle.

**Deliverable and oracle.** One CLI integration target owns both checker
ingresses and the public lanes:

```sh
cargo nextest run -p chelis-cli --test issue_1339_top_level_initialization --no-fail-fast
```

Its negatives are scalar, `List`, tensor, multi-function, nested-lambda, and
later-external-input dependencies. Its controls are the direct-forward rows,
backward scalar/heap dependencies, an independent later value, legal forward
and recursive functions whose value closure is already available, and exact
eval-versus-compiled-C output including root order. `check`, `prove`, `eval`,
and `build` reject before proof work, execution, lowering, or artifact
creation. [#1362]'s launch gate runs this complete target beside the ownership launch
subset; the ownership oracle's direct rows alone are not evidence that [#1339]
is closed.


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
  `i32` scrutinee scored 1.0 and `chelis eval` printed a result from an arm
  that can never match. The numbered spec had not decided the rule: §3's Match
  rule never defined `bindings` for `pat-lit`, and [04-LIT-1]'s closed
  atom-to-primitive matrix is scoped to a literal's declared `lit` metadata,
  which a `pat-lit` structurally cannot carry. [04-PAT-1] and a `bindings`
  definition at the Match rule were authored first, then implemented. The rule
  is family agreement rather than unification, because a `pat-lit` admits no
  suffix: unifying with §5.3's `i32` default would reject a `| 1 =>` arm over
  an `i64` scrutinee and leave no spelling for an `i64` literal pattern.
  Two sub-clauses are errors because each arm is provably dead: a non-primitive
  scrutinee admits no literal pattern, and an integer pattern outside the
  scrutinee width's range is rejected under §5.3's and §5.6's range rule. The
  check sits in the one recursive walk both ingresses share, so the tuple,
  record, and constructor nestings are the same finding at depth, and the tests
  assert ingress parity rather than assuming it. The evidence is
  `crates/chelis-types/tests/issue_1494_literal_pattern_scrutinee.rs` and
  `crates/chelis-cli/tests/issue_1494_literal_pattern_cli.rs`, whose rejections
  were proved red on the pre-fix tree, with the score-1 inputs also in §C4.4 as
  the float-versus-`i32` and out-of-range-`i8` members beside a
  matching-family positive control. One boundary is recorded rather than moved:
  a `pat-lit` whose child is not a scalar atom still scores 1.0, because that is
  Deep well-formedness rather than typing, and it is tracked as [#1525].

---

### Checked collection-operation transport (chelis#1654)

[04-INF-9] controls this slice. A newly authored generic wrapper may not publish
a collection admission predicate inferred from its body: `def size(x) = len(x)`
and an authored `a -> i64` signature both fail at their declaration boundary.
An explicit `List[a]` or `Dict[k, v]` parameter supplies the constructor
information the operation requires. [04-INF-1] still permits a local
unannotated lambda to settle monomorphically at its first application.

An already-checked function value is different. The checked contracts of
`len`, `index`, `append`, and `concat` survive aliases, instantiation,
higher-order passage and return, aggregates, recursive and indirect calls,
imports, and serialized TypeEnv checker metadata. Calls decide those
transported contracts without inspecting the callee body.

#### Mechanism

1. `CollectionConstraint` records each checked builtin's fixed operand/result
   relation. `len`, `index`, `append`, and `concat` all carry the result
   equations their ordinary rules impose. These are checked operation
   contracts, not user-authored §5.9 dtype bounds.
2. `Scheme::constraints` transports those relations with the same quantified
   variables as the function type. `Env::instantiate_scheme` performs the one
   renaming and installs a fresh inference-local contract instance with an
   opaque identity and lexical owner. Variable-bearing and fully monomorphic
   relations use this same ledger. All relation variables must occur in the
   callable type; there is no connected hidden-intermediate graph.
3. Aliasing, returning, aggregating, or passing a function value leaves its
   relation transportable. Generalization moves only the exact child-scope
   instances owned by that value onto its scheme; recursive siblings that share
   a level retain distinct identities. Applying a value captures the exact
   instances minted while inferring that callee. If a consumed operand remains
   generic, the declaration boundary reports the missing collection contract.
   Thus transport does not become body-inferred wrapper publication. Projecting
   or returning a result that no longer contains the function-bearing subvalue
   removes its detached transport instance.
4. After a clean recursive-function component is first inferred, generalized,
   and published as one batch, the checker may run at most one further batch
   sweep per recursive member. The function-plan's actual recursive subset is
   authoritative; eager or mixed value cycles do not enter this replay. Each
   sweep resolves in-group references from the preceding complete batch,
   reuses ordinary `infer_top_level`, scheme instantiation, application
   consumption, uniform-recursion validation, and generalization, then
   publishes every member together. A component whose initial complete schemes
   contain no checked relation skips replay: under [04-INF-9], closure cannot
   create a relation without a checked seed. It stops when a batch adds no
   relation or when the first new diagnostic appears. For N members, at most
   N-1 propagation edges are possible, so sweep N must be the stable
   confirmation; growth on that final sweep rejects with one internal checker
   diagnostic instead of publishing an unverified incomplete closure. The
   prior complete scheme is a compiler-owned expected type and does not make an
   unsigned member authored; a member that already owns a `defsig` retains its
   ordinary signature checks. There is no separate AST contract flow walk,
   body-derived contract synthesis, alpha-variable rewrite, or second contract
   representation.
5. Direct syntactic calls keep `app_post.rs`'s operation-specific rules and
   diagnostics. The scheme copy is discarded for that call. Before an indirect
   call unifies its arguments, its exact consumed tensor-`concat` instance
   receives that application's axis value, literal element shapes, or
   binding-carried list length. After call unification, one discharge returns
   the operation rule's exact result rather than the callable type's possibly
   wider wildcard result. Direct and indirect routes use the same
   `tensor_concat_result_type` decision, so exact concat-axis sums and
   out-of-bounds axes do not disappear when the function value was aliased,
   returned, passed, aggregated, imported, or restored from TypeEnv. Nested
   function-valued parameters and results remain transport rather than being
   consumed by the outer call. Explicit failed-call cleanup prevents axis or
   extent evidence from leaking into later calls or independently specialized
   aliases.
6. Serialized TypeEnv checker reuse and published package identities both
   include the relation. TypeEnv format 4 follows the relation-bearing format
   3 and additionally retains callable provenance for contextual named
   gradient selectors in a deterministic JSON-safe root/module structure; its
   exact provenance digest is bound into the checked-library proof so cache
   reconstruction rejects reordered formals, changed lexical origins, or other
   provenance drift. It is the source-free checker-reuse path exercised here.
   CHB format 5 follows #2071's format 4, and Reef schema format 3 follows
   schema format 2; those two
   surfaces protect package publication and identity, not compiler reuse from
   CHB or Reef schema. All predecessors are rejected rather than decoded as
   unconstrained. Canonical package relations share the function type's
   alpha-renamed variables, use exact canonical Deep rendering, are strictly
   ordered and unique, and reject hidden variables. CHB validation,
   encode/decode, and public Reef-schema deserialization all enforce that
   ledger invariant rather than treating parse-equivalent relation strings or
   duplicate rows as distinct package identities.

This mechanism is the collection constructor/result-relation specialization of
the general operation-admission design already recorded in §C3.1 and
[04-INF-9]. It adds no collection-only generalization exception and does not
alter PP7, PP9, runtime-extent guards, or unrelated deferred operation rules.

#### Required acceptance

The acceptance set is:

```text
cargo nextest run -p chelis-types --test issue_1654_generic_collection_constraints --no-fail-fast
cargo nextest run -p chelis-cli --test issue_1654_generic_collection_cli --no-fail-fast
```

```text
cargo check -p chelis-reef --tests
cargo nextest run -p chelis-reef --lib --no-fail-fast
cargo nextest run -p chelis-reef --test scheme_restriction_schema --no-fail-fast
```

Every case must reach both checker APIs. CLI cases must assert the verdict,
error kind, and score, not merely a nonzero command exit.
The negatives cover declaration rejection plus invalid direct, alias,
higher-order, aggregate, recursive, imported, and serialized TypeEnv-reuse
uses. Positive controls cover explicit `List`/`Dict` contracts, both `concat`
relations, and transport of already-checked values. #1506 remains a regression
lock; #1537's PP9 shared-driver/pass-set work is separate and is not implied by
paired checker-ingress coverage here. #1639's alias diagnostic-identity work
is also separate.

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
- **With [#730] again, for PP9**: the pass-set sweep RELOCATES five checker
  rejections that describe what a backend can lower rather than what the language
  admits - trivially non-terminating defs, symbolic conv2d tensor metadata, the
  `mean` / `layer_norm` axis-concreteness rules, and conv2d's literal-operand
  demand, which [05-RWIN-1] forbids as a checker rejection shape. [#730] receives
  them; PP9 does not design their diagnostics. Until it does, those five program
  classes are accepted by every checker entry, which is a deliberate loosening and
  is stated in the change that makes it. The two conv2d relocations move together:
  the metadata-concreteness gate returns early ahead of the literal-operand check,
  so relocating it alone would make the literal demand fire on strictly more
  programs than it does today.
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
| PP5 (partial) | [#668]; unification is the one rank authority for elementwise tensor operands, at both ingresses, and the identity-rank side channel that ran beside it on `check_ir_program` only is deleted (D8 PR A). The claim covers the operations the PP5 oracle spells, not an enumerated registry; see PP5. The tensor-DAG C emitter aborts on a positive-rank operand disagreement, and under [#1484] so does the host-value emitter for its six binary elementwise builtins. The residue is routed: [#597] (the runtime lanes execute `expand` as the unit-extent broadcast the language assigns, with §2.4.1's guard, which S2b landed), [#1512] (reductions over a genuinely unresolved operand), and [#1506] (comparison scalar rewrite, PR B); the row stays partial until the acceptance list in PP5 holds |
| [#1247] residue | integer nominal arguments are kind-checked and concrete dimensions constrain every checker/test/compiler lane; [#1258] round trips the same representation |
| [#1125] nominal-rank ingress residual | ordinary `.dp` ingress, `surf`, and `validate --deep` reject `d-rank` in nominal argument slots while preserving legal dimension arguments and tensor rank spreads; the broader reader-audit/lint issue remains open |
| PP6 | [#1486] (a hole is never quantified and no reference observes it before the body; an authored binder is rigid), [#1487] (lambda bodies and applied values are eager references), and [#1485] (every reference-graph component is inferred as one group; the three spellings reject as `CycleDetected` identically at both ingresses) are delivered with the shared-graph, schedule, paired-ingress, and CLI oracles. [#1854] is the explicit-binder follow-up: Surf and Deep declarations carry complete binder lists and undeclared type/dimension/rank variables reject |
| PP7 | [#1125]'s carrier axis: the seven probed divergences receive the same verdict from `check_ir_program` and `check_typed_program`, and one shared total accessor plus the lint make a carrier a reader cannot decode a diagnostic rather than an absent subtree. PP9 closes the separately owned pass-set axis; the unswept guarded-arm inventory remains outside PP7's claim |
| [#1134] forward-reference parity | both checker ingresses reject eager forward values, accept backward values from value initializers and function bodies where allowed, accept declaration-local explicitly typed external inputs, retain sequential local scope, and reject bare self-reference and every [04-INF-7] eager value cycle identically; the schedule's order invariants are asserted directly |
| PP8 | [#874]'s class statement, restated as coverage rather than tag-keying, and [#887]'s Tier 1 residue. Seven named programs over `vmap`'s axis, `pat-ctor`/`pat-record` heads, and `grad`'s operand are rejected instead of scoring 1.0, and `kv`'s unreadable key reports its own form instead of an `internal:` stamp violation naming a different node; the selector-read seam makes a silently-defaulted slot unspellable, and `infer_expr` reaches it from either Deep carrier. The `Selector` role is enumerated and all eight of its slots are claimed; the other roles are spot-checked only, and converting them into a claim needs an enumerator this item does not deliver (decision row 18) |
| PP9 | [#1537]'s pass-set axis: the four checker entries apply one ordered semantic protocol, every pass in it carries a spec sentence or has moved to the stage whose capability it describes, and 23 logical rows receive identical ordered diagnostics from all four across every admitted carrier. Each entry uses declaration-local external-input prebinding under [04-INF-4] plus defsig-less function-header prebinding under [04-INF-2]/[04-INF-3]; no universal claim is made outside the corpus |
| [#1339] top-level initialization frontier | an eager value whose acyclic closure reaches a later non-function value rejects as `UnboundVariable` under [04-INF-8]; cycles retain [04-INF-7]'s `CycleDetected`, while backward and independent controls preserve source-ordered manifest output; `issue_1339_top_level_initialization` covers both checker ingresses plus `check`, `prove`, `eval`, and C `build` as the authoritative oracle |

## Decisions and remaining questions

| # | question | decided in | recorded where |
|---|---|---|---|
| 1 | `handle-effect`'s checked signature details | DECIDED 2026-07-17 (revised same day, explicit over implicit: this code is agent-written, so there is no ergonomic case for contextual binding). Phase 1 checks FORM, [#735] authors meaning. Seed = an EXPLICITLY i64-suffixed signed integer literal (`42i64` or `-1i64`, spec/02 §P5/§P10a); [05-RNG-1] governs its signed seed bits; an unsuffixed literal is a type error whose diagnostic names the requirement and the suffix spelling; non-literal seed expressions are rejected, diagnostic citing §P5's shipped constraint and [#735]. Device = a string literal; the checker validates literal-ness only, never the device-name vocabulary (target knowledge, [#735]'s territory). No spec/02 §P10 change needed - the width is visible in the source itself. Existing `with seed(n)` fixtures/examples migrate to the suffixed form in P1's change set (Public-Surface Change Rule) | §C1.5 + spec/04 effect section |
| 2 | typecheck-cache deserialization as a witness mint (accepted, or cache entries re-validated?) | Phase 2 | §C3 note + the cache module doc |
| 3 | whether printers/desugar also migrate to `DeepTag` (nice-to-have; they are not chokepoints) | DECIDED 2026-07-23: deferred; REVERSED 2026-07-24 by the decode-once rework directive - printers, desugar, and every other producer/consumer migrated; no string-keyed tag idiom survives outside the parse/serialize boundary | this doc |
| 4 | score semantics for `UnknownForm`/`MalformedForm` | DECIDED 2026-07-17: severity parity with `TypeMismatch` (the existing 0.5-class precedent), no new weight class. The invariant that matters - any pushed error forces score < 1.0 - is locked by §C4.4's corpus independently of the weights, so calibration can move later without touching it | scoring code + this doc |
| 5 | how a lambda-bound or function-valued parameter's type binds, and which checks re-run once it is bound | DECIDED 2026-08-04: shape-constrained lambdas with an unknown outer parameter constructor are monomorphic bind-on-first-use within their enclosing declaration, replay the ordinary semantic rule, and reject unresolved at that declaration's own boundary; a result annotation or later top-level caller does not bind them; symbolic declared tensors remain polymorphic and rigid dimensions remain distinct absent a real equality constraint | [04-INF-1] + PP1 |
| 6 | whether two closures may each consume one underlying value through two user-visible names (`y = x`, one capture per name), or capture forwards through the alias chain generally | DECIDED 2026-08-21: preserved and made normative. A capture consumes the binding it names; distinct user-visible bindings of one value are distinct for capture; only a destructured component (or an alias of one) forwards to its carrier. Nautilus `lu_solve` and coral depend on the spelling; the reviewer guidance on [#1209] was to specify the choice explicitly and keep any tightening separate | [04-LIN-2] + PP3 |
| 7 | whether one linked module's uniquely matching terminal name or one batched test file's declaration can confer unimported scope on another file | DECIDED 2026-08-31: no. Value lookup is exact-only after reef rewriting; batch entries are independently module-rewritten before combination; terminal matching is diagnostic-only | spec/02 P2 + [04-FIT-2] + PP4 |
| 8 | whether an unannotated nominal parameter is a type, a dimension, or contextually reinterpreted per application | DECIDED 2026-08-31: one checker-owned header kind is fixed before body resolution. Dimension-only evidence selects `Dimension`; mixed use rejects; unused defaults to `Type`; transitive nominal uses propagate by least fixed point | [04-ADT-3]/[04-ADT-4] + [#1247] residue |
| 9 | whether a top-level eager value may refer to a later value, and whether the two checker ingresses may differ | DECIDED 2026-09-01: no. Both ingresses reject a later eager value as unbound; serialized body metadata cannot create scope. Scope is read from declaration position, never from a binding timeline the inference schedule advances, and the schedule infers an eager value before any function that legally reads it, using only reference edges over the hoist order so a program without such a read keeps its previous grouped order. Only an explicitly typed self-reference receives a declaration-local external-input type; bare self-reference remains an eager cycle. Function inference groups remain separately governed by [04-INF-2]/[04-INF-3] | [04-INF-4] + [#1134] residue |
| 10 | whether a wildcard slot in a signature is a polymorphic binder, and what a reference sees before the declaration's body is inferred | DECIDED 2026-09-03: a hole, never quantified; every reference is typed at the body-determined signature wherever it sits, so readers of a hole-signature function are scheduled after its body in every region. A shared monomorphic hole was rejected because it makes a partial header monomorphic in its own dimension binders | [04-INF-5] + PP6 |
| 11 | whether a body may narrow an authored type binder | DECIDED 2026-09-03: no; the user confirmed the rigid rule and accepted the ten-site stdlib migration (`cast(lit, p)` plus bundle regeneration) that it costs. Explicit binders are rigid in the body, as dimension parameters already are under §4.4; the scheme is the declared signature. Ten stdlib declarations that narrow a bounded binder with an unsuffixed literal migrate to `cast(literal, p)`. The 2026-09-17 [#1854] decision removed implicit binders rather than weakening rigidity | [04-INF-6] + PP6 |
| 12 | which references inside a top-level value's initializer are eager for cycle detection | DECIDED 2026-09-03: all of them, lambda bodies included, transitively through every referenced top-level declaration, with an applied value required like a read one. The argument-position refinement was rejected as unsound for stored and returned closures; the over-rejection is accepted and is already the detector's treatment of a bare function reference | [04-INF-7] + PP6 |
| 13 | whether one checker entry may check a program the other does not, when the difference is which admitted carrier represents it | DECIDED 2026-09-03: no. Verdict is independent of entry and of carrier. Normalizing at the one non-normalizing entry is rejected as the mechanism because three of the seven probed divergences live outside every checker entry (`prune.rs` before the check, `tier_b_lower.rs` and `count_invariant_opaque_deep` after it); the reader, not the entry, is the unit that must be total. A carrier a reader cannot decode is diagnosed, never observed as empty | [04-TOT-5] + PP7 |
| 14 | whether the seven comparison identities admit a scalar beside a tensor ([#1506]) | DECIDED 2026-09-03: no; the checker rejects and the spec wins. `[05-OP-36]` ("Mixed surfaces ... are type errors"), `spec/05` §1.2, and `spec/04` §4.2-§4.3 already decide it, and `add`/`max_elem` reject the same pair today, the `spec/05` §2.1 prose above `[05-OP-40]` naming their scalar form "the rank-zero instance of the tensor rule, not scalar/tensor broadcasting". The rewrite at `infer/app.rs:457-506` goes; the three accepting `issue5_cmp_broadcast_both_forms` rows and `coral_comparison_ops_broadcast_tensor_scalar` become negative controls; scalar-scalar and same-shape tensor-tensor forms stay; the diagnostic names `[05-OP-36]` and the explicit `expand(to_tensor([c]), axis, shape(x, axis))` spelling. ALTERNATIVE, not taken and requiring the user's explicit choice: amend `[05-OP-36]` to admit one active-numeric scalar beside one tensor of the same dtype, comparing every element against the scalar and returning `tensor[D, bool]`. Its cost: it contradicts §1.2's "hard rule" and §4.2's rationale; it must explain why comparison broadcasts when `add`, `max_elem`, and `[05-OP-17..19]` do not, or extend them too; it leaves a rank-0 tensor beside a tensor undecided; the evaluator already implements it, so the runtime cost is nil; and Coral depends on the accepting behaviour today (`coral_prerequisites.rs:315`), so the rejection has a downstream migration cost that the alternative avoids. Sequencing under the decision taken: the explicit spelling executes as an insertion on every lane until [#597] closes (Slice B2a item b2.5 for C, Slice B2h for eval), so PR B of PP5 D8 lands after both or states the gap | PP5 D4/D8 + `[05-OP-36]` |
| 15 | whether PP5's identity-rank validator stays as a second rank model beside unification | DECIDED 2026-09-03 by `spec/04` §4.7.2 and the D1-D5 evidence: it goes. **What the user is asked to confirm, in one sentence:** for `s = stride(x, 2i64); e = expand(x, 0i32, 2i64); add(s, e)` on `x: tensor[n, f32]`, the checker accepts the program with `e` at `tensor[2, f32]`, the only shape the language assigns to `expand` (row 16, [#1532]; the `f5ec5ca63` checker already stamps it), and PP5 stops rejecting it; the loud outcome for `n != 1` is the `Domain` trap `spec/05` §2.4.1 states ([#1523], merged; narrowed by [#1532]) once Slice B2a item b2.5 carries its guard, a literal operand extent other than 1 being a check-time type error instead. Today, for this program, eval rejects with `[3] vs [2, 6]` and the C insertion path trips R3's rank guard, because the runtime still inserts ([#597]) and the consumer `add` exposes the inserted axis; that loudness is the consumer's, not the language's: the bare spelling `sig f: tensor[6, f32] -> tensor[2, f32]; def f(x) = expand(x, 0, 2i64)` with no consumer is silent on both lanes today (`chelis check` score 1 at `-> tensor[2, f32]`, `chelis eval` and the compiled C both print `shape=[2, 6]` with no diagnostic; measured by the [#1277] owner and re-measured here on the f5ec5ca63 build and the 09-01 binary), which is [#597] as filed, and under the single-meaning rule plainly so. Unification is the one rank authority; the validator's `expand` derivation, input rank plus one, is a rank the language never assigns to `expand`, and it rejects programs whose stamped types agree (measured, D2 third row). The checker half of [#668] resolves into [#597] (the runtime must execute `expand` as the unit-extent broadcast, with §2.4.1's guard), [#1512] (reductions over a genuinely unresolved operand), and [#1506]. Because this was expected to flip ten whole [#1463] rows, the control half of an eleventh, and the corpus row in `issue_731_fitness_honesty_corpus.rs` from rejection to acceptance, the implementation PR (D8 PR A) states the flip in its body, lands after S2a (the [#1277] stream's `insert` builtin plus mechanical rename, branch `agent/1277-s2a-insert-rename`; [#1532] itself is merged as `b8e08e5b9`), and lands only after the user confirms the sentence above. The single-meaning decision (row 16) removed the earlier alternative reading of positional `expand` as insertion-only; that meaning is now `insert`'s. MEASURED OUTCOME, recorded when PR A landed: the flip shipped in [#1277] S2b, not in PR A. S2b gave the validator rank delta 0 for `expand`, which stopped it rejecting those programs, and re-vehicled every `issue_668_elementwise_rank_honesty` row onto `insert`, whose rank 2 is genuine. PR A therefore changed no verdict for the elementwise family; what it removed there was a duplicate diagnostic on one ingress. Its first pushed head DID change verdicts at the environment's other consumer, the `conv2d` validator, by turning the three deleted operations into permanent failed derivations; red-team round 1 found it, and re-keying the failed-derivation marker restores byte-identical verdicts against the base on every probe. See the measured-outcome paragraph in D8 | PP5 D6 (d) |
| 16 | whether positional `expand` is one operation or two | DECIDED 2026-09-03 by the user: "I don't want ambiguity that is resolved at runtime. Make expand canonically only insert or increase the number of dimension (whichever is more canonical). Use something else (e.g. insert) for alternative. I don't want overloaded uses like this." Confirmed as: `expand` keeps only the same-rank singleton broadcast (rank unchanged; the operand's extent at `axis` must be 1; a literal violation is a check-time type error; a symbolic extent is a §4.7 runtime `Domain` guard, per [#1523]); a new primitive `insert` adds an axis of extent `size` at `axis`; `spec/04` §4.7.2's two-candidate deferred model is deleted. Recorded in PR [#1532], merged as `b8e08e5b9` (`spec/02`, `spec/03`, `spec/04`, `spec/05`, `spec/06`, `spec/08`, `runtime_extents.md`; `spec/05` §2.4 rows and adjoints for both primitives under `[05-AXIS-1]`/`[05-MOV-1]`, §4.7.2 without the two-candidate model, §4.5.3's named forms as `insert`; no new `[05-OP-N]`). PP5 consumes the decision and adds nothing to it | [#1532] + PP5 D2 |
| 17 | whether the source-coverage obligation is a new atom or a tightening of an existing one, and whether [#887] closes | DECIDED 2026-09-03: a new atom that EXTENDS [04-TOT-3] rather than replacing it. [04-TOT-3] already governs the live instances and the shipped `access`/`record` rejections cite it, so R1 through R3 are unimplemented [04-TOT-3] cases and an implementer fixing them cites [04-TOT-3]. [04-TOT-4] carries that obligation from the form to each of the form's slots and adds the two sentences no earlier atom states: an omitted optional child and a present unreadable one are distinct inputs with only the omission permitted to default, and coverage quantifies over the submitted program rather than the checked result. The second is the one [04-TOT-2] structurally cannot express, and R1 proves it by satisfying [04-TOT-2] completely while being wrong. [#887] is RE-SCOPED, not closed: its Tier 2 shipped via [#998]/[#1019]/[#1041], and its Tier 1 consumption-boundary residue is this item's Slice 2 | [04-TOT-4] + PP8 |
| 18 | whether a parsed-vs-checked coverage census belongs at `finalize_checked_program` | OPEN, recorded 2026-09-03, no deliverable attached. It closes none of PP8's five named instances, which Slices 1 and 2 close between them, and its three candidate justifications do not survive a necessity trace: the roles it would guard have no demonstrated defect, `child_stamp_role` already makes an unclassified tag a compile error, and the cancellation route it would subsume is closed at the surface [#874] named. It is also the only proposal here touching the public fitness surface. Revisit if a coverage-keyed instance appears that the selector-read seam does not reach | PP8 + [#874] |
| 19 | whether an eager value may initialize through a function or nested lambda that reaches a later non-function value | DECIDED 2026-09-04: no. Every non-function value in the initiating value's [04-INF-7] eager-reference set is compared with the initiating value's source position; an acyclic later member is `UnboundVariable`, while a return to the origin is `CycleDetected`. Dependency-ordering whole initializers was rejected because top-level effects and traps make it observably different from eval's demand forcing; the implementation follows PP6 B/C and reuses their single graph | [04-INF-8] + [#1339] frontier section |
| 20 | which of the two checker pass sets is correct, and whether one shared driver replaces the two inference functions | DECIDED 2026-09-08 and IMPLEMENTED 2026-09-16: union plus dispositions. All four entries run the spec-required surviving checks through `validate_semantic_program`; `report_initialization_errors` lives there, `chelis_deep::validate` is deleted as a duplicate, and stamped input is preserved rather than normalized. [05-OP-51] owns convolution's static-versus-runtime domain split. Five backend-capability restrictions (termination, symbolic conv metadata, `mean` and `layer_norm` axis concreteness, and conv's literal-operand demand) relocate to [#730]. Both drivers use declaration-local external-input prebinding under [04-INF-4] and defsig-less function-header prebinding under [04-INF-2]/[04-INF-3]; body metadata never publishes an eager value. Union alone and union plus normalization remain rejected by the measured contract and carrier failures | [04-TOT-5] + PP9 |
| 21 | whether a borrow's target type is decided at the borrow arm or after def-level resolution, and whether the #256 deferred classification survives [04-INF-6] ([#1589]) | DECIDED by `spec/04` §8.2, which already states it: the inner "must be — or must ultimately resolve to — a tensor or a tensor-carrying value", and classification is deferred when it is not yet known. No language decision is open. The reading that a borrow is decided where it is written is REFUTED by execution: disabling `validate_deferred_borrow_vars` makes `def use_it[a](seed: a) -> bool = { v = seed  consume_any(&v) }` score 1.00 with no errors, reopening the #256 round-2 unsoundness, and turns all three of the suite's deferred-path tests red, so the validator is live code and its two acceptance tests were merely relabelled by [#1542]. The issue's original premise that an inferred parameter "rejects at the borrow arm" is also wrong: measured, the borrow arm defers, the validator resolves it `sound=true`, and the 0.80 `InvalidBorrow` comes from linearity's `check_borrow_arg`, which failed closed because `expr_type` returns `None` for a `(var ..)` node whose parameter annotation is a synthesized hole. The repair reads the resolved `&T` the annotate pass already stamps on the `borrow` node. Rows C/E/F/G of the [#1589] header matrix become accepted regression rows; rows I/J/K stay rejected as locks on the validator's reject branch; §8.2's `relu` example is corrected, because unresolved dimension variables never reach the deferral | `spec/04` §8.2 + PP6 residue |
| 22 | whether a `defsig` occurrence may introduce a type, dimension, or rank binder | DECIDED 2026-09-17 for [#1854]: no. Surf's `[..]` list and Deep's structural `defsig` binder-list child are the only authored declaration-binder sources. Unlisted variable nodes reject; unknown scalar/precision names remain primitive requests and receive an unknown-dtype diagnostic with a nearest active spelling. The regex/alias heuristic alternative is rejected because it leaves an open typo class | spec/02 P4b + spec/03 §2.2/§2.5.1 + spec/04 §3.1.3/§5.8.1 + PP6 |

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
demonstrated and which are not. Its completion design records that the rank
the validator derives for `expand`, the input rank plus one, is a rank the
language never assigns to `expand` under the single-meaning rule merged in
[#1532], so the side channel is retired in favour of unification once
implemented; until then the shipped behaviour is as stated.
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
PP7 extends the same honesty rule from the entry to the reader. [04-TOT-5] makes
a program's verdict independent of which checker entry receives it and of which
admitted representation carries it, so a check one representation receives is a
check every representation receives, and a carrier a reader cannot decode is a
silent exemption to be diagnosed rather than an empty subtree to be skipped.
PP9 applies that rule to the pass set: every public entry runs one ordered
semantic protocol, while backend capability restrictions stay outside the
language checker and remain loud-unsupported work under [#730].

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
[#1126]: https://github.com/Chelis-Lang/chelis/pull/1126
[#1107]: https://github.com/Chelis-Lang/chelis/issues/1107
[#1320]: https://github.com/Chelis-Lang/chelis/issues/1320
[#1362]: https://github.com/Chelis-Lang/chelis/issues/1362
[#1537]: https://github.com/Chelis-Lang/chelis/issues/1537
[#1543]: https://github.com/Chelis-Lang/chelis/pull/1543
[#1546]: https://github.com/Chelis-Lang/chelis/pull/1546
[#1602]: https://github.com/Chelis-Lang/chelis/pull/1602
[#1134]: https://github.com/Chelis-Lang/chelis/issues/1134
[#887]: https://github.com/Chelis-Lang/chelis/issues/887
[#930]: https://github.com/Chelis-Lang/chelis/issues/930
[#1085]: https://github.com/Chelis-Lang/chelis/issues/1085
[#998]: https://github.com/Chelis-Lang/chelis/issues/998
[#1019]: https://github.com/Chelis-Lang/chelis/issues/1019
[#1041]: https://github.com/Chelis-Lang/chelis/issues/1041
[#1485]: https://github.com/Chelis-Lang/chelis/issues/1485
[#1486]: https://github.com/Chelis-Lang/chelis/issues/1486
[#1487]: https://github.com/Chelis-Lang/chelis/issues/1487
[#1355]: https://github.com/Chelis-Lang/chelis/issues/1355
[#219]: https://github.com/Chelis-Lang/chelis/issues/219
[#1484]: https://github.com/Chelis-Lang/chelis/issues/1484
[#1494]: https://github.com/Chelis-Lang/chelis/issues/1494
[#1525]: https://github.com/Chelis-Lang/chelis/issues/1525
[#1457]: https://github.com/Chelis-Lang/chelis/pull/1457
[#1542]: https://github.com/Chelis-Lang/chelis/pull/1542
[#1589]: https://github.com/Chelis-Lang/chelis/issues/1589
[#1551]: https://github.com/Chelis-Lang/chelis/pull/1551
[#1512]: https://github.com/Chelis-Lang/chelis/issues/1512
[#1339]: https://github.com/Chelis-Lang/chelis/issues/1339
[#1277]: https://github.com/Chelis-Lang/chelis/issues/1277
[#597]: https://github.com/Chelis-Lang/chelis/issues/597
[#664]: https://github.com/Chelis-Lang/chelis/issues/664
[#1380]: https://github.com/Chelis-Lang/chelis/issues/1380
[#1463]: https://github.com/Chelis-Lang/chelis/pull/1463
[#1506]: https://github.com/Chelis-Lang/chelis/issues/1506
[#1508]: https://github.com/Chelis-Lang/chelis/pull/1508
[#1511]: https://github.com/Chelis-Lang/chelis/pull/1511
[#1513]: https://github.com/Chelis-Lang/chelis/pull/1513
[#1519]: https://github.com/Chelis-Lang/chelis/pull/1519
[#1523]: https://github.com/Chelis-Lang/chelis/pull/1523
[#1532]: https://github.com/Chelis-Lang/chelis/pull/1532
[#1612]: https://github.com/Chelis-Lang/chelis/issues/1612
[#1619]: https://github.com/Chelis-Lang/chelis/issues/1619
[#1621]: https://github.com/Chelis-Lang/chelis/issues/1621
[#1124]: https://github.com/Chelis-Lang/chelis/issues/1124
[#1552]: https://github.com/Chelis-Lang/chelis/pull/1552
[#1590]: https://github.com/Chelis-Lang/chelis/pull/1590
