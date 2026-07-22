# Checker Totality: every construct is checked or loudly rejected

**Status:** Phase 2 implemented; Phase 3 (`DeepTag`) remains open. Tracking
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

**Scope verdict** (the question this document answers first): the immediate
fix is genuinely SMALL (one checker case + one loud wildcard + the [#710]
guard pushes - Phase 1 is days, not weeks). What merits design is the
ratchet: (a) a **type-state change** that makes a silent `Type::Error`
*unconstructible* - the variant carries a witness token obtainable only by
pushing a diagnostic or propagating an existing error - and (b) the
**`DeepTag` enum** at the dispatch chokepoints, so that the day tag 63 is
added, every consumer that has not decided what to do with it stops
compiling. No rewrite; the Deep AST keeps its shape; only dispatch entry
points and the `Type::Error` variant change.

## Shipped Phase 2 architecture

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
   the handled body remains a type-owned runtime child.
6. Annotation consumes the completed epoch's canonical owner stamps. It does
   not semantically re-infer expressions with a fresh `VarGen`/`Subst`; a
   missing or conflicting owner write is a diagnostic, never a default type.
7. `finalize_checked_program` checks fresh inference results: the annotated
   runtime tree and pattern/function stamps, with input and output signature
   metadata as structural backstops. There is no public raw `from_parts` or
   `try_from_parts` API. The effects pass can call only
   `CheckedProgram::try_with_effect_annotations`: it proves every root, span,
   atom, child, and metadata entry is identical to the checked input except
   the effects-owned `effects` entry, preserves the original type environment,
   signature inference, and linearity, and reruns totality validation. A
   violation maps to `EffectErrorKind::TypeTotality` exactly once.
   `CheckedProgram::compose` is the distinct trusted operation that combines
   two already-successful checked halves.
8. Recursive callable availability is planned by the canonical function SCC
   schedule. Only a genuinely recursive SCC receives provisional monomorphic
   bindings; its members infer, unify, remove the provisional entries, and
   generalize as a unit. Acyclic generic helpers stay polymorphic, bare
   acyclic forward calls retain textual semantics, and no diagnostic is
   erased after it has been reported.

PR #802's checker snapshot is advisory evidence only. The active specs, code,
and executable tests above are the authority; this document does not derive or
invent atom identifiers or content hashes from that snapshot.

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
  and it pushes the diagnostic in the same expression - the two can no
  longer be separated by a refactor, a review miss, or a new contributor.
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
type-context, and compiler-api cache module docs (open question 2).

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
2. **`DeepTag` enum at the chokepoints** (Phase 3): the parser already
   validates strings against the closed vocabulary; it starts producing
   `DeepTag` (string kept alongside for spans/printing). `infer_expr`,
   `lower_expr`, and the `.dp` structural validators match the enum
   **exhaustively - no `_` arm** ([#730]'s `rust-no-wildcard-dispatch`
   lint adds `DeepTag` to its enum list, §I1). Tag 63 then stops the build
   at every consumer that has not chosen a disposition. Raw-string entry
   points (anything that never went through the parser) keep the §C1.2
   loud arm.
3. **The canary stays forever**: the wrapper battery is cheap, runs in the
   default suite, and is the behavioral proof the structural claims cash
   out. New wrapper constructs added to the language get a row in it as
   part of landing (a one-line reviewer checklist item, recorded here).
   Known fixture caveat (PR #757): six of the eleven wrapper strings are
   parse-rejected Surf as written, so those rows score below 1 via the
   parser, not the checker; re-probed with corrected syntax, the
   ill-typed variants ARE checker-caught, so the coverage claim
   survives. Fixtures corrected at Phase 1 (B2.1: rows may be fixed to
   test what they claim, never removed or weakened).
4. **The fitness-honesty corpus** (2026-07 review integration): a CI
   suite of known-ill-typed programs - the wrapper battery plus every
   census-verified silent-hole repro ([#709]/[#710]/[#755]/[#756] and
   future finds) - asserting every member scores strictly below 1.0;
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

## Phase 3 - `DeepTag` at the chokepoints

**You inherit:** a checker that cannot silently exempt (Phase 2) - this
phase is about the NEXT tag, not the current ones.

**You deliver:**

1. `enum DeepTag` (62 variants, `parse`/`as_str`, produced by the Deep
   parser alongside the validated string).
2. `infer_expr`, `lower_expr`, and the `.dp` structural validators
   dispatch on it **exhaustively**; the string-keyed `tag(list) ==
   Some("...")` chains at those three chokepoints retire. (Printers and
   producers may migrate opportunistically; they are not chokepoints -
   they cannot exempt or substitute.)
3. `DeepTag` added to [#730]'s lint enum list (§I1); the §C1.2 loud arm
   retained only at raw-string entry boundaries.

**Frozen at your exit:** the variant set = the vocabulary, changing only
per B1's one-change-set rule.

**Explicitly not yours:** adding tag 63 or any vocabulary change.

**Oracle:** the build itself - the mutation test: adding a scratch
variant to `DeepTag` must produce compile errors in `infer.rs` AND
`lower.rs` AND the validators (verified once in the PR, recorded, then
the scratch variant deleted); the canary and full matrix stay green.

---

# Part IV - bookkeeping

## I1. Interlocks

- **With [#730] (`loud_unsupported.md`)**: `EffectKind` is delivered there
  (Phase 2) and consumed here (§C1.5) - if this plan's Phase 1 lands
  first, it matches the two kind strings with a loud else and migrates to
  the enum when available; the bogus-effect `.dp` repro must be rejected
  by whichever side lands first, and BOTH once both land (checker: unknown
  kind is `MalformedForm`; lowering: its catch-all raises). `DeepTag`
  joins the `rust-no-wildcard-dispatch` enum list on landing (their B1
  freeze anticipates the addition).
- **With [#729] (`dtype_semantics.md`)**: none structural. Phase 4's
  capability table derives op x dtype acceptance; this plan governs
  CONSTRUCT-level totality. The two meet only in that both make `chelis
  check`'s score-1 claim honest.
- **With [#721]**: none (eval ingestion, no checker code); listed so nobody
  searches for it here.

## Issue map

| phase | goes green / becomes unwritable |
|---|---|
| 0 | detection; the invariant exists |
| 1 | [#709] (all three escalations), [#710]'s silent half |
| 2 | the future supply of silent exemptions (type-state) |
| 3 | the future supply of undecided TAGS (compile-time totality) |

## Open questions and where they get decided

| # | question | decided in | recorded where |
|---|---|---|---|
| 1 | `handle-effect`'s checked signature details | DECIDED 2026-07-17 (revised same day, explicit over implicit: this code is agent-written, so there is no ergonomic case for contextual binding). Phase 1 checks FORM, [#735] authors meaning. Seed = an EXPLICITLY int64-suffixed integer literal (`42i64`, spec/02 §P10a); an unsuffixed literal is a type error whose diagnostic names the requirement and the suffix spelling; non-literal seed expressions are rejected, diagnostic citing §P5's shipped constraint and [#735]. Device = a string literal; the checker validates literal-ness only, never the device-name vocabulary (target knowledge, [#735]'s territory). No spec/02 §P10 change needed - the width is visible in the source itself. Existing `with seed(n)` fixtures/examples migrate to the suffixed form in P1's change set (Public-Surface Change Rule) | §C1.5 + spec/04 effect section |
| 2 | typecheck-cache deserialization as a witness mint (accepted, or cache entries re-validated?) | Phase 2 | §C3 note + the cache module doc |
| 3 | whether printers/desugar also migrate to `DeepTag` (nice-to-have; they are not chokepoints) | Phase 3, may defer | this doc |
| 4 | score semantics for `UnknownForm`/`MalformedForm` | DECIDED 2026-07-17: severity parity with `TypeMismatch` (the existing 0.5-class precedent), no new weight class. The invariant that matters - any pushed error forces score < 1.0 - is locked by §C4.4's corpus independently of the weights, so calibration can move later without touching it | scoring code + this doc |

## The one-sentence summary for a reviewer

Make the checker's default loud (one wildcard, one new case, the [#710]
guards), then take the pen away twice: `Type::Error` without a pushed
diagnostic becomes unconstructible (witness token), and a Deep tag without
a checker disposition becomes uncompilable (`DeepTag` exhaustive matches) -
with an always-on invariant (empty errors implies no `Type::Error` in the
tree) standing guard over both claims.

[#696]: https://github.com/Chelis-Lang/chelis/pull/696
[#703]: https://github.com/Chelis-Lang/chelis/issues/703
[#709]: https://github.com/Chelis-Lang/chelis/issues/709
[#710]: https://github.com/Chelis-Lang/chelis/issues/710
[#721]: https://github.com/Chelis-Lang/chelis/issues/721
[#735]: https://github.com/Chelis-Lang/chelis/issues/735
[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#731]: https://github.com/Chelis-Lang/chelis/issues/731
[#755]: https://github.com/Chelis-Lang/chelis/issues/755
[#756]: https://github.com/Chelis-Lang/chelis/issues/756
