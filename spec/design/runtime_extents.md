# Runtime Extents: one resolver for non-literal tensor extents

**Status:** PROPOSED. No phase is implemented. Tracking issue: [#1277].
Code evidence in this document was measured on `main` at `647c76a9` unless a
different commit is named; function names are the durable anchors and line
numbers are a convenience of that commit.
**Owning specs:** `spec/04-type-system.md` §4.7 (the runtime-size contract as
amended by PR [#1283], which deleted the Form-1/2/3 taxonomy) and
`spec/05-risc-primitives.md` [05-DIM-1..3] (extent-domain `int64`, axis-domain
`int32`) plus the owning `[05-OP-N]` atoms for `expand`, `reshape`, `shrink`,
`pad`, and `stride`. This plan implements those decisions; it decides no
language semantics of its own. The one open language decision it surfaces
(freeze-point settlement order, §C3.3) must land as a numbered-spec amendment
before the phase that implements it.
**Class fixed:** [#1277] - a tensor extent that is not a literal (a runtime
`shape()` read, a deferred or positional `expand` size, a symbolic dimension)
is resolved by four cooperating mechanisms, and each has gaps the others do
not cover.
**Sibling plans:** [`dtype_semantics.md`](dtype_semantics.md) ([#729]) owns
what extents are *typed as* and delivered the [05-DIM] boundary; its Phase 4
issues [#1296]/[#1298] own runtime axes and reduction windows.
[`hash_order_determinism.md`](hash_order_determinism.md) ([#1341]) owns the
ordered-iteration mechanism this plan's §C3.2 consumes.
[`checker_totality.md`](checker_totality.md) ([#731]) owns the diagnostic
channel discipline the deleted rejections leave behind.
[`chelis_616_runtime_movement_dims.md`](chelis_616_runtime_movement_dims.md)
is the shipped predecessor of §C4 and records why the current backend
machinery has its shape.

## Summary

The controlling spec text is newer than the implementation, and the class
closes by moving the implementation to the spec, not by patching the
implementation's gaps. PR [#1283] (2026-08-24, one day after [#1277] was
filed) amended `spec/04-type-system.md` §4.7 to say, of `expand` sizes:

> A function parameter, top-level or local binding, user-function result,
> cast, or checked integer-arithmetic expression is equally admissible; no
> stage may reject an extent because of its provenance or default it to one.

and, of every runtime extent (§4.7.4):

> Runtime extent expressions are lowered as ordinary typed integer dataflow.
> ... Eval, C, HIP, and Metal execute the same graph and checks. A backend
> may optimize a proven constant but may not require one, infer provenance to
> narrow the language, or substitute a guessed extent.

The words "Form-3" and "sourceless" no longer appear in any numbered spec.
The implementation, at the measured commit, still contains a provenance
classifier whose rejection diagnostic cites "§4.7.2 Form-3" and recommends a
`cast(N, int32)` size spelling that [05-DIM-1] has since made a type error.
Every child of [#1277] is a gap in one of four mechanisms that the amended
contract makes unnecessary, under-specified, or non-conforming. The plan:
delete the provenance classifier rather than extend it (§C2), specify and
totalize the one genuine deferral (§C3), move the backend from name-resolved
symbolic dims toward value edges (§C4), sweep the dtype-boundary residue
(§C1.3), and hold the whole surface with one cross-lane parity-and-stability
oracle (§C5).

## The evidence: four mechanisms, and why instances keep arriving

The four mechanisms, with their measured gaps:

1. **The provenance walk** (`classify_expand_size`,
   `crates/chelis-types/src/infer/app_shape_helpers.rs:243`; `SizeClass`
   at `:183`). Before accepting a runtime `expand` size, the checker walks
   the expression looking for a path back to a tensor: literals and `cast`
   chains, `shape(t, axis)` on a bare or borrowed variable, `let`-bound
   aliases via a lexically scoped provenance map (`Env::size_provenance`),
   and applications of exactly `add|sub|mul|div|mod|neg`
   (`INT_ARITH`, `:334`). Everything else hits a fail-closed
   `_ => SizeClass::Sourceless` arm. A record field access ([#1266]), the
   lambda `chelis fmt` itself emits for `|> cast(int64)` ([#569]), and a
   symbolic-rank body ([#578]) are all just missing arms of that match.
2. **Deferred shape candidates**
   (`crates/chelis-types/src/unify.rs:131,:137`). Positional
   `expand(x, axis, n)` genuinely defers a choice - insert a new axis or
   replace an extent - as constraints keyed by the unresolved result type
   variable. Selection fires only in `bind_tvar` when that variable unifies
   against a non-variable type; the comparison family constructs its `bool`
   result and returns it without unifying anything
   (`infer/app_post.rs:568`), so a comparison can never select ([#1265]).
   The freeze-point fallback (`materialize_deferred_expand_defaults`,
   `unify.rs:996`) iterates a `HashMap`, so coupled defaults settle in hash
   order and the same file is accepted or rejected at random ([#1338]).
3. **Backend name resolution** (`symbolic_occurrences`,
   `crates/chelis-ir/src/dag.rs:1627`). The DAG resolves symbolic dims by
   *name*: every symbol must be declared by a `Load` or by the
   `op_declared_output_axes` table (`:1859`), and an undeclared occurrence
   is an ICE (`:1793`). The table's `Expand` arm covers only the expand's
   own axis in the anon-size + shape-dep form; an op-declared runtime axis
   arriving on the expand's *input* is dropped, which is [#665] and (via
   grad+vmap) [#592].
4. **The dtype boundary.** [05-DIM-1..3] shipped in v0.18.3 (chelis#1130,
   for [#1112]); `shape()` returns `int64`, every extent consumer takes
   `int64`, the C ABI carrier is `int64_t` ([#1149]). The live residue is
   diagnostics and doc comments that still recommend the pre-migration
   `int32` spelling (§C1.3).

Three structural facts explain why fixing instances has not closed the
class:

- **Three walks claim to be mirrors of each other and drift by
  construction.** The checker's `classify_expand_size`, the lowerer's
  static folder and shape-source triad (`fold_static_size`,
  `lower.rs:9939`), and the DAG's `shape_source_for_axis` (`dag.rs:2114`)
  each re-derive "where does this extent come from" over a different
  expression or node vocabulary. Measured drift at `647c76a9`: `div` is in
  the checker's `INT_ARITH` but not in `fold_static_size`'s folded ops; the
  checker classifies `mul(shape(x, 0), 2)` as accepted while the lowerer
  rejects exactly that spelling with a fatal "cannot materialize as an
  extent" error (`lower.rs`, the `raise_fatal_lowering_error` under the
  expand-size dispatch) - the latter in direct conflict with amended
  §4.7.6, which says producers "may emit any well-typed expression and need
  not rewrite it into a privileged syntactic shape".
- **Polarity is inconsistent.** `SizeClass::Unknown` is the *accepting*
  class and `Sourceless` the rejecting one
  (`infer/app_tensor.rs:1009`), while the walk's catch-all arm *rejects*;
  `shape_source_for_axis` returning `None` means "op-declare the axis" at
  one call site (`op_declared_output_axes`) and "ICE" at another
  (`symbolic_occurrences`).
- **The checker-to-backend channel is one `Dim::Name` in annotated type
  metadata.** The classifier's verdict is never serialized. The single arm
  `_ => return subst.apply(result_ty)` in `check_expand_signature`
  (`infer/app_tensor.rs:1022`) is simultaneously the [#597] hole (a
  Static-sized expand result stays a type variable, gets no `type`
  metadata, and eval's `assert_ir_typed` then rejects what `build`
  accepts) and the [#609] hole (the same early return skips validating the
  declared rank against the statically known result rank, so a wrong-rank
  ascription is accepted and eval silently returns a contradicting rank).

## Part I: contracts

### C1 The extent contract, restated

This section restates the controlling normative text so the phases can cite
one place; the numbered spec remains the authority and this restatement
decides nothing.

- **C1.1 Admissibility is typing, not provenance.** An `expand` size is any
  well-typed `int64` expression; a `reshape` shape is a `List[int64]` of
  statically known arity whose elements are any well-typed `int64`
  expressions (§4.7.2, §4.7.3). No stage - checker, lowerer, or backend -
  may reject an extent for its provenance, require a literal, or substitute
  a default (§4.7.2, §4.7.4).
- **C1.2 Identity is proof-gated, equality is guarded.** A literal produces
  a literal extent; an in-scope symbolic dimension may preserve its name; a
  direct shape read may preserve an input dimension identity only when
  ordinary type reasoning proves it; every other expression produces a
  fresh runtime extent `(d-name {} *)`. A surrounding literal or named
  claim imposes an execution-time equality guard that traps `Domain` on
  mismatch; it never makes the expression illegal (§4.7.2, §4.7.3, §4.7.6).
  Name-preservation walks may therefore survive as best-effort
  *refinement*, but their failure mode is "fresh anonymous extent plus
  guard", never rejection.
- **C1.3 Extents are `int64`, axes are `int32`** ([05-DIM-1..3]), shipped.
  The residue this plan owns: the `sourceless_expand_size_error` text
  (`app_shape_helpers.rs:210`) and the lowerer's arithmetic-size and
  post-node rejection texts still recommend `cast(N, int32)` extents and
  cite "§4.7.2 Form-3"; `extract_int_for_dim`'s doc comment still
  describes an `int32`-pinned outer unification; a `lower.rs` comment
  still names `cast(len, int32)` as the canonical idiom. (The axis-slot
  diagnostics that recommend `cast(N, int32)` *axes* are correct and stay.)
  [#1112] should be verified against the shipped state and closed rather
  than re-implemented.

### C2 One resolver: the provenance classifier is deleted, not extended

The checker types an extent expression as ordinary `int64` dataflow and
lowers it as such. Deliverables:

- **C2.1** `SizeClass`, `classify_expand_size`, `classify_arith_app`,
  `sourceless_expand_size_error`, `Env::size_provenance` and its
  mark/clear plumbing, and the lowerer's mirror rejections are removed.
  The structural lock is deletion itself: after this phase there is no
  construct in the tree that can express "reject an extent for its
  provenance", so [#1266], [#569], and [#578]'s expand half stop being
  fixable bugs and become unwritable ones. A re-introduction requires
  re-authoring a rejection channel, which review evaluates against §4.7.2's
  explicit prohibition.
- **C2.2** What survives, reclassified as refinement: static folding of
  literal arithmetic (one folder, shared or contract-tested against both
  consumers, closing the `div` drift), and Form-2-style named-dimension
  stamping where the name is proven (C1.2). Refinement failure yields a
  fresh anonymous extent plus a runtime guard, never an error.
- **C2.3** Every expand result is a constructed tensor type whose rank is
  statically known (operand rank, or operand rank + 1, per the deferral
  choice of §C3). Consequences, delivered together: the result always
  passes the annotation stamp gate, so the [#597] check-vs-eval asymmetry
  closes; and the declared or ascribed type is validated against that
  statically known rank, so the [#609] wrong-rank acceptance closes. The
  `_ => return subst.apply(result_ty)` early exit that currently produces
  both holes is removed with the classifier.
- **C2.4** The lowering and backends materialize any well-typed extent
  expression as scalar `int64` dataflow (§4.7.4). The C lane's existing
  declare-vs-guard site machinery is the fallback for every unproven
  identity claim.

### C3 The deferral: one language choice, total and deterministic

The insert-vs-replace ambiguity of positional `expand` is the one genuine
deferred decision in this surface, and §4.7.2 already specifies its
selection rule (declared result or first shape-bearing consumer; the
same-rank default at shape-neutral forcing; trailing insertion at
`axis == rank(x)`; materialized default when nothing fixes it). Three
deliverables make the implementation total against that rule:

- **C3.1 A closed consumer-disposition vocabulary.** Every builtin declares
  whether it is shape-bearing (selects the pending candidate),
  shape-neutral (defers), or shape-forcing (triggers the documented
  default), in a closed table consumed exhaustively - the
  dependency-bottom pattern of [#730] Phase 2, with in-tree precedent in
  `AxisArgumentLayout` on `BuiltinDecl`. A builtin added without a
  disposition fails to compile. The comparison family's special-cased
  result construction (`app_post.rs:568`) is rewritten to route through
  unification so a consumer that produces a shape necessarily binds its
  operands' pending candidates; [#1265] closes here, and its
  `expand -> add -> eq` variant closes with it because relocation-without-
  selection stops being expressible.
- **C3.2 Deterministic stores.** The deferred-constraint stores iterate in
  a canonical order (for `TypeVar` keys, allocation order, which is source
  order). The ordered-store mechanism, its tripwire, and the repeated-run
  stability oracle are owned by
  [`hash_order_determinism.md`](hash_order_determinism.md) ([#1341]);
  this plan consumes them and owns *which* verdict the settled order
  produces.
- **C3.3 The settlement-order amendment (numbered-spec prerequisite).**
  Which default a freeze-point expand takes is specified; the *order in
  which coupled deferred defaults settle* is not, and [#1338] proves the
  order is semantically visible (the two hash orders disagree on rank).
  Before or with the phase that implements C3.1/C3.2, `spec/04` §4.7.2
  must be amended to state the settlement order. This plan proposes
  source order (first-deferred settles first) as the amendment content,
  because it matches allocation order, is implementable without lookahead,
  and settled the [#1338] reproducer on the accepting verdict; the
  numbered spec, not this document, decides. Landing the mechanism without
  the amendment would be permission-to-mandate escalation in the wrong
  direction: a deterministic implementation of an unspecified rule.

### C4 Backend representation: value edges, not resolved names

§4.7.4 forbids a backend from requiring a declared provenance; the current
occurrence pass is a provenance requirement enforced by ICE. Target state:
a runtime extent in the DAG is a value edge - a scalar `int64` node input
of the consuming op (`RtDim::Node` generalized to the canonical
non-literal form) - so "find the declaring Load for this name" ceases to
exist as an operation, and the `dag.rs:1793`/`:1840` ICE class becomes
unconstructible. [#665] and [#592] close here.

This is the phase with real design risk. The shipped [#616] machinery
(op-declared dims, mid-evaluation binding, first-site-declares +
later-site-guards, the wildcard rules) exists because of measured
corrections recorded in
[`chelis_616_runtime_movement_dims.md`](chelis_616_runtime_movement_dims.md)
§2, and grad/vmap interact with symbolic dims through [#513]'s open gaps.
The phase therefore carries an interim ratchet that is landable
immediately and independently:

- **C4.1 (interim ratchet)** `op_declared_output_axes` and
  `op_declarable_axes` become total by construction: one declaration per
  `RiscOp` variant of its output-axis sources, consumed exhaustively, so
  adding an op - or, as in [#665], adding a *flow* (an op-declared axis
  crossing an `Expand`'s kept axes) - without a disposition is a compile
  error, and an undeclared occurrence downgrades from ICE to a typed
  `Unsupported` rejection under [#730]'s §C2 channel until C4.2 lands.
- **C4.2 (target)** the value-edge representation, with the eval, C, HIP,
  and Metal lanes reading extents as scalar operands and the equality
  guards of C1.2 emitted where identity claims are unproven.

### C5 The class oracle

One authoritative runner, `scripts/runtime_extent_oracle.py`, exit 0 with
final line `RUNTIME EXTENT ORACLE: PASS`. Legs:

1. **Parity corpus.** A generated matrix of extent-producing forms
   (literal; suffixed literal arithmetic; function parameter; `let`
   binding; record field access; pipe/lambda spelling; user-function
   result; module-scope binding; `shape()` reads through each) crossed
   with extent consumers (`expand`, `reshape`, `shrink`, `pad`, `stride`,
   `reduce_window_*`) crossed with lanes (`check`, `eval`,
   `build --target c` compiled and executed). Assertion: acceptance
   parity across lanes and value agreement where executed. Every [#1277]
   child's reproducer is a named corpus row.
2. **Negative controls.** Genuinely invalid extents - negative size,
   non-`int64` dtype, out-of-range axis, rank-contradicting ascription -
   fail on every lane with the owning diagnostic. Negative-test parity is
   the repo baseline; the controls keep C2's deletion from overshooting
   into fail-open.
3. **Stability.** Each corpus verdict is asserted across K fresh-process
   runs (mechanism owned by [#1341]'s oracle; this leg pins the extent
   corpus specifically, because [#1338] demonstrated a single-run
   assertion passes about two thirds of the time).
4. **Guard execution.** Programs whose declared extents are contradicted
   at runtime trap `Domain` (C1.2) rather than mis-executing, on eval and
   compiled C.

Until Phase 2 lands, legs 1 and 4 run in a documented expected-failure
mode over the rows the current tree rejects; a row flipping to accepted is
ratchet progress and a row flipping to rejected is a regression. The
oracle is a guard artifact under the repo rule: editing it to make a
change pass is never the fix.

## Part II: boundary law

Per the sibling plans' discipline:

- Freeze points are declared at each phase exit (Part III) and changing
  one requires updating this document and [#1277] in the same change set.
- Controls never move: a red oracle row turns green only by the tree
  changing, never by editing the row.
- A discovery mid-phase forks into its own child issue and corpus row
  rather than widening the phase in flight.
- Spec-first: any behavior this plan changes that is user-visible lands
  with its `spec/04`/`spec/05` reading confirmed, and where the numbered
  spec is silent (C3.3) the amendment lands first. New `[05-OP-N]` or
  `[04-NUM-N]` atoms, if any prove necessary, follow the registry
  regeneration rule (`scripts/generate_rejection_registries.py --write`).

## Part III: phases

Every phase names one authoritative oracle. Recommended order below;
Phase 1 is independent and may land any time, and Phase 4's C4.1 ratchet
may land ahead of Phases 2-3.

- **Phase 0 - oracle skeleton and measured baseline.**
  *Deliver:* `scripts/runtime_extent_oracle.py` with legs 1-4, the
  generated corpus, and a checked-in baseline of the current tree's
  per-row verdicts (accepted / rejected / ICE / lane-divergent), so every
  later phase's effect is a measured diff.
  *Frozen at exit:* corpus row identities; the baseline file as a
  ratchet floor.
  *Oracle:* the runner in baseline mode - exit 0 with
  `RUNTIME EXTENT ORACLE: BASELINE OK` and zero unexplained divergences
  from the checked-in baseline.
- **Phase 1 - dtype residue sweep (C1.3).**
  *Deliver:* corrected diagnostic and comment texts; [#1112] verified
  against the shipped [05-DIM] state and closed.
  *Explicitly not yours:* any extent-typing change ([#729] owns dtype
  semantics).
  *Oracle:* `cargo nextest run -p chelis-types --test
  issue_1112_extent_dtypes` green plus a repo grep gate in the class
  runner asserting no remaining `cast(N, int32)` *extent* recommendation
  (axis recommendations exempt).
- **Phase 2 - one resolver (C2).**
  *Inherit:* Phase 0's baseline.
  *Deliver:* C2.1-C2.4; [#1266], [#569], [#597], [#609] close; [#578]'s
  expand-mechanism half closes (its named-axis-at-symbolic-rank legality
  question stays with the rank-polymorphism docs, §I1).
  *Frozen at exit:* the deleted-channel invariant (no
  provenance-rejection construct in `chelis-types` or `chelis-ir`); the
  single static folder.
  *Oracle:* oracle legs 1, 2, and 4 fully green (no expected-failure rows
  remaining for checker- and lowerer-owned rejections).
- **Phase 3 - deferral totality and determinism (C3).**
  *Inherit:* Phase 2's constructed result types; the §4.7.2
  settlement-order amendment (C3.3), which must be merged first or in the
  same change set.
  *Deliver:* the disposition vocabulary; the comparison-family rewrite;
  ordered stores via [#1341]; [#1265] closes, [#1338]'s selection half
  closes.
  *Frozen at exit:* the disposition table's variant set and the
  settlement-order behavior.
  *Oracle:* oracle leg 3 green with the [#1338] reproducer as a named
  row, plus the disposition table's exhaustiveness compile-lock.
- **Phase 4 - backend value edges (C4).**
  *Deliver:* C4.1 immediately; C4.2 behind its own design review against
  the [#616] corrections; [#665] and [#592] close (C4.1 downgrades them
  to typed rejections; C4.2 makes them execute).
  *Explicitly not yours:* grad's symbolic-window machinery ([#513]) and
  runtime reduction windows ([#1298]).
  *Oracle:* oracle leg 1's build-lane rows green including the [#665]
  and [#592] reproducers executed, and the DAG occurrence-pass ICE tests
  retired in favor of typed-rejection or execution tests.

## Part IV: bookkeeping

### I1 Interlocks

- **[#729] / [`dtype_semantics.md`](dtype_semantics.md):** owns extent and
  axis dtypes (shipped) and, through [#1296]/[#1298], runtime axes and
  reduction-window operands. Boundary: that plan owns what an extent *is
  typed as* and which ops gain runtime-valued operands; this plan owns how
  an already-typed extent *resolves* through checker, deferral, and
  backend. Neither edits the other's oracles.
- **[#1341] / [`hash_order_determinism.md`](hash_order_determinism.md):**
  owns ordered-iteration stores, the lint ratchet, and the K-run
  stability harness. This plan consumes them in C3.2 and C5 leg 3 and
  owns which verdict determinism settles on. [#1338] is parented to
  [#1277] with an "Also part of [#1341]" cross-link; the split is
  recorded in both bodies.
- **[#731] / [`checker_totality.md`](checker_totality.md):** owns the
  witnessed-diagnostic channel. C2's deletions must not create silent
  paths: every remaining rejection in this surface flows through the
  [#731] channel, and the [#609] rank validation lands as an ordinary
  witnessed `CheckError`.
- **Rank polymorphism
  ([`rank_polymorphism.md`](rank_polymorphism.md) and
  [`rank_polymorphism_tier3_followups.md`](rank_polymorphism_tier3_followups.md)):**
  owns whether `shape(x, name)` and named-axis `expand` are legal inside a
  `..r` body and under what restrictions ([#578]'s legality half). This
  plan delivers the resolution mechanism wherever those docs and the
  numbered spec make the form legal.
- **[#730] / [`loud_unsupported.md`](loud_unsupported.md):** C4.1's
  interim downgrade emits through the [#730] §C2 typed `Unsupported`
  channel with a registered authority; no new panic and no silent
  substitute.

### Issue map

| issue | one clause | phase |
|---|---|---|
| [#1266] | the walk does not follow a record field access | 2 |
| [#569] | lint's pipe rewrite defeats the walk; `lint --fix` breaks a build | 2 |
| [#597] | Static-sized expand gets no type metadata; eval rejects what build accepts | 2 |
| [#609] | wrong-rank ascription on an expand result accepted; eval silently contradicts it | 2 |
| [#578] | named-axis expand broadcast-back in a `..r` body rejects | 2 (mechanism) + rank-poly docs (legality) |
| [#1265] | a comparison consumer never selects a deferred expand shape | 3 |
| [#1338] | freeze-point defaults settle in hash order; verdict is random | 3 (selection + spec amendment) / [#1341] (mechanism) |
| [#665] | expand over a movement-op runtime wildcard ICEs the occurrence pass | 4 |
| [#592] | grad+vmap backward `Expand{size:Sym}` trips the same guard | 4 |
| [#1112] | extent producer/consumer dtype mismatch (shipped v0.18.3; residue only) | 1 |

### What this plan does not own

Data-dependent output shapes ([#600]), type-level dimension arithmetic
([#526]), grad's symbolic-window gaps ([#513]), the sibling backend
symbolic-dim defects not parented here ([#593], [#888] - candidates for
[#1277] parenting when picked up, but not deliverables of a phase above),
and every dtype-semantics decision.

[#513]: https://github.com/Chelis-Lang/chelis/issues/513
[#526]: https://github.com/Chelis-Lang/chelis/issues/526
[#569]: https://github.com/Chelis-Lang/chelis/issues/569
[#578]: https://github.com/Chelis-Lang/chelis/issues/578
[#592]: https://github.com/Chelis-Lang/chelis/issues/592
[#593]: https://github.com/Chelis-Lang/chelis/issues/593
[#597]: https://github.com/Chelis-Lang/chelis/issues/597
[#600]: https://github.com/Chelis-Lang/chelis/issues/600
[#609]: https://github.com/Chelis-Lang/chelis/issues/609
[#616]: https://github.com/Chelis-Lang/chelis/issues/616
[#665]: https://github.com/Chelis-Lang/chelis/issues/665
[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#731]: https://github.com/Chelis-Lang/chelis/issues/731
[#888]: https://github.com/Chelis-Lang/chelis/issues/888
[#1112]: https://github.com/Chelis-Lang/chelis/issues/1112
[#1149]: https://github.com/Chelis-Lang/chelis/pull/1149
[#1265]: https://github.com/Chelis-Lang/chelis/issues/1265
[#1266]: https://github.com/Chelis-Lang/chelis/issues/1266
[#1277]: https://github.com/Chelis-Lang/chelis/issues/1277
[#1283]: https://github.com/Chelis-Lang/chelis/pull/1283
[#1296]: https://github.com/Chelis-Lang/chelis/issues/1296
[#1298]: https://github.com/Chelis-Lang/chelis/issues/1298
[#1338]: https://github.com/Chelis-Lang/chelis/issues/1338
[#1341]: https://github.com/Chelis-Lang/chelis/issues/1341
