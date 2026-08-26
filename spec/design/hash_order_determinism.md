# Hash-Order Determinism: observable behavior never depends on hash iteration order

**Status:** PROPOSED. No phase is implemented. Tracking issue: [#1341].
Code evidence was measured on `main` at `647c76a9` unless a different commit
is named; function names are the durable anchors and line numbers are a
convenience of that commit.
**Owning specs:** none normative - this plan changes no language semantics.
It enforces an implementation invariant the repo already states piecewise:
the Contract Invariants section of `AGENTS.md` (machine-facing contracts are
invariants locked by tests), the byte-determinism regression suite
(`crates/chelis-backend-c/tests/codegen_determinism.rs`), and the instance
record `spec/upstream-bugs/host-emit-hashmap-iteration-nondeterminism.md`.
Where a settled iteration order is user-visible language behavior (the
freeze-point `expand` settlement order), the decision belongs to
`spec/04-type-system.md` and is owned by
[`runtime_extents.md`](runtime_extents.md) §C3.3, not here.
**Class fixed:** [#1341] - `std::collections::HashMap`/`HashSet` are
randomly seeded, so any iteration whose order can reach observable behavior
(a checker verdict, a dispatch choice, an ABI slot assignment, emitted
artifact bytes, serialized metadata) makes that behavior random across runs.
**Sibling plans:** [`runtime_extents.md`](runtime_extents.md) ([#1277])
consumes this plan's ordered stores and stability harness for the deferred
expand machinery; [`loud_unsupported.md`](loud_unsupported.md) ([#730]) is
the census-discipline precedent (execution-backed dispositions, guard
artifacts, ratchets); [`checker_totality.md`](checker_totality.md) ([#731])
is the unconstructibility precedent this plan's Phase 2 imitates at the
store level.

## Summary

Six instances of one defect are confirmed across four subsystems, three of
them verdict- or dispatch-affecting. Every resolved instance was found by a
downstream symptom, repaired by a local sort, and pinned by a site-specific
regression test that structurally cannot see the next site. The invariant
"iteration order does not reach behavior" is today maintained by site
comments and vigilance - the weakest rung of the repo's enforcement ladder
(compile-error > lint > tripwire > gate > prose). The plan: an
execution-backed census of iteration sites (Phase 0), landing the two known
open sites with a stability regression (Phase 1), converting
behavior-reaching stores to types whose public API only iterates in a
canonical order - making the defect unwritable rather than remembered
(Phase 2) - and a supporting lint plus one fresh-process stability oracle
(Phase 3).

## The instance record

| site | observable effect | status |
|---|---|---|
| `Subst::materialize_deferred_expand_defaults` (`crates/chelis-types/src/unify.rs:996`; `.keys()` over `HashMap<TypeVar, _>`) | the checker accepts or rejects the same well-typed file at random: 17 accept / 7 reject over 24 `check` runs; `eval` splits three ways; `build` rejected 5 of 10. Proof by intervention: sorting this one iteration yields 24/24 accept, the correct verdict | OPEN - [#1338] |
| `Subst::take_deferred_reshapes_for_input` (`crates/chelis-types/src/unify.rs:929`; `iter_mut()` over the same map shape) | coupled deferred-reshape resolution order is hash order; latent - no known reproducer lands on it, which is itself the problem | OPEN - [#1338], same change set |
| linearity capture walk (`free_vars`, `crates/chelis-types/src/linearity.rs`) | the same program compiled or failed `UseAfterConsume` run to run when two captures share an alias chain; measured 11/12 reject, 1/12 accept | resolved in the chelis#1200 fix; the site comment records the sort as "semantically necessary, not merely cosmetic", and chelis#1209's generation ids did NOT remove the order-sensitivity |
| ADT variant dispatch (`lookup_variant_preferring_shape`, `crates/chelis-types/src/adt.rs`) | `self.defs.iter()` leaked hash order into which variant "first match" selects, in both the shape-match and fallback paths | resolved by a name sort; no owning issue recorded at the site |
| input-validation preamble (`emit_input_shape_preamble` in `chelis-backend-c` and the HIP host/device siblings) | validation-block order varied across builds; byte-nonidentical C from identical input | resolved 2026-05-01 at `17a28b9`; record: `spec/upstream-bugs/host-emit-hashmap-iteration-nondeterminism.md`; regression: `codegen_determinism.rs` |
| Load pre-creation (`lower_subexpr_program_inner`, `crates/chelis-ir/src/lower.rs`; `HashMap<String, TensorType>`) | tensor-kernel input-slot order flipped build to build, so emitted C bytes differed; slipped past `codegen_determinism.rs`, which constructs DAGs directly and bypasses lowering | resolved 2026-07 in the chelis#469 wave: name-sorted pre-creation; regression: `rank_poly_tier3::form3_bias_broadcast_c_is_byte_deterministic` |

The escalation across instances is why this is a class and not a
coincidence: presentation bytes, then an ABI slot assignment, then dispatch
and linearity verdicts, now the type checker's accept/reject itself. The
audit surface is real: 44 files across `chelis-types`, `chelis-ir`, and the
three backends mention `HashMap`/`HashSet` at the measured commit, and
nothing marks which of their iterations reach behavior.

**Not in this class:** language-level nondeterminism semantics. `scatter`
duplicate-index collisions and
`AdRejectionReason::NonDeterministicAtDuplicateIndices` are deliberate,
specified rejections about the *program's* semantics, not the compiler's
hash seeds. Also out of scope: [#1198] (Reef archive reproducibility) and
[#823] (benchmark reproducibility), which concern packaging and
measurement.

## Why the class persists

- The defect is invisible at the site: `map.iter()` reads identically
  whether or not its order reaches behavior. The taint is a property of the
  *consumer*, discovered only when a symptom appears downstream.
- Site-specific tests do not generalize: `codegen_determinism.rs` pinned
  the preamble instance and structurally could not see the lowering
  instance; a single-run test on the [#1338] reproducer passes roughly two
  runs in three.
- CI hides it: a rejected run followed by a clean retry reads as
  infrastructure flake, and nothing in the suite pins determinism of a
  checker verdict as an invariant. Measured cost: the transcript preserved
  for [#1222] shows an agent chasing type errors that appeared and
  disappeared across identical rebuilds - 72 episodes, no deliverable.

## Part I: contracts

### C1 The determinism contract

For a fixed input (source bytes, toolchain, target, flags, environment as
the reproducibility docs already scope it), every observable output of
`chelis check`, `eval`, `build`, and the library entry points is invariant
under the process's hash seeds. Observable means: the verdict and complete
diagnostic list (content *and* order), evaluation results, emitted artifact
bytes, serialized/cached metadata, and any ABI-visible ordering (kernel
input slots, exported declaration order). Internal orders that provably
never reach those surfaces are exempt, but the exemption is a census
disposition (C2), never an assumption.

### C2 The census

One census under `docs/investigations/hash_order_census.md` enumerates
every iteration over an unordered map/set in the compiler crates
(`chelis-types`, `chelis-ir`, `chelis-backend-*`, `chelis-eval`,
`chelis-cli`'s compilation path, `chelis-compiler-api`). Schema per row:
site (crate, function), key type, consumer chain (what the order feeds),
and exactly one disposition:

- **(a) unreachable** - order provably cannot reach C1's surfaces, with
  the reason stated (e.g. the iteration folds into an order-insensitive
  reduction);
- **(b) converted** - the store is an ordered structure or a newtyped
  store per C3;
- **(c) boundary-sorted** - a sort at the consumption boundary, with the
  canonical order named and a comment citing this document.

Dispositions rest on execution where feasible (a regression test or a
planted-mutation check that perturbs the order and observes the surface),
inspection where not, and the census says which per row - the [#730]
census discipline. Mechanical seeding (grep over iteration idioms, the
Phase 3 lint's findings) feeds the census but does not close a row.

### C3 Store discipline

- **C3.1 Canonical orders.** For `TypeVar` keys, allocation order (a
  `u32`, which for the checker's deferred stores is source order). For
  string keys, lexicographic byte order. For `NodeId`-keyed DAG maps, node
  id order. A store may not invent a per-site bespoke order when one of
  these applies.
- **C3.2 Ordered structures.** `BTreeMap`/`BTreeSet` are the default
  conversion target (no new dependency; iteration is the canonical key
  order). An insertion-ordered map is admissible where insertion order
  *is* the canonical order, as a deliberate dependency decision recorded
  in the census row.
- **C3.3 Unwritability.** For stores with a history of instances (the
  `Subst` deferred-constraint stores first), conversion goes through a
  newtyped store whose public API offers only ordered iteration - raw
  map access is private, so reintroducing an unordered walk is a compile
  error at the store boundary, the same move as [#731]'s unconstructible
  silent `Type::Error`. The newtype may additionally expose a test-only
  adversarial mode (reversed canonical order) so order-sensitivity is
  testable deterministically instead of statistically.
- **C3.4 Serialization dividend.** `Subst`'s deferred stores serialize
  into cached library contexts; ordered stores make those bytes
  deterministic too, which the reproducibility work ([#1198]) inherits
  for free but does not depend on.

### C4 The lint ratchet, with its gap stated

`clippy::iter_over_hash_type` is enabled at `deny` for the census crates.
Its gap is stated here so it is never mistaken for the oracle: the lint
fires on `for`-loop iteration over hash types, and the exact [#1338] shape
(`.keys().copied().collect::<Vec<_>>()`, then iterate the `Vec`) passes it
without a warning. The lint is a supply-reduction ratchet on the easy
majority of sites; C2's census and C5's oracle carry the guarantee.
Sites with a census disposition of (a) or (c) carry the narrowest
`#[allow]` with a comment naming their census row; a bare `#[allow]`
without a row is a review-blocking finding.

### C5 The class oracle

One authoritative runner, `scripts/hash_order_determinism_oracle.py`, exit
0 with final line `HASH ORDER DETERMINISM ORACLE: PASS`. Legs:

1. **Verdict stability.** The [#1338] reproducer and one representative
   program per verdict-affecting census row, each run K times as *fresh
   subprocesses* per lane (`check`, `eval`, `build`), asserting an
   identical verdict, diagnostic list, and output every run. Fresh
   processes are the perturbation mechanism because the standard
   library's hasher is randomly seeded per process (and per map instance
   within one); an in-process loop is not trusted to vary the order. This
   is the same mechanism `form3_bias_broadcast_c_is_byte_deterministic`
   already uses for build bytes, generalized to check and eval verdicts.
2. **Byte stability.** The existing `codegen_determinism.rs` and
   CLI-level byte-equality tests run under the oracle's identity check so
   they cannot be silently dropped from the suite.
3. **Census liveness.** The census file parses, every row carries exactly
   one disposition, every (c) row's named sort exists at the cited site
   (a source-anchored check), and every `#[allow]` of the C4 lint in the
   census crates maps to a row.
4. **Adversarial order** (after Phase 2): the newtyped stores' reversed
   mode runs the leg-1 corpus in-process and asserts identical verdicts,
   turning the statistical property into a deterministic one.

A K-run leg is probabilistic by nature; K is chosen per row from the
measured flake rate (for [#1338], 24 runs bounded the accept rate at
~2/3, so K=24 gives better than one-in-ten-thousand odds of a silent
pass) and recorded in the census row. The oracle is a guard artifact:
editing a row, a baseline, or K to make a change pass is never the fix.

## Part II: phases

Every phase names one authoritative oracle.

- **Phase 0 - census.**
  *Deliver:* `docs/investigations/hash_order_census.md` per C2, seeded
  mechanically, dispositioned by hand, with the six recorded instances as
  the first rows (the four resolved ones as (c) rows pointing at their
  existing sorts and tests).
  *Frozen at exit:* the census schema and the disposition vocabulary.
  *Oracle:* oracle leg 3 (census liveness) green.
- **Phase 1 - the open sites.**
  *Deliver:* [#1338]'s two `unify.rs` iterations settled in canonical
  `TypeVar` order, landed with the leg-1 stability regression for the
  reproducer. The freeze-point *settlement-order* language decision is
  [`runtime_extents.md`](runtime_extents.md) §C3.3's numbered-spec
  amendment and is a prerequisite of, or lands with, this change - the
  sort must implement the specified order, not merely *an* order.
  *Explicitly not yours:* the consumer-selection totality work
  ([#1265]), which is [#1277] Phase 3.
  *Oracle:* oracle leg 1 green with the [#1338] row at its recorded K.
- **Phase 2 - unwritability at the store.**
  *Deliver:* C3.3 newtyped stores for the `Subst` deferred-constraint
  maps and every (b)-dispositioned census row; the test-only adversarial
  mode.
  *Frozen at exit:* the store boundary (raw map access private).
  *Oracle:* oracle leg 4 green plus the compile-time boundary (a
  `compile_fail` doctest on raw access, per the repo's existing
  visibility-lock pattern).
- **Phase 3 - lint and continuous execution.**
  *Deliver:* the C4 lint at deny with row-mapped allows; the full runner
  wired into the gate's owning stage so legs 1-4 run continuously; the
  census row for any site the lint newly surfaces.
  *Oracle:* `scripts/hash_order_determinism_oracle.py` exit 0 with
  `HASH ORDER DETERMINISM ORACLE: PASS` on the tree, and red on a planted
  unordered iteration in a converted store (the negative control).

## Part III: interlocks and non-goals

- **[#1277] / [`runtime_extents.md`](runtime_extents.md):** this plan owns
  that the compiler gives the *same* answer every run; that plan owns
  *which* answer. [#1338] is structurally parented to [#1277] and carries
  an "Also part of [#1341]" cross-link; the mechanism half (Phases 1-2
  here) and the selection/spec half ([#1277] Phase 3) may land in either
  order, and each plan's oracle is unaffected by the other's landing
  state except where a leg is explicitly shared (leg 1's reproducer).
- **[#730] / [`loud_unsupported.md`](loud_unsupported.md):** census
  discipline precedent; no contract dependency.
- **Non-goals:** language-level nondeterminism semantics (scatter
  duplicate-index rejection, AD rejections); floating-point
  reduction-order semantics (owned by the numbered specs and [#729]'s
  behavior contracts); packaging and benchmark reproducibility ([#1198],
  [#823]); parallelism-induced nondeterminism (no compiler stage is
  currently parallel over these stores; if one becomes parallel, that
  change owns extending C1 to it).

[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#731]: https://github.com/Chelis-Lang/chelis/issues/731
[#823]: https://github.com/Chelis-Lang/chelis/issues/823
[#1198]: https://github.com/Chelis-Lang/chelis/issues/1198
[#1222]: https://github.com/Chelis-Lang/chelis/issues/1222
[#1265]: https://github.com/Chelis-Lang/chelis/issues/1265
[#1277]: https://github.com/Chelis-Lang/chelis/issues/1277
[#1338]: https://github.com/Chelis-Lang/chelis/issues/1338
[#1341]: https://github.com/Chelis-Lang/chelis/issues/1341
