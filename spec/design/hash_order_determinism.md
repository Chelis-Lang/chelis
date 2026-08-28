# Hash-Order Determinism: observable behavior never depends on hash iteration order

**Status:** PROPOSED. No phase is implemented. Tracking issue: [#1341].
Code evidence was rechecked on `main` at `12c22520` unless a different commit is
named. Function names are the durable anchors; line numbers are conveniences of
that commit.
**Owning specs:** none for the compiler-wide determinism invariant. This plan
changes no language semantics. It implements the machine-facing contract in
`AGENTS.md` that observable invariants are explicit and executable. When an
iteration order selects a user-visible language result, its canonical order must
first be decided by the owning numbered spec. In particular, [#1277] owns the
missing `spec/04-type-system.md` §4.7.2 settlement-order rule exposed by [#1338].
**Class fixed:** [#1341] - `std::collections::HashMap` and `HashSet` seed their
hashers per process, so any order that reaches a checker verdict, diagnostic,
dispatch choice, evaluation result, emitted artifact, serialized cache payload,
or ABI-visible sequence makes that output vary across identical runs.
**Precedents:** [`loud_unsupported.md`](loud_unsupported.md) ([#730]) supplies
the execution-backed census discipline; [`checker_totality.md`](checker_totality.md)
([#731]) supplies the private, structurally unconstructible boundary pattern.

## Summary

Six instances of one defect are confirmed across four subsystems, three of them
verdict- or dispatch-affecting. Each resolved instance was found from a downstream
symptom, repaired by a local sort, and pinned by a test that could not see the next
site. The invariant is therefore still maintained by comments and vigilance.

Closure has four structural parts:

1. derive the production compiler universe from Cargo metadata and maintain an
   exact source-to-census bijection for hash-backed carriers and their consumers;
2. require one authority-backed canonical order wherever order reaches behavior;
3. inventory the complete Serde graph of every persisted cache payload and make
   its bytes canonical; and
4. make those boundaries private, mutation-tested, and continuously executable
   through one class oracle.

Fresh-process repetition remains useful regression coverage, but it is not proof
of completeness and carries no false-pass probability claim.

## The instance record

| site | observable effect | status |
|---|---|---|
| `Subst::materialize_deferred_expand_defaults` (`unify.rs`; `.keys()` over `HashMap<TypeVar, _>`) | identical source is accepted or rejected across processes; the original 24-run sample split 17/7, and a later review split 15/9 | OPEN - [#1338] |
| `Subst::take_deferred_reshapes_for_input` (`unify.rs`; `iter_mut()` over the same map shape) | coupled deferred-reshape resolution order is hash order; no reproducer currently isolates this second site | OPEN - [#1338], same implementation change |
| linearity capture walk (`linearity.rs`, `free_vars`) | one program alternated between success and `UseAfterConsume` | resolved in the chelis#1200 fix by a semantically required sort |
| ADT variant dispatch (`adt.rs`, `lookup_variant_preferring_shape`) | hash order selected the first matching variant | resolved by sorting candidates by ADT name |
| input-validation preamble (C and HIP emitters) | validation-block order changed emitted bytes | resolved at `17a28b9`; regression in `codegen_determinism.rs` |
| Load pre-creation (`chelis-ir`, `lower_subexpr_program_inner`) | input-slot order and emitted C bytes varied after lowering | resolved in the chelis#469 wave; CLI regression in `rank_poly_tier3` |

The escalation is the class evidence: presentation bytes, ABI slot assignment,
dispatch and linearity verdicts, and now the type checker's accept/reject result.

**Not in this class:** deliberate language-level nondeterminism semantics such as
duplicate-index `scatter` rejection; floating-point reduction-order semantics;
packaging reproducibility ([#1198]); benchmark reproducibility ([#823]); or future
parallel scheduling. Each has a different authority and oracle.

## Part I: contracts

### C1 Observable determinism

For fixed source bytes, compiler build, target, flags, and documented environment,
every observable result of `chelis check`, `eval`, `build`, and compiler library
entry points is invariant under process hash seeds. Observable results include:

- verdict and the complete ordered diagnostic list;
- evaluated values and traps;
- emitted source, object, and artifact bytes;
- persisted payload and envelope bytes;
- dispatch choices and ABI-visible ordering.

An internal order may be exempt only through a census row proving that its entire
consumer chain is order-insensitive. Absence from a hand-written list is never an
exemption.

### C2 Exact production census

The checked-in census is `docs/investigations/hash_order_census.md`. An executable
enumerator produces the identities that the census must classify exactly.

#### C2.1 Production universe

The enumerator reads `cargo metadata --format-version 1` rather than naming a
partial crate list. Its core anchors are `chelis-types`, `chelis-ir`, and the C,
HIP, and Metal backends. The audited package set is:

1. every workspace package with a non-test target that reaches a core anchor
   through normal or build dependencies; plus
2. the workspace dependency closure of those packages and the anchors.

It scans production `src/**/*.rs` for that set. Test, example, and benchmark
sources are evidence consumers, not production census inputs. A new package or
dependency enters automatically; no stale crate allowlist can hide it. This rule
includes the parser, Deep, effects, validation, pipeline-core, compiler API, CLI,
bindings, proof, and editor/server routes when they are in the derived graph.

#### C2.2 Source identities and bijection

A Rust liveness test parses the production sources with `syn` and resolves local
imports and type aliases conservatively. It enumerates:

- every `HashMap`/`HashSet` field, local, parameter, return carrier,
  constructor, and typed `collect` origin;
- every operation that can expose or consume its order, including `IntoIterator`,
  direct loops, `iter*`, `keys`, `values*`, `drain`, collected iterators, and
  iterator combinators; and
- every `Serialize`/`Deserialize` edge for a hash-backed carrier.

An identity is `(crate, module, enclosing item, carrier field or binding,
consumer)`. Anonymous or ambiguous carriers must gain a stable census marker;
unresolved aliases, duplicate identities, and unknown order-consuming forms fail
closed instead of being skipped.

The generated identities and census rows are bijective: a missing source row, a
stale census row, or two rows claiming one identity is an oracle failure. The
Clippy lint and grep may seed review, but neither defines this universe.

Each row records the carrier/key type, consumer chain, C1 surface, cache roots if
any, canonical-order authority, evidence, and exactly one disposition:

- **order-insensitive** - executable evidence or a checked algebraic argument
  proves that every consumer is insensitive to order;
- **ordered-store** - the carrier is a private ordered/newtyped store exposing
  only the row's canonical order; or
- **canonical-boundary** - a named boundary sorts or canonically serializes the
  complete carrier before order can escape.

#### C2.3 Required negative mutations

The liveness oracle must reject all of these planted changes:

1. a direct loop over a new hash map;
2. `keys().collect::<Vec<_>>()` followed by iteration;
3. the same collect with its vector type inferred;
4. an iterator combinator such as `for_each`;
5. a new hash-backed field under derived Serde;
6. a new carrier or consumer with no census row; and
7. an `#[allow(clippy::iter_over_hash_type)]` with no matching row.

The mutation suite is the completeness lock that the lint alone cannot provide.

### C3 Canonical order and private stores

#### C3.1 Authority before mechanism

A behavior-reaching order is not chosen merely because `Ord` exists. Its row must
name the authority for the order:

- language-result selection cites the owning numbered-spec rule;
- byte/presentation-only rows cite this implementation contract and state their
  key order (for example UTF-8 byte order or `NodeId` numeric order); and
- cache-only rows cite C4's canonical serialization rule.

For [#1338], `TypeVar(u32)` allocation order is an implementation candidate, not
semantic authority. Phase 1 is blocked until [#1277] amends `spec/04` to decide
the coupled deferred-obligation settlement order. The implementation must encode
that decided order exactly.

#### C3.2 Ordered representation

`BTreeMap`/`BTreeSet` are the default only when their `Ord` is the canonical order.
Otherwise the row uses a private newtype with a canonical iterator or a boundary
sort. An insertion-ordered map is allowed only when the owning authority explicitly
makes insertion order canonical.

Stores with prior behavior-affecting instances, beginning with the deferred
constraint stores, are newtyped. Raw map access is private. Production consumers
receive only the canonical iterator or an order-insensitive operation, so a new
unordered walk fails at compile time.

Tests construct equal contents through canonical, reverse, and fixed shuffled
insertion orders and through fresh hash states, then assert the same canonical
iterator and observable result. They never reverse the canonical settlement order:
that would test the stronger, out-of-scope property of order-independent resolution.

### C4 Complete canonical serialization

Persisted bytes are part of C1, so derived Serde is part of the census even though
its iteration executes in dependency code.

The serialization enumerator starts at every concrete call to
`cache_envelope::save<T>`, records each payload root, and walks its complete
Serde-reachable struct/enum graph. It includes derived implementations, manual
`Serialize` implementations, `serde(with)` modules, and nested containers. An
unknown custom edge fails closed and requires an explicit census row.

The `LibraryContext` graph is a mandatory representative. Its wire carrier reaches
`TypeEnv` and `CheckedProgram`; those reach `Subst`, `Env`, `AdtRegistry`, and other
hash-backed state. `Subst` alone currently serializes `types`, `dims`, `ranks`, both
deferred-constraint maps, and three lowered-level maps. Converting only the two
deferred stores cannot satisfy this contract.

Every reachable unordered carrier is either converted to an ordered representation
or serialized by a canonical adapter that sorts the complete key/value sequence.
The cache format version is bumped whenever the canonical wire shape changes under
the existing cache exactness policy. Representative complete payloads are built with
multiple insertion orders and serialized in 24 fresh processes; payload bytes,
payload SHA-256, and final envelope bytes must all match exactly.

This track owns deterministic compiler-cache bytes. [#1198] may consume that result
for archive reproducibility, but it is not the authority or a prerequisite here.

### C5 Lint ratchet

`clippy::iter_over_hash_type` is denied throughout the C2 production universe. Its
known limitation is explicit: indirect collection and iterator-combinator forms can
escape it. A narrow allow is valid only when it names one census identity whose
disposition requires the direct hash carrier. The C2 source enumerator and mutations,
not Clippy, prove coverage.

### C6 Class oracle

The authoritative runner is `scripts/hash_order_determinism_oracle.py`. Exit 0 and
the final line `HASH ORDER DETERMINISM ORACLE: PASS` require every leg:

1. **Universe and bijection:** Cargo-derived production scope, source identities,
   serialization roots, and census rows are exact and duplicate-free.
2. **Mutation rejection:** all C2.3 mutations fail for the intended missing-row or
   forbidden-order reason.
3. **Canonical boundaries:** ordered/newtyped stores pass insertion/hash-order
   perturbations, and raw access fails to compile.
4. **Cache bytes:** every representative complete payload is byte-identical across
   insertion permutations and 24 fresh processes.
5. **Public surfaces:** the [#1338] reproducer and at least one representative for
   each C1 output class run in 24 fresh processes with identical verdicts,
   diagnostics, values, and bytes.
6. **Existing regressions and lint:** the backend and CLI byte-determinism tests
   remain present and green, and every lint allow maps to the census.

The 24-process legs are supporting randomized regressions with a fixed execution
budget. They make no statistical confidence or false-pass claim and can never replace
the deterministic structural and mutation legs.

## Part II: phases

### Phase 0 - executable universe and census

**Deliver:** the Cargo-derived package enumerator, source/Serde identity extractor,
census schema, all currently discovered rows, and the C2.3 mutation suite.

**Frozen at exit:** universe derivation, identity schema, dispositions, and the rule
that unknown syntax/serialization edges fail closed.

**Oracle:** `.venv/bin/python scripts/hash_order_determinism_oracle.py --phase census`
exits 0 with `HASH ORDER CENSUS: PASS` after every negative mutation is observed.

### Phase 1 - specified open sites

**Prerequisite:** [#1277] has amended `spec/04` with the canonical coupled
settlement order.

**Deliver:** [#1338]'s two deferred-constraint sites use a private store implementing
that exact order. Add compile-fail raw-access coverage, insertion-order perturbations,
and stable `check`/`eval`/`build` results for the reproducer.

**Explicitly excluded:** consumer-selection totality such as [#1265]. This phase
guarantees one specified order, not equal results under arbitrary orders.

**Oracle:** `.venv/bin/python scripts/hash_order_determinism_oracle.py --phase open-sites`
exits 0 with `HASH ORDER OPEN SITES: PASS`.

### Phase 2 - all behavior and serialization carriers

**Deliver:** every `ordered-store` and `canonical-boundary` census row, the complete
cache payload graph, cache format bumps required by changed bytes, and exact payload/
envelope tests. No behavior-reaching raw hash iterator or derived unordered cache
carrier remains.

**Frozen at exit:** private store APIs and canonical serialization adapters.

**Oracle:** `.venv/bin/python scripts/hash_order_determinism_oracle.py --phase carriers`
exits 0 with `HASH ORDER CARRIERS: PASS`.

### Phase 3 - continuous ratchet and full public oracle

**Deliver:** deny the lint throughout the derived universe, map every allow to the
census, wire the full runner into the gate, and retain all public-surface fresh-process
representatives.

**Oracle:** the complete C6 command exits 0 with
`HASH ORDER DETERMINISM ORACLE: PASS`, and the full controlled-mutation suite turns it
red.

## Part III: interlocks and non-goals

- **[#1277] / [#1338]:** #1277 owns which deferred `expand` result is selected;
  this plan owns that every compiler process follows that one specified order. Only
  Phase 0 may land before the `spec/04` amendment. Phase 1 and later cannot claim
  their oracles while the semantic order is unsettled.
- **[#730]:** census and mutation discipline precedent only.
- **[#731]:** private/unconstructible boundary precedent only.
- **Non-goals:** arbitrary-order-independent deferred resolution; language-level
  nondeterminism; floating-point reassociation; packaging/benchmark reproducibility;
  and parallel scheduling. A future parallel compiler phase must extend C1 and its
  row evidence in the same change.

[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#731]: https://github.com/Chelis-Lang/chelis/issues/731
[#823]: https://github.com/Chelis-Lang/chelis/issues/823
[#1198]: https://github.com/Chelis-Lang/chelis/issues/1198
[#1265]: https://github.com/Chelis-Lang/chelis/issues/1265
[#1277]: https://github.com/Chelis-Lang/chelis/issues/1277
[#1338]: https://github.com/Chelis-Lang/chelis/issues/1338
[#1341]: https://github.com/Chelis-Lang/chelis/issues/1341
