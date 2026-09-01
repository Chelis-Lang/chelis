# Hash-Order Determinism: observable behavior never depends on hash iteration order

**Status:** ACTIVE. Phase A is implemented; Phase B remains proposed. Tracking
issue: [#1341].
Amended 2026-08-28: the mechanism moved from the census enumerator that PR #1366
merged to a type-level ban, then was cut to what defends a known instance or a spec
sentence; see § Rationale and alternatives set aside. Phase A was implemented and
its named oracle passed on 2026-08-31. Function names are the durable anchors.
**Owning specs:** `spec/00-context.md` §5 states the rule this plan implements:
for fixed program text, compiler build, target, and declared inputs, every check,
evaluation, and build result is a function of those inputs. `spec/04-type-system.md`
§4.7.2 decides the coupled `expand` default settlement order that [#1338] exposed
and, with §4.7.3, that a `reshape` result is shape-bearing from its shape list.
This plan changes no other language semantics. When an iteration order selects a
user-visible language result, its canonical order must first be decided by the
owning numbered spec.
**Class fixed:** [#1341] - `std::collections::HashMap` and `HashSet` seed their
hashers per process, so any order that reaches a checker verdict, diagnostic,
dispatch choice, evaluation result, emitted artifact, serialized cache payload,
or ABI-visible sequence makes that output vary across identical runs.
**Precedents:** [`checker_totality.md`](checker_totality.md) ([#731]) supplies
the private, structurally unconstructible boundary pattern;
`rustc_data_structures::unord` (`UnordMap` and `UnordSet` in the Rust compiler)
supplies the order-free map API.

## Summary

Six source-order instances of one defect are confirmed across four subsystems, three
of them verdict- or dispatch-affecting, and exact-head review confirmed a seventh in
the whole-package cache graph. Each resolved instance was found from a downstream
symptom, repaired by a local sort, and pinned by a test that could not see the next
site. The invariant is still maintained by comments and vigilance.

The class is closed by construction: every instance is an iteration over one of two
named standard-library types. Closure therefore lands at the compile-error rung:

1. ban `std::collections::HashMap` and `HashSet` in every workspace target through
   `clippy::disallowed_types`, and provide an order-free workspace map and set whose
   API cannot expose iteration order; use `BTreeMap`/`BTreeSet` where iteration is
   needed and `Ord` is the canonical order;
2. require one authority-backed canonical order wherever order reaches behavior;
3. keep every cache root byte-locked by an exact-byte fresh-process test; and
4. lock the ban with a configuration-blind token tripwire whose allow-list is exact.

Fresh-process repetition is regression coverage, not proof of completeness, and
carries no false-pass probability claim.

## The instance record

| site | observable effect | status |
|---|---|---|
| `Subst::materialize_deferred_expand_defaults` (`unify.rs`; formerly `.keys()` over `HashMap<TypeVar, _>`) | identical source was accepted or rejected across processes; the original 24-run sample split 17/7, and a later review split 15/9 | resolved in Phase A by source-ordinal settlement |
| deferred reshape obligations (`unify.rs`; formerly an `iter_mut()` walk over the same map shape) | coupled deferred-reshape resolution order was hash order; no reproducer isolated this second site | resolved in Phase A by reshape-source ordering and immediate result-shape publication |
| linearity capture walk (`linearity.rs`, `free_vars`) | one program alternated between success and `UseAfterConsume` | resolved in the chelis#1200 fix by a semantically required sort |
| ADT variant dispatch (`adt.rs`, `lookup_variant_preferring_shape`) | hash order selected the first matching variant | resolved by sorting candidates by ADT name |
| input-validation preamble (C and HIP emitters) | validation-block order changed emitted bytes | resolved at `17a28b9`; regression in `codegen_determinism.rs` |
| Load pre-creation (`chelis-ir`, `lower_subexpr_program_inner`) | input-slot order and emitted C bytes varied after lowering | resolved in the chelis#469 wave; CLI regression in `rank_poly_tier3` |
| `CompiledContext::{encode, save}` and CLI worker handoff | identical fixed-source package contexts produced eight distinct encoded digests in eight fresh processes | OPEN - Phase B |

The escalation is the class evidence: presentation and persisted bytes, ABI slot
assignment, dispatch and linearity verdicts, and now the type checker's accept/reject
result.

**Not in this class:** deliberate language-level nondeterminism semantics such as
duplicate-index `scatter` rejection; floating-point reduction-order semantics;
packaging reproducibility ([#1198]); benchmark reproducibility ([#823]); future
parallel scheduling; and the degraded build-fingerprint nonce in
`chelis-compiler-api/src/lib.rs` (`fingerprint_string`), which deliberately draws
process-random bytes from `RandomState` so that two degraded processes never share a
cache entry and is documented at its site. Each has a different authority and
oracle.

## Part I: contracts

### C1 Observable determinism

For fixed source bytes, compiler build, target, flags, and declared inputs, every
observable result of `chelis check`, `eval`, `build`, and compiler library entry
points is invariant under process hash seeds. Observable results include:

- verdict and the complete ordered diagnostic list;
- evaluated values and traps;
- emitted source, object, and artifact bytes;
- persisted payload and envelope bytes;
- dispatch choices and ABI-visible ordering.

**Scope of the mechanism.** Workspace code, its four build scripts and every test
included, is covered by construction: the ban and the tripwire in C2 leave no raw
hash carrier for an order to escape from. Third-party runtime code is not scanned
and is outside this claim; it is covered only by the fresh-process stability tests
named in C5, and an instance those expose is fixed by a canonical boundary at the
workspace call site and added to the tripwire's allow-list with its reason. External
procedural-macro and build-script output is a property of the pinned `Cargo.lock`
and belongs to [#1198].

### C2 The ban, the wrapper, and the tripwire

#### C2.1 The ban

A root `clippy.toml` lists `std::collections::HashMap` and
`std::collections::HashSet` under `disallowed-types`, each with a `reason` naming
this document and a `replacement` naming the workspace type. The lint is in
Clippy's `style` group, so the gate's existing
`cargo clippy --workspace --all-targets -- -D warnings` (`scripts/gate.py`,
`STAGES`) promotes it to an error with no `[workspace.lints]` change. The lint
resolves definition paths, so `hash_map::HashMap`, `use ... as` renames, glob
imports, a `HashMap::new()` receiver, and the definition site of a `type` alias are
all caught; Phase B proves each spelling with a compile-fail fixture rather than by
citation. The ban applies to every target Clippy compiles, tests included: a test
whose expected value passes through a hash walk is itself a coin flip, and a retried
flake is the CI-hiding failure mode [#1341] records. Exactly one production
`#[allow(clippy::disallowed_types)]` exists, on the wrapper's private field, where
`clippy::iter_over_hash_type` is also denied. The perturbation fixtures of C3.2
need raw maps to construct fresh hash states; each carries a narrow allow and an
allow-list entry.

#### C2.2 The order-free workspace map and set

The wrapper lives in a new leaf crate, `chelis-unord`, that depends on `std` and
`serde` only. `chelis-vocab` cannot host it: that crate is `#![no_std]`,
dependency-free, and purity-locked by `crates/chelis-vocab/tests/crate_purity.rs`.
Every crate whose sources carry a hash map (types, ir, the C, HIP, and Metal
backends, compiler-api, cli, effects, deep, surf, prove, lint, pipeline-core, lsp,
validate, reef, macros, e2e) gains one direct dependency edge in Phase B, and
`scripts/pipeline_core_dependency_guard.py` adds the crate to
`APPROVED_DIRECT_DEPENDENCIES` in the same change. A test-only hit
(`chelis-tide/tests/mcp.rs`) migrates to `BTreeMap` without the edge.

The API is hashed lookup and mutation (`insert`, `get`, `get_mut`, `remove`,
`contains_key`, `len`, `is_empty`, `entry`, `extend`, `FromIterator`),
order-insensitive queries (`all`, `any`, `count`, `map_values`, `merge`), one
ordered exit `to_sorted_by_key` whose key must be injective over the contents (a
tie would fall back to hash order, so a tying key panics in debug builds), and
`Serialize` as the key-sorted sequence with a `Deserialize` that rejects a duplicate
key, so a wrapper field under derived Serde is canonical without an adapter. There
is no `iter`, `keys`, `values`, `drain`, `IntoIterator`, or `Deref` to the inner
map, and a test pins the public method set so an order-exposing method cannot be
added without changing this document.

Most hash-map uses in the workspace are lookup-only and move to the wrapper
unchanged; the rest iterate and move to `BTreeMap` when their key is `Ord` and the
canonical order, or to `to_sorted_by_key` with the numeric payload of `TypeVar`,
`DimVar`, `RankVar`, or `NodeId`, which C3.1 classifies as byte or cache order,
never semantic authority. Hot lookup paths (`Subst::apply` and `apply_dim`,
`Env::bindings`, the `lower.rs` symbol tables, `dag.rs` remaps) stay hashed.

#### C2.3 The token tripwire

Clippy sees only the configuration it compiles: a raw map behind a non-default
feature or a platform `cfg` is invisible to it, and such carriers exist today
(`chelis-prove/src/tier_b.rs`, `carcara_audit.rs`, and `z3_engine.rs` behind
`smt`, `carcara`, and `z3`). The completeness lock is therefore a text scan.

`scripts/hash_order_determinism_oracle.py` scans every tracked `.rs` file in the
repository (`git ls-files`), which is a superset of every Cargo target of every
kind, every build script, and every path outside `src/`, and is blind to the
feature or `cfg` that gates a file, a target, or a function. It matches the source
spellings `HashMap`, `HashSet`, `hash_map::`, `hash_set::`, `FxHashMap`,
`FxHashSet`, `rustc_hash::`, `fxhash::`, `ahash::`, `hashbrown::`, and
`indexmap::` outside the wrapper module, and every `#[allow(clippy::disallowed_types)]`
site. Every hit must appear in the script's allow-list, a constant of
(path, item, reason) entries, and every entry must still hit, so a stale entry and an
unlisted hit fail alike. The script's `unittest` twin plants a raw map in a library
target, behind a non-default feature (`checkpoint-compile-probe`), in a `build.rs`,
in a test target, and as a new `#[allow]` without an entry, and asserts each is
rejected. Migrated code behind a feature compiles in the workflows that already
build those features (`smt-full-prove.yml`; the `generalize-sweep-oracle` job in
`ci.yml`); this plan cites that evidence and adds no per-PR feature-matrix Clippy
run.

### C3 Canonical order and private stores

#### C3.1 Authority before mechanism

A behavior-reaching order is not chosen merely because `Ord` exists. Its authority
is one of: the owning numbered-spec rule, for language-result selection; this
implementation contract with a stated key order (UTF-8 byte order, `NodeId` numeric
order), for byte and presentation order; or C4, for cache bytes.

For [#1338], `spec/04-type-system.md` §4.7.2 decides the order: deferred results
settle in the source order of their `expand` expressions in the canonical Deep
program, a result carrying several obligations takes the position of its earliest
`expand` and settles to the first candidate satisfying all of them (obligations in
source order, same-rank before insertion), and the element-count relations of the
`reshape` expressions that consume a settled result resolve in `reshape` source
order before the next result settles. `TypeVar(u32)` allocation order is not that
order: inference visits definitions in dependency order (`infer/declarations.rs`
Tarjan SCCs), so the implementation records a source ordinal on each obligation: the
pre-order index of its `expand` in the program, or, for an obligation carried out of
a serialized library context, the index of its instantiation site in the downstream
program, assigned at first reference and placed after every program ordinal when
never referenced. `bind_tvar`'s obligation `extend` becomes an ordered merge.

§4.7.2 also states, with §4.7.3, that a `reshape` result is shape-bearing for its
consumers from its shape list alone while only its input's form stays open. That
sentence decides the [#1338] family:
`sub(reshape(expand(a, 0, 6i64), [3i64, 2i64]), expand(b, 0, 3i64))` and its
mirrored operand order both accept, because the `reshape` result is `tensor[3, 2]`
under either form of `a`, so `sub` fixes `b` by rank before any default fires and
`a` defaults to `[6]` at the freeze point. Today's
`resolve_deferred_expand_for_reshape` instead keeps the `reshape` result as a type
variable carrying one output per input candidate until the input settles, even
when every candidate agrees; in the mirror, `bind_tvar` then aliases `b`'s
obligations onto that variable and `materialize_deferred_expand_default` selects
`[3]` from the `expand` obligations alone and rejects at the first `reshape`
requirement. That deferral, not the settlement order, is what makes the mirror
reject in some runs, and Phase A removes it. §4.7.2 further states that a `shape`
read of an open result, inside a `reshape` list or elsewhere, requires the tensor
type and selects the replacement form, so `reshape(e, [shape(e, 1), 6i64])` over
`e = expand(to_tensor([0.0f32]), 0, 6i64)` is a type error (axis 1 on rank 1)
rather than today's per-form acceptance as `tensor[1, 6]`; Phase A adds that
program as a rejecting row. For an aliased pair such as
`add(expand(x, 0, 3i64), expand(y, 0, 3i64))` with `x: tensor[2]` and
`y: tensor[3, 2]`, today's unification-direction producer order and the rule both
select `[3, 2]`; only the first-rejection message moves for rejecting pairs. The
settlement order still decides which named extents and diagnostics a program
carries, which C1 counts as observable; no program is known whose verdict changes
with the order once a `reshape` publishes its shape. Order-independent resolution
beyond what §4.7.2 states is the totality question that [#1277] and [#1265] own and
remains a non-goal here.

#### C3.2 Ordered representation

The wrapper is the workspace default; `BTreeMap`/`BTreeSet` when a consumer needs
iteration and their `Ord` is the canonical order; otherwise a private newtype with
a canonical iterator or a boundary sort. An insertion-ordered map is allowed only
when the owning authority makes insertion order canonical. The two deferred
constraint stores in `Subst` are newtyped with the recorded ordinal and a canonical
iterator; raw access is private, so a new unordered walk fails at compile time.
Tests construct equal contents through canonical, reverse, and fixed shuffled
insertion orders and through fresh hash states, then assert the same canonical
iterator and observable result. They never reverse the canonical settlement order:
that would test the stronger, out-of-scope property of order-independent resolution.

### C4 Cache bytes

Persisted, hashed, returned, and cross-process handoff bytes are part of C1. With
the ban in place, no derived `Serialize` reaches an unordered carrier, so canonical
bytes follow from the types; one exact-byte test per root keeps that executable.
The roots are `cache_envelope::save<T>` payload and envelope bytes for
`LibraryContext` and `StdLibContext`, `CompiledContext::encode` public bytes,
`CompiledContext::save` payload and envelope, the CLI worker handoff that writes
encoded `CompiledContext` bytes, and the serialized inputs to the stdlib, library,
compiled-context, and lowering cache keys. Each test builds a complete payload
through several insertion orders and serializes it in 24 fresh processes; the bytes
and digests must match exactly. At reviewed head `5b2dfd14`, eight fixed-source
`CompiledContext::encode` probes produced eight distinct SHA-256 digests; that probe
is the regression. The relevant cache format version is bumped whenever canonical
bytes change under the existing cache exactness policy. This track owns
deterministic compiler-cache bytes; [#1198] may consume that result for archive
reproducibility but is not the authority or a prerequisite here.

### C5 Oracles

Each phase names one command. Phase A's Python runner composes the compile-fail,
source-order perturbation, cache-version, executable-example, executable eval/C
parity, reshape-regression, and fresh-process CLI suites; the raw-store compile-fail
leg also runs in the gate's
`lint-and-unit` stage and the `--local` subset, whose membership
`scripts/test_gate.py` locks. Phase B's tripwire script and named cache-byte tests
remain proposed. The fresh-process stability tests cover the [#1338] reproducer, its
mirrored operand order, the aliased pair, and the three-way case of Phase A, and the
existing byte-determinism regressions (`codegen_determinism.rs`,
`rank_poly_tier3::form3_bias_broadcast_c_is_byte_deterministic`) stay present and
green. Repeated fresh-process runs have a fixed budget of 24 and make no statistical
confidence or false-pass claim.

## Part II: phases

### Phase A - the specified open sites - IMPLEMENTED

**Prerequisite:** none. The settlement order and the `reshape` rule are decided in
`spec/04` §4.7.2. This phase may land before Phase B.

**Deliver:** the two deferred-constraint stores in `Subst` newtyped over a recorded
source ordinal exactly as §4.7.2 states; a `reshape` result that publishes the shape
§4.7.3 assigns it at once, with its input's element-count relation the only pending
obligation; compile-fail raw-access coverage; insertion-order perturbations;
stability rows, all accepting in 24 fresh processes, for the reproducer, its
mirrored operand order, `add(expand(x, 0, 3i64), expand(y, 0, 3i64))`, and
`sub(reshape(expand(a, 0, 6i64), [3i64, 2i64]), add(expand(x, 0, 3i64), expand(y, 0, 3i64)))`
in both operand orders, and a rejecting row for `reshape(e, [shape(e, 1), 6i64])`;
and a freeze-point rejection diagnostic that names the settlement rule and
suggests a declared result shape. Because the ordered obligation fields change the
serialized `TypeEnv` shape, the compiled-context, stdlib, and library cache format
versions advance in the same phase; Phase B still owns canonical cache bytes.

**Explicitly excluded:** consumer-selection totality such as [#1265]. This phase
guarantees one specified order, not equal results under arbitrary orders.

**Oracle:** `.venv/bin/python scripts/hash_order_phase_a_oracle.py` exits 0 with
the final line `HASH ORDER PHASE A ORACLE: PASS`. It passed on 2026-08-31.

### Phase B - class closure

**Deliver:** `clippy.toml`; `chelis-unord` with its API pin, compile-fail spelling
fixtures, and Serde tests; migration of production and test code off the std
types, with the dependency edges and the dependency-guard constant; the tripwire
script, its `unittest` twin, and gate wiring; and the C4 exact-byte tests with the
cache format bumps they require.

**Frozen at exit:** the wrapper's public API and the tripwire's token list.

**Oracle:** `.venv/bin/python scripts/hash_order_determinism_oracle.py` exits 0
with the final line `HASH ORDER DETERMINISM ORACLE: PASS`, and the C4 tests are
green.

## Part III: interlocks and non-goals

- **[#1277] / [#1338] / [#1343]:** `spec/04` §4.7.2 names which deferred `expand`
  result settles first and that a `reshape` result is shape-bearing from its list;
  this plan owns that every compiler process follows that one order; [#1343],
  merged as `2919cf89`, owns the resulting extent verdict rows in
  `runtime_extents.md`. Rebasing this document's pull request onto that merge
  folded the two §4.7.2 paragraphs into one that keeps [#1343]'s sentence and its
  required-literal anchor (`their defaults settle in source order`, the
  `positional expand settlement order` contract) together with this document's
  definitions of the freeze point, the merged-obligation rule, the `reshape`
  publication rule, and the `shape`-read rule, and recomputes the shared
  frozen-file digest. `runtime_extents.md` C1.4, its Freeze action, and its Slice C
  key the deferred stores by source-introduction position and cite this document
  for the ordered-store mechanism, which matches C3.1; its deferral-stability row 6
  depends on the `reshape` publication rule.
- **[#731]:** private/unconstructible boundary precedent only.
- **[#1198]:** owns build-script and external expansion reproducibility.
- **Non-goals:** arbitrary-order-independent deferred resolution; language-level
  nondeterminism; floating-point reassociation; packaging/benchmark reproducibility;
  and parallel scheduling. A future parallel compiler phase must extend C1 and its
  evidence in the same change.

## Rationale and alternatives set aside

This document's first merged form (`f1a6d162`, PR #1366) closed the class through a
census: a post-expansion, type-resolved enumerator over every Cargo target and every
feature, `cfg`, profile, and host configuration; a traced sandbox for build scripts
and procedural macros; 24 re-executions of every procedural-macro invocation per
configuration fixture; and a bijection between enumerated sites and census rows.
Seven exact-head review rounds are recorded on that pull request; the first six
each closed a completeness gap by adding a piece of that machinery, and the seventh
passed. That design was set aside because a type-resolved enumerator and isolated
procedural-macro workers need `rustc_private`, which the pinned stable toolchain
(`rust-toolchain.toml`) does not expose; because its oracle required sandbox-proved
evidence from every supported host and had no shown path to green on macOS;
because a bijection keyed on crate, target, host, profile, feature assignment,
module, item, field, and consumer turns every rename in `chelis-types` into a
census edit; and because the class is closed by construction, while the workspace
has no procedural-macro crate and no build script that touches a hash map, so the
procedural-macro and sandbox legs defended hypotheticals with no instance.

The first revision of this amendment (PR #1370, heads `23fd333e` through
`f5d21709`) kept the census template around the ban: a census file with identity,
disposition, and evidence rows; a dependency-graph leg with rows for transitive
`hashbrown`, `indexmap`, `foldhash`, and `dashmap` versions; a `RandomState` token
and a disposition for the fingerprint nonce; seventeen enumerated mutations; a
six-leg phase oracle; four phases; one fresh-process representative per C1 output
class; and row bookkeeping for manual `Serialize` implementations. Each piece was
added in answer to a review finding that named no instance. The revision at this
head keeps only what defends a known instance or a spec sentence, and the rule for
future review is the same: a coverage finding is valid where this document claims
coverage, and a proposed addition names the instance it defends or is recorded here
as out of scope.

Keeping `HashMap` with a fixed hasher (`FxHash`) was also set aside: it removes the
cross-process flake but leaves an arbitrary order that changes with insertion
history and table growth, so a verdict-affecting site would still settle by accident
and C3.1 would still apply.

The census design remains the record of what a complete site enumeration would
require. It is revived, as an extension of this document filed under [#1341], only
by a confirmed instance the current mechanism cannot see: hash order inside a
compile-time executor, or inside third-party runtime code that C5's stability tests
expose.

[#731]: https://github.com/Chelis-Lang/chelis/issues/731
[#823]: https://github.com/Chelis-Lang/chelis/issues/823
[#1198]: https://github.com/Chelis-Lang/chelis/issues/1198
[#1265]: https://github.com/Chelis-Lang/chelis/issues/1265
[#1277]: https://github.com/Chelis-Lang/chelis/issues/1277
[#1338]: https://github.com/Chelis-Lang/chelis/issues/1338
[#1341]: https://github.com/Chelis-Lang/chelis/issues/1341
[#1343]: https://github.com/Chelis-Lang/chelis/pull/1343
