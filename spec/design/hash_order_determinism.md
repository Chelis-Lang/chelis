# Hash-Order Determinism: observable behavior never depends on hash iteration order

**Status:** Phase A is implemented. Phase B is implemented and its named oracle
passes; PR #1444 merged it as `bcde1133` and [#1341] closed.
Tracking issue: [#1341].
Amended 2026-08-28: the mechanism moved from the census enumerator that PR #1366
merged to a type-level ban, then was cut to what defends a known instance or a spec
sentence. Amended 2026-09-01: Phase B's completeness leg moved from a
configuration-blind source census to compiler-reported configuration closure, and
the wrapper's storage from a hand-rolled arena hash table to an ordered
collection; see § Rationale and alternatives set aside. Phase A's named oracle
passed on 2026-08-31. Function names are the durable anchors.
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
4. bound the configuration space with a compile error, compile all of it with a
   registered Clippy matrix, and reconcile every source against rustc's dep-info.

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
| `CompiledContext::{encode, save}` and CLI worker handoff | identical fixed-source package contexts produced eight distinct encoded digests in eight fresh processes | resolved in Phase B by canonical collection serialization and a 24-process exact-byte oracle |

The escalation is the class evidence: presentation and persisted bytes, ABI slot
assignment, dispatch and linearity verdicts, and now the type checker's accept/reject
result.

**Not in this class:** deliberate language-level nondeterminism semantics such as
duplicate-index `scatter` rejection; floating-point reduction-order semantics;
packaging reproducibility ([#1198]); benchmark reproducibility ([#823]); future
parallel scheduling; the order in which a container's internal allocations are
released, which a process-global allocator can observe but which reaches none of
C1's outputs because nothing in the compiler orders by address; and the degraded build-fingerprint nonce in
`chelis-image-id/src/lib.rs` (`fingerprint_string`), which deliberately draws
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
included, is covered by construction: C2's ban is enforced by the compiler over a
configuration space that is declared, compiled, and reconciled, so no raw hash
carrier is left for an order to escape from. Third-party runtime code is not scanned
and is outside this claim; it is covered only by the fresh-process stability tests
named in C5, and an instance those expose is fixed by a canonical boundary at the
workspace call site. External
procedural-macro and build-script output is a property of the pinned `Cargo.lock`
and belongs to [#1198].

### C2 The ban, the wrapper, and the configuration closure

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
flake is the CI-hiding failure mode [#1341] records. No production `#[allow(clippy::disallowed_types)]` exists: with ordered storage the
wrapper needs no exemption, so the ban covers the workspace without a hole. The disallowed-type compile-fail
fixture intentionally names raw maps; it is a standalone Cargo project outside the
workspace build, recorded in `UNCOMPILED_EXCEPTIONS` with the gate that compiles
it, and it carries no lint allowance because rejection is the test.

#### C2.2 The order-free workspace map and set

The wrapper lives in a new leaf crate, `chelis-unord`, that depends on `std` and
`serde` only. `chelis-vocab` cannot host it: that crate is `#![no_std]`,
dependency-free, and purity-locked by `crates/chelis-vocab/tests/crate_purity.rs`.
Every crate whose sources carry a hash map (types, ir, the C, HIP, and Metal
backends, compiler-api, cli, effects, surf, prove, lint, pipeline-core, lsp,
validate, reef, macros, e2e, and runtime behind its `ownership-ledger` feature:
eighteen) gains one direct dependency edge in Phase B, and
`scripts/pipeline_core_dependency_guard.py` adds the crate to
`APPROVED_DIRECT_DEPENDENCIES` in the same change. A test-only hit
(`chelis-deep/src/tag.rs` and `chelis-tide/tests/mcp.rs`) migrates to an ordered
collection without the edge.

The API is keyed lookup and mutation (`new`, `insert`, `get`, `get_key_value`,
`get_mut`, `remove`, `contains_key`, `len`, `is_empty`, `entry`, `clear`,
`extend`, `Default`, `Index`, `From<[(K, V); N]>`, `FromIterator`, and `merge`),
the `Entry`/`OccupiedEntry`/`VacantEntry` trio, plus borrowed and consuming
forms of one no-callback ordered exit (`to_sorted`, `into_sorted`). A key used in lookup or
mutation has `Ord`: its owning authority defines that order as canonical
identity order, never as semantic priority. The wrapper's
storage is a private `BTreeMap`/`BTreeSet`. Every key already carries `Ord`
because the ordered exits require it, so the structure has exactly one order and
there is no randomized one to leak: `Clone`, `PartialEq`, the ordered exits,
`Serialize`, `clear`, and destruction all traverse key order because that is the
only order available, and no wrapper-side index has to be maintained to make that
true. What the type buys over a bare `BTreeMap` is the boundary, not the storage:
there is no callback-bearing query, mutation, or projection exit, because even an
apparently read-only closure can observe visitation order through interior
mutability, logging, or atomics, and a consumer that wants an order must spell
`to_sorted` or `into_sorted` and thereby claim C3.1 authority for key order.
`Debug` reports only the wrapper kind and length.
`Serialize` emits a key-sorted sequence and `Deserialize` rejects a duplicate
key, so a wrapper field under derived Serde is canonical without an adapter.
There is no `iter`, `keys`, `values`, `drain`, `IntoIterator`, or `Deref` to the
inner collection; tests pin the public inherent and trait sets, including derived
traits, assert that the private storage is the ordered collection, and assert that
the crate claims no `disallowed_types` allowance of its own.

Most former hash-map uses in the workspace are lookup-only and move to the
wrapper unchanged; the rest iterate and move to `BTreeMap` when their key is `Ord` and
the canonical order, or materialize the wrapper's key-sorted exit before any
application callback runs. Numeric identity keys such as `TypeVar`, `DimVar`,
`RankVar`, and `NodeId` implement `Ord` by their payload; `Prim` uses its
canonical dtype spelling and `LoadStoreName` its validated string payload. C3.1
classifies each as byte or cache order, never semantic authority. A consumer that needs value
order first materializes key order, then sorts the resulting ordinary vector.
Lookup is therefore `O(log n)` over cheap key comparisons rather than `O(1)`; the
complexity gate is `scripts/compiler_front_end_performance.py`, which counts
deterministic work rather than wall time. Restoring a hash index for a measured
hot path is a private change inside the crate, not an API move.

#### C2.3 Configuration closure

Clippy sees only the configuration it compiles: a raw map behind a non-default
feature or a platform `cfg` is invisible to it, and such carriers exist today
(`chelis-prove/src/tier_b.rs`, `carcara_audit.rs`, and `z3_engine.rs` behind
`smt`, `carcara`, and `z3`). Closure therefore bounds the configuration space
and compiles all of it. It does not search source text for banned spellings in
configurations nobody builds; see § Rationale for why that alternative was set
aside.

**The space is declared.** `[workspace.lints.rust] unexpected_cfgs = "deny"`,
inherited through every member's `[lints]` table, makes a `cfg` name that is
neither a declared Cargo feature nor a well-known rustc cfg a compile error
rather than a warning. Beyond features, workspace source uses only `test`, `unix`, `target_os`, and
`debug_assertions`, and declares no module gated on one of those, so the space
is exactly declared features times supported hosts: finite and enumerable.
(Feature-gated module declarations do exist, in `chelis-prove` and
`chelis-cli`; the matrix compiles them, and `z3_engine.rs` is one of the three
nightly-only sources named below.)

**The matrix compiles it.** `scripts/check_configuration_closure.py` holds
`CLIPPY_MATRIX`, a constant whose rows each carry a command, the file that must
issue it, the hosts it runs on, and its cadence:

| row | compiles | owner | cadence |
|---|---|---|---|
| `default-features` | default features | `scripts/gate.py` | per pull request |
| `solver-free-features` | `sleef`, `hip-local-gpu`, `clarabel`, `extension-module`, `ownership-ledger`, and the two `chelis-types` probe features | `scripts/gate.py` | per pull request |
| `no-default-features` | every default feature in its off-state | `scripts/gate.py` | per pull request |
| `core-without-migration` | Deep and Surf without the CLI-enabled `pre-020-pipe-migration` reader | `scripts/gate.py` | per pull request |
| `cvc5-features` | `smt` for `chelis-cli`, `chelis-prove`, `chelis-tide` | `ci.yml`'s `SMT Feature Build (Linux)`, which already provisions cvc5 | per pull request |
| `all-features` | every declared feature at once | `smt-full-prove.yml`, the only runner that provisions every solver | nightly |

**Both states of every feature are compiled, and cargo decides what those are.**
`#[cfg(feature = "f")]` is linted only by a row that enables `f`, and
`#[cfg(not(feature = "f"))]` only by a row that leaves it off, so an additive
matrix never lints the off-state of a default feature however many rows it has;
`--all-features` makes that worse rather than better, because it enables
everything at once. The check therefore requires each declared feature to be
enabled by some row and disabled by some row.

Which features exist, and which a workspace row actually enables, both come from
`cargo metadata` rather than from the manifests or the command strings.
Package-restricted rows use `cargo tree` with the same package and feature
selection, including normal, build and development dependencies. Only a package
present in that resolved graph can supply off-state coverage; omitting a package
does not compile its disabled configuration. The core row is required because
the CLI enables the migration reader through its Surf dependency even when
workspace default features are disabled. An
optional dependency creates an implicit feature of the same name that the
`[features]` table never lists, and `chelis-cli`'s optional `chelis-prove`
dependency is exactly that shape and is a *default* feature guarding 25
`#[cfg(not(feature = "chelis-prove"))]` regions. A feature can also be turned on
transitively by another, as `smt` turns on `cvc5-rs`. Reading either fact off
the source would repeat the mistake C2.3 exists to correct: ask the tool that
resolves it.

The check also fails if a command drifts from the file that owns it. For
`scripts/gate.py` the owner check reads the gate's own `--list` rendering rather
than grepping its source, so a constant that is defined but reaches no stage
does not count as coverage.

The ban itself is Clippy's `disallowed_types`, applied to real HIR. Aliases,
glob imports, a `type` alias definition site, macro-expanded code, and
`include!`-ed `OUT_DIR` code are therefore the compiler's problem rather than a
scanner's, and each is proved by a compile-fail fixture in Phase B rather than
by citation.

**Nothing is left out.** After the matrix runs, the same script reconciles every
repository `.rs` file (`git ls-files --cached --others --exclude-standard`)
against the prerequisite lists rustc itself wrote to `target/<profile>/deps/*.d`.
A file that no registered configuration compiled fails the gate by name. The
only accepted absences are `UNCOMPILED_EXCEPTIONS`: the standalone compile-fail
fixture projects, each naming the gate that compiles it, each verified to exist
and to hold a Cargo project so a stale entry cannot survive.

Dep-info accumulates, and every cargo invocation writes it, not only a
registered row. A developer who has once built an unregistered configuration in
that worktree therefore has dep-info for it, and leg 3 would count those files
as covered. The accumulated union is therefore supporting completeness evidence:
it can overstate coverage, and it carries no provenance that can prove a
nightly-only entry stale. The gate therefore makes no automatic stale inference
from it. Reviewers own the source-to-feature attribution when the exact residual
changes; the nightly `--require-complete` run remains the executable proof that
the named sources are compiled by the nightly matrix.

**Executable controls.** `scripts/test_check_configuration_closure.py` proves
the two properties the rest rests on, with rustc rather than assertion:
an undeclared `cfg` name does not compile under the workspace lint config, while
a declared feature `cfg` still does; and a `.gitignore`d `#[path]` target that
rustc compiles appears in its dep-info and so in the reconciled set. The
remaining tests cover dep-info parsing, a member that fails to inherit the lint,
a workspace that only warns, an uncovered feature, an invented feature, an
unqualified feature spelling, a run its owner does not issue, an uncompiled
source, an empty dep-info set, accumulated provenance that cannot prune a
nightly-only source, and a stale or ungated exception.

**Residual: the host dimension.** Both registered hosts are unix, so
`#[cfg(not(unix))]` is compiled by no row at any cadence. Eighteen such sites
exist, seventeen of them in `src/`, including production code in `chelis-cli`,
`chelis-reef`, `chelis-compiler-api`, `chelis-image-id`, `chelis-python`,
`chelis-conformance`, and `chelisup`. Windows is not a supported host: no
workflow uses a Windows runner and `release.yml` builds only Linux and macOS,
so those regions ship nowhere and the class cannot reach a user through them.
They are nonetheless outside the ban's enforcement, and this document does not
claim otherwise. Leg 3 does not see the gap either, for the same reason it does
not see a partly feature-gated file: dep-info is file-granular, so a source
whose `#[cfg(not(unix))]` regions rustc never lowered still reads as covered.
The two residuals differ in that the feature one has a nightly `--all-features`
backstop and this one has none at any cadence. Bringing it in requires a
supported Windows host, not a further row.

The Linux per-pull-request rows remain `default-features`,
`solver-free-features`, and `no-default-features`. Separate macOS nightly rows
register the first two configurations in `.github/workflows/macos-nightly.yml`.
Shard 1 of `macos-workspace-shard` installs Clippy and runs both configurations; the
`macos-smoke` aggregate requires that shard to succeed. Attribute-form
`#[cfg(target_os = "macos")]` regions therefore receive these two lint
configurations nightly and on manual dispatch, not on ordinary PRs or main
pushes. `python3 scripts/gate.py --validation` remains an optional
local reproduction. The
`no-default-features` row is Linux-only: `--validation` dropped it and kept the
other two because the closure check's source-reconciliation leg needs the
solver-free row on a fresh target (`crates/chelis-prove/src/clarabel_sos.rs`
is compiled per pull request by that row alone), while the no-default row
compiles a strict subset of the default row. The residual that follows is
narrow but real: an attribute-form `#[cfg(target_os = "macos")]` region that
sits under `#[cfg(not(feature = ...))]` is linted on no host at any cadence.
The `cfg!()` macro form is not in this residual: it compiles both arms on
every host, which is what covers the link-flag and BLAS-provider selection in
`chelis-backend-c/src/toolchain.rs`. Recording the attribute-form gap here is
what the Manual Gates rule requires.

**Residual: the feature dimension.** `z3`, `carcara`, and `arb` need external
solver toolchains
whose per-pull-request cost this repository has already declined, so everything
they gate is linted nightly rather than per pull request. That is three whole
sources plus every region those three features gate inside sources the
per-pull-request matrix does compile: `NIGHTLY_ONLY_SOURCES` names the whole
files (`chelis-prove/src/z3_engine.rs` and the two `certify_*_envelope`
binaries), while leg 3's reconciliation is file-granular and therefore records
a partly-gated file as covered. The nightly `--all-features` row compiles both
kinds, and the matrix records that cadence. An entry whose file is gone is
reported as stale. There is no automatic source-to-feature attribution or
per-pull-request pruning: neither accumulated dep-info nor a row's resolved
feature set proves that the row compiled a particular source. The nightly job
runs the reconciliation with `--require-complete`, which drops the allowance so
a new uncovered file cannot be parked there.
`--all-features` is also where a `#[cfg(all(feature = ..., feature = ...))]`
combination is compiled.

### C3 Canonical order and private stores

#### C3.1 Authority before mechanism

A behavior-reaching order is not chosen merely because `Ord` exists. Its authority
is one of: the owning numbered-spec rule, for language-result selection; this
implementation contract with a stated key order (UTF-8 byte order, `NodeId` numeric
order), for byte and presentation order; or C4, for cache bytes.

For [#1338], `spec/04-type-system.md` §4.7.2 decided the order: deferred results
settled in the source order of their `expand` expressions in the canonical Deep
program, a result carrying several obligations took the position of its earliest
`expand`, and the element-count relations of the `reshape` expressions consuming a
settled result resolved in `reshape` source order before the next result settled.
`TypeVar(u32)` allocation order is not that order, because inference visits
definitions in dependency order (`infer/declaration_graph.rs` Tarjan SCCs), so the
implementation recorded a source ordinal on each obligation and `bind_tvar`'s
obligation `extend` became an ordered merge.

**That whole paragraph is superseded, and with it the defect it describes.**
`spec/04` §4.7.2 now gives `expand` and `insert` one result shape each ([#1532]),
so no result is deferred, there are no obligations to order, and [#1338] is
resolved by construction rather than by ordering: with nothing recorded there is no
order to get wrong. The stores, the ordinals and their index, the alias merge and
the freeze loop are deleted (chelis#1277 S2b and S2c). The reproducer's own
operands moved with the meaning. `expand(to_tensor([0.0f32]), 0, 6i64)` is now a
same-rank broadcast of a unit axis, which is legal and has one answer, while
`expand(to_tensor([1.0f32, 2.0f32]), 0, 3i64)` claims extent 1 at an axis carrying
2 and is a check-time type error; `examples/hash_order_determinism.ch` therefore
spells `insert`, and 12 of 12 `check` runs accept it at score 1.

Order-independent resolution beyond what §4.7.2 states was the totality question
[#1277] and [#1265] owned, and it is answered the same way: one meaning per
primitive leaves nothing to resolve.

#### C3.2 Ordered representation

The wrapper is the workspace default; `BTreeMap`/`BTreeSet` when a consumer needs
iteration and their `Ord` is the canonical order; otherwise a private newtype with
a canonical iterator or a boundary sort. An insertion-ordered map is allowed only
when the owning authority makes insertion order canonical. The two deferred
constraint stores in `Subst` were the worked example of that rule: newtyped with a
recorded ordinal and a canonical iterator, raw access private so a new unordered
walk failed at compile time, and tested through canonical, reverse and fixed
shuffled insertion orders against fresh hash states. Both stores are deleted with
their subject, so the example is historical; the rule it illustrates still binds
every remaining ordered store.

### C4 Cache bytes

Persisted, hashed, returned, and cross-process handoff bytes are part of C1. With
the ban in place, no derived `Serialize` reaches an unordered carrier, so canonical
bytes follow from the types; one exact-byte test per root keeps that executable.

**This section's claim is exactly that**, and it is frozen at that width. Review
of the pull request that implemented it also surfaced cache-*key* defects, where
a key omitted an input and a stale entry could be reused: the stdlib key hashed
parsed declarations rather than exact source bytes, and the prepared-graph key
and envelope carried a bare package version rather than the running build's
identity. Those are stale-hit correctness, not hash order. They belong to the
[#1156] class and were already tracked as [#952] (bundled chelis-std identity,
stale stdlib replay) and [#1249] (prepared-graph key on release version rather
than compiler build); no new tracker is filed. Their repairs landed here, and
are named below, because removing green fixes to re-land them elsewhere would be
churn, not because C4 claims coverage of that class. A coverage finding against C4 is valid where C4
claims coverage: that no unordered carrier reaches a serialized payload, and that
each named root has an exact-byte fresh-process test.
The roots are `cache_envelope::save<T>` payload and envelope bytes for
`LibraryContext` and `StdLibContext`, `CompiledContext::encode` public bytes,
`CompiledContext::save` payload and envelope, the CLI worker handoff that writes
encoded `CompiledContext` bytes, `PreparedReefGraph::encode`, the prepared-graph
payload/envelope, its cache filename and serialized key inputs, its persisted
chelis-std exact-source determinant, and the serialized inputs to the stdlib,
library, and compiled-context cache keys. The stdlib, library, and prepared-graph
input artifacts come from production preimage helpers and are the exact ordered
SHA-256 preimages used by key derivation, not parallel approximations; the test
hashes each artifact back to its recorded final key. The prepared-graph preimage is
domain-separated and length-delimits the exact running-build identity and lossless
canonical package-root bytes. Its envelope carries that same build identity, so two
same-version compiler builds cannot share an entry. The compiled input artifact
includes package name and version as well as source hash and compiler identity. One
closed artifact enum is used by every worker producer and the reader; a worker fails
unless the executed producer set is exactly bijective with that registry. There is
no separate persisted
lowering-cache key; lowered products are covered inside those complete payloads.
The named test builds complete payloads through several filesystem insertion
orders and serializes them in 24 fresh processes; the bytes and digests must match
exactly. The stdlib cache key carries both canonical parsed declarations and the
prepared graph's exact manifest/inventory/source-byte determinant, so a comment-only
edit clean-misses even though the AST bytes do not change. The authoritative oracle
runs the production CLI test that performs that source-only mutation; the synthetic
24-process worker is supporting byte evidence and cannot substitute for production
threading from `PreparedReefGraph` into the stdlib cache call. After rebasing over the
independent nominal-kind formats, Phase B advances the compiled-context cache to
V14, the library cache to V7, the stdlib cache to V11, and the prepared-graph
cache to V5, each strictly above the corresponding constant on `main`, so a
payload written by `main` is a clean version mismatch rather than a misparse.
Version numbers alone do not separate this branch's own pre-rebase head, which
carried the same four constants with a different `TypeEnv` shape; every root
keys on the running compiler's exact build fingerprint rather than its release
version, so two differently built compilers never share a cache filename in the
first place. That is the property doing the work here. At reviewed head
`5b2dfd14`, eight fixed-source
`CompiledContext::encode` probes produced eight distinct SHA-256 digests; that probe
is the regression. The relevant cache format version is bumped whenever canonical
bytes change under the existing cache exactness policy. This track owns
deterministic compiler-cache bytes; [#1198] may consume that result for archive
reproducibility but is not the authority or a prerequisite here.

### C5 Oracles

Each phase names one command. Phase A's Python runner composed the compile-fail,
source-order perturbation, cache-version, executable-example, executable eval/C
parity, reshape-regression, and fresh-process CLI suites. The legs whose subject
was the two-candidate settlement are deleted with it, and the runner keeps the
four that were never about it: the cache-version test, the executable-example and
eval/C parity rows, and `issue_942_inferred_tensor_cast`. The raw-store
compile-fail leg is gone from the gate's `lint-and-unit` stage and the `--validation`
subset with the store it probed; `scripts/test_gate.py` still locks that
membership. Phase B's configuration-closure check and named
cache-byte tests run in its named oracle, and two of its legs also run in the
gate's `lint-and-unit` stage and the `--validation` subset: the closure check,
ordered after the Clippy commands that produce the dep-info it reads, and the
disallowed-type compile-fail fixture. That fixture is the ban's liveness proof.
Nothing else continuous reads `clippy.toml`, and no workspace source spells the
banned types, so deleting the two `disallowed-types` entries would otherwise
leave every job green. Phase A's fresh-process stability tests covered the [#1338]
reproducer, its mirrored operand order, the aliased pair, and the three-way case;
they are deleted with the settlement they exercised. The existing
byte-determinism regressions (`codegen_determinism.rs`,
`rank_poly_tier3::form3_bias_broadcast_c_is_byte_deterministic`) are unrelated to
that model and stay present and green. The Phase B command registry also pins the exact production
`stale_stdlib_byte_mutation_misses_not_stale_hit` test; it uses `cargo test --exact`
because the repository's default nextest filter can select zero tests. Repeated
fresh-process runs have a fixed budget of 24 and make no statistical
confidence or false-pass claim.

## Part II: phases

### Phase A - the specified open sites - IMPLEMENTED, THEN SUPERSEDED WITH ITS SUBJECT

**Prerequisite:** none. The settlement order and the `reshape` rule were decided in
`spec/04` §4.7.2. This phase could land before Phase B.

**Delivered:** the two deferred-constraint stores in `Subst` newtyped over a recorded
source ordinal exactly as §4.7.2 then stated; a `reshape` result that published the
shape §4.7.3 assigns it at once, with its input's element-count relation the only
pending obligation; compile-fail raw-access coverage; insertion-order perturbations;
stability rows, all accepting in 24 fresh processes, for the reproducer, its
mirrored operand order, and the aliased and three-way cases; a rejecting row for
`reshape(e, [shape(e, 1), 6i64])`; and a freeze-point rejection diagnostic naming
the settlement rule. Because the ordered obligation fields changed the serialized
`TypeEnv` shape, the compiled-context, stdlib, and library cache format versions
advanced in the same phase; Phase B still owns canonical cache bytes.

**Superseded.** `spec/04` §4.7.2 now gives `expand` and `insert` one result shape
each ([#1532]), so no result is deferred, no store is populated, and nothing is
iterated. Everything above whose subject was the ordering of that settlement is
deleted (chelis#1277 S2b and S2c): the two stores, the source ordinals and their
whole index, the alias merge, the freeze loop and its diagnostic, the raw-access
compile-fail fixture with its checker script and gate stage, the perturbation and
ordering unit rows, and the K=24 stability target. The cache formats advance once
more for the removal. [#1338] is resolved by construction rather than by ordering:
with nothing recorded there is no order to get wrong, measured as 12 of 12 accepts
at score 1 on `examples/hash_order_determinism.ch`, which now spells `insert`.

**Explicitly excluded:** consumer-selection totality such as [#1265]. This phase
guaranteed one specified order, not equal results under arbitrary orders.

**Oracle:** `.venv/bin/python scripts/hash_order_phase_a_oracle.py` exits 0 with
the final line `HASH ORDER PHASE A ORACLE: PASS`. It passed on 2026-08-31, and
again after the removal reduced it to its four components that were never about
the two-candidate model: the cache-version test, the executable-example and
eval/C parity rows over `hash_order_determinism.ch`, and
`issue_942_inferred_tensor_cast`.

### Phase B - class closure - IMPLEMENTED

**Deliver:** `clippy.toml`; `chelis-unord` over ordered storage, with its
inherent/trait API pin, private-storage assertion, no-self-exemption assertion,
compile-fail spelling fixtures, and Serde tests; migration of production and test
code off the std types, with the dependency edges and the dependency-guard
constant; the workspace `unexpected_cfgs` deny and its per-member inheritance;
`scripts/check_configuration_closure.py` with its Clippy matrix, owner and
feature-coverage checks, dep-info reconciliation, nightly-only inventory, and
`unittest` twin including the two compile-backed controls; the matrix's gate and
workflow wiring; and the C4 exact-byte tests plus the production source-only cache
mutation with the cache format bumps they require.

**Frozen at exit:** the wrapper's public API, `CLIPPY_MATRIX` and its owners, and
`NIGHTLY_ONLY_SOURCES`.

**Oracle:** `.venv/bin/python scripts/hash_order_determinism_oracle.py` exits 0
with the final line `HASH ORDER DETERMINISM ORACLE: PASS`. The script is a command
registry, not an analysis: its first command is
`scripts/check_configuration_closure.py`, because without proven coverage of the
compiled configuration space the remaining evidence is about an unbounded surface.
It must run after the gate's Clippy stages, which produce the dep-info it reads.

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
  frozen-file digest. That §4.7.2 language is itself superseded by [#1532]'s one
  shape per primitive, so `runtime_extents.md`'s Freeze action and its Slice C are
  withdrawn and the ordered-store mechanism they cited is deleted. C3.1 still
  governs every ordered store that remains.
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

The third design set aside is the one this revision replaces: a configuration-blind
source census. It kept the ban and the wrapper but added a Python scanner that
reconstructed rustc's compiled-source set from source text and matched banned
token spellings against it, with an exact allow-list of (path, item, spelling,
cardinality) rows and a SHA-256 registry of every workspace build script.

Nine exact-head review rounds are recorded on its pull request. Seven of them
found live compiled Rust the reconstruction did not see: a `#[path]` module, a
macro-synthesised path attribute, an attribute split across metavariable
boundaries, a doc comment desugaring to a `#[doc = "..."]` literal that a macro
then captured, a path literal hidden between two lifetime tokens that the
hand-written character-literal lexer mis-scanned, an ordinary `mod name;` with
Rust's inline-module directory semantics, and a macro-emitted `mod name;` that
resolves at its invocation site rather than its definition site. Each was
repaired by adding the missing piece of a Rust lexer, module resolver, or macro
model, and the next round found the next piece. That is the same shape as the
first census above, and it has the same cause: the set of files the compiler
compiles is not computable beside the compiler.

Two observations ended it. Every one of those probes was *proved* live by running
Clippy under the escaping configuration, so Clippy already saw them all; and every
one but the last hid behind an undeclared `cfg`, which `unexpected_cfgs` makes a
compile error. The scanner was therefore strictly weaker than the check it existed
to supplement, defending a configuration space that cannot legally exist. C2.3
keeps the obligation and changes the authority: bound the space with a compile
error, compile all of it with the registered matrix, and let rustc's own dep-info
say what it read. The durable rule, for this document and any successor: **the set
of files the compiler compiles is reported by the compiler, never recomputed
beside it.**

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
[#952]: https://github.com/Chelis-Lang/chelis/issues/952
[#1156]: https://github.com/Chelis-Lang/chelis/issues/1156
[#1198]: https://github.com/Chelis-Lang/chelis/issues/1198
[#1249]: https://github.com/Chelis-Lang/chelis/issues/1249
[#1265]: https://github.com/Chelis-Lang/chelis/issues/1265
[#1277]: https://github.com/Chelis-Lang/chelis/issues/1277
[#1338]: https://github.com/Chelis-Lang/chelis/issues/1338
[#1341]: https://github.com/Chelis-Lang/chelis/issues/1341
[#1343]: https://github.com/Chelis-Lang/chelis/pull/1343
[#1532]: https://github.com/Chelis-Lang/chelis/pull/1532
