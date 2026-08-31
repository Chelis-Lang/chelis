# Compiled Value Ownership

**Status:** Design freeze and source/FFI ownership-contract freeze for [#1286]. No delivery
phase in this document is implemented by the design change that introduces it.

**Owning specs:** `spec/04-type-system.md` [04-LIN-1..8],
`spec/05-risc-primitives.md` [05-OP-31..33], and `spec/11-ffi.md` §2.1.
The exact target heap and C-callable contract described below becomes normative
only through Phase 1's atomic numbered-spec entry amendment.

**Class fixed when:** an invalid compiled ownership state cannot be constructed
at the backend boundary; every heap kind has one exhaustive clone/release path;
every backend receives the same proof before reusing storage; and the complete
`compiled-value-ownership` suite is green with no expected-failure receipt.

## Summary

Compiled values currently cross four mechanisms that do not share one ownership
model:

1. source linearity inserts `Copy` and `Drop`, then erases borrow information;
2. host C emission reconstructs aliases, escaped arguments, and releases from
   `HostProgram` names and expression shapes;
3. tensors use `owns_data` and `chelis_free`, while strings and aggregates use
   independent refcounts and `chelis_value_release` omits tensors; and
4. C and HIP decide in-place reuse with separate backend-local predicates.

That separation is the defect class. A missing arm leaks tensors in aggregates;
an extra release frees a borrowed capture or fresh argument twice; a branch or
fold loses which path owns the result; recursive frames keep dead tensors; and a
backend can reuse caller storage after omitting one guard.

The repair is one structural boundary. The compiler lowers every checked program
to an ownership-explicit form, verifies it, and gives backends only the verified
form. The runtime uses one closed heap-kind and strong-owner model for tensors,
storage, strings, aggregates, and the existing mapped-file resource. Storage
reuse requires one unforgeable proof
created by the shared ownership/memory planner. The selected target C ABI exposes
opaque handles, not ownership fields. Phase 1 may not edit the runtime until its
first change amends the numbered primitive atoms and exact registries for that
ABI, updates the numeric-surface census, and adds the spec-derived tests.

This is a pre-compatibility break. There is no dual ABI, legacy ownership mode,
or fallback inference path.

## Why a refactor and not another patch

Each open instance can be patched locally:

- add a tensor arm to `chelis_value_release`;
- balance one aggregate construction path;
- release one recursive temporary sooner;
- copy C's caller-storage condition into HIP;
- special-case returned arguments, mixed branches, or fold targets.

None removes the next omission. The current compiler allows a backend to receive
an ownership-erased expression and answer “who owns this?” for itself. The current
runtime lets the same tensor be described as a unique allocation, a non-owning
view, or an untyped heap payload without one shared owner identity. Any future
expression form, heap variant, or backend can repeat the class while all existing
tests stay green.

The enforcement target is therefore:

```text
checked program
    -> ownership lowering
    -> total ownership verification
    -> VerifiedOwnershipProgram
    -> shared memory plan with ReusableOwnedStorage proofs
    -> C / HIP / Metal / evaluator consumers
```

No arrow after verification may accept the pre-verification form.

## Non-goals

- Garbage collection. Supported heap graphs are immutable and acyclic, so strong
  ownership is complete.
- User-authored lifetime annotations or Rust-style lifetime syntax.
- Preserving the layout-visible tensor ABI, `chelis_free`, `chelis_alloc_view`,
  or ambiguous value-conversion spellings.
- Treating arbitrary foreign C that violates a documented borrowed-pointer
  lifetime as a safe Chelis program. The supported ABI removes forgeable ownership
  fields and validates every handle state it can observe.
- Closing [#1339], [#763], [#893], or [#912] from this design change. Their
  contracts interlock with this plan, but their parentage and assignees remain.
- Claiming the Tier 1 launch blocker fixed before the implementation phases and
  final oracle are complete.

## Vocabulary

- **owner**: one accountable right to keep a heap value live and eventually
  consume or release it.
- **borrow**: a non-owning use bounded by one verified owner.
- **move**: transfer of one owner into a call, join, aggregate, root, or drop.
- **clone**: creation of one additional owner. For a tensor it may retain a
  descriptor/storage graph; source `copy` may instead produce independent bytes
  where [04-LIN-4] or [04-LIN-7] forbids shared mutable storage.
- **entry borrow**: a handle supplied by an external caller. It is immutable and
  never eligible for in-place reuse.
- **owner join**: one owned block/result parameter with exactly one moved incoming
  owner from every predecessor.
- **storage owner**: the heap allocation whose strong count protects tensor bytes.
- **reuse proof**: a consumed compiler token proving one storage owner is uniquely
  writable at an exact operation.
- **terminal operation**: an owned move into a consuming destination or a `Drop`.

# Part I — the contracts

## C1. The ownership algebra

The normative rules are [04-LIN-1..8]. The implementation model has exactly
three use dispositions:

```rust
pub enum OwnershipUse {
    Borrow,
    Move,
    Clone,
}
```

The actual enum and all fields are private to the ownership module; this shape
records the closed domain, not a public construction API.

Each owned definition mints one `OwnerId`. A borrow names that owner without
changing its state. A move terminates that owner at the source and transfers it
to one destination. A clone reads a live owner and mints another `OwnerId`.
There is no `Unknown`, `MaybeOwned`, raw boolean, or backend-specific use kind.

Aliases remain binding facts for diagnostics, never owner creation. Rebinding a
name does not change an existing owner's identity. A physical pointer is runtime
evidence only and cannot create, merge, or transfer an owner.

The verifier enforces for every path:

- every use refers to a live definition;
- a moved owner has no later use;
- every owner has exactly one terminal operation;
- a borrow does not escape its owner or cross an owner-consuming edge;
- clones and copies produce fresh owner identities;
- heap-kind and value-tag identities agree; and
- no backend-visible value has an undecided disposition.

## C2. The ownership IR boundary

`chelis-ir` gains an ownership-lowered host/control-flow form with private
constructors:

```rust
pub struct OwnershipProgram { /* private blocks, owners, uses, roots */ }
pub struct VerifiedOwnershipProgram(OwnershipProgram);

pub fn lower_ownership(
    checked: &CheckedProgram,
    host: &HostProgram,
    manifest: &RootManifest,
) -> Result<OwnershipProgram, OwnershipError>;

pub fn verify_ownership(
    program: OwnershipProgram,
) -> Result<VerifiedOwnershipProgram, OwnershipError>;
```

The exact crate may use references rather than owned arguments, but the type
boundary is fixed: only `verify_ownership` constructs
`VerifiedOwnershipProgram`, its fields are private, and every compiled backend
entry point takes that verified type. `HostProgram` remains an earlier logical
form and is not itself an emission contract.

The representation contains:

- stable `OwnerId` and `BlockId` identities;
- explicit borrow/move/clone operands;
- owned block parameters for joins and loops;
- terminal consumes and drops;
- owned function results on every return edge;
- ordered root-consume sinks; and
- storage provenance attached by the shared memory planner.

Lowering errors are typed compiler errors. Verification has no warning mode,
debug-only assertion, wildcard acceptance, or “best effort” program result.

At final cutover, the C emitter deletes `analyze_returns_arg`,
`result_alias_set`, `may_return_outer`, `record_alias`, `scope_releases`,
`owned_destinations`, and `retain_call_escaped_args`, plus any successor that
answers the same ownership question from backend-local syntax. A source scan is
supporting evidence; the compile-time backend signature is the structural guard.

## C3. Calls, joins, folds, and roots

### Calls and returns

The resolved parameter type chooses `Borrow` or `Move`. A callee returns a new
owner on every owned result edge. If a result value is an argument or capture:

- a moved argument may become the result owner because its owner transfers
  through the callee;
- a borrowed argument cannot become an owned result without an ordinary copy;
- a still-live captured or outer owner requires a copy before the result; and
- a fresh argument passed to an identity-returning function has one transfer
  chain, not a caller release plus a callee release.

These rules close the mechanisms behind [#1344] and [#1356] without a
“returns argument N” summary.

The artifact boundary is an adapter before these internal edges. It may pass an
`EntryBorrow` directly to a borrowed formal, but it must copy before an owned
formal so the internal move never consumes the caller's owner.

### Branches and matches

Each owned result is an owner join. Every predecessor moves exactly one owner
into it. A predecessor that returns an alias while preserving another live use
clones before the join. A fresh predecessor transfers the fresh owner. No join
has `alias | fresh | unknown` state, so [#1352]'s mixed-arm case is ordinary.

### Loops and folds

A loop-carried accumulator is one owned block parameter. The initializer moves
into iteration zero; each iteration consumes the previous owner once and moves
one successor owner to the back-edge or result. Source rebindings are display
names only and cannot overwrite the accumulator's provenance. This removes
[#1346]'s double-target representation.

### Manifested roots

After [#912]'s root manifest exists, ownership lowering appends one terminal
consume per manifest entry in manifest order. Existing copy insertion handles
fan-out. If roots `a` and `b` denote one value, the earlier root receives a
copy and the final root consumes the remaining owner. A root result exported by
an artifact is owned by the caller. Non-root top-level owners receive a drop.

## C4. One runtime heap

[04-LIN-3..8] and `spec/11-ffi.md` §2.1 own the already-decided owner and entry
semantics. This section selects the implementation architecture that Phase 1
must lift into a numbered primitive atom before changing the runtime. Internally
every heap allocation begins with the same private header:

```rust
struct HeapHeader {
    kind: HeapKind,
    strong: AtomicUsize,
}

enum HeapKind {
    String,
    Tensor,
    TensorStorage,
    List,
    Tuple,
    Dict,
    Adt,
    MappedFile,
}
```

The final universe is exact:

| compiled/runtime identity | `HeapKind` | public ownership carrier |
|---|---|---|
| `string` | `String` | opaque `chelis_string` handle |
| tensor value or internal tensor view | `Tensor` | opaque `chelis_tensor *` handle |
| tensor bytes | `TensorStorage` | private; reached only through a tensor descriptor |
| `List` | `List` | opaque `chelis_list *` handle and `CHELIS_VALUE_LIST` |
| tuple | `Tuple` | opaque `chelis_tuple *` handle and `CHELIS_VALUE_TUPLE` |
| dictionary | `Dict` | opaque `chelis_dict *` handle and `CHELIS_VALUE_DICT` |
| ADT | `Adt` | opaque `chelis_adt *` handle and `CHELIS_VALUE_ADT` |
| `MappedFile` resource | `MappedFile` | opaque `chelis_mapped_file *`; never a `chelis_value` |

The implementation derives a bijection across heap-backed `ConcreteHostType`
variants, heap `chelis_value` tags, private heap allocations, and `HeapKind`.
Scalar, bool, and unit carriers are structurally nonheap. A new heap-backed host
type, value tag, or private allocation fails the same executable registry until
its kind, validation, clone/retain rule, and finalizer are all present.

The real types remain private and may add validation metadata, but they may not
split the kind, count, and finalizer authorities. Retain uses checked relaxed
increment; final release uses release/acquire synchronization before the
kind-specific finalizer. Overflow traps. All clone/release/finalize matches are
wildcard-free, and adding `HeapKind` fails compilation until every consumer is
updated.

`MappedFile` is included even though no current [#1286] child names it. It is
already a refcounted compiled heap value; leaving it on a separate counter with
no exhaustive release edge would preserve the omission mechanism this plan
claims to eliminate. It remains a resource handle and never becomes a
`chelis_value` payload or numeric transport.

Tensor descriptors and tensor storage are distinct heap kinds. A descriptor
retains storage. A view owns a descriptor with shape/stride/offset metadata and
retains the storage for its entire lifetime. Finalizing a descriptor releases
storage once; finalizing storage frees bytes once. There is no `owns_data`, raw
dtype id, or tensor-special free path.

Strings and aggregate nodes are immutable. Constructors clone borrowed children
exactly once; accessors returning by-value `chelis_value` clone exactly once;
finalizers release each stored child exactly once. Because child sets cannot be
mutated after construction, a value cannot be inserted into itself or create a
cycle through a later update.

## C5. Target opaque C ownership ABI

The existing exact ABI remains [05-OP-31], [05-OP-33], and their registries
until Phase 1. The Phase 1 entry amendment must re-check the highest current
`[05-OP-N]`, author the exact lifetime/heap semantics and identity registry,
replace the tensor identities in [05-OP-31]/[05-OP-33], and update every
numeric-capacity authority artifact atomically. Only then does implementation
begin. That amendment must encode this selected target:

- `chelis_tensor` and `chelis_tensor_write` are opaque;
- `chelis_tensor_retain` and `chelis_tensor_release` are the sole public
  tensor lifetime operations;
- `chelis_mapped_file_retain` and `chelis_mapped_file_release` put the existing
  mapped-file resource under the same lifetime authority;
- `chelis_tensor_read_view` and `chelis_tensor_write_view` pair the pointer
  with its exact dtype and element count;
- `chelis_tensor_begin_write` succeeds only for unique runtime storage and
  returns an opaque exclusive guard;
- `chelis_tensor_end_write` consumes the guard and invalidates the write view;
- `chelis_value_clone`, `chelis_value_release`, explicit `take`, and explicit
  `borrow` conversions carry heap ownership; and
- `chelis_free`, `chelis_alloc_view`, public tensor fields,
  `chelis_value_retain`, `chelis_value_from_*`, and
  `chelis_value_as_*` do not survive the cut.

The cut is atomic across the runtime header, generated C/HIP, compiler API,
Python bindings, examples, benchmarks, and every header consumer. No header or
runtime library exposes both ownership models.

An entry argument is tagged `EntryBorrow` before ownership lowering. Retaining
its descriptor does not make the bytes mutable: the provenance remains
external for the invocation. An owned result must use runtime-owned storage.
The caller may later pass that result back as a new entry borrow, but no address
comparison changes the boundary mode.

## C6. Storage provenance and in-place reuse

The common memory planner owns this private vocabulary:

```rust
enum StorageProvenance {
    RuntimeOwned,
    EntryBorrow,
    SharedView,
}

pub struct ReusableOwnedStorage { /* private, linear token */ }
```

Only the planner constructs `ReusableOwnedStorage`, and only when:

1. the source has one live program owner;
2. its descriptor and storage are uniquely owned;
3. it is runtime-owned and writable;
4. no live view or borrow can observe the mutation;
5. shape, dtype, capacity, and operation-specific reuse requirements hold; and
6. consuming the operation is the source owner's terminal use.

The token names the exact source owner, storage identity, and consuming node.
It is moved into the fused operation and cannot be reused. C, HIP, and Metal
consume the same proof-bearing plan. Backend-local `fused_in_place_spec`
functions may select target mechanics only after receiving the proof; they may
not decide ownership eligibility. The runtime checks the live strong counts
and write state again before mutation.

Removing any one condition must fail a controlled test for both C and HIP. The
HIP hardware leg is mandatory on the repository workstation; compiling an
ignored test is not execution evidence.

## C7. Last-use reclamation

Ownership lowering computes terminal positions using control-flow
post-dominance, not source scope alone:

- a straight-line owner drops immediately after its last borrow;
- a branch-local owner drops on each path where it remains live;
- an owner moved into a join is dead in the predecessor;
- a loop iteration drops non-carried owners before the back-edge;
- a tail call happens only after all non-argument frame owners are terminated;
  and
- a manifested root consumes its owner before process teardown.

The runtime's test-only allocation ledger records allocation identity, kind,
size, retain/release events, live owners, and peak live bytes. It is compiled
out of release artifacts and is not a second ownership authority. For the
recursive fixture, the verifier exports the expected live-set bound and the
ledger asserts that peak live bytes do not grow with recursion depth beyond
that bound. RSS is supporting telemetry, not the oracle.

## C8. The authoritative suite

The repository will add one Python driver and one named CI suite:

```text
.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase <N>
```

The final invocation is:

```text
.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase complete --require-hip
```

It exits zero only with final line:

```text
COMPILED VALUE OWNERSHIP ORACLE: PASS
```

The driver owns a typed manifest of fixtures, enumerators, exact commands,
expected outcomes, platform requirements, and controlled mutations. A missing,
duplicate, skipped, or malformed row is a failure. An unavailable required HIP
device reports `BLOCKED` and exits nonzero.

The complete run covers:

1. caller input bytes unchanged after every attempted reusable-input fusion;
2. balanced tensor/string/List/tuple/dictionary/ADT/mapped-file ownership,
   including the exact size and nesting thresholds in [#543] and [#544];
3. top-level aliases, argument/capture/fresh returns, mixed branch/match arms,
   fold-carried aliases, and repeated tensor insertion into aggregates;
4. peak live bytes bounded by the verified working set at recursion depths 32,
   128, and 288;
5. C and HIP rejection of entry-backed or shared-view in-place reuse;
6. clean normal teardown with zero live allocations and no invalid release;
7. the existing [#1222] and [#1344] regressions; and
8. [#1339]'s initialization-order regression as a separately classified Tier
   1 prerequisite, not as a member of this ownership class.

Required mutations include:

- add a `HeapKind` without clone/release/finalize handling: compilation fails;
- omit tensor cloning from `chelis_value_clone`: aggregate balance fails;
- turn one branch clone into a borrow: verifier or runtime fixture fails;
- permit `EntryBorrow` to mint `ReusableOwnedStorage`: C and HIP fixtures fail;
- delay a frame drop past the tail call: the peak-live-byte fixture fails; and
- restore one backend-local ownership predicate: the structural boundary test
  fails even if the positive fixture remains green.

At Phase 0 the manifest may contain typed `ExpectedFailure(issue, detector)`
receipts for still-open children. A phase removes a receipt only when the same
change makes the positive regression green. Assertions are never inverted to
accept current behavior. The final manifest contains zero expected failures.

# Part II — process rules

## B1. Freeze points

| contract | frozen when | later change protocol |
|---|---|---|
| [04-LIN-1..8] owner/call/join/root/entry semantics | this design/spec change merges | amend `spec/04`, this document, and [#1286] together before implementation changes meaning |
| heap, transport, aggregate, and reuse architecture | this design change merges | Phase 1 first lifts it into `spec/05`, an exact normative registry, this document, and the dtype/C-surface interlock together |
| opaque tensor and tagged guarded-access ABI target | this design change merges | Phase 1 first freezes exact identities in the numbered spec and numeric-capacity authority; no compatibility alias |
| `OwnershipProgram`/verified-backend boundary | Phase 2 | change the verifier, all backend signatures, mutations, and oracle manifest together |
| `ReusableOwnedStorage` proof domain | Phase 3 | change the shared planner, every backend consumer, runtime defense, and C/HIP negative controls together |
| complete issue/corpus manifest | Phase 4 | new class member is added to [#1286] and the typed manifest before its fix |

Frozen means an implementation may elaborate representation but may not weaken a
condition, restore inference, or add a compatibility path. A discovered conflict
with a controlling spec stops the phase and amends the controlling document first.

## B2. Invariants at every phase boundary

1. **One owner question, one answer.** Backend syntax never decides ownership.
2. **No unchecked heap kind.** Every discovered heap variant is exhaustively
   validated, cloned, and finalized.
3. **Negative parity.** Every positive ownership fixture has the corresponding
   rejected or balance-failure control.
4. **No expected-failure laundering.** A receipt names an open issue and a
   detector that currently catches it; removing the detector is not progress.
5. **No compatibility bridge.** Old and new tensor ABIs never coexist.
6. **Both compiled backends.** A reuse rule is incomplete until C and HIP consume
   the shared proof and HIP executes on hardware.
7. **Independent semantics.** Lane agreement is supporting evidence only; exact
   ownership counts, bytes, and expected results come from the spec-derived
   ledger and fixtures.
8. **Honest status.** A phase closes only the issue rows whose complete
   reproductions and secondary observations are green.

## B3. How to pick up a phase

1. Read this document, [04-LIN-1..8], [05-OP-31..33], `spec/11-ffi.md` §2.1,
   the complete [#1286] thread, and every issue assigned to the phase.
2. Run the phase oracle before editing and record the exact red/expected-failure
   set in the PR.
3. Write or un-ignore the positive and negative tests before production code.
4. Implement only the phase's structural boundary; file discoveries rather
   than growing the phase oracle after entry.
5. Run the exact phase oracle, the default repository gate, the example corpus,
   and any documented hardware leg.
6. Red-team the exact PR head. Only the latest exact-head review counts.

# Part III — delivery phases

## Phase 0 — executable detectors

**You inherit:** the current runtime split, ownership-erased `HostProgram`, the
existing [#1222]/[#1344] regressions, and the open issue reproducers.

**You deliver:**

- `scripts/compiled_value_ownership_oracle.py` and its unit tests;
- the typed fixture/command/mutation manifest;
- a test-only runtime allocation ledger with deterministic owner/byte counts;
- compile/link/run helpers for generated C and the hardware HIP path;
- exact positive and negative fixtures for all nine children; and
- an explicit external-prerequisite row for [#1339].

The current failing children are represented as typed expected failures. Closed
[#1222] and [#1344] are green controls. No production ownership code changes.

**Not this phase:** refcounts, ABI changes, ownership IR, release scheduling, or
backend reuse fixes.

**Frozen at exit:** the complete initial fixture universe, detector semantics,
ledger event schema, mutation identities, and zero-vacuity rule.

**Authoritative oracle:**
`.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 0`;
exit zero and final line `COMPILED VALUE OWNERSHIP PHASE 0: PASS`.

## Phase 1 — unified heap and atomic ABI cutover

**You inherit:** Phase 0's detectors, the frozen [04-LIN]/FFI contract, and the
selected heap/ABI architecture. The exact heap and callable atoms do not yet
exist.

**You deliver:**

- before production code, an atomic numbered-spec amendment defining the heap,
  transport, aggregate, reuse, and exact ABI identities, plus its registry,
  dtype-capacity updates, and positive/negative test stubs;
- the closed heap header/kind and checked strong-owner operations;
- tensor descriptor/storage ownership and retained internal views;
- exhaustive `chelis_value` clone/release over every heap kind;
- mapped-file retain/release under the same heap header and finalizer dispatch;
- immutable aggregate construction/access/finalization balance;
- the opaque tensor, lifetime registry, and guarded tagged-data ABI;
- atomic migration of generated C, HIP, bindings, examples, and benchmarks; and
- deletion of `owns_data`, `chelis_free`, `chelis_alloc_view`, public field
  reads, and ambiguous conversion aliases.

This phase makes [#544] green and the aggregate-tensor half of [#543] green.
It coordinates the typed runtime seal with [#893] but does not claim that issue
unless its own complete oracle is satisfied. [#543] remains open until its
function-internal tensor-literal temporary is green in Phase 2.

**Not this phase:** backend-local alias inference or final last-use scheduling.

**Frozen at exit:** one heap kind/count/finalizer authority and one public ABI.

**Authoritative oracle:**
`.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 1`;
exit zero and final line `COMPILED VALUE OWNERSHIP PHASE 1: PASS`.

## Phase 2 — verified ownership lowering

**You inherit:** the unified heap and Phase 0 call/branch/fold/root fixtures.

**You deliver:**

- `OwnershipProgram`, private construction, and the total verifier;
- explicit use dispositions, owner joins, loop parameters, and root sinks;
- backend signatures that accept only `VerifiedOwnershipProgram`;
- owned return behavior for arguments, captures, and fresh values;
- ownership-directed releases for heap-valued host code; and
- deletion of C emitter ownership inference for converted forms.

This phase makes [#1346], [#1352], [#1356], and the remaining [#543]
temporary leak green. It preserves [#1222] and [#1344] as ordinary regressions.

**Not this phase:** moving terminal operations earlier than the verifier's
initial correct placement or enabling in-place reuse.

**Frozen at exit:** the verified-backend type boundary and owner algebra.

**Authoritative oracle:**
`.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 2`;
exit zero and final line `COMPILED VALUE OWNERSHIP PHASE 2: PASS`.

## Phase 3 — last-use reclamation and shared reuse proof

**You inherit:** verified owners and the unified storage graph.

**You deliver:**

- post-dominance/last-use terminal placement;
- release of non-carried frame owners before loop back-edges and tail calls;
- `StorageProvenance` and private `ReusableOwnedStorage` construction;
- shared memory-plan consumption by C, HIP, and Metal;
- runtime unique-write defense; and
- executed C and HIP positive/negative reuse fixtures.

This phase makes [#1206] and [#1214] green. The recursive ledger bound is
derived from the verified live set, not a hand-tuned RSS threshold. HIP hardware
absence blocks completion.

**Not this phase:** a backend-specific escape hatch or a caller-buffer copy-on-
write compatibility mode.

**Frozen at exit:** last-use scheduling and the unique-storage proof domain.

**Authoritative oracle:**
`.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 3 --require-hip`;
exit zero and final line `COMPILED VALUE OWNERSHIP PHASE 3: PASS`.

## Phase 4 — class elimination

**You inherit:** all earlier phase oracles and zero unresolved child behavior.

**You deliver:**

- deletion ratchets for old tensor fields/symbols, backend ownership analyses,
  wildcard/no-op release arms, and unverified backend entry points;
- exact header/source and heap-kind/consumer bijections;
- zero expected-failure receipts in the ownership manifest;
- the full example and embedding corpus under the allocation ledger; and
- final exact-head red-team evidence across specs, IR, runtime, C, HIP, docs,
  and the public header.

Every open child closes only after its own full thread, secondary observations,
negative control, and positive regression are satisfied. [#1286] closes only
after all children are closed and the final oracle is green.

**Authoritative oracle:**
`.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase complete --require-hip`;
exit zero and final line `COMPILED VALUE OWNERSHIP ORACLE: PASS`.

# Part IV — bookkeeping

## Issue map

| issue | phase and exact disposition |
|---|---|
| [#543] | Phase 1 adds tensor heap cloning/finalization; Phase 2 closes the function-internal tensor-literal temporary before the issue closes |
| [#544] | Phase 1 makes aggregate child clone/release balance independent of count, capacity growth, and nesting |
| [#1206] | Phase 3 moves dead frame releases before tail calls and proves peak live bytes independent of recursion depth |
| [#1214] | Phase 3 removes backend-local eligibility and executes the shared caller-storage negative on HIP hardware |
| [#1222] | closed instance; Phase 0 onward retains teardown/alias regressions |
| [#1344] | closed instance; Phase 0 onward retains captured-borrow regressions |
| [#1346] | Phase 2 represents fold state as one owned block parameter |
| [#1352] | Phase 2 represents mixed alias/fresh results as one owned join |
| [#1356] | Phase 2 gives fresh call arguments one transfer chain and owned return |

## Interlocks

- **[#912]:** owns root identity, topology, order, lane, and artifact routing.
  [04-LIN-6] and this plan own what observing each already-manifested root does
  to ownership. Neither reconstructs the other's data.
- **[#893]:** owns the typed runtime representation seal. Phase 1 should be one
  coordinated implementation if its exact oracle remains aligned; this design
  does not mark it implemented.
- **[#763]/[#1351]:** lane-check may provide reusable compile/run machinery,
  but lane agreement cannot prove ownership balance or shared-lane defects. The
  ownership suite remains independently callable and spec-derived.
- **[#1339]:** initialization order is a distinct Tier 1 class. Its regression
  runs beside the ownership suite so the launch gate is complete, but it is not
  parented here and does not change this issue map.
- **[#729]:** numeric representation and operation semantics remain controlling.
  The target opaque tagged carrier changes the C construction boundary without
  changing any per-dtype value rule. The Phase 1 entry amendment and
  implementation update the Phase 4B freeze oracle and numeric census with
  every exact ABI identity change in the same change; this design freeze does
  not alter today's exact partition.

## Design-freeze PR boundary

The PR that introduces this document changes the controlling specs and design
records only. It does not add `OwnershipProgram`, change the runtime/header, add
the future oracle, or make a child reproducer green. Its PR body says
`Part of #1286`, records the unimplemented phases, and uses no closing keyword.

[#543]: https://github.com/Chelis-Lang/chelis/issues/543
[#544]: https://github.com/Chelis-Lang/chelis/issues/544
[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#763]: https://github.com/Chelis-Lang/chelis/issues/763
[#893]: https://github.com/Chelis-Lang/chelis/issues/893
[#912]: https://github.com/Chelis-Lang/chelis/issues/912
[#1206]: https://github.com/Chelis-Lang/chelis/issues/1206
[#1214]: https://github.com/Chelis-Lang/chelis/issues/1214
[#1222]: https://github.com/Chelis-Lang/chelis/issues/1222
[#1286]: https://github.com/Chelis-Lang/chelis/issues/1286
[#1339]: https://github.com/Chelis-Lang/chelis/issues/1339
[#1344]: https://github.com/Chelis-Lang/chelis/issues/1344
[#1346]: https://github.com/Chelis-Lang/chelis/issues/1346
[#1351]: https://github.com/Chelis-Lang/chelis/issues/1351
[#1352]: https://github.com/Chelis-Lang/chelis/issues/1352
[#1356]: https://github.com/Chelis-Lang/chelis/issues/1356
