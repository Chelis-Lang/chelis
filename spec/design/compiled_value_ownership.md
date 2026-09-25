# Compiled Value Ownership

**Status:** Design freeze and source/FFI ownership-contract freeze for [#1286].
Phases 0 through 2 are implemented. The Phase 3 source implementation is
delivered separately from its pending AMD hardware acceptance under [#1286]
and [#1214]; merging the implementation does not complete Phase 3. Phase 4
remains unimplemented. The stable
`compiled-value-ownership-phase0-oracle` CI job enforces the latest landed
phase oracle.

**Owning specs:** `spec/04-type-system.md` [04-LIN-1..8],
`spec/05-risc-primitives.md` [05-OP-31..33] and [05-OP-44] with the four
`spec/registry/c_*.md` registries, and `spec/11-ffi.md` §2.1.
The exact target heap and C-callable contract described below is normative in
[05-OP-44] and the amended [05-OP-31..33]; this document records only how it
is implemented and sequenced.

**Class fixed when:** an invalid compiled ownership state cannot be constructed
at the backend boundary; every ownership-bearing host carrier maps totally to
one exhaustive heap clone/release path; every backend receives the same proof
before reusing storage; and the complete `compiled-value-ownership` suite is
green with no expected-failure receipt.

## Summary

Compiled values currently cross four mechanisms that do not share one ownership
model:

1. source linearity inserts `Copy` and `Drop`, then erases borrow information;
2. host C emission reconstructs aliases, escaped arguments, and releases from
   `HostProgram` names and expression shapes;
3. tensors use `owns_data` and `chelis_free`, strings and aggregates use
   independent refcounts, heap-carrying `Option` values have no release path,
   and `chelis_value_release` omits tensors; and
4. C and HIP decide in-place reuse with separate backend-local predicates.

That separation is the defect class. A missing arm leaks tensors in aggregates;
an extra release frees a borrowed capture or fresh argument twice; a branch or
fold loses which path owns the result; recursive frames keep dead tensors; and a
backend can reuse caller storage after omitting one guard.

The repair is one structural boundary. The compiler lowers every checked program
to an ownership-explicit form, verifies it, and gives backends only the verified
form. The runtime uses one closed heap-kind and strong-owner model for tensors,
storage, strings, aggregates including `Option`, and the existing mapped-file
resource. Storage reuse requires one unforgeable proof
created by the shared ownership/memory planner. The selected target C ABI exposes
opaque heap objects, not ownership fields. Phase 1 may not edit the runtime until its
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
- Claiming [#1339] fixed from this plan's direct forward-capture rows. The
  ownership launch subset and the distinct top-level-initialization release
  blocker have separate oracles. The stronger full-class ownership oracle is
  not the [#1362] release gate.

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

`chelis-ir` gains a sealed, payload-typed ownership boundary with private
constructors. Host emission and tensor-DAG emission are distinct
specializations of the same verified transition:

```rust
pub struct OwnershipProgram<P: EmissionPayload> {
    /* private exact payload, owners, uses, roots, and directives */
}
pub struct VerifiedOwnershipProgram<P: EmissionPayload>(OwnershipProgram<P>);

pub struct HostEmissionPayload { /* private post-selection host program */ }
pub struct DagEmissionPayload { /* private post-optimization DAG */ }

pub type VerifiedHostProgram =
    VerifiedOwnershipProgram<HostEmissionPayload>;
pub type VerifiedDagProgram =
    VerifiedOwnershipProgram<DagEmissionPayload>;

pub fn lower_host_ownership(
    manifested: &ManifestedProgram,
    host: ConcreteHostProgram,
) -> Result<OwnershipProgram<HostEmissionPayload>, OwnershipError>;

pub fn lower_dag_ownership(
    dag: Dag,
) -> Result<OwnershipProgram<DagEmissionPayload>, OwnershipError>;

pub fn verify_ownership<P: EmissionPayload>(
    program: OwnershipProgram<P>,
) -> Result<VerifiedOwnershipProgram<P>, OwnershipError>;
```

`EmissionPayload` is sealed inside `chelis-ir`; downstream crates cannot add a
payload kind: only `verify_ownership` constructs
`VerifiedOwnershipProgram`. Its host and DAG specializations own the exact
post-entry-selection, post-optimization payload together with its directives:
a caller cannot retain a mutable sibling payload, extract the raw payload, or
reorder it after lowering. Backends receive read-only verified views and every
compiled backend entry point takes the specialization for its lane.
`HostProgram` and `Dag` remain earlier logical forms and are not themselves
emission contracts.

The host payload assigns an opaque `HostSiteId` by structural traversal, never
from an identifier's spelling. Every binding, expression, argument, branch or
match edge, loop edge, function entry and return, and manifested root has one
site and one directive-list entry. The independent payload census derives both
the structural kind and owning unit; every attached action must name that same
unit. Ownership operations reference those sites, and verification proves the
payload-site and directive-site universes are bijective before a backend can
observe them. Generic owner operands expose identity, type, and heap class but
no binder spelling. Name projection is a separate sealed capability on the
exact verified binding action or function-parameter association that emits it,
so a clone/drop operand cannot seed backend-local lifetime reconstruction. Host
ABI projection preserves the site identities and parameter associations and
consumes one verified payload if it must produce another; it cannot clone or
rebuild a raw sibling program after verification.

The DAG specialization uses the existing stable `NodeId` identity. Loads are
borrowed entries; producers mint owners; and `Copy` clones. A `Drop` over an
owned producer is an explicit terminal, while a `Drop` over a borrowed `Load`
is a typed logical discard that neither releases nor terminates the external
owner. `Realize` clones a borrowed or still-needed owner and moves a last-use
owned source; `Store` likewise clones an entry borrow but consumes an owned
source. Roots and stores remain explicit sinks. Standalone DAGs and every
nested `HostTensorHelper` run the same DAG ownership verifier. The host payload
keeps each nested helper and its proof inseparable, so a backend cannot route a
raw helper DAG around verification. Phase 2 adds no reusable-storage proof.

An ownership `Apply` carries a closed typed operation schema containing its
resolved operand modes and result class. Its free-form label exists only for
diagnostics and stable rendering. Verification compares every operand and
result against the schema; lowering output is not accepted merely because its
label or self-selected disposition looks plausible.

The representation contains:

- stable `OwnerId` and `BlockId` identities;
- opaque, non-spelling host-site identities and existing DAG `NodeId`s;
- a total payload-site/directive-site bijection;
- typed operation schemas independent of diagnostic labels;
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

A list-building loop's step (`list_push` for `map` and `scan`, `list_extend`
for `flat_map`, `filter_step`, `partition_step`) moves its item into the
accumulator. The emitter's accumulator ABI therefore declares only consuming
entry points, `chelis_list_push_moved` and `chelis_list_extend_moved`, so the
move cannot be realized as a retaining copy that leaves the moved owner live
(chelis#2508). A `filter` step releases the item its predicate rejects.

### Manifested roots

After [#912]'s root manifest exists, ownership lowering appends one terminal
consume per manifest entry in manifest order. Existing copy insertion handles
fan-out. If roots `a` and `b` denote one value, the earlier root receives a
copy and the final root consumes the remaining owner. A root result exported by
an artifact is owned by the caller. Non-root top-level owners receive a drop.

## C4. One runtime heap

[04-LIN-3..8], [05-OP-44], and `spec/11-ffi.md` §2.1 own the already-decided
owner, heap-kind, and entry semantics. This section records the implementation
architecture behind [05-OP-44]; where the two disagree, the atom and its
registry decide and this section has a bug. Internally
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
    Option,
    MappedFile,
}
```

The final universe is exact:

| compiled/runtime identity | `HeapKind` | public ownership carrier |
|---|---|---|
| `string` | `String` | fixed `chelis_string` wrapper with an opaque target and `CHELIS_VALUE_STRING` |
| tensor value or internal tensor view | `Tensor` | opaque `chelis_tensor *` handle and `CHELIS_VALUE_TENSOR` |
| tensor bytes | `TensorStorage` | private; reached only through a tensor descriptor |
| `List` | `List` | opaque `chelis_list *` handle and `CHELIS_VALUE_LIST` |
| tuple | `Tuple` | opaque `chelis_tuple *` handle and `CHELIS_VALUE_TUPLE` |
| dictionary | `Dict` | opaque `chelis_dict *` handle and `CHELIS_VALUE_DICT` |
| ADT | `Adt` | opaque `chelis_adt *` handle and `CHELIS_VALUE_ADT` |
| `Option<T>` where `T` has a target recursive-value representation | `Option` | opaque `chelis_option *` handle and `CHELIS_VALUE_OPTION` |
| `MappedFile` resource | `MappedFile` | opaque `chelis_mapped_file *` handle and `CHELIS_VALUE_MAPPED_FILE` |

The implementation derives one total, wildcard-free ownership classification
over every `ConcreteHostType` variant and, for `Function`, the closed
`ConstructContext` from `host_function_values.md`, plus every public carrier,
heap `chelis_value` tag, private allocation, and `HeapKind`. Each identity has
exactly one disposition: structurally nonheap, target-rejected with an owning
capability issue, direct heap carrier, tagged heap payload, or private heap
allocation. Each heap disposition names exactly one kind and its one finalizer;
tensor storage is the sole private heap allocation with no public tag. Every
directly carried public heap kind also has the table's exact tagged
representation for recursive aggregates. Numeric scalars, bool, unit, and a
function in an already-supported `ContextualCallback` position are structurally
nonheap.

Every target-representable `Option<T>`, including `Option` of a scalar, mapped
resource, or another `Option`, is one immutable `Option` heap node: `None` owns
no child and `Some(value)` owns exactly one tagged child. This deliberately
replaces the current by-value
`chelis_option_scalar` / `chelis_option_value` split, whose type-recursive
ownership cannot be recovered by the emitter.

Function and closure values remain valid Chelis logical values. A function
stored in `Option`, `List`, tuple, dictionary, or ADT is a `FirstClassValue`,
not a contextual callback. Until [#879] supplies the general closure carrier,
the C-host projection rejects the complete recursively containing type with
`UnsupportedKind::HostAbi`, `Stage::Codegen("c")`, and
`Unimplemented { issue: #879 }` after the sealed ownership boundary certifies
the exact selected payload and before backend emission. The ownership plan
tracks the opaque logical function identity as non-heap without inventing a
runtime representation. The same rule applies to HIP or Metal builds that
select the C-host fallback. It is a target capability result, not a language type error, scalar
substitution, empty value, or permission to omit the type from the registry.
This plan neither defines the closure ABI nor closes [#879] or its [#909]
tracker.

A new host type, placement, value tag, or private allocation fails the same
executable registry until it is classified; a heap classification is
incomplete until its kind, validation, clone/retain rule, child walk, and
finalizer are all present.

The real types remain private and may add validation metadata, but they may not
split the kind, count, and finalizer authorities. Retain uses checked relaxed
increment; final release uses release/acquire synchronization before the
kind-specific finalizer. Overflow traps. All clone/release/finalize matches are
wildcard-free, and adding `HeapKind` fails compilation until every consumer is
updated. The runtime validates null and live wrong-kind handles, but it does not
retain freed allocations as tombstones: using a pointer after its owner or guard
has been consumed is outside [05-OP-44]'s C caller precondition.

`MappedFile` is included even though no current [#1286] child names it. It is
already a refcounted compiled heap value; leaving it on a separate counter with
no exhaustive release edge would preserve the omission mechanism this plan
claims to eliminate. Its direct ABI stays an opaque resource handle.
`CHELIS_VALUE_MAPPED_FILE` is the exact tagged representation when that handle
is stored in `Option`, `List`, tuple, dictionary, or ADT. It carries no numeric
payload and does not turn the resource into numeric transport.

Tensor descriptors and tensor storage are distinct heap kinds. A descriptor
retains storage. A view owns a descriptor with shape/stride/offset metadata and
retains the storage for its entire lifetime. Finalizing a descriptor releases
storage once; finalizing storage frees bytes once. There is no `owns_data`, raw
dtype id, or tensor-special free path.

Strings, `Option`, and aggregate nodes are immutable. Constructors clone borrowed
children exactly once; accessors returning by-value `chelis_value` clone exactly once;
finalizers release each stored child exactly once. Because child sets cannot be
mutated after construction, a value cannot be inserted into itself or create a
cycle through a later update.

That sentence is about values, and the consuming container entry points below
do not contradict it, but the reason is worth stating because it is the
argument the whole last-use optimisation rests on and it is not obvious from
either half.

A consuming entry point mutates an allocation, not a live value. The operand
it rewrites is one the verifier proved dead at that point, so no value whose
child set anyone can still read changes, and [05-OP-44]'s premise is intact at
the level it speaks about. What changes is the allocation, which the emitter
reuses for the result instead of allocating a second one and copying.

Acyclicity survives that reuse, and not by assumption. Suppose an in-place
push made a container `X` reachable from the child `c` it just gained. The
push does not change `c`, so `X` was already reachable from `c` beforehand.
Every edge in this heap graph is a strong-owner edge, since each stored child
is retained by its holder and released by its holder's finalizer, so that
path ends in a heap-resident strong owner of `X`. The moved operand is a
second owner on top of it, so the strong count is at least two and the
in-place arm never runs: the entry point clones and releases instead. The
same argument covers every child a merge adds, and it is why the runtime's
count test is the whole check rather than one of several. A value still
cannot be inserted into itself.

## C5. Target opaque C ownership ABI

The exact ABI is [05-OP-31..33], [05-OP-44], and all four registries:
`c_scalar_carrier.md`, `c_container_boundary.md`, `c_tensor_runtime.md`, and
`c_heap_lifetime.md`. The Phase 1 entry amendment authored [05-OP-44] as the
exact lifetime/heap atom with its identity registry, replaced the affected
carrier, container, and tensor identities throughout [05-OP-31..33], and
updated the numeric-capacity authority artifacts in the same change;
implementation begins only after that amendment. The amended atoms encode
this selected target:

- `chelis_tensor` and `chelis_tensor_write` are opaque;
- `chelis_tensor_retain` and `chelis_tensor_release` are the sole public
  tensor lifetime operations;
- every target-representable `Option<T>` uses an opaque `chelis_option *`;
  `chelis_option_retain`,
  `chelis_option_release`, and checked constructor/accessor operations share
  the tagged child clone/release authority, while `CHELIS_VALUE_OPTION` is its
  only `chelis_value` representation;
- `chelis_mapped_file_retain` and `chelis_mapped_file_release` put the existing
  mapped-file resource under the same lifetime authority, and
  `CHELIS_VALUE_MAPPED_FILE` is its only recursively embedded representation;
- `chelis_tensor_read_view` and `chelis_tensor_write_view` pair the pointer
  with its exact dtype and element count;
- a read view remains valid only until its descriptor owner ends or the
  descriptor is passed to `chelis_tensor_begin_write`, whichever comes first.
  A successful begin invalidates every previously returned read view; dereferencing
  such a stale view violates the caller precondition;
- `chelis_tensor_begin_write` succeeds only for unique runtime storage and
  activates and returns an opaque exclusive non-owning guard embedded in the
  descriptor, borrowing the descriptor owner without allocating a guard;
- `chelis_tensor_write_view` borrows that guard, while
  `chelis_tensor_end_write` consumes and deactivates it without freeing an
  allocation or consuming the descriptor owner, and invalidates the write view;
- `chelis_value_clone`, `chelis_value_release`, explicit `take`, and explicit
  `borrow` conversions carry heap ownership; and
- `chelis_free`, `chelis_alloc_view`, public tensor fields,
  `chelis_option_scalar`, `chelis_option_value`, `chelis_value_retain`,
  `chelis_value_from_*`, and
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
It is moved into the fused operation and cannot be reused. C and HIP consume
the same proof-bearing plan. Backend-private C and HIP mechanics own the token
and may derive only target spelling such as pointer qualifiers, a contiguity
fallback, or whether a fallback slot has a later owner. They contain no
ownership or capacity eligibility predicate. Metal consumes
`VerifiedOwnershipProgram` through a typed `MetalNeverReuse` plan whose input cannot carry `ReusableOwnedStorage`; it allocates distinct storage for every
produced node. Enabling Metal reuse is a later amendment to this frozen
boundary and requires Metal positive/negative execution rows in the same
change. The runtime checks the live strong counts and write state again before
mutation.

The shared plan compares exact `CapacityKey` equality, exact `Repr`, and exact
per-axis shape before minting a `FusedElem` reuse token. An unresolved equality
allocates fresh storage. A larger slot is not interchangeable with a smaller
request, a resolved source-axis name has no semantic force beyond its exact
value, and neither backend may consult `DimExprKey`, a machine-integer element
count, or runtime-observed equality. Slot allocation and peak-byte accounting
consume the same plan: distinct live physical slots are counted once, shared
views add no bytes, C caller storage is excluded, HIP input mirrors are
included, and C materialized stores own independent storage. The concrete upper
bound sums distinct slot capacities, never logical-owner live intervals: logical
death permits recycling but does not itself release an allocation. HIP retains
the whole slot pool through cleanup; C can release a descriptor at explicit
`Drop`, so its slot sum can conservatively exceed the observed peak. A known
`LiveByteBound::Exact` value is a concrete upper bound, not an assertion of an
exact observed peak.

An owned `Drop` ends any live exclusive write guard before releasing its
descriptor. Because that release destroys the descriptor as well as ending the
storage owner, the shared plan retires the physical slot after the dropped
owner; a later owner allocates a fresh descriptor even when its `CapacityKey`
is exactly equal. Slots whose prior owner ends without a destructive `Drop`
remain eligible for ordinary exact-capacity recycling.

Target placement follows shipped mechanics. C movement operations currently
materialize canonical tensors and therefore receive `OwnedSlot` placements;
they are not modeled as metadata views the public C runtime cannot construct.
HIP movement operations retain `SharedView` placement. When C transfers an
exact dead slot to a new logical owner, `chelis_tensor_repurpose` resets the
unique runtime-owned descriptor to a same-byte shape before opening the new
write guard. The numbered [05-OP-44] rule and `c_heap_lifetime` registry own
that public runtime operation.

Removing any one condition must fail a controlled test for both C and HIP. The
HIP hardware leg is mandatory on the repository workstation; compiling an
ignored test is not execution evidence. Replacing `MetalNeverReuse` with an
input that accepts a reuse token must fail the structural mutation suite.

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

The consumed operand of a container-producing builtin (a row of the ownership
IR's `CONTAINER_CONSUMERS` table: `append`, `concat` and `skip` at the list
kind, `dict_insert`, `dict_merge` and `dict_remove` at the dictionary kind,
and `string_concat` at the string kind) is moved
into the builtin when the scheduler places that owner's terminal directly
after the application: lowering borrows every builtin operand, and the
last-use scheduler upgrades the borrow to a move and drops the terminal, so
the verifier re-checks the move as it would any other (no live borrow, no
later use). The move establishes only the borrow half of exclusivity. The
sharing half is the runtime's: as with tensor reuse in C6, the consuming entry
point (`chelis_list_append_owned`, `chelis_list_concat_owned`,
`chelis_list_drop_owned`, `chelis_dict_insert_owned`,
`chelis_dict_merge_owned`, `chelis_dict_remove_owned`,
`chelis_string_concat_owned`, private to the emitter like the accumulator
ABI)
re-checks the strong-owner count and mutates in place only at one, otherwise
cloning and releasing the consumed input; a right-hand side that aliases the
consumed left-hand side is such a retained owner and takes the same cloning
path. A retained alias, whether a tuple, an option, an ADT, or a callee that
stored the container, therefore never observes a mutation, and no static rule
inside one unit has to prove exclusivity for a parameter whose callers may
have retained it. A runtime `refcount == 1` test on its own is not this rule:
without the verified move it cannot exclude an un-retained borrow, which is
what chelis#943 measured and rejected.

Each row names its own heap kind, and the match requires that kind on both
the application's result and the named operand. The kind is not read off the
application, because result-class-equals-operand-class is a weaker test than
membership: `chunk`, `map`, `flatten`, `zip` and `enumerate` all take a list
and return a list without being consumers, and a future
`concat(tensor, tensor) -> tensor` would satisfy it while needing a different
entry point entirely. Naming the kind keeps the tensor `concat` that shares
the label `builtin:concat` out of the table by construction rather than by an
emitter guard firing after the scheduler has already retired the operand's
terminal. Adding a heap kind to the table is therefore never a row edit
alone: the kind owes its own consuming entry points, with the same
in-place-at-count-one and otherwise-clone-and-release behaviour, before any
row naming it can land, and it owes an answer for any derived state or
interior pointer its representation publishes.

Membership is narrower than "produces its own kind". A row is for a callable
whose result is its operand with an edit applied, so that reusing the
allocation replaces a copy of the whole operand with the edit alone. The edit
need not grow the container: `append`, `concat`, `dict_insert`, `dict_merge`
and `string_concat` add, `dict_remove` deletes a keyed entry and `skip`
deletes a leading run, but in each the surviving content is carried over in
place rather than rebuilt.

A callable whose result is a positional sub-range of its operand is a row only
where reusing the allocation removes the copy. Two distinct facts decide it,
and only one is normative.

The normative one is [05-OP-44]'s closed heap-kind universe. A sub-range that
shares its operand's buffer needs a private storage kind behind the handle, as
`TensorStorage` already is behind a `Tensor` descriptor, so it is an amendment
to that atom and to `spec/registry/c_heap_lifetime.md` before it is a row here.
It is also unsound against the rows that already exist: `ys = take(xs, 3)`
sharing `xs`'s buffer stands at its own strong count of one, so
`chelis_list_append_owned(ys, v)` takes the in-place arm, writes slot 3, and
clobbers `xs[3]`.

The other is whether an exclusive offset earns its complexity, and for `skip`
it now does. An offset carried inside the single allocation, reached only
through the `builtin:skip` row, needs no new heap kind and conflicts with
nothing normative: it moves only in `chelis_list_drop_owned`, only at strong
count one, only under a verified move, so there is never a second handle over
one buffer and the clobber above cannot arise. Its result is the operand's
suffix, so the surviving content needs no move at all, which is what makes the
call O(count) where the cloning path is O(length). A retired prefix would
otherwise stay allocated until the list dies, so the entry point compacts once
the prefix exceeds the live window, rebuilding the buffer at the live length.
That bounds the waste over the **allocation** rather than the length, which
matters because the allocation is what the ledger reports: a compaction that
drained in place would keep the original capacity and leave a one-element list
holding the buffer of the list it was skipped from. The cost is amortised O(1)
per skipped element. [#2334] delivered it.

`take` is not a row and does not need to be: its result is a prefix, so the
in-place form is a truncation with no offset involved, and its callers are one
standard-library wrapper and two examples. `string_slice` and `string_trim` are
not rows either, because a string carries the derived state described below and
a sub-range of one is not the single mutation a sub-range of a list is. Those
three remain dispositions rather than derivations and could be revisited on
their own evidence.

The string row carries an obligation neither of the other kinds has, and it
is the reason a heap kind is not interchangeable here. `RuntimeString` stores
derived state beside its bytes: `nul_terminated`, which `chelis_string_data`
serves as an interior pointer, and `char_count`, which the character-indexed
`chelis_string_len` returns and which `chelis_string_slice` reads to decide
whether byte indices are character indices. A list or a dictionary has no such
field, so appending to one is a single mutation, while
`chelis_string_concat_owned` maintains all three together or leaves the string
describing itself wrongly. `char_count` gains the right-hand side's count,
which is exact because concatenating two UTF-8 sequences concatenates their
scalar sequences and creates no scalar at the seam. A stale count is invisible
to any ASCII fixture, because ASCII makes bytes and characters agree, so the
tests that cover it are multibyte by construction.

The interior pointer is the one published surface this optimisation can
invalidate. `chelis_string_data` returns a pointer into `nul_terminated`,
which an in-place growth may reallocate.

The safe condition is not that generated code never holds such a pointer
across a statement, because it does. `chelis_json_compare_strings`, emitted
verbatim by `append_json_canonical_object_helpers`, binds two of them and
reads both across a `while` loop and the statements after it. The condition
that actually holds is narrower: no interior pointer in generated code is
derived from an operand a `CONTAINER_CONSUMERS` row can move. Those two point
into single-character slices the helper allocates and releases itself, which
no `string_concat` can consume, so nothing can grow the buffer under them.

Adding `skip` puts a list operand under the same question, and lists answer it
by publishing no interior pointer at all: `chelis_list_index` returns a
`chelis_value` by value and the layout behind `chelis_list` is opaque, so
there is nothing for an advanced offset to invalidate. The string case remains
the one with a published interior pointer, and it remains the reason a kind
whose public surface hands one out with a longer contract would need a
different answer before taking a row.

An in-place growth invalidates such a pointer exactly as the cloning path
already does by releasing the consumed input, and the consuming entry point
is private to the emitter, so no published-ABI caller can reach it. A new
emitted call site owes this check: if it derives an interior pointer from a
value that a row's operand position can name, that pointer must not outlive
the consuming call. A kind whose public surface hands out an interior pointer
with a longer contract would need a different answer before it could take a
row here.

Reading the allocation ledger as an oracle for these rows needs one caution.
A `resize` event at a consuming entry point's site is the only signal that
separates the in-place arm from the cloning one; allocation counts cannot,
because the cloning arm's extra allocation is indistinguishable from any
other. `chelis_string_concat_owned` therefore records that event on every
in-place return, including an empty right-hand side that changes no byte, so
for strings the absence of the event means the cloning arm ran.

`chelis_list_drop_owned` records on the same rule and for the same reason. A
skip on its own frees nothing, so most of its events repeat the figure the list
already had and are purely the arm signal; the compaction rebuilds the buffer at
the live length, and that one records a real shrink.

A cursor gives a second reading the per-call caution does not forbid. Over a
whole walk the consuming arm allocates no list at all while the cloning arm
allocates one per step, so an allocation count against a known baseline
separates them in aggregate even though one extra allocation cannot be
attributed in isolation. That is the form chelis#2334's receipt takes.

The dictionary entry points are uneven on this, so the same reading does not
carry to them. `chelis_dict_merge_owned` records on every in-place return;
`chelis_dict_insert_owned` records only when it pushes, not when it replaces
an existing key; and `chelis_dict_remove_owned` never records. For those two,
a zero count still means "cloned, or edited nothing". chelis#2252 owns
closing the gap.

One obligation belongs to `dict_insert` alone. The cloning
`chelis_dict_insert` releases the value it replaces before cloning the
incoming one, which is safe because the caller still owns the incoming value.
The consuming entry point clones first and releases second: the two may be
the same heap value held exactly once, and releasing first would free it
before the clone reads it.

A second obligation belongs to every row that removes content rather than
adding it, which is now `dict_remove` and `skip`. `chelis_dict_remove_owned`
releases the removed entry's key and value itself, and
`chelis_list_drop_owned` releases each element it retires, because in both the
consumed container keeps its allocation and no finalizer will reach those
children again. The cloning entry points leave that to the caller's own
release of the untouched input, which is why the obligation appears only on
the consuming side.

`skip` adds one corollary the dictionary case does not have, because its
removal leaves the allocation holding slots the container no longer owns: the
list's finalizer walks the live window rather than the allocation. Walking the
allocation would release each retired element a second time, against
[05-OP-44]'s "releases each stored child exactly once", and it is observable
only when a list is finalized while a retired prefix survives -- a cursor
walked to the end compacts that prefix away, so the shape that catches it is a
partial skip released afterwards.

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

The [#1362] Tier 1 A launch invocation is:

```text
.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase launch
```

It requires the C-lane regressions for [#1344], [#1346], and [#1356], plus the
direct-forward [#1339] rejection controls, and exits zero only with final line
`COMPILED VALUE OWNERSHIP LAUNCH SUBSET: PASS`. It excludes Python, HIP, Metal,
and the non-launch child rows exactly as [#1362] does. The implementation phases
may be structural prerequisites for those rows without making every issue they
also serve a launch blocker.

[#1339]'s **direct** forward capture, where a compiled function names the later
value itself, is now an exact checker rejection rather than a compiled expected
failure: both annotated heap-capture shapes and the unannotated control report
an unbound forward value before build, following [04-INF-4]'s source-order
decision. Those three rows leave no emitted ownership behavior for the
ownership phases to repair. [#1339]'s **indirect** shape remains outside this
oracle: without [04-INF-8], an earlier value's initializer could call a
function that reads a later-assigned value before source-ordered `main`
assigns it. [04-INF-8] rejects that program at the checker using PP6's
complete eager-reference graph; runtime initialization remains source ordered.
It is not an ownership defect, and only [#1339]'s dedicated oracle can close
it.

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

The `complete --require-hip` invocation is the eventual [#1286] class-closure
oracle. It is deliberately stronger than the launch invocation and does not
delay v0.19 after the launch subset and the other independent [#1362] gates are
green.

The complete run covers:

1. caller input bytes unchanged after every attempted reusable-input fusion;
2. balanced tensor/string/List/tuple/dictionary/ADT/Option/mapped-file ownership,
   including `Option[MappedFile]`, nested resource aggregates, and the exact
   size and nesting thresholds in [#543] and [#544];
3. exact `Unimplemented { issue: #879 }` C-host rejection for
   `Option[function]` and every other recursive aggregate containing a
   first-class function, with the same rejection through HIP/Metal host fallback;
4. top-level aliases, argument/capture/fresh returns, mixed branch/match arms,
   fold-carried aliases, and repeated tensor insertion into aggregates;
5. peak live bytes bounded by the verified working set at recursion depths 32,
   128, and 288;
6. C and HIP rejection of entry-backed or shared-view in-place reuse;
7. Metal emission with distinct storage for every produced node and no input,
   view, or intermediate alias;
8. clean normal teardown with zero live allocations and no invalid release;
9. the existing [#1222] and [#1344] regressions; and
10. [#1339]'s annotated list/tensor forward captures and unannotated control as
    exact source-order rejections, separately classified controls rather than
    members of this ownership class. Its indirect initializer-through-a-call
    shape is owned by [04-INF-8] and the dedicated [#1339] oracle; it is not
    covered by these rows and is not an ownership defect.

Required mutations include:

- add a `HeapKind` without clone/release/finalize handling: compilation fails;
- omit `ConcreteHostType::Option` or `CHELIS_VALUE_OPTION` from the closed
  host/carrier/heap projection: compilation or the structural registry fails;
- omit `CHELIS_VALUE_MAPPED_FILE` or its `Option[MappedFile]` fixture: the
  recursive-carrier registry or ownership balance fails;
- admit `Option[function]` or another recursive function container without the
  exact [#879] target rejection: the host-type/placement registry fails;
- omit tensor cloning from `chelis_value_clone`: aggregate balance fails;
- turn one branch clone into a borrow: verifier or runtime fixture fails;
- permit `EntryBorrow` to mint `ReusableOwnedStorage`: C and HIP fixtures fail;
- let the Metal plan accept `ReusableOwnedStorage`: the structural no-reuse
  mutation fails;
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
| heap, transport, aggregate, and reuse architecture | [05-OP-44] and `spec/registry/c_heap_lifetime.md` merge | amend [05-OP-44], its registry, this document, and the dtype/C-surface interlock together |
| opaque tensor and tagged guarded-access ABI target | the amended [05-OP-31..33] and [05-OP-44] merge | amend the exact identities in the numbered spec and the numeric-capacity authority together; no compatibility alias |
| `OwnershipProgram`/verified-backend boundary | Phase 2 | change the verifier, all backend signatures, mutations, and oracle manifest together |
| `ReusableOwnedStorage` proof domain | Phase 3 | change the shared planner, every backend consumer, runtime defense, and C/HIP negative controls together |
| complete issue/corpus manifest | Phase 4 | new class member is added to [#1286] and the typed manifest before its fix |

Frozen means an implementation may elaborate representation but may not weaken a
condition, restore inference, or add a compatibility path. A discovered conflict
with a controlling spec stops the phase and amends the controlling document first.

## B2. Invariants at every phase boundary

1. **One owner question, one answer.** Backend syntax never decides ownership.
2. **No unchecked ownership carrier.** Every host variant is classified, and
   every discovered heap variant is exhaustively validated, cloned, walked,
   and finalized.
3. **Negative parity.** Every positive ownership fixture has the corresponding
   rejected or balance-failure control.
4. **No expected-failure laundering.** A receipt names an open issue and a
   detector that currently catches it; removing the detector is not progress.
5. **No compatibility bridge.** Old and new tensor ABIs never coexist.
6. **Every compiled backend is safe.** A reuse rule is incomplete until C and
   HIP consume the shared proof and HIP executes on hardware. Metal remains a
   typed no-reuse consumer until an amendment adds equivalent execution rows.
7. **Independent semantics.** Lane agreement is supporting evidence only; exact
   ownership counts, bytes, and expected results come from the spec-derived
   ledger and fixtures.
8. **Honest status.** A phase closes only the mapped ownership rows whose exact
   ownership reproductions and ownership-specific secondary observations are
   green. A support, syntax, diagnostic, or reachability observation outside
   this class must have an explicit external owner before phase exit; this plan
   neither absorbs its implementation nor waits for it to become green.

## B3. How to pick up a phase

1. Read this document, [04-LIN-1..8], [05-OP-31..33], [05-OP-44], `spec/11-ffi.md` §2.1,
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
- exact positive and negative fixtures for all nine children;
- scalar, heap-child, and recursively nested target-representable `Option`
  balance fixtures;
- negative `Option[function]` and recursive function-container fixtures that
  require the exact [#879] target rejection; and
- explicit direct-forward rejection controls for [#1339].

The current failing children are represented as typed expected failures. Closed
[#1222] and [#1344] are green controls, and [#1339]'s direct-forward rows are
green exact checker rejections. The acyclic indirect initialization blocker is
not part of this oracle. No production ownership code changes.

**Not this phase:** refcounts, ABI changes, ownership IR, release scheduling,
backend reuse fixes, or the general closure ABI owned by [#879].

**Frozen at exit:** the complete initial fixture universe, detector semantics,
ledger event schema, mutation identities, and zero-vacuity rule.

**Authoritative oracle:**
`.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 0`;
exit zero and final line `COMPILED VALUE OWNERSHIP PHASE 0: PASS`.

**Phase 0 delivery receipt (2026-08-31):** the oracle freezes the complete
initial fixture identity, checked-in source, child-issue, command, detector,
mutation-activation, exact-output, and ledger-receipt universes. Deleting an
issue and its rows together, or adding an undeclared source file, fails the
independent frozen census. The recursive function-container fixtures use named
function values so C and HIP reach the C-host ABI projection; Metal's current
successful emission is an exact typed expected failure owned by [#879], not a
passing rejection receipt. The Phase 3 C/HIP reuse rows name exact behavioral
tests, preflight the test list against zero-test success, and freeze ignored
HIP device-entry tests that compare the caller's bytes after rejected reuse
and execute a certified program-owned reuse.
The ledger is compiled only by the private `ownership-ledger` runtime feature;
normal runtime behavior and the public ABI are unchanged. Its JSONL event stream
uses deterministic logical owner identities and portable payload-byte counts,
then validates exact per-event keys, kinds, bytes, identities, sites, transition
state, invalid-operation class, and final reconciliation. A malformed event,
wrong result or stdout, empty allocation stream, or mismatched summary fails
closed before an ownership receipt can pass. The typed expected failures remain
red for their exact current reasons. In particular, the ledger finds the small
aggregate-string leaks that the historical platform leak thresholds treated as
controls. Phase 0 is hardware-independent: it compiles and lists the HIP
execution test through the normal workspace surface and freezes its exact
zero-test preflight; the phase oracle executes that preflight and the device
test only when `--require-hip` becomes mandatory in Phase 3.

## Phase 1 — unified heap and atomic ABI cutover

**You inherit:** Phase 0's detectors, the frozen [04-LIN]/FFI contract, and the
selected heap/ABI architecture. This phase's entry amendment establishes the
exact heap and callable atoms before any production implementation changes.

**You deliver:**

- before production code, an atomic numbered-spec amendment defining the heap,
  transport, aggregate, reuse, and exact ABI identities, plus its registry,
  dtype-capacity updates, and positive/negative test stubs;
- the closed heap header/kind and checked strong-owner operations;
- tensor descriptor/storage ownership and retained internal views;
- exhaustive `chelis_value` clone/release over every heap kind;
- canonical opaque `Option` nodes and deletion of both by-value option carriers;
- mapped-file retain/release under the same heap header and finalizer dispatch;
- immutable aggregate construction/access/finalization balance;
- the opaque tensor, lifetime registry, and guarded tagged-data ABI;
- atomic migration of generated C, HIP, bindings, examples, and benchmarks; and
- deletion of `owns_data`, `chelis_free`, `chelis_alloc_view`, public field
  reads, and ambiguous conversion aliases.

This phase promotes exactly twenty-three oracle rows: the five [#543]
aggregate-tensor rows (including the function-internal tensor-literal
temporary), the eight [#544] size/nesting rows, the five direct/nested
`Option` and mapped-file rows, and the five [#879] Metal rejection rows. It
therefore makes [#543] and [#544] green in full. It coordinates the typed
runtime seal with [#893] but does not claim that issue unless its own complete
oracle is satisfied.

**Not this phase:** backend-local alias inference, final last-use scheduling,
or the general closure ABI owned by [#879].

**Frozen at exit:** one heap kind/count/finalizer authority and one public ABI.

**Authoritative oracle:**
`.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 1`;
exit zero and final line `COMPILED VALUE OWNERSHIP PHASE 1: PASS`.

**Phase 1 delivery receipt (2026-09-03):** the committed transition removes
the twenty-three Phase 1 expected-failure receipts and executes their exact
positive/rejection behavior. The runtime proof also runs the four ownership
semantic suites under `--features ownership-ledger`, with frozen test listings
and per-test execution receipts; zero matches, ignored/skipped outcomes,
listing-only evidence, and forged supervisor transcripts fail closed. The
separate forged `__main__` and forged import-transcript controls prove that
only the oracle-owned callback receipt can certify Python test execution. The
`compiled-value-ownership-phase0-oracle` job retains its identity in
`heavy-e2e.yml`, invoking Phase 2 and the launch subset daily at 03:17 UTC or
on manual dispatch. It is not a required PR check. The receipt is valid only when the
Phase 1 oracle and all supporting representation, dtype, rejection, capacity,
and fast-gate checks pass on the same committed head.

## Phase 2 — verified ownership lowering

**You inherit:** the unified heap and Phase 0 call/branch/fold/root fixtures.

**You deliver:**

- `OwnershipProgram`, private construction, and the total verifier;
- explicit use dispositions, owner joins, loop parameters, and root sinks;
- backend signatures that accept only `VerifiedOwnershipProgram`;
- payload-owning host and DAG verified specializations, including verified
  nested tensor-helper DAGs;
- owned return behavior for arguments, captures, and fresh values;
- ownership-directed releases for heap-valued host code; and
- deletion of C emitter ownership inference for converted forms.

Phase 2 owns the closed disposition for every `RiscOp::Drop` consumer. For an
owned source, C and HIP emit the same single release selected by the verified
DAG directive and exclude that owner from epilogue cleanup. For a borrowed
`Load`, both consume a distinct borrowed-discard directive and emit no release;
the external owner remains live. Metal consumes both verified directives
through its typed no-reuse/no-device-owner plan and never invents a device
owner or a reuse decision. The backend emission mechanics land after the
sealed boundary and DAG verifier exist, but remain part of this phase's exit
contract.

This phase promotes exactly six oracle rows: the [#1346] fold row, the two
[#1352] mixed fresh-arm rows, the two [#1356] fresh-argument rows, and
`recursive-depth-1-control`. Real scope-exit `Drop` balances the depth-one
recursive frame completely; Phase 3 still owns the last-use peak bound for
depths 32, 128, and 288. The launch subset remains the same four rows. Phase 2
preserves [#1222] and [#1344] as ordinary regressions.

**Not this phase:** moving terminal operations earlier than the verifier's
initial correct placement or enabling in-place reuse.

**Frozen at exit:** the verified-backend type boundary and owner algebra.

**Authoritative oracle:**
`.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 2`;
exit zero and final line `COMPILED VALUE OWNERSHIP PHASE 2: PASS`.

**Phase 2 delivery receipt:** the committed transition removes the six Phase 2
expected-failure receipts, consumes the sealed host and DAG ownership actions
in C/HIP/Metal, and deletes the parallel C-host ownership inference. The same
head must also pass `--phase launch`, whose final line is
`COMPILED VALUE OWNERSHIP LAUNCH SUBSET: PASS`. The three remaining [#1206]
peak receipts record conservative scope-exit placement and retain Phase 3 as
their promotion phase.

## Phase 3 — last-use reclamation and shared reuse proof

**You inherit:** verified owners and the unified storage graph.

**You deliver:**

- post-dominance/last-use terminal placement;
- release of non-carried frame owners before loop back-edges and tail calls;
- `StorageProvenance` and private `ReusableOwnedStorage` construction;
- shared proof-bearing memory-plan consumption by C and HIP;
- a typed `MetalNeverReuse` plan that cannot accept reusable storage;
- runtime unique-write defense; and
- executed C and HIP positive/negative reuse fixtures.

This phase makes [#1206] and [#1214] green. The recursive ledger bound is
derived from the verified live set, not a hand-tuned RSS threshold. HIP hardware
absence blocks completion.

**Phase 3 source receipt (AMD gate outstanding):** the current draft consumes
the landed exact `CapacityKey` from one sealed C/HIP storage planner. The plan
owns `VerifiedDagProgram`, makes token construction private and one-take, maps
target physical placement separately from semantic provenance, performs exact
slot reuse, and computes the physical live-byte bound. C and HIP no longer
contain capacity or ownership eligibility predicates; each backend owns only
the token-derived emission mechanics. C emits real slot transfer through the
same-byte `chelis_tensor_repurpose` defense, while HIP derives both kernel
qualifiers and wrapper aliasing from one retained token. Host C execution,
planner collision controls, caller-owned C/HIP negatives, the isolated
six-condition matrix, Metal's owning no-reuse boundary, and runtime repurpose
positive/negative tests execute locally. The two HIP hardware rows are typed
must-passes but have no local execution receipt; until the authoritative AMD
command below passes on the tested implementation commit, this receipt is source-complete rather
than Phase 3 completion evidence.

**Delivery boundary:** the implementation PR may land after its review, hosted
CI, and local gate pass, with AMD execution retained as follow-up work under
[#1286] and [#1214]. The hardware fixtures remain mandatory must-passes in the
Phase 3 oracle; neither issue closure nor Phase 3 completion is authorized by
that implementation merge. The manually dispatched `ownership-hip.yml`
workflow runs this unchanged oracle and uploads its log and commit receipt.
It requires a separately registered AMD runner; see
[`docs/local_hip_environment.md`](../../docs/local_hip_environment.md#ownership-phase-3-in-ci).

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
- exact host/header/source/value-tag and heap-kind/consumer classifications;
- zero expected-failure receipts in the ownership manifest;
- the full example and embedding corpus under the allocation ledger; and
- final exact-head red-team evidence across specs, IR, runtime, C, HIP, the
  Metal no-reuse boundary, docs, and the public header.

Every child ownership row closes only after its exact ownership reproduction,
negative control, and positive regression are green. Before the child issue
closes, every other observation in its full thread must be explicitly assigned
outside this class or shown to be evidence rather than a separate obligation;
the unrelated implementation is not added to this oracle. [#1286] closes only
after every child ownership row and issue disposition is complete and the final
oracle is green.

**Authoritative oracle:**
`.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase complete --require-hip`;
exit zero and final line `COMPILED VALUE OWNERSHIP ORACLE: PASS`.

# Part IV — bookkeeping

## Issue map

| issue | phase and exact disposition |
|---|---|
| [#543] | Phase 1 adds tensor heap cloning/finalization and closes all five aggregate-tensor rows, including the function-internal tensor-literal temporary. The top-level tuple missing-`main` observation is [#545], not an ownership-oracle row |
| [#544] | Phase 1 makes aggregate child clone/release balance independent of count, capacity growth, and nesting |
| [#1206] | Phase 2 balances the depth-one recursive frame with real scope-exit `Drop`; Phase 3 moves dead frame releases before tail calls and proves the depths 32/128/288 peak live bytes independent of recursion depth. Runtime-valued `with seed` remains [#735] syntax/semantics work; recursive-host operation support remains [#729]/[#730] capability work |
| [#1214] | Phase 3 removes backend-local eligibility and executes the shared caller-storage negative on HIP hardware. [#1172] owns the span-key cause that can over-broaden hints; Surf reachability is exposure evidence, not another ownership mechanism |
| [#1222] | closed instance; Phase 0 onward retains teardown/alias regressions |
| [#1344] | closed instance; Phase 0 onward retains captured-borrow regressions |
| [#1346] | Phase 2 represents fold state as one owned block parameter |
| [#1352] | Phase 2 represents mixed alias/fresh results as one owned join |
| [#1356] | Phase 2 gives fresh call arguments one transfer chain and owned return |

## Interlocks

- **[#912]:** owns root identity, topology, order, lane, and artifact routing.
  [04-LIN-6] and this plan own what observing each already-manifested root does
  to ownership. Neither reconstructs the other's data.
- **[#893]:** owns the typed runtime representation seal. PR #1400 freezes the
  current layout-visible [05-OP-31] tensor carrier on `main`, while this plan
  selects an incompatible opaque carrier for the ownership cut. Phase 1 must
  explicitly supersede its numbered-spec citations, `runtime_representation.md`
  target, guards, and public-layout promise in the same atomic change before
  the opaque implementation calls the representation sealed. The two carrier
  contracts never coexist. This design does not mark [#893] implemented.
- **[#909]/[#879]:** own shared first-class function representation and the
  general C-host closure ABI. This plan imports their closed contextual-versus-
  first-class placement decision and exact [#879] target rejection; it does not
  invent a callable tag, environment layout, capture lifetime, or closure
  copy/drop rule.
- **[#763]/[#1351]:** lane-check may provide reusable compile/run machinery,
  but lane agreement cannot prove ownership balance or shared-lane defects. The
  ownership suite remains independently callable and spec-derived.
- **[#1339]:** initialization order is a distinct Tier 1 class. Its annotated
  list/tensor captures and unannotated control must reject as unbound before
  backend emission; [04-INF-8] separately requires its indirect
  initializer-through-a-call shape to reject after PP6 completes the shared
  eager graph. The direct regressions run beside the ownership suite, but they
  do not complete that release blocker. The issue is not parented here and
  does not change this issue map.
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
[#545]: https://github.com/Chelis-Lang/chelis/issues/545
[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#735]: https://github.com/Chelis-Lang/chelis/issues/735
[#763]: https://github.com/Chelis-Lang/chelis/issues/763
[#879]: https://github.com/Chelis-Lang/chelis/issues/879
[#893]: https://github.com/Chelis-Lang/chelis/issues/893
[#909]: https://github.com/Chelis-Lang/chelis/issues/909
[#912]: https://github.com/Chelis-Lang/chelis/issues/912
[#1172]: https://github.com/Chelis-Lang/chelis/issues/1172
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
[#1362]: https://github.com/Chelis-Lang/chelis/issues/1362
[#2334]: https://github.com/Chelis-Lang/chelis/issues/2334
