# Named-dimension constraint transport

Design under review for [#1875](https://github.com/Chelis-Lang/chelis/issues/1875).
This document does not change language semantics or declare a compiler repair.
The implementation baseline inspected here is `01e91e766fe48020aff447392d1fb846bdebcdf0`.

## Objective and non-goals

Preserve the information needed to enforce authored dimension-binder rigidity
through named dimensions, calls, aliases, and serialized checker contexts.
The authority is `spec/04-type-system.md` §§4.1, 4.4, 4.4.1 and [04-INF-6],
with `spec/02-surf-syntax.md` P15 for unlisted concrete names.

Keep concrete names concrete: this is not implicit ADT polymorphism. Different
names still mismatch. Name/literal compatibility, wildcard permissiveness,
list widening, and the documented return-only exceptions remain distinct from
an equality between protected authored binders. Do not repair School by erasing
dimensions, inserting casts, or treating its nongeneric ADTs as generic.
No tensor layout, runtime ABI, RNG, effect, AD, or public shell signature change
is part of this design. Runtime preservation of a named extent claim is a
separate obligation; a checker test cannot establish it.

## #1889 bounded lowering progress

The #1889 helper-result repair is deliberately below this proposed checker
carrier. For an inline plain callable, lowering now prefers the non-default,
checker-annotated application result claim over the helper's raw generic return
binder. It may connect that claim only to the exact current-activation Caller
extent witness already created for the bound actual tensor axis. This preserves
the existing declared-result validation and permits a literal actual through a
shape-preserving body operation without treating equal extents as identity.

It adds no checker value-root/provenance carrier, global name equivalence,
extent-based anchor recovery, wire field, cache version, worker protocol, or
semantic/normative rule. Earlier activation witnesses are excluded even when a
later call reuses the same tensor node; unrelated copies remain excluded. The
evaluator/context regression covers direct, alias, and both elementwise operand
orders. A native-C attempt stopped before emission at generic [05-UNS-1]
(chelis#730); its cause is unclassified and outside this repair. This is a
lowering-only repair, not approval of the broader origin/template design below.

## The information-loss witness

These are diagnostic source sketches, not new normative examples:

```chelis
def named_id(x: tensor[fixed, f32]) = copy(x)
def independent(gain: tensor[fixed, f32]) =
  (named_id(to_tensor([1.0f32, 2.0f32])), named_id(gain))
def shared(gain: tensor[fixed, f32]) =
  (named_id(gain), named_id(gain))
```

Renamed to the same exported binding, their published `Scheme` bytes are equal
on the diagnosed baseline: both have type `tensor[fixed] ->
(tensor[fixed], tensor[fixed])`. Their checked bodies differ. An origin-aware
repair must retain any distinction it needs after publication; merely adding
a serde-skipped unification ledger and projecting all witnesses to `Name`
cannot do so. Bare `TypeEnv` snapshots contain no function bodies from which
to recover the relation. This is a specific obstruction to that projection,
not proof that every private representation is impossible.

The original candidate policy was one declaration/instantiation-local origin
per concrete spelling, fresh at function instantiation, with actual argument
constraints joining origins. Reject that policy under the source-compatibility
constraint: it conflates independent ADT parameters in the discriminator below.
Origins are not ordinary universally quantified dimensions, and equal axis
spelling alone must not become global extent equality.

### Minimal semantic discriminator and alternative

```chelis
type Affine = | Affine { gain: tensor[fixed, f32] }
def kept[d](x: tensor[d, f32], left: Affine, right: Affine) -> tensor[d, f32] = {
  _ = mul(left.gain, to_tensor([1.0f32, 2.0f32]))
  mul(x, right.gain)
}
```

The paired variant changes only the final `right.gain` to `left.gain`.
Both currently check through whole-unit, live-context and direct decoded-context
routes. A declaration-wide `fixed` witness puts the literal and `d` in the same
class in BOTH variants, rejecting the disconnected case as well. This is a
deduction about the proposed model, not execution of an origin-aware checker.
The intended positive is a source-compatibility discriminator, not settled law
merely because the current compiler accepts it.

Smallest alternative to design next: give each value parameter a private root
and each tensor axis a structural occurrence path. An ADT field projection is
keyed by (value root, constructor owner, field path, axis), not axis spelling.
Two `Affine` parameters get distinct roots; repeated projection, destructuring,
copy and value aliases preserve the root/path. Constructors connect payload
occurrences to their argument roots. Callable templates express argument/result
links and fresh result roots; instantiation substitutes actual roots and retains
captures. A function alias retains that template, not a freshly unrelated
value. For nested aggregates this uses finite paths in the observed type/body,
not unbounded expansion of recursive ADTs. Branch joins and recursive/unknown
returned aggregate relations require an explicit conservative disposition before
implementation; an absent summary cannot mean independent. This is a design
option, not approval of a general provenance framework.

These identities stay checker-private, alongside semantic types. They neither
add ADT parameters nor alter `Dim::Name` spelling, runtime object identity or
tensor layout. However they must travel with ALL value bindings and projected
results, not only callable schemes. `Type::Adt("Affine", [])` itself carries no
field-root identity. Merely adding callable templates does not solve this case.

The smallest remaining normative decision is when equal concrete names provide
only compatibility and when actual value/operation relationships propagate
protected-binder constraints across ADT projection, calls and aliases. Specify
this pair and preservation laws in the owning chapter before choosing the
carrier. Preserve §4.4.1's expressly stated exceptions; do not replace its
conservative return-only comparison rule with origin inference accidentally.
Explicit generic `Affine[p]` is the already-demonstrated alternative, but changes
the source API and requires separate approval. No such migration is made here.

## Proposed representation boundary

Prefer one checker-owned constraint representation used by inference and
publication, not a second source checker or an encoded suffix in name strings.
Private witnesses may reuse `DimVar` storage only if ordinary quantification,
semantic name queries, and authored-binder checks cannot confuse their roles.
Maintain both the semantic axis name and its constraint representative.

Publish a private relation template beside each callable binding: occurrence
paths, equivalence classes, retained literal facts, and distinction between
fresh invocation-local origins and captured origins. Instantiation must carry
the template through function aliases and returned values, including functions
with no ordinary quantified variables. Raw public `Scheme` inspection may
remain a semantic view, but it must not become a route to silently discard
required checker information. Audit every binding/import constructor before
promising unchanged Rust embedding behavior.

This is a proposed mechanism, not an approved wire layout. In particular,
`Env` currently serializes bindings and `TypeEnv` serializes its private
environment/substitution; those are real compatibility boundaries despite
private Rust fields. Enumerate stdlib, dependency, compiled-context, worker,
and direct serde routes. Use existing owner-specific versioned envelopes and
keys; reject obsolete direct payloads explicitly. Where source is available,
rebuild through the existing checked-library path. Do not default absent
relations to an empty ledger, trust a digest as semantic validation, or invent
source reconstruction for source-free contexts. Specify relation validation
and its trust boundary before accepting a decoded template.

### Concrete carrier and consumer inventory

All references below are against the baseline named above; paths omit `crates/`.

| Boundary | Current path and authority | Required design disposition |
|---|---|---|
| Semantic schemes | `chelis-types/src/types.rs:647` public `Scheme`, including serde and `mono`; `context.rs:220` immutable `TypeEnv::scheme` | Keep the semantic display shape; it is not a complete relation summary. |
| Checker bindings | `chelis-types/src/env.rs:103,387,394,405,642`: ordinary, constructor, lexical bindings and instantiation; `adt.rs:134` constructor schemes | One private binding product pairs semantic scheme with value-root/template evidence. Shadowing, cloning and constructor lookup must move both together; no stale side table keyed only by spelling. |
| Local utilities | Public `Env::bind`, `lookup`, `instantiate`, `generalize` and Env serde | No public accepted-program entry currently takes Env. They are low-level utilities, not a bypass into `CheckedProgram`; keep this distinction. Internal trusted builtin/declaration construction must explicitly establish its evidence. |
| Live checker context | `chelis-types/src/context.rs:118,197`; `infer/program.rs:483,644,794,967` builders, layering, checking | Preserve relations and root allocation high-water marks through resume/base extension. A naked Scheme cannot recreate a checked binding. |
| Direct serde | `TypeEnv` derives Deserialize; `check_ir_with_context` resumes it without rechecking library bodies (`infer/program.rs:1012`) | This IS checker-evidence ingress, unlike display. Settle its trust policy: faithful trusted-snapshot transport or source-backed admission under an untrusted-payload boundary; absent relations must reject. Direct Env/Scheme deserialization must not be promoted into that authority. |
| Dependency and stdlib caches | `chelis-compiler-api/src/library_cache.rs:185,215`; `stdlib_cache.rs:137,184`; payload versions 14/18 in `cache_envelope.rs:80,86` | Both deserialize TypeEnv plus CheckedProgram through `validate_cached_library`; bump owner versions/keys and re-establish relation authority, including direct serde of these wrappers. |
| Compiled contexts | `chelis-compiler-api/src/context.rs:179,205,271,792,857`; version 20 | Disk/encode/decode share an envelope, but the public Deserialize implementation also exists. Guard both, not only `decode`; preserve proof pairing, re-lowering and identity checks. |
| Workers and bindings | `chelis-cli/src/main.rs:6823,8892`: tempfile encode/decode and package-root check; `chelis-python/src/lib.rs:869`: context loader | Workers consume the same accepted context, not a separate raw TypeEnv route. Invalid explicit handoffs stay fatal; absence may use the existing source-build path. |
| Package metadata | `chelis-reef/src/lib.rs:8084,8104,8265`; `chelis-shell/src/lib.rs:40` type_repr/restrictions | Semantic export/query metadata, not a Scheme-to-checker importer. The linked source path builds checked libraries. Do not impose an origin wire change here merely because it prints types; add a guard that metadata alone never gains checker authority. |

The binding carrier therefore cannot be only `Scheme + optional ledger`.
Use a mandatory private product for accepted checker bindings, with explicit
trusted construction for builtins/declared contracts and inference-produced
relations for bodies. Private witness storage may reuse DimVar allocation, but
must retain a distinct role in free-variable, name-query and generalization
operations. Roots and relation templates are not ordinary quantifiers.

### Decode authority: admission alternatives, not an approved wire layout

Structural validation can check occurrence paths, owner/kind/name agreement,
references, allocation bounds, and fresh/captured partitions. It cannot prove
that a semantically required relation was not omitted. A hash of the payload or
matching LibraryProofIds proves neither that omission property nor inference.
`chelis-pipeline-core/src/semantic.rs:71` currently checks pair identity and
effects/linearity, not dimension re-inference; re-lowering checks another
obligation and does not fill this gap.

One sufficient option under an untrusted-payload boundary is source-backed
rebuilding through the existing checked-library builders. Under that policy,
serialized templates are not independent authority: recompute and use
checker-produced bindings, or compare the complete canonical relation product
before reusing one. In-memory checked contexts remain cheaply reusable. Selecting
this policy would exclude source-free direct TypeEnv snapshots from resuming
checking; removing their Deserialize admission would be a deliberate Rust
embedding/serialization compatibility change, not a shell source-API change.
The information-loss witness does not establish that this option is minimal or
that faithful serialization of a trusted checker snapshot is insufficient.
Retaining trusted-snapshot transport is an alternative requiring explicit
producer/decoder trust assumptions and complete preservation of private binding
products. Settle that boundary before choosing either policy; do not promise
unchanged embedding behavior. Neither option permits a versionless empty-ledger
fallback.

Do not equate retained CheckedProgram bodies with authentic authored source.
Both `exprs()` and `annotated_exprs()` return the same annotated vector
(`chelis-types/src/infer/checked.rs:1689`). Inferred type metadata and authored
signature scope must not be mistaken for interchangeable source on reconstruction.
The source-backed option re-prepares from available package source, or transports an
explicit original prepared-source recipe with its declaration/signature context
through the existing owner envelope. The exact representation is future work,
not invented here. A layered library needs its base plus extension and their
scope context; old proof IDs cannot simply be recomputed from concatenated
annotated bodies (the semantic validator documents that distinction).

Under the source-backed option, direct payload-only rebuilds without this source
authority reject. Workers would need a sufficient source recipe or an already
admitted live context; rebuilding per process has a performance cost that must
be measured before selecting that option. Trusted-snapshot transport instead
relies on its stated trust boundary; structural checks or a digest must not be
described as proving semantic completeness of an untrusted snapshot.

## Mergeable implementation sequence

1. Settle the origin policy and carrier/consumer inventory with executable
   discrimination cases. Amend the owning normative text only where a new
   semantic decision is necessary; this design is not that authority.
2. Implement the constraint mechanism and publication transport together with
   versioned cache admission, whole/live/decoded-context tests, and migration
   notes. These are one soundness slice: do not merge a local-only repair that
   can still be bypassed through a warm context.
3. Recheck affected School dependency closures and ordinary package gates at
   the candidate. Report remaining concrete-name errors honestly. Public ADT
   parameterization, if required, needs a separately announced API decision;
   this repair does not grant it. Publish/adopt only through the existing
   reviewed release and School migration sequence.

The recent #1832 deferred-result repair excludes pending type/dimension/rank
variables from both generalization algorithms. Preserve that invariant and
its parity oracle; origin witnesses must not reopen premature generalization.
Concretely `Subst::pending_gate_result_vars` (`unify.rs:861`) collects all free
variables of each applied pending result, including nested dimensions/ranks;
both `env.rs:769` and `:829` exclude them. Private origin/capture relations add
another dependency that levels alone cannot see: do not freshen a result root
while its operand gate is pending. Extend both algorithms and their parity
comparison to the new relation product. The current CLI regression target is
`chelis-cli::issue_1489_deferred_result_generalization`, not a types target.

## Acceptance and falsifiers

The authoritative repair oracle must cover the same source in whole-unit,
live-context, direct round-trip, and rebuilt checked-context routes, comparing
verdicts and authored/inferred result relationships. Retain the original
19-case rigidity matrix and 21 publication controls as diagnostic history;
their proposed-origin expectations are not all settled language requirements.

Required paired cases include literal pins through names, distinct-binder
collapse, legal named/literal calls, independent versus shared call results,
aliases/captures, repeated ADT access and matches, ascriptions, reduction and
insertion name queries, wildcard/list widening, return-only exceptions, and
failed-unification retries. Test late constraints in both operand orders.
Mutation controls must erase a relation, freshen a capture, conflate independent
calls, admit an obsolete payload, and remove the generalization exclusion.
Run the existing generalization parity, cache/worker, and source corpus gates.

Record commands, exact compiler/artifact identities, denominators, failures,
and consumer deltas. A green checker oracle establishes this bounded repair,
not backend shape safety or LaCaDiLE replacement acceptance. Every merge still
requires fresh-context executable review and exact-head green required CI.
