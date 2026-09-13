# Named-dimension constraint transport

Design for [#1875](https://github.com/Chelis-Lang/chelis/issues/1875).
The bounded implementation slice below is selected for the repair experiment;
the broader origin-aware options remain unselected. This document does not
change language semantics or claim runtime preservation from checker tests.
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

## Selected bounded implementation slice

Retain an authored input dimension's checker identity when it is compatible
with a concrete axis name. Store the name separately in one mandatory private
`Subst` table keyed by the existing allocated `DimVar`. Ordinary unprotected
Var/Name unification remains unchanged. Mul's already-shared tensor type
retains that identity in either operand order; its ordinary result equality
can therefore no longer escape an authored-result rigidity check through Name.
Both the declared signature and separately resolved annotated parameters must
be protected before body inference. The existing §4.4.1 comparisons remain
authoritative, not replaced by a new origin policy.

Schemes keep their dimension IDs. Instantiation copies labels onto fresh IDs;
aliases use the same generalization and instantiation path, while free captures
keep their IDs. Labelled IDs remain available even after a concrete binding:
the constraint view resolves that binding, the semantic view answers name
queries, and neither generalizer quantifies a concretely bound ID. Final
annotations project names only after authored-result guards. This is not a
claim that two separately named values have equal extents.

An outer unification journals only changed label entries, including copies
made through scheme instantiation, and restores them in reverse on failure.
The journal is checker-local, omitted from serialization, and reset on clone;
unrelated unifications do not copy the accumulated label table. This preserves
the narrow label/shared-type-refinement rollback boundary, not atomicity of
all existing substitution mutations. Context cloning and transactional
substitution composition remain separate whole-state operations.

The §4.5.2 list join classifies axes using constraints, semantic labels and
current authored rigidity, not the private `Dim::Var` carrier tag. Concrete
axes retain the head only when their names agree (or both are literals) and
no known extents conflict. Distinct names and Name/Lit pairs widen even at
equal extents; equal names retain queries when no extent conflict is known.
Unresolved protected binders and unlabelled inference variables still unify,
and wildcard head bias and list-uniformity guards remain authoritative.
This classification is ephemeral: both live and decoded contexts clear
historical protection before a new check; instantiation copies labels, not
protection, while active lexical captures retain their canonical binder IDs.

The constraint-sensitive consumers described below derive an ephemeral
`DimObservation` rather than interpreting a carrier `Var` as a language-level
category. The interface keeps
known constraint extents, semantic names, canonical unresolved IDs and current
protection separate; it has no generic equality or conversion back to `Dim`.
Each operation owns its policy. Scatter-elements containment checks known
numeric bounds before symbolic equality. Where/clamp retain strict matching
shapes, including Name/Lit distinction and rejection of known contradictions
behind equal labels. Window, stride, pad, diagonal and convolution inference
read known constraints for their existing calculations. Surviving axes and
existing identity operations keep the original IDs for authored-result guards.
The joined gather/scatter/trace deferred routes copy IDs and keep their result
dependencies; arithmetic routes are not added to that deferral ledger.

Known constraints do not globally turn named runtime claims into literals.
Expand's named operand claim, reshape's named-input product obligation, and
concat's named-axis unknown rule retain their existing admission policies
(§§4.5.4, 4.7.2–4.7.3). Truly unknown arithmetic remains unknown. The consumer
oracle pairs known contradictions with runtime-admissible unknown claims and
tests whole, live, decoded and layered contexts in both multiplication orders.
The convolution formula has private-seam coverage only: the labelled source
positive still encounters the baseline concrete-metadata restriction in the
final checker validator. That is not claimed fixed by an inference formula.

For example, `aligned[d](x: tensor[d,f32], gain: tensor[fixed,f32]) ->
tensor[d,f32] = mul(x,gain)` establishes the output label `fixed`; the same
signature with body `copy(x)` does not. The private table preserves that
distinction through whole/live/decoded checks and function aliases, without
all-value roots, ADT field paths, or a second checker. A caller still cannot
give the mul result a false literal dimension while promising polymorphic d.

### Serialization and Rust embedding migration

Direct `TypeEnv` serde is now explicitly versioned (format 1), with a mandatory
dimension-label table. Old raw snapshots must be regenerated; missing tables,
unallocated label IDs, and structurally invalid labels reject. Dependency,
stdlib and compiled-context payload versions advance to 15, 19 and 21.
An embedding that previously serialized `TypeEnv` or its owning wrappers must
rebuild those snapshots using the matching checker. Public Scheme inspection
alone does not recreate a reusable binding's private label summary.

The selected source-free boundary is a faithful snapshot from a trusted
checker producer. Structural admission, hashes, build IDs, and proof IDs do
not prove that a malicious producer included every required label. Existing
owner-cache effect/linearity and lowering validation remains necessary.
No public Dim/Type/Scheme variant, School API, source signature, or runtime
tensor layout change is selected.

## #1889 bounded lowering progress

Public `check_in_context` enters lowering too: successful checker snapshot
decode alone is not an end-to-end compatibility result. The separate
[#1889](https://github.com/Chelis-Lang/chelis/issues/1889) helper-result repair
addresses the bounded lowering cases below without changing this checker carrier.
For an inline plain callable, lowering first preserves the helper's
authored return claim, including its runtime obligation and source diagnostic.
When checked and produced ranks agree, it then transports only a distinct,
non-wildcard *named* checker-annotated call-result axis as a caller-side label.
Resolve that axis in the saved caller substitutions before transporting a
remaining name. A caller binder already actualized to a literal must not
reappear as an unbound runtime symbol at an inlined root (#1917).
The label is not resolved against the callee's signature: the same spelling
may name an unrelated callee parameter. Label transport creates no equality
claim and imports neither checker-substituted literals nor optional extents;
an existing produced extent is retained. Authored two-source guards remain
on the unchanged preservation path. Parameter-witness construction and rank
substitution timing are unchanged: checked label transport does not need to
reclassify names copied through a rank splice as callee-authored binders.

It adds no checker value-root/provenance carrier, global name equivalence,
extent-based anchor recovery, wire field, cache version, worker protocol, or
semantic/normative rule. The evaluator/context regression covers direct, alias,
both elementwise operand orders, caller alpha-renaming, rank-spliced caller
names, independent same-spelled callee axes and real
authored mismatches. Native-C attempts stopped before emission at an undeclared
axis (#1277) or generic unresolved-host diagnostic; those residuals remain
unclassified and outside this repair. This is a
lowering-only repair, not approval of the broader origin/template design below.

## Unselected broader origin-aware investigation

The following witness and alternatives concern a stronger proposed value-flow
policy. They are not prerequisites for the bounded existing-spec repair above,
and their desired shared-versus-independent answer is not normative law.

### The information-loss witness

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

One unselected alternative: give each value parameter a private root
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

## Unselected broader representation boundary

Prefer one checker-owned constraint representation used by inference and
publication, not a second source checker or an encoded suffix in name strings.
Private witnesses may reuse `DimVar` storage only if ordinary quantification,
semantic name queries, and authored-binder checks cannot confuse their roles.
Maintain both the semantic axis name and its constraint representative.

Under that broader option, publish a private relation template beside each callable binding: occurrence
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

## Unselected broader implementation sequence

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

## Broader-option acceptance and falsifiers

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
