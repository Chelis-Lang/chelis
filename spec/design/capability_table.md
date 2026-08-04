# The Capability Table: schema for the op x dtype x lane authority

**Status:** Schema contract, pre-implementation. The table is delivered by
`spec/design/dtype_semantics.md` ([#729]) Phase 4. This document owns the
schema consumed by [#729] Phase 4, `loud_unsupported.md` [#730] Phase 3, and
`spec_provenance.md` [#733] Phase 3/§C5. The schema freezes at [#729] Phase 4
entry and may then change only through this document and every consuming
plan in one change set.
**Owning specs:** `spec/05-risc-primitives.md` (op semantics the rows
cite), `spec/04-type-system.md` (dtype rules), and the four sibling plans.

## The two-table design

### Dependency owner

The machine form lives above dependency-free `chelis-vocab`, which owns
closed identities shared by the checker, runtimes, and backends.
`loud_unsupported.md` Phase 2 establishes `EffectKind` and `RuntimeDType` in
that crate. Phase 4 places `Prim` and `BuiltinId` there, with compatibility
re-exports from `chelis-types`. `chelis-vocab` owns identity and wire
representation only. Numeric finalization, storage, operation semantics, and
kernel behavior remain in `dtype_semantics.md` and its consumers.

The schema separates target-independent operation semantics from
per-backend implementation status.

The same separation governs host types. `HostTypeTerm -> ConcreteHostType` is
a logical-resolution boundary over checked metadata and does not consult a
backend. Exact primitive identity survives it. `ConcreteHostType ->
HostAbiType` consumes the Table-B target decision and returns `Unsupported`
for `Unimplemented` or `RejectedByDesign`; there is no "unknown logical type"
success case. Before Table B is generated, [#730] permits only a private,
exhaustive target adapter whose negative decisions cite a spec atom or
implementation issue. Table B replaces those decisions without changing the
typed boundary. In particular, f16, bf16, int8, and int16 remain known
logical scalar types even while C-host cells are unimplemented; no target may
substitute int64, f32, `void *`, or a default emitted value.

### Pre-table root-realizability projections ([#912])

The current `BuiltinDecl.realizability` declarations and target capability
sets are pre-Phase-4 routing projections, not a third capability authority.
Before Tables A/B exist they may remain only as exhaustive private adapters
whose negative decisions cite the controlling atom or open implementation
issue. At Phase 4 they derive from the exact Table-A legality and Table-B
backend cells; adding or changing a builtin then changes the tables first and
regenerates the projection.

The root manifest may combine those generated projections with the checked
program's root set, which is [#912]'s subject. It SHALL NOT author operation
legality or backend support itself. Conversely, the manual `KNOWN_TAGS` table
is not a capability-table input at all: Deep structural classification belongs
to the typed `DeepTag` successor governed by [#908]/[#731]. Replacing that raw
string table with exhaustive typed dispositions is a structural-AST handoff,
not a Table-A or Table-B row expansion.

### Table A - the semantic table (target-independent; the checker's law)

One row per **(builtin, surface, dtype)**:

- **builtin** - the user-facing name (`add`, `mean`, `bitand`, `abs`,
  `reduce_window_max`, `to_string`, ...). NOT `RiscOp`: several audited
  builtins have no IR op at all (the bitwise family, [#682]/[#695]), and users
  hit the table at the name level. The builtin -> RiscOp/kernel mapping is
  a per-backend implementation detail below.
- **surface** - `Scalar | Tensor`, always separate rows. Non-negotiable:
  the audit measured opposite behaviors per surface within single lanes
  ([#715] scalar-stubs vs correct tensors; [#718]'s inverted width matrix).
- **dtype** - one row per `Prim` (rows may be authored via dtype-class
  macros - "all integer widths" - but EXPAND to per-Prim rows in the
  machine-readable form, so a new Prim variant leaves visible holes the
  generator turns into compile errors).
- **cell** - `Supported { sig, atom }` or `Rejected { reason, atom }`:
  - `sig`: the result type/shape rule (e.g. `mean: tensor[n, f32] ->
    tensor[f32]`), which the checker derives from - deleting the
    hand-mirrored lists (`TRANSCENDENTAL_FLOAT_ONLY_OPS` becomes a view).
  - `atom`: the controlling current atom revision ([#733] §C5 item 1). The
    machine-readable table schema requires this field, and the pinned Buoy
    shell-side integration checks authority and freshness. A row without an
    atom fails table construction or the blocking provenance policy. This is the authoring-forcing function: an undecided
    cell (integer `mean` [#724], bool `add` [#726]) cannot be made `Supported`
    OR `Rejected` without someone writing the normative sentence and crossing
    the `spec/**` signoff.
  - `Rejected.reason` is the user-facing diagnostic fragment, rendered in
    `loud_unsupported.md` §C2's format by whichever stage reports it.

Effects are NOT rows (they are constructs, not ops - `EffectKind` +
checker totality, [#730]/[#731]). Movement ops and reductions are ordinary
rows; parameter constraints (axis validity, window literalness per [#725]'s
resolution) live in `sig`, not in extra axes.

### Table B - the backend table (per-target reality)

One row per **(A-row, backend)**, backends = `eval | c-host | c-dag |
hip | metal` (eval is a backend here on purpose: [#717] proved the
reference lane needs conformance rows too):

- **cell** - `Implemented { kernel-ref }` or `Unimplemented { issue,
  diagnostic }` or `RejectedByDesign { reason, atom }`:
  - `Implemented`: names the kernel/template/host-op the dispatch
    generator wires; the macro-generated skeleton makes an A-`Supported`
    row with no B-cell in some backend a **compile error in that
    backend** - the "add a builtin" inversion.
  - `Unimplemented`: legal and LOUD - the build gate rejects programs
    hitting the cell with the diagnostic, and the cell must carry a
    tracking issue. This is where the orphaned HIP work gets owned:
    HIP x int64-tensor cells become `Unimplemented { issue: #689 }`
    (and the div-guard gap [#690] rides the same rows) the day the table
    lands, converting silent-F32-kernels into clean rejections until
    someone writes the templates.
  - `RejectedByDesign`: permanent, atom-cited (Metal x f64 - the
    existing exemplary diagnostic becomes this cell's rendering; Metal's
    rank-1 limit is a `sig`-level constraint on its B-cells).

## Derivations (what consumes which table)

| consumer | derives from | mechanism |
|---|---|---|
| checker acceptance | A | generated predicate; hand lists deleted ([#712]'s class dies here) |
| `chelis check` reporting | A | check always reports A-`Rejected` cells because they are target-independent type facts; B-level rejections surface at build where the target is known |
| build gates | B | generated early-UX gates per [#730] Phase 3's gate contract (earlier/more specific, never the sole defense) |
| backend dispatch | B | macro-generated skeletons; missing arm = compile error |
| [#912] builtin realizability and target sets | A + B | generated routing projection; target-independent legality comes from A and per-backend availability from B, with no independently authored support list |
| conformance suite | A x B | every (`Supported`, `Implemented`) cell executed in every backend, exact agreement or [#732]'s tolerance table; every `Rejected`/`Unimplemented` cell asserts its diagnostic from every stage that renders it |
| [#733] citations | A + B | the table schema requires controlling atom revisions; the pinned Buoy policy and Chelis shell adapter check authority, freshness, and selected-surface completeness |

## Seed decisions the table must ship with

The audited cells that were never authored, listed so Phase 4 cannot ship
around them (proposal defaults from the plans; final call is the atom
author's). Rows marked DECIDED were settled in the 2026-07 design review
on the named issues; Phase 4 still ratifies them as atoms per the [#733]
signoff - the decision is recorded, the normative sentence still gets
authored:

| cell | proposal default |
|---|---|
| `mean` x Tensor x int widths ([#724]) | DECIDED (2026-07, on the issue) and IMPLEMENTED in the [#729] Phase 1 stack (originally draft PR #857; not yet landed): `Rejected` at check time on every application form - integer mean requires an explicit cast (`mean(cast(x, f32))`) or `floor_div(sum(x), n)`; `mean`'s sig is float-only (bool rejects with the integers). Scope: only fractional-producing reductions reject; `sum`/`max`/`min`/`prod` over integers stay valid. Phase 4 mechanizes the cell |
| `add`/`sub`/`mul` x (any) x bool ([#726]) | DECIDED (2026-07, on the issue) and IMPLEMENTED in the [#729] Phase 1 stack (originally draft PR #857; not yet landed): `Rejected` per [04-NUM-4] at check time on every application form. The interim authored roster also includes `neg` and `floor_div`, whose bool-typed results fall under [04-NUM-4]. It does NOT claim `sum`/`prod_reduce`: their pre-existing dispositions remain outside this #726 decision until first-class `count` and the reduction cells are authored together. Diagnostic points at `and`/`or`/`not`, the existing explicit-cast counting idiom (`sum(cast(x, int64), 0)`), and the future `count`. Phase 4 mechanizes only the authored cells |
| scalar `relu`/`sigmoid`/`silu`/`gelu`/`tanh` ([#712], [#704]) | DECIDED (2026-07, on the issue): `Supported` on float dtypes at BOTH surfaces (a scalar is a rank-0 tensor; kills the three-lane disagreement in the direction the tensor forms already behave); non-float is a clean domain `Rejected`, never a silent 0 |
| scalar `floor`/`ceil`/`round` x int widths ([#715]'s rows) | DECIDED (2026-07, via [#712]'s comment): `Supported` as identity (the checker's existing stance, made real) |
| `max_elem`/`min_elem` x Scalar x all dtypes ([#715]) | DECIDED (2026-07, via [#712]'s comment): `Supported` at their valid dtypes (eval already correct; C implements via [#730] Phase 1 + kernel work) |
| C-DAG x int64 x `max_elem`/`abs` etc. ([#691]) | B-cells `Unimplemented { issue: #691 }` until integer kernels land - the fmaxf/fabsf substitution becomes a rejection. Status 2026-08-04: the direct DAG integer path is repaired ([#729] Phase 3 dispatches integer min/max/abs through exact checked integer paths, per the roadmap's unclaimed-issue ledger), and [#691] stays OPEN only under the [#730] rejection-authority liveness pin until PR #1164 rehomes the emitter citations |
| Metal x int64 x `abs` ([#693]/[#699]) | A is `Supported`; Metal B-cell `Implemented` once [#699]'s raise lands and the MSL integer path is wired; until then `Unimplemented { issue: #693 }` |
| `bitand`/`bitor`/`bitxor`/`shl`/`shr` x Scalar x int widths ([#682]) | `Supported`; C B-cells `Implemented`; shifts use the width-bounded unsigned helpers required by [04-NUM-13], never raw signed C shifts |
| `to_string` x Tensor/List ([#734]) | `Supported` (eval already stringifies); C B-cell `Unimplemented { issue: #734 }` until the emitter renders via [#732]'s formatter |
| `wrap_add`/`wrap_sub`/`wrap_mul` x (both surfaces) x int widths (spec/04 [04-NUM-7], [#753]) | A `Supported` on int8/16/32/64, `Rejected` on bool/float ("no modular arithmetic on non-integer dtypes; see [04-NUM-7]"); B-cells `Unimplemented { issue: #753 }` until kernels land ([#729] Phase 2's natural moment; SMT lowers to `bvadd`/`bvsub`/`bvmul` exactly, no tolerance row) |
| named lossy cast x directions x dtypes ([#759]) | future explicit truncating/narrowing rung over the checked-cast DEFAULT. Phase 1 implements the default only ([04-NUM-14]: target finalization; fractional float-to-int traps `Domain`; strict 0/1 bool; int-to-float IEEE RNE may lose exactness). Per-direction lossy rules remain to be authored as atoms (same discipline as [#753]) and implemented with [#729] Phase 2's kernel work - never the default; bool remains out of scope per [04-NUM-4] |

## New numeric ops before the table lands (added 2026-07-30)

Recorded after the 2026-07-30 PR sweep found a new numeric op
(`round_to`, PR #891, unmerged) at review with per-dtype semantics
stated only in a Rust doc comment. Between now and [#729] Phase 4, a
NEW numeric op entering the public surface (builtin, prelude, stdlib,
or runtime export) requires a `spec/05-risc-primitives.md` entry in the
same change set: signature, per-dtype semantics at [04-NUM-8]'s
declared arithmetic widths, adjoint or a non-differentiability
statement, and an accumulator rule where applicable. A doc comment is
not an authority (`AGENTS.md` §Numbered Specs Decide; the ops of
chelis#898 are the standing backlog of exactly this omission).

The requirement is structural per family. Table A remains the
language-builtin registry (and checker acceptance is derived from it at
Phase 4). Runtime exports and exported stdlib defs use the §C6
operation-semantic registry delivered with the capacity tripwire:
each structurally discovered numeric callable's exact canonical
identity is a key whose value is one exact `[05-OP-N]` authority. The
registry validator requires chapter `05`, group `OP`, and verbatim atom
existence as a normative line beginning `> **[05-OP-N]**`; it does not
accept a free-text `spec/05` substring, a cross-reference,
`[05-OBS-1]`, or an absent `[05-OP-999]`. Existing numeric-callable
rows at the 2026-07-30 baseline are explicitly grandfathered because
the OP atoms do not yet exist. That grandfathering is an exact IDENTITY
list frozen in the tripwire source, not a citation string a new row can
copy: both pre-ratchet citations are locked that way, so the exemption
covers precisely the rows that predate the ratchet and no others. A NEW
runtime or exported stdlib
numeric callable authors a new `[05-OP-N]` normative atom in spec/05
and adds its exact registry mapping in the same change set. The
deferred PyO3 leg must deliver the same identity-to-authority shape
before Phase 1 entry. Table A's (builtin, surface, dtype) key
deliberately does NOT stretch to those families: runtime exports and
PyO3 functions have no `BuiltinId`, and container/boundary callables
have no `Scalar|Tensor` surface - the `to_string` x Tensor/List seed
row above already strains that axis (PR #950 red team P1-2; open
question 5).

Numeric-ness is signature-derived: a callable whose signature mentions
a numeric dtype requires a registry entry, and the non-numeric
classification is available only for genuinely dtype-free surface.
For the C family this is deliberately conservative: every
non-boolean/non-character built-in arithmetic value type, including bare
`int` and the pointer-sized integer spellings, yields `numeric-op` - and
a spelling the census does not recognize at all is a build failure rather
than a dtype-free row, which is what makes "conservative" true rather
than aspirational (`dtype_semantics.md` §C6, the inverted type-word
rule). PR #956
commit `6ddf1a72d6dea6770a330d5c2ef3b8fa7d023c43` permits exactly three
pre-ratchet plumbing identities to remove that flag - `chelis_alloc`,
`chelis_tensor_from_value_list_typed`, and `chelis_dtype_size`, with their
complete canonical declarations frozen verbatim in
`dtype_semantics.md` §C6. A callable name, parameter name, or substring is
never an exemption. Conditional macro definitions are propagated across
their connected local-include component (either include spelling) before
classification, so a
cross-file type alias cannot make a configuration-varying numeric callable
disappear from this obligation.
Positive controls bind a discovered callable to the exact atom that
decides it. Negative mutation controls add one runtime export and one
exported stdlib numeric def with no entry, bind a callable to a missing
OP atom, and bind it to a non-OP atom; each must fail. Tooling validates
the structured authority kind and existence; it does not infer whether
the human-selected OP atom is semantically relevant, which remains a
normative review check. Review cannot make a mismatched atom
authoritative: if no OP atom's normative text governs the callable, the
numbered spec must gain the decision before the mapping can land.
Changing a callable's canonical identity also invalidates its old
registration. These controls are permanent parts of the §C6 tripwire,
not review instructions.

Table A registration decides LANGUAGE legality - what is legal in Surf,
Deep, and the RISC DAG, target-independently, reported by the checker.
The other family registries record which existing normative decision
controls each discovered callable; they do not create language
semantics. Per-backend executability is Table B's separate decision,
reported at build through [#730]'s `Unsupported` channel where the
target is known: a language-legal op a backend cannot run is a
CAPABILITY rejection, never a checker type error (§Derivations owns
this split; restated here because a new-op author is the person most
tempted to collapse it).

`round_to` specifically: widening an f32 operand to f64, rounding
decimally, and re-narrowing is computing at other than the declared
arithmetic width - non-conforming under [04-NUM-8] as of its 2026-07-28
amendment, which provides no exception vocabulary. Its semantics must
be authored at declared widths, or its dtype set restricted to f64,
before it lands.

## Open questions (decided at [#729] Phase 4 entry, recorded here)

1. Machine form: `const` Rust table vs a checked-in data file with a
   build-script parser (proposal: `const` Rust in one crate; the pinned Buoy
   shell and versioned Chelis adapter read authorities and table metadata, and
   Chelis does not add a second provenance parser).
2. Whether `c-host` and `c-dag` stay distinct backends in B (proposal:
   yes - the audit's divergences between them, [#691] vs host-lane
   exactness, are per-path facts).
3. Signature language for `sig` (how much shape/param constraint is
   expressible; where [#725]'s "window must be literal" rule sits).
4. Row count management (builtins x 2 surfaces x 10 dtypes is a few
   thousand cells; the dtype-class authoring macro's ergonomics).
5. Surface axis for container/boundary callables: the `to_string` x
   Tensor/List seed row already names a `List` surface the
   `Scalar|Tensor` axis forbids (PR #950 red team P1-2). Decide before
   Phase 4 entry: extend the axis, or move container ops to a sibling
   registry per `dtype_semantics.md` §C6's registries-per-family rule.

[#682]: https://github.com/Chelis-Lang/chelis/issues/682
[#690]: https://github.com/Chelis-Lang/chelis/issues/690
[#691]: https://github.com/Chelis-Lang/chelis/issues/691
[#692]: https://github.com/Chelis-Lang/chelis/issues/692
[#693]: https://github.com/Chelis-Lang/chelis/issues/693
[#695]: https://github.com/Chelis-Lang/chelis/issues/695
[#699]: https://github.com/Chelis-Lang/chelis/issues/699
[#704]: https://github.com/Chelis-Lang/chelis/issues/704
[#712]: https://github.com/Chelis-Lang/chelis/issues/712
[#715]: https://github.com/Chelis-Lang/chelis/issues/715
[#718]: https://github.com/Chelis-Lang/chelis/issues/718
[#724]: https://github.com/Chelis-Lang/chelis/issues/724
[#725]: https://github.com/Chelis-Lang/chelis/issues/725
[#726]: https://github.com/Chelis-Lang/chelis/issues/726
[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#731]: https://github.com/Chelis-Lang/chelis/issues/731
[#732]: https://github.com/Chelis-Lang/chelis/issues/732
[#733]: https://github.com/Chelis-Lang/chelis/issues/733
[#734]: https://github.com/Chelis-Lang/chelis/issues/734
[#753]: https://github.com/Chelis-Lang/chelis/issues/753
[#759]: https://github.com/Chelis-Lang/chelis/issues/759
[#908]: https://github.com/Chelis-Lang/chelis/issues/908
[#912]: https://github.com/Chelis-Lang/chelis/issues/912
[#717]: https://github.com/Chelis-Lang/chelis/issues/717
