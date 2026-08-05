# The Capability Table: schema for the op x parameter x lane authority

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
per-backend implementation status. Tables A and B cover operations; the
companion host-ABI constructor table applies the same closed-disposition rule
to recursive host types without becoming a third operation authority.

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

One row per **(builtin, surface, operand dtype, semantic-parameter case)**.
Every builtin declares a finite semantic-parameter product. The product is
`Unit` for an op with no dtype-valued parameter. A dtype-valued argument adds
one closed `Prim` axis, so checked `cast` expands to every
`(source Prim, target Prim, surface)` case rather than one ambiguous "cast x
dtype" row. Multiple dtype-valued arguments form the ordinary Cartesian
product. Authoring macros may state a class, but the machine form expands
every `Prim`; a new `Prim` or parameter axis therefore creates compile-time
holes rather than silently inheriting a neighbor's disposition.

The row fields are:

- **builtin** - the user-facing name (`add`, `mean`, `bitand`, `abs`,
  `reduce_window_max`, `to_string`, ...). NOT `RiscOp`: several audited
  builtins have no IR op at all (the bitwise family, [#682]/[#695]), and users
  hit the table at the name level. The builtin -> RiscOp/kernel mapping is
  a per-backend implementation detail below.
- **surface** - `Scalar | Tensor`, always separate rows. Non-negotiable:
  the audit measured opposite behaviors per surface within single lanes
  ([#715] scalar-stubs vs correct tensors; [#718]'s inverted width matrix).
- **operand dtype** - one row per `Prim` (rows may be authored via dtype-class
  macros - "all integer widths" - but EXPAND to per-Prim rows in the
  machine-readable form, so a new Prim variant leaves visible holes the
  generator turns into compile errors).
- **semantic-parameter case** - the expanded tuple of dtype-valued arguments
  that can change legality or semantics. For `cast`, this is the target Prim;
  paired with the operand dtype it is the complete source x target matrix.
  Literal values, axes, and windows do not become unbounded table axes: their
  validity remains a rule in `sig`.
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
resolution) live in `sig`, not in extra axes. The finite product is only for
closed semantic choices such as a target dtype; it is not an enumeration of
runtime values.

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

### Companion host-ABI constructor table

Concrete host types are recursive, so enumerating observed complete shapes
would recreate [#955] whenever constructors nest in a new order. The machine
schema therefore carries a companion closed table keyed by
**(host constructor, position, backend)**. `position` distinguishes a
constructor's represented fields when their ABI roles differ. Each cell is:

- `Represented { abi-constructor }`;
- `Unimplemented { issue, diagnostic }`; or
- `RejectedByDesign { reason, atom }`.

Primitive leaves use the operation-specific typed-carrier projection of the
corresponding Table-B row. Product,
tuple, `List`, `Option`, and every other admitted host constructor compose
recursively from their cell and their children's dispositions. A composite is
represented if and only if its constructor and every child are represented;
otherwise the first structural child in source order returns its exact typed
authority. There is no `Unknown`, wildcard ABI, or concrete-shape exception.
Consequently, if `List`, `Option`, and `T` are represented for `c-host`, then
both `List<Option<T>>` and `Option<List<T>>` are represented without new rows.

The generated constructor-pair suite nests every constructor inside every
other constructor in each legal position, in both orders where distinct, and
checks a represented leaf plus every negative disposition. Adding a
constructor makes both the recursive resolver and this suite incomplete at
compile time. This is [#730] LU4's permanent mechanism for [#955], not a table
of the shapes that happened to appear in the report.

## Derivations (what consumes which table)

| consumer | derives from | mechanism |
|---|---|---|
| checker acceptance | A | generated predicate; hand lists deleted ([#712]'s class dies here) |
| `chelis check` reporting | A | check always reports A-`Rejected` cells because they are target-independent type facts; B-level rejections surface at build where the target is known |
| build gates | B | generated early-UX gates per [#730] Phase 3's gate contract (earlier/more specific, never the sole defense) |
| backend dispatch | B | macro-generated skeletons; missing arm = compile error |
| [#912] builtin realizability and target sets | A + B | generated routing projection; target-independent legality comes from A and per-backend availability from B, with no independently authored support list |
| conformance suite | expanded A parameter product x B + host constructor table | every (`Supported`, `Implemented`) source x semantic-parameter x surface x backend cell executed, exact agreement or [#732]'s tolerance table; every `Rejected`/`Unimplemented` cell asserts its diagnostic from every stage that renders it; every legal constructor pair is composed in both nesting orders |
| checked host-cast plan | expanded `cast` A rows + c-host B cells | The pre-Table adapter is `CheckedCastPlan`: C host ABI projection maps every admitted scalar/tensor source and target onto that exhaustive plan, exact same-type pairs alone are identity, and every other pair selects its checked implementation or exact typed rejection. [#730] LU6 owns this typed boundary; Phase 4 replaces its interim product source with the expanded rows |
| host-type resolution | host constructor table + operation-specific Table-B typed-carrier projections | recursive composition; no inventory of concrete nested shapes and no default ABI |
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
| `mean` x Tensor x int widths ([#724]) | DECIDED (2026-07, on the issue) and LANDED on main 2026-08-04 (validated at 013b947d, all three lanes citing the issue in `crates/chelis-cli/tests/reduction_and_bitwise_matrix.rs`; [#724] closed. The decision is release-visible at v0.19 per the roadmap's anti-churn invariant 4): `Rejected` at check time on every application form - integer mean requires an explicit cast (`mean(cast(x, f32))`) or `floor_div(sum(x), n)`; `mean`'s sig is float-only (bool rejects with the integers). Scope: only fractional-producing reductions reject; `sum`/`max`/`min`/`prod` over integers stay valid. Phase 4 mechanizes the cell |
| `add`/`sub`/`mul` x (any) x bool ([#726]) | DECIDED (2026-07, on the issue) and LANDED on main 2026-08-04 (validated at 013b947d, a check-time rejection in every surface form per `crates/chelis-cli/tests/issue_860_checker_chokepoint.rs`; [#726] closed. The decision is release-visible at v0.19 per the roadmap's anti-churn invariant 4): `Rejected` per [04-NUM-4] at check time on every application form. The interim authored roster also includes `neg` and `floor_div`, whose bool-typed results fall under [04-NUM-4]. It does NOT claim `sum`/`prod_reduce`: their pre-existing dispositions remain outside this #726 decision until first-class `count` and the reduction cells are authored together. Diagnostic points at `and`/`or`/`not`, the existing explicit-cast counting idiom (`sum(cast(x, int64), 0)`), and the future `count`. Phase 4 mechanizes only the authored cells |
| scalar `relu`/`sigmoid`/`silu`/`gelu`/`tanh` ([#712], [#704]) | DECIDED (2026-07, on the issue) and IMPLEMENTED by the 2026-08-05 pre-table scalar remediation: `Supported` on float dtypes at BOTH surfaces (a scalar is a rank-0 tensor); non-float is a check-time domain `Rejected`, never a silent 0. Eval and C now execute every active float width through dtype-aware kernels/helpers, and the always-run scalar matrix requires byte-identical observation. Phase 4 replaces this interim hand registration and matrix with generated A/B projections and the generated product |
| scalar `tan`/`atan`/`recip` ([#704]) | DECIDED by existing `spec/05-risc-primitives.md` §2.2 and IMPLEMENTED by the 2026-08-05 pre-table scalar remediation on float types only (f32, f64, f16, bf16); non-float is a check-time domain `Rejected`. In particular `recip(0)` is IEEE infinity, not [04-NUM-9] `DivZero`. Eval and C now agree at every admitted width, and the C host path has no silent-zero fallback. Like [#1009]'s callables these three predate the semantic-registration ratchet and have §2.2 table semantics rather than their own `[05-OP-N]` atoms; Phase 4 authors those atoms and replaces the interim hand registration/matrix |
| scalar `floor`/`ceil`/`round` x int widths ([#715]'s rows) | DECIDED (2026-07, via [#712]'s comment) and IMPLEMENTED by the 2026-08-05 pre-table scalar remediation: `Supported` as an exact identity at int8/int16/int32/int64 in checker, eval, and C, with no float conversion. Phase 4 mechanizes the cells |
| `max_elem`/`min_elem` x Scalar x all dtypes ([#715]) | DECIDED (2026-07, via [#712]'s comment) and IMPLEMENTED by the 2026-08-05 pre-table scalar remediation at every admitted signed-integer and float width in checker, eval, and C. Phase 4 mechanizes the cells |
| C-DAG x int64 x `max_elem`/`abs` etc. ([#691]) | B-cells `Unimplemented { issue: #691 }` until integer kernels land - the fmaxf/fabsf substitution becomes a rejection. Status 2026-08-04: the direct DAG integer path is repaired ([#729] Phase 3 dispatches integer min/max/abs through exact checked integer paths, per the roadmap's unclaimed-issue ledger), and [#691] CLOSED 2026-08-04 once PR #1164 rehomed the emitter citations and released the [#730] rejection-authority liveness pin. That closure leaves a live obligation on this row rather than settling it: a `Unimplemented { issue: #691 }` B-cell now cites a CLOSED owner, which is the exact state the pin exists to prevent, so any surviving C-DAG integer cell is re-cited to an open owner (or dispositioned `Implemented` where Phase 3's exact integer dispatch already covers it) before Phase 4 mechanizes this row |
| Metal x int64 x `abs` ([#693]/[#699]) | A is `Supported`; Metal B-cell `Implemented` once [#699]'s raise lands and the MSL integer path is wired; until then `Unimplemented { issue: #693 }` |
| `abs`/`floor`/`ceil`/`round` x Tensor x int widths x compiled backends, grad path included ([#722]/[#699]) | A is `Supported`: `abs` is exact at the declared width and traps at the signed minimum per [04-NUM-9], and `floor`/`ceil`/`round` are the identity on an already-integral value. The B-cells are the seed decision, and [#699]'s `Const { value: 0.0 }` placeholder satisfies neither disposition - a fabricated zero is a third state the schema does not admit, which is exactly what [#722] measured. Because `grad` is built over the lowered DAG, the placeholder reached the reference lane too and both lanes agreed on all-zero gradients, so no cross-lane oracle could see it; a cell's atom owns the adjoint and no Table-B cell may invent one. Status 2026-08-04: the eval integer unary rows and the compiled C `abs` row landed (PR #1065, exact signed widths and MIN traps); the remaining compiled unary cells stay `Unimplemented { issue: #722 }` under [#699] |
| `uniform_like` x Tensor x active float widths x eval/C, plus f32/f64 x HIP ([#937]) | `Supported` under [05-OP-8]. The implementation cell is typed rather than an f32 storage convention: f64 uses a 53-bit unit and one f64 FMA from exactly widened f32 bounds; f32 uses one f32 FMA; f16/bf16 round that f32 result once. Eval and C cover all four active float widths. HIP covers its existing f32/f64 `ElemKind` surface; the f16/bf16 HIP cells remain explicit unsupported cells under [#174], not implied support. Raw-bit C and HIP rows lock the f64 result against the shared authority; a structural negative asserts that no f64 store calls the f32 sampler. This is the concrete seed for Phase 4's generated all-cell suite |
| `bitand`/`bitor`/`bitxor`/`shl`/`shr` x Scalar x int widths ([#682]) | `Supported`; C B-cells `Implemented`; shifts use the width-bounded unsigned helpers required by [04-NUM-13], never raw signed C shifts |
| `to_string` x Tensor/List ([#1059]) | `Supported` (eval already stringifies); C B-cell `Unimplemented { issue: #1059 }` until the emitter renders via [#732]'s formatter. Re-cited 2026-08-04: [#734] owned only removing the `<value>` substitution and closed when the rejection landed, so it can no longer authorize a cell; [#1059] owns implementing the compiled capability and is the open owner |
| `wrap_add`/`wrap_sub`/`wrap_mul` x (both surfaces) x int widths (spec/04 [04-NUM-7], [#753]) | A `Supported` on int8/16/32/64, `Rejected` on bool/float ("no modular arithmetic on non-integer dtypes; see [04-NUM-7]"); B-cells `Unimplemented { issue: #753 }` until kernels land ([#729] Phase 2's natural moment; SMT lowers to `bvadd`/`bvsub`/`bvmul` exactly, no tolerance row) |
| named lossy cast x directions x dtypes ([#759]) | future explicit truncating/narrowing rung over the checked-cast DEFAULT. Phase 1 implements the default only ([04-NUM-14]: target finalization; fractional float-to-int traps `Domain`; strict 0/1 bool; int-to-float IEEE RNE may lose exactness). Per-direction lossy rules remain to be authored as atoms (same discipline as [#753]) and implemented with [#729] Phase 2's kernel work - never the default; bool remains out of scope per [04-NUM-4] |
| checked `cast` x source dtype x target dtype x both surfaces ([#1150], [#1152]) | `Supported` and governed by [04-NUM-14] for every admitted pair; identity exists only where source and target Prim are equal. The interim 9 x 9 product is executable through `CheckedCastPlan` and the generated Phase 3 matrix; every B-cell will replace that adapter with either its named checked implementation or an exact typed rejection. The conformance rows include in-range, fractional, non-finite, overflow, and mixed-offender tensors. Per [04-NUM-15], mixed offenders select the trap attached to the lowest row-major flat index in every lane, including parallel C. [#730] LU6 consumes these cells for host emission; [#729] owns conversion and indexed-trap semantics |

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
before Phase 1 entry. Table A's
(builtin, surface, operand dtype, semantic-parameter case) key
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
4. Row count management (builtins x surfaces x operand dtypes x each finite
   semantic-parameter product is a few thousand cells before cast-like
   products; the authoring macros must stay compact while the machine form
   remains fully expanded).
5. Surface axis for container/boundary callables: the `to_string` x
   Tensor/List seed row already names a `List` surface the
   `Scalar|Tensor` axis forbids (PR #950 red team P1-2). Decide before
   Phase 4 entry: extend the axis, or move container ops to a sibling
   registry per `dtype_semantics.md` §C6's registries-per-family rule.

`pad_sequences`/`pad_sequences_to` ([#1009]) are the completed backfill case:
they predate the §C6 semantic-registration ratchet, and now bind their exact
callable identities to [05-OP-9]/[05-OP-10]. Their cells enter the table at
Phase 4 under those atoms, not as a grandfathered exemption.

[#174]: https://github.com/Chelis-Lang/chelis/issues/174
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
[#722]: https://github.com/Chelis-Lang/chelis/issues/722
[#724]: https://github.com/Chelis-Lang/chelis/issues/724
[#725]: https://github.com/Chelis-Lang/chelis/issues/725
[#726]: https://github.com/Chelis-Lang/chelis/issues/726
[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#731]: https://github.com/Chelis-Lang/chelis/issues/731
[#732]: https://github.com/Chelis-Lang/chelis/issues/732
[#733]: https://github.com/Chelis-Lang/chelis/issues/733
[#734]: https://github.com/Chelis-Lang/chelis/issues/734
[#1059]: https://github.com/Chelis-Lang/chelis/issues/1059
[#753]: https://github.com/Chelis-Lang/chelis/issues/753
[#759]: https://github.com/Chelis-Lang/chelis/issues/759
[#908]: https://github.com/Chelis-Lang/chelis/issues/908
[#912]: https://github.com/Chelis-Lang/chelis/issues/912
[#937]: https://github.com/Chelis-Lang/chelis/issues/937
[#955]: https://github.com/Chelis-Lang/chelis/issues/955
[#1009]: https://github.com/Chelis-Lang/chelis/issues/1009
[#1150]: https://github.com/Chelis-Lang/chelis/issues/1150
[#1152]: https://github.com/Chelis-Lang/chelis/issues/1152
[#717]: https://github.com/Chelis-Lang/chelis/issues/717
