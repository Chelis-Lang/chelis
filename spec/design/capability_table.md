# The Capability Table: schema for the op x dtype x lane authority

**Status:** Schema specification, pre-implementation. No tracking issue of
its own: the table is DELIVERED by `spec/design/dtype_semantics.md`
([#729]) Phase 4; this document owns its SCHEMA, authored now because
three plans consume it ([#729] Phase 4, `loud_unsupported.md` [#730] Phase 3,
`spec_provenance.md` [#733] Phase 3/§C5) and each was otherwise gesturing at
an artifact nobody had specified. Freeze point: the schema freezes at
[#729] Phase 4 ENTRY, changing after only via this doc + both consuming
plans, one change set.
**Owning specs:** `spec/05-risc-primitives.md` (op semantics the rows
cite), `spec/04-type-system.md` (dtype rules), and the four sibling plans.

## The two-table design

### Dependency owner

The machine form will live above the dependency-free `chelis-vocab` crate,
which owns closed identities shared by the checker, runtimes, and backends.
Phase 2 of `loud_unsupported.md` first establishes that owner for
`EffectKind` and `RuntimeDType`. Phase 4 then moves `Prim` and introduces
`BuiltinId` there, with temporary re-exports from `chelis-types`. This avoids
making the C runtime depend on the type checker and ensures table rows,
backend dispatch, runtime dtype IDs, and generated C spellings consume one
closed declaration without a dependency cycle.

The audit's lane-skew findings force a separation the single-table sketch
in [#729] glossed over: *what an op means* is target-independent, while
*whether a backend implements it* is not. Conflating them is how "the
checker accepts what eval rejects" ([#712]) and "Metal rejects rank-2 while
C panics" ([#692]) coexisted. So:

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
| `chelis check` reporting | A | check reports A-`Rejected` hits ALWAYS - target-independent truths need no target. **This resolves [#730]'s open question 2** (formerly orphaned between [#730] and [#731]): yes, check pre-reports, because A-rejections are type-level facts; B-level (target) rejections surface at build, where the target is known |
| build gates | B | generated early-UX gates per [#730] Phase 3's gate contract (earlier/more specific, never the sole defense) |
| backend dispatch | B | macro-generated skeletons; missing arm = compile error |
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
| `mean` x Tensor x int widths ([#724]) | DECIDED (2026-07, on the issue): `Rejected` - integer mean requires an explicit cast (`mean(cast(x, f32))`) or `floor_div(sum(x), n)`; `mean`'s sig is float-only. Scope: only fractional-producing reductions reject; `sum`/`max`/`min`/`prod` over integers stay valid |
| `add`/`sub`/`mul` x (any) x bool ([#726]) | DECIDED (2026-07, on the issue): `Rejected` per [04-NUM-4], diagnostic points at `and`/`or`/`cast`; the mask-counting idiom is preserved via a first-class `count` builtin (chosen over bool-accepting reductions; tracked on [#726]) |
| scalar `relu`/`sigmoid`/`silu`/`gelu`/`tanh` ([#712], [#704]) | DECIDED (2026-07, on the issue): `Supported` on float dtypes at BOTH surfaces (a scalar is a rank-0 tensor; kills the three-lane disagreement in the direction the tensor forms already behave); non-float is a clean domain `Rejected`, never a silent 0 |
| scalar `floor`/`ceil`/`round` x int widths ([#715]'s rows) | DECIDED (2026-07, via [#712]'s comment): `Supported` as identity (the checker's existing stance, made real) |
| `max_elem`/`min_elem` x Scalar x all dtypes ([#715]) | DECIDED (2026-07, via [#712]'s comment): `Supported` at their valid dtypes (eval already correct; C implements via [#730] Phase 1 + kernel work) |
| C-DAG x int64 x `max_elem`/`abs` etc. ([#691]) | B-cells `Unimplemented { issue: #691 }` until integer kernels land - the fmaxf/fabsf substitution becomes a rejection |
| Metal x int64 x `abs` ([#693]/[#699]) | A is `Supported`; Metal B-cell `Implemented` once [#699]'s raise lands and the MSL integer path is wired; until then `Unimplemented { issue: #693 }` |
| `bitand`/`bitor`/`bitxor`/`shl`/`shr` x Scalar x int widths ([#682]) | `Supported`; C B-cells `Unimplemented { issue: #682 }` until emitted |
| `to_string` x Tensor/List ([#734]) | `Supported` (eval already stringifies); C B-cell `Unimplemented { issue: #734 }` until the emitter renders via [#732]'s formatter |
| `wrap_add`/`wrap_sub`/`wrap_mul` x (both surfaces) x int widths (spec/04 [04-NUM-7], [#753]) | A `Supported` on int8/16/32/64, `Rejected` on bool/float ("no modular arithmetic on non-integer dtypes; see [04-NUM-7]"); B-cells `Unimplemented { issue: #753 }` until kernels land ([#729] Phase 2's natural moment; SMT lowers to `bvadd`/`bvsub`/`bvmul` exactly, no tolerance row) |
| named lossy cast x directions x dtypes ([#759]) | the explicit truncating/narrowing rung over the checked-cast default (same discipline as [#753]): per-direction rules authored as atoms - proposals: float->float RNE at target width; float->int truncate-toward-zero with an authored out-of-range rule; int->narrower-int ONE authored rule; never the default; bool out of scope per [04-NUM-4]; B-cells land with [#729] Phase 2's kernel work |

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
[#717]: https://github.com/Chelis-Lang/chelis/issues/717
