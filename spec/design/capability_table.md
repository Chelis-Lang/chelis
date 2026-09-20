# The Capability Table: schema for the op x parameter x lane authority

**Status:** Phase 4B schema frozen; machine tables and generated consumers are
not implemented. This document owns the schema consumed by [#729] Phase 4,
`loud_unsupported.md` [#730] Phase 3, and `spec_provenance.md` [#733]
Phase 3/§C5. A schema change requires this document and every consuming plan
to change together.
**Owning specs:** `spec/05-risc-primitives.md` (op semantics the rows
cite), `spec/04-type-system.md` (dtype rules), and the four sibling plans.

## The two-table design

### Dependency owner

The machine form lives in `chelis-types`, which owns `Prim`, `BuiltinId`, the
capability keys and cells, and target-independent numeric policy. Those types
do not move to or duplicate into `chelis-vocab`. Dependency-free
`chelis-vocab` owns only the cross-layer representation vocabulary required by
lower layers, including `EffectKind`, `RuntimeDType`, `Repr`, and canonical C
wire spellings. [`runtime_representation.md`](runtime_representation.md) C1
adds `DTypeContract` as the closed [04-NUM-8] projection that binds each runtime
dtype to its stored and arithmetic representation; it does not add an operation
or backend-capability decision. Backends consume `chelis-types` for numeric
identity and capability policy and `chelis-vocab` for representation
vocabulary.

The schema separates target-independent operation semantics from
per-backend implementation status. Tables A and B cover operations; the
companion host-ABI primitive-leaf and constructor tables and the
exported-stdlib effect-disposition registry apply the same closed-disposition
rule to recursive host types and effect dependencies without becoming
additional operation authorities.

The same separation governs host types. `HostTypeTerm -> ConcreteHostType` is
a logical-resolution boundary over checked metadata and does not consult a
backend. Exact primitive identity survives it. `ConcreteHostType ->
HostAbiType` consumes the operation-independent primitive-leaf and constructor
tables and returns `Unsupported` for `Unimplemented` or `RejectedByDesign`;
there is no "unknown logical type" success case. Table B decides whether an
operation is implemented on a backend and is never consulted to decide
whether a primitive host-ABI leaf exists. Before the companion tables are
populated, [#730] permits only a private, exhaustive target adapter whose
negative decisions cite a spec atom or implementation issue. The companion
tables replace those decisions without changing the typed boundary. In
particular, f16, bf16, i8, and i16 remain known logical scalar types even
while one host-ABI leaf or operation cell is unimplemented; no target may
substitute i64, f32, `void *`, or a default emitted value.

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
legality or backend support itself. Conversely, the wildcard-free typed
`DeepTag` lane disposition is not a capability-table input at all: Deep
structural classification belongs to the successor governed by [#908]/[#731].
Maintaining that exhaustive typed disposition is a structural-AST handoff,
not a Table-A or Table-B row expansion.

### Table A - the semantic table (target-independent; the checker's law)

One row per
**(`BuiltinId`, `SurfaceClass`, operand `Prim`, `SemanticParams`)**, where
`SurfaceClass` is exactly `Scalar | Tensor`.
Every builtin declares its applicable capability domains and a finite
semantic-parameter product. The product is
`Unit` for an op with no dtype-valued parameter. A dtype-valued argument adds
one closed `Prim` axis, so checked `cast` expands to every
`(source Prim, target Prim, surface)` case rather than one ambiguous "cast x
dtype" row. Multiple dtype-valued arguments form the ordinary Cartesian
product. Authoring macros may state a class, but the machine form expands
every `Prim`; a new `Prim` or parameter axis therefore creates compile-time
holes rather than silently inheriting a neighbor's disposition.

The row fields are:

- **builtin** - a closed `BuiltinId` whose declaration supplies the
  user-facing name (`add`, `mean`, `bitand`, `abs`,
  `reduce_window_max`, `cast`, ...). NOT `RiscOp`: several audited
  builtins have no IR op at all (the bitwise family, [#682]/[#695]), and users
  hit the table at the name level. The builtin -> RiscOp/kernel mapping is
  a per-backend implementation detail below.
- **surface** - `SurfaceClass::Scalar | SurfaceClass::Tensor`, always
  separate rows. Non-negotiable:
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
- **cell** - one of these typed shapes; no field is free-form prose:
  - `Supported { signature_rule: SignatureRuleId, result_dtype_rule:
    ResultDtypeRule, op_atom: SpecAtomRef }`;
  - `Rejected { op_atom: SpecAtomRef, diagnostic_kind: DiagnosticKind }`.
  `SignatureRuleId` names a closed checker rule for shape and bounded
  parameter constraints. `ResultDtypeRule` determines the result dtype
  without reparsing a signature string. `SpecAtomRef` is a structured
  normative authority; for a numeric callable it resolves to the exact
  governing `[05-OP-N]`. A missing, duplicate, malformed, or nonexistent
  authority makes table construction fail. An undecided cell therefore
  cannot become either `Supported` or `Rejected` without first authoring its
  language rule.

Effects are NOT Table-A or Table-B rows: they are constructs, not operations.
The exported-stdlib effect-disposition registry below gives their execution
dependencies a separate typed authority. `chelis-vocab::EffectKind` remains
the closed syntax/handler vocabulary and is not that authority. Movement ops
and reductions are ordinary rows; parameter constraints live behind
`SignatureRuleId`, not in extra axes. Reduction-window rules use
[05-RWIN-1..2]'s runtime `window_shape` and `strides` validation and
output-shape computation; a literal-only signature is not an admitted rule,
and [#1298] owns its implementation. The finite
product is only for closed semantic choices such as a target dtype; it is not
an enumeration of runtime values. Container and boundary builtins do not
stretch `SurfaceClass`: their declarations select the sibling builtin
registry defined below. An overloaded builtin may select more than one
domain, but every applicable domain is explicit and total.

### Table B - the backend table (per-target reality)

One row per **(Table-A `Supported` row, backend)**, backends exactly
`eval | c-host | c-dag | hip | metal` (eval is a backend here on purpose:
[#717] proved the reference lane needs conformance rows too). A Table-A
`Rejected` row has no Table-B product:

- **cell** - one of these typed shapes:
  `Implemented { kernel_id: KernelId }`,
  `Unimplemented { issue: IssueRef, diagnostic_kind: DiagnosticKind }`, or
  `RejectedByDesign { atom: SpecAtomRef, diagnostic_kind: DiagnosticKind }`:
  - `Implemented`: names the kernel/template/host-op the dispatch
    generator wires; the macro-generated skeleton makes an A-`Supported`
    row with no B-cell in some backend a **compile error in that
    backend** - the "add a builtin" inversion. It covers the complete
    target-independent `SignatureRuleId` domain of that Table-A row. A kernel
    that covers only one rank, dtype, or runtime-parameter subset leaves the
    whole B-cell `Unimplemented`; Table B has no target-local signature
    narrowing.
  - `Unimplemented`: legal and LOUD - the build gate rejects programs
    hitting the cell with the diagnostic, and the cell must carry an open
    implementation issue. This is where the orphaned HIP work gets owned:
    HIP x i64-tensor cells become `Unimplemented { issue: #689 }`
    (and the div-guard gap [#690] rides the same rows) the day the table
    lands, converting silent-F32-kernels into clean rejections until
    someone writes the templates.
  - `RejectedByDesign`: permanent and atom-cited. Current backend capacity,
    including a rank- or dtype-limited kernel, is never design authority and
    therefore uses `Unimplemented` under an open implementation owner.

### Sibling builtin registry - container and boundary callables

Builtins whose legal applications cannot be keyed by a numeric scalar/tensor
operand use one sibling semantic registry. Its key is exactly
**(`BuiltinId`, `SiblingDomain`, `SiblingCaseId`, `SemanticParams`)**:

- `SiblingDomain` is the closed enum `Container | Boundary`. `Container`
  covers operations whose governing input is an ADT or recursive container;
  `Boundary` covers conversions between the language value domain and a
  non-numeric result or external representation.
- `SiblingCaseId` is a closed, builtin-owned identifier for one finite type
  shape relevant to legality or implementation, such as
  `ToStringTensor`, `ToStringList`, or `PadSequencesListOfTensor`. It is not a
  free-form type string. The builtin's exhaustive case enumerator maps every
  accepted checked application to exactly one case; no case and multiple
  cases are both construction failures. Recursive ABI representability is
  not duplicated here: it remains the companion constructor table's job.
- `SemanticParams` has the same finite-product meaning as in Table A. A
  dtype-valued choice such as `pad_sequences_to`'s output dtype expands to
  one `Prim` case per admitted dtype.

The semantic cell is one of these typed shapes:

- `Supported { signature_rule: SignatureRuleId, result_type_rule:
  ResultTypeRuleId, op_atom: SpecAtomRef }`;
- `Rejected { op_atom: SpecAtomRef, diagnostic_kind: DiagnosticKind }`.

`ResultTypeRuleId` is the general result-type counterpart to Table A's
`ResultDtypeRule`; it may select a string, container, tensor, or other checked
type without encoding prose. Every sibling `Supported` row expands across
the same exact backend set as Table B (`eval | c-host | c-dag | hip | metal`),
and those backend cells use the identical `Implemented | Unimplemented |
RejectedByDesign` shapes. Thus `to_string` is owned only by the `Boundary`
domain: its disjoint `ToStringScalar`, `ToStringTensor`, and `ToStringList`
applications are separate sibling cases and may carry different backend
dispositions without putting `List` into `SurfaceClass`. It has no Table-A
row. The exhaustive `to_string` case enumerator must name every additional
checked type shape it accepts; its closed set is scalar, tensor, and List. A
wildcard, fallthrough, or additional accepted type shape is forbidden.

Every `BuiltinDecl` carries a non-empty closed set of
`Numeric | Container | Boundary` domains plus the exact case enumerator for
each selected sibling domain. Numeric means Table A. A declaration may select
multiple domains, but an accepted application must resolve to exactly one
semantic row across them. Adding a builtin, domain, or sibling case without a
semantic disposition fails construction; adding a supported sibling row
without every backend cell fails compilation. Runtime exports, exported
stdlib definitions, and binding functions have no `BuiltinId` and therefore
do not enter this registry. Their separate semantic and execution authorities
are defined below. They do not feed [#912] builtin-root realizability.

### External callable registries - semantic authority and execution reality

The §C6 enumerators discover three closed external callable families:
`RuntimeCExport | ExportedStdlibDef | PyO3Binding`. Their canonical identities
include the family and complete callable declaration, so a rename or signature
change removes the old key and adds a new one.

For every structurally numeric callable, the existing semantic registry is
keyed by **(`ExternalCallableFamily`, `CanonicalCallableId`)** and contains the
exact `SemanticRegistration { callable, atom }` required by §C6. It binds the
callable to one controlling `[05-OP-N]`; it is not an implementation-status
cell. A dtype-free callable has no semantic registration, and classification
is derived by the family enumerator rather than declared by the row.

Runtime C exports and PyO3 bindings additionally populate an external target
disposition registry keyed by **(`ExternalCallableFamily`,
`CanonicalCallableId`, `ExternalTargetContext`)**. The family fixes the only
admitted target context: `RuntimeCExport -> c-runtime` and `PyO3Binding ->
python-extension`. Every discovered callable in those two families has
exactly one cell, numeric or not:

- `Implemented { implementation_id: ExternalImplementationId }`;
- `Unimplemented { issue: IssueRef, diagnostic_kind: DiagnosticKind }`; or
- `RejectedByDesign { atom: SpecAtomRef, diagnostic_kind: DiagnosticKind }`.

Missing, duplicate, family/target-mismatched, and stale-identity rows fail the
registry constructor. A numeric row also fails unless its separate semantic
registration exists. The target disposition says whether the published
external callable exists and executes in its owning environment; it does not
decide language legality or substitute for a backend cell of a builtin used
inside that implementation.

### Exported-stdlib effect-disposition registry

Effects reached by an exported stdlib definition use a companion registry
owned by `chelis-types`, not an implied field on Table B. Its key is exactly
**(`CanonicalEffectRequirement`, `BackendId`)**. The requirement is a closed
typed value matching the type layer's complete effect domain:
`Random | Accum | IO | Test | Resource(ResourceId)`. `ResourceId` preserves
the exact checked UTF-8 string literal; it is not an author-written row label,
an inferred device class, or a normalization rule.

The finite row universe is the four payload-free requirements crossed with
the exact backend set `eval | c-host | c-dag | hip | metal`, plus every exact
`ResourceId` discovered by the completed checked-body dependency closure of
the exported-stdlib manifest crossed with that same backend set. Adding a
variant to `chelis_types::Effect` makes the conversion to
`CanonicalEffectRequirement` non-exhaustive at compile time. Adding or
changing a reachable resource literal adds or removes exact rows; it cannot
inherit another resource's disposition.

Every row has exactly one typed cell:

- `Implemented { implementation_id: EffectImplementationId }`;
- `Unimplemented { issue: IssueRef, diagnostic_kind: DiagnosticKind }`; or
- `RejectedByDesign { atom: SpecAtomRef, diagnostic_kind: DiagnosticKind }`.

Missing, duplicate, or stale rows fail registry construction. There is no
missing-row, wildcard, or default disposition. A permanent rejection requires
an existing normative atom and an implementation gap requires an open concrete
issue; the parent [#729] is not a substitute for either. In particular,
resource-selection meaning remains owned by [#735], so a resource/backend case
whose result depends on that unauthored decision remains
`Unimplemented { issue: #735, diagnostic_kind: UnsupportedFeature }` until
that issue authors the controlling semantics and implementation receipt.

Dependency traversal returns a sealed `CompleteEffectDependencies` only after
the checked-body fixed point has completed. Its explicit `Pure` case means the
completed set is empty. An absent traversal result, an unresolved call, or an
unfinished recursive fixed point cannot be converted to `Pure`.

An exported stdlib definition has no independent external target cell: it is
Chelis source compiled for the selected backend. Its executability is derived
transitively from every statically resolved builtin Table-B cell, sibling
backend cell, host primitive-leaf or constructor cell, and exact
effect-registry row reachable
from its checked body. The exported-stdlib manifest must expose a checked body
and declared signature to this generated dependency closure. An unresolved
call, missing owning row, absent dependency result, or recursive cycle without
a completed fixed point is a construction failure. For each backend, the
definition is executable exactly when every reachable dependency is
implemented; otherwise the first dependency in canonical source order returns
its typed rejection authority. Its §C6 semantic registration still binds a
numeric exported identity to the language atom; it never supplies execution
status.

### Companion host-ABI primitive-leaf and constructor tables

Concrete host types are recursive, so enumerating observed complete shapes
would recreate [#955] whenever constructors nest in a new order. The machine
schema therefore carries an operation-independent primitive-leaf table keyed
by **(primitive, primitive position, backend)**. `primitive` is every active
`Prim`; `primitive position` is the closed `HostPrimitivePosition` enum
`FunctionParameter | FunctionResult | TensorElement | AggregateField |
ContainerElement | CallbackParameter | CallbackResult`; and backend is the
same closed backend set as Table B. Every product cell is exactly one of:

- `Represented { abi_leaf: HostAbiPrimitiveId }`;
- `Unimplemented { issue: IssueRef, diagnostic_kind: DiagnosticKind }`; or
- `RejectedByDesign { atom: SpecAtomRef, diagnostic_kind: DiagnosticKind }`.

Missing, duplicate, or stale primitive-leaf cells fail construction. Adding a
`Prim` or primitive position makes the product non-exhaustive at compile time.
No leaf may inherit a carrier from a Table-B operation, a wider primitive, a
different position, or a default arm. Consequently an identity function,
`Option[f16]`, and a callback carrying `f16` resolve the same exact primitive
without depending on whether an unrelated f16 arithmetic kernel exists.

The schema also carries a companion closed constructor table keyed by
**(host constructor, position, backend)**. `position` distinguishes a
constructor's represented fields when their ABI roles differ. Each cell is
typed:

- `Represented { abi_constructor: HostAbiConstructorId }`;
- `Unimplemented { issue: IssueRef, diagnostic_kind: DiagnosticKind }`; or
- `RejectedByDesign { atom: SpecAtomRef, diagnostic_kind: DiagnosticKind }`.

Primitive leaves use the exact primitive-leaf cell for their role. Product,
tuple, `List`, `Option`, and every other admitted host constructor compose
recursively from their cell and their children's dispositions. A composite is
represented if and only if its constructor and every child are represented;
otherwise the first structural child in source order returns its exact typed
authority. There is no `Unknown`, wildcard ABI, or concrete-shape exception.
Consequently, if `List`, `Option`, and `T` are represented for `c-host`, then
both `List<Option<T>>` and `Option<List<T>>` are represented without new rows.

The generated primitive-leaf suite removes or defaults every leaf cell in
turn and requires construction to fail. The generated constructor-pair suite
nests every constructor inside every other constructor in each legal
position, in both orders where distinct, and checks a represented leaf plus
every negative disposition. Adding a constructor makes both the recursive
resolver and this suite incomplete at compile time. This is [#730] LU4's
permanent mechanism for [#955], not a table of the shapes that happened to
appear in the report.

## Derivations (what consumes which table)

| consumer | derives from | mechanism |
|---|---|---|
| checker acceptance | A + sibling semantic registry | generated predicate; hand lists deleted ([#712]'s class dies here) |
| `chelis check` reporting | A + sibling semantic registry | check always reports semantic `Rejected` cells because they are target-independent type facts; backend-level rejections surface at build where the target is known |
| build gates | B + sibling backend product + effect-disposition registry + exported-stdlib dependency closure | generated early-UX gates per [#730] Phase 3's gate contract (earlier/more specific, never the sole defense) |
| backend dispatch | B + sibling backend product + external target dispositions | macro-generated skeletons; missing arm or external disposition = compile error |
| [#912] builtin realizability and target sets | A + B + host/sibling-builtin registries | generated routing projection; a root capability comes from the domain that owns it, with no independently authored support list |
| conformance suite | expanded A parameter product x B + host primitive-leaf and constructor tables + sibling builtin registry + external target dispositions + effect-disposition registry + exported-stdlib dependency closure | every (`Supported`, `Implemented`) source x semantic-parameter x surface x backend cell executed, exact agreement or [#732]'s tolerance table; every `Rejected`/`Unimplemented` cell asserts its diagnostic from every stage that renders it; every primitive ABI leaf and legal constructor pair is composed in both nesting orders; every sibling-registry, external callable, and effect requirement is exercised under its declared domain; every stdlib export is derived and exercised per backend |
| checked host-cast plan | expanded `cast` A rows + c-host B cells | The pre-Table adapter is `CheckedCastPlan`: C host ABI projection maps every admitted scalar/tensor source and target onto that exhaustive plan, exact same-type pairs alone are identity, and every other pair selects its checked implementation or exact typed rejection. [#730] LU6 owns this typed boundary; Phase 4D replaces its interim product source with the Phase-4C rows |
| host-type resolution | operation-independent host primitive-leaf table + host constructor table | recursive composition; no operation-dependent leaf carrier, inventory of concrete nested shapes, or default ABI |
| [#733] citations | A + B + host/sibling-builtin registries + external semantic/target registries + effect-disposition registry + exported-stdlib dependency closure | Table A and the sibling builtin registry bind every semantic cell to its controlling atom; their backend products add typed implementation or rejection authority without replacing that semantic binding. The host domain, external-family `SemanticRegistration` registries, external target dispositions, effect dispositions, and derived stdlib dependencies carry their corresponding typed authorities. The pinned Buoy policy and Chelis shell adapter check authority, freshness, and selected-surface completeness |

## Delivery split and oracles

Phase 4 deliberately separates authoring from generated consumption:

- **4B - decided-contract and schema freeze.** Author the numbered-spec atoms
  in this slice and freeze the numeric, sibling-builtin, host primitive-leaf,
  host-constructor, and
  external-family routing schemas plus the exported-stdlib effect-disposition
  registry before machine population. The authoritative
  oracle is `.venv/bin/python scripts/dtype_phase4b_oracle.py`, whose success
  line is `DTYPE PHASE 4B ORACLE: PASS`.
- **Pre-4C - exact builtin-atom closure ([#1294]).** Introduce the closed
  builtin domain/case declaration types, attach a non-empty exhaustive
  declaration to every `BuiltinDecl`, discover every declared domain and
  finite case, author every missing exact
  `[05-OP-N]` authority, and prove a total bijection for every Table-A and
  sibling-builtin identity. The authoritative oracle is
  `.venv/bin/python scripts/dtype_builtin_atom_closure_oracle.py`, whose
  success line is `DTYPE BUILTIN ATOM CLOSURE ORACLE: PASS`. It admits no
  count allowlist, unnumbered authority, issue citation, default, alias, age,
  or compatibility exception. It must be green and merged before any Phase
  4C key/cell type, macro, or row lands. The domain and case declarations
  themselves are [#1294] prerequisite artifacts; they are discovery metadata,
  not Table-A/Table-B cells.
- **Pre-4C - composite executable gate ([#1296]).** Wire the exact child
  behavior, storage, census, and atom-closure oracles into
  `.venv/bin/python scripts/dtype_pre_phase4c_oracle.py`. Its success line is
  `DTYPE PRE-PHASE-4C ORACLE: PASS`; a missing, duplicate, skipped, stale,
  nonzero, or success-line-free leg fails. This runner is part of the normal
  gate and is the only Phase 4C entry receipt. Per the 2026-09-01 amendment in
  `dtype_semantics.md`, a leg passes on executable `eval`/`c-host`/`c-dag`
  behavior plus a typed `Unimplemented { issue }` receipt for every unbuilt
  `hip` or `metal` cell; device execution is not an entry condition, and that
  receipt is the disposition this document already assigns to current backend
  capacity.
- **4C - machine authority population.** Add the closed key/cell types and
  compact authoring macros in `chelis-types`; expand them into a complete machine
  table; consume every already-exhaustive `BuiltinDecl` domain/case declaration;
  populate Tables A/B, the host primitive-leaf and constructor tables, sibling
  container/boundary
  registries, external target dispositions, and exact effect-disposition
  rows; and consume [#1294]'s already complete atom membership. Compact macros are an
  authoring convenience only: duplicate or missing expanded rows fail. The
  authoritative 4C oracle is
  `.venv/bin/python scripts/dtype_phase4c_oracle.py`,
  whose success line is `DTYPE PHASE 4C ORACLE: PASS`.
- **4D - generated consumers.** Derive checker acceptance/reporting, build
  gates, backend dispatch skeletons, root-realizability projections, checked
  host casts, recursive host-ABI resolution, effect-policy lookup, and
  exported-stdlib dependency closure. Delete the hand-authored
  mirrors only when the corresponding generated consumer is live. The
  authoritative 4D oracle is
  `.venv/bin/python scripts/dtype_phase4d_oracle.py`,
  whose success line is `DTYPE PHASE 4D ORACLE: PASS`.
- **4E - conformance.** Generate positive and negative execution cases over
  the complete supported/implemented and rejected/unimplemented products,
  including every primitive ABI leaf, host-constructor nesting, external
  callables, effect requirements,
  and exported stdlib definitions. The authoritative Phase 4 oracle is
  `.venv/bin/python scripts/dtype_phase4_oracle.py`, whose success line is
  `DTYPE PHASE 4 ORACLE: PASS`; it invokes the 4B, 4C, and 4D oracles and the
  4E suite. Phase 4 is not complete until this final oracle passes and the
  required fresh-context red team validates the generated surface.

## Seed dispositions the table must ship with

The audited cells are listed so Phase 4 cannot ship around them. Normative
atoms decide the language rule; these rows record the initial machine
dispositions and the issue that owns any unimplemented backend cell.

| cell | frozen disposition |
|---|---|
| `mean` x Tensor x int widths ([#724]) | DECIDED (2026-07, on the issue) and LANDED on main 2026-08-04 (validated at 013b947d, all three lanes citing the issue in `crates/chelis-cli/tests/reduction_and_bitwise_matrix.rs`; [#724] closed. The decision is release-visible at v0.19 per the roadmap's anti-churn invariant 4): `Rejected` at check time on every application form - integer mean requires an explicit cast (`mean(cast(x, f32))`) or `floor_div(sum(x), n)`; `mean`'s sig is float-only (bool rejects with the integers). Scope: only fractional-producing reductions reject; `sum`/`max`/`min`/`prod` over integers stay valid. Phase 4C records the cell |
| arithmetic operators and arithmetic reductions x (any) x bool ([04-NUM-4], [#726]) | `Rejected` at check time on every application form. This includes `add`, `sub`, `mul`, `div`, `neg`, `floor_div`, `trunc_div`, `sum`, `mean`, and `prod_reduce`; no pre-existing disposition or compatibility exception survives. Diagnostics point to `and`/`or`/`not` for logic and [05-OP-29] `count` for bool-tensor counting. They do not prescribe an explicit cast or arithmetic lowering |
| `count` x Tensor x bool ([05-OP-29], [#1287], [#1291]) | Table A is `Supported` for the bool-tensor domain, with `i64` result and structural `grad` rejection; scalar, numeric, string, and deferred operands are `Rejected`. The named signature rule accepts [05-OP-29]'s unique static axis sets and rejects an invalid application before any Table-B lookup; axis values are not table axes. Eval, C-host, and C-DAG cells are `Implemented`, delivered by [#1287] with `scripts/dtype_count_oracle.py` as the executable core receipt. HIP and Metal cells are `Implemented { kernel_id }` by the dedicated device Count kernels, which `scripts/dtype_count_device_oracle.py` proves structurally; those two cells still owe [#1291]'s real-hardware numerical gates, which remain that issue's closing condition, and no host fallback or stub counts as implementation |
| `and`/`or`/`not`, comparison, and `where` ([05-OP-26..28], [05-OP-53], [05-OP-36], [#1284]) | Logical semantic cells are `Supported` exactly under their bool-only truth tables. The five ordered identities `cmplt`/`lt`/`gt`/`gte`/`lte` occupy numeric scalar/tensor rows. `eq`/`neq` additionally occupy bool scalar/tensor cases and sibling cases for string, unit, List, tuple, Dict, Option, and ADT under [05-OP-36]'s recursive equality domain; they are not mislabeled numeric ordered rows. [05-OP-53] governs `where`: a bool condition selects same-shape, same-dtype branches by stored bits. Mixed types/surfaces/shapes are rejected by the exact signature rule. Numeric or mask-arithmetic aliases are not `Implemented` receipts. Eval and compiled C implement the direct identities across their admitted surfaces. HIP implements numeric tensor comparisons, Bool8 equality/logic, and raw stored-bit tensor `where` across every active device dtype; `scripts/dtype_nonnumeric_lowering_oracle.py` is the always-run receipt and the ignored exact-bit hardware matrix is registered in `docs/manual_gates.md`. Metal retains `Unimplemented { issue: #2266, diagnostic_kind: UnsupportedFeature }` with a stable typed rejection until exact kernels land |
| scalar `relu`/`sigmoid`/`silu`/`gelu`/`tanh` ([#712], [#704]) | DECIDED (2026-07, on the issue) and IMPLEMENTED by the 2026-08-05 pre-table scalar remediation: `Supported` on float dtypes at BOTH surfaces (a scalar is a rank-0 tensor); non-float is a check-time domain `Rejected`, never a silent 0. Eval and C now execute every active float width through dtype-aware kernels/helpers, and the always-run scalar matrix requires byte-identical observation. Phase 4C records the rows, 4D replaces the hand registration, and 4E replaces the curated matrix |
| `relu` adjoint x active floats ([05-OP-43], [#1313]) | Table A is `Supported` on every active float surface. The dedicated identity and adjoint are implemented by the sealed evaluator and compiled C host/DAG lanes at f16/bf16/f32/f64; the adjoint selects the complete cotangent only where `0 < x` and constructs exact positive zero at both signed zeros and NaN. HIP code generation implements all four widths (raw stored-bit predicates for f16/bf16), and Metal implements its [04-TGT-1]-admitted f32/f16/bf16 widths; Metal f64 remains the deliberate target rejection, not an unimplemented ReLU cell. `.venv/bin/python scripts/dtype_relu_oracle.py` is the always-run exact identity/value oracle; the ignored HIP exact-bit execution case runs through `scripts/hip_test.py`, while [#737] owns the general macOS Metal execution-evidence boundary. No backend cell cites [#1313] after it closes, and the pre-amendment first-operand tie gradient is not an implementation receipt |
| `stop_gradient` x every checked type ([05-OP-42], [#1312]) | Table A is `Supported` as the identity for exactly one value of any checked type; a `SurfaceClass` split does not apply and the declaration selects the sibling `Container`/`Boundary` machinery only if Phase 4C finds the identity domain needs it. The differentiation barrier (no traversal of the argument subgraph for adjoints or structural rejection; shape-preserving zero cotangent) is the whole semantic content. Every backend cell is `Unimplemented { issue: #1312, diagnostic_kind: UnsupportedFeature }` until the barrier oracle, including the straight-through-estimator composition and the bare-`round` negative control, is green |
| `io/json::json_bigint` and the `JsonBigInt(string)` variant ([05-OP-2..5], [05-OP-34..35], [#1314]) | Stdlib-family rows, not Table-A cells: the variant and accessor bind to [05-OP-34]/[05-OP-35] (the manifest is eighty-four definitions), ingestion to [05-OP-2], projection to [05-OP-3], construction to [05-OP-4], and canonical-form serialization to [05-OP-5]. The census semantic registration lands with [#1314]'s implementing change set per the AGENTS.md discipline; until then discovery finds no `json_bigint` row and the manifest names the final contract |
| scalar `tan`/`atan`/`recip` ([#704]) | `Supported` by the controlling `spec/05-risc-primitives.md` §2.2 scalar-unary rule on f32, f64, f16, and bf16 and `Rejected` on every non-float dtype. In particular `recip(0)` is IEEE infinity, not [04-NUM-9] `DivZero`. Eval and C implement every admitted width without a silent-zero fallback. Phase 4C records these exact Table-A rows against that controlling rule, 4D derives the consumers, and 4E replaces the curated matrix |
| scalar `floor`/`ceil`/`round` x int widths ([#715]'s rows) | DECIDED (2026-07, via [#712]'s comment) and IMPLEMENTED by the 2026-08-05 pre-table scalar remediation: `Supported` as an exact identity at i8/i16/i32/i64 in checker, eval, and C, with no float conversion. Phase 4C records the cells and 4D derives their consumers |
| `max_elem`/`min_elem` x Scalar/Tensor x active numeric dtypes ([05-OP-40], [#715], [#1306]) | Table A is `Supported` exactly under [05-OP-40]'s stored-bit selection and adjoint rule. Earlier ordinary-value scalar coverage does not prove NaN payload, signed-zero, equality, signed-minimum, tensor, or device behavior. Every backend cell is `Unimplemented { issue: #1306, diagnostic_kind: UnsupportedFeature }` until the exact extrema oracle is green; arithmetic negation, target-native `fmax`/`fmin` semantics, host fallback, and partial-width support are not implementation receipts |
| `sub` x Scalar/Tensor x active numeric dtypes ([05-OP-41], [#1306]) | Table A is `Supported` exactly under direct checked subtraction. Every backend cell is `Unimplemented { issue: #1306, diagnostic_kind: UnsupportedFeature }` until the cross-surface oracle proves representable signed-minimum cases, true overflow, and declared-width float behavior. `add(a, neg(b))` is not an implementation receipt |
| C-DAG x i64 x `max_elem`/`abs` etc. ([#691]) | B-cells `Unimplemented { issue: #691 }` until integer kernels land - the fmaxf/fabsf substitution becomes a rejection. Status 2026-08-04: the direct DAG integer path is repaired ([#729] Phase 3 dispatches integer min/max/abs through exact checked integer paths, per the roadmap's unclaimed-issue ledger), and [#691] CLOSED 2026-08-04 once PR #1164 rehomed the emitter citations and released the [#730] rejection-authority liveness pin. That closure leaves a live obligation on this row rather than settling it: a `Unimplemented { issue: #691 }` B-cell now cites a CLOSED owner, which is the exact state the pin exists to prevent, so any surviving C-DAG integer cell is re-cited to an open owner (or dispositioned `Implemented` where Phase 3's exact integer dispatch already covers it) before Phase 4 mechanizes this row. Resolved 2026-09-01: no live cell cites [#691] - the generated `REGISTERED_OPEN_ISSUES` carries no 691 entry and the surviving source references are historical comments - so this row reserves no closed-issue citation and asserts no disposition of its own. Its cells are governed by the live rows that own them: `max_elem`/`min_elem` at every active numeric dtype, i64 included, by the [05-OP-40]/[#1306] row above, which holds them `Unimplemented { issue: #1306 }` until the exact extrema oracle is green; and `abs` by the [#722]/[#699] row below, whose compiled C `abs` receipt PR #1065 already records. Phase 3's exact integer dispatch is repair evidence, not a Table-B receipt for either |
| Metal x i64 x `abs` ([#693]/[#699]) | A is `Supported`; Metal B-cell `Implemented` once [#699]'s raise lands and the MSL integer path is wired; until then `Unimplemented { issue: #693 }` |
| `abs`/`floor`/`ceil`/`round` x Tensor x int widths x compiled backends, grad path included ([#722]/[#699]) | A is `Supported`: `abs` is exact at the declared width and traps at the signed minimum per [04-NUM-9], and `floor`/`ceil`/`round` are the identity on an already-integral value. The B-cells are the seed decision, and [#699]'s `Const { value: 0.0 }` placeholder satisfies neither disposition - a fabricated zero is a third state the schema does not admit, which is exactly what [#722] measured. Because `grad` is built over the lowered DAG, the placeholder reached the reference lane too and both lanes agreed on all-zero gradients, so no cross-lane oracle could see it; a cell's atom owns the adjoint and no Table-B cell may invent one. Status 2026-08-04: the eval integer unary rows and the compiled C `abs` row landed (PR #1065, exact signed widths and MIN traps); the remaining compiled unary cells stay `Unimplemented { issue: #722 }` under [#699] |
| `uniform_like` and `dropout` x Tensor x active float widths ([05-OP-8], [05-OP-37], [#937], [#1295]) | Table A is `Supported` for every active float dtype with same-dtype bounds/rate. [05-OP-8] owns validation, exact f64/f32 FMA behavior, f16/bf16 exact widening into the f32 arithmetic domain and one final narrow, RNG ordinal consumption, and pathwise bound adjoints. [05-OP-37] owns the exact inverted-dropout graph, saved mask, and pathwise adjoint. #937 delivered the earlier f64 template/output-width repair but does not authorize an f32-bound signature or any remaining cell. Every backend cell is `Unimplemented { issue: #1295, diagnostic_kind: UnsupportedFeature }` until the all-active-dtype oracle proves the complete signature domain; a pre-existing partial lane is not `Implemented` |
| `bitand`/`bitor`/`bitxor`/`shl`/`shr` x Scalar x int widths ([#682]) | `Supported`; C B-cells `Implemented`; shifts use the width-bounded unsigned helpers required by [04-NUM-13], never raw signed C shifts |
| sibling `Boundary` cases for scalar, tensor, and recursive-List `to_string` ([05-OP-25], [#1059], [#1282]) | The exact case enumerator is the closed set scalar, tensor, and List. A List recursively rejects any element outside that set. Unit, tuple, `Dict`, `Option`, ADT, function, deferred, and resource cases are semantic `Rejected` rows, not implementation gaps. [#1282] owns checker/evaluator alignment over exactly the admitted and rejected cases. [#1059] owns compiled C-host Tensor/List rendering only; it does not authorize another language case. Each backend cell retains its own exact disposition and owner until its acceptance oracle is green |
| runtime-axis `shape`, `ReduceWindow`, and `ReduceWindowGrad` ([05-OP-7], [05-SHAPE-1], [05-OP-39], [#1298]) | Their Table-A signatures are target-independent: `shape` admits literal or computed i32 axes and the window operations retain their complete reducer/rank/dtype/AD domains. A backend implementing only constant axes, selected ranks, or first-order adjoints leaves the whole B-cell `Unimplemented { issue: #1298, diagnostic_kind: UnsupportedFeature }`; no host fallback, permanent target rejection, zero adjoint, or signature narrowing is an implementation receipt |
| host numeric builtins `tensor_scan`, `process_run`, and generic assertions ([05-OP-38], [#1297]) | Their numeric/sibling semantic cases are `Supported` exactly under [05-OP-38]. Compiled-host backend and effect cells remain `Unimplemented { issue: #1297, diagnostic_kind: UnsupportedFeature }` until exact execution is green. Eval-only routing, whole-module assertion rejection, dtype-named assertion aliases, inert behavior, and device capacity presented as language illegality are forbidden |
| `wrap_add`/`wrap_sub`/`wrap_mul` x (both surfaces) x int widths ([05-OP-17..19], [#753]) | A `Supported` on i8/16/32/64 and `Rejected` on bool/float. Every backend B-cell is `Unimplemented { issue: #753, diagnostic_kind: UnsupportedFeature }` until its exact-width modular kernels land; SMT lowers to `bvadd`/`bvsub`/`bvmul` exactly, with no tolerance row |
| `mean`/`max_reduce`/`min_reduce`/`argmax_reduce`/`argmin_reduce` x admitted tensor dtypes ([05-OP-11..13], [05-OP-15..16], [#898], [#1281]) | A follows the exact dtype, result, empty-axis, order, NaN, infinity, tie, and AD contracts in the atoms. A pre-existing partial kernel is not an `Implemented` cell. Until the behavior work lands, each affected backend cell is `Unimplemented { issue: #1281, diagnostic_kind: UnsupportedFeature }`; #898 closes only the spec gap and never becomes a Table-B authority |
| `prod_reduce` x admitted tensor dtypes ([05-OP-14], [#1290], [#170], [#898]) | A follows [05-OP-14]'s exact dtype, empty identity, canonical adjacent-pair balanced tree, overflow, and reverse-tree AD contract. Affected backend cells are `Unimplemented { issue: #1290, diagnostic_kind: UnsupportedFeature }`; #1290 is also part of #170. The more specific HIP wrong-dtype cells use [#689]. #898 closes only the earlier spec gap |
| `is_nan`/`is_finite`/`is_infinite` x both surfaces x active floats ([05-OP-20..22], [#965]) | A `Supported` for f16/bf16/f32/f64 and `Rejected` for every other Prim. B-cells are `Unimplemented { issue: #965, diagnostic_kind: UnsupportedFeature }` until stored-width classification exists in each lane |
| named lossy casts x source/target dtypes x both surfaces ([05-OP-6], [05-OP-23..24], [#759]) | `cast_trunc` is the float-to-integer truncating rung; `cast_saturate` admits active signed-integer/float sources and signed-integer targets; `cast_wrap` admits signed-integer source/target pairs. All reject bool and deferred dtypes. There is no `cast_round` cell: programs compose `round` with checked `cast`. Table B records each lane as implemented or as `Unimplemented { issue: #759, diagnostic_kind: UnsupportedFeature }` without weakening the Table-A rule |
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

The requirement is structural per family. Table A remains the numeric
language-builtin registry and the sibling registry owns container/boundary
builtins (checker acceptance is derived from both at Phase 4). Runtime exports
and exported stdlib defs use the §C6
operation-semantic registry delivered with the capacity tripwire:
each structurally discovered numeric callable's exact canonical
identity is a key whose value is one exact `[05-OP-N]` authority. The
registry validator requires chapter `05`, group `OP`, and verbatim atom
existence as a normative line beginning `> **[05-OP-N]**`; it does not
accept a free-text `spec/05` substring, a cross-reference,
`[05-OBS-1]`, or an absent `[05-OP-999]`. Every numeric callable in the
discovered baseline and every later addition has one exact semantic
registration. There is no grandfather, permanent-disposition, successor-
override, or integer-plumbing exception. A callable without a governing atom
authors a new `[05-OP-N]` normative atom in spec/05 and adds its exact registry
mapping in the same change set. The
deferred PyO3 leg must deliver the same identity-to-authority shape
before Phase 1 entry. Table A's
(`BuiltinId`, `SurfaceClass`, operand `Prim`, `SemanticParams`) key
deliberately does NOT stretch to those families: runtime exports and
PyO3 functions have no `BuiltinId`. Container/boundary builtins use the
frozen sibling registry above because they have no scalar/tensor surface;
external-family callables continue to use their §C6 semantic registry plus
the external execution authority defined above (PR #950 red team P1-2).

Numeric-ness is signature-derived: a callable whose signature mentions
a numeric dtype requires a registry entry, and the non-numeric
classification is available only for genuinely dtype-free surface.
For the C family this is deliberately conservative: every
non-boolean/non-character built-in arithmetic value type, including bare
`int` and the pointer-sized integer spellings, yields `numeric-op` - and
a spelling the census does not recognize at all is a build failure rather
than a dtype-free row, which is what makes "conservative" true rather
than aspirational (`dtype_semantics.md` §C6, the inverted type-word
rule). The checked-in pre-ratchet census still records exactly three legacy
integer-plumbing dispositions - `chelis_alloc`,
`chelis_tensor_from_value_list_typed`, and `chelis_dtype_size` - solely as
deletion debt owned by [#1288]. They confer no final authority and no
successor may inherit them. Their final numeric forms are exact [05-OP-31] or
[05-OP-33] identities and require ordinary semantic registration. A callable
name, parameter name, or substring is never an exemption. Conditional macro
definitions are propagated across
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

Table A and sibling semantic registration decide LANGUAGE legality - what is
legal in Surf, Deep, and the RISC DAG, target-independently, reported by the
checker.
The other family semantic registries record which existing normative decision
controls each discovered callable; they do not create language semantics.
Per-backend executability is Table B's, the sibling backend product's, the
external target disposition's, the effect-disposition registry's, or the
exported-stdlib dependency closure's separate decision,
reported at build through [#730]'s `Unsupported` channel where the
target is known: a language-legal op a backend cannot run is a
CAPABILITY rejection, never a checker type error (§Derivations owns
this split; restated here because a new-op author is the person most
tempted to collapse it).

`round_to` is the resolved worked case: [05-OP-1] admits every active float
dtype, rounds the exact binary value at the declared operand width, and
finalizes once to that same dtype. A Table-A row or backend implementation
that widens through f64, narrows the signature, or omits one admitted float is
nonconforming; there is no unresolved semantic choice or compatibility cell.

## Frozen schema decisions

1. The machine form is const Rust in `chelis-types`. Compact Rust macros
   author dtype classes and parameter products, but expansion produces the
   complete checked row set. A checked-in editable data file is not an
   authority.
2. `c-host` and `c-dag` remain distinct Table-B backends alongside `eval`,
   `hip`, and `metal`.
3. `SignatureRuleId`, `ResultDtypeRule`, `ResultTypeRuleId`, and
   `SiblingCaseId` are closed typed identifiers. Shape and bounded
   runtime-parameter checks live in the named signature rule; they are not
   strings and do not create unbounded table axes.
4. Row count is managed only at the authoring layer. The machine product is
   exhaustive, and expansion rejects missing or duplicate cells rather than
   filtering them.
5. Table A's `SurfaceClass` is exactly scalar or tensor. Container and
   boundary builtins use the exact sibling registry defined above, selected by
   `BuiltinDecl`; host ABI recursion uses the companion constructor table.
   Runtime, stdlib, and binding callables remain in §C6's per-family
   `SemanticRegistration` registries. Runtime and binding callables also use
   the external target-disposition registry; exported stdlib executability is
   derived transitively from its checked body. An overloaded builtin declares
   every domain it occupies.
6. Enumerable identity inventories live as normative registry files under
   `spec/registry/`, one per owning `[05-OP-N]` atom, identity-keyed with no
   semantic ordinals; the atom incorporates its registry by reference and
   keeps the semantic rules, and neither duplicates the other. These files
   are numbered-spec-tier content under the AGENTS.md Documentation Authority
   rules and are frozen by the owning guard oracles, so they are not the
   "editable data file" decision 1 prohibits. Phase 4C's const-Rust machine
   form is validated byte-wise against them (the [05-OBS-3] generated-table
   pattern); the registry is the reviewed source and the machine form is the
   executable authority, and a disagreement fails construction rather than
   selecting either silently.

`pad_sequences`/`pad_sequences_to` ([#1009]) are the completed backfill case:
they predate the §C6 semantic-registration ratchet, and now bind their exact
callable identities to [05-OP-9]/[05-OP-10]. Their builtin applications enter
the sibling `Container` registry at Phase 4 under those atoms, with separate
finite cases and dtype-valued `SemanticParams`; their exported identities stay
in the §C6 family registry. Neither path is a grandfathered exemption.

[#170]: https://github.com/Chelis-Lang/chelis/issues/170
[#174]: https://github.com/Chelis-Lang/chelis/issues/174
[#682]: https://github.com/Chelis-Lang/chelis/issues/682
[#689]: https://github.com/Chelis-Lang/chelis/issues/689
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
[#898]: https://github.com/Chelis-Lang/chelis/issues/898
[#908]: https://github.com/Chelis-Lang/chelis/issues/908
[#912]: https://github.com/Chelis-Lang/chelis/issues/912
[#937]: https://github.com/Chelis-Lang/chelis/issues/937
[#955]: https://github.com/Chelis-Lang/chelis/issues/955
[#965]: https://github.com/Chelis-Lang/chelis/issues/965
[#1009]: https://github.com/Chelis-Lang/chelis/issues/1009
[#1150]: https://github.com/Chelis-Lang/chelis/issues/1150
[#1152]: https://github.com/Chelis-Lang/chelis/issues/1152
[#717]: https://github.com/Chelis-Lang/chelis/issues/717
[#1281]: https://github.com/Chelis-Lang/chelis/issues/1281
[#1282]: https://github.com/Chelis-Lang/chelis/issues/1282
[#1284]: https://github.com/Chelis-Lang/chelis/issues/1284
[#1294]: https://github.com/Chelis-Lang/chelis/issues/1294
[#1295]: https://github.com/Chelis-Lang/chelis/issues/1295
[#1296]: https://github.com/Chelis-Lang/chelis/issues/1296
[#1297]: https://github.com/Chelis-Lang/chelis/issues/1297
[#1298]: https://github.com/Chelis-Lang/chelis/issues/1298
[#1306]: https://github.com/Chelis-Lang/chelis/issues/1306
[#1312]: https://github.com/Chelis-Lang/chelis/issues/1312
[#1313]: https://github.com/Chelis-Lang/chelis/issues/1313
[#1314]: https://github.com/Chelis-Lang/chelis/issues/1314
