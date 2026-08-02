# Grounded Dtype Semantics

**Status:** Active phased plan. Phase 0, the PR #956 §C6 covered-family
tripwire, and Phase 1's typed wire-schema, registered-PyO3 entry legs, and
sealed dtype-semantics layer have landed through PRs #1033 and #1049. Draft
PR #1054 consolidates the complete Phase 2 kernel split, exact prover
carriers, trap freeze, integer-unary evaluator rows, and the ordinary,
windowed, argument-reduction, and overlapping window-adjoint consumers
discovered by its red team. Its single Phase 2 oracle is
`.venv/bin/python scripts/dtype_phase2_oracle.py`. Draft PR #1065 begins
Phase 3 with the exact, minimum-trapping signed-integer `abs` kernel in the C
tensor and scalar-host lanes. General fused integer kernels and the remaining
C rows mean Phase 3 is not complete, and Phase 4 has not started. Tracking
issue: [#729].
**Owning specs:** `spec/04-type-system.md` (gains an authored overflow/rounding
section, today silent), `spec/05-risc-primitives.md` (op result semantics),
and the audit record in `docs/investigations/numeric_audit_next_sweeps.md` /
`docs/investigations/numeric_audit_structural_prevention.md`.
**Class fixed:** [#727] (no dtype's semantics are enforced at any single
point), subsuming [#695] (the integer instance). Sibling classes [#703],
[#709] have their own plans (`spec/design/loud_unsupported.md` [#730],
`spec/design/checker_totality.md` [#731]); [#728]'s plan is
`spec/design/faithful_observation.md` ([#732]), which this plan's
Phase 3 inherits or delivers depending on landing order (its §I1).

## Summary

Chelis has no component that owns what a dtype *means*. Rounding, width,
overflow, value domain, comparison, and formatting are decided independently
at every op x lane x surface site - and the 2026-07 numeric audit measured
the result: forty-plus verified defects across every dtype family, with each
lane correct exactly where another is wrong, and three prior rounds of
correct-but-local fixes that did not stop the class ([#695]'s history).

This proposal introduces one: a **dtype semantics layer** that is the single
source of truth for per-dtype behavior, made *unavoidable* rather than
available. The mechanism is not "a helper everyone should call" - the audit
found four correct helpers sitting uncalled next to their bug sites - but
representation types whose constructors are private, so that producing a
numeric value without passing through the semantics is a compile error.

The work is split into five phases. **Each phase is specified below as a
contract**: what it inherits from the previous phase, what it must deliver,
what is *frozen* at its exit (the parts the next phase is allowed to build
on without re-checking), what is explicitly not its job, and the one oracle
that decides whether it is done. Someone picking up Phase N should be able
to work from §"Phase N" plus the normative contracts in §C1-§C5 alone.

## Why a refactor and not another patch

Recorded so the next reader does not have to re-derive it:

1. **The patch strategy has a measured failure history.** Round 1
   (`TensorElement`, PRs #64/#67/#72) built the right abstraction and
   deferred two dtypes by explicit decision. Round 2 (`checked_int_binop`,
   [#387]) built the right helper and wired 4 of ~17 call sites. Round 3's
   audit found `tensor_float_unop_f32`, `convert_cast_data`, and
   `fold_static_size`'s `checked_i64` - each correct, each optional, each
   with siblings that do not call it. Optionality is the root cause, and
   optionality is the one thing a patch cannot remove.
2. **There is no reference lane to patch toward.** eval-scalar rounds f16
   correctly while eval-tensor does not ([#717]); C-tensor wraps int8 while
   C-scalar does not ([#718]); prove's interpreter collapses what the SMT
   tier keeps exact ([#688]). "Make X match Y" is undefined when no Y holds
   the semantics.
3. **Dtype discipline is the language's stated value proposition.** The
   spec's differentiators - no implicit precision promotion, explicit
   casts, named dimensions - are precision-centric promises. A numeric
   layer with no grounded notion of `f16` or `int8` contradicts the
   product's own core claim.

## Non-goals

- **Not** the fix for [#703] (unsupported cases must fail loudly - dispatch
  and fallback discipline, `numeric_audit_structural_prevention.md` item 3)
  or [#709] (checker holes - item 5). Those are cheaper, independent, and
  must not wait for this.
- **Not** a numerics-accuracy project. Transcendental ulp bounds ([#719]'s
  vvsqrtf, SLEEF vs libm differences) get a documented per-op tolerance
  table as part of the formatting/oracle contract (§C4), not new kernels.
- **Not** a general change to surface syntax or user-facing checker rules.
  Phase 1 authors and enforces the already-decided integer-`mean` [#724]
  and bool-arithmetic [#726] cells through [#860]'s operand-dtype
  chokepoint; Phase 4 owns their table-derived mechanization, not their
  authorship. Other checker policy remains out of scope.

## Vocabulary

Used precisely throughout:

- **Lane** - an execution path from source to a value: the evaluator
  (`chelis eval`), the compiled C binary, HIP, Metal, prove's interpreter.
- **Surface** - scalar vs tensor within a lane (the audit showed these
  diverge *within* lanes: [#718]).
- **Cell** - one (op, dtype, lane, surface) combination. The unit of both
  bugs and tests.
- **Control** - a test locking a cell that is correct today. Controls are
  load-invariant across all phases: no phase may change a control's
  expected value (§B2).
- **Finalize** - apply a dtype's rounding/width/domain rule to a wide
  intermediate, or trap. The central verb of this design.

---

# Part I - the normative contracts (§C1-§C5)

These sections are the *interfaces between phases*. Phase 1 implements
§C1-§C4; later phases consume them and may not reinterpret them. Any change
to a frozen contract after its freeze point (§B1) requires editing THIS
document and the tracking issue in the same change set - never a silent
drift in code.

## C1. Per-dtype value semantics

**Normative home: `spec/04-type-system.md` §9.1**, the consolidated
per-dtype table, backed cell by cell by atoms [04-NUM-1] through
[04-NUM-14]. That table is what a later reader cites and is the only
normative copy; this section is the implementation-facing elaboration.
Where the two disagree the spec wins and this section has a bug.

The rows below are reproduced for convenience while this plan's phases are
in flight, and are edited in the same change set as §9.1 or not at all:

| dtype | value set | arithmetic width | finalize(wide) | overflow / out of range | special values |
|---|---|---|---|---|---|
| `f64` | IEEE binary64 | f64 | identity | n/a (IEEE handles it) | NaN/±inf/-0.0 preserved |
| `f32` | IEEE binary32 | f32 | round-to-nearest-even to 24-bit mantissa | rounds to ±inf per IEEE | NaN preserved (quiet), ±inf, -0.0 preserved |
| `f16` | IEEE binary16 | f32 | RNE to 11-bit mantissa, incl. subnormals | overflow -> ±inf (locked: `mul(65504f16, 2f16) = inf`) | as f32 |
| `bf16` | bfloat16 | f32 | RNE to 8-bit mantissa | overflow -> ±inf | as f32 |
| `int64` | integers in [-2^63, 2^63-1] | exact int64 | must be integral and in range, else **trap** | **trap** (`Overflow` out of range, `Domain` non-integral, [04-NUM-9]) | none |
| `int32/16/8` | integers at width | exact at width | same rule at width | **trap** (`Overflow` / `Domain` at width) | none |
| `bool` | {0, 1} | n/a (not arithmetic) | must be exactly 0 or 1, else **trap** (`NumericTrap::Domain`) | trap | none |
| deferred names (spec §1.1.1) | rejected by the checker | - | unreachable: `finalize` for them is a compile-time-visible `Rejected` row in the capability table, not a runtime arm | - | - |

Normative notes, each pinned by an existing test:

1. **Kernels compute at the dtype's arithmetic width and finalize ONCE
   per op** (spec/04 [04-NUM-8]). This note previously read "kernels
   therefore compute float ops in f64", derived from a permissive clause
   in [04-NUM-2] that was removed on 2026-07-28; f64 computation of an
   f32 op is now non-conforming. The narrow-float rows compute at f32,
   which yields the correctly-rounded narrow result for the basic
   operations by construction (f32 carries >= 2p+2 bits for p <= 11) and
   is what every target's hardware does. Chained ops still finalize
   per-op: the eval scalar lane's `add(add(2048f16,1f16),1f16) = 2048`
   lock (`narrow_dtype_matrix.rs::eval_scalar_f16_rounds_per_op`) is the
   sequential-rounding contract and is unaffected.
2. **The f64 collision stays.** `f64 add(2^53, 1) == 2^53` is CORRECT
   IEEE behavior and must not change
   (`precision_matrix.rs::f64_add_at_mantissa_boundary_is_correctly_lossy`).
   The identical numbers at int64 trap or stay exact - never silently
   collapse.
3. **Integer overflow traps at every width in every lane** ([#680]'s decided
   contract). Today int8/16/32 wrap in eval and int64 saturates, and the
   compiled lane does neither ([#718]) - all four behaviors are replaced by
   the trap. In-range arithmetic is exact and must not trap
   (negative-parity locks exist: `int8_add_just_below_overflow_does_not_trap`).
4. **Comparisons compare finalized values.** `lt(cast(2048.0,f16),
   cast(2049.0,f16))` is `false` because both casts finalize to 2048
   before comparing. This kills the wrong-branch family ([#680]'s 2^53 case,
   [#714]'s 2049 case, [#720]'s fold) in one rule.
5. **Integer division stays authored as-is**: `div` on integers rejected
   by the checker; `floor_div`/`trunc_div`/`mod` exact via the checked
   path ([#387], verified in both lanes by `scalar_stub_matrix.rs`).

## C2. The trap contract

**Normative home: `spec/04-type-system.md` [04-NUM-9] (the closed kind set
and cross-lane identity), [04-NUM-10] (traps are values until the lane
boundary, including the device-lane error-flag shape), [04-NUM-12] (trap
OCCURRENCE for multi-step ops is defined by each lane's documented
accumulation order; trap-versus-exact at accumulator range edges is the
one permitted cross-lane trap divergence), and [04-NUM-5] (the
fold-decline rule).** Those atoms outlive this document and are what a
later reader should cite. This section is the elaboration: the Rust shape,
the working message strings, and the evidence behind the device-lane
decision. Where the two disagree the atoms win and this section has a bug.
Phase 2's trap-string freeze inherits [04-NUM-12] as a standing
constraint: the strings it freezes render a trap whose occurrence is
per-lane-order-defined, and the [#687] corpus may not paper over a
trap-versus-complete divergence that the atom's conditions permit.

One error type, one message shape, identical in every lane:

```rust
pub enum NumericTrap {
    /// Integer result outside the dtype's range.
    Overflow { op: &'static str, prim: Prim },
    /// Value not a member of the dtype's set (fractional -> int,
    /// non-0/1 -> bool).
    Domain   { op: &'static str, prim: Prim },
    /// Division/remainder by zero.
    DivZero  { op: &'static str, prim: Prim },
}
```

The `prim` on `DivZero` is required by controlling [04-NUM-9], which says
EVERY trap names the dtype. The earlier one-field design sketch was a design
bug; the numbered spec wins.

- **Message format (frozen by controlling [04-NUM-9]):** the exact three
  forms and the operation-name rule live in that atom. The operation slot is
  the canonical lowered primitive whose numeric kernel raised the trap; a
  composed operation forwards that trap unchanged, without an internal
  evaluator prefix. The exact fragments are recorded in the module as
  `pub const` and every lane emits them verbatim - the C lane via generated
  guard snippets (Phase 3), eval via the module directly. [#687]'s later
  oracle compares them byte-for-byte. This resolves [#861]'s Phase 2 decision
  without introducing source-operation provenance that no lowered lane owns.
- Traps are *values* (`Result::Err`) inside the lanes and become process
  aborts only at the lane boundary (eval: `error:` + nonzero exit;
  compiled C: stderr + nonzero exit). No lane may `panic!` for a user-input
  trap ([#692]'s rule).
- **Deciding fold behavior:** compile-time folds (`fold_static_cond`,
  const propagation) that would trap **decline to fold** - the condition
  falls to runtime, mirroring `fold_static_size`'s refusal ([#711]'s rule).
  A fold must never bake a trap away NOR bake one in.
- **GPU lanes (evidence-backed riders, 2026-07-17; provisional pending
  the HIP half, [#736]):** a GPU kernel cannot raise mid-flight, so the
  GPU trap shape is a device-side error-flag buffer (atomic flag +
  first-failing index) checked on the host after dispatch completion,
  which then raises with the same branded message. This is C2-conforming
  by construction: the completion wait IS the lane boundary, and traps
  remain values until it. The Metal spike ([#737] report) measured the
  detection as ~free for the memory-bound elementwise shape the backend
  emits (worst +0.7% median, int64 mul included) and verified MSL int64
  bit-exact. Implementation rider from the same spike: the clang
  overflow builtins are BANNED in emitted MSL (reproducible backend
  compiler crashes on `__builtin_mul_overflow(long)`; at-scale
  vectorization miscompiles false-positive the add/sub forms) - the
  hand-written checks (widening for int32, sign-bit XOR for add/sub,
  `mulhi` for int64 mul; never the division-based form, which also
  crashes the backend) are the implementation. Ratifying trap-everywhere
  for the GPU lanes stays with the [04-NUM-3] freeze decision once
  [#736] runs.

## C3. Representation and construction (the privacy contract)

**Scalars.** The existing tagged scalar (`ScalarBits`-family: exact i64
storage for ints, f64 for floats) keeps its shape. What changes is
visibility: raw variant construction becomes `pub(in dtype_semantics)`.
The public constructors are:

```rust
/// Op results. THE chokepoint: applies C1, or traps.
pub fn finalize_scalar(prim: Prim, raw: RawScalar) -> Result<ScalarValue, NumericTrap>;
/// Literals and host-boundary ingress (already-exact values;
/// still domain-checked, cannot trap for in-range input by construction).
pub fn scalar_from_i64(prim: Prim, v: i64) -> Result<ScalarValue, NumericTrap>;
pub fn scalar_from_f64(prim: Prim, v: f64) -> Result<ScalarValue, NumericTrap>;

pub enum RawScalar { Int(i64), Float(f64) }   // what kernels produce
```

**Tensors.** The element buffer becomes private and per-dtype:

```rust
pub enum TensorStorage {          // constructors pub(in dtype_semantics)
    F64(Vec<f64>), F32(Vec<f32>), F16(Vec<u16>), Bf16(Vec<u16>),
    I64(Vec<i64>), I32(Vec<i32>), I16(Vec<i16>), I8(Vec<i8>),
    Bool(Vec<u8>),                // 0/1, one byte - ends PR #79's deferral
}
/// Bulk finalize: one monomorphized loop per dtype, never per-element
/// dynamic dispatch (performance contract §C5).
pub fn finalize_tensor(prim: Prim, raw: RawTensor) -> Result<TensorValue, NumericTrap>;
```

This is the **storage decision** ([#684]/[#686]/[#685]): per-dtype buffers, not
finalize-on-write over `Vec<f64>` - f64 storage cannot represent exact
int64 above 2^53 regardless of write discipline, so it fails [#684] by
construction. The census has five declaration layers. Phase 1 lands the exact
representation atomically at four of them - `chelis-ir/src/eval.rs`
(`TensorValue`), `chelis-compiler-api/src/schema.rs` (wire schema - tensor
`data` and every numeric `ExecutionValue` scalar leaf become tagged
per-dtype payloads; this is a wire-format break, versioned as such),
`bindings/python/chelis/__init__.py` (per-dtype
tuples / numpy dtypes, ending the `np.float64` cast of [#685]), and the
layer the plan's original enumeration missed and [#856] filed: IR constant payloads
(`RiscOp::Const`/`ConstTensor`), implemented in the Phase 1 stack
(originally draft PR #857). The fifth census layer, prove's numeric env,
keeps its existing f64 map until Phase 2 under [#688]; the Phase 1 wire/type
change forced mechanical adapter edits there, and every surviving flattening
read is explicitly named `*_lossy` rather than masquerading as exact.

The constant payloads become the SEALED module types
(`ScalarValue`/`TensorStorage`), so a `RiscOp::Const` or `ConstTensor` value
cannot exist un-finalized: literals finalize once in `lower_lit` at their
desugarer-ascribed dtype (integer atoms travel their exact i64),
compiler-synthesized constants construct through
`RiscOp::synth_const`/`synth_const_tensor`, the `WireDag` carries the
dtype-tagged payloads (v4, finalize-on-decode with loud rejection of
corrupt reduced-float images), the bincode caches bumped
(`CHELIS_CTX_V6`, stdlib format 3), and constant folds decline rather
than bake a collapsed integer or a trap in. This is deliberately scoped to
the constant families, not a globally total claim about every IR field:
`RiscOp::Pad { fill: f64 }` remains a raw numeric-capacity seam tracked by
open [#878]. Partial adoption of the
storage decision is forbidden: it is the one all-layers-or-nothing
element of this plan, because a mixed state re-creates the very
boundary bugs ([#684]/[#686]) it exists to end. The mechanical
no-sixth-layer census over every remaining f64/Vec<f64> payload field
is the in-tree `crates/chelis-cli/tests/issue_729_payload_census.rs`, whose
scope is `RiscOp`/`WireRiscOp` carrier fields rather than every numeric form
in the repository.

**Normative home for the GUARANTEE this delivers:**
`spec/04-type-system.md` [04-NUM-11] - a value survives storage,
transport, and every boundary crossing at its declared dtype without
collapse. That atom is what a later reader cites; this section owns the
mechanism that achieves it (per-dtype buffers, private constructors, the
sealed types) and is free to change form as long as the atom keeps
holding.

**The four layers are a census, not a closed list (amended 2026-07-30).**
The 2026-07-30 PR sweep (the class-conflict review recorded on [#729])
found a fifth candidate layer in flight before Phase 1 had begun: a
prelude `Json` ADT whose only numeric carrier is an `f64` variant, plus
f64-only CSV accessors (PR #891, unmerged - executed there:
`9007199254740993` returns `9007199254740992.0` silently). Any such
channel that lands becomes part of this section's atomic set the moment
it merges. Phase 1 entry therefore RE-DERIVES the layer list by
enumerating every public channel that carries numeric values - eval
storage, wire schema, Python payloads, prove's env, prelude value ADTs,
published runtime-header exports - and the storage decision covers all
of them in the one change set, or Phase 1 is not entered. For a prelude
JSON/CSV surface specifically, the integer-capacity decision (a
`JInt`-shaped variant and integer accessors) ships WITH the 0.19
storage break per the roadmap's anti-churn invariants - never as a
later addition that changes a shipped prelude type twice. §C6 owns the
standing guard that keeps this census from silently growing stale.

**Access for consumers.** Reads are free-form (`as_f64_lossy()` explicitly
named lossy, `as_i64_exact() -> Option<i64>`, typed slices per dtype).
Only *construction* is gated. Movement ops (reshape/permute/shrink) that
provably preserve elements may clone/re-slice storage without
re-finalizing via a `pub(crate) fn reuse_storage` escape hatch whose doc
contract is "element-preserving ops only"; every use site cites it. That
hatch is the ONE deliberate hole, kept greppable.

**Implementation-signature notes (Phase 1 stack; same protocol as the
[#732] `ElementRef` note - the doc records the implemented form):**

- every constructor takes a leading `op: &'static str` so the C2 trap
  message can name the operation (the sketch had no op channel);
- `finalize_tensor` returns `TensorStorage` (shape stays with the
  evaluator's `TensorValue`, which wraps the storage);
- `F16`/`Bf16` buffers store `half::f16`/`half::bf16` (both
  `repr(transparent)` over the sketch's `u16`);
- the reuse hatch is implemented as the greppable `reuse_*` method family on
  `TensorStorage` (`reuse_gather`, `reuse_fill_gather`,
  `reuse_overwrite`), pub because its consumers live in `chelis-ir` and
  `chelis-compiler-api`; the "element-preserving ops only" doc contract
  and per-site citations are unchanged;
- typed read slices are implemented as the borrowed `StorageView` enum;
- the runtime's `ScalarPayload` wraps the module's sealed `ScalarValue`
  (the old in-crate `ScalarBits` enum and its wrapping `from_*_as` raw
  constructors are deleted), and tensor arguments ingress-finalize at
  their DECLARED param dtype at call binding (the host-lane mirror of
  the DAG evaluator's Load ingress).

**The cast ladder ([#759]'s one-rule-per-direction obligation, executed at
chelis#729 Phase 1).** The pre-Phase-1 design already selected the checked
finalize-or-trap rung and left truncation for a future named lossy form. The
rework replaces the split behavior in production code with that ONE authored
rule per direction, identical on every
eval surface: the shared ladder is `chelis_types::cast_raw`, consumed
by `chelis_ir::eval::convert_cast_data` / `cast_value` (the DAG
evaluator and the host tensor cast's delegation) and by the host
scalar `eval_cast` via `cast_scalar`. The CHECKED cast is the DEFAULT;
ratified at spec/04 §5.2 in the same change set:

- any source -> float target: finalize (IEEE RNE at the target width;
  overflow is the correctly signed infinity per [04-NUM-2]); total.
- integer/bool source -> integer target: exact value; out of the
  target range TRAPS `Overflow` (no wrap; int32 `300 -> int8` traps,
  formerly `44`).
- float source -> integer target: finalize only if finite and integral;
  fractional values and NaN/inf TRAP `Domain`, and integral values outside
  the target width TRAP `Overflow` (no saturation; `cast(300.0, int8)`
  traps, formerly `127`). The user spells a rounding choice first, e.g.
  `cast(floor(x), int32)` or `cast(round(x), int32)`.
- any source -> bool target: STRICT {0, 1} membership - exactly 0/1
  encodes false/true, anything else TRAPS `Domain` (`cast(2, bool)`
  traps, formerly `true`). Scalar->bool now WORKS under this rule
  (formerly a loud "unsupported cast" hole). Evidence for strict: the
  2026-07-24 corpus sweep (grep plus full-suite execution under the
  strict rule) found NO test, fixture, or example depending on the old
  nonzero-to-1 encoding; the counting idiom already casts explicitly
  (`sum(cast(x, int64))`).
- int/bool source -> float target can lose integer exactness by design while
  remaining total IEEE RNE: `cast(9007199254740993i64, f64)` yields
  `9007199254740992.0`, and `cast(16777217i32, f32)` yields `16777216.0`.
  These are explicit ByDesign controls, not a hidden f64 intermediate.
- **Fold rule (the §C2 decline clause, applied to casts):** a
  compile-time constant fold whose cast would trap DECLINES TO FOLD -
  the condition falls to runtime, where the trap fires with its full
  diagnostic (`lower.rs`'s static-`if` Cast arm).

The trap op slot is `cast` (a real op name; the former host-scalar
spelling `overflow in arithmetic at int8` is gone with the rewire, and
[#861]'s naming decision set still owns the Phase 2 freeze). The NAMED
lossy/truncating forms ([#759]) remain the future explicit escape
hatch, the same species as [#753]'s `wrap_*`: never the default,
greppable, with capability-table rows and cross-lane oracle coverage
like any other cell. The compiled C lane stays documented-divergent
until Phase 3 (issue-linked ignored rows in
`crates/chelis-cli/tests/issue_759_checked_cast_default.rs`); spelling
and atoms for the named forms land with Phase 2's kernel work; cells
ratified at Phase 4.

The 2026-08-02 dependency sweep found no executable example or fixture relying
on fractional default-cast truncation. Six test-only sites exercised or
described it; the trapping oracles now trap, and the one checker-pipeline
fixture that needed an integral intermediate now spells `floor` before
`cast`. Integral float-to-int controls remain green on both eval surfaces.

## C4. The observation contract (formatting; fixes [#728])

**Ownership note:** the authoritative elaboration and the delivery plan
for this contract is `spec/design/faithful_observation.md` ([#732]),
which is independently landable BEFORE this plan; its §I1 pins both
landing orders. The rules below are identical to its §C1 by construction -
an edit to either updates both in the same change set. This section
remains as the interface this plan's phases rely on.

One function, one output, every exit, both lanes:

```rust
/// THE printed form of one element. Used by eval's printer directly and
/// used to GENERATE the C print helper (Phase 3). No other formatting
/// path may exist for tensor/scalar payloads.
pub fn format_element(prim: Prim, value: ElementRef) -> String;
```

(Landed at [#732] Phase 1, 2026-07-20, as `chelis-types::observation`:
`ElementRef` is a `Copy` enum carrying the element at its dtype's own
width - the sketch's lifetime was dropped; a `prim`/variant mismatch
panics loudly, mirroring `ScalarPayload`'s dtype/bits invariant.)

Frozen rules (Phase 1 freezes the Rust side; Phase 3 makes C emit the
identical bytes):

1. **Integers print as integers.** `data=[750]`, never `750.0` ([#723]'s
   `.0` lie ends). int64 prints all 19 digits exactly (never via double).
2. **Floats print shortest-round-trip for their OWN width**: an f32
   element prints the shortest string that parses back to that f32 (Rust
   `{:?}` shortest-round-trip semantics - see [#732]'s §C1.3 for why this
   is `{:?}` and not `Display`); f16/bf16 print the shortest string
   round-tripping through their exact value. This replaces the C helper's `%.1f`/`%.16g`
   split and eval's f64-width formatting - it is what makes byte-equal
   lane comparison possible (the audit's `1.4142135381698608` vs
   `1.414213538169861` divergence was two formatters, one value).
3. **bool prints `true`/`false`** in tensors, matching `to_list` (ends the
   `1.0`-vs-`true` split, [#726]'s observation half).
4. **`print`, `to_list`, diagnostics, and the wire schema agree** with the
   stored bits and each other. Acceptance is literal: for every dtype,
   dump the same tensor through all exits in both lanes and diff bytes.
5. **The transcendental tolerance table.** Where lanes legitimately differ
   in VALUE (libm vs SLEEF vs vForce, > 0.5 ulp ops), the language-level
   per-op bound belongs in `spec/05-risc-primitives.md` next to the op, and
   the [#687] oracle will consult it; `sqrt` is required correctly rounded
   ([#719]) and has no tolerance row. Formatting itself never has tolerance.
   The Phase 1 `f64` `tan`/`exp` cross-lane controls use `1e-12` only as an
   implementation-chosen test margin: macOS and glibc differed by one ulp in
   the observed repros, while the defect those controls detect (computing an
   `f64` program through `f32`) differs by roughly `1e-7`. `1e-12` was an
   arbitrary separating margin, not a language decision, not an
   [05-OBS-3] tolerance row, and not authority for another operation. The
   normative tolerance table remains pending until its bounds are separately
   decided and authored in `spec/05`.
6. **Containers and scalar roots** (decided with [#732] Phase 1, identical
   to its §C1.5; ratified as [05-OBS-4]/[05-OBS-5]): a scalar-typed value
   renders as the BARE scalar at every exit in both lanes, including as a
   top-level labeled root; a rank-0 tensor renders as its single element,
   bare (`tensor(shape=[], data=[..])` is not an exit form - the [#775]
   decision). Tensor element rendering truncates after 32 elements with
   the `, ...` marker at every exit in both lanes; `to_list` and the wire
   never truncate.
7. **The root envelope is governed by [05-OBS-6].** Each emitted root renders
   as `name = value` in both lanes; when an exit emits multiple roots, their
   order is manifest entry order. The `value` half still obeys rules 1-6; this
   prefix neither reopens the frozen number grammar nor changes the
   scalar/rank-0/container decision. [#912] owns the root set and artifact
   obligation. An unavailable owed root enters [#730]'s typed failure channel
   with root, lane, and reason rather than disappearing. Full manifested-path
   acceptance is tracked by [#1023].

## C5. Kernels and the consumer map

**Kernel split (D3 of the original sketch, [#695]'s decided fix):**

```rust
// The ONLY arithmetic entry points. An integer operand reaching a float
// kernel is a type error at the call site, not a wrong answer later.
pub fn int_binop(op: IntOp, prim: Prim, a: i64, b: i64) -> Result<i64, NumericTrap>;
pub fn int_unop (op: IntUnop, prim: Prim, a: i64)       -> Result<i64, NumericTrap>;
pub fn float_binop(op: FloatOp, prim: Prim, a: f64, b: f64) -> Result<ScalarValue, NumericTrap>;
pub fn float_unop (op: FloatUnop, prim: Prim, a: f64)       -> Result<ScalarValue, NumericTrap>;
```

`IntOp`/`FloatOp` are closed enums (no strings). Dispatch from builtin
NAMES to these enums stays in the consumers and is [#703]'s territory (its
fallback must be `Err`, not a value) - this plan only guarantees that once
dispatched, the semantics are right.

**Signature note (2026-07-28, [04-NUM-8]).** The sketch above takes
`a: f64, b: f64` and `a: i64` because it was written under the removed
"compute in f64" clause. Those signatures now encode a non-conforming
contract: under [04-NUM-8] an f32 op computes at f32 and an int32 op
computes exactly at int32, so a kernel entry point that can only accept
f64/i64 cannot express the rule it is meant to enforce. Phase 2 owns the
replacement and SHALL design it against [04-NUM-8]; this doc deliberately
does not pre-specify the form (monomorphized per width, a width-parameter,
or a sealed wide-value enum are all open). What Phase 2 may NOT do is keep
the f64/i64 signatures and finalize afterwards - that is the current
behavior and is exactly what the atom forbids.

**Consumer map** - who calls what, and which audit issue each row retires:

| consumer | adopts | retires |
|---|---|---|
| eval scalar (`chelis-compiler-api/src/runtime/host_ops.rs`) | kernel split + `finalize_scalar` | [#680], [#718] eval-scalar cells |
| eval tensor (same file + `chelis-ir/src/eval.rs`) | `finalize_tensor` bulk paths; `tensor_float_unop_f32` and raw `binary_map` deleted; ordinary/window/argument reducers and the window adjoint plan ordered index groups and delegate all arithmetic/comparison/overlap accumulation to the sealed reduction kernels | [#717], [#684], [#724] eval half, [#726] eval half |
| `convert_cast_data` / tensor `cast_value` (`chelis-ir/src/eval.rs`) and host `eval_cast` | `cast_raw` / `cast_scalar`; checked target finalization with identical scalar/tensor rules | [#717] cast rows, [#720] (via next row) |
| `fold_static_cond` / const folds (`chelis-ir/src/lower.rs`) | `int_binop` + finalize in the Cast arm; decline-on-trap | [#711], [#720] |
| prove (`graph_extract.rs`, `obligation_engine.rs`, `opaque.rs`) | Phase 1 adapter state: existing f64 env with explicit `*_lossy` reads forced by the typed wire change; Phase 2 end state: exact sealed-value env and deleted flatteners | [#688] |
| C host lane (`chelis-ir/src/host.rs`, `chelis-backend-c/src/host_emit.rs`) | `parse_host_type` narrow arms (forced by exhaustive `Prim`), per-dtype scalar storage + generated trap guards | [#714], [#718] C cells, [#715]'s dtype rows |
| C emitted helpers (print, dtype switches) | GENERATED from `format_element` / exhaustive matches | [#716], [#723], [#728] |
| Metal / HIP | capability table only (already honestly typed / cleanly rejecting) | - |

**Performance contract:** finalize is per-buffer monomorphized loops (or
direct element-type compute once storage is per-dtype), never per-element
dyn dispatch; `cargo test --workspace` stays inside the ~60s inner-loop
budget; the conformance matrix's compile+run cells live in the per-crate
integration tier, not the workspace loop.

## C6. The covered-family capacity ratchet and Phase 1 entry edges (added 2026-07-30)

**Current enforcement status.** The merged PR #956 change set implements the
structurally enforced successor for the header and stdlib families:
canonical C declaration identity, matched-row metadata freeze, stdlib
ADT-shape identity AND capacity classification, non-function ABI
inventory, total linemarker attribution, a derived and recursively
walked published-header set, an INVERTED type-word rule that rejects
unrecognized arithmetic spellings rather than classifying them
dtype-free, runtime/stdlib numeric-callable authority registration,
public-header context invariance, both pre-ratchet citation sets frozen
by identity, and issue-kind-aware liveness are executable tripwires. The
typed wire-schema and registered-PyO3 legs in this change complete the
pre-Phase-1 inventory with separate generated baselines and mutation
controls. Their named commands below remain the hard Phase 1 entry evidence;
editing coverage metadata is not a substitute for making them green.

§C1-§C5 make the EXISTING numeric surface correct and make
supported-cell semantics unavoidable at op-result construction. The
2026-07-30 PR sweep (recorded on [#729]) demonstrated what they do not
cover: the class reproducing through NEW surface while the phases are
still queued - a new published-ABI `(value, int dtype)` pair, a new
f64-only prelude value channel, raw dtype-integer dispatch in the
bindings, and a new numeric op whose per-dtype semantics existed only
in a Rust doc comment. All four arrived in open PRs; none crosses the
finalize chokepoint, so none is touched by the privacy contract. The
sibling plans each carry a standing guard against new instances of
their class ([#730]'s token tripwire and append-only census; [#732]'s
exit census plus its no-third-formatter and no-untyped-decode rules);
this plan had none. This section is that guard.

**Exact PR #956 follow-up calibration.** Commit
`6ddf1a72d6dea6770a330d5c2ef3b8fa7d023c43` closes two specific
classification escapes, and this paragraph freezes its exact strength:

- `closure_conditional_macro_taint` propagates every macro defined under a
  non-include-guard conditional across the whole connected component of
  local includes in the header closure. A public declaration
  consuming any such token is rejected even when the definition and use are
  in different headers
  (`conditional_macro_taint_across_include_closure_is_rejected`). The
  component follows BOTH include spellings - `6ddf1a72` scoped it to
  quoted includes, and the round-3 angle-include closure below widened it,
  since `cc -E -I` resolves `<x>` and `"x"` identically inside the include
  directory. That widening had no control of its own until round-4 red
  team N4: narrowing the taint edges back to quoted includes left the
  whole suite green, because every angle-include test in it also tripped
  a different guard. The control plants an UNCONDITIONAL
  `#include <shared.h>` beside a conditional `#define`, so the taint
  component is the only thing that can reject it
  (`conditional_macro_taint_follows_angle_spelled_includes`).
- Numeric-callable classification treats every non-boolean, non-character
  built-in arithmetic value type conservatively as `numeric-op`: `double`,
  `float`, `int`, `short`, `long`, `signed`, `unsigned`, `size_t`,
  `ptrdiff_t`, `intptr_t`, `uintptr_t`, and the exact-width signed/unsigned
  integer types. Bare `int` is not generally control plumbing
  (`bare_int_export_is_numeric_op_and_requires_registration`).
- **The classification rule is INVERTED, so the conservatism above is real**
  (round-4 red team N1, folded 2026-07-31). `NUMERIC_C_TYPES` alone is an
  allowlist, and an allowlist of arithmetic spellings can never be complete:
  `_Float16`, `__fp16`, `__bf16`, `_Decimal64`, and `__int128` all
  classified as `[]`, so a bare 16-bit float export entered on an ordinary
  issue citation, past the rule that flagged rows have no citation path.
  The CLOSED list is therefore the other one, `NON_NUMERIC_C_TYPE_WORDS`:
  the qualifier and aggregate keywords plus the two non-arithmetic value
  spellings. A TYPE word in neither list is a build failure naming the
  unknown word, on the same footing an unresolvable typedef already has,
  unless it is a local typedef alias (whose own statement is checked by the
  same rule) or a `chelis_`-prefixed opaque handle whose layout is its own
  inventory row. Type words are separated from declarator names by POSITION,
  and a declarator name shaped like a type spelling - a leading underscore
  or a `_t` suffix - is rejected too, because it means the positional read
  was wrong. Controls:
  `an_unknown_c_type_word_is_rejected_rather_than_classified_empty`,
  `a_typedef_alias_cannot_introduce_an_unknown_type_word`,
  `a_reserved_shaped_declarator_name_is_rejected`, with
  `known_c_type_words_still_classify_without_rejection` as the positive leg.
  The language side of the same gap is closed by an EXHAUSTIVE match over
  `chelis_types::Prim` (`prim_census_class`): the census enumerates dtypes
  by their `(t-prim {} <name>)` spelling, so widening `Prim` would otherwise
  never reach the census at all. A new variant now stops the tripwire
  compiling until it is classified
  (`census_prim_lists_cover_every_prim_variant`).
- `NON_NUMERIC_INTEGER_PLUMBING_EXPORTS` contains exactly THREE exemptions,
  reproduced byte-for-byte here:

  ```text
  chelis_runtime.h: chelis_tensor * chelis_alloc ( int ndim , const int * shape , int dtype ) ;
  chelis_runtime.h: chelis_tensor * chelis_tensor_from_value_list_typed ( const chelis_list * list , int dst_dtype ) ;
  chelis_runtime.h: int chelis_dtype_size ( int dtype ) ;
  ```

  `apply_exact_integer_plumbing_exemption` removes `numeric-op` only when the
  complete canonical identity is in that list AND
  `GRANDFATHER_SEAM_IDS`; `integer_plumbing_exemptions_are_exact_and_closed`
  locks the count, membership, and same-shaped-neighbor behavior. A name,
  parameter spelling, substring, or newly added identity cannot inherit the
  exemption. Neither this follow-up nor the round-3 closure adds a
  configuration MATRIX: the policy remains that published ABI has exactly
  one preprocessing context, and both changes only widen what the guard can
  see.

The architecture is this plan's own move applied one layer up: three
chokepoints, each total over a DISCOVERED surface, never a
hand-maintained list (hand lists are how `HOST_ONLY_BUILTINS` rotted,
[#682]/[#705]):

1. **Values** (§C1-§C3, unchanged): an op result exists only through
   the private finalize constructors.
2. **Carriers**: a numeric value crosses a public boundary only inside
   a dtype-tagged carrier, enforced per FAMILY at the strongest rung
   that family can reach - the end-state table below. (An earlier
   revision made one blanket "unwritable by type via
   chelis#893/chelis#894" claim here; the PR #950 red team showed that
   overstated chelis#893's reach, which is the runtime tensor-data
   path, not every carrier family.) Until a family's typed mechanism
   lands, the guard is the capacity census and tripwire (deliverable 1
   below).
3. **Ops**: a numeric operation exists only with a decided row in the
   registry that owns its FAMILY. Table A's key is
   (builtin, surface, dtype) and stays the LANGUAGE-BUILTIN law;
   runtime exports, prelude/stdlib defs, and binding functions do not
   fit that key - no `BuiltinId`, no `Scalar|Tensor` surface (PR #950
   red team P1-2, which also caught the pre-existing strain in the
   `to_string` x Tensor/List seed row). Totality is therefore
   REGISTRIES PER FAMILY, each exhaustive over its own enumeration:
   language builtins in Table A, with checker acceptance DERIVED from
   it at Phase 4 so an unregistered builtin is `UnknownForm` by
   construction; runtime exports and exported prelude/stdlib defs in
   deliverable 1's structured operation-semantic registry; binding
   callables in the same registry shape once the rustdoc-JSON leg
   lands. Each non-Table-A entry binds the callable's exact canonical
   identity to one verbatim `[05-OP-N]` authority. The registry
   validates chapter `05`, group `OP`, and a normative definition line
   beginning `> **[05-OP-N]**`; a free-text chapter substring or
   cross-reference, a missing `[05-OP-999]`, or an observation atom such
   as `[05-OBS-1]` is not semantic registration. Tooling does not
   infer whether the selected existing OP atom is semantically relevant;
   review verifies that its normative text already governs the callable
   but cannot make a mismatched atom authoritative. If no atom governs
   the callable, the numbered spec gains the decision first. Existing
   numeric-callable rows at the initial baseline are grandfathered, and
   the grandfathering is an IDENTITY set, never a citation string: BOTH
   pre-ratchet citations (the seam one and the plain one) are frozen to
   exact `(id)` lists in the tripwire source, which ordinary baseline
   regeneration cannot rewrite and which may only shrink. The prior
   revision froze only the seam list and recorded the gap as a known
   residual - the plain citation was copyable onto a brand-new numeric
   row to skip the hook, which the 2026-07-31 red team executed.
   Freezing both lists closes it structurally, so the next sentence is
   now enforced rather than intended:
   every NEW runtime or exported stdlib numeric callable authors its new
   `[05-OP-N]` atom and exact mapping in the same change set. The
   implementation record is `SemanticRegistration { callable, atom }` in
   `SEMANTIC_REGISTRATIONS`; `callable` is exactly
   `[<kind>] <canonical id>`, so family identity is part of the key.
   This registration is deliberately separate from the capacity row's issue citation: the
   citation owns liveness, while the structured authority binding owns
   meaning. An operation registered in no family's registry is a build
   failure, not a doc comment (`capability_table.md` §New numeric ops
   carries the interim authoring rule).

   **Language legality and backend capability stay decoupled, here as
   everywhere** (the standing Table A / Table B split,
   `capability_table.md` §Derivations): registering an op decides what
   is legal in Surf, Deep, and the RISC DAG - a target-independent
   fact the CHECKER reports. What a given backend can execute is Table
   B's separate, later decision: a language-legal op a target cannot
   run is a build-stage capability rejection through [#730]'s
   `Unsupported` channel where the target is known, NEVER a checker
   type error. Both stages' diagnostic machinery already exists
   ([#731]'s witnessed checker diagnostics for language-level
   rejections; [#730]'s staged `Unsupported` for target-known ones);
   the family registries must preserve that stage split rather than
   collapse a backend gap into language illegality.

   Numeric-ness is STRUCTURAL, never declared: any callable whose
   canonical signature mentions a numeric dtype requires a row and an
   authority binding, and the
   non-numeric classification exists only for genuinely dtype-free
   surface - so the cheapest evasion (an op classifying itself
   non-numeric) is not representable. That sentence holds because the
   classification rule is inverted (the calibration bullet above): an
   arithmetic spelling the census does not recognize is a build failure,
   not an unflagged row. While the closed list was the NUMERIC one, an
   unrecognized spelling classified as dtype-free, which is exactly the
   evasion this paragraph disclaims - the round-4 red team executed it
   with `_Float16`. Wherever a consumer can be
   DERIVED from the table rather than compared against it, derive:
   Phase 4 already plans checker acceptance as a generated table view,
   which makes an unregistered op `UnknownForm` by construction.
   Comparison-based totality has an enumeration boundary (the
   `.dp`-reachable internals of [#730] census row 13 / chelis#794);
   derivation does not. Dependency, recorded: this chokepoint is only
   as total as the chelis#850-class closure (a `sig` with no `def`
   reaching an undeclared C call) in the [#730]/[#731] tracks - the
   FFI seam is a numeric-surface entry the table cannot see until it
   is closed.

**The carrier end state, per family.** Each family ends on the
strongest rung it can reach, stated so nobody infers a type seal that
cannot exist (PR #950 red team P1-3):

| carrier family | end-state mechanism | owner | final rung |
|---|---|---|---|
| runtime tensor data (`chelis_tensor.data`) | sealed typed access path | chelis#893 - a PEER class OUTSIDE [#729], unsequenced by the waves; a hard edge, never an assumption | types |
| execution wire schema | per-dtype tagged payload | [#729] Phase 1 (§C3) | types |
| language ops | checker acceptance derived from Table A | [#729] Phase 4 | derivation |
| published C signatures | tokenized canonical declaration inventory over callables AND non-function data, frozen derived classification, and a mechanically enforced ban on context-varying public ABI; this detects and blocks drift but does not generate the whole header | deliverable 1; the `RuntimeDType` fragment remains [#729] Phase 3 | census/tripwire |
| prelude / stdlib value ADTs | permanent census whose identity preserves type, variant, and numeric field shape, CLASSIFIED on the same rule as the C families: a float primitive in an untagged variant or field is a `float-carrier` seam with no citation path, an integer primitive is a `numeric-op` owing a semantic decision. A numeric field is legal Surf, so no type seal can exist for this family - the seam classification is the strongest rung available | deliverable 1 | census/tripwire |
| binding (PyO3) signatures | rustdoc-JSON registry + typed raw-dtype mutation oracle | the named pre-Phase-1 binding leg below | census/registry |

[#729] can close Phase 4 while chelis#893 remains open. Any
C6-complete claim at the [#729] close is therefore scoped to the
families [#729] owns; the runtime-data family completes at
chelis#893's exit, tracked as a hard edge in the roadmap, never
absorbed silently.

Deliverables, with phase homes:

1. **Now, pre-Phase-1: the capacity census + tripwire.** PR #956
   checks the covered families generated from actual artifacts:
   published-header declarations and struct layouts, stdlib ADT
   numeric carrier shapes, and exported runtime/stdlib numeric
   callables. The header legs use
   `preprocessed_headers -> header_rows`; the stdlib legs use exactly
   `stdlib_rows -> scan_deftypes + scan_exported_numeric_defs`.
   The prelude / stdlib value-ADT family has a second enumeration
   source since chelis#890: **Rust-registered prelude ADTs**
   (`register_prelude_adts` in `crates/chelis-types/src/builtins.rs`)
   enumerate through `prelude_adt_rows -> chelis_types::prelude_adt_defs`
   as the `prelude-adt-numeric` leg, classified on the identical
   float-carrier / numeric-op rule with a shape-complete identity.
   Before chelis#890 no prelude ADT carried a numeric payload, so the
   `.ch` enumerator had nothing to miss; the prelude `Json` ADT
   (`JInt int64` beside `JNum f64`, split decided by [05-OP-2]) made
   the Rust registry a numeric surface, and this leg closes what would
   otherwise be exactly the unenumerated-surface blind spot this
   section forbids. A prelude ADT registered outside
   `register_prelude_adts` cannot exist (there is one registration
   path), so the enumeration is complete by construction.
   Its leg manifest is a typed executable contract, not editable
   `covered` prose. This change adds the typed wire-schema and PyO3
   signature legs and moves them to `covered` only with their separate
   commands and red mutations green. The payload-work citation is chelis#893, an
   ISSUE - never PR #894. Implemented command:
   `cargo nextest run -p chelis-cli --test capacity_census_tripwire --no-fail-fast`
   is the authoritative covered-family oracle (regeneration: the same
   command with `CHELIS_CAPACITY_CENSUS_WRITE=1`). The liveness command
   is `.venv/bin/python scripts/capacity_census_liveness.py`; success is
   exit 0 with final line `CAPACITY CENSUS LIVENESS: PASS`. Execution
   is CI-owned through [#730]'s §C7.5 channel (converged 2026-07-31,
   superseding the earlier manual-only contract): the change-gated
   blocking job runs it when the census or its citations change, and
   the nightly full sweep re-runs it for standing-citation drift;
   release cuts and red-team passes remain additional manual
   invocations, not the only ones.

   `coverage_manifest()` in
   `crates/chelis-cli/tests/capacity_census_tripwire.rs` is the fixed
   executable coverage contract. Its typed `CoverageManifest` records
   `covered: Vec<CoveredLeg>` and `deferred: Vec<DeferredLeg>`; both leg
   types carry `leg`, `artifact`, `enumerator`, `command`,
   `expected_success`, and `mutations`, and a deferred leg additionally
   carries `owner`. `Baseline { version, legs: CoverageManifest, rows }`
   is version 2, and its serialized `legs` value must equal the
   executable manifest exactly. Editing
   `spec/design/capacity_census.json` therefore cannot promote a
   deferred leg. The completed typed Phase 1 entry commitments are:

   | typed leg | artifact | live enumerator | command and expected success | standing red mutation |
   |---|---|---|---|---|
   | `wire-schema-numeric-fields` | `crates/chelis-compiler-api/src/schema.rs` public serialized type graph | rustdoc JSON public schema type graph -> numeric fields | `cargo nextest run -p chelis-compiler-api --test capacity_census_wire`; `wire_schema_numeric_fields_match_the_reviewed_baseline` passes | `adding_or_removing_a_public_serialized_f64_field_changes_the_census` |
   | `binding-raw-dtype-params` | `crates/chelis-python/src/lib.rs` registered PyO3 callables | live registered PyCFunctions/pyclasses joined to rustdoc JSON signatures | `cargo nextest run -p chelis-python --test capacity_census_bindings`; `registered_pyfunctions_match_the_reviewed_rustdoc_signatures` passes | `a_registered_pyfunction_with_a_raw_dtype_parameter_is_rejected` |

   Each leg is a live enumerator, executable command, exact success
   condition, and mutation test recorded in `coverage_manifest()`. The wire
   baseline freezes public serialized numeric carrier shapes only. Its
   `float-carrier` classification is shape metadata for this census: a
   `TensorElements::{F64,F32,F16,Bf16}` variant inside the serde dtype-tagged
   payload is the intended carrier, not an untagged seam and not authority for
   execution semantics. In
   particular, post-PR-#956 root structures are census inputs, not authority
   for root identity, manifest order, dotted-root expansion, `requires_main`,
   artifact routing, or `HostReason`; those remain chelis#912 work. The PyO3
   leg freezes registered signatures and rejects raw dtype ingress; it does
   not inspect or redesign private runtime-dtype decoding. `KNOWN_TAGS` and
   Deep stamping are outside this task entirely.

   Binding-baseline dispositions (each entry is the C6 review a frozen
   fingerprint update cites):

   - 2026-08-02, chelis#816 (PRs #819/#822): `compile_and_load` gains
     `project_root: Option<&str>, force_bare: bool` and `eval_json` gains
     `project_root: Option<&str>` for reef-context resolution. All three
     parameters are dtype-free control/path inputs (a filesystem path and a
     lane selector); the enumerator classifies both rows `[]`, no numeric
     capacity enters the surface, and no raw dtype id is introduced.

   Acceptance requirements, from the 2026-07-30 and 2026-07-31
   adversarial passes. Each bullet names the standing control that turns
   RED when its guard is reverted; the round-4 additions are the
   inverted type-word rule and its `Prim` totality lock, the array-typedef
   resolution, the signature-less stdlib export failure, the
   angle-spelled macro-taint control, the recursive published-header
   walk, the stdlib-side registration controls, the removal of the
   redundant seam count, and the splice-aware `#line` ban:

   - **C declaration identity is canonical, not pretty-print text.**
     The real preprocessor supplies the transitive published-header
     artifact, including macro expansion; the census then tokenizes
     declarations into a punctuation/whitespace-independent canonical
     identity. A missing compiler fails loudly. Typedefs are resolved
     for classification, while the canonical declaration key remains
     stable across supported toolchains. The standing
     `c_identity_is_token_canonical_across_preprocessor_whitespace`
     control feeds the observed Apple-clang and Linux spellings through
     `canonical_c_tokens` and requires identical rows; the hosted
     toolchain lanes then compare one canonical baseline. The
     Apple-clang-versus-Linux spacing difference that failed PR #956's
     first hosted oracle is the motivating case. Adding a second
     observed spelling to the baseline is not a fix.
   - **Published ABI has one preprocessing context.** Public
     declarations may not vary under feature macros or include-root
     context. `assert_context_invariant_headers` mechanically rejects conditional
     declaration regions and multiple expansions of one public header
     that produce different canonical inventories. This is the chosen
     alternative to a supported configuration matrix: first-expansion
     wins is forbidden, and a future decision to permit
     configuration-varying ABI requires a B1 amendment defining
     context identities and inventorying the union. The raw-source scans
     that back this rule follow BOTH include spellings: `cc -E -I` resolves
     `#include <x>` against the include path exactly as it resolves
     `#include "x"`, so a guard reading only quoted includes would leave an
     angle-included local header outside every raw-source check
     (`angle_included_local_header_is_inside_the_context_guard`).
   - **Attribution is total, and `#line` is banned.** The census decides
     which published header a declaration belongs to by reading `cc -E`
     linemarkers back, so a `#line` directive in a published header can
     attribute a real, callable export to a path outside the include
     directory and delete it from the inventory while `cc -fsyntax-only`
     still accepts calls to it. Published headers may contain no `#line`
     directive and no hand-written linemarker
     (`line_directive_in_a_published_header_is_rejected`). Independently
     of that ban, every declarator the RAW published closure declares must
     reappear in some attributed bucket
     (`a_declaration_missing_from_every_attributed_bucket_fails`); the
     comparison is on declarator NAMES so that legal macro expansion of a
     type spelling does not read as a missing declaration. The ban names
     the known channel; totality does not depend on having enumerated the
     channels - which the round-4 pass demonstrated rather than argued: a
     `#line` split across a phase-2 line splice reached the backstop and
     was rejected there, correctly but under the wrong name. The ban now
     joins splices before reading directives, so the diagnostic matches
     the defect (`a_spliced_line_directive_is_caught_by_the_ban_itself`).
     The export never entered the inventory either way; this is
     diagnostic quality, not a closed hole.
   - **The published header set is derived, not declared.** `HEADER_ROOTS`
     survives as a record of WHY each root is published, but it is not
     trusted: the roots' INCLUDE closure must equal the `.h` files actually
     present in the published include directory, so a header dropped in
     and reachable from no root fails loudly instead of contributing
     nothing (`a_published_header_reachable_from_no_root_fails`).
     The directory walk RECURSES, keyed by the path an `#include` spells
     (`sub/x.h`). A flat scan left a subdirectory as an uninventoried
     publishing channel, which is the same hole one level down
     (round-4 red team N5;
     `a_published_header_in_a_subdirectory_is_reached_or_fails`).
     Reachability is asked of the include graph rather than of the
     preprocessed buckets because whether a header emits a locally
     attributed bucket at all is a preprocessor detail: `chelis_blas.h`
     yields one under Apple clang and none under Linux gcc, since its body
     is entirely include and conditional directives
     (`a_declaration_free_root_is_still_reached`).
   - **Non-function ABI is inventoried too.** A published header's
     numeric surface is not only its callables. `extern <type> <name>;`
     data declarations are inventoried as `header-data` rows and
     classified on the same rules, so an exported bare `double` global or
     raw dtype id is a seam with no citation path
     (`extern_data_declarations_are_inventoried_and_classified`).
     Function-pointer typedefs are RESOLVED rather than skipped - a
     setter taking `typedef double (*cb)(double, int elem_dtype)` inherits
     the callback's numeric and dtype words
     (`function_pointer_typedef_cannot_launder_a_seam`) - and a
     parenthesized typedef the resolver does not understand is rejected
     rather than silently skipped, because skipping is exactly how that
     seam laundered itself
     (`unresolvable_parenthesized_typedef_is_rejected`).
     ARRAY typedefs are resolved on the same footing, and were the third
     shape (round-4 red team N2): the generic word split popped the array
     EXTENT as the alias, so `typedef double chelis_vec4[4];` registered
     `4` and a setter taking `chelis_vec4` inherited nothing. The alias is
     the identifier before the first `[`, and a shape the resolver cannot
     read that way is rejected rather than guessed at
     (`an_array_typedef_cannot_launder_a_float_carrier`,
     `an_unresolvable_array_typedef_is_rejected`). A `chelis_`-prefixed
     array alias is not covered by the inverted type-word rule, which
     accepts project-named handles, so this needed its own closure.
   - **The stdlib carrier families carry capacity flags.** A
     `std-adt-numeric` or `std-def-numeric` row is classified, not merely
     inventoried: a float primitive (`f64`/`f32`/`f16`/`bf16`) in an
     untagged public position makes the row a `float-carrier` SEAM with no
     citation path, and an integer primitive makes it `numeric-op`, owing
     the same semantic registration a runtime callable owes. Before this
     the ADT leg emitted no flags at all, so adding `| JsonBigNum(f64)` to
     `io/json.ch` landed by regenerating and citing an open issue - the
     P1-1 shape closed for the header family only, and a direct
     contradiction of the `AGENTS.md` rule it was meant to enforce
     (`std_adt_bare_f64_variant_has_no_issue_citation_path`, with
     `std_adt_integer_carrier_is_numeric_op_not_a_seam` as the positive
     leg for the source-faithful `JsonInt(int64)` shape this plan wants).
     The def leg reads capacity off the DECLARED signature, so an
     exported `def` that declares none is public numeric surface the
     census cannot see. That case used to produce no row and no
     complaint, and the Surf style guide recommends exactly that shape
     for load-style top-level bindings, which put the silent path one
     stdlib commit away; it is now a loud census failure naming the
     definition (round-4 red team N3;
     `an_exported_stdlib_def_without_a_signature_fails_loudly`, with
     `a_declared_stdlib_signature_enumerates_and_a_type_export_does_not`
     keeping a dtype-free type export from reading as one).
   - **Matched rows freeze enforcement metadata.** Equality is not
     merely `(kind, id)`: the tripwire compares the complete
     enforcement-relevant derived classification for every matched
     row. A typedef target changing from an exact integer to `double`
     must fail as classification drift even when the declaration's
     surface spelling and key are unchanged. Positive controls keep
     exact-width types unflagged; negative controls cover drift into
     `float-carrier`, `raw-dtype-int`, and `numeric-op`
     (`matched_row_float_carrier_metadata_change_fails` and
     `matched_row_typedef_int64_to_double_metadata_change_fails`).
   - **Stdlib carrier identity is shape-complete.** A numeric ADT row
     preserves its public type, variant, and field position/type shape,
     including multiplicity. Adding a second same-dtype variant or a
     numeric field to an existing variant changes the identity and
     fails. Collapsing a type to the SET of dtypes it mentions is not a
     permanent carrier census.
   - **Operation meaning is a structured exact binding.** Runtime
     exports and exported stdlib numeric defs are exhaustively
     discovered and keyed by exact canonical callable identity. Their
     separate semantic registry names one exact `[05-OP-N]` atom and
     validates its chapter/group and normative `> **[05-OP-N]**`
     definition. Initial numeric-callable
     rows are grandfathered; a new callable authors a new OP atom and
     mapping together. Positive controls bind known callables to their
     decisions.
     Negative mutations add an unregistered runtime export and stdlib
     `export def`, name absent `[05-OP-999]`, and substitute
     `[05-OBS-1]`; all fail. An issue citation or a bare chapter
     substring is never semantic authority. The STDLIB half of that
     claim had no control until round-4 red team N6, which is why it is
     named twice now: a new exported numeric def cited with only an
     issue fails, and - the branch that binds `std-def-numeric` by KIND
     rather than by flags - a float-only def whose capacity seam a
     maintainer override disposed of STILL owes its registration
     (`a_new_stdlib_numeric_def_requires_semantic_registration`,
     `a_stdlib_registration_against_a_nonexistent_atom_fails`). The tool does not infer
     semantic relevance within the OP group; review verifies the
     numbered-spec decision but cannot create it.
   - **A new row cannot self-bless, and a new SEAM cannot be cited
     into existence at all.** Regeneration emits `citation: TODO` and
     CI fails on TODO. An UNFLAGGED row names an OPEN issue (invariant
     7 governs the release). A FLAGGED capacity row has NO
     issue-citation path: the grandfathered 2026-07-30 seam set is
     frozen by exact citation, row identity, and derived
     classification. Removing one seam cannot relocate its citation to
     a new row, and adding one alongside the whole original set is the
     same rejection (`grandfather_citation_cannot_be_copied_onto_new_rows`).
     There is deliberately no separate COUNT lock. The identity freeze
     subsumes it - `GRANDFATHER_SEAM_IDS` IS the set, so a row carrying
     the citation is either one of those identities or already a
     rejection - and the count branch it replaced could not be reached
     by any input, which makes it an untested claim rather than a second
     guard (round-4 red team N7). The PLAIN pre-ratchet citation is frozen the same way
     and for the same reason - a citation string any new row may copy
     is not a disposition, and leaving it unfrozen let a brand-new
     numeric export skip the semantic hook
     (`plain_baseline_citation_cannot_be_copied_onto_a_new_row`).
     Both lists are hand-maintained and SHRINK-ONLY, deliberately not
     regenerated: a generator that re-derived them from the baseline
     would re-bless whatever a contributor had just pasted the citation
     onto. The only sanctioned outcomes are redesign onto the
     tagged carrier, removal, or
     `maintainer-override(<reason>, chelis#N)`, which is assigned to
     human review; opening an issue is not authorization. That marker
     is validated for SHAPE rather than matched as a prefix: a balanced
     closing paren, a nonempty reason, and the `chelis#N` reference
     INSIDE the parentheses, so an unterminated marker or a reference
     that sits after the closing paren is a forgery and fails
     (`malformed_maintainer_overrides_fail_and_the_exact_form_passes`).
   - **Liveness is issue-typed.** Every sanctioned `chelis#N`
     reference must exist, must be an ISSUE rather than a pull request,
     and must be OPEN. A closed, missing, or PR reference fails and
     forces re-adjudication. `capacity_census_liveness.py` represents
     the result as `IssueRecord { kind: IssueKind, state: IssueState }`;
     `fetch_issue` calls
     `gh api repos/Chelis-Lang/chelis/issues/N` and treats the REST
     payload's `pull_request` field as the wrong object kind. The unit
     controls include an open issue, closed issue, missing issue, and
     open PR.
     The division of labour is deliberate and stated here so nobody
     infers more from a green tripwire run than it proves: **the
     tripwire checks citation SHAPE only** - that a `chelis#N`
     reference is present, and that a `maintainer-override(...)` marker
     is balanced with its issue inside the parentheses - because it
     runs offline and network access would make it flaky and
     unrunnable in a sandbox. **Existence, kind, and open-state are the
     LIVENESS gate's job**: `.venv/bin/python
     scripts/capacity_census_liveness.py`. As originally landed this
     was a manual gate run at release cuts and red-team passes, with
     the offline constraint as its rationale (round-4 red team N8,
     recorded); **converged 2026-07-31 onto [#730]'s §C7.5 CI channel**,
     which resolves the same offline/online split without leaving the
     live half manual: the change-gated blocking `ci.yml` job (with
     `issues: read`, failing CLOSED on tracker unavailability) runs the
     liveness command pre-merge whenever the census or a citation
     changes, and the nightly full sweep re-runs it so a cited issue
     closing goes red within a day rather than at the next release
     cut. A citation naming a missing issue or a PR now fails
     pre-merge, not only at the next manual pass; the shape/liveness
     boundary itself is unchanged. Scheduling and gating are [#730]
     §C7.5's; this script, its pass line, and the census contract
     remain this plan's (the same contents-vs-scheduling split as the
     [#732] oracle interlock).
   - **Failure messages teach** the rule, the sanctioned actions, and
     the §C6 pointer. For a context-poor agent the error text is the
     only documentation that provably gets read; the cheapest passing
     action must be visible IN the message and must be the wanted one.
   - **The census and tripwire files are review-ROUTED, stated at the
     enforcement actually configured.** CODEOWNERS auto-requests the
     human owner but does not block (`require_code_owner_reviews` is
     off and bypasses exist). Enabling the code-owner toggle is a
     separate repository-policy decision. The structural enforcement
     here is the executable identity/metadata/registry freeze plus the
     ordinary required review; this document claims no more.
2. **Phase 1**: the census re-derivation defines §C3's atomic set (the
   §C3 amendment above); the tripwire baseline regenerates in the same
   change set.
3. **Phases 2-3, coordinated with chelis#893** (and PR #964's merged `Repr` vocabulary (2026-07-31; chelis#894 is the tracking issue)): the families with typed end states per the table above
   become types; each tripwire row retires by citation as its seam
   unwinds. The known `(double, int)` seams are the kill-list,
   starting with `chelis_format_shortest`'s pair
   (`faithful_observation.md` §I1 records it) and any survivor of PR
   #891's rework. chelis#893's own completion is that peer class's
   exit, not this plan's to claim.
4. **Phase 3**: the `RuntimeDType` header FRAGMENT is already
   generated with a byte-for-byte regeneration test ([#730] §C4.3);
   the census asserts the fragment-owned rows against regeneration
   rather than diff. The rest of the public header is structurally
   guarded by canonical inventory, metadata freeze, semantic
   registration, and the no-context-variance rule; it is NOT generated,
   and a hand-added export remains representable but cannot pass the
   tripwire without its sanctioned disposition. FULL header generation
   is in no phase contract today. Adopting it would be a stronger
   B1-protocol amendment to Phase 3, not a prerequisite silently inferred
   from the standing census (red team P1-3).
5. **Phase 4**: the totality leg over the reachable surface, plus the
   table's mandatory atom citation, make every new numeric op force
   spec authorship - the authoring-forcing function generalized from
   dtype cells to op EXISTENCE. The added-variant mutation oracle's
   build set is derived from cargo metadata (every workspace member
   depending on `chelis-vocab`), never hand-listed; the chelis-python
   escape in the sweep is the motivating instance ([#730] §C6 owns the
   oracle; this row records the derivation rule).

Scope honesty, sharpened by the 2026-07-30 and 2026-07-31 adversarial
passes. On the
families PR #956 covers, additions, removals, classification drift,
missing semantic registrations, ADT-shape growth, ADT and stdlib-def
capacity classification, non-function data ABI, unattributed
declarations, unreached published headers, and context-varying
public ABI are executable build failures. Those mechanisms are
structural guards, not review-only visibility, but they do not make the
underlying source forms unrepresentable: a contributor can still write
a C export or stdlib variant, and the tripwire then blocks it until the
sanctioned disposition is present. The registry can validate an
`[05-OP-N]` identity and existence, not infer whether a human selected
the right OP atom. CODEOWNERS routes that judgment but does not enforce
owner approval.

The three baselines cover the currently named families after the wire and
PyO3 hard edges land; a genuinely new surface kind remains invisible until
an enumerator is added. For every covered family (and any future explicitly
deferred family), the design criterion is CHEAPEST-PASSING-ACTION: an agent
blocked by the tripwire
must find that redesign/removal, extending the enumerator, or authoring
the exact spec entry is the cheapest sanctioned action. The TODO flow,
seam/metadata freeze, typed legs, liveness, mutation controls, and
teaching messages exist to make that true. The doctrine also lives
where context-poor agents read it:
`AGENTS.md` §Numeric Surface Discipline carries the binding rules, and
this section is its elaboration.

Named non-goals, each with its owner, so coverage is never inferred:

- **Ingress dtype SELECTION - restated 2026-07-30; the first version
  of this bullet overclaimed the non-goal.** Where BOTH ends of a
  conversion are typed, dtype selection IS the type checker's job, and
  the language already does it structurally: no implicit precision
  promotion exists, every conversion is a spelled `cast`, and the
  [#759] ladder makes the CHECKED cast the default (it traps when the
  value does not survive the target) with lossiness only as a named
  spelling. §C6's chokepoints add nothing there and disclaim nothing
  there. The genuine non-goal is one step earlier, where the source
  side has no type yet: host-lane ingestion that parses external
  text or bytes and CHOOSES the first dtype (a CSV column read as
  f64; a JSON number funneled to float). `csv_f64(...) ->
  tensor[n, f64]` is a well-typed program - the defect is that the
  source's integer-ness never existed as a type for the checker to
  defend, and no checker can govern a conversion whose source type is
  not in the program. The fix is still type-system-shaped: TYPE THE
  BOUNDARY - source-faithful ingestion ADTs, the in-tree `io/json`
  precedent (`JsonInt(int64)` beside `JsonFloat(f64)`; JSON syntax
  distinguishes the two, so a parse that erases it discards
  information the source format carried) - after which the checker
  governs everything downstream and the [#759] discipline covers the
  now-visible casts. [04-NUM-11] owns that obligation, the census's
  std-ADT leg keeps every ingestion ADT's numeric variants visible,
  and `AGENTS.md` §Numeric Surface Discipline states the
  never-silently-narrow rule for boundary authors.
- **Shell-side surface.** A shell wrapping the runtime with its own
  `(double, int)` helper is invisible to chelis CI; the mirror is a
  conform-contract row (chelis#738's lane), not this plan.
- **Raw FFI buffer writes** until chelis#893's seal lands; the C
  runtime's `pub` untyped data pointer is that issue's subject.
- **A genuinely new surface KIND** (a new serialization format, IPC
  channel, or export mechanism) is unguarded until the enumerators are
  taught it. Introducing one without extending them in the same change
  set is a review-blocking finding; the rule lives in `AGENTS.md`
  because the enumerators cannot see what they were never pointed at.

Within the covered bounds, silent or undecided additions fail the
tripwire; bad human authority selection is still possible and is forced
to a visible, structured, reviewed point. Deferred families have only
the explicit pre-Phase-1 hard edge until their enumerators land. This is
the enforceable surface successor to §C3, not a claim that every public
numeric form is type-unrepresentable.

---

# Part II - process rules that hold at every phase boundary

## B1. Freeze points

| contract | frozen at end of | may change after only by |
|---|---|---|
| §C1 semantics table + spec/04 section | Phase 1 | spec change + this doc + re-run of the full matrix |
| §C3 public API + storage layout + wire schema | Phase 1 | versioned schema bump, all four layers together |
| §C2 trap kinds + exact message strings | Phase 2 | this doc + [#687] corpus update in the same PR |
| §C4 element-formatting rules 1-4 and 6 | Phase 1 (Rust) / Phase 3 (C parity) | this doc + [#687] corpus update |
| §C4 root envelope rule 7 | [05-OBS-6] authored 2026-07-31; full implementation acceptance pending [#1023] | spec/05 [05-OBS-6] + `faithful_observation.md` + this doc + the release roadmap + root-boundary corpus, one change set |
| §C5 kernel signatures | Phase 2 | this doc |
| §C6 covered-family census + tripwire | at PR #956 landing: canonical row identities, complete derived classifications, stdlib ADT shapes AND their capacity flags, callable-to-`[05-OP-N]` registrations, public-header context invariance, and BOTH grandfathered identity sets (seam and plain) all freeze; unflagged rows append by live ISSUE citation, new numeric callables also author/register a new OP atom, and FLAGGED rows are shrink-only (a shape-validated maintainer override is the sole human exception) | this doc + the executable tripwire/registry artifacts and their positive/negative controls, same change set |
| §C6 typed wire/PyO3 leg state | before Phase 1 entry: frozen when each named enumerator and mutation command is green; an editable baseline field cannot change coverage | this doc + the typed leg manifest + owning enumerator/oracle in the same change set |
| capability table schema | Phase 4 entry | `capability_table.md` (the owning doc) + this doc |

"Frozen" means: later phases may ADD consumers but not reinterpret
behavior. If your phase needs a frozen contract to change, stop, update
this document and [#729] first, and say so in the PR - that is the
protocol, not a failure.

## B2. Invariants that hold across every boundary

1. **Controls never move.** Every green control in the audit test files
   (correct cells, `ByDesign` float rows, the documented lucky-greens)
   keeps its exact expected string through all five phases. A phase PR
   that edits a control's expectation is wrong until proven otherwise -
   the burden is on the PR to show the control itself was wrong.
2. **Red-to-green only by un-ignoring.** The `#[ignore]`d tests assert
   correct behavior and fail today. A phase completes cells by making the
   test pass and REMOVING the ignore attribute in the same PR - never by
   editing the assertion to match behavior (the audit's "never invert an
   assertion" rule).
3. **No new dispatch wildcards.** Any `match` a phase touches in the
   numeric crates loses its `_` arm over closed enums (`Prim`, `RiscOp`,
   dtype ids) rather than gaining one, ratcheting toward
   `numeric_audit_structural_prevention.md` item 3a.
4. **Discoveries fork, they do not scope-creep.** Mid-phase findings
   (there will be some; every audit pass found more) are filed as issues
   and linked to [#729] - a phase's exit oracle does not grow after entry.
5. **Both lanes or neither.** Any behavior change lands with its eval and
   compiled-lane expectations updated in the same PR, per the
   Public-Surface Change Rule in the repo contract.

## B3. How to pick up a phase

1. Read this doc's Part I, your phase's section, and the previous phase's
   "frozen at exit" list. You may rely on frozen items without re-review.
2. Run your phase's oracle suite first; record the red set in the PR
   description (it is your work-list and your done-list).
3. `docs/investigations/probes/` has the raw probe drivers if you need to
   re-derive any cell's current behavior from scratch; do not trust
   comments, including this document's - the oracle tests are the truth.
4. Land against the gate (`scripts/gate.py --local`), open the PR early,
   let CI's macOS Smoke run the workspace suite.

---

# Part III - the phases

## Phase 0 - detectors (before any refactor; afternoon-scale)

**You inherit:** the PR [#696] test surface as-is; no code changes exist yet.

**You deliver:**

1. **The domain-validity checker**: a test-support function
   `assert_elements_in_domain(prim, printed_or_stored: &Values)` that
   implements §C1's *value set* column only (no finalize, no traps - pure
   membership), plus its wiring into the existing lane drivers
   (`eval_lane_str` / `c_lane_str` / `build_and_run_c` families) so every
   matrix test gets domain-checking for free.
2. **The [#687] exact-integer oracle lanes**: `eval_agreement.rs` gains an
   exact-string lane (or is superseded by the PR [#696] drivers - decision
   recorded in the PR); `parity.rs` loses the silent float-parsing
   fallback (a mismatch REPORTS, and only ops carrying a tolerance row in
   spec/05 §8, which [05-OBS-3] names as that table's single address, may
   compare tolerantly). Constraint, verified by execution 2026-07-17:
   the current run-mode parity corpus is byte-identical across lanes
   (123/123 lines) only because it prints dyadic floats exclusively, so
   this lands green - but until [#732] Phase 2 delivers byte-identical
   rendering, it freezes that float diet. A new example printing any
   computed float (a `sqrt`, a non-dyadic product) goes red for
   formatting reasons, not value reasons; [#732] Phase 2 is the release
   valve. [#732] Phase 3's "fallback deleted" item is satisfied by this
   deliverable (recorded in both docs).
3. New `#[ignore]`d rows where the domain checker exposes cells the audit
   did not enumerate (expected: few; the audit was thorough, but the
   checker is mechanical).

**Frozen at your exit:** the domain-checker's API and the drivers'
verbatim-string discipline. Later phases treat "domain checker green" as
load-bearing evidence.

**Explicitly not yours:** fixing anything the detectors reveal; touching
production code at all.

**Oracle:** the invariant harness runs in CI, red (ignored) on exactly the
audit's known bad cells, green on all controls. `parity.rs` still passes
on the existing corpus with the fallback removed.

## Phase 1 - the semantics module, the storage decision, eval adoption

**You inherit:** Phase 0's detectors (your acceptance instruments),
Part I as the spec of what to build, and the complete §C6 capacity census
and tripwire: PR #956's header/stdlib legs plus the typed wire-schema and
registered-PyO3 legs landed before this phase (the roadmap's Wave 3 entry
gate). Before touching the storage decision, re-run
`capacity_census_tripwire`, `capacity_census_wire`, and
`capacity_census_bindings`; all three must be green against the current
surface. "Every public numeric channel" is the state those executable
enumerators establish, never a prose or baseline claim.

**You deliver:**

1. The `dtype_semantics` module implementing §C1-§C4's Rust side:
   `finalize_scalar` / `finalize_tensor` / `scalar_from_*` /
   `format_element` / `NumericTrap`, with exhaustive `Prim` matches and
   unit tests per cell of the §C1 table (positive AND negative per the
   repo's negative-test-parity rule: every rounding case, every trap
   case, every special value).
2. **The spec/04 section**: spec/04 §9 carries the decided contract as
   current blockquote authorities, including [04-NUM-14]'s checked-cast
   ladder, with an honest status banner
   (seeded ahead of this phase; implementation tracked here). This phase
   RATIFIES and refines those semantics (and their §C1/§C2 correspondence) in
   the same PR as the module, so spec and code cannot diverge at the moment the
   semantics become real. Chelis#733 Phase 1 separately migrates the authority
   form and revisions through the pinned Buoy shell-side integration.
3. **The storage decision at every layer of the §C3 census as
   re-derived at entry** (the four layers named in §C3 - eval
   `TensorStorage`, the versioned wire schema, the Python boundary,
   prove's env - are the 2026-07 floor, not the set; the §C6 capacity
   census is the enumeration that decides what else joined). The actual
   Phase 1 outcome is recorded in §C3/§C5: prove's exact env swap is deferred
   to Phase 2 under [#688], while the typed wire/storage change mechanically
   converted its necessary flattening reads to explicitly named-lossy
   adapters. This recorded adapter-only change replaces the earlier
   "compiles untouched" condition.
4. **Eval adoption**: eval scalar and tensor paths construct exclusively
   through the module (raw constructors are now private - the compiler
   gives you the site list; the PR description records the count).
   `tensor_float_unop_f32` and the raw f64 `binary_map`/`unary_map` paths
   are deleted, not deprecated.
5. Crate-placement decision (open question 4) recorded in this doc.
6. **Checker adoption of decided dtype cells:** [#860]'s single
   post-desugar operand-dtype policy chokepoint covers direct applications,
   reduction data arguments, bare pipe stages, and polymorphic
   instantiations for integer `mean` [#724] and the authored bool-arithmetic
   roster [#726]. Direct and polymorphic paths reuse one canonical diagnostic;
   Phase 4 replaces this interim policy with the generated Table-A view.

**Frozen at your exit:** §C1 table + spec section; §C3 API + storage +
wire schema; §C4 rules 1-4 as implemented in Rust. **Eval is now the
reference RENDERER**: Phases 2-3 validate other lanes against eval's
output strings. It is NOT the value authority - `spec/08-backends.md` §2
names the C backend the reference implementation, and spec/04
[04-NUM-8] fixes the arithmetic width both lanes owe independently. A
lane conforms to the spec, not to eval; where eval and the spec disagree
eval has the bug, and today it has several ([04-NUM-8]'s divergence
note).

**Explicitly not yours:** the compiled lanes (C still wrong in all its
audited ways at your exit - expected); trap wiring in `host_ops`' scalar
kernels beyond what finalize forces (Phase 2); any generated-C work.

**Oracle:** `.venv/bin/python scripts/dtype_phase1_oracle.py` is this phase's
single authoritative command. Acceptance is exit 0 with the final line
`PHASE 1 ORACLE: PASS`. Its tested command manifest runs the three §C6 entry
censuses; the sealed semantics, typed Load, and execution-wire exactness
controls; `eval_tensor_narrowing_matrix.rs`; the Phase 1 eval rows of
`narrow_dtype_matrix.rs` ([#717] tensor cells), `precision_matrix.rs`
([#684] storage/binding cells), `int_width_lane_matrix.rs` (the eval tensor
trap cell), and `reduction_and_bitwise_matrix.rs` ([#684] sum and [#724]'s
eval half); the checker/payload controls; the Phase 0 domain checker over all
eval outputs; and the Python/Hull tagged-wire readers. It never runs ignored
rows or compiled-C phase work. The exact scalar-kernel rows for [#680],
[#688], [#718], and [#722] remain Phase 2 obligations; this oracle does not
pull them across the kernel-split boundary.

## Phase 2 - the kernel split and prove

**At phase entry you inherit:** the module (frozen §C1/§C3/§C4-Rust), eval
as reference lane, and the not-yet-split `host_ops` helpers now visibly
awkward (they finalize but still accept `Fn(f64,f64)`).

**Implementation status:** draft PR #1054 delivers this phase as one
consolidated change. It replaces the open host and IR arithmetic closures
with sealed dtype-keyed kernels, preserves declared-width integer and float
semantics through standalone and fused evaluation, and makes static-condition
folding typed and decline on traps. Prove's concrete and generated
environments carry exact `ScalarValue`s, including every post-#956 public
wire scalar variant without dtype substitution. The exact trap grammar is
public module data and the active Phase 2 corpus asserts byte-exact
diagnostics. Integer `abs` lowers to the typed trapping kernel, integer
`floor`/`ceil`/`round` lower to identity, and the C, HIP, and Metal emitters
remain loud instead of routing integer `abs` through float-only templates.
The reduction follow-up routes ordinary, windowed, and argument reductions,
plus the overlapping window adjoint, through the same sealed boundary:
consumers retain only shape and ordered index-group planning. Declared-width
float witnesses, exact int64 comparison, all four integer-width
intermediate-overflow rows, an overlap-add adjoint witness at every float
arithmetic width, and structural no-bypass locks are part of the oracle.
Implementing backend kernels remains Phase 3 / [#699].

**You deliver:**

1. §C5's kernel signatures as the ONLY arithmetic entry points in
   `host_ops.rs`; the `Fn(f64, f64) -> f64`-shaped helpers
   (`numeric_binop`, `numeric_unop`, `tensor_numeric_binop`,
   `dispatch_scalar_binop`) are deleted. The compiler enumerates every
   call site; the PR records the count against [#695]'s "~17".
2. The trap contract end-to-end in eval: overflow/domain/divzero traps
   surface with §C2's exact strings; the strings become `pub const` and
   are FROZEN in this PR.
3. Prove: `concrete_eval`'s env becomes exact-typed; the two `as f64`
   flatteners (`obligation_engine.rs:1857`, `opaque.rs:1536`) will no
   longer compile - replace with exact reads; `fuzz` samples integers as
   integers.
4. `fold_static_cond` moves to `int_binop` + finalize-in-Cast-arm with
   decline-on-trap (this is here, not Phase 1, because it needs the
   frozen trap semantics to define "decline").

**Frozen at your exit:** §C2 message strings; §C5 signatures. Phase 3 may
generate C guard code emitting those exact strings without asking.

**Explicitly not yours:** C backend behavior (still unwrapped/untrapped at
your exit); formatting anywhere.

**Oracle:** `.venv/bin/python scripts/dtype_phase2_oracle.py` is this phase's
single authoritative command. Acceptance is exit 0 with the final line
`DTYPE PHASE 2 ORACLE: PASS`. Its tested manifest inherits the complete Phase
1 oracle; exercises all sealed numeric-kernel and frozen trap-string tests;
runs the IR and host structural exclusivity locks; runs declared-width
ordinary, windowed, and argument-reduction behavior including negative
parity at every integer width and reverse-mode overlap accumulation at every
float arithmetic width; runs the active eval matrices for [#680], precision,
exact int64 values, and static-condition folding; and finishes with the exact
prover-carrier boundary. It never runs ignored rows or Phase 3 backend suites.

## Phase 3 - backends adopt; the observation channel is generated

**You inherit:** frozen everything (§C1-§C5); eval as the reference
RENDERER whose printed strings are the expected values for yours (the
rendering contract, not the value contract - values are owed to
[04-NUM-8] by both lanes independently).

**Implementation status:** draft PR #1065 delivers the first bounded C-kernel
row on top of #1054. Direct signed-integer `abs` uses its declared width for
int8/int16/int32/int64 in both the tensor emitter and scalar host emitter,
traps on each width's minimum before C negation, and emits §C2's byte-exact
diagnostic generated from `NumericTrap`. Integer `abs` stays materialized
until the general fused-integer emitter carries the same semantics;
externally supplied fused integer-`abs` IR remains loud. This also makes
[#722]'s compiled integer-`abs` gradient row green. It does not deliver the
other integer kernels, generated observation helper, or Phase 3 oracle, and
it does not widen the HIP/Metal scope.

**You deliver:**

1. Inherit [#730]'s completed
   `HostTypeTerm -> ConcreteHostType -> HostAbiType` boundary: exact narrow
   scalar identity already survives host lowering, `int8`/`int16` already
   select their exact existing C integer ABIs, and the former
   `HostType::Unknown -> int64_t/void*` route is unrepresentable. This phase
   adds the per-dtype operation semantics and overflow traps for those integer
   ABIs. For f16/bf16 it adds exact C storage and rounding, then changes the
   target decision from structured rejection to the new ABI representation
   ([#714]); it does not reopen or duplicate the host-type boundary.
2. Scalar C arithmetic at width with generated trap guards emitting §C2's
   frozen strings (the `chelis_int_div_guard` pattern, generalized), and
   f16/bf16 scalar C storage/rounding matching §C1 (likely via uint16
   payload + the same conversion helpers the WS-1 kernels already use).
3. **The generated print helper**: the emitted C tensor/scalar printers
   are produced from `format_element`'s per-dtype logic (a Rust function
   emitting the C switch, exhaustive over `Prim`, `default:` aborts with
   the dtype id). `to_list`, print, and the wire schema now agree in the
   compiled lane (§C4.4).
4. The [#687] cross-lane oracle turned fully on: byte-identical expected
   strings for every cell in the matrix files, tolerance only where
   §C4.5's table says so (that table is authored into spec/05 §8 per
   [05-OBS-3]; §C4.5 is this doc's pointer to it, not a second home).

**Frozen at your exit:** §C4 C-side parity - both lanes print identical
bytes. This is [#728]'s acceptance and the precondition Phase 4's generated
conformance matrix asserts against.

**Explicitly not yours:** which cells EXIST (capability decisions like
[#724]/[#726] arrive in Phase 4; until then those cells stay ignored with
their issue numbers); HIP/Metal kernel work (none needed).

**Oracle:** the C rows of `narrow_dtype_matrix.rs` and
`int_width_lane_matrix.rs`, `scalar_stub_matrix.rs`'s dtype rows, the
[#723] row in `reduction_and_bitwise_matrix.rs`, and
`fold_static_cond_matrix.rs`'s C rows - green and un-ignored; plus the
cross-lane byte-diff pass over the whole corpus (print/to_list/wire vs
stored bits, both lanes, every dtype).

## Phase 4 - the capability table becomes the permanent guard

The table's SCHEMA is owned by `spec/design/capability_table.md` (the
two-table design: semantic table A, per-backend table B; per-Prim rows;
scalar/tensor surfaces separate; atom citations mandatory) - read it
before this phase; its seed-decision list is this phase's work-list.

**You inherit:** two agreeing lanes and a hand-curated matrix of tests.

**You deliver:**

1. The `const` op x dtype table (`Supported | Rejected(&'static str)`),
   with the never-authored cells decided on the record: integer `mean`
   ([#724] - reject, widen, or authored floor-mean), bool arithmetic
   ([#726] - proposal default: reject, diagnostics pointing at explicit
   casts), scalar floor/ceil/round on ints ([#715]'s three-lane row - the
   checker's existing stance says integer-valid; make eval and C honor
   it or change the stance, once, here).
2. Checker acceptance derived from the table (delete the hand-mirrored
   lists, e.g. `TRANSCENDENTAL_FLOAT_ONLY_OPS` becomes a table view).
3. Backend dispatch skeletons macro-generated from the table: a
   `Supported` cell with no kernel is a compile error in that backend; a
   kernel with no cell is dead code the build flags.
4. The generated conformance suite: every `Supported` cell executed in
   every lane asserting exact agreement (or the spec/05 §8 tolerance row
   §C4.5 points at), every
   `Rejected` cell asserting the same diagnostic from every lane. This
   suite REPLACES the hand-written matrix files as the standing guard;
   the audit files remain as regression archaeology.
5. The pre-table [#912] `BuiltinDecl.realizability` declarations and target
   capability sets become generated projections of Tables A/B. The root
   manifest may consume those projections and the checked root set, but it
   cannot remain an independently authored builtin/backend authority.

**Frozen at your exit:** the table schema and the rule that lanes derive
from it. After this phase, "add a builtin" without deciding every lane is
a build failure, which is the class-level end state.

**Explicitly not yours:** relitigating §C1 semantics (frozen since
Phase 1).

**Oracle:** the generated matrix is the named suite (this phase's single
authoritative oracle per the repo contract); mutation check: deleting any
lane's arm for a Supported cell must fail the BUILD, not just the tests.

---

## I1. Interlock with loud unsupported ([#730])

The plans share representation identities and backend call sites but own
different decisions:

- `RuntimeDType` and `Repr` in `chelis-vocab` own stable ABI representation
  identity. `RuntimeDType` owns numeric IDs, external spelling, and the mapping
  to `Repr`. `Repr` describes each current physical encoding, and byte width
  derives from it. `Repr` does not select a storage format. This document owns
  finalization, value domains, the §C3 storage decision, cast behavior,
  operation legality, and kernel behavior.
- `loud_unsupported.md` defines how every negative decision reaches the user.
  This document defines correct behavior for supported cells. A cell may move
  from silently wrong to loudly rejected under [#730], then to correctly
  implemented under this plan; it may never move through a substituted value.
- `HostTypeTerm -> ConcreteHostType` preserves checked logical identity and
  does not consult a backend. `ConcreteHostType -> HostAbiType` consumes the
  target implementation decision. Table B is the permanent authority for
  that decision.
- Before Table B is generated, [#730] may use only a private exhaustive target
  adapter whose negative decisions cite a spec atom or implementation issue.
  Phase 4 replaces those decisions without changing the HostType/ABI boundary.
- Table A rejections are reported by the checker because they are
  target-independent. Table B `Unimplemented` and `RejectedByDesign` cells are
  reported by build/lowering where the target is known, using [#730]'s
  diagnostic contract.
- [#912]'s root manifest owns root identity, order, and artifact routing, not
  operation legality. Its pre-table builtin realizability and target sets are
  exhaustive adapters; Phase 4 replaces their hand-authored decisions with
  generated Table-A/Table-B projections.
- `KNOWN_TAGS` is deliberately outside this interlock. It classifies Deep
  syntax and must become an exhaustive typed `DeepTag` disposition under
  [#908]/[#731], rather than being generated from numeric capability rows.

Neither plan may duplicate the other's authority. Any change to this boundary
updates this section, `loud_unsupported.md` §I1, and
`capability_table.md` in the same change set.

---

# Part IV - bookkeeping

## Issue map

| phase | goes green / becomes unwritable |
|---|---|
| 0 | detection for everything below; [#687] partially |
| 1 | [#684], [#717], [#720] (with Phase 2's fold work), [#724] eval half, [#726] eval half |
| 2 | [#680], [#688], [#711], [#718] eval cells, [#722] eval half |
| 3 | [#714], [#715] dtype rows, [#716], [#718] C cells, [#723], [#728]; [#687] fully unblocked |
| 4 | [#692], [#712], [#715] lane skew, [#724]/[#726] authored, future lane skew as a class |

Orthogonal, do not wait: [#703]'s loud-fallback discipline (now its own
plan: `spec/design/loud_unsupported.md`, tracking [#730], whose §I1
pins the interlock with this document) and [#709]'s
DeepTag/EffectKind enums (`numeric_audit_structural_prevention.md` items
3 and 5), the tripwires (item 8), and [#699]/[#725]'s raise-instead-of-
substitute fixes (needed for [#722]'s C half regardless of this plan).

## Open questions and where they get decided

| # | question | decided in | recorded where |
|---|---|---|---|
| 1 | per-dtype buffers vs finalize-on-write f64 | DECIDED 2026-07-17: per-dtype buffers. The proposal default is ratified: f64 storage cannot meet [#684] by construction, and the 2026-07 review endorsed per-dtype `TensorStorage` as the structural core | §C3 of this doc + the PR |
| 2 | integer `mean` / bool arithmetic / int floor-ceil-round capability rows | cells DECIDED 2026-07 on the issues ([#724] reject; [#726] reject + first-class `count`; [#712]/[#715] support - see capability_table.md's seed rows); Phase 4 ratifies each as an atom | capability table + spec/05 |
| 3 | trap surface form and exact strings | Phase 2 | §C2 + `pub const` in the module |
| 4 | crate placement | DECIDED 2026-07-17: a `chelis-types` MODULE. `Prim` already lives there (`types.rs`); the checker already consumes value-domain semantics (literal range diagnostics today, table-A acceptance at Phase 4); every §C5 consumer already depends on the crate; and §C3's privacy contract is module-scoped (`pub(in dtype_semantics)`), so the firewall is identical to a crate boundary. Constraint check passed: chelis-runtime stays dependency-light (libc+memmap2 only) - the generated helpers are emitted by chelis-backend-c, and C-side parity is enforced by tests, not a link edge. Discipline: the module stays import-clean (only `Prim` + std from the surrounding crate) so a later lift to a leaf crate remains mechanical. This also fixes [#732] Phase 1's `format_element` placement as FINAL (its §C3.1 pre-[#729] fallback is the answer - no Wave 2 -> Wave 3 migration) | §C5 + this doc + faithful_observation.md §C3.1 |
| 5 | wire-schema versioning mechanics for the storage change | DECIDED in the Phase 1 implementation: `EXECUTION_VALUE_SCHEMA_VERSION = 2` in `schema.rs`. The tensor payload is the tagged per-dtype `TensorElements` (`{"dtype": ..., "values": [...]}`; integer families exact at width, f16/bf16 as their exact f64 images, bool as true/false); `EvalResult` stamps `schema_version` (serde default 1 on deserialize, so a version-less payload identifies a v1 producer loudly); v1 clients posting the old bare-array `data` binding fail loudly at serde (a type error at the payload position; the untagged-enum message does not name the field), never a reinterpretation. The manifest is NOT bitten: `chelis_manifest_spec.md` carries type strings, not `ExecutionValue` payloads. The constant is independent of `WIRE_DAG_SCHEMA_VERSION` (which still governs `WireDag`) | schema.rs |

[#387]: https://github.com/Chelis-Lang/chelis/issues/387
[#680]: https://github.com/Chelis-Lang/chelis/issues/680
[#682]: https://github.com/Chelis-Lang/chelis/issues/682
[#684]: https://github.com/Chelis-Lang/chelis/issues/684
[#685]: https://github.com/Chelis-Lang/chelis/issues/685
[#686]: https://github.com/Chelis-Lang/chelis/issues/686
[#687]: https://github.com/Chelis-Lang/chelis/issues/687
[#736]: https://github.com/Chelis-Lang/chelis/issues/736
[#737]: https://github.com/Chelis-Lang/chelis/issues/737
[#753]: https://github.com/Chelis-Lang/chelis/issues/753
[#759]: https://github.com/Chelis-Lang/chelis/issues/759
[#775]: https://github.com/Chelis-Lang/chelis/issues/775
[#688]: https://github.com/Chelis-Lang/chelis/issues/688
[#692]: https://github.com/Chelis-Lang/chelis/issues/692
[#695]: https://github.com/Chelis-Lang/chelis/issues/695
[#696]: https://github.com/Chelis-Lang/chelis/pull/696
[#699]: https://github.com/Chelis-Lang/chelis/issues/699
[#703]: https://github.com/Chelis-Lang/chelis/issues/703
[#705]: https://github.com/Chelis-Lang/chelis/issues/705
[#709]: https://github.com/Chelis-Lang/chelis/issues/709
[#711]: https://github.com/Chelis-Lang/chelis/issues/711
[#712]: https://github.com/Chelis-Lang/chelis/issues/712
[#714]: https://github.com/Chelis-Lang/chelis/issues/714
[#715]: https://github.com/Chelis-Lang/chelis/issues/715
[#716]: https://github.com/Chelis-Lang/chelis/issues/716
[#717]: https://github.com/Chelis-Lang/chelis/issues/717
[#718]: https://github.com/Chelis-Lang/chelis/issues/718
[#719]: https://github.com/Chelis-Lang/chelis/issues/719
[#720]: https://github.com/Chelis-Lang/chelis/issues/720
[#722]: https://github.com/Chelis-Lang/chelis/issues/722
[#723]: https://github.com/Chelis-Lang/chelis/issues/723
[#724]: https://github.com/Chelis-Lang/chelis/issues/724
[#725]: https://github.com/Chelis-Lang/chelis/issues/725
[#726]: https://github.com/Chelis-Lang/chelis/issues/726
[#727]: https://github.com/Chelis-Lang/chelis/issues/727
[#728]: https://github.com/Chelis-Lang/chelis/issues/728
[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#731]: https://github.com/Chelis-Lang/chelis/issues/731
[#732]: https://github.com/Chelis-Lang/chelis/issues/732
[#908]: https://github.com/Chelis-Lang/chelis/issues/908
[#912]: https://github.com/Chelis-Lang/chelis/issues/912
[#1023]: https://github.com/Chelis-Lang/chelis/issues/1023
