# Grounded Dtype Semantics

**Status:** Active phased plan. Phases 0-3 and their nested oracle chain have
landed. Phase 4A's covered-family capacity ratchet has landed. This change
completes Phase 4B by deciding its named language contracts and freezing the
typed capability schema. The exact builtin-atom closure [#1294] is a hard
pre-4C gate. Phase 4C's machine tables, Phase 4D's generated consumers,
and Phase 4E's generated conformance product have not landed.
Tracking issue: [#729].
**Owning specs:** `spec/04-type-system.md` (the authored overflow/rounding and
per-dtype value contract), `spec/05-risc-primitives.md` (op result semantics),
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
dtype-tagged payloads (exact-only v6 with loud rejection of corrupt
reduced-float images), the bincode caches bumped
(`CHELIS_CTX_V6`, stdlib format 3), and constant folds decline rather
than bake a collapsed integer or a trap in. The [#878] follow-through
extends the same carrier to `RiscOp::Pad.fill` and `WireRiscOp::Pad.fill`:
WireDag v6 carries a typed `ScalarValue`; a missing version, every v1-v5
payload, every future version, and every raw-number Pad spelling is rejected
before node decode. There is no migration grammar. The serialized-shape change bumps the
bincode caches to `CHELIS_CTX_V7` and stdlib format 4. Partial adoption of the
storage decision is forbidden: it is the one all-layers-or-nothing
element of this plan, because a mixed state re-creates the very
boundary bugs ([#684]/[#686]) it exists to end. The mechanical
no-sixth-layer census over every remaining f64/Vec<f64> payload field
is the in-tree `crates/chelis-cli/tests/issue_729_payload_census.rs`, whose
scope is `RiscOp`/`WireRiscOp` carrier fields rather than every numeric form
in the repository.

**The Pad fill is not an exemption ([#878]).** `Pad.fill` is finalized once at
the padded tensor's dtype during lowering and remains tagged through eval,
wire transport, proving, and every backend emitter. Exact int64 values above
2^53 therefore never acquire an f64 image. The carrier replacement shrinks both guards:
the IR payload census now requires the sealed `ScalarValue`, and §C6's typed
wire manifest no longer carries the former FLAGGED `float-carrier` row. The
WireDag v6 consumer break remains a cross-repository coordination fact:
Beacon advertises only versions 1-3, so the chelis launcher must reject that
mismatch through [#708]'s version-negotiation gate rather than dispatching v6
bytes optimistically.

**Normative home for the GUARANTEE this delivers:**
`spec/04-type-system.md` [04-NUM-11] - a value survives storage,
transport, and every boundary crossing at its declared dtype without
collapse. That atom is what a later reader cites; this section owns the
mechanism that achieves it (per-dtype buffers, private constructors, the
sealed types) and is free to change form as long as the atom keeps
holding.

**The four layers are a census, not a closed list.** Phase entry re-derives
the layer list by enumerating every public channel that carries numeric values
- eval storage, wire schema, Python payloads, prove's environment, registered
value ADTs, exported stdlib definitions, and published runtime-header exports
- and the storage decision covers all of them in one change set. The final JSON
surface has one public value identity: `io/json::Json` with tagged
`JsonInt(int64)` and `JsonFloat(f64)` variants under [05-OP-2]/[05-OP-34].
[#1293] removes the duplicate prelude `Json`/`JInt`/`JNum` surface and moves CSV
to its untyped text-table contract before Phase 4C. §C6 owns the standing guard
that keeps this census from silently growing stale.

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
  strict rule) found no valid use of the old nonzero-to-1 encoding. Boolean
  counting is the dedicated [05-OP-29] `count` operation; an arithmetic cast
  composition is not the language contract.
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
[#861]'s naming decision set still owns the diagnostic freeze). The explicit
non-default conversion ladder is fully decided: [05-OP-6] governs
`cast_trunc`, [05-OP-23] governs `cast_saturate`, and [05-OP-24] governs
`cast_wrap`. Each spelling is greppable, has its own exact type, trap, and AD
contract, and receives ordinary capability-table rows and cross-lane oracle
coverage. [#759] owns implementation of those authored contracts, not their
spelling or semantic authority. The default `cast` remains checked and no
named form creates a compatibility mode for it.

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
   in VALUE (libm vs SLEEF vs vForce, > 0.5 ulp ops), the per-op bound is
   authored in `spec/05-risc-primitives.md` §8 and represented by
   `chelis_types::agreement::OP_TOLERANCES`; the [#687] oracle consults that
   machine form. `sqrt` is required correctly rounded ([#719]) and therefore
   has an explicit zero-bound row. A row is eligible only when both lanes
   compute at [04-NUM-8]'s declared arithmetic width; [#897]'s current eval
   float path is not eligible. A differing f16/bf16 result additionally
   requires both pre-final f32 bit patterns and evidence that each rounds to
   its observed stored value; finalized strings alone do not prove an f32 ULP
   distance. Formatting itself never has tolerance.
   The Phase 1 `f64` `tan`/`exp` cross-lane controls use `1e-12` only as an
   implementation-chosen test margin: macOS and glibc differed by one ulp in
   the observed repros, while the defect those controls detect (computing an
   `f64` program through `f32`) differs by roughly `1e-7`. `1e-12` was an
   arbitrary separating margin, not a language decision, not an
   [05-OBS-3] tolerance row, and not authority for another operation. Those
   controls are looser than the authored `tan`/`exp` rows and remain
   implementation controls; they are not the [05-OBS-3] oracle.
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
dtype-free, post-ratchet runtime/stdlib numeric-callable authority
registration, public-header context invariance, and issue-kind-aware liveness
are executable tripwires. Its grandfather, permanent-disposition, successor-
override, and integer-plumbing lists are transitional implementation debt,
not accepted end states: chelis#1288 removes them and chelis#1293 aligns the
recursively discovered stdlib definitions before Phase 4C authority
population may begin. The
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
  (`new_post_ratchet_bare_int_export_is_numeric_op_and_requires_registration`).
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
- Integer-valued plumbing is not an exemption class. A public callable whose
  signature contains an arithmetic value is a numeric operation and owes an
  exact semantic registration, even when the value represents an extent,
  allocation size, or dtype selector. Raw dtype selectors are replaced by the
  tagged carrier from chelis#1289. The final tripwire contains no
  `NON_NUMERIC_INTEGER_PLUMBING_EXPORTS` path. Published ABI still has exactly
  one preprocessing context; this rule changes classification, not that
  invariance.

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
3. **Ops**: a new numeric operation exists only with a decided row in the
   registry that owns its FAMILY. Numeric Table A and the sibling builtin
   registry's exact keys stay the LANGUAGE-BUILTIN law;
   runtime exports, prelude/stdlib defs, and binding functions do not
   fit those keys because they have no `BuiltinId`. Container and boundary
   builtins use the sibling registry rather than inventing a List-valued
   scalar/tensor surface (PR #950 red team P1-2). Post-ratchet growth totality
   is
   therefore REGISTRIES PER FAMILY, each exhaustive over new members of its
   own enumerated surface:
   language builtins in Table A, with checker acceptance DERIVED from
   it at Phase 4 so an unregistered builtin is `UnknownForm` by
   construction; new runtime exports and exported prelude/stdlib defs in
   deliverable 1's structured operation-semantic registry; binding
   callables in the same registry shape once the rustdoc-JSON leg
   lands. Each registered non-Table-A entry binds the callable's exact
   canonical identity to one verbatim `[05-OP-N]` authority. The registry
   validates chapter `05`, group `OP`, and a normative definition line
   beginning `> **[05-OP-N]**`; a free-text chapter substring or
   cross-reference, a missing `[05-OP-999]`, or an observation atom such
   as `[05-OBS-1]` is not semantic registration. Tooling does not
   infer whether the selected existing OP atom is semantically relevant;
   review verifies that its normative text already governs the callable but
   cannot make a mismatched atom authoritative. If no atom governs the
   callable, the numbered spec gains the decision first. Every discovered
   numeric callable, regardless of age, authors or cites its governing
   `[05-OP-N]` atom and exact mapping. A rename, signature change, or
   reclassification removes the old registered identity and adds a fully
   registered successor; resemblance confers no authority. The final census
   has zero grandfather, permanent-disposition, successor-override, or
   integer-plumbing exception rows. The
   implementation record is `SemanticRegistration { callable, atom }` in
   `SEMANTIC_REGISTRATIONS`; `callable` is exactly
   `[<kind>] <canonical id>`, so family identity is part of the key.
   Atom allocation re-checks the highest existing `[05-OP-N]` on current
   `main`; parallel branches do not reserve numbers. After adding the
   normative atom, the same change runs
   `.venv/bin/python scripts/generate_rejection_registries.py --write` and
   commits `crates/chelis-types/src/rejection_registry_generated.rs`. That
   generated membership artifact keeps rejection-authority validation aware
   of the new atom; it does not replace the callable's exact
   `SemanticRegistration` or create semantic authority.
   This registration is the callable's semantic authority. Backend delivery
   may separately carry an open issue in the target-disposition registry, but
   that issue never substitutes for semantic registration. An operation
   registered in no family's registry is a build failure, not a doc comment
   (`capability_table.md` §New numeric ops
   carries the interim authoring rule).

   **Language legality and backend capability stay decoupled, here as
   everywhere** (the standing Table A / Table B split,
   `capability_table.md` §Derivations): registering an op decides what
   is legal in Surf, Deep, and the RISC DAG - a target-independent
   fact the CHECKER reports. What a given execution target can run is Table
   B's, the sibling backend product's, the external target registry's, or an
   exported-stdlib body's derived dependency closure's separate, later
   decision: a language-legal op a target cannot
   run is a build-stage capability rejection through [#730]'s
   `Unsupported` channel where the target is known, NEVER a checker
   type error. Both stages' diagnostic machinery already exists
   ([#731]'s witnessed checker diagnostics for language-level
   rejections; [#730]'s staged `Unsupported` for target-known ones);
   the family registries must preserve that stage split rather than
   collapse a backend gap into language illegality.

   Numeric-ness is STRUCTURAL, never declared: any callable whose
   canonical signature mentions a numeric dtype requires a row and an
   authority binding. The non-numeric classification exists only for genuinely dtype-free
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
| runtime tensor data (`chelis_tensor.data`) | sealed typed access path plus exact tagged public carrier | chelis#893 and chelis#1289; prerequisite to [#729] Phase 4C | types |
| execution wire schema | per-dtype tagged payload | [#729] Phase 1 (§C3) | types |
| language ops | checker acceptance derived from Table A | [#729] Phase 4 | derivation |
| published C signatures | generated typed dtype and scalar carriers, tokenized canonical declaration inventory over callables and non-function data, exact semantic registrations, and a mechanically enforced ban on context-varying public ABI | chelis#1289 and chelis#1288 | types + census/registry |
| prelude / stdlib value ADTs | shape-complete census; exact tagged carriers are structurally recognized and every numeric callable/constructor has exact semantic registration. Untagged float carriers are illegal regardless of age | chelis#1288 | census/registry |
| binding (PyO3) signatures | rustdoc-JSON registry + typed raw-dtype mutation oracle | the named pre-Phase-1 binding leg below | census/registry |

[#729] Phase 4 cannot close while the chelis#893 access seal or chelis#1289
exact public carrier prerequisite remains open. A scoped or compatibility-
qualified C6 completion claim is not permitted.

Deliverables, with phase homes:

1. **Now, pre-Phase-1: the capacity census + tripwire.** PR #956
   checks the covered families generated from actual artifacts:
   published-header declarations and struct layouts, stdlib ADT
   numeric carrier shapes, and exported runtime/stdlib numeric
   callables. The header legs use
   `preprocessed_headers -> header_rows`; the stdlib legs use exactly
   `stdlib_rows -> scan_deftypes + scan_exported_numeric_defs`. The exported-
   definition scan expands every nominal ADT and container in a public
   signature recursively to a fixed point. It classifies the definition as
   numeric when any reachable field is numeric, and treats a precision type
   variable in a tensor element or linked scalar position as numeric; a
   direct-primitive-only scan is incomplete and fails its mutation control.
   The prelude / stdlib value-ADT family has a second enumeration
   source: **Rust-registered prelude ADTs**
   (`register_prelude_adts` in `crates/chelis-types/src/builtins.rs`)
   enumerate through `prelude_adt_rows -> chelis_types::prelude_adt_defs`
   as the `prelude-adt-numeric` leg, classified on the identical
   float-carrier / numeric-op rule with a shape-complete identity.
   The final public JSON ADT is exported from stdlib and is discovered by the
   recursive `.ch` enumerator. The Rust leg must therefore contain no numeric
   prelude JSON row after [#1293]; retaining `JInt`, `JNum`, or a second `Json`
   identity fails the zero-exception oracle. A prelude ADT registered outside
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
   not inspect or redesign private runtime-dtype decoding. The typed `DeepTag`
   lane disposition and Deep stamping are outside this task entirely.

   Binding-baseline dispositions (each entry is the C6 review a frozen
   descriptor-manifest update cites):

   - 2026-08-02, chelis#816 (PRs #819/#822): `compile_and_load` gains
     `project_root: Option<&str>, force_bare: bool` and `eval_json` gains
     `project_root: Option<&str>` for reef-context resolution. All three
     parameters are dtype-free control/path inputs (a filesystem path and a
     lane selector); the enumerator classifies both rows `[]`, no numeric
     capacity enters the surface, and no raw dtype id is introduced.

   **Zero-exception closure.** Closing the plan requires every discovered row
   to be structurally nonnumeric, a structurally recognized exact tagged
   carrier, or an exactly registered numeric operation. Age, an earlier
   review, an old issue, and an unchanged descriptor are not dispositions.
   `PERMANENT_PLAIN_ROWS`, `GRANDFATHER_SEAM_ROWS`, successor overrides, and
   integer-plumbing exemptions are deleted rather than frozen. The
   source-faithful `io/json::Json` carrier is legal because its tagged
   `JsonInt`/`JsonFloat` shape is recognized and its constructors bind to
   [05-OP-2]/[05-OP-34], not because any descriptor predates the ratchet.
   Typed wire and PyO3 rows follow the same structural/registration rule.

   The final C/runtime authority partition is exact:

   - `Nonnumeric` contains only `chelis_file_exists`,
     `chelis_string_contains`, `chelis_string_ends_with`, `chelis_string_eq`,
     `chelis_string_starts_with`, `chelis_string_concat`,
     `chelis_string_from_cstr`, `chelis_string_trim`, `chelis_string_data`,
     `chelis_adt_get_tag`, `chelis_adt_tag_equals`, `chelis_fail`,
     `chelis_read_file`, `chelis_write_file`, `chelis_read_lines`,
     `chelis_list_dir`, `chelis_mmap_file`, and the opaque `chelis_string`
     layout.
   - `TaggedTransport` contains the exact `chelis_dtype`, `chelis_scalar`,
     `chelis_tensor`, `chelis_value_tag`, `chelis_value`,
     `chelis_option_scalar`, `chelis_option_value`, and `chelis_dict_entry`
     layouts; `chelis_free`; `chelis_string_retain`, `chelis_string_release`,
     `chelis_list_retain`, `chelis_list_release`, `chelis_tuple_retain`,
     `chelis_tuple_release`, `chelis_dict_retain`, `chelis_dict_release`,
     `chelis_adt_retain`, `chelis_adt_release`, `chelis_value_retain`, and
     `chelis_value_release`; `chelis_value_from_string`,
     `chelis_value_from_tensor`, `chelis_value_from_list`,
     `chelis_value_from_tuple`, `chelis_value_from_dict`,
     `chelis_value_from_adt`; `chelis_value_as_string`,
     `chelis_value_as_tensor`, `chelis_value_as_list`,
     `chelis_value_as_tuple`, `chelis_value_as_dict`, and
     `chelis_value_as_adt`;
     `chelis_list_empty`, `chelis_list_append`, `chelis_list_concat`,
     `chelis_list_flatten`, `chelis_list_zip`; and
     `chelis_dict_keys`, `chelis_dict_values`, and `chelis_dict_entries`.
     A transport row only moves an
     already validated tagged value and never sizes, compares, indexes,
     observes, or interprets its numeric payload.
   - `Numeric([05-OP-31])` contains exactly the ten scalar/dtype identities
     enumerated by [05-OP-31]; `Numeric([05-OP-32])` contains exactly its
     container/index/observation identities; and `Numeric([05-OP-33])`
     contains exactly its tensor-runtime identities. A name present in two
     groups or absent from all three fails the bijection.
   - Every exported numeric stdlib ADT maps to [05-OP-34], every exported
     numeric stdlib `def`
     maps to [05-OP-35]. Compiler-owned numeric builtins remain separately
     enumerated in Table A or the sibling registry and map to their exact
     controlling numbered-spec atoms.

   A new row must enter exactly one of those structural or semantic sets in
   the same change that exposes it. The sets have no wildcard, family-prefix
   match, issue citation, or default arm.

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
   - **The stdlib language families carry semantic classifications.** A
     `std-adt-numeric` or `std-def-numeric` row is classified, not merely
     inventoried. A declared Chelis primitive field retains its dtype tag:
     `f64`, `f32`, `f16`, `bf16`, and every signed-integer field therefore
     classify the owning public constructor or definition as `numeric-op`.
     Every such row, including the frozen initial rows, requires the same
     exact semantic registration as a runtime callable. `float-carrier` is
     reserved for an external carrier that erases the language dtype, such
     as an exported bare C `double`; it does not describe a typed Chelis ADT
     field. Before this, the ADT leg emitted no flags at all, so adding
     `| JsonBigNum(f64)` to `io/json.ch` could land by regenerating and citing
     an open issue. The final control instead rejects that addition because
     the new constructor identity has no exact operation registration; the
     paired integer and float mutations prove the rule uniformly. The
     source-faithful `JsonInt(int64)` and `JsonFloat(f64)` variants are both
     wanted tagged numeric operations, not carrier seams.
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
     discovered and keyed by exact canonical callable identity. Stdlib
     discovery follows nominal ADT/container fields recursively to a fixed
     point, so a callable accepting `Decimal`, `Date`, `Duration`, `Json`,
     `Tokenizer`, or a container that reaches one of them cannot disappear
     merely because its signature has no directly spelled primitive. A
     tensor precision variable and every linked scalar occurrence are numeric
     for the same reason. Their
     separate semantic registry names one exact `[05-OP-N]` atom and
     validates its chapter/group and normative `> **[05-OP-N]**`
     definition. Every numeric callable, including every row inherited from
     the initial census, has an exact registration. Positive controls bind
     registered callables to their decisions.
     Negative mutations add an unregistered runtime export and stdlib
     `export def`, name absent `[05-OP-999]`, and substitute
     `[05-OBS-1]`; all fail. An issue citation or a bare chapter
     substring is never semantic authority. The STDLIB half of that
     claim had no control until round-4 red team N6, which is why it is
     named twice now: a new exported numeric def cited with only an
     issue fails, and - the branch that binds `std-def-numeric` by KIND
     rather than by flags - a float-only def remains illegal until it is
     semantically registered. Its declared Chelis float type is already a
     tagged language value, not an untagged carrier seam
     (`a_new_stdlib_numeric_def_requires_semantic_registration`,
     `a_stdlib_registration_against_a_nonexistent_atom_fails`). The tool does not infer
     semantic relevance within the OP group; review verifies the
     numbered-spec decision but cannot create it.
   - **A row cannot self-bless and a seam cannot be cited into existence.**
     Regeneration fails on an unclassified row. A bare numeric carrier is
     redesigned onto the exact tagged carrier or removed; neither an issue
     citation nor a maintainer override authorizes it. A numeric callable
     without semantic registration fails. An identity change is removal plus
     addition, and the successor independently satisfies the final rule.
   - **Liveness is issue-typed.** Every
     sanctioned `chelis#N` reference must exist, must be an ISSUE rather than
     a pull request, and must be OPEN. An invented reference-free disposition,
     or a closed, missing, or PR reference, fails and
     forces re-adjudication. `capacity_census_liveness.py` represents
     the result as `IssueRecord { kind: IssueKind, state: IssueState }`;
     `fetch_issue` calls
     `gh api repos/Chelis-Lang/chelis/issues/N` and treats the REST
     payload's `pull_request` field as the wrong object kind. The unit
     controls include an open issue, closed issue, missing issue, and
     open PR.
     The division of labour is deliberate and stated here so nobody
     infers more from a green tripwire run than it proves: **the tripwire
     checks structural carrier recognition, exact semantic registrations,
     and syntax for issue-backed target dispositions**. It runs offline;
     network access would make it flaky and unrunnable in a sandbox.
     **Existence, kind, and open-state are the
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
     here is the executable descriptor/registry freeze plus the
     ordinary required review; this document claims no more.
2. **Phase 1**: the census re-derivation defines §C3's atomic set (the
   §C3 amendment above); the tripwire baseline regenerates in the same
   change set.
3. **Phases 2-3, coordinated with chelis#893 and chelis#1289** (and PR #964's merged `Repr` vocabulary (2026-07-31; chelis#894 is the tracking issue)): the families with typed end states per the table above
   become types; each bare seam is removed or redesigned onto the exact tagged
   carrier. The known `(double, int)` seams are the kill-list,
   starting with `chelis_format_shortest`'s pair
   (`faithful_observation.md` §I1 records it) and any survivor of PR
   #891's rework. The typed access seal and carrier are prerequisites to this
   plan's Phase 4C entry.
4. **Phase 3**: the `RuntimeDType` header FRAGMENT is already
   generated with a byte-for-byte regeneration test ([#730] §C4.3);
   the census asserts the fragment-owned rows against regeneration
   rather than diff. The rest of the public header is structurally
   guarded by canonical inventory, a total authority map for every descriptor,
   exact semantic registration for every numeric callable, and the no-context-
   variance rule; it is NOT generated, and a hand-added export remains
   representable but cannot pass the tripwire without final classification.
   FULL header generation
   is in no phase contract today. Adopting it would be a stronger
   B1-protocol amendment to Phase 3, not a prerequisite silently inferred
   from the standing census (red team P1-3).
5. **Phase 4**: the totality leg over the reachable surface, plus the
   table's mandatory atom citation, make every numeric op force
   spec authorship - the authoring-forcing function generalized from
   dtype cells to op EXISTENCE. The added-variant mutation oracle's
   build set is derived from cargo metadata (every workspace member
   depending on `chelis-vocab`), never hand-listed; the chelis-python
   escape in the sweep is the motivating instance ([#730] §C6 owns the
   oracle; this row records the derivation rule).

Scope honesty, sharpened by the 2026-07-30 and 2026-07-31 adversarial
passes. On the
families PR #956 covers, additions, removals, classification drift,
missing semantic registrations on new numeric callables, ADT-shape growth,
ADT and stdlib-def
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
| §C6 covered-family census + tripwire | canonical row identities, complete derived classifications, stdlib ADT shapes and capacity flags, exact callable-to-`[05-OP-N]` registrations for every numeric identity, structural recognition of exact tagged carriers, public-header context invariance, and a total bijection from every discovered row to `Nonnumeric`, `TaggedTransport`, or `Numeric(atom)` with zero exception rows | this doc + the executable tripwire/registry artifacts and their positive/negative controls, same change set |
| §C6 typed wire/PyO3 leg state | before Phase 1 entry: frozen when each named enumerator and mutation command is green; an editable baseline field cannot change coverage | this doc + the typed leg manifest + owning enumerator/oracle in the same change set |
| capability table schema | Phase 4B | `capability_table.md` (the owning doc) + this doc + every named consumer plan |

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

**Oracle:** `.venv/bin/python scripts/dtype_phase0_oracle.py` is this phase's
single authoritative command. Acceptance is exit 0 with the final line
`DTYPE PHASE 0 ORACLE: PASS`. Its tested manifest runs the frozen domain
checker and every active dtype matrix carrying a detector chokepoint, the
byte-exact executable-example parity corpus with the fallback removed, and
the retained floating agreement controls. It never opts into ignored rows.
The command executes continuously by inheritance through the Phase 2 oracle.

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
2. **The spec/04 section**: spec/04 §9 carries the timeless decided contract,
   including [04-NUM-14]'s checked-cast rule. Implementation status remains in
   this plan and current-state documents, never in the normative section.
   Chelis#733 Phase 1 separately migrates the authority form and revisions
   through the pinned Buoy shell-side integration.
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
   Phase 4D replaces this interim policy with the generated Table-A view.

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
`PHASE 1 ORACLE: PASS`. Its tested command manifest first inherits the
complete Phase 0 oracle, then runs the three §C6 entry censuses; the sealed
semantics, typed Load, and execution-wire exactness controls; the Phase
1-specific checker/payload controls; and the Python/Hull tagged-wire readers.
It never repeats Phase 0's matrix commands, runs ignored rows, or pulls in
compiled-C phase work. The exact scalar-kernel rows for [#680], [#688],
[#718], and [#722] remain Phase 2 obligations; this oracle does not pull them
across the kernel-split boundary.

## Phase 2 - the kernel split and prove

**At phase entry you inherit:** the module (frozen §C1/§C3/§C4-Rust), eval
as reference lane, and the not-yet-split `host_ops` helpers now visibly
awkward (they finalize but still accept `Fn(f64,f64)`).

**Implementation status:** PR #1054 landed this phase as one
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
1 oracle and therefore Phase 0; exercises all sealed numeric-kernel and frozen
trap-string tests;
runs the IR and host structural exclusivity locks; runs declared-width
ordinary, windowed, and argument-reduction behavior including negative
parity at every integer width and reverse-mode overlap accumulation at every
float arithmetic width; runs the active eval matrices for [#680], precision,
exact int64 values, and static-condition folding; and finishes with the exact
prover-carrier boundary. It never runs ignored rows or Phase 3 backend suites.
The required Linux Integration job runs this command after the normal
`scripts/gate.py integration` stage, so all three phase contracts are
continuous without changing the developer gate's command set.

## Phase 3 - backends adopt; the observation channel is generated

**You inherit:** frozen everything (§C1-§C5); eval as the reference
RENDERER whose printed strings are the expected values for yours (the
rendering contract, not the value contract - values are owed to
[04-NUM-8] by both lanes independently).

**Implementation status:** PR #1065 landed the first bounded C-kernel row on
top of #1054. This revision completes the phase's C value layer. Exact narrow
scalar ABI types survive as `uint16_t` f16/bf16 payloads; scalar and tensor
arithmetic finalize at the declared float width; integer scalar, tensor,
cast, and reduction paths use checked helpers with §C2's byte-exact traps.
Integer tensor operations stay materialized at the checked DAG boundary, so
the float-only fused emitter cannot bypass their guards. The existing tagged
rank-0 tensor carrier boxes non-f64 floats without adding a public numeric ABI
channel. Exact-bit literal emission retires [#751]'s four C-ingress corpus
exclusions, host lowering finalizes suffixed literal leaves before widening
([#1110]), and exact f32 subnormal ingress closes [#761]. [#713], [#714],
[#715]'s authored float rows, [#718], [#699], [#691], and [#865] leave their
Phase 3 red/ignored state on their original assertions. The same exact narrow
scalar storage returns f16/bf16 `to_string` to the own-width observation
corpus ([#734]) through private generated helpers, without adding a public
numeric ABI channel. Post-phase maintenance makes the decided scalar rows for
[#704], [#712], and [#715] behavior-preserving before table generation:
float activations and scalar transcendental/rounding operations execute in
eval and C at every active float width, scalar integer `floor`/`ceil`/`round`
are exact identities at every signed width, and scalar `max_elem`/`min_elem`
execute at every admitted numeric width. The permanent Table-A/Table-B cells,
generated dispatch, and generated conformance product remain Phase 4 work;
this maintenance does not widen the HIP/Metal scope.

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
   f16/bf16 scalar C storage/rounding matching §C1 via exact `uint16_t`
   payloads and the same conversion helpers the WS-1 kernels use.
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

**Explicitly not yours:** the permanent generated inventory of which cells
exist (Phase 4C records the already-decided interim rows and Phase 4D derives
their consumers); HIP/Metal kernel
work (none needed).

**Oracle:** `.venv/bin/python scripts/dtype_phase3_oracle.py` is this phase's
single authoritative command. Acceptance is exit 0 with the final line
`DTYPE PHASE 3 ORACLE: PASS`. Its tested manifest inherits the complete dtype
Phase 2 oracle and the faithful-observation Phase 3 oracle; runs the C rows of
`narrow_dtype_matrix.rs`, `int_width_lane_matrix.rs`,
`scalar_stub_matrix.rs`, `precision_matrix.rs`,
`reduction_and_bitwise_matrix.rs`, `fold_static_cond_matrix.rs`, the checked
cast/subnormal/reduced-float `to_string` fixtures, and the complete observation
harness; runs the C
emitter structural locks; and finishes with the numeric-surface capacity
censuses. It never opts into Phase 4's ignored capability cells or HIP/Metal.
The required Linux Integration job invokes this nested oracle after the
normal integration gate, so Phases 0-3 remain continuous without duplicating
their commands in `scripts/gate.py`.

## Post-Phase-3 checked-cast maintenance ([#1150], [#1152])

This is a named maintenance item under [#729], not a new phase and not a
license to patch either reported site independently. [04-NUM-14] already
decides checked conversion and [04-NUM-15] already decides multi-offender
order. The implementation must make those rules one construction shared by
every checked-cast lane.

The capability schema expands checked `cast` over the finite
`source Prim x target Prim x surface x backend` product. Every admitted pair
has an exact conversion plan; identity is legal only when source and target
are the same Prim. The evaluator finalizer, C DAG emitter, and C host emitter
consume that plan or a mechanically equivalent generated projection. No lane
may carry an `_ => input` arm or invent its own source/target roster. [#730]
LU6 owns the typed host-emission boundary and its rejection rendering; this
item owns which checked conversion a supported cell performs.

Elementwise trapping conversions produce a shared semantic value equivalent
to:

```text
IndexedTrapCandidate { flat_index, trap }
```

where `flat_index` is the row-major flat index governed by [04-NUM-15] and
`trap` is the exact [04-NUM-14] failure at that element. A lane reduces
candidates by minimum `flat_index` and reports that candidate's trap. The
evaluator therefore has no domain-first whole-buffer pre-pass. A parallel C
lane accumulates a private candidate per worker and reduces those candidates
after the parallel region; it never races on a shared trap flag and need not
serialize correct element conversion. Empty candidate sets complete
successfully.

The work item lands atomically with a generated interim conformance matrix
over every active source/target pair and both scalar/tensor surfaces in eval,
C DAG, and C host lanes. Each applicable pair has positive in-range coverage
and negative fractional, non-finite, and overflow coverage. Mixed-offender
tensors permute trap kinds across lower and higher flat indices and run the C
case at multiple thread counts; all lanes select the lowest index. Structural
mutations replace one non-identity conversion with identity, restore the
domain-first pre-pass, and make the C winner schedule-dependent; each is red.
Phase 4 generates this same product permanently from Tables A/B rather than
retaining a cast-specific hand list.

**Implementation receipt.** The interim construction is
`chelis_types::CheckedCastPlan`, whose exhaustive active-`Prim` product feeds
sealed scalar/tensor evaluation and the shared C conversion-expression
projection used by both emitters. Tensor evaluation reduces explicit
`IndexedTrapCandidate` values. The C DAG emitter performs an OpenMP
`min(flat_index)` reduction without calling an aborting helper in a worker,
then reclassifies the selected element after the parallel region; C host
tensor emission uses the same plan in row-major order and has no identity
fallback. C DAG identity casts materialize logical row-major order from the
source strides rather than copying a view's backing order.
`issue_759_checked_cast_default.rs` generates the complete 9 x 9 x
three-surface positive matrix for eval and compiled C, locks direct
f64/int64-to-f16/bf16 rounding with midpoint witnesses on scalar, DAG tensor,
and host tensor surfaces, and runs both mixed-offender permutations at
`OMP_NUM_THREADS=1,2,4,8`. The plan-level negative matrix covers every
applicable fractional, non-finite, overflow, and strict-bool case. A
compile-run `permute -> same-type cast` regression guards logical
materialization. This maintenance item remains part of the inherited
`.venv/bin/python scripts/dtype_phase3_oracle.py`; Phase 4 still replaces the
interim generator with Tables A/B.

## Phase 4 - the capability table becomes the permanent guard

The schema is owned by `spec/design/capability_table.md`. It keeps `Prim`,
`BuiltinId`, and capability policy in `chelis-types`; separates semantic Table
A from per-backend Table B; expands finite semantic parameters per `Prim` and
scalar/tensor surface; routes container/boundary builtins to the exact sibling
registry while retaining §C6's external-family semantic registries; gives
runtime and binding callables total external target dispositions; gives exact
effect dependencies a separate typed backend disposition; derives an exported
stdlib definition's executability transitively from its checked body; and
composes host ABI types through a separate constructor table.
Phase 4 is split so table authoring, generated routing, and executable coverage
cannot be conflated.

### Phase 4A - capacity-ratchet permanence (landed)

**Delivered:** §C6's covered-family census, typed coverage manifests,
callable-to-`[05-OP-N]` semantic registrations, public-header invariance, and
the shrink-only flagged-seam rule. This prevents a new numeric channel or
callable from bypassing review while the capability tables are built.

### Phase 4B - decided-contract and schema freeze (this change)

**Delivered:**

1. This slice freezes the complete timeless [05-OP-1..39] authority set,
   including exact decimal rounding; source-faithful JSON/CSV boundaries;
   checked, truncating, saturating, and wrapping casts; runtime shape and
   movement values; all-active-dtype random and dropout parameters; recursive
   List/ADT padding and host-value adjoints; `mean`, extrema, product,
   argument reductions, and canonical multi-axis sum/count; modular
   arithmetic; float classification; scalar/tensor/recursive-List `to_string`; boolean
   `and`/`or`/`not`; NaN-aware comparison/equality; exact scalar/container/C
   tensor carriers; the 83-definition stdlib manifest; legal compiled host
   effects; and target-independent runtime reduction windows. Logical
   operations are bool-only and do not alias numeric primitives; arithmetic
   reductions reject bool and `count` is the dedicated bool-tensor reduction.
   The exact scalar, container, tensor-runtime, exported numeric ADT, and
   exported numeric stdlib families are [05-OP-31..35], including their final
   breaking C identities and corrected stdlib signatures.
   `sum`, `prod_reduce`, and `count` use the canonical adjacent-pair balanced
   tree, with no stride-4 compatibility mode. The extrema contract divides every non-NaN tie,
   including equal infinities, routes a NaN cotangent to the first NaN selected
   by the forward rule, and applies the same rule to windowed extrema.
2. `capability_table.md` freezes the exact typed Table-A/Table-B cells,
   backend set, companion primitive-leaf and constructor tables, exact sibling-builtin key and
   cells, external-family semantic and target routing, exact effect-disposition
   registry, exported-stdlib dependency derivation, ownership boundary, and
   macro-expanded machine form. There are no open schema questions after this
   slice.
3. Behavior-changing implementation work remains assigned to v0.19 and may
   not be disguised as table population. The v0.20 table mechanism records
   honest `Unimplemented` cells until those implementations land. [#1284]
   owns replacing the pre-table boolean numeric aliases used by logical
   operations, comparison-derived negation, and `where`.

**Frozen at exit:** the numbered-spec atoms authored or amended by this slice
and the capability schema. This is not a claim that every older builtin already
has exact atom authority: [#1294] owns that all-or-nothing closure before 4C.
Changing a frozen atom or the schema follows §B1; table implementation may not
reinterpret either.

**Authoritative 4B oracle:**
`.venv/bin/python scripts/dtype_phase4b_oracle.py`; exit 0 and final line
`DTYPE PHASE 4B ORACLE: PASS`. It validates the normative atoms in this slice,
named-cast exclusion, typed numeric, sibling, and effect schema markers, frozen
contract digests, phase naming, and generated rejection-registry agreement.
Normative and schema inputs carry complete-file integrity fingerprints, and
narrower atom/section fingerprints supply owning diagnostics. Updating a
design or status document's integrity fingerprint is ordinary synchronized
maintenance, not a semantic decision; only a change to the controlling
numbered-spec atoms or frozen schema follows §B1. Roadmap and current-state
documents are validated through required contract markers instead of whole-
file semantic fingerprints.
Its success proves this freeze, not complete builtin-atom closure or any Phase
4C implementation.

### Pre-4C - exact builtin-atom closure ([#1294])

Before any Phase 4C key/cell type, authoring macro, or partial machine row may
land, [#1294] first introduces the closed builtin domain/case declaration types and
attaches a non-empty exhaustive declaration to every `BuiltinDecl`. It then
discovers the union of every canonical Table-A IR/RISC operation identity and
every declared `BuiltinDecl` sibling domain/case and proves an exact
bijection from every Table-A and sibling-builtin identity to one
semantically governing normative `[05-OP-N]` line. It authors every missing
atom, regenerates the rejection registry, and admits no count allowlist,
unnumbered table/prose authority, issue citation, default, alias, age, or
compatibility exception. An open implementation issue can authorize only a
later Table-B `Unimplemented` receipt; it never satisfies semantic closure.

**Authoritative closure oracle:**
`.venv/bin/python scripts/dtype_builtin_atom_closure_oracle.py`; exit 0 and
final line `DTYPE BUILTIN ATOM CLOSURE ORACLE: PASS`. The oracle and its
adversarial mutations must be green and merged before Phase 4C begins.

### Pre-4C - composite executable gate ([#1296])

After the individual behavior, storage, census, and [#1294] atom-closure
oracles land, [#1296] wires every exact oracle in the `v0.19 behavior` row —
including [#722], [#753], [#759], [#893], [#965], [#1059], [#1281], [#1282],
[#1284], and [#1287]-[#1298] — into
`.venv/bin/python scripts/dtype_pre_phase4c_oracle.py`. Success ends with
`DTYPE PRE-PHASE-4C ORACLE: PASS`. The runner fails for a missing, duplicate,
skipped, stale, nonzero, or success-line-free leg and includes the structural
mutations named by each child. It is wired to the normal gate; prose coverage
or a manual waiver is not an entry receipt.

### Phase 4C - populate the machine authorities

**Entry condition:** the [#1296] composite pre-4C oracle is green, merged, and
wired to the normal gate. It includes [#1294]'s exact builtin-atom closure and
every acceptance oracle in the `v0.19 behavior` row of the issue map. In
particular, chelis#1288's zero-exception census,
chelis#893/chelis#1289's typed carrier, chelis#1290's balanced reductions,
chelis#1287/chelis#1291's first-class count cells, chelis#1292's own-width
tensor comparison, chelis#1293's complete 83-definition stdlib alignment,
chelis#1295's all-active-float random/rounding rules, chelis#1297's compiled
host effects, and chelis#1298's runtime-axis/window operations have landed.
No grandfather, permanent-disposition, successor-override,
integer-plumbing, bare numeric-carrier, legacy callable, or semantics-divergent
registered identity remains. Phase 4C may not populate tables around a
compatibility exception or record a cell before its controlling behavior is
conformant. This is the executable requirement for zero capacity exceptions.

**You deliver:**

1. Closed Rust key and cell types in `chelis-types`: Table A keyed by
   `(BuiltinId, SurfaceClass, operand Prim, SemanticParams)` and Table B over
   every A-`Supported` row times `eval | c-host | c-dag | hip | metal`.
2. Compact authoring macros that expand to the complete machine rows. A
   missing or duplicate expansion, new `Prim`, undeclared `BuiltinDecl`
   domain, absent atom, or unsupported backend hole fails construction or
   compilation.
3. Fully populated Tables A/B, the recursive host-constructor table, and the
   sibling builtin registry. Its exact key is `(BuiltinId, SiblingDomain,
   SiblingCaseId, SemanticParams)` and its supported rows expand over the same
   backend set; §C6's external-family semantic registry remains separate, and
   runtime/binding callables populate the exact
   `(ExternalCallableFamily, CanonicalCallableId, ExternalTargetContext)`
   target-disposition registry. Exact effect dependencies populate the
   `(CanonicalEffectRequirement, BackendId)` disposition registry over
   `Random | Accum | Io | Test | Resource(ResourceId)`. Table A and sibling
   semantic cells use typed signature, result, atom, and diagnostic identities;
   backend and effect cells use typed implementation, issue, or
   rejected-by-design authorities.
4. Missing, duplicate, stale, or defaulted effect rows fail construction. An
   explicit `CompleteEffectDependencies::Pure` is distinct from an absent or
   unfinished checked-body traversal.
5. An exact `SpecAtomRef` lookup for every row, consuming [#1294]'s already
   complete generated membership. Phase 4C does not author or backfill atoms.

**Authoritative 4C oracle (supporting the overall Phase 4 oracle):**
`.venv/bin/python scripts/dtype_phase4c_oracle.py`; exit 0 and final line
`DTYPE PHASE 4C ORACLE: PASS`.

### Phase 4D - replace hand-authored consumers

**You deliver:** checker acceptance and reporting, early build gates, backend
dispatch skeletons, [#912] root-realizability projections, checked host-cast
planning, recursive host-ABI resolution, exact effect-disposition lookup, and
exported-stdlib dependency closures generated from the owning registries and
checked bodies. Delete each hand-mirrored list only after its generated
consumer is live. A root capability may derive from numeric Table B, the host
constructor table, or a sibling registry; the root manifest authors none of
those decisions.

**Authoritative 4D oracle (supporting the overall Phase 4 oracle):**
`.venv/bin/python scripts/dtype_phase4d_oracle.py`; exit 0 and final line
`DTYPE PHASE 4D ORACLE: PASS`.

### Phase 4E - generated conformance and class elimination

**You deliver:** an executable product over every Table-A semantic cell,
Table-B backend cell, finite parameter, surface, sibling semantic/backend
cell, legal host-constructor composition, external target disposition, effect
disposition, and derived exported-stdlib/backend result. Supported/implemented cells
execute with exact agreement or the one owning tolerance rule;
rejected/unimplemented cells assert the typed diagnostic at every rendering
stage. The suite includes structural mutations
for a missing A row, missing B row, stale atom, missing dispatch arm, omitted
constructor nesting, missing or defaulted effect row, incomplete effect
traversal, extrema finite-tie/NaN/infinity-tie routing, and a missing generated
case.

Finite-difference checks cover smooth reduction points and the two-way tie
subgradient. General `k`-way extrema ties, including equal positive and
negative infinities, additionally assert permutation symmetry and that routed
cotangents sum to the upstream cotangent; no plan may claim that generic
finite differences prove the nonsmooth `g/k` convention.

**Authoritative Phase 4 oracle:**
`.venv/bin/python scripts/dtype_phase4_oracle.py`; exit 0 and final line
`DTYPE PHASE 4 ORACLE: PASS`. It invokes the 4B, 4C, and 4D oracles and the
4E product. Phase 4 is complete only after this command passes and a fresh local
red-team agent validates the exact head. Supporting-oracle success alone is
not phase completion.

**Explicitly not Phase 4's job:** relitigating §C1 semantics or declaring an
`Unimplemented` cell implemented because a partial pre-table path happens to
pass a happy-path test.

---

## I1. Interlock with loud unsupported ([#730])

The plans share representation identities and backend call sites but own
different decisions:

- `RuntimeDType` and `Repr` in `chelis-vocab` own stable ABI representation
  identity. `RuntimeDType` owns numeric IDs, external spelling, and the mapping
  to `Repr`. `Repr` describes each current physical encoding, and byte width
  derives from it. `Repr` does not select a storage format. `Prim`,
  `BuiltinId`, and capability policy remain in `chelis-types`; they do not
  move to or duplicate into vocab. This document owns finalization, value
  domains, the §C3 storage decision, cast behavior, operation legality, and
  kernel behavior.
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
  Phase 4C populates the typed cells and Phase 4D replaces those decisions
  without changing the HostType/ABI boundary.
- The companion host-ABI constructor table composes recursive host types from
  closed constructor and child dispositions. [#730] LU4 owns the total typed
  resolver and rejection channel; this plan owns the backend dispositions.
  Neither plan may enumerate concrete `Option`/`List` nestings as policy.
- Checked `cast` is a finite source x target x surface Table-A product; Table
  B adds the backend. This plan owns conversion semantics and
  `IndexedTrapCandidate`'s lowest-flat-index selection ([#1150], [#1152]);
  [#730] LU6 owns exhaustive host-emission planning and the typed negative
  channel. Identity is not a fallback.
- Table-A and sibling semantic rejections are reported by the checker because
  they are target-independent. Table-B, sibling-backend, and external-target
  `Unimplemented`/`RejectedByDesign` cells, plus typed failures derived from an
  exported stdlib body's dependencies, are reported where the execution target
  is known, using [#730]'s diagnostic contract.
- [#912]'s root manifest owns root identity, order, and artifact routing, not
  operation legality. Its pre-table builtin realizability and target sets are
  exhaustive adapters; Phase 4D replaces their hand-authored decisions with
  projections generated from numeric Table B, the companion host-constructor
  table, or the callable's sibling registry.
- `deep_tag_lane_contribution` is deliberately outside this interlock. It
  classifies Deep syntax through an exhaustive typed `DeepTag` disposition
  under [#908]/[#731], rather than being generated from numeric capability
  rows.

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
| 4A-4B | §C6 capacity detection; [#898] reduction authorities; [#753]/[#759]/[#965] language decisions; [05-OP-29] first-class `count` authority; canonical reduction-order authority; WireDag v6 schema freeze |
| pre-4C authority and executable closure | [#1294] closed exhaustive `BuiltinDecl` domain/case declarations plus exact `[05-OP-N]` authority for every discovered Table-A IR/RISC operation and sibling-builtin identity; [#1296] one normal-gate composite over every prerequisite oracle and structural mutation; no machine key/cell type, authoring macro, or row may land first |
| v0.19 behavior | [#1290] balanced sum/product backend work (also part of [#170]); [#1281] mean/extrema/argument-reduction and windowed-extrema behavior; [#722] remaining compiled integer unary/AD cells; [#753]/[#759]/[#965] numeric callables; [#1282] [05-OP-25] scalar/tensor/recursive-List `to_string` domain; [#1059] compiled C-host Tensor/List rendering cells; [#1284] typed non-numeric logical/comparison/`where` lowering; [#893]/[#1289] typed C carrier; [#1288] zero-exception census; [#1287]/[#1291] exact-only WireDag v6 plus first-class count delivery; [#1292] own-width tensor-close assertions; [#1293] the exact 83-definition stdlib, sole public JSON surface, pathwise random/List adjoints, and stub removal; [#1295] all-active-float `round_to`/`uniform_like`/`dropout` and all-dtype padding; [#1297] legal compiled host-effect operations; [#1298] runtime-axis shape and target-independent window reductions |
| 4C-4E | [#692], [#712], [#715] lane-skew mechanisms; [#724]/[#726] generated policy; future lane skew as a class |
| maintenance | [#878] delivered the internal typed Pad carrier but not the exact-only v6 break owned by [#1287]; [#937] delivered the earlier f64 sampler repair but [#1295] owns the final same-dtype parameter contract; [#1150]/[#1152] are one checked-cast source x target construction with [#730] LU6 owning only host-emission totality and rejection rendering |

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
| 2 | integer `mean` / bool arithmetic / int floor-ceil-round capability rows | DECIDED: [#724] reject; [#726] rejects bool arithmetic, [05-OP-29] supplies first-class `count`, and [#712]/[#715] support the integer identity rows. No cast-counting compatibility path exists. Phase 4B freezes the corresponding atom/schema bindings | capability table + spec/05 |
| 3 | trap surface form and exact strings | Phase 2 | §C2 + `pub const` in the module |
| 4 | crate placement | DECIDED 2026-07-17: a `chelis-types` MODULE. `Prim` already lives there (`types.rs`); the checker already consumes value-domain semantics (literal range diagnostics today, table-A acceptance at Phase 4); every §C5 consumer already depends on the crate; and §C3's privacy contract is module-scoped (`pub(in dtype_semantics)`), so the firewall is identical to a crate boundary. Constraint check passed: chelis-runtime stays dependency-light (libc+memmap2 only) - the generated helpers are emitted by chelis-backend-c, and C-side parity is enforced by tests, not a link edge. Discipline: the module stays import-clean (only `Prim` + std from the surrounding crate) so a later lift to a leaf crate remains mechanical. This also fixes [#732] Phase 1's `format_element` placement as FINAL (its §C3.1 pre-[#729] fallback is the answer - no Wave 2 -> Wave 3 migration) | §C5 + this doc + faithful_observation.md §C3.1 |
| 5 | wire-schema versioning mechanics for the storage change | DECIDED: WireDag schema v6 is explicitly present and exact-only. Versionless, v1-v5, and future payloads are rejected before node decode; there is no legacy Pad migration or additive-variant compatibility. `WireRiscOp::Count { axes }` is a v6 variant. `EXECUTION_VALUE_SCHEMA_VERSION` remains a separately typed payload contract and never substitutes for WireDag validation | spec/10 §3 + schema.rs |

[#170]: https://github.com/Chelis-Lang/chelis/issues/170
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
[#878]: https://github.com/Chelis-Lang/chelis/issues/878
[#688]: https://github.com/Chelis-Lang/chelis/issues/688
[#692]: https://github.com/Chelis-Lang/chelis/issues/692
[#695]: https://github.com/Chelis-Lang/chelis/issues/695
[#696]: https://github.com/Chelis-Lang/chelis/pull/696
[#699]: https://github.com/Chelis-Lang/chelis/issues/699
[#703]: https://github.com/Chelis-Lang/chelis/issues/703
[#705]: https://github.com/Chelis-Lang/chelis/issues/705
[#708]: https://github.com/Chelis-Lang/chelis/issues/708
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
[#878]: https://github.com/Chelis-Lang/chelis/issues/878
[#898]: https://github.com/Chelis-Lang/chelis/issues/898
[#908]: https://github.com/Chelis-Lang/chelis/issues/908
[#912]: https://github.com/Chelis-Lang/chelis/issues/912
[#937]: https://github.com/Chelis-Lang/chelis/issues/937
[#955]: https://github.com/Chelis-Lang/chelis/issues/955
[#965]: https://github.com/Chelis-Lang/chelis/issues/965
[#1023]: https://github.com/Chelis-Lang/chelis/issues/1023
[#1150]: https://github.com/Chelis-Lang/chelis/issues/1150
[#1152]: https://github.com/Chelis-Lang/chelis/issues/1152
[#1281]: https://github.com/Chelis-Lang/chelis/issues/1281
[#1282]: https://github.com/Chelis-Lang/chelis/issues/1282
[#1284]: https://github.com/Chelis-Lang/chelis/issues/1284
[#1294]: https://github.com/Chelis-Lang/chelis/issues/1294
[#1295]: https://github.com/Chelis-Lang/chelis/issues/1295
[#1296]: https://github.com/Chelis-Lang/chelis/issues/1296
[#1297]: https://github.com/Chelis-Lang/chelis/issues/1297
[#1298]: https://github.com/Chelis-Lang/chelis/issues/1298
