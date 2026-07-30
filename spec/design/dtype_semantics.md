# Grounded Dtype Semantics

**Status:** Design proposal, pre-implementation. Tracking issue: [#729].
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
- **Not** a change to surface syntax or user-facing checker rules, except
  where the capability table (Phase 4) surfaces cells that were never
  authored (integer `mean` [#724], bool `add` [#726]) - each becomes an
  explicit spec decision rather than a lane accident.

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
[04-NUM-12]. That table is what a later reader cites and is the only
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
    /// Division/remainder by zero (existing behavior, absorbed here).
    DivZero  { op: &'static str },
}
```

- **Message format (frozen at Phase 2 exit):**
  `numeric trap: <kind> in <op> at <prim>` following the branding precedent
  of `chelis_int_div_guard` / `integer division or remainder by zero`
  (`chelis-runtime/include/chelis_runtime.h:165-171`). The EXACT strings are
  recorded in the module as `pub const` and every lane emits them verbatim -
  the C lane via generated guard snippets (Phase 3), eval via the module
  directly. [#687]'s oracle compares them byte-for-byte.
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
construction. The decision is expressed at all four declaration layers in
ONE change set (Phase 1): `chelis-ir/src/eval.rs` (`TensorValue`),
`chelis-compiler-api/src/schema.rs` (wire schema - `data` becomes a tagged
per-dtype payload; this is a wire-format break, versioned as such),
`bindings/python/chelis/__init__.py` (per-dtype tuples / numpy dtypes,
ending the `np.float64` cast of [#685]), and prove's env (§C5-consumer
table). Partial adoption of the storage decision is forbidden: it is the
one all-layers-or-nothing element of this plan, because a mixed state
re-creates the very boundary bugs ([#684]/[#686]) it exists to end.

**Normative home for the GUARANTEE this delivers:**
`spec/04-type-system.md` [04-NUM-11] - a value survives storage,
transport, and every boundary crossing at its declared dtype without
collapse. That atom is what a later reader cites; this section owns the
mechanism that achieves it (per-dtype buffers, private constructors, the
sealed types) and is free to change form as long as the atom keeps
holding.

**Access for consumers.** Reads are free-form (`as_f64_lossy()` explicitly
named lossy, `as_i64_exact() -> Option<i64>`, typed slices per dtype).
Only *construction* is gated. Movement ops (reshape/permute/shrink) that
provably preserve elements may clone/re-slice storage without
re-finalizing via a `pub(crate) fn reuse_storage` escape hatch whose doc
contract is "element-preserving ops only"; every use site cites it. That
hatch is the ONE deliberate hole, kept greppable.

**The cast ladder (2026-07 review; [#759]).** The explicit `cast`
surface mirrors the read-side split above: the CHECKED cast is the
default - `convert_cast_data` is a thin wrapper over `finalize_scalar`
(consumer map), so a cast whose value does not survive the target dtype
traps per §C1/§C2 - and a NAMED lossy/truncating form ([#759]) is the
explicit escape hatch, the same species as [#753]'s `wrap_*`: never the
default, greppable, per-direction semantics AUTHORED as an atom rather
than inherited from a lane (proposal defaults: float->float is RNE at
the target width; float->int truncates toward zero with the
out-of-range rule authored, not accidental; int->narrower-int gets ONE
authored rule), with capability-table rows and cross-lane oracle
coverage like any other cell. Spelling and atom land with Phase 2's
kernel work; cells ratified at Phase 4.

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
   recorded in `spec/05-risc-primitives.md` next to the op, and the [#687]
   oracle consults it; `sqrt` is required correctly rounded ([#719]) and has
   no tolerance row. Formatting itself never has tolerance.
6. **Containers and scalar roots** (decided with [#732] Phase 1, identical
   to its §C1.5; ratified as [05-OBS-4]/[05-OBS-5]): a scalar-typed value
   renders as the BARE scalar at every exit in both lanes, including as a
   top-level labeled root; a rank-0 tensor renders as its single element,
   bare (`tensor(shape=[], data=[..])` is not an exit form - the [#775]
   decision). Tensor element rendering truncates after 32 elements with
   the `, ...` marker at every exit in both lanes; `to_list` and the wire
   never truncate.

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
| eval tensor (same file + `chelis-ir/src/eval.rs`) | `finalize_tensor` bulk paths; `tensor_float_unop_f32` and raw `binary_map` deleted | [#717], [#684], [#724] eval half, [#726] eval half |
| `convert_cast_data` (`chelis-ir/src/eval.rs`) | thin wrapper over `finalize_scalar` | [#717] cast rows, [#720] (via next row) |
| `fold_static_cond` / const folds (`chelis-ir/src/lower.rs`) | `int_binop` + finalize in the Cast arm; decline-on-trap | [#711], [#720] |
| prove (`concrete_eval.rs`, `obligation_engine.rs:1857`, `opaque.rs:1536`) | env `HashMap<String, ScalarValue>`; flatteners deleted (will not compile post-split) | [#688] |
| C host lane (`chelis-ir/src/host.rs`, `chelis-backend-c/src/host_emit.rs`) | `parse_host_type` narrow arms (forced by exhaustive `Prim`), per-dtype scalar storage + generated trap guards | [#714], [#718] C cells, [#715]'s dtype rows |
| C emitted helpers (print, dtype switches) | GENERATED from `format_element` / exhaustive matches | [#716], [#723], [#728] |
| Metal / HIP | capability table only (already honestly typed / cleanly rejecting) | - |

**Performance contract:** finalize is per-buffer monomorphized loops (or
direct element-type compute once storage is per-dtype), never per-element
dyn dispatch; `cargo test --workspace` stays inside the ~60s inner-loop
budget; the conformance matrix's compile+run cells live in the per-crate
integration tier, not the workspace loop.

---

# Part II - process rules that hold at every phase boundary

## B1. Freeze points

| contract | frozen at end of | may change after only by |
|---|---|---|
| §C1 semantics table + spec/04 section | Phase 1 | spec change + this doc + re-run of the full matrix |
| §C3 public API + storage layout + wire schema | Phase 1 | versioned schema bump, all four layers together |
| §C2 trap kinds + exact message strings | Phase 2 | this doc + [#687] corpus update in the same PR |
| §C4 formatting rules 1-4 | Phase 1 (Rust) / Phase 3 (C parity) | this doc + [#687] corpus update |
| §C5 kernel signatures | Phase 2 | this doc |
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

**You inherit:** Phase 0's detectors (your acceptance instruments) and
Part I as the spec of what to build.

**You deliver:**

1. The `dtype_semantics` module implementing §C1-§C4's Rust side:
   `finalize_scalar` / `finalize_tensor` / `scalar_from_*` /
   `format_element` / `NumericTrap`, with exhaustive `Prim` matches and
   unit tests per cell of the §C1 table (positive AND negative per the
   repo's negative-test-parity rule: every rounding case, every trap
   case, every special value).
2. **The spec/04 section**: spec/04 §9 carries the decided contract as
   current blockquote authorities [04-NUM-1..6] with an honest status banner
   (seeded ahead of this phase; implementation tracked here). This phase
   RATIFIES and refines those semantics (and their §C1/§C2 correspondence) in
   the same PR as the module, so spec and code cannot diverge at the moment the
   semantics become real. Chelis#733 Phase 1 separately migrates the authority
   form and revisions through the pinned Buoy shell-side integration.
3. **The storage decision at all four layers** (§C3): `TensorStorage`
   per-dtype buffers in eval, the versioned wire-schema change, the
   Python boundary, prove's env type swap can be deferred to Phase 2 ONLY
   if prove keeps compiling untouched (record which).
4. **Eval adoption**: eval scalar and tensor paths construct exclusively
   through the module (raw constructors are now private - the compiler
   gives you the site list; the PR description records the count).
   `tensor_float_unop_f32` and the raw f64 `binary_map`/`unary_map` paths
   are deleted, not deprecated.
5. Crate-placement decision (open question 4) recorded in this doc.

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

**Oracle:** `eval_tensor_narrowing_matrix.rs` fully green and un-ignored;
the eval rows of `narrow_dtype_matrix.rs`, `precision_matrix.rs`,
`int_width_lane_matrix.rs`, `reduction_and_bitwise_matrix.rs` ([#724]'s
eval half traps or is table-rejected - see open question 2, decided in
this phase) green and un-ignored; every control untouched; the Phase 0
domain checker green on ALL eval outputs, not just audited cells.

## Phase 2 - the kernel split and prove

**You inherit:** the module (frozen §C1/§C3/§C4-Rust), eval as reference
lane, and the not-yet-split `host_ops` helpers now visibly awkward (they
finalize but still accept `Fn(f64,f64)`).

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

**Oracle:** `issue_680_int_exactness.rs` and `precision_matrix.rs`
overflow rows green and un-ignored (eval side); `prove_int64_exactness.rs`
green and un-ignored; `fold_static_cond_matrix.rs` eval expectations and
the [#711] row green; the [#722] grad rows green in EVAL (grad's lowering now
folds/computes exactly; the C half of [#722] waits for Phase 3 plus [#699]'s
own fix, which is [#703]-track).

## Phase 3 - backends adopt; the observation channel is generated

**You inherit:** frozen everything (§C1-§C5); eval as the reference
RENDERER whose printed strings are the expected values for yours (the
rendering contract, not the value contract - values are owed to
[04-NUM-8] by both lanes independently).

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

- `RuntimeDType` in `chelis-vocab` owns stable ABI identity, external
  spelling, and byte width. It does not define finalization, value domains,
  storage, cast behavior, operation legality, or kernel behavior; those remain
  owned by this document.
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
| 5 | wire-schema versioning mechanics for the storage change | Phase 1 | schema.rs + `spec/design/chelis_manifest_spec.md` if it bites the manifest |

[#387]: https://github.com/Chelis-Lang/chelis/issues/387
[#680]: https://github.com/Chelis-Lang/chelis/issues/680
[#684]: https://github.com/Chelis-Lang/chelis/issues/684
[#685]: https://github.com/Chelis-Lang/chelis/issues/685
[#686]: https://github.com/Chelis-Lang/chelis/issues/686
[#687]: https://github.com/Chelis-Lang/chelis/issues/687
[#736]: https://github.com/Chelis-Lang/chelis/issues/736
[#737]: https://github.com/Chelis-Lang/chelis/issues/737
[#753]: https://github.com/Chelis-Lang/chelis/issues/753
[#759]: https://github.com/Chelis-Lang/chelis/issues/759
[#688]: https://github.com/Chelis-Lang/chelis/issues/688
[#692]: https://github.com/Chelis-Lang/chelis/issues/692
[#695]: https://github.com/Chelis-Lang/chelis/issues/695
[#696]: https://github.com/Chelis-Lang/chelis/pull/696
[#699]: https://github.com/Chelis-Lang/chelis/issues/699
[#703]: https://github.com/Chelis-Lang/chelis/issues/703
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
