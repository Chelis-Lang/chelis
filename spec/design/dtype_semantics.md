# Grounded Dtype Semantics

**Status:** Design proposal, pre-implementation. Tracking issue: chelis#729.
**Owning specs:** `spec/04-type-system.md` (gains an authored overflow/rounding
section, today silent), `spec/05-risc-primitives.md` (op result semantics),
and the audit record in `docs/investigations/numeric_audit_next_sweeps.md` /
`docs/investigations/numeric_audit_structural_prevention.md`.
**Class fixed:** chelis#727 (no dtype's semantics are enforced at any single
point), subsuming chelis#695 (the integer instance). Sibling classes #703,
#709, #728 have their own fixes; #728's falls out of Phase 3 here.

## Summary

Chelis has no component that owns what a dtype *means*. Rounding, width,
overflow, value domain, comparison, and formatting are decided independently
at every op x lane x surface site - and the 2026-07 numeric audit measured
the result: forty-plus verified defects across every dtype family, with each
lane correct exactly where another is wrong, and three prior rounds of
correct-but-local fixes that did not stop the class (chelis#695's history).

This proposal introduces one: a **dtype semantics layer** that is the single
source of truth for per-dtype behavior, made *unavoidable* rather than
available. The mechanism is not "a helper everyone should call" - the audit
found four correct helpers sitting uncalled next to their bug sites - but
representation types whose constructors are private, so that producing a
numeric value without passing through the semantics is a compile error.

The refactor is phased so each phase lands green against an oracle that
already exists: the ~70 `#[ignore]`d red tests on PR #696 that assert
correct behavior per cell, plus the exact-string cross-lane harness (#687)
and a domain-validity invariant. This is spec-first development applied to a
refactor: the failing acceptance surface was written before this design.

## Why a refactor and not another patch

Recorded so the next reader does not have to re-derive it:

1. **The patch strategy has a measured failure history.** Round 1
   (`TensorElement`, PRs #64/#67/#72) built the right abstraction and
   deferred two dtypes by explicit decision. Round 2 (`checked_int_binop`,
   #387) built the right helper and wired 4 of ~17 call sites. Round 3's
   audit found `tensor_float_unop_f32`, `convert_cast_data`, and
   `fold_static_size`'s `checked_i64` - each correct, each optional, each
   with siblings that do not call it. Optionality is the root cause, and
   optionality is the one thing a patch cannot remove.
2. **There is no reference lane to patch toward.** eval-scalar rounds f16
   correctly while eval-tensor does not (#717); C-tensor wraps int8 while
   C-scalar does not (#718); prove's interpreter collapses what the SMT
   tier keeps exact (#688). "Make X match Y" is undefined when no Y holds
   the semantics.
3. **Dtype discipline is the language's stated value proposition.** The
   spec's differentiators - no implicit precision promotion, explicit
   casts, named dimensions - are precision-centric promises. A numeric
   layer with no grounded notion of `f16` or `int8` contradicts the
   product's own core claim.

## Non-goals

- **Not** the fix for #703 (unsupported cases must fail loudly - dispatch
  and fallback discipline, `numeric_audit_structural_prevention.md` item 3)
  or #709 (checker holes - item 5). Those are cheaper, independent, and
  should not wait for this.
- **Not** a numerics-accuracy project. Transcendental ulp bounds (#719's
  vvsqrtf, SLEEF vs libm differences) get a documented per-op tolerance
  table as part of the formatting/oracle contract, not new kernels.
- **Not** a change to surface syntax or the type checker's user-facing
  rules, except where the capability table (Phase 4) surfaces cells that
  were never authored (integer `mean` #724, bool `add` #726) - each of
  those becomes an explicit spec decision rather than a lane accident.

## Design

### D1. The semantics module

A new module (working name `chelis_types::dtype_semantics`; may graduate to
a `chelis-dtype` crate if the dependency graph wants it) defining, for every
`Prim`, in one place:

```rust
pub struct DtypeSemantics {
    /// Is this f64 value a member of the dtype's value set?
    /// (f16: exactly representable in binary16; int8: integral in
    /// [-128, 127]; bool: 0.0 or 1.0.)
    pub fn contains(prim: Prim, value: f64) -> bool;

    /// Round/narrow a wide intermediate into the dtype, or trap.
    /// f16/bf16/f32: round-to-nearest-even at the dtype's mantissa
    /// (single rounding from f64 is correctly rounded for all three,
    /// the 2p+2 rule the eval scalar lane already relies on).
    /// f64: identity. Integers: must be integral and in range, else
    /// NumericTrap::Overflow (the #680 contract: errors, not wraps,
    /// at every width, in every lane). Bool: must be 0 or 1.
    pub fn finalize(prim: Prim, raw: RawResult) -> Result<Bits, NumericTrap>;

    /// Exact integer operations at width (checked; trap on overflow)
    /// and float operations with one terminal rounding. The kernel
    /// SIGNATURES are the enforcement: an integer cannot reach an
    /// f64 kernel (see D3).
    pub fn int_binop(op: IntOp, a: i64, b: i64, prim: Prim) -> Result<i64, NumericTrap>;
    pub fn float_binop(op: FloatOp, a: f64, b: f64, prim: Prim) -> Result<Bits, NumericTrap>;

    /// THE printed form, used by every lane and every exit
    /// (print, to_list, wire schema, diagnostics). Integers print as
    /// integers; narrow floats print their exact narrowed value; the
    /// per-op transcendental tolerance table lives with this contract.
    pub fn format(prim: Prim, bits: Bits) -> String;
}
```

Two rules govern the module itself:

- **Exhaustive matches only.** Every `match` over `Prim` in this module and
  its consumers has no `_` arm (`numeric_audit_structural_prevention.md`
  item 3a, enforced by a chelis-lint rule). Adding a `Prim` variant makes
  the compiler enumerate every place that must decide what it means.
- **The spec moves with it.** `spec/04-type-system.md` gains the overflow
  and rounding section it currently lacks - the absence is why int8-wraps
  and int64-saturates coexisted unnoticed (#680). The module is the
  executable form of that section; divergence between them is a spec bug.

### D2. Private constructors: the semantics is unavoidable

`ScalarBits` (already a tagged union, already exact - the audit confirmed
the scalar front end is clean) keeps its shape but makes raw construction
private to the semantics module. `TensorValue` is the larger change: its
element buffer becomes private, constructed only via
`TensorValue::finalize(prim, raw_elems)` (bulk form of D1's finalize, so
hot loops finalize per-buffer, not per-element - see Performance).

After D2, "forgot to call the narrowing helper" - the literal mechanism of
#680, #717, and #720 - is code that does not compile. The compiler emits
the complete site work-list that three rounds of manual fixing never
assembled.

**The storage decision.** `TensorValue { data: Vec<f64> }` is currently
declared independently at three layers (`chelis-ir/src/eval.rs`,
`chelis-compiler-api/src/schema.rs`, `bindings/python/chelis/__init__.py`)
plus prove's `HashMap<String, f64>` env. Phase 1 makes ONE decision -
per-dtype buffers (the C runtime's `TensorElement` shape, extended to the
deferred `CHELIS_I32`/`CHELIS_BOOL` per PR #79's outstanding deferral) vs
f64 storage with finalize-on-write - and expresses it at all four layers in
the same change set, per #695's sequencing note ("one storage decision
expressed at three layers"). The proposal's default is **per-dtype
buffers**: finalize-on-write over f64 storage cannot represent exact int64
above 2^53 (#684) no matter how disciplined the writes are, so f64 storage
fails requirement one. The wire schema and Python boundary change with it
(#686, #685).

### D3. The kernel type split

Unchanged from #695's decided contract, restated because D1/D2 do not
subsume it: the shared evaluator helpers stop being typed
`impl Fn(f64, f64) -> f64`. Integer ops route through
`int_binop(i64, i64) -> Result<i64, _>`; float ops through the float
kernel plus terminal `finalize`. An integer operand reaching a float
kernel becomes `E0308` at compile time. This is also what makes the trap
contract implementable - an i64 overflow cannot be detected inside an
`Fn(f64, f64)` kernel that already lost the values.

### D4. Consumers

Every lane consumes D1 through D2/D3; none re-implements semantics:

| consumer | change |
|---|---|
| eval scalar (`host_ops.rs`) | kernels split (D3); results finalized (D2). The one already-correct surface (scalar f16/bf16 rounding) becomes shared instead of local. |
| eval tensor (`host_ops.rs`, `eval.rs`) | `tensor_float_unop_f32` and the un-narrowed binop paths (#717) replaced by bulk finalize; storage per D2. |
| `fold_static_cond` / `convert_cast_data` (`lower.rs`, `eval.rs`) | cast folding calls `finalize` (kills #720); integer condition folding uses `int_binop` and declines on trap (kills #711, mirroring `fold_static_size`). |
| prove (`concrete_eval.rs`, `obligation_engine.rs`, `opaque.rs`) | env becomes exact-value typed; the `as f64` flatteners (#688) fail to compile after D3. |
| C backend host lane (`host.rs`, `host_emit.rs`) | `parse_host_type` gains the narrow arms (#714) as a forced consequence of exhaustive `Prim` matches; scalar C storage per dtype with trap guards (#718, mirroring `chelis_int_div_guard`). |
| backend emitted code (print helper, dtype switches) | GENERATED from D1's exhaustive `format`/dispatch functions (`prevention` item 3b) - kills #716/#723 and gives both lanes byte-identical printed output, which is #728's fix and #687's precondition. |
| Metal / HIP | already the best-behaved consumers (typed kernels / clean rejection); they adopt the capability table (D5) but need no semantic change. |

### D5. The capability table (parallelizable)

One `const` table, op x dtype -> `Supported | Rejected(reason)`, from which
the checker's acceptance rules derive, backend dispatch skeletons are
macro-generated (a table row without a kernel is a compile error in that
backend), and a conformance test executes every Supported cell in every
lane asserting exact agreement, and every Rejected cell asserting the same
diagnostic. This retires lane skew as a class (#712, #715's int rows,
#724, #726, #692's panic-vs-silent split) and converts "add a builtin"
from silently-becomes-a-stub into the-build-breaks-until-every-lane-
decides. It can begin as a generated test over the CURRENT dispatch before
any migration, which makes it a Phase 0 detector as well as the Phase 4
end state.

## Performance

Two obligations, addressed by construction rather than hope:

- **Bulk finalize.** Tensor ops finalize per-buffer with a monomorphized
  per-dtype loop (or, with per-dtype storage, compute directly in the
  element type) - never a per-element dynamic dispatch. The C backend
  already works this way; eval adopting per-dtype buffers makes the
  reference lane's cost model match the compiled one.
- **The inner-loop budget stands.** `cargo test --workspace` stays under
  the ~60s contract; the conformance matrix (D5) is table-generated and
  cheap per cell, and the heavyweight cells (compile+run per dtype) stay
  in the per-crate integration tier that PR #696's harness already uses.

## Phases, each with its oracle

Per the repo contract: one authoritative oracle per phase, red tests first
(they exist), no phase claimed done on narrative progress.

**Phase 0 - detectors (afternoon-scale, before any refactor).**
Domain-validity invariant wired into the PR #696 lane drivers (every
element of every printed/returned tensor is a member of its declared
dtype's value set - mechanically catches 2049.0-in-f16, 187.5-in-int64,
2-in-bool, 200-in-int8), plus the #687 exact-integer oracle lanes.
*Oracle:* the invariant harness runs in CI and is red on today's known
cells (ignored), green on the clean ones.

**Phase 1 - the semantics module + eval adoption.**
D1 lands with exhaustive matches and the spec/04 section; eval scalar and
tensor adopt it (D2, including the storage decision at all four layers).
*Oracle:* the eval-side ignored tests flip green as a class -
`eval_tensor_narrowing_matrix.rs` (all cells), `narrow_dtype_matrix.rs`
eval rows, `precision_matrix.rs` eval rows, `int_width_lane_matrix.rs`
eval cells - with zero regressions in the locked controls (the ByDesign
float rows and the correct-scalar-rounding locks must NOT change).

**Phase 2 - the kernel split + prove.**
D3 in `host_ops.rs`; prove's interpreter and flatteners move to exact
values. *Oracle:* `issue_680_int_exactness.rs` and
`prove_int64_exactness.rs` ignored rows green; the overflow-trap rows trap
with the branded diagnostic in eval.

**Phase 3 - backends consume, observation channel generated.**
`parse_host_type` narrow arms, C scalar width+trap parity, generated print
helper and dtype switches (D4 backend rows). *Oracle:*
`scalar_stub_matrix.rs` dtype rows, `narrow_dtype_matrix.rs` C rows,
`reduction_and_bitwise_matrix.rs` #723 row, `fold_static_cond_matrix.rs`,
and cross-lane byte-identical printed output on the #687 corpus (the #728
acceptance: print, to_list, and the wire schema agree with the stored
bits, both lanes, every dtype).

**Phase 4 - the capability table as the permanent guard.**
D5 generation replaces hand-mirrored dispatch; the conformance matrix
becomes the standing per-PR guard. *Oracle:* the generated matrix is the
named suite; deleting any lane's arm for a Supported cell fails the build,
not the numbers.

Phases 1-3 each unblock issue clusters independently; none requires a
big-bang merge. The #703/#709 fixes (loud fallbacks, DeepTag enum) are
orthogonal and can land any time.

## Issue map

| phase | goes green / becomes unwritable |
|---|---|
| 0 | detection for everything below; #687 partially |
| 1 | #684, #717, #720, #724 (eval half), #726 (eval half), the #711 fold via int_binop |
| 2 | #680, #688, #718 (eval cells), #722 (via #699's raise + exact grad lane) |
| 3 | #714, #715 (dtype rows), #716, #718 (C cells), #723, #728, #687 unblocked |
| 4 | #692, #712, #715 (lane skew), #724/#726 (authored cells), future lane skew as a class |

## Open questions (decide in Phase 1, on the record)

1. Per-dtype buffers vs finalize-on-write f64 storage (proposal default:
   per-dtype; f64 storage cannot meet #684's exactness requirement).
2. Where `mean` on integer tensors lands in the capability table (#724's
   three options - reject, widen-to-float, or authored integer mean) and
   whether bool arithmetic exists at all (#726; proposal default: reject
   both, pointing at explicit casts).
3. Whether `NumericTrap` surfaces as the existing branded diagnostic
   string (`chelis_int_div_guard` precedent) or a structured error kind -
   both lanes must emit the same text either way (#687).
4. Crate placement (`chelis-types` module vs new `chelis-dtype` crate) -
   decided by whether chelis-runtime can depend on it for the generated
   C helper templates without a cycle.
