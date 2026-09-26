import ChelisBridge.Protocol.Dropout
import ChelisBridge.Protocol.F16Widen
import ChelisBridge.Protocol.BF16Widen

/-!
# The corrected non-f64 dropout decision for every half-width dtype

For f32, f16 and bf16 storage the extracted `apply` widens the rate with `as_f64_lossy`
(exact for finite rates: `realize_widen_f32`, `realize_f16_to_f64`, `realize_bf16_to_f64`) and
compares `f64::from(unit as f32)` against it (`arithUnit`). The branch taken by
`apply_loop_spec` at index `i` is therefore "drop ⇔ value (unit as f32) < value w" for every
non-f64 dtype, not the reference "unitValue < r".

The apply-level theorems `mask_agrees_f16` / `mask_agrees_bf16` state the whole extracted output:
the kept value is the half-width quotient `xs[i] / (1 - w)` that `float_binop` computes through
f32 (`f16Div`, `bf16Div`), and the dropped value is the converted zero `cast_raw(Int(0))`.
The `*_naive_counterexample` theorems show that the reference exact-unit threshold is not what
Rust computes on these paths.
-/

namespace ChelisBridge.Random

open Aeneas.Std LaCaDiLECore ChelisBridge ChelisBridge.Protocol
open_chelis_model

/-- **Corrected non-f64 decision.** For an f32, f16 or bf16 rate `w` (finite) and every flat
index `i`, the widened rate Rust computes is exact and the loop's drop test equals
`value (unit as f32) < value w`. -/
theorem nonf64_decision (seed ordinal : U64) (i : ℕ) :
    (∀ w : F32, w.isFinite = true → ∃ wr,
      chelis_types.dtype_semantics.ScalarValue.as_f64_lossy ⟨.F32 w⟩ = .ok wr ∧
      F64.lt (arithUnit .F32 (unitAt seed ordinal i)) wr =
        decide (value (F64.toF32 (unitAt seed ordinal i)) < value w)) ∧
    (∀ w : F16, w.isFinite = true → ∃ wr,
      chelis_types.dtype_semantics.ScalarValue.as_f64_lossy ⟨.F16 w⟩ = .ok wr ∧
      F64.lt (arithUnit .F16 (unitAt seed ordinal i)) wr =
        decide (value (F64.toF32 (unitAt seed ordinal i)) < value w)) ∧
    (∀ w : BF16, w.isFinite = true → ∃ wr,
      chelis_types.dtype_semantics.ScalarValue.as_f64_lossy ⟨.Bf16 w⟩ = .ok wr ∧
      F64.lt (arithUnit .Bf16 (unitAt seed ordinal i)) wr =
        decide (value (F64.toF32 (unitAt seed ordinal i)) < value w)) := by
  have hu := realize_unit (keyOfDraw seed ordinal) i
  have hrange := DrawKey.unitValue_range (keyOfDraw seed ordinal) i
  have h1 : |(keyOfDraw seed ordinal).unitValue i| < 1 := by
    rw [abs_of_nonneg hrange.1]; exact hrange.2
  have ha : ∀ p : Prim, isF64 p = false →
      arithUnit p (unitAt seed ordinal i) = F32.toF64 (F64.toF32 (unitAt seed ordinal i)) := by
    intro p hp; simp [arithUnit, hp]
  refine ⟨fun w hw => ⟨_, rfl, ?_⟩, fun w hw => ⟨_, rfl, ?_⟩, fun w hw => ⟨_, rfl, ?_⟩⟩
  · rw [ha .F32 rfl]
    exact nonf64_lt_eq (u := unitAt seed ordinal i) hu h1 (realize_widen_f32 hw)
  · rw [ha .F16 rfl]
    exact nonf64_lt_eq (u := unitAt seed ordinal i) hu h1 (realize_f16_to_f64 hw)
  · rw [ha .Bf16 rfl]
    exact nonf64_lt_eq (u := unitAt seed ordinal i) hu h1 (realize_bf16_to_f64 hw)

/-! ## The f16 instance of `PreparedDropout::apply` -/

/-- `half::f16::to_f32` (the model's software widening). -/
def f16To32 (x : F16) : F32 := BinaryFloat.ofNat _ (Half.f16ToF32 x.bits.toNat)

/-- `half::f16::from_f32` (the model's software narrowing). -/
def f16Of32 (y : F32) : F16 := BinaryFloat.ofNat _ (Half.f32ToF16 y.bits.toNat)

/-- `f64::from(f16)`: the widening `as_f64_lossy` applies to a f16 rate. -/
def f16To64 (x : F16) : F64 := BinaryFloat.ofNat _ (Half.f16ToF64 x.bits.toNat)

/-- The f16 arithmetic of `float_binop`: both operands widened to f32, one binary32 operation,
one narrowing back to f16. -/
def f16Div (x d : F16) : F16 := f16Of32 (F32.div (f16To32 x) (f16To32 d))

def f16Sub (x d : F16) : F16 := f16Of32 (F32.sub (f16To32 x) (f16To32 d))

theorem f16_from_i64_zero :
    chelis_types.dtype_semantics.f16_from_i64_rne 0#i64 = .ok F16.ZERO := by rfl

theorem f16_from_i64_one :
    chelis_types.dtype_semantics.f16_from_i64_rne 1#i64 = .ok F16.ONE := by rfl

theorem prim_eq_i64_f16 :
    chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq .Int64 .F16 = .ok false := by
  simp [chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq,
    chelis_types.types.Prim.read_discriminant]

theorem prim_eq_f16_f16 :
    chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq .F16 .F16 = .ok true := by
  simp [chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq]

theorem cast_raw_zero_f16 (h) :
    chelis_types.dtype_semantics.cast_raw (toStr "dropout" h) (.Int 0#i64) .F16 =
      .ok (.Ok ⟨.F16 F16.ZERO⟩) := by
  simp [chelis_types.dtype_semantics.cast_raw, chelis_types.dtype_semantics.CheckedCastPlan.new,
    chelis_types.dtype_semantics.CheckedCastPlan.cast_raw,
    core.result.Result.unwrap_or_else, prim_eq_i64_f16,
    chelis_types.dtype_semantics.finalize_scalar, lift, f16_from_i64_zero]

theorem cast_raw_one_f16 (h) :
    chelis_types.dtype_semantics.cast_raw (toStr "dropout" h) (.Int 1#i64) .F16 =
      .ok (.Ok ⟨.F16 F16.ONE⟩) := by
  simp [chelis_types.dtype_semantics.cast_raw, chelis_types.dtype_semantics.CheckedCastPlan.new,
    chelis_types.dtype_semantics.CheckedCastPlan.cast_raw,
    core.result.Result.unwrap_or_else, prim_eq_i64_f16,
    chelis_types.dtype_semantics.finalize_scalar, lift, f16_from_i64_one]

theorem float_binop_div_f16 (x d : F16) :
    chelis_types.dtype_semantics.float_binop .Div ⟨.F16 x⟩ ⟨.F16 d⟩ =
      .ok (.Ok ⟨.F16 (f16Div x d)⟩) := by
  simp [chelis_types.dtype_semantics.float_binop, chelis_types.dtype_semantics.require_family,
    chelis_types.dtype_semantics.require_same_dtype, chelis_types.dtype_semantics.ScalarValue.prim,
    chelis_types.types.Prim.is_float, prim_eq_f16_f16, chelis_types.dtype_semantics.FloatBinOp.name,
    chelis_types.dtype_semantics.apply_float_binop_f32, F16.to_f32, F16.from_f32, f16Div, f16Of32,
    f16To32]

theorem float_binop_sub_f16 (x d : F16) :
    chelis_types.dtype_semantics.float_binop .Sub ⟨.F16 x⟩ ⟨.F16 d⟩ =
      .ok (.Ok ⟨.F16 (f16Sub x d)⟩) := by
  simp [chelis_types.dtype_semantics.float_binop, chelis_types.dtype_semantics.require_family,
    chelis_types.dtype_semantics.require_same_dtype, chelis_types.dtype_semantics.ScalarValue.prim,
    chelis_types.types.Prim.is_float, prim_eq_f16_f16, chelis_types.dtype_semantics.FloatBinOp.name,
    chelis_types.dtype_semantics.apply_float_binop_f32, F16.to_f32, F16.from_f32, f16Sub, f16Of32,
    f16To32]

/-- `DropoutParameters::denominator` of a f16 rate `w` is `1 - w` in the f16 arithmetic. -/
theorem denominator_f16 (w : F16) :
    chelis_types.dtype_semantics.DropoutParameters.denominator ⟨⟨.F16 w⟩⟩ =
      .ok (.Ok ⟨.F16 (f16Sub F16.ONE w)⟩) := by
  simp only [chelis_types.dtype_semantics.DropoutParameters.denominator,
    chelis_types.dtype_semantics.ScalarValue.prim, bind_tc_ok, cast_raw_one_f16,
    core.result.TryResult.branch]
  exact float_binop_sub_f16 F16.ONE w

theorem as_f64_lossy_f16 (w : F16) :
    chelis_types.dtype_semantics.ScalarValue.as_f64_lossy ⟨.F16 w⟩ = .ok (f16To64 w) := rfl

theorem scalar_at_f16 (xs : alloc.vec.Vec F16) (idx : Usize) (h : idx.val < xs.val.length) :
    chelis_types.dtype_semantics.TensorStorage.scalar_at ⟨.F16 xs⟩ idx =
      .ok ⟨.F16 (xs.val[idx.val]'h)⟩ := by
  simp [chelis_types.dtype_semantics.TensorStorage.scalar_at, alloc.vec.Vec.index_slice_index,
    alloc.vec.Vec.index_usize, h]

def f16Payload (x : SV) : Result F16 :=
  match x.bits with
  | .F16 y => .ok y
  | _ => .fail .panic

theorem mapM_f16Payload (ys : List F16) :
    (ys.map (fun y => (⟨.F16 y⟩ : SV))).mapM f16Payload = .ok ys := by
  induction ys with
  | nil => rfl
  | cons y ys ih =>
    simp only [List.map_cons, List.mapM_cons, f16Payload, bind_tc_ok, ih]
    rfl

/-- `tensor_from_scalars(F16, values)` of f16 scalars is the f16 buffer of their payloads. -/
theorem tensor_from_scalars_f16 (s : Slice SV) (ys : List F16)
    (h : s.val = ys.map (fun y => (⟨.F16 y⟩ : SV))) :
    ∃ v : alloc.vec.Vec F16, v.val = ys ∧
      chelis_types.dtype_semantics.tensor_from_scalars .F16 s = .ok ⟨.F16 v⟩ := by
  have hf : ∀ x ∈ s.val,
      core.ops.function.FnMut.call_mut
          chelis_types.dtype_semantics.tensor_from_scalars.closure_2.Insts.CoreOpsFunctionFnMutTupleSharedScalarValuef16
          .F16 x = (do let y ← f16Payload x; .ok (y, .F16)) := by
    intro x hx
    rw [h] at hx
    obtain ⟨y, _, rfl⟩ := List.mem_map.mp hx
    simp [chelis_types.dtype_semantics.tensor_from_scalars.closure_2.Insts.CoreOpsFunctionFnMutTupleSharedScalarValuef16.call_mut,
      chelis_types.dtype_semantics.tensor_from_scalars.read, f16Payload]
  have hm : (s.val.drop 0).mapM f16Payload = .ok ys := by
    rw [List.drop_zero, h]; exact mapM_f16Payload ys
  obtain ⟨it', hit⟩ := iterToList_map_slice_ok _ _ f16Payload s hf _ 0 [] ys rfl hm
  have hlen : ys.length ≤ Usize.max := by
    have := s.property; rw [h, List.length_map] at this; exact this
  refine ⟨⟨ys, hlen⟩, rfl, ?_⟩
  unfold chelis_types.dtype_semantics.tensor_from_scalars
  simp only [core.slice.Slice.iter, core.iter.traits.iterator.Iterator.map.default,
    core.iter.traits.iterator.Iterator.collect.default, alloc.vec.FromIteratorVec.from_iter,
    bind_tc_ok]
  simp only [core.iter.traits.collect.IntoIterator.Blanket.into_iter, bind_tc_ok]
  simp only [List.reverse_nil, List.nil_append] at hit
  rw [hit]
  simp [hlen]

/-- The f16 element Rust pushes at index `i`: the comparison uses `f64::from(unit as f32)`
against `f64::from(w)`. -/
def keptF16 (xs : alloc.vec.Vec F16) (w : F16) (seed ordinal : U64) (i : ℕ) : F16 :=
  if F64.lt (F32.toF64 (F64.toF32 (unitAt seed ordinal i))) (f16To64 w) then F16.ZERO
  else f16Div (xs.val.getD i F16.ZERO) (f16Sub F16.ONE w)

theorem arithUnit_f16 (u : F64) : arithUnit .F16 u = F32.toF64 (F64.toF32 u) := by
  simp [arithUnit, isF64]

theorem step_f16 (xs : alloc.vec.Vec F16) (w : F16) (seed ordinal : U64) (idx : Usize)
    (hidx : idx.val < (alloc.vec.Vec.len xs).val) :
    (F64.lt (arithUnit .F16 (unitAt seed ordinal idx.val)) (f16To64 w) = true ∧
        (⟨.F16 (keptF16 xs w seed ordinal idx.val)⟩ : SV) = ⟨.F16 F16.ZERO⟩) ∨
      (F64.lt (arithUnit .F16 (unitAt seed ordinal idx.val)) (f16To64 w) = false ∧
        ∃ sv, chelis_types.dtype_semantics.TensorStorage.scalar_at ⟨.F16 xs⟩ idx = .ok sv ∧
          chelis_types.dtype_semantics.float_binop .Div sv ⟨.F16 (f16Sub F16.ONE w)⟩ =
            .ok (.Ok ⟨.F16 (keptF16 xs w seed ordinal idx.val)⟩)) := by
  have hlen : idx.val < xs.val.length := by simpa using hidx
  rw [arithUnit_f16]
  cases hlt : F64.lt (F32.toF64 (F64.toF32 (unitAt seed ordinal idx.val))) (f16To64 w)
  · refine .inr ⟨rfl, _, scalar_at_f16 xs idx hlen, ?_⟩
    rw [float_binop_div_f16]
    simp only [keptF16, hlt, Bool.false_eq_true, if_false, List.getD_eq_getElem _ _ hlen]
  · exact .inl ⟨rfl, by simp [keptF16, hlt]⟩

/-- **f16 dropout, extracted `apply`.** For f16 storage `xs` and a f16 rate `w`, the extracted
`apply` succeeds with f16 storage `ys`; element `i` is the converted zero when
`f64::from(unit as f32) < f64::from(w)` and `xs[i] / (1 - w)` in the f16 arithmetic otherwise. -/
theorem apply_f16 (xs : alloc.vec.Vec F16) (w : F16) (seed ordinal : U64) :
    ∃ ys : alloc.vec.Vec F16,
      chelis_types.dtype_semantics.PreparedDropout.apply
          { input := ⟨.F16 xs⟩, «parameters» := ⟨⟨.F16 w⟩⟩ } seed ordinal =
        .ok (.Ok ⟨.F16 ys⟩) ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if F64.lt (F32.toF64 (F64.toF32 (unitAt seed ordinal i))) (f16To64 w) then F16.ZERO
        else f16Div x (f16Sub F16.ONE w)) := by
  obtain ⟨out, hout, hloop⟩ :=
    apply_loop_spec { input := ⟨.F16 xs⟩, «parameters» := ⟨⟨.F16 w⟩⟩ } ⟨⟨.F16 w⟩⟩ seed ordinal
      .F16 (f16To64 w) ⟨.F16 F16.ZERO⟩ ⟨.F16 (f16Sub F16.ONE w)⟩ (alloc.vec.Vec.len xs)
      (fun i => ⟨.F16 (keptF16 xs w seed ordinal i)⟩)
      (step_f16 xs w seed ordinal) xs.val.length 0#usize
      (alloc.vec.Vec.with_capacity SV (alloc.vec.Vec.len xs)) (by simp) (by simp)
      (by simp [alloc.vec.Vec.with_capacity])
  have hdo : (alloc.vec.Vec.deref out).val =
      ((List.range xs.val.length).map (keptF16 xs w seed ordinal)).map
        (fun y => (⟨.F16 y⟩ : SV)) := by
    simp only [alloc.vec.Vec.deref, hout, alloc.vec.Vec.with_capacity, List.nil_append,
      List.map_map, List.range_eq_range']
    simp
  obtain ⟨ys, hys, htfs⟩ := tensor_from_scalars_f16 (alloc.vec.Vec.deref out) _ hdo
  refine ⟨ys, ?_, ?_⟩
  · simp only [chelis_types.dtype_semantics.PreparedDropout.apply,
      chelis_types.dtype_semantics.TensorStorage.prim,
      chelis_types.dtype_semantics.DropoutParameters.impl.rate, as_f64_lossy_f16, bind_tc_ok,
      cast_raw_zero_f16, core.result.TryResult.branch, denominator_f16,
      chelis_types.dtype_semantics.TensorStorage.len, hloop, htfs]
  · rw [hys]
    apply List.ext_getElem?
    intro i
    rw [List.getElem?_mapIdx]
    by_cases hi : i < xs.val.length
    · rw [List.getElem?_map, List.getElem?_range hi, List.getElem?_eq_getElem hi]
      simp [keptF16, List.getElem?_eq_getElem hi]
    · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none (by omega)]
      rfl

/-- **Corrected non-f64 relation (f16, universally quantified).** For f16 storage and a finite
f16 rate `w`, the extracted `apply` drops element `i` exactly when the f32 rounding of the
reference unit is below the rate's exact value, and otherwise stores the f16 quotient
`xs[i] / (1 - w)`. -/
theorem mask_agrees_f16 (xs : alloc.vec.Vec F16) (w : F16) (hw : w.isFinite = true)
    (seed ordinal : U64) :
    ∃ ys : alloc.vec.Vec F16,
      chelis_types.dtype_semantics.PreparedDropout.apply
          { input := ⟨.F16 xs⟩, «parameters» := ⟨⟨.F16 w⟩⟩ } seed ordinal =
        .ok (.Ok ⟨.F16 ys⟩) ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if value (F64.toF32 (unitAt seed ordinal i)) < value w then F16.ZERO
        else f16Div x (f16Sub F16.ONE w)) := by
  obtain ⟨ys, happ, hys⟩ := apply_f16 xs w seed ordinal
  refine ⟨ys, happ, ?_⟩
  rw [hys]
  congr 1
  funext i x
  have hu := realize_unit (keyOfDraw seed ordinal) i
  have hrange := DrawKey.unitValue_range (keyOfDraw seed ordinal) i
  have hlt := nonf64_lt_eq (u := unitAt seed ordinal i) hu
    (by rw [abs_of_nonneg hrange.1]; exact hrange.2) (realize_f16_to_f64 hw)
  rw [show f16To64 w = BinaryFloat.ofNat .binary64 (Half.f16ToF64 w.bits.toNat) from rfl, hlt]
  simp only [decide_eq_true_eq]

/-- `1/2` as a f16 (bits `0x3800`). -/
def halfF16 : F16 := BinaryFloat.ofNat _ 0x3800

theorem value_halfF16 : value halfF16 = 1 / 2 := by decide +kernel

/-- **The naive relation fails on the extracted f16 `apply`.** At `cexSeed`, ordinal 0 and
f16 rate `1/2`, the reference unit of index 0 is below the rate (the reference `dropoutMask`
drops it), yet the extracted f16 `apply` keeps element 0 and scales it. -/
theorem f16_naive_counterexample (xs : alloc.vec.Vec F16) :
    (keyOfDraw cexSeed 0#u64).unitValue 0 < value halfF16 ∧
    ∃ ys : alloc.vec.Vec F16,
      chelis_types.dtype_semantics.PreparedDropout.apply
          { input := ⟨.F16 xs⟩, «parameters» := ⟨⟨.F16 halfF16⟩⟩ } cexSeed 0#u64 =
        .ok (.Ok ⟨.F16 ys⟩) ∧
      ys.val[0]? = xs.val[0]?.map (fun x => f16Div x (f16Sub F16.ONE halfF16)) := by
  refine ⟨?_, ?_⟩
  · have h : (keyOfDraw cexSeed 0#u64).unitValue 0 = (2 ^ 52 - 1) / 2 ^ 53 := cex_unit_value
    rw [h, value_halfF16]
    norm_num
  · obtain ⟨ys, happ, hys⟩ := apply_f16 xs halfF16 cexSeed 0#u64
    refine ⟨ys, happ, ?_⟩
    have hlt : F64.lt (F32.toF64 (F64.toF32 (unitOfWord 0x7FFFFFFFFFFFF800)))
        (f16To64 halfF16) = false := by decide +kernel
    rw [hys, List.getElem?_mapIdx, unitAt_cex, hlt]
    simp only [Bool.false_eq_true, if_false]

/-! ## The bf16 instance of `PreparedDropout::apply` -/

/-- `half::bf16::to_f32` (the model's software widening). -/
def bf16To32 (x : BF16) : F32 := BinaryFloat.ofNat _ (Half.bf16ToF32 x.bits.toNat)

/-- `half::bf16::from_f32` (the model's software narrowing). -/
def bf16Of32 (y : F32) : BF16 := BinaryFloat.ofNat _ (Half.f32ToBf16 y.bits.toNat)

/-- `f64::from(bf16)`: the widening `as_f64_lossy` applies to a bf16 rate. -/
def bf16To64 (x : BF16) : F64 := BinaryFloat.ofNat _ (Half.bf16ToF64 x.bits.toNat)

/-- The bf16 arithmetic of `float_binop`: both operands widened to f32, one binary32 operation,
one narrowing back to bf16. -/
def bf16Div (x d : BF16) : BF16 := bf16Of32 (F32.div (bf16To32 x) (bf16To32 d))

def bf16Sub (x d : BF16) : BF16 := bf16Of32 (F32.sub (bf16To32 x) (bf16To32 d))

theorem bf16_from_i64_zero :
    chelis_types.dtype_semantics.bf16_from_i64_rne 0#i64 = .ok BF16.ZERO := by rfl

theorem bf16_from_i64_one :
    chelis_types.dtype_semantics.bf16_from_i64_rne 1#i64 = .ok BF16.ONE := by rfl

theorem prim_eq_i64_bf16 :
    chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq .Int64 .Bf16 = .ok false := by
  simp [chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq,
    chelis_types.types.Prim.read_discriminant]

theorem prim_eq_bf16_bf16 :
    chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq .Bf16 .Bf16 = .ok true := by
  simp [chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq]

theorem cast_raw_zero_bf16 (h) :
    chelis_types.dtype_semantics.cast_raw (toStr "dropout" h) (.Int 0#i64) .Bf16 =
      .ok (.Ok ⟨.Bf16 BF16.ZERO⟩) := by
  simp [chelis_types.dtype_semantics.cast_raw, chelis_types.dtype_semantics.CheckedCastPlan.new,
    chelis_types.dtype_semantics.CheckedCastPlan.cast_raw,
    core.result.Result.unwrap_or_else, prim_eq_i64_bf16,
    chelis_types.dtype_semantics.finalize_scalar, lift, bf16_from_i64_zero]

theorem cast_raw_one_bf16 (h) :
    chelis_types.dtype_semantics.cast_raw (toStr "dropout" h) (.Int 1#i64) .Bf16 =
      .ok (.Ok ⟨.Bf16 BF16.ONE⟩) := by
  simp [chelis_types.dtype_semantics.cast_raw, chelis_types.dtype_semantics.CheckedCastPlan.new,
    chelis_types.dtype_semantics.CheckedCastPlan.cast_raw,
    core.result.Result.unwrap_or_else, prim_eq_i64_bf16,
    chelis_types.dtype_semantics.finalize_scalar, lift, bf16_from_i64_one]

theorem float_binop_div_bf16 (x d : BF16) :
    chelis_types.dtype_semantics.float_binop .Div ⟨.Bf16 x⟩ ⟨.Bf16 d⟩ =
      .ok (.Ok ⟨.Bf16 (bf16Div x d)⟩) := by
  simp [chelis_types.dtype_semantics.float_binop, chelis_types.dtype_semantics.require_family,
    chelis_types.dtype_semantics.require_same_dtype, chelis_types.dtype_semantics.ScalarValue.prim,
    chelis_types.types.Prim.is_float, prim_eq_bf16_bf16, chelis_types.dtype_semantics.FloatBinOp.name,
    chelis_types.dtype_semantics.apply_float_binop_f32, BF16.to_f32, BF16.from_f32, bf16Div, bf16Of32,
    bf16To32]

theorem float_binop_sub_bf16 (x d : BF16) :
    chelis_types.dtype_semantics.float_binop .Sub ⟨.Bf16 x⟩ ⟨.Bf16 d⟩ =
      .ok (.Ok ⟨.Bf16 (bf16Sub x d)⟩) := by
  simp [chelis_types.dtype_semantics.float_binop, chelis_types.dtype_semantics.require_family,
    chelis_types.dtype_semantics.require_same_dtype, chelis_types.dtype_semantics.ScalarValue.prim,
    chelis_types.types.Prim.is_float, prim_eq_bf16_bf16, chelis_types.dtype_semantics.FloatBinOp.name,
    chelis_types.dtype_semantics.apply_float_binop_f32, BF16.to_f32, BF16.from_f32, bf16Sub, bf16Of32,
    bf16To32]

/-- `DropoutParameters::denominator` of a bf16 rate `w` is `1 - w` in the bf16 arithmetic. -/
theorem denominator_bf16 (w : BF16) :
    chelis_types.dtype_semantics.DropoutParameters.denominator ⟨⟨.Bf16 w⟩⟩ =
      .ok (.Ok ⟨.Bf16 (bf16Sub BF16.ONE w)⟩) := by
  simp only [chelis_types.dtype_semantics.DropoutParameters.denominator,
    chelis_types.dtype_semantics.ScalarValue.prim, bind_tc_ok, cast_raw_one_bf16,
    core.result.TryResult.branch]
  exact float_binop_sub_bf16 BF16.ONE w

theorem as_f64_lossy_bf16 (w : BF16) :
    chelis_types.dtype_semantics.ScalarValue.as_f64_lossy ⟨.Bf16 w⟩ = .ok (bf16To64 w) := rfl

theorem scalar_at_bf16 (xs : alloc.vec.Vec BF16) (idx : Usize) (h : idx.val < xs.val.length) :
    chelis_types.dtype_semantics.TensorStorage.scalar_at ⟨.Bf16 xs⟩ idx =
      .ok ⟨.Bf16 (xs.val[idx.val]'h)⟩ := by
  simp [chelis_types.dtype_semantics.TensorStorage.scalar_at, alloc.vec.Vec.index_slice_index,
    alloc.vec.Vec.index_usize, h]

def bf16Payload (x : SV) : Result BF16 :=
  match x.bits with
  | .Bf16 y => .ok y
  | _ => .fail .panic

theorem mapM_bf16Payload (ys : List BF16) :
    (ys.map (fun y => (⟨.Bf16 y⟩ : SV))).mapM bf16Payload = .ok ys := by
  induction ys with
  | nil => rfl
  | cons y ys ih =>
    simp only [List.map_cons, List.mapM_cons, bf16Payload, bind_tc_ok, ih]
    rfl

/-- `tensor_from_scalars(Bf16, values)` of bf16 scalars is the bf16 buffer of their payloads. -/
theorem tensor_from_scalars_bf16 (s : Slice SV) (ys : List BF16)
    (h : s.val = ys.map (fun y => (⟨.Bf16 y⟩ : SV))) :
    ∃ v : alloc.vec.Vec BF16, v.val = ys ∧
      chelis_types.dtype_semantics.tensor_from_scalars .Bf16 s = .ok ⟨.Bf16 v⟩ := by
  have hf : ∀ x ∈ s.val,
      core.ops.function.FnMut.call_mut
          chelis_types.dtype_semantics.tensor_from_scalars.closure_3.Insts.CoreOpsFunctionFnMutTupleSharedScalarValuebf16
          .Bf16 x = (do let y ← bf16Payload x; .ok (y, .Bf16)) := by
    intro x hx
    rw [h] at hx
    obtain ⟨y, _, rfl⟩ := List.mem_map.mp hx
    simp [chelis_types.dtype_semantics.tensor_from_scalars.closure_3.Insts.CoreOpsFunctionFnMutTupleSharedScalarValuebf16.call_mut,
      chelis_types.dtype_semantics.tensor_from_scalars.read, bf16Payload]
  have hm : (s.val.drop 0).mapM bf16Payload = .ok ys := by
    rw [List.drop_zero, h]; exact mapM_bf16Payload ys
  obtain ⟨it', hit⟩ := iterToList_map_slice_ok _ _ bf16Payload s hf _ 0 [] ys rfl hm
  have hlen : ys.length ≤ Usize.max := by
    have := s.property; rw [h, List.length_map] at this; exact this
  refine ⟨⟨ys, hlen⟩, rfl, ?_⟩
  unfold chelis_types.dtype_semantics.tensor_from_scalars
  simp only [core.slice.Slice.iter, core.iter.traits.iterator.Iterator.map.default,
    core.iter.traits.iterator.Iterator.collect.default, alloc.vec.FromIteratorVec.from_iter,
    bind_tc_ok]
  simp only [core.iter.traits.collect.IntoIterator.Blanket.into_iter, bind_tc_ok]
  simp only [List.reverse_nil, List.nil_append] at hit
  rw [hit]
  simp [hlen]

/-- The bf16 element Rust pushes at index `i`: the comparison uses `f64::from(unit as f32)`
against `f64::from(w)`. -/
def keptBF16 (xs : alloc.vec.Vec BF16) (w : BF16) (seed ordinal : U64) (i : ℕ) : BF16 :=
  if F64.lt (F32.toF64 (F64.toF32 (unitAt seed ordinal i))) (bf16To64 w) then BF16.ZERO
  else bf16Div (xs.val.getD i BF16.ZERO) (bf16Sub BF16.ONE w)

theorem arithUnit_bf16 (u : F64) : arithUnit .Bf16 u = F32.toF64 (F64.toF32 u) := by
  simp [arithUnit, isF64]

theorem step_bf16 (xs : alloc.vec.Vec BF16) (w : BF16) (seed ordinal : U64) (idx : Usize)
    (hidx : idx.val < (alloc.vec.Vec.len xs).val) :
    (F64.lt (arithUnit .Bf16 (unitAt seed ordinal idx.val)) (bf16To64 w) = true ∧
        (⟨.Bf16 (keptBF16 xs w seed ordinal idx.val)⟩ : SV) = ⟨.Bf16 BF16.ZERO⟩) ∨
      (F64.lt (arithUnit .Bf16 (unitAt seed ordinal idx.val)) (bf16To64 w) = false ∧
        ∃ sv, chelis_types.dtype_semantics.TensorStorage.scalar_at ⟨.Bf16 xs⟩ idx = .ok sv ∧
          chelis_types.dtype_semantics.float_binop .Div sv ⟨.Bf16 (bf16Sub BF16.ONE w)⟩ =
            .ok (.Ok ⟨.Bf16 (keptBF16 xs w seed ordinal idx.val)⟩)) := by
  have hlen : idx.val < xs.val.length := by simpa using hidx
  rw [arithUnit_bf16]
  cases hlt : F64.lt (F32.toF64 (F64.toF32 (unitAt seed ordinal idx.val))) (bf16To64 w)
  · refine .inr ⟨rfl, _, scalar_at_bf16 xs idx hlen, ?_⟩
    rw [float_binop_div_bf16]
    simp only [keptBF16, hlt, Bool.false_eq_true, if_false, List.getD_eq_getElem _ _ hlen]
  · exact .inl ⟨rfl, by simp [keptBF16, hlt]⟩

/-- **bf16 dropout, extracted `apply`.** For bf16 storage `xs` and a bf16 rate `w`, the extracted
`apply` succeeds with bf16 storage `ys`; element `i` is the converted zero when
`f64::from(unit as f32) < f64::from(w)` and `xs[i] / (1 - w)` in the bf16 arithmetic otherwise. -/
theorem apply_bf16 (xs : alloc.vec.Vec BF16) (w : BF16) (seed ordinal : U64) :
    ∃ ys : alloc.vec.Vec BF16,
      chelis_types.dtype_semantics.PreparedDropout.apply
          { input := ⟨.Bf16 xs⟩, «parameters» := ⟨⟨.Bf16 w⟩⟩ } seed ordinal =
        .ok (.Ok ⟨.Bf16 ys⟩) ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if F64.lt (F32.toF64 (F64.toF32 (unitAt seed ordinal i))) (bf16To64 w) then BF16.ZERO
        else bf16Div x (bf16Sub BF16.ONE w)) := by
  obtain ⟨out, hout, hloop⟩ :=
    apply_loop_spec { input := ⟨.Bf16 xs⟩, «parameters» := ⟨⟨.Bf16 w⟩⟩ } ⟨⟨.Bf16 w⟩⟩ seed ordinal
      .Bf16 (bf16To64 w) ⟨.Bf16 BF16.ZERO⟩ ⟨.Bf16 (bf16Sub BF16.ONE w)⟩ (alloc.vec.Vec.len xs)
      (fun i => ⟨.Bf16 (keptBF16 xs w seed ordinal i)⟩)
      (step_bf16 xs w seed ordinal) xs.val.length 0#usize
      (alloc.vec.Vec.with_capacity SV (alloc.vec.Vec.len xs)) (by simp) (by simp)
      (by simp [alloc.vec.Vec.with_capacity])
  have hdo : (alloc.vec.Vec.deref out).val =
      ((List.range xs.val.length).map (keptBF16 xs w seed ordinal)).map
        (fun y => (⟨.Bf16 y⟩ : SV)) := by
    simp only [alloc.vec.Vec.deref, hout, alloc.vec.Vec.with_capacity, List.nil_append,
      List.map_map, List.range_eq_range']
    simp
  obtain ⟨ys, hys, htfs⟩ := tensor_from_scalars_bf16 (alloc.vec.Vec.deref out) _ hdo
  refine ⟨ys, ?_, ?_⟩
  · simp only [chelis_types.dtype_semantics.PreparedDropout.apply,
      chelis_types.dtype_semantics.TensorStorage.prim,
      chelis_types.dtype_semantics.DropoutParameters.impl.rate, as_f64_lossy_bf16, bind_tc_ok,
      cast_raw_zero_bf16, core.result.TryResult.branch, denominator_bf16,
      chelis_types.dtype_semantics.TensorStorage.len, hloop, htfs]
  · rw [hys]
    apply List.ext_getElem?
    intro i
    rw [List.getElem?_mapIdx]
    by_cases hi : i < xs.val.length
    · rw [List.getElem?_map, List.getElem?_range hi, List.getElem?_eq_getElem hi]
      simp [keptBF16, List.getElem?_eq_getElem hi]
    · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none (by omega)]
      rfl

/-- **Corrected non-f64 relation (bf16, universally quantified).** For bf16 storage and a finite
bf16 rate `w`, the extracted `apply` drops element `i` exactly when the f32 rounding of the
reference unit is below the rate's exact value, and otherwise stores the bf16 quotient
`xs[i] / (1 - w)`. -/
theorem mask_agrees_bf16 (xs : alloc.vec.Vec BF16) (w : BF16) (hw : w.isFinite = true)
    (seed ordinal : U64) :
    ∃ ys : alloc.vec.Vec BF16,
      chelis_types.dtype_semantics.PreparedDropout.apply
          { input := ⟨.Bf16 xs⟩, «parameters» := ⟨⟨.Bf16 w⟩⟩ } seed ordinal =
        .ok (.Ok ⟨.Bf16 ys⟩) ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if value (F64.toF32 (unitAt seed ordinal i)) < value w then BF16.ZERO
        else bf16Div x (bf16Sub BF16.ONE w)) := by
  obtain ⟨ys, happ, hys⟩ := apply_bf16 xs w seed ordinal
  refine ⟨ys, happ, ?_⟩
  rw [hys]
  congr 1
  funext i x
  have hu := realize_unit (keyOfDraw seed ordinal) i
  have hrange := DrawKey.unitValue_range (keyOfDraw seed ordinal) i
  have hlt := nonf64_lt_eq (u := unitAt seed ordinal i) hu
    (by rw [abs_of_nonneg hrange.1]; exact hrange.2) (realize_bf16_to_f64 hw)
  rw [show bf16To64 w = BinaryFloat.ofNat .binary64 (Half.bf16ToF64 w.bits.toNat) from rfl, hlt]
  simp only [decide_eq_true_eq]

/-- `1/2` as a bf16 (bits `0x3F00`). -/
def halfBF16 : BF16 := BinaryFloat.ofNat _ 0x3F00

theorem value_halfBF16 : value halfBF16 = 1 / 2 := by decide +kernel

/-- **The naive relation fails on the extracted bf16 `apply`.** At `cexSeed`, ordinal 0 and
bf16 rate `1/2`, the reference unit of index 0 is below the rate (the reference `dropoutMask`
drops it), yet the extracted bf16 `apply` keeps element 0 and scales it. -/
theorem bf16_naive_counterexample (xs : alloc.vec.Vec BF16) :
    (keyOfDraw cexSeed 0#u64).unitValue 0 < value halfBF16 ∧
    ∃ ys : alloc.vec.Vec BF16,
      chelis_types.dtype_semantics.PreparedDropout.apply
          { input := ⟨.Bf16 xs⟩, «parameters» := ⟨⟨.Bf16 halfBF16⟩⟩ } cexSeed 0#u64 =
        .ok (.Ok ⟨.Bf16 ys⟩) ∧
      ys.val[0]? = xs.val[0]?.map (fun x => bf16Div x (bf16Sub BF16.ONE halfBF16)) := by
  refine ⟨?_, ?_⟩
  · have h : (keyOfDraw cexSeed 0#u64).unitValue 0 = (2 ^ 52 - 1) / 2 ^ 53 := cex_unit_value
    rw [h, value_halfBF16]
    norm_num
  · obtain ⟨ys, happ, hys⟩ := apply_bf16 xs halfBF16 cexSeed 0#u64
    refine ⟨ys, happ, ?_⟩
    have hlt : F64.lt (F32.toF64 (F64.toF32 (unitOfWord 0x7FFFFFFFFFFFF800)))
        (bf16To64 halfBF16) = false := by decide +kernel
    rw [hys, List.getElem?_mapIdx, unitAt_cex, hlt]
    simp only [Bool.false_eq_true, if_false]

end ChelisBridge.Random
