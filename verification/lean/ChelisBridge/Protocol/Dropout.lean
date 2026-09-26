import ChelisBridge.Protocol.Random
import ChelisBridge.Tensor

open Aeneas.Std LaCaDiLECore ChelisBridge ChelisBridge.Protocol ChelisBridge.Random
open_chelis_model

namespace ChelisBridge.Random

abbrev Prim := chelis_types.types.Prim
abbrev SV := chelis_types.dtype_semantics.ScalarValue
abbrev Storage := chelis_types.dtype_semantics.TensorStorage

/-- Whether the extracted `prim == Prim::F64` holds. -/
def isF64 : Prim → Bool
  | .F64 => true
  | _ => false

theorem prim_eq_f64 (p : Prim) :
    chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq p .F64 = .ok (isF64 p) := by
  cases p <;>
    simp [chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq,
      chelis_types.types.Prim.read_discriminant, isF64]

/-- The unit Rust compares with the widened rate: the exact unit for f64 storage, and
`f64::from(unit as f32)` (the f32 rounding) for every other dtype. -/
def arithUnit (p : Prim) (u : F64) : F64 := if isF64 p then u else F32.toF64 (F64.toF32 u)

theorem cast_u64_val (x : Usize) : (UScalar.cast .U64 x).val = x.val := by
  apply UScalar.cast_val_mod_pow_greater_numBits_eq
  simp only [UScalarTy.Usize_numBits_eq, UScalarTy.U64_numBits_eq]
  cases System.Platform.numBits_eq <;> simp [*]

/-- The unit drawn for flat index `i`. -/
def unitAt (seed ordinal : U64) (i : ℕ) : F64 :=
  unitOfWord (DrawKey.word ⟨UInt64.ofBitVec seed.bv, ordinal.val⟩ i)

/-- One iteration of the extracted dropout loop at index `k < n`: it pushes `x`, which is the
converted zero when the arithmetic unit is below the widened rate and the successful quotient
`input[k] / denominator` otherwise. -/
theorem body_step (self : chelis_types.dtype_semantics.PreparedDropout)
    (dp : chelis_types.dtype_semantics.DropoutParameters) (seed ordinal : U64) (prim : Prim)
    (wr : F64) (zero den : SV) (output : alloc.vec.Vec SV) (k n : Usize)
    (hk : k.val < n.val) (hlen : output.val.length < Usize.max) (x : SV)
    (hx : (F64.lt (arithUnit prim (unitAt seed ordinal k.val)) wr = true ∧ x = zero) ∨
      (F64.lt (arithUnit prim (unitAt seed ordinal k.val)) wr = false ∧
        ∃ sv, chelis_types.dtype_semantics.TensorStorage.scalar_at self.input k = .ok sv ∧
          chelis_types.dtype_semantics.float_binop .Div sv den = .ok (.Ok x))) :
    ∃ (k' : Usize) (out' : alloc.vec.Vec SV), k'.val = k.val + 1 ∧
      out'.val = output.val ++ [x] ∧
      chelis_types.dtype_semantics.PreparedDropout.apply_loop.body self ⟨k, n⟩ dp seed ordinal
          prim wr zero den output =
        .ok (.cont (⟨k', n⟩, dp, seed, ordinal, prim, wr, zero, den, out')) := by
  obtain ⟨k', hk', hnext⟩ := range_next_lt k n hk
  have hpush : ∀ y, ∃ o, alloc.vec.Vec.push output y = .ok o ∧ o.val = output.val ++ [y] := by
    intro y
    unfold alloc.vec.Vec.push
    rw [dif_pos (by simp; omega)]
    exact ⟨_, rfl, List.concat_eq_append⟩
  have hunit := word_eq_reference seed ordinal (UScalar.cast .U64 k)
  rw [cast_u64_val] at hunit
  simp only [chelis_types.dtype_semantics.PreparedDropout.apply_loop.body, hnext, bind_tc_ok,
    Function.uncurry_apply_pair, lift, hunit, prim_eq_f64]
  have harith : ∀ u : F64, (if isF64 prim = true then Result.ok u
      else core.convert.num.FromF64F32.from_ (F64.toF32 u)) = .ok (arithUnit prim u) := by
    intro u
    cases h : isF64 prim <;> simp [arithUnit, h, core.convert.num.FromF64F32.from_]
  simp only [harith, bind_tc_ok]
  simp only [unitAt] at hx
  rcases hx with ⟨hlt, rfl⟩ | ⟨hlt, sv, hsv, hfb⟩
  · obtain ⟨o, ho, hov⟩ := hpush x
    simp only [hlt, if_true, ho, bind_tc_ok]
    exact ⟨k', o, hk', hov, rfl⟩
  · obtain ⟨o, ho, hov⟩ := hpush x
    simp only [hlt, Bool.false_eq_true, if_false, hsv, hfb, core.result.TryResult.branch, ho,
      bind_tc_ok]
    exact ⟨k', o, hk', hov, rfl⟩

/-- The extracted dropout loop from index `k` (with `k` elements already pushed) pushes the
per-index values `v k, …, v (n-1)` and finishes with `tensor_from_scalars prim` of the whole
output, provided every index takes one of the two branches with value `v i`. -/
theorem apply_loop_spec (self : chelis_types.dtype_semantics.PreparedDropout)
    (dp : chelis_types.dtype_semantics.DropoutParameters) (seed ordinal : U64) (prim : Prim)
    (wr : F64) (zero den : SV) (n : Usize) (v : ℕ → SV)
    (hv : ∀ idx : Usize, idx.val < n.val →
      (F64.lt (arithUnit prim (unitAt seed ordinal idx.val)) wr = true ∧ v idx.val = zero) ∨
      (F64.lt (arithUnit prim (unitAt seed ordinal idx.val)) wr = false ∧
        ∃ sv, chelis_types.dtype_semantics.TensorStorage.scalar_at self.input idx = .ok sv ∧
          chelis_types.dtype_semantics.float_binop .Div sv den = .ok (.Ok (v idx.val)))) :
    ∀ (m : ℕ) (k : Usize) (output : alloc.vec.Vec SV), n.val - k.val = m → k.val ≤ n.val →
      output.val.length = k.val →
      ∃ out : alloc.vec.Vec SV, out.val = output.val ++ (List.range' k.val m).map v ∧
        chelis_types.dtype_semantics.PreparedDropout.apply_loop self ⟨k, n⟩ dp seed ordinal prim wr
            zero den output =
          (do
            let t ← chelis_types.dtype_semantics.tensor_from_scalars prim
              (alloc.vec.Vec.deref out)
            .ok (.Ok t)) := by
  intro m
  induction m with
  | zero =>
    intro k output hm hkn hlen
    refine ⟨output, by simp, ?_⟩
    unfold chelis_types.dtype_semantics.PreparedDropout.apply_loop
    rw [loop.eq_def]
    simp only [Function.uncurry_apply_pair,
      chelis_types.dtype_semantics.PreparedDropout.apply_loop.body,
      range_next_ge k n (by omega), bind_tc_ok]
    cases chelis_types.dtype_semantics.tensor_from_scalars prim (alloc.vec.Vec.deref output) <;>
      simp
  | succ m ih =>
    intro k output hm hkn hlen
    have hk : k.val < n.val := by omega
    obtain ⟨k', out', hk', hout', hbody⟩ :=
      body_step self dp seed ordinal prim wr zero den output k n hk
        (by have := n.hBounds; scalar_tac) (v k.val) (hv k hk)
    obtain ⟨out, hout, hloop⟩ := ih k' out' (by omega) (by omega)
      (by rw [hout', List.length_append, hlen, hk']; rfl)
    refine ⟨out, ?_, ?_⟩
    · rw [hout, hout', hk', List.range'_succ, List.map_cons, List.append_assoc]; rfl
    · unfold chelis_types.dtype_semantics.PreparedDropout.apply_loop at hloop ⊢
      rw [loop.eq_def]
      simp only
      rw [hbody]
      exact hloop

/-- Payload of an f64 scalar (the only kind the f64 collector reads without panicking). -/
def f64Payload (x : SV) : Result F64 :=
  match x.bits with
  | .F64 y => .ok y
  | _ => .fail .panic

theorem mapM_f64Payload (ys : List F64) :
    (ys.map (fun y => (⟨.F64 y⟩ : SV))).mapM f64Payload = .ok ys := by
  induction ys with
  | nil => rfl
  | cons y ys ih =>
    simp only [List.map_cons, List.mapM_cons, f64Payload, bind_tc_ok, ih]
    rfl

/-- `tensor_from_scalars(F64, values)` of f64 scalars is the f64 buffer of their payloads. -/
theorem tensor_from_scalars_f64 (s : Slice SV) (ys : List F64)
    (h : s.val = ys.map (fun y => (⟨.F64 y⟩ : SV))) :
    ∃ v : alloc.vec.Vec F64, v.val = ys ∧
      chelis_types.dtype_semantics.tensor_from_scalars .F64 s = .ok ⟨.F64 v⟩ := by
  have hf : ∀ x ∈ s.val,
      core.ops.function.FnMut.call_mut
          chelis_types.dtype_semantics.tensor_from_scalars.closure.Insts.CoreOpsFunctionFnMutTupleSharedScalarValueF64
          .F64 x = (do let y ← f64Payload x; .ok (y, .F64)) := by
    intro x hx
    rw [h] at hx
    obtain ⟨y, _, rfl⟩ := List.mem_map.mp hx
    simp [chelis_types.dtype_semantics.tensor_from_scalars.closure.Insts.CoreOpsFunctionFnMutTupleSharedScalarValueF64.call_mut,
      chelis_types.dtype_semantics.tensor_from_scalars.read, f64Payload]
  have hm : (s.val.drop 0).mapM f64Payload = .ok ys := by
    rw [List.drop_zero, h]; exact mapM_f64Payload ys
  obtain ⟨it', hit⟩ := iterToList_map_slice_ok _ _ f64Payload s hf _ 0 [] ys rfl hm
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

/-! ## The f64 instance of `PreparedDropout::apply` -/

/-- `f64` zero and one as produced by `cast_raw(Int(0|1), F64)`. -/
def zero64 : F64 := F64.ofIScalar 0#i64
def one64 : F64 := F64.ofIScalar 1#i64

theorem realize_zero64 : realize zero64 = some 0 := by decide +kernel
theorem realize_one64 : realize one64 = some 1 := by decide +kernel

theorem prim_eq_i64_f64 :
    chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq .Int64 .F64 = .ok false := by
  simp [chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq,
    chelis_types.types.Prim.read_discriminant]

theorem cast_raw_int_f64 (h) (n : I64) :
    chelis_types.dtype_semantics.cast_raw (toStr "dropout" h) (.Int n) .F64 =
      .ok (.Ok ⟨.F64 (F64.ofIScalar n)⟩) := by
  simp [chelis_types.dtype_semantics.cast_raw, chelis_types.dtype_semantics.CheckedCastPlan.new,
    chelis_types.dtype_semantics.CheckedCastPlan.cast_raw,
    core.result.Result.unwrap_or_else, prim_eq_i64_f64,
    chelis_types.dtype_semantics.finalize_scalar, lift]

theorem prim_eq_f64_f64 :
    chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq .F64 .F64 = .ok true := by
  simp [chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq]

theorem float_binop_div_f64 (x d : F64) :
    chelis_types.dtype_semantics.float_binop .Div ⟨.F64 x⟩ ⟨.F64 d⟩ =
      .ok (.Ok ⟨.F64 (F64.div x d)⟩) := by
  simp [chelis_types.dtype_semantics.float_binop, chelis_types.dtype_semantics.require_family,
    chelis_types.dtype_semantics.require_same_dtype, chelis_types.dtype_semantics.ScalarValue.prim,
    chelis_types.types.Prim.is_float, prim_eq_f64_f64, chelis_types.dtype_semantics.FloatBinOp.name,
    chelis_types.dtype_semantics.apply_float_binop_f64]

theorem float_binop_sub_f64 (x d : F64) :
    chelis_types.dtype_semantics.float_binop .Sub ⟨.F64 x⟩ ⟨.F64 d⟩ =
      .ok (.Ok ⟨.F64 (F64.sub x d)⟩) := by
  simp [chelis_types.dtype_semantics.float_binop, chelis_types.dtype_semantics.require_family,
    chelis_types.dtype_semantics.require_same_dtype, chelis_types.dtype_semantics.ScalarValue.prim,
    chelis_types.types.Prim.is_float, prim_eq_f64_f64, chelis_types.dtype_semantics.FloatBinOp.name,
    chelis_types.dtype_semantics.apply_float_binop_f64]

theorem cast_raw_zero_f64 (h) :
    chelis_types.dtype_semantics.cast_raw (toStr "dropout" h) (.Int 0#i64) .F64 =
      .ok (.Ok ⟨.F64 zero64⟩) := cast_raw_int_f64 h 0#i64

/-- `DropoutParameters::denominator` of an f64 rate `w` is `1 - w` (one binary64 subtraction). -/
theorem denominator_f64 (w : F64) :
    chelis_types.dtype_semantics.DropoutParameters.denominator ⟨⟨.F64 w⟩⟩ =
      .ok (.Ok ⟨.F64 (F64.sub one64 w)⟩) := by
  simp only [chelis_types.dtype_semantics.DropoutParameters.denominator,
    chelis_types.dtype_semantics.ScalarValue.prim, bind_tc_ok, cast_raw_int_f64,
    core.result.TryResult.branch]
  exact float_binop_sub_f64 one64 w

theorem scalar_at_f64 (xs : alloc.vec.Vec F64) (idx : Usize) (h : idx.val < xs.val.length) :
    chelis_types.dtype_semantics.TensorStorage.scalar_at ⟨.F64 xs⟩ idx =
      .ok ⟨.F64 (xs.val[idx.val]'h)⟩ := by
  simp [chelis_types.dtype_semantics.TensorStorage.scalar_at, alloc.vec.Vec.index_slice_index,
    alloc.vec.Vec.index_usize, h]

/-- The f64 element Rust pushes at index `i` (`xs[i]` read as `zero64` out of range). -/
def keptF64 (xs : alloc.vec.Vec F64) (w : F64) (seed ordinal : U64) (i : ℕ) : F64 :=
  if F64.lt (unitAt seed ordinal i) w then zero64
  else F64.div (xs.val.getD i zero64) (F64.sub one64 w)

theorem arithUnit_f64 (u : F64) : arithUnit .F64 u = u := by simp [arithUnit, isF64]

theorem step_f64 (xs : alloc.vec.Vec F64) (w : F64) (seed ordinal : U64) (idx : Usize)
    (hidx : idx.val < (alloc.vec.Vec.len xs).val) :
    (F64.lt (arithUnit .F64 (unitAt seed ordinal idx.val)) w = true ∧
        (⟨.F64 (keptF64 xs w seed ordinal idx.val)⟩ : SV) = ⟨.F64 zero64⟩) ∨
      (F64.lt (arithUnit .F64 (unitAt seed ordinal idx.val)) w = false ∧
        ∃ sv, chelis_types.dtype_semantics.TensorStorage.scalar_at ⟨.F64 xs⟩ idx = .ok sv ∧
          chelis_types.dtype_semantics.float_binop .Div sv ⟨.F64 (F64.sub one64 w)⟩ =
            .ok (.Ok ⟨.F64 (keptF64 xs w seed ordinal idx.val)⟩)) := by
  have hlen : idx.val < xs.val.length := by simpa using hidx
  rw [arithUnit_f64]
  cases hlt : F64.lt (unitAt seed ordinal idx.val) w
  · refine .inr ⟨rfl, _, scalar_at_f64 xs idx hlen, ?_⟩
    rw [float_binop_div_f64]
    simp only [keptF64, hlt, Bool.false_eq_true, if_false, List.getD_eq_getElem _ _ hlen]
  · exact .inl ⟨rfl, by simp [keptF64, hlt]⟩

/-- **f64 dropout, extracted `apply`.** For f64 storage `xs` and an f64 rate `w`, the extracted
`PreparedDropout::apply` succeeds with f64 storage `ys` whose element `i` is `0` (Rust
`cast_raw(Int(0))`) when the drawn unit is below the rate (Rust `<` on f64) and
`xs[i] / (1 - w)` (binary64 operations) otherwise. -/
theorem apply_f64 (xs : alloc.vec.Vec F64) (w : F64) (seed ordinal : U64) :
    ∃ ys : alloc.vec.Vec F64,
      chelis_types.dtype_semantics.PreparedDropout.apply
          { input := ⟨.F64 xs⟩, «parameters» := ⟨⟨.F64 w⟩⟩ } seed ordinal =
        .ok (.Ok ⟨.F64 ys⟩) ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if F64.lt (unitAt seed ordinal i) w then zero64 else F64.div x (F64.sub one64 w)) := by
  obtain ⟨out, hout, hloop⟩ :=
    apply_loop_spec { input := ⟨.F64 xs⟩, «parameters» := ⟨⟨.F64 w⟩⟩ } ⟨⟨.F64 w⟩⟩ seed ordinal
      .F64 w ⟨.F64 zero64⟩ ⟨.F64 (F64.sub one64 w)⟩ (alloc.vec.Vec.len xs)
      (fun i => ⟨.F64 (keptF64 xs w seed ordinal i)⟩)
      (step_f64 xs w seed ordinal) xs.val.length 0#usize
      (alloc.vec.Vec.with_capacity SV (alloc.vec.Vec.len xs)) (by simp) (by simp)
      (by simp [alloc.vec.Vec.with_capacity])
  have hdo : (alloc.vec.Vec.deref out).val =
      ((List.range xs.val.length).map (keptF64 xs w seed ordinal)).map
        (fun y => (⟨.F64 y⟩ : SV)) := by
    simp only [alloc.vec.Vec.deref, hout, alloc.vec.Vec.with_capacity, List.nil_append,
      List.map_map, List.range_eq_range']
    simp
  obtain ⟨ys, hys, htfs⟩ := tensor_from_scalars_f64 (alloc.vec.Vec.deref out) _ hdo
  refine ⟨ys, ?_, ?_⟩
  · simp only [chelis_types.dtype_semantics.PreparedDropout.apply,
      chelis_types.dtype_semantics.TensorStorage.prim,
      chelis_types.dtype_semantics.DropoutParameters.impl.rate,
      chelis_types.dtype_semantics.ScalarValue.as_f64_lossy, bind_tc_ok, cast_raw_zero_f64,
      core.result.TryResult.branch, denominator_f64,
      chelis_types.dtype_semantics.TensorStorage.len, hloop, htfs]
  · rw [hys]
    apply List.ext_getElem?
    intro i
    rw [List.getElem?_mapIdx]
    by_cases hi : i < xs.val.length
    · rw [List.getElem?_map, List.getElem?_range hi, List.getElem?_eq_getElem hi]
      simp [keptF64, List.getElem?_eq_getElem hi]
    · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none (by omega)]
      rfl

/-- The reference key of the extracted `(seed, ordinal)` pair. -/
def keyOfDraw (seed ordinal : U64) : DrawKey := ⟨UInt64.ofBitVec seed.bv, ordinal.val⟩

/-- **Mask agreement (f64, element form).** For f64 storage `xs` and an f64 rate `w` with exact
value `r`, the extracted `apply` succeeds and element `i` of its output is the converted zero
exactly when the reference unit `DrawKey.unitValue key i` is below `r`, and the binary64
quotient `xs[i] / (1 - w)` otherwise. (The element form avoids the false "output is zero iff
dropped": a kept `xs[i] = 0` is also zero.) -/
theorem mask_agrees_f64 (xs : alloc.vec.Vec F64) (w : F64) (r : ℚ) (hw : realize w = some r)
    (seed ordinal : U64) :
    ∃ ys : alloc.vec.Vec F64,
      chelis_types.dtype_semantics.PreparedDropout.apply
          { input := ⟨.F64 xs⟩, «parameters» := ⟨⟨.F64 w⟩⟩ } seed ordinal =
        .ok (.Ok ⟨.F64 ys⟩) ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if (keyOfDraw seed ordinal).unitValue i < r then zero64
        else F64.div x (F64.sub one64 w)) := by
  obtain ⟨ys, happ, hys⟩ := apply_f64 xs w seed ordinal
  refine ⟨ys, happ, ?_⟩
  rw [hys]
  congr 1
  funext i x
  have hlt : F64.lt (unitAt seed ordinal i) w = decide ((keyOfDraw seed ordinal).unitValue i < r) :=
    f64_lt_eq (realize_unit _ i) hw
  rw [hlt]
  simp only [decide_eq_true_eq]

theorem forall₂_mapIdx {α β γ δ : Type} {R : α → β → Prop} {S : γ → δ → Prop}
    {F : ℕ → α → γ} {G : ℕ → β → δ} {l₁ : List α} {l₂ : List β} (h : List.Forall₂ R l₁ l₂)
    (hFG : ∀ j a b, R a b → S (F j a) (G j b)) :
    List.Forall₂ S (l₁.mapIdx F) (l₂.mapIdx G) := by
  rw [List.forall₂_iff_get] at h ⊢
  obtain ⟨hlen, hget⟩ := h
  refine ⟨by simp [hlen], fun i h1 h2 => ?_⟩
  simp only [List.get_eq_getElem, List.getElem_mapIdx]
  exact hFG i _ _ (by
    have := hget i (by simpa using h1) (by simpa using h2)
    simpa using this)

/-- **Scale (element level, f64 model ops).** Under the per-execution exactness hypotheses
`1 - r` representable and `a / (1 - r)` representable, the kept value
`x / (1 - rate)` computed by the Aeneas binary64 `sub` then `div` is exactly `a * (1 - r)⁻¹`. -/
theorem scale_exact_agrees (x one rate : F64) (a r : ℚ) (hx : realize x = some a)
    (h1 : realize one = some 1) (hr : realize rate = some r) (hr1 : r < 1)
    (hden : Representable .binary64 (1 - r)) (hq : Representable .binary64 (a / (1 - r))) :
    realize (F64.div x (F64.sub one rate)) = some (a * (1 - r)⁻¹) := by
  have hd := realize_sub h1 hr hden
  rw [← div_eq_mul_inv]
  exact realize_div hx hd (by linarith) hq

/-- **Tensor corollary (f64).** If the f64 input decodes to the reference tensor `x`, the rate
realizes to `rate`, `1 - rate` is a binary64 value and every quotient `x_j / (1 - rate)` is a
binary64 value (the per-execution exactness hypotheses of the scale), then the extracted
`apply` output decodes to `x * dropoutMask rate key`, coordinatewise through `Index.flat`. -/
theorem mask_tensor_f64 {s : Shape} (xs : alloc.vec.Vec F64) (w : F64) (rate : DropoutRate)
    (hw : realize w = some rate.val) (seed ordinal : U64) (x : Tensor ℚ s)
    (hx : decodeTensor s xs.val = some x)
    (hden : Representable .binary64 (1 - rate.val))
    (hq : ∀ (j : ℕ) (hj : j < xs.val.length) (a : ℚ), realize xs.val[j] = some a →
      Representable .binary64 (a / (1 - rate.val))) :
    ∃ (ys : alloc.vec.Vec F64) (y : Tensor ℚ s),
      chelis_types.dtype_semantics.PreparedDropout.apply
          { input := ⟨.F64 xs⟩, «parameters» := ⟨⟨.F64 w⟩⟩ } seed ordinal =
        .ok (.Ok ⟨.F64 ys⟩) ∧
      decodeTensor s ys.val = some y ∧
      ∀ i, y i = x i * dropoutMask rate (keyOfDraw seed ordinal) i := by
  obtain ⟨ys, happ, hys⟩ := mask_agrees_f64 xs w rate.val hw seed ordinal
  simp only [decodeTensor, Option.bind_eq_bind, Option.bind_eq_some_iff] at hx
  obtain ⟨avs, has, hxt⟩ := hx
  have hfa := realizeList_eq_some.mp has
  let maskv : ℕ → ℚ := fun j =>
    if (keyOfDraw seed ordinal).unitValue j < rate.val then 0 else (1 - rate.val)⁻¹
  have hys' : realizeList ys.val = some (avs.mapIdx (fun j a => a * maskv j)) := by
    rw [realizeList_eq_some, hys]
    have hfa' : List.Forall₂ (fun x a => realize x = some a ∧ ∃ j, ∃ hj : j < xs.val.length,
        xs.val[j] = x) xs.val avs := by
      rw [List.forall₂_iff_get] at hfa ⊢
      refine ⟨hfa.1, fun i h1 h2 => ⟨hfa.2 i h1 h2, i, h1, rfl⟩⟩
    refine forall₂_mapIdx hfa' ?_
    rintro j x a ⟨hxa, k, hk, rfl⟩
    simp only [maskv]
    split
    · rw [mul_zero]; exact realize_zero64
    · exact scale_exact_agrees _ one64 w a rate.val hxa realize_one64 hw rate.property.2 hden
        (hq k hk a hxa)
  have hlen : (avs.mapIdx (fun j a => a * maskv j)).length = s.prod := by
    rw [List.length_mapIdx]; exact ofList_length hxt
  obtain ⟨y, hy⟩ : ∃ y, Tensor.ofList s (avs.mapIdx (fun j a => a * maskv j)) = some y := by
    unfold Tensor.ofList
    rw [dif_pos hlen]
    exact ⟨_, rfl⟩
  refine ⟨ys, y, happ, ?_, ?_⟩
  · simp only [decodeTensor, Option.bind_eq_bind, hys', Option.bind_some, hy]
  · intro i
    obtain ⟨hi, hxi⟩ := ofList_apply hxt i
    obtain ⟨hi', hyi⟩ := ofList_apply hy i
    rw [hxi, hyi]
    simp only [List.getElem_mapIdx, dropoutMask, maskv, Rat.cast_id]


/-! ## The f32 instance: the corrected (f32-rounded unit) relation -/

/-- `f32` zero and one as produced by `cast_raw(Int(0|1), F32)`. -/
def zero32 : F32 := F32.ofIScalar 0#i64
def one32 : F32 := F32.ofIScalar 1#i64

theorem prim_eq_i64_f32 :
    chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq .Int64 .F32 = .ok false := by
  simp [chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq,
    chelis_types.types.Prim.read_discriminant]

theorem prim_eq_f32_f32 :
    chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq .F32 .F32 = .ok true := by
  simp [chelis_types.types.Prim.Insts.CoreCmpPartialEqPrim.eq]

theorem cast_raw_int_f32 (h) (n : I64) :
    chelis_types.dtype_semantics.cast_raw (toStr "dropout" h) (.Int n) .F32 =
      .ok (.Ok ⟨.F32 (F32.ofIScalar n)⟩) := by
  simp [chelis_types.dtype_semantics.cast_raw, chelis_types.dtype_semantics.CheckedCastPlan.new,
    chelis_types.dtype_semantics.CheckedCastPlan.cast_raw,
    core.result.Result.unwrap_or_else, prim_eq_i64_f32,
    chelis_types.dtype_semantics.finalize_scalar, lift]

theorem cast_raw_zero_f32 (h) :
    chelis_types.dtype_semantics.cast_raw (toStr "dropout" h) (.Int 0#i64) .F32 =
      .ok (.Ok ⟨.F32 zero32⟩) := cast_raw_int_f32 h 0#i64

theorem float_binop_div_f32 (x d : F32) :
    chelis_types.dtype_semantics.float_binop .Div ⟨.F32 x⟩ ⟨.F32 d⟩ =
      .ok (.Ok ⟨.F32 (F32.div x d)⟩) := by
  simp [chelis_types.dtype_semantics.float_binop, chelis_types.dtype_semantics.require_family,
    chelis_types.dtype_semantics.require_same_dtype, chelis_types.dtype_semantics.ScalarValue.prim,
    chelis_types.types.Prim.is_float, prim_eq_f32_f32, chelis_types.dtype_semantics.FloatBinOp.name,
    chelis_types.dtype_semantics.apply_float_binop_f32]

theorem float_binop_sub_f32 (x d : F32) :
    chelis_types.dtype_semantics.float_binop .Sub ⟨.F32 x⟩ ⟨.F32 d⟩ =
      .ok (.Ok ⟨.F32 (F32.sub x d)⟩) := by
  simp [chelis_types.dtype_semantics.float_binop, chelis_types.dtype_semantics.require_family,
    chelis_types.dtype_semantics.require_same_dtype, chelis_types.dtype_semantics.ScalarValue.prim,
    chelis_types.types.Prim.is_float, prim_eq_f32_f32, chelis_types.dtype_semantics.FloatBinOp.name,
    chelis_types.dtype_semantics.apply_float_binop_f32]

theorem denominator_f32 (w : F32) :
    chelis_types.dtype_semantics.DropoutParameters.denominator ⟨⟨.F32 w⟩⟩ =
      .ok (.Ok ⟨.F32 (F32.sub one32 w)⟩) := by
  simp only [chelis_types.dtype_semantics.DropoutParameters.denominator,
    chelis_types.dtype_semantics.ScalarValue.prim, bind_tc_ok, cast_raw_int_f32,
    core.result.TryResult.branch]
  exact float_binop_sub_f32 one32 w

theorem scalar_at_f32 (xs : alloc.vec.Vec F32) (idx : Usize) (h : idx.val < xs.val.length) :
    chelis_types.dtype_semantics.TensorStorage.scalar_at ⟨.F32 xs⟩ idx =
      .ok ⟨.F32 (xs.val[idx.val]'h)⟩ := by
  simp [chelis_types.dtype_semantics.TensorStorage.scalar_at, alloc.vec.Vec.index_slice_index,
    alloc.vec.Vec.index_usize, h]

def f32Payload (x : SV) : Result F32 :=
  match x.bits with
  | .F32 y => .ok y
  | _ => .fail .panic

theorem mapM_f32Payload (ys : List F32) :
    (ys.map (fun y => (⟨.F32 y⟩ : SV))).mapM f32Payload = .ok ys := by
  induction ys with
  | nil => rfl
  | cons y ys ih =>
    simp only [List.map_cons, List.mapM_cons, f32Payload, bind_tc_ok, ih]
    rfl

/-- `tensor_from_scalars(F32, values)` of f32 scalars is the f32 buffer of their payloads. -/
theorem tensor_from_scalars_f32 (s : Slice SV) (ys : List F32)
    (h : s.val = ys.map (fun y => (⟨.F32 y⟩ : SV))) :
    ∃ v : alloc.vec.Vec F32, v.val = ys ∧
      chelis_types.dtype_semantics.tensor_from_scalars .F32 s = .ok ⟨.F32 v⟩ := by
  have hf : ∀ x ∈ s.val,
      core.ops.function.FnMut.call_mut
          chelis_types.dtype_semantics.tensor_from_scalars.closure_1.Insts.CoreOpsFunctionFnMutTupleSharedScalarValueF32
          .F32 x = (do let y ← f32Payload x; .ok (y, .F32)) := by
    intro x hx
    rw [h] at hx
    obtain ⟨y, _, rfl⟩ := List.mem_map.mp hx
    simp [chelis_types.dtype_semantics.tensor_from_scalars.closure_1.Insts.CoreOpsFunctionFnMutTupleSharedScalarValueF32.call_mut,
      chelis_types.dtype_semantics.tensor_from_scalars.read, f32Payload]
  have hm : (s.val.drop 0).mapM f32Payload = .ok ys := by
    rw [List.drop_zero, h]; exact mapM_f32Payload ys
  obtain ⟨it', hit⟩ := iterToList_map_slice_ok _ _ f32Payload s hf _ 0 [] ys rfl hm
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

/-- The f32 element Rust pushes at index `i`: the comparison uses `f64::from(unit as f32)`. -/
def keptF32 (xs : alloc.vec.Vec F32) (w : F32) (seed ordinal : U64) (i : ℕ) : F32 :=
  if F64.lt (F32.toF64 (F64.toF32 (unitAt seed ordinal i))) (F32.toF64 w) then zero32
  else F32.div (xs.val.getD i zero32) (F32.sub one32 w)

theorem arithUnit_f32 (u : F64) : arithUnit .F32 u = F32.toF64 (F64.toF32 u) := by
  simp [arithUnit, isF64]

theorem step_f32 (xs : alloc.vec.Vec F32) (w : F32) (seed ordinal : U64) (idx : Usize)
    (hidx : idx.val < (alloc.vec.Vec.len xs).val) :
    (F64.lt (arithUnit .F32 (unitAt seed ordinal idx.val)) (F32.toF64 w) = true ∧
        (⟨.F32 (keptF32 xs w seed ordinal idx.val)⟩ : SV) = ⟨.F32 zero32⟩) ∨
      (F64.lt (arithUnit .F32 (unitAt seed ordinal idx.val)) (F32.toF64 w) = false ∧
        ∃ sv, chelis_types.dtype_semantics.TensorStorage.scalar_at ⟨.F32 xs⟩ idx = .ok sv ∧
          chelis_types.dtype_semantics.float_binop .Div sv ⟨.F32 (F32.sub one32 w)⟩ =
            .ok (.Ok ⟨.F32 (keptF32 xs w seed ordinal idx.val)⟩)) := by
  have hlen : idx.val < xs.val.length := by simpa using hidx
  rw [arithUnit_f32]
  cases hlt : F64.lt (F32.toF64 (F64.toF32 (unitAt seed ordinal idx.val))) (F32.toF64 w)
  · refine .inr ⟨rfl, _, scalar_at_f32 xs idx hlen, ?_⟩
    rw [float_binop_div_f32]
    simp only [keptF32, hlt, Bool.false_eq_true, if_false, List.getD_eq_getElem _ _ hlen]
  · exact .inl ⟨rfl, by simp [keptF32, hlt]⟩

/-- **f32 dropout, extracted `apply`.** For f32 storage `xs` and an f32 rate `w`, the extracted
`apply` succeeds with f32 storage `ys`; element `i` is the converted zero when
`f64::from(unit as f32) < f64::from(w)` and `xs[i] / (1 - w)` (binary32 operations) otherwise. -/
theorem apply_f32 (xs : alloc.vec.Vec F32) (w : F32) (seed ordinal : U64) :
    ∃ ys : alloc.vec.Vec F32,
      chelis_types.dtype_semantics.PreparedDropout.apply
          { input := ⟨.F32 xs⟩, «parameters» := ⟨⟨.F32 w⟩⟩ } seed ordinal =
        .ok (.Ok ⟨.F32 ys⟩) ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if F64.lt (F32.toF64 (F64.toF32 (unitAt seed ordinal i))) (F32.toF64 w) then zero32
        else F32.div x (F32.sub one32 w)) := by
  obtain ⟨out, hout, hloop⟩ :=
    apply_loop_spec { input := ⟨.F32 xs⟩, «parameters» := ⟨⟨.F32 w⟩⟩ } ⟨⟨.F32 w⟩⟩ seed ordinal
      .F32 (F32.toF64 w) ⟨.F32 zero32⟩ ⟨.F32 (F32.sub one32 w)⟩ (alloc.vec.Vec.len xs)
      (fun i => ⟨.F32 (keptF32 xs w seed ordinal i)⟩)
      (step_f32 xs w seed ordinal) xs.val.length 0#usize
      (alloc.vec.Vec.with_capacity SV (alloc.vec.Vec.len xs)) (by simp) (by simp)
      (by simp [alloc.vec.Vec.with_capacity])
  have hdo : (alloc.vec.Vec.deref out).val =
      ((List.range xs.val.length).map (keptF32 xs w seed ordinal)).map
        (fun y => (⟨.F32 y⟩ : SV)) := by
    simp only [alloc.vec.Vec.deref, hout, alloc.vec.Vec.with_capacity, List.nil_append,
      List.map_map, List.range_eq_range']
    simp
  obtain ⟨ys, hys, htfs⟩ := tensor_from_scalars_f32 (alloc.vec.Vec.deref out) _ hdo
  refine ⟨ys, ?_, ?_⟩
  · simp only [chelis_types.dtype_semantics.PreparedDropout.apply,
      chelis_types.dtype_semantics.TensorStorage.prim,
      chelis_types.dtype_semantics.DropoutParameters.impl.rate,
      chelis_types.dtype_semantics.ScalarValue.as_f64_lossy, bind_tc_ok, cast_raw_zero_f32,
      core.result.TryResult.branch, denominator_f32,
      chelis_types.dtype_semantics.TensorStorage.len, hloop, htfs]
  · rw [hys]
    apply List.ext_getElem?
    intro i
    rw [List.getElem?_mapIdx]
    by_cases hi : i < xs.val.length
    · rw [List.getElem?_map, List.getElem?_range hi, List.getElem?_eq_getElem hi]
      simp [keptF32, List.getElem?_eq_getElem hi]
    · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none (by omega)]
      rfl

/-- The exact value of a finite f32 rate survives the widening `f64::from(w)`. -/
theorem realize_widen_f32 {w : F32} (hw : w.isFinite = true) :
    realize (F32.toF64 w) = some (value w) :=
  realize_cast hw binary32_to_binary64

/-- **Corrected non-f64 relation (f32, universally quantified).** For f32 storage and a finite
f32 rate `w`, the extracted `apply` drops element `i` exactly when the **f32 rounding** of the
reference unit is below the rate's value: `value (unit as f32) < value w`. The reference
`dropoutMask`, which thresholds the exact unit `DrawKey.unitValue`, is *not* what Rust computes
on this path (`f32_naive_counterexample`). -/
theorem mask_agrees_f32 (xs : alloc.vec.Vec F32) (w : F32) (hw : w.isFinite = true)
    (seed ordinal : U64) :
    ∃ ys : alloc.vec.Vec F32,
      chelis_types.dtype_semantics.PreparedDropout.apply
          { input := ⟨.F32 xs⟩, «parameters» := ⟨⟨.F32 w⟩⟩ } seed ordinal =
        .ok (.Ok ⟨.F32 ys⟩) ∧
      ys.val = xs.val.mapIdx (fun i x =>
        if value (F64.toF32 (unitAt seed ordinal i)) < value w then zero32
        else F32.div x (F32.sub one32 w)) := by
  obtain ⟨ys, happ, hys⟩ := apply_f32 xs w seed ordinal
  refine ⟨ys, happ, ?_⟩
  rw [hys]
  congr 1
  funext i x
  have hu := realize_unit (keyOfDraw seed ordinal) i
  have hrange := DrawKey.unitValue_range (keyOfDraw seed ordinal) i
  have hlt := nonf64_lt_eq (u := unitAt seed ordinal i) hu
    (by rw [abs_of_nonneg hrange.1]; exact hrange.2) (realize_widen_f32 hw)
  rw [hlt]
  simp only [decide_eq_true_eq]


/-! ## Counterexamples -/

/-- `1/2` as an f32 (the Rust-side rate of the counterexamples), built by the model's own
conversion of the unit of word `2^63`. -/
def half32 : F32 := F64.toF32 (unitOfWord 0x8000000000000000)

theorem value_half32 : value half32 = 1 / 2 := by decide +kernel

theorem half32_isFinite : half32.isFinite = true := by decide +kernel

/-- **Non-f64 mask counterexample (element decision).** For `cexSeed`, ordinal 0, index 0 the
extracted unit is exactly `1/2 - 2^-53`, so the reference drops at rate `1/2`
(`unitValue < 1/2`), but the non-f64 branch compares the f32-rounded unit `f64::from(u as f32)`,
which is exactly `1/2`, against the rate: `1/2 < 1/2` is false and Rust keeps the element.
The same unit compared exactly (the f64 branch) is dropped. -/
theorem nonf64_mask_counterexample :
    chelis_types.dtype_semantics.dropout_random_unit cexSeed 0#u64 0#u64 =
        .ok (unitOfWord 0x7FFFFFFFFFFFF800) ∧
      DrawKey.unitValue ⟨UInt64.ofBitVec cexSeed.bv, (0#u64 : U64).val⟩ (0#u64 : U64).val < 1 / 2 ∧
      realize (F32.toF64 half32) = some (1 / 2) ∧
      realize (F32.toF64 (F64.toF32 (unitOfWord 0x7FFFFFFFFFFFF800))) = some (1 / 2) ∧
      F64.lt (F32.toF64 (F64.toF32 (unitOfWord 0x7FFFFFFFFFFFF800))) (F32.toF64 half32) = false ∧
      F64.lt (unitOfWord 0x7FFFFFFFFFFFF800) (F32.toF64 half32) = true := by
  refine ⟨?_, ?_, by decide +kernel, cex_f32_round, by decide +kernel, by decide +kernel⟩
  · rw [word_eq_reference, cex_word]
  · rw [cex_unit_value]; norm_num

theorem unitAt_cex : unitAt cexSeed 0#u64 0 = unitOfWord 0x7FFFFFFFFFFFF800 := by
  unfold unitAt
  exact congrArg unitOfWord cex_word

/-- **The naive relation fails on the extracted f32 `apply`.** At `cexSeed`, ordinal 0 and
f32 rate `1/2`, the reference unit of index 0 is below the rate (the reference `dropoutMask`
drops it), yet the extracted f32 `apply` keeps element 0 and scales it. -/
theorem f32_naive_counterexample (xs : alloc.vec.Vec F32) :
    (keyOfDraw cexSeed 0#u64).unitValue 0 < value half32 ∧
    ∃ ys : alloc.vec.Vec F32,
      chelis_types.dtype_semantics.PreparedDropout.apply
          { input := ⟨.F32 xs⟩, «parameters» := ⟨⟨.F32 half32⟩⟩ } cexSeed 0#u64 =
        .ok (.Ok ⟨.F32 ys⟩) ∧
      ys.val[0]? = xs.val[0]?.map (fun x => F32.div x (F32.sub one32 half32)) := by
  refine ⟨?_, ?_⟩
  · have h : (keyOfDraw cexSeed 0#u64).unitValue 0 = (2 ^ 52 - 1) / 2 ^ 53 := cex_unit_value
    rw [h, value_half32]
    norm_num
  · obtain ⟨ys, happ, hys⟩ := apply_f32 xs half32 cexSeed 0#u64
    refine ⟨ys, happ, ?_⟩
    have hlt : F64.lt (F32.toF64 (F64.toF32 (unitOfWord 0x7FFFFFFFFFFFF800)))
        (F32.toF64 half32) = false := by decide +kernel
    rw [hys, List.getElem?_mapIdx, unitAt_cex, hlt]
    simp only [Bool.false_eq_true, if_false]

/-- binary64 `0.1` (bits `0x3FB999999999999A`). -/
def rate01 : F64 := ⟨0x3FB999999999999A#64⟩

/-- Without the exactness hypotheses the scale is wrong: at rate `0.1` the computed
`1 / (1 - 0.1)` differs from the exact `(1 - r)⁻¹`. -/
theorem scale_counterexample :
    realize rate01 = some (3602879701896397 / 36028797018963968) ∧
      realize (F64.div (BinaryFloat.ofInt .binary64 1) (F64.sub (BinaryFloat.ofInt .binary64 1) rate01))
        ≠ some ((1 - 3602879701896397 / 36028797018963968 : ℚ)⁻¹) := by
  decide +kernel

end ChelisBridge.Random
