import ChelisBridge.Float

/-!
# Exact widening and narrowing of finite values

The Rust f32 paths materialize values through binary64: constants via `as_f64_lossy` and
`x as f32` (`finalize_tensor`), typed `Sum` results via `as_f64_lossy` and `x as f32`. For a
finite f32 value both conversions are exact: `realize_cast_of_representable`,
`representable_value` and `f32_roundtrip_realize`.
-/

namespace ChelisBridge.Numeric

open Aeneas.Std ChelisBridge

variable {f : BinaryFormat}

/-- A finite encoding's exact value is representable in its own format. -/
theorem representable_value {x : BinaryFloat f} (hx : x.isFinite = true) :
    Representable f (value x) := by
  have hexp : x.exponent < f.maxExponent := by
    simp only [BinaryFloat.isFinite, bne_iff_ne, ne_eq] at hx
    have h2 : 2 ^ (f.width - 1) = 2 ^ f.fractionBits * 2 ^ f.exponentBits := by
      rw [width_sub_one, pow_add, Nat.mul_comm]
    have hmag : x.magnitude < 2 ^ f.fractionBits * 2 ^ f.exponentBits := by
      unfold BinaryFloat.magnitude; rw [← h2]; exact Nat.mod_lt _ (Nat.two_pow_pos _)
    have hlt : x.exponent < 2 ^ f.exponentBits := by
      unfold BinaryFloat.exponent; exact Nat.div_lt_of_lt_mul hmag
    have hm := maxExponent_eq f
    omega
  have hfr : x.fraction < 2 ^ f.fractionBits := Nat.mod_lt _ (Nat.two_pow_pos _)
  refine ⟨significand x, unitExp x, ?_, ?_, ?_, ?_⟩
  · have h2p : 2 ^ (f.fractionBits + 1) = 2 ^ f.fractionBits * 2 := pow_succ _ _
    have hfr' := hfr
    unfold significand
    split <;> omega
  · unfold unitExp minUnit; omega
  · unfold unitExp maxUnit
    have : max 1 x.exponent ≤ f.maxExponent - 1 := by
      have : 2 ≤ f.maxExponent := by cases f <;> decide
      omega
    omega
  · rw [value_eq]
    have hpos : (0 : ℚ) ≤ (significand x : ℚ) * (2 : ℚ) ^ unitExp x := by positivity
    split <;> simp [abs_of_nonneg hpos]

/-- Converting a finite value into a format where it is representable is exact. -/
theorem realize_cast_of_representable {g : BinaryFormat} {x : BinaryFloat f}
    (hx : x.isFinite = true) (hr : Representable g (value x)) :
    realize (BinaryFloat.cast g x) = some (value x) := by
  unfold BinaryFloat.cast
  rw [isNaN_eq_false_of_isFinite hx, isInfinite_eq_false_of_isFinite hx]
  simp only [Bool.false_eq_true, if_false]
  have hd := ratio_snd_pos x
  have hq : ((x.ratio.1 : ℚ) / x.ratio.2) = |value x| := by
    unfold value
    have : (0 : ℚ) ≤ (x.ratio.1 : ℚ) / x.ratio.2 := by positivity
    split <;> simp [abs_of_nonneg this]
  rw [round_exact g x.sign x.ratio.1 x.ratio.2 hd (by rw [hq, representable_abs_iff]; exact hr)]
  unfold value
  split <;> rfl

/-- Every finite binary32 value is representable in binary64. -/
theorem representable_f32_in_f64 {q : ℚ} (h : Representable .binary32 q) :
    Representable .binary64 q := by
  obtain ⟨m, u, hm, hu1, hu2, hq⟩ := h
  refine ⟨m, u, lt_of_lt_of_le hm (Nat.pow_le_pow_right (by norm_num) (by decide)), ?_, ?_, hq⟩
  · simp only [minUnit] at hu1 ⊢; simp [BinaryFormat.bias, BinaryFormat.exponentBits,
      BinaryFormat.fractionBits] at hu1 ⊢; omega
  · simp only [maxUnit] at hu2 ⊢; simp [BinaryFormat.bias, BinaryFormat.exponentBits,
      BinaryFormat.fractionBits, BinaryFormat.maxExponent] at hu2 ⊢; omega

/-- **The f32 round trip through binary64 is exact** on finite values
(`(x as f64) as f32` realizes to `x`). -/
theorem f32_roundtrip_realize {x : F32} (hx : x.isFinite = true) :
    realize (F64.toF32 (F32.toF64 x)) = some (value x) ∧
      realize (F32.toF64 x) = some (value x) := by
  have hr32 := representable_value hx
  have h64 := realize_cast_of_representable hx (representable_f32_in_f64 hr32)
  refine ⟨?_, h64⟩
  obtain ⟨hfin, hval⟩ := realize_eq_some.mp h64
  have := realize_cast_of_representable (g := .binary32) hfin (by rw [hval]; exact hr32)
  rw [hval] at this
  exact this

end ChelisBridge.Numeric
