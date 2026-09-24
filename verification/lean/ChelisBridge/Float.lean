import Aeneas.Std.Float
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Exact rational observation of the Aeneas IEEE model

`realize` maps a finite `BinaryFloat` to its exact binary rational value (never a
printed decimal), identifying the two signed zeros; NaN and infinities have no value.

The main theorem `round_exact` states that the model's single rounding step returns a
finite encoding of exactly `±n/d` whenever `n/d` is `Representable` in the format
(normal or subnormal). The corollaries `realize_add` and `realize_mul` give exact
real-number arithmetic for one IEEE operation **only under the per-operation
hypothesis that the exact result is representable** (`ExactAdd`, `ExactMul`).
Finiteness of operands and result is not sufficient: `finite_add_not_exact` and
`add_not_associative` are concrete binary64 counterexamples evaluated in the kernel.
-/

namespace ChelisBridge

open Aeneas.Std

section Format

variable (f : BinaryFormat)

/-- Unit exponent (weight of the last significand bit) of the subnormal binade. -/
def minUnit : ℤ := 1 - (f.bias : ℤ) - f.fractionBits

/-- Unit exponent of the largest finite binade. -/
def maxUnit : ℤ := (f.maxExponent : ℤ) - 1 - f.bias - f.fractionBits

/-- `q` is exactly a finite value of format `f`: its magnitude is an integer
significand of at most `fractionBits + 1` bits times a unit `2^u` in the finite range.
This covers normal and subnormal values and both zeros (`m = 0`). -/
def Representable (q : ℚ) : Prop :=
  ∃ (m : ℕ) (u : ℤ), m < 2 ^ (f.fractionBits + 1) ∧ minUnit f ≤ u ∧ u ≤ maxUnit f ∧
    |q| = (m : ℚ) * (2 : ℚ) ^ u

theorem representable_abs_iff (q : ℚ) : Representable f |q| ↔ Representable f q := by
  simp only [Representable, abs_abs]

theorem representable_neg_iff (q : ℚ) : Representable f (-q) ↔ Representable f q := by
  simp only [Representable, abs_neg]

theorem width_sub_one : f.width - 1 = f.exponentBits + f.fractionBits := by
  simp only [BinaryFormat.width]; omega

theorem exponentBits_pos : 0 < f.exponentBits := by
  cases f <;> decide

theorem maxExponent_eq : f.maxExponent + 1 = 2 ^ f.exponentBits := by
  have := Nat.one_le_two_pow (n := f.exponentBits)
  simp only [BinaryFormat.maxExponent]; omega

end Format

section Value

variable {f : BinaryFormat}

/-- Significand of a finite encoding (hidden bit made explicit for normals). -/
def significand (x : BinaryFloat f) : ℕ :=
  if x.exponent = 0 then x.fraction else 2 ^ f.fractionBits + x.fraction

/-- Weight of the last significand bit of a finite encoding. -/
def unitExp (x : BinaryFloat f) : ℤ :=
  ((max 1 x.exponent : ℕ) : ℤ) - (f.bias : ℤ) - (f.fractionBits : ℤ)

/-- Exact signed rational value of the encoding's finite interpretation. -/
def value (x : BinaryFloat f) : ℚ :=
  let v : ℚ := (x.ratio.1 : ℚ) / (x.ratio.2 : ℚ)
  if x.sign then -v else v

/-- Finite elements map to their exact binary rational value; both zeros map to `0`;
NaN and infinities are rejected. -/
def realize (x : BinaryFloat f) : Option ℚ :=
  if x.isFinite then some (value x) else none

theorem ratio_snd_pos (x : BinaryFloat f) : 0 < x.ratio.2 := by
  unfold BinaryFloat.ratio
  dsimp only
  split <;> simp

theorem ratio_value (x : BinaryFloat f) :
    (x.ratio.1 : ℚ) / (x.ratio.2 : ℚ) = (significand x : ℚ) * (2 : ℚ) ^ unitExp x := by
  unfold BinaryFloat.ratio significand unitExp
  dsimp only
  have hm : (if (x.exponent == 0) = true then x.fraction else 2 ^ f.fractionBits + x.fraction) =
      (if x.exponent = 0 then x.fraction else 2 ^ f.fractionBits + x.fraction) := by
    simp [beq_iff_eq]
  rw [hm]
  generalize (if x.exponent = 0 then x.fraction else 2 ^ f.fractionBits + x.fraction) = m
  generalize ((max 1 x.exponent : ℕ) : ℤ) - (f.bias : ℤ) - (f.fractionBits : ℤ) = e
  split
  · rename_i h
    simp only [Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat, Nat.cast_one, div_one]
    rw [← zpow_natCast, Int.toNat_of_nonneg h]
  · rename_i h
    simp only [Nat.cast_pow, Nat.cast_ofNat]
    rw [← zpow_natCast, Int.toNat_of_nonneg (by omega), zpow_neg, div_eq_mul_inv, inv_inv]

theorem value_eq (x : BinaryFloat f) :
    value x = (if x.sign then -1 else 1) * ((significand x : ℚ) * (2 : ℚ) ^ unitExp x) := by
  unfold value
  rw [ratio_value]
  split <;> ring

theorem realize_eq_some {x : BinaryFloat f} {q : ℚ} :
    realize x = some q ↔ x.isFinite = true ∧ value x = q := by
  unfold realize
  split <;> simp_all

theorem isNaN_eq_false_of_isFinite {x : BinaryFloat f} (h : x.isFinite = true) :
    x.isNaN = false := by
  simp_all [BinaryFloat.isFinite, BinaryFloat.isNaN]

theorem isInfinite_eq_false_of_isFinite {x : BinaryFloat f} (h : x.isFinite = true) :
    x.isInfinite = false := by
  simp_all [BinaryFloat.isFinite, BinaryFloat.isInfinite]

end Value

section Pack

variable (f : BinaryFormat)

theorem maxExponent_pos : 0 < f.maxExponent := by
  cases f <;> decide

theorem two_pow_width : 2 ^ f.width = 2 ^ (f.width - 1) + 2 ^ (f.width - 1) := by
  have h : f.width = (f.width - 1) + 1 := by simp only [BinaryFormat.width]; omega
  conv_lhs => rw [h]
  rw [pow_succ]; omega

theorem pack_low (e m : ℕ) (he : e < f.maxExponent) (hm : m < 2 ^ f.fractionBits) :
    m + 2 ^ f.fractionBits * e < 2 ^ (f.width - 1) := by
  have hmax := maxExponent_eq f
  rw [width_sub_one, pow_add]
  have h1 : m + 2 ^ f.fractionBits * e < 2 ^ f.fractionBits * (e + 1) := by
    rw [Nat.mul_succ]; omega
  have h2 : 2 ^ f.fractionBits * (e + 1) ≤ 2 ^ f.fractionBits * 2 ^ f.exponentBits :=
    Nat.mul_le_mul_left _ (by omega)
  rw [Nat.mul_comm (2 ^ f.exponentBits)]; omega

theorem pack_toNat (neg : Bool) (e m : ℕ) (he : e < f.maxExponent)
    (hm : m < 2 ^ f.fractionBits) :
    (BinaryFloat.pack f neg e m).bits.toNat =
      (if neg then 2 ^ (f.width - 1) else 0) + (m + 2 ^ f.fractionBits * e) := by
  unfold BinaryFloat.pack BinaryFloat.ofNat
  rw [BitVec.toNat_ofNat]
  have hlow := pack_low f e m he hm
  have hw := two_pow_width f
  have heq : (if neg then 2 ^ (f.width - 1) else 0) + e * 2 ^ f.fractionBits + m =
      (if neg then 2 ^ (f.width - 1) else 0) + (m + 2 ^ f.fractionBits * e) := by
    rw [Nat.mul_comm e]; omega
  rw [heq, Nat.mod_eq_of_lt (by split <;> omega)]

theorem pack_fields (neg : Bool) (e m : ℕ) (he : e < f.maxExponent)
    (hm : m < 2 ^ f.fractionBits) :
    (BinaryFloat.pack f neg e m).sign = neg ∧ (BinaryFloat.pack f neg e m).exponent = e ∧
      (BinaryFloat.pack f neg e m).fraction = m := by
  have hbits := pack_toNat f neg e m he hm
  have hlow := pack_low f e m he hm
  have hmag : (BinaryFloat.pack f neg e m).magnitude = m + 2 ^ f.fractionBits * e := by
    unfold BinaryFloat.magnitude
    rw [hbits]
    split
    · rw [Nat.add_mod_left, Nat.mod_eq_of_lt hlow]
    · rw [Nat.zero_add, Nat.mod_eq_of_lt hlow]
  have hp : 0 < 2 ^ f.fractionBits := Nat.two_pow_pos _
  refine ⟨?_, ?_, ?_⟩
  · unfold BinaryFloat.sign
    rw [hbits]
    cases neg
    · simp only [Bool.false_eq_true, if_false, Nat.zero_add, ge_iff_le, decide_eq_false_iff_not,
        Nat.not_le]
      exact hlow
    · simp
  · unfold BinaryFloat.exponent
    rw [hmag, Nat.add_mul_div_left _ _ hp, Nat.div_eq_of_lt hm, Nat.zero_add]
  · unfold BinaryFloat.fraction
    rw [hmag, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hm]

/-- A packed finite encoding denotes `±significand × 2^unit` exactly. -/
theorem realize_pack (neg : Bool) (e m : ℕ) (he : e < f.maxExponent)
    (hm : m < 2 ^ f.fractionBits) :
    realize (BinaryFloat.pack f neg e m) =
      some ((if neg then -1 else 1) *
        (((if e = 0 then m else 2 ^ f.fractionBits + m : ℕ) : ℚ) *
          (2 : ℚ) ^ (((max 1 e : ℕ) : ℤ) - (f.bias : ℤ) - (f.fractionBits : ℤ)))) := by
  obtain ⟨hs, hexp, hfr⟩ := pack_fields f neg e m he hm
  rw [realize_eq_some]
  refine ⟨?_, ?_⟩
  · simp only [BinaryFloat.isFinite, hexp, bne_iff_ne, ne_eq]; omega
  · rw [value_eq, hs]
    unfold significand unitExp
    rw [hexp, hfr]

/-- Signed-zero identification: both zeros are finite with exact value `0`. -/
theorem realize_zero (neg : Bool) : realize (BinaryFloat.zero f neg) = some 0 := by
  unfold BinaryFloat.zero
  rw [realize_pack f neg 0 0 (maxExponent_pos f) (Nat.two_pow_pos _)]
  simp

/-- The two zeros are distinct encodings (different sign bits) with the same value. -/
theorem zero_ne_zero_neg : BinaryFloat.zero f true ≠ BinaryFloat.zero f false := by
  intro h
  have h1 := (pack_fields f true 0 0 (maxExponent_pos f) (Nat.two_pow_pos _)).1
  have h2 := (pack_fields f false 0 0 (maxExponent_pos f) (Nat.two_pow_pos _)).1
  unfold BinaryFloat.zero at h
  rw [h] at h1
  rw [h1] at h2
  exact Bool.noConfusion h2

end Pack

section Rounding

theorem two_zpow_pos (k : ℤ) : (0 : ℚ) < 2 ^ k := zpow_pos (by norm_num) k

theorem two_pow_nat_eq_zpow (k : ℕ) : (2 : ℚ) ^ k = (2 : ℚ) ^ (k : ℤ) := (zpow_natCast 2 k).symm

/-- `ratioExponent n d` is `⌊log₂ (n / d)⌋` for positive `n`, `d`. -/
theorem ratioExponent_spec {n d : ℕ} (hn : 0 < n) (hd : 0 < d) :
    (2 : ℚ) ^ BinaryFloat.ratioExponent n d ≤ (n : ℚ) / d ∧
      (n : ℚ) / d < (2 : ℚ) ^ (BinaryFloat.ratioExponent n d + 1) := by
  have hn1 : (2 : ℚ) ^ ((n.log2 : ℕ) : ℤ) ≤ n := by
    rw [← two_pow_nat_eq_zpow]; exact_mod_cast Nat.log2_self_le hn.ne'
  have hn2 : (n : ℚ) < 2 ^ ((n.log2 : ℤ) + 1) := by
    rw [show ((n.log2 : ℤ) + 1) = ((n.log2 + 1 : ℕ) : ℤ) by push_cast; rfl,
      ← two_pow_nat_eq_zpow]
    exact_mod_cast Nat.lt_log2_self
  have hd1 : (2 : ℚ) ^ ((d.log2 : ℕ) : ℤ) ≤ d := by
    rw [← two_pow_nat_eq_zpow]; exact_mod_cast Nat.log2_self_le hd.ne'
  have hd2 : (d : ℚ) < 2 ^ ((d.log2 : ℤ) + 1) := by
    rw [show ((d.log2 : ℤ) + 1) = ((d.log2 + 1 : ℕ) : ℤ) by push_cast; rfl,
      ← two_pow_nat_eq_zpow]
    exact_mod_cast Nat.lt_log2_self
  have hdq : (0 : ℚ) < d := by exact_mod_cast hd
  obtain ⟨e, he⟩ : ∃ e : ℤ, e = (n.log2 : ℤ) - (d.log2 : ℤ) := ⟨_, rfl⟩
  have hbelow : (if e ≥ 0 then n < d * 2 ^ e.toNat else n * 2 ^ (-e).toNat < d) ↔
      (n : ℚ) / d < 2 ^ e := by
    rw [div_lt_iff₀ hdq]
    split
    · rename_i h
      rw [show (2 : ℚ) ^ e = 2 ^ e.toNat by rw [two_pow_nat_eq_zpow, Int.toNat_of_nonneg h],
        mul_comm]
      norm_cast
    · rename_i h
      rw [show (2 : ℚ) ^ e = ((2 : ℚ) ^ (-e).toNat)⁻¹ by
          rw [two_pow_nat_eq_zpow, Int.toNat_of_nonneg (by omega), zpow_neg, inv_inv],
        inv_mul_eq_div, lt_div_iff₀ (by positivity)]
      norm_cast
  have hre : BinaryFloat.ratioExponent n d = if (n : ℚ) / d < 2 ^ e then e - 1 else e := by
    unfold BinaryFloat.ratioExponent
    simp only [← he]
    simp only [hbelow]
  rw [hre]
  split
  · rename_i hb
    refine ⟨?_, by rw [sub_add_cancel]; exact hb⟩
    rw [le_div_iff₀ hdq]
    have : (2 : ℚ) ^ (e - 1) * 2 ^ ((d.log2 : ℤ) + 1) = 2 ^ ((n.log2 : ℕ) : ℤ) := by
      rw [← zpow_add₀ two_ne_zero]; congr 1; omega
    have hmono : (2 : ℚ) ^ (e - 1) * d < 2 ^ (e - 1) * 2 ^ ((d.log2 : ℤ) + 1) :=
      mul_lt_mul_of_pos_left hd2 (two_zpow_pos _)
    linarith
  · rename_i hb
    refine ⟨le_of_not_gt hb, ?_⟩
    rw [div_lt_iff₀ hdq]
    have : (2 : ℚ) ^ (e + 1) * 2 ^ ((d.log2 : ℕ) : ℤ) = 2 ^ ((n.log2 : ℤ) + 1) := by
      rw [← zpow_add₀ two_ne_zero]; congr 1; omega
    have hmono : (2 : ℚ) ^ (e + 1) * 2 ^ ((d.log2 : ℕ) : ℤ) ≤ 2 ^ (e + 1) * d :=
      mul_le_mul_of_nonneg_left hd1 (le_of_lt (two_zpow_pos _))
    linarith

/-- An exact quotient is returned unchanged by round-half-even. -/
theorem roundQuotient_exact (b Q : ℕ) (hb : 0 < b) :
    BinaryFloat.roundQuotient (b * Q) b = Q := by
  unfold BinaryFloat.roundQuotient
  simp only [Nat.mul_div_cancel_left Q hb, Nat.mul_mod_right, Nat.mul_zero]
  have : ¬ (0 > b) := by omega
  have : (0 == b) = false := by simp; omega
  simp_all

variable (f : BinaryFormat)

/-- **Exact rounding.** If `n / d` is representable in format `f`, the model's single
rounding step returns a finite encoding whose exact value is `±n/d` (sign `neg`).
For `n = 0` the result is the zero with the requested sign, whose value is `0`. -/
theorem round_exact (neg : Bool) (n d : ℕ) (hd : 0 < d)
    (h : Representable f ((n : ℚ) / d)) :
    realize (BinaryFloat.round f neg n d) =
      some (if neg then -((n : ℚ) / d) else (n : ℚ) / d) := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · have : BinaryFloat.round f neg 0 d = BinaryFloat.zero f neg := by
      simp [BinaryFloat.round, hd.ne']
    rw [this, realize_zero]
    simp
  obtain ⟨m, u, hm, hu1, hu2, hq⟩ := h
  have hdq : (0 : ℚ) < d := by exact_mod_cast hd
  have hnq : (0 : ℚ) < n := by exact_mod_cast hn
  have hpos : (0 : ℚ) < (n : ℚ) / d := div_pos hnq hdq
  rw [abs_of_pos hpos] at hq
  obtain ⟨hL1, hL2⟩ := ratioExponent_spec hn hd
  obtain ⟨L, hL⟩ : ∃ L, L = BinaryFloat.ratioExponent n d := ⟨_, rfl⟩
  rw [← hL] at hL1 hL2
  have hm2 : (m : ℚ) < 2 ^ ((f.fractionBits : ℤ) + 1) := by
    rw [show ((f.fractionBits : ℤ) + 1) = ((f.fractionBits + 1 : ℕ) : ℤ) by push_cast; rfl,
      ← two_pow_nat_eq_zpow]
    exact_mod_cast hm
  -- The exact value fixes the binade: `L - p ≤ u`.
  have hLu : L - f.fractionBits ≤ u := by
    by_contra hc
    have hlt : (m : ℚ) * 2 ^ u < 2 ^ ((f.fractionBits : ℤ) + 1) * 2 ^ u :=
      mul_lt_mul_of_pos_right hm2 (two_zpow_pos u)
    rw [← zpow_add₀ two_ne_zero] at hlt
    have hle : (2 : ℚ) ^ ((f.fractionBits : ℤ) + 1 + u) ≤ 2 ^ L :=
      zpow_le_zpow_right₀ (by norm_num) (by omega)
    linarith
  obtain ⟨unit, hunit⟩ : ∃ unit : ℤ,
      unit = max (1 - (f.bias : ℤ) - (f.fractionBits : ℤ)) (L - (f.fractionBits : ℤ)) :=
    ⟨_, rfl⟩
  have hunit_le : unit ≤ u := by
    rw [hunit]; exact max_le (by simpa [minUnit] using hu1) hLu
  have hunit_min : 1 - (f.bias : ℤ) - f.fractionBits ≤ unit := by rw [hunit]; exact le_max_left _ _
  have hunit_L : L - f.fractionBits ≤ unit := by rw [hunit]; exact le_max_right _ _
  obtain ⟨k, hk⟩ : ∃ k : ℕ, (k : ℤ) = u - unit := ⟨(u - unit).toNat, Int.toNat_of_nonneg (by omega)⟩
  have hQv : (n : ℚ) / d = ((m * 2 ^ k : ℕ) : ℚ) * 2 ^ unit := by
    rw [hq]; push_cast
    rw [two_pow_nat_eq_zpow, mul_assoc, ← zpow_add₀ two_ne_zero, hk]; ring_nf
  obtain ⟨Q, hQ⟩ : ∃ Q, Q = m * 2 ^ k := ⟨_, rfl⟩
  rw [← hQ] at hQv
  -- The significand has at most `p + 1` bits: no carry.
  have hQlt : Q < 2 ^ (f.fractionBits + 1) := by
    have h1 : (Q : ℚ) * 2 ^ unit < 2 ^ (L + 1) := hQv ▸ hL2
    have h2 : (2 : ℚ) ^ (L + 1) ≤ 2 ^ ((f.fractionBits : ℤ) + 1 + unit) :=
      zpow_le_zpow_right₀ (by norm_num) (by omega)
    rw [zpow_add₀ two_ne_zero ((f.fractionBits : ℤ) + 1) unit] at h2
    have h3 : (Q : ℚ) < 2 ^ ((f.fractionBits : ℤ) + 1) :=
      lt_of_mul_lt_mul_right (lt_of_lt_of_le h1 h2) (le_of_lt (two_zpow_pos unit))
    rw [show ((f.fractionBits : ℤ) + 1) = ((f.fractionBits + 1 : ℕ) : ℤ) by push_cast; rfl,
      ← two_pow_nat_eq_zpow] at h3
    exact_mod_cast h3
  -- The quotient computed by `round` is exact.
  have hnd : (n : ℚ) = d * Q * 2 ^ unit := by
    rw [div_eq_iff hdq.ne'] at hQv; rw [hQv]; ring
  have hqc : (if unit ≥ 0 then BinaryFloat.roundQuotient n (d * 2 ^ unit.toNat)
      else BinaryFloat.roundQuotient (n * 2 ^ (-unit).toNat) d) = Q := by
    split
    · rename_i hu0
      have : n = d * 2 ^ unit.toNat * Q := by
        have : (n : ℚ) = d * 2 ^ unit.toNat * Q := by
          rw [hnd, two_pow_nat_eq_zpow, Int.toNat_of_nonneg hu0]; ring
        exact_mod_cast this
      rw [this]; exact roundQuotient_exact _ _ (by positivity)
    · rename_i hu0
      have : n * 2 ^ (-unit).toNat = d * Q := by
        have : (n : ℚ) * 2 ^ (-unit).toNat = d * Q := by
          rw [hnd, two_pow_nat_eq_zpow, Int.toNat_of_nonneg (by omega), mul_assoc,
            ← zpow_add₀ two_ne_zero]
          simp
        exact_mod_cast this
      rw [this]; exact roundQuotient_exact _ _ hd
  have hround : BinaryFloat.round f neg n d =
      if Q < 2 ^ f.fractionBits then BinaryFloat.pack f neg 0 Q
      else BinaryFloat.pack f neg (unit + f.fractionBits + f.bias).toNat (Q - 2 ^ f.fractionBits) := by
    unfold BinaryFloat.round
    simp only [beq_iff_eq, hd.ne', hn.ne', if_false, ← hL, ← hunit, hqc]
    have hc : ¬ (Q ≥ 2 ^ (f.fractionBits + 1)) := by omega
    simp only [hc, if_false]
    split
    · rename_i hQp
      have : ¬ (0 ≥ f.maxExponent) := by have := maxExponent_pos f; omega
      simp [this]
    · rename_i hQp
      have hmaxu : u ≤ (f.maxExponent : ℤ) - 1 - f.bias - f.fractionBits := by
        simpa [maxUnit] using hu2
      have : ¬ ((unit + f.fractionBits + f.bias).toNat ≥ f.maxExponent) := by omega
      have hne : ¬ ((unit + f.fractionBits + f.bias).toNat = 0) := by omega
      simp [this, hne]
  rw [hround]
  have hmax := maxExponent_pos f
  split
  · rename_i hQp
    -- Subnormal: the unit is the minimum one.
    have hunit_eq : unit = 1 - (f.bias : ℤ) - f.fractionBits := by
      by_contra hne
      have hul : unit = L - f.fractionBits := by rw [hunit] at hne ⊢; omega
      have : (2 : ℚ) ^ L ≤ Q * 2 ^ unit := hQv ▸ hL1
      rw [hul] at this
      have hQp' : (Q : ℚ) < 2 ^ (f.fractionBits : ℤ) := by
        rw [← two_pow_nat_eq_zpow]; exact_mod_cast hQp
      have : (Q : ℚ) * 2 ^ (L - f.fractionBits) < 2 ^ (f.fractionBits : ℤ) * 2 ^ (L - f.fractionBits) :=
        mul_lt_mul_of_pos_right hQp' (two_zpow_pos _)
      rw [← zpow_add₀ two_ne_zero, add_sub_cancel] at this
      linarith
    rw [realize_pack f neg 0 Q hmax hQp]
    congr 1
    rw [hQv, hunit_eq]
    split <;> simp
  · rename_i hQp
    have hmaxu : u ≤ (f.maxExponent : ℤ) - 1 - f.bias - f.fractionBits := by
      simpa [maxUnit] using hu2
    have he : (unit + f.fractionBits + f.bias).toNat < f.maxExponent := by omega
    rw [realize_pack f neg _ _ he (by omega)]
    congr 1
    have hne : ¬ ((unit + f.fractionBits + f.bias).toNat = 0) := by omega
    have hmax1 : ((max 1 (unit + f.fractionBits + f.bias).toNat : ℕ) : ℤ) - f.bias - f.fractionBits
        = unit := by omega
    have hsig : (2 ^ f.fractionBits + (Q - 2 ^ f.fractionBits) : ℕ) = Q := by omega
    rw [if_neg hne, hsig, hmax1, hQv]
    split <;> simp

/-- Integer-numerator form of `round_exact`: a signed exact quotient is returned exactly
whenever the requested sign agrees with the numerator's sign (either sign for zero). -/
theorem round_exact_int (neg : Bool) (N : ℤ) (D : ℕ) (hD : 0 < D)
    (hneg : N < 0 → neg = true) (hpos : 0 < N → neg = false)
    (h : Representable f ((N : ℚ) / D)) :
    realize (BinaryFloat.round f neg N.natAbs D) = some ((N : ℚ) / D) := by
  have hDq : (0 : ℚ) < D := by exact_mod_cast hD
  have habs : ((N.natAbs : ℕ) : ℚ) / D = |(N : ℚ) / D| := by
    rw [abs_div, abs_of_pos hDq, Nat.cast_natAbs, Int.cast_abs]
  rw [round_exact f neg N.natAbs D hD (by rw [habs, representable_abs_iff]; exact h)]
  congr 1
  rcases lt_trichotomy N 0 with hN | hN | hN
  · have hq : (N : ℚ) / D < 0 := div_neg_of_neg_of_pos (by exact_mod_cast hN) hDq
    rw [hneg hN, if_pos rfl, habs, abs_of_neg hq, neg_neg]
  · subst hN; simp
  · have hq : 0 < (N : ℚ) / D := div_pos (by exact_mod_cast hN) hDq
    rw [hpos hN, habs, abs_of_pos hq]
    rfl

end Rounding

section Arithmetic

variable {f : BinaryFormat}

/-- Per-operation exact-execution hypothesis for one IEEE addition: both operands are
finite and the exact rational sum is representable in the format. Finiteness alone does
not imply this (see `finite_add_not_exact`). -/
def ExactAdd (x y : BinaryFloat f) : Prop :=
  ∃ a b, realize x = some a ∧ realize y = some b ∧ Representable f (a + b)

/-- Per-operation exact-execution hypothesis for one IEEE multiplication. -/
def ExactMul (x y : BinaryFloat f) : Prop :=
  ∃ a b, realize x = some a ∧ realize y = some b ∧ Representable f (a * b)

theorem value_ratio (x : BinaryFloat f) :
    value x = (((if x.sign then -(x.ratio.1 : ℤ) else (x.ratio.1 : ℤ)) : ℤ) : ℚ) /
      (x.ratio.2 : ℚ) := by
  unfold value
  split <;> push_cast <;> ring

/-- One exact IEEE addition is real addition of the realized operands. -/
theorem realize_add {x y : BinaryFloat f} {a b : ℚ} (hx : realize x = some a)
    (hy : realize y = some b) (h : Representable f (a + b)) :
    realize (BinaryFloat.add x y) = some (a + b) := by
  obtain ⟨hxf, rfl⟩ := realize_eq_some.mp hx
  obtain ⟨hyf, rfl⟩ := realize_eq_some.mp hy
  have hxd := ratio_snd_pos x
  have hyd := ratio_snd_pos y
  rw [value_ratio x, value_ratio y] at h ⊢
  unfold BinaryFloat.add
  rw [isNaN_eq_false_of_isFinite hxf, isNaN_eq_false_of_isFinite hyf,
    isInfinite_eq_false_of_isFinite hxf, isInfinite_eq_false_of_isFinite hyf]
  simp only [Bool.or_false, Bool.false_eq_true, if_false]
  generalize x.ratio = rx at hxd h ⊢
  generalize y.ratio = ry at hyd h ⊢
  obtain ⟨xn, xd⟩ := rx
  obtain ⟨yn, yd⟩ := ry
  dsimp only at hxd hyd h ⊢
  generalize (if x.sign = true then -(xn : ℤ) else (xn : ℤ)) = A at h ⊢
  generalize (if y.sign = true then -(yn : ℤ) else (yn : ℤ)) = B at h ⊢
  have hxq : (xd : ℚ) ≠ 0 := by exact_mod_cast hxd.ne'
  have hyq : (yd : ℚ) ≠ 0 := by exact_mod_cast hyd.ne'
  have hN : (A : ℚ) / xd + B / yd = ((A * yd + B * xd : ℤ) : ℚ) / ((xd * yd : ℕ) : ℚ) := by
    push_cast; field_simp
  rw [hN] at h ⊢
  apply round_exact_int f _ _ _ (Nat.mul_pos hxd hyd)
  · intro hlt; simp [hlt.ne, hlt]
  · intro hgt; simp [hgt.ne', not_lt.mpr hgt.le]
  · exact h

/-- One exact IEEE multiplication is real multiplication of the realized operands. -/
theorem realize_mul {x y : BinaryFloat f} {a b : ℚ} (hx : realize x = some a)
    (hy : realize y = some b) (h : Representable f (a * b)) :
    realize (BinaryFloat.mul x y) = some (a * b) := by
  obtain ⟨hxf, rfl⟩ := realize_eq_some.mp hx
  obtain ⟨hyf, rfl⟩ := realize_eq_some.mp hy
  have hxd := ratio_snd_pos x
  have hyd := ratio_snd_pos y
  unfold BinaryFloat.mul
  rw [isNaN_eq_false_of_isFinite hxf, isNaN_eq_false_of_isFinite hyf,
    isInfinite_eq_false_of_isFinite hxf, isInfinite_eq_false_of_isFinite hyf]
  simp only [Bool.or_false, Bool.false_and, Bool.false_eq_true, if_false]
  unfold value at h ⊢
  generalize x.ratio = rx at hxd h ⊢
  generalize y.ratio = ry at hyd h ⊢
  obtain ⟨xn, xd⟩ := rx
  obtain ⟨yn, yd⟩ := ry
  dsimp only at hxd hyd h ⊢
  have hax : ∀ (s : Bool) (n d : ℕ),
      |(if s = true then -((n : ℚ) / d) else (n : ℚ) / d)| = (n : ℚ) / d := by
    intro s n d; split <;> simp [abs_div]
  have key : Representable f (((xn * yn : ℕ) : ℚ) / ((xd * yd : ℕ) : ℚ)) := by
    rw [← representable_abs_iff, abs_mul, hax, hax] at h
    push_cast; rw [mul_div_mul_comm]; exact h
  rw [round_exact f _ _ _ (Nat.mul_pos hxd hyd) key]
  congr 1
  cases x.sign <;> cases y.sign <;> push_cast <;> simp <;> ring

theorem realize_add_of_exact {x y : BinaryFloat f} (h : ExactAdd x y) :
    ∃ a b, realize x = some a ∧ realize y = some b ∧ realize (BinaryFloat.add x y) = some (a + b) :=
  let ⟨a, b, hx, hy, hr⟩ := h; ⟨a, b, hx, hy, realize_add hx hy hr⟩

theorem realize_mul_of_exact {x y : BinaryFloat f} (h : ExactMul x y) :
    ∃ a b, realize x = some a ∧ realize y = some b ∧ realize (BinaryFloat.mul x y) = some (a * b) :=
  let ⟨a, b, hx, hy, hr⟩ := h; ⟨a, b, hx, hy, realize_mul hx hy hr⟩

end Arithmetic

section Counterexample

/-- `10^16`, `-10^16` and `1` in binary64, built by the model's own integer conversion. -/
def c16 : F64 := BinaryFloat.ofInt .binary64 (10 ^ 16)
def cneg16 : F64 := BinaryFloat.ofInt .binary64 (-(10 ^ 16))
def cone : F64 := BinaryFloat.ofInt .binary64 1

theorem realize_c16 : realize c16 = some (10 ^ 16) := by decide +kernel
theorem realize_cneg16 : realize cneg16 = some (-(10 ^ 16)) := by decide +kernel
theorem realize_cone : realize cone = some 1 := by decide +kernel

/-- `(10^16 + -10^16) + 1 = 1` but `10^16 + (-10^16 + 1) = 0` in the model: `-10^16 + 1`
is a binary64 tie that rounds (to even) back to `-10^16`. Every value involved is finite. -/
theorem add_not_associative :
    realize (BinaryFloat.add (BinaryFloat.add c16 cneg16) cone) = some 1 ∧
      realize (BinaryFloat.add c16 (BinaryFloat.add cneg16 cone)) = some 0 := by
  decide +kernel

/-- Finite operands and a finite result do not make one addition exact:
`-10^16 + 1` is finite but realizes to `-10^16`. -/
theorem finite_add_not_exact :
    cneg16.isFinite = true ∧ cone.isFinite = true ∧
      (BinaryFloat.add cneg16 cone).isFinite = true ∧
      realize (BinaryFloat.add cneg16 cone) = some (-(10 ^ 16)) := by
  decide +kernel

/-- Hence the unconditional law "finite operands and result ⇒ exact sum" is false;
the corrected hypothesis is `ExactAdd` (representability of the exact sum). -/
theorem not_finite_add_exact :
    ¬ ∀ (x y : F64) (a b : ℚ), realize x = some a → realize y = some b →
      (BinaryFloat.add x y).isFinite = true → realize (BinaryFloat.add x y) = some (a + b) := by
  intro h
  have h1 := h cneg16 cone _ _ realize_cneg16 realize_cone finite_add_not_exact.2.2.1
  rw [finite_add_not_exact.2.2.2] at h1
  norm_num at h1

end Counterexample

end ChelisBridge
