import ChelisBridge.Float

namespace ChelisBridge.Protocol

open Aeneas.Std ChelisBridge

variable {f : BinaryFormat}

/-- Unsigned exact value of an encoding's finite interpretation. -/
def uval (x : BinaryFloat f) : ℚ := (significand x : ℚ) * (2 : ℚ) ^ unitExp x

theorem magnitude_eq (x : BinaryFloat f) :
    x.magnitude = x.exponent * 2 ^ f.fractionBits + x.fraction := by
  unfold BinaryFloat.exponent BinaryFloat.fraction
  rw [Nat.mul_comm]; exact (Nat.div_add_mod _ _).symm

theorem fraction_lt (x : BinaryFloat f) : x.fraction < 2 ^ f.fractionBits :=
  Nat.mod_lt _ (Nat.two_pow_pos _)

theorem uval_nonneg (x : BinaryFloat f) : 0 ≤ uval x :=
  mul_nonneg (Nat.cast_nonneg _) (le_of_lt (two_zpow_pos _))

theorem value_eq_uval (x : BinaryFloat f) : value x = if x.sign then -uval x else uval x := by
  rw [value_eq]; unfold uval; split <;> ring

/-- Upper bound of a binade: `uval x < 2 ^ (e + 1 - bias)`. -/
theorem uval_lt (x : BinaryFloat f) :
    uval x < (2 : ℚ) ^ ((x.exponent : ℤ) + 1 - f.bias) := by
  have ht := fraction_lt x
  unfold uval significand unitExp
  by_cases he : x.exponent = 0
  · simp only [he, if_true]
    have hmax : ((max 1 0 : ℕ) : ℤ) = 1 := by simp
    rw [hmax]
    have : (2 : ℚ) ^ ((0 : ℕ) + 1 - (f.bias : ℤ) : ℤ) =
        (2 : ℚ) ^ (f.fractionBits : ℤ) * 2 ^ ((1 : ℤ) - f.bias - f.fractionBits) := by
      rw [← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring
    rw [this]
    apply mul_lt_mul_of_pos_right _ (two_zpow_pos _)
    rw [← two_pow_nat_eq_zpow]; exact_mod_cast ht
  · simp only [he, if_false]
    have hmax : ((max 1 x.exponent : ℕ) : ℤ) = x.exponent := by
      congr 1; omega
    rw [hmax]
    have : (2 : ℚ) ^ ((x.exponent : ℤ) + 1 - f.bias) =
        (2 : ℚ) ^ ((f.fractionBits + 1 : ℕ) : ℤ) * 2 ^ ((x.exponent : ℤ) - f.bias - f.fractionBits) := by
      rw [← zpow_add₀ two_ne_zero]; congr 1; push_cast; ring
    rw [this]
    apply mul_lt_mul_of_pos_right _ (two_zpow_pos _)
    rw [← two_pow_nat_eq_zpow, pow_succ]; push_cast
    have : (x.fraction : ℚ) < 2 ^ f.fractionBits := by exact_mod_cast ht
    linarith

/-- Lower bound of a normal binade: `2 ^ (e - bias) ≤ uval x`. -/
theorem le_uval (x : BinaryFloat f) (he : x.exponent ≠ 0) :
    (2 : ℚ) ^ ((x.exponent : ℤ) - f.bias) ≤ uval x := by
  unfold uval significand unitExp
  simp only [he, if_false]
  have hmax : ((max 1 x.exponent : ℕ) : ℤ) = x.exponent := by congr 1; omega
  rw [hmax]
  have : (2 : ℚ) ^ ((x.exponent : ℤ) - f.bias) =
      (2 : ℚ) ^ (f.fractionBits : ℤ) * 2 ^ ((x.exponent : ℤ) - f.bias - f.fractionBits) := by
    rw [← zpow_add₀ two_ne_zero]; congr 1; ring
  rw [this]
  apply mul_le_mul_of_nonneg_right _ (le_of_lt (two_zpow_pos _))
  rw [← two_pow_nat_eq_zpow]; push_cast
  have : (0 : ℚ) ≤ x.fraction := Nat.cast_nonneg _
  linarith

/-- `uval` is strictly monotone in the magnitude field. -/
theorem uval_lt_uval {x y : BinaryFloat f} (h : x.magnitude < y.magnitude) :
    uval x < uval y := by
  rw [magnitude_eq, magnitude_eq] at h
  have htx := fraction_lt x
  have hty := fraction_lt y
  have hp := Nat.two_pow_pos f.fractionBits
  rcases Nat.lt_or_ge x.exponent y.exponent with hlt | hge
  · have hy0 : y.exponent ≠ 0 := by omega
    calc uval x < (2 : ℚ) ^ ((x.exponent : ℤ) + 1 - f.bias) := uval_lt x
      _ ≤ (2 : ℚ) ^ ((y.exponent : ℤ) - f.bias) :=
          zpow_le_zpow_right₀ (by norm_num) (by omega)
      _ ≤ uval y := le_uval y hy0
  · have heq : x.exponent = y.exponent := by
      rcases Nat.lt_or_ge y.exponent x.exponent with h2 | h2
      · exfalso
        have : y.exponent * 2 ^ f.fractionBits + 2 ^ f.fractionBits ≤
            x.exponent * 2 ^ f.fractionBits := by
          rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ h2
        omega
      · omega
    have hfr : x.fraction < y.fraction := by rw [heq] at h; omega
    unfold uval
    have hu : unitExp x = unitExp y := by unfold unitExp; rw [heq]
    rw [hu]
    apply mul_lt_mul_of_pos_right _ (two_zpow_pos _)
    unfold significand
    rw [heq]
    split <;> push_cast <;> exact_mod_cast (by omega)

theorem uval_eq_of_magnitude {x y : BinaryFloat f} (h : x.magnitude = y.magnitude) :
    uval x = uval y := by
  have he : x.exponent = y.exponent := by unfold BinaryFloat.exponent; rw [h]
  have hf : x.fraction = y.fraction := by unfold BinaryFloat.fraction; rw [h]
  unfold uval significand unitExp; rw [he, hf]

theorem uval_lt_iff {x y : BinaryFloat f} : uval x < uval y ↔ x.magnitude < y.magnitude := by
  constructor
  · intro h
    by_contra hn
    rcases Nat.lt_or_eq_of_le (Nat.le_of_not_lt hn) with h2 | h2
    · exact absurd (uval_lt_uval h2) (not_lt.mpr h.le)
    · rw [uval_eq_of_magnitude h2] at h; exact lt_irrefl _ h
  · exact uval_lt_uval

theorem uval_eq_zero_iff (x : BinaryFloat f) : uval x = 0 ↔ x.magnitude = 0 := by
  constructor
  · intro h
    by_contra hn
    have hz := pack_fields f false 0 0 (maxExponent_pos f) (Nat.two_pow_pos _)
    have hm0 : (BinaryFloat.pack f false 0 0).magnitude = 0 := by
      rw [magnitude_eq, hz.2.1, hz.2.2]; simp
    have := uval_lt_uval (x := BinaryFloat.pack f false 0 0) (y := x) (by omega)
    have h0 : uval (BinaryFloat.pack f false 0 0) = 0 := by
      unfold uval significand; rw [hz.2.1, hz.2.2]; simp
    rw [h0, h] at this; exact lt_irrefl _ this
  · intro h
    have he : x.exponent = 0 := by unfold BinaryFloat.exponent; rw [h]; simp
    have hf : x.fraction = 0 := by unfold BinaryFloat.fraction; rw [h]; simp
    unfold uval significand; rw [he, hf]; simp

/-- **IEEE `<` on finite encodings is exact rational `<`** (signed zeros compare equal). -/
theorem lt_eq_decide {x y : BinaryFloat f} (hx : x.isFinite = true) (hy : y.isFinite = true) :
    BinaryFloat.lt x y = decide (value x < value y) := by
  have hxn := isNaN_eq_false_of_isFinite hx
  have hyn := isNaN_eq_false_of_isFinite hy
  have hx0 := uval_nonneg x
  have hy0 := uval_nonneg y
  unfold BinaryFloat.lt
  rw [hxn, hyn, value_eq_uval, value_eq_uval]
  simp only [Bool.false_or]
  by_cases hz : (x.isZero && y.isZero) = true
  · rw [if_pos hz]
    simp only [Bool.and_eq_true, BinaryFloat.isZero, beq_iff_eq] at hz
    rw [(uval_eq_zero_iff x).mpr hz.1, (uval_eq_zero_iff y).mpr hz.2]
    simp
  · rw [if_neg hz]
    simp only [Bool.and_eq_true, BinaryFloat.isZero, beq_iff_eq, not_and_or] at hz
    have hpos : 0 < uval x ∨ 0 < uval y := by
      rcases hz with h | h
      · exact Or.inl (lt_of_le_of_ne hx0 (fun e => h ((uval_eq_zero_iff x).mp e.symm)))
      · exact Or.inr (lt_of_le_of_ne hy0 (fun e => h ((uval_eq_zero_iff y).mp e.symm)))
    cases hsx : x.sign <;> cases hsy : y.sign <;> simp only [bne_self_eq_false, Bool.false_eq_true,
      if_false, if_true, bne_iff_ne, ne_eq, not_false_eq_true, reduceCtorEq]
    · exact decide_eq_decide.mpr uval_lt_iff.symm
    · symm; rw [decide_eq_false_iff_not, not_lt]; rcases hpos with h | h <;> linarith
    · symm; rw [decide_eq_true_iff]; rcases hpos with h | h <;> linarith
    · exact decide_eq_decide.mpr (by rw [neg_lt_neg_iff]; exact uval_lt_iff.symm)

theorem abs_value (x : BinaryFloat f) : |value x| = uval x := by
  rw [value_eq_uval]; split
  · rw [abs_neg, abs_of_nonneg (uval_nonneg x)]
  · exact abs_of_nonneg (uval_nonneg x)

theorem isZero_eq_false_of_value_ne {x : BinaryFloat f} (h : value x ≠ 0) :
    x.isZero = false := by
  cases hz : x.isZero
  · rfl
  · exfalso; apply h
    have hm : x.magnitude = 0 := by simpa [BinaryFloat.isZero] using hz
    rw [value_eq_uval, (uval_eq_zero_iff x).mpr hm]; simp

theorem ratio_fst_pos_of_value_ne {x : BinaryFloat f} (h : value x ≠ 0) : 0 < x.ratio.1 := by
  rcases Nat.eq_zero_or_pos x.ratio.1 with h0 | h0
  · exfalso; apply h; unfold value; rw [h0]; simp
  · exact h0

/-- One exact IEEE division is real division of the realized operands (nonzero divisor). -/
theorem realize_div {x y : BinaryFloat f} {a b : ℚ} (hx : realize x = some a)
    (hy : realize y = some b) (hb : b ≠ 0) (h : Representable f (a / b)) :
    realize (BinaryFloat.div x y) = some (a / b) := by
  obtain ⟨hxf, rfl⟩ := realize_eq_some.mp hx
  obtain ⟨hyf, rfl⟩ := realize_eq_some.mp hy
  have hyz := isZero_eq_false_of_value_ne hb
  have hyn := ratio_fst_pos_of_value_ne hb
  have hxd := ratio_snd_pos x
  have hyd' := ratio_snd_pos y
  unfold BinaryFloat.div
  rw [isNaN_eq_false_of_isFinite hxf, isNaN_eq_false_of_isFinite hyf,
    isInfinite_eq_false_of_isFinite hxf, isInfinite_eq_false_of_isFinite hyf, hyz]
  simp only [Bool.or_false, Bool.and_false, Bool.false_eq_true, if_false]
  unfold value at h ⊢
  generalize x.ratio = rx at hxd h ⊢
  generalize y.ratio = ry at hyn hyd' h ⊢
  obtain ⟨xn, xd⟩ := rx
  obtain ⟨yn, yd⟩ := ry
  dsimp only at hxd hyn hyd' h ⊢
  have hyd : (0 : ℚ) < yd := by exact_mod_cast hyd'
  have hax : ∀ (s : Bool) (n d : ℕ),
      |(if s = true then -((n : ℚ) / d) else (n : ℚ) / d)| = (n : ℚ) / d := by
    intro s n d; split <;> simp [abs_div]
  have key : Representable f (((xn * yd : ℕ) : ℚ) / ((xd * yn : ℕ) : ℚ)) := by
    rw [← representable_abs_iff, abs_div, hax, hax, div_div_div_eq] at h
    push_cast; exact h
  rw [round_exact f _ _ _ (Nat.mul_pos hxd hyn) key]
  congr 1
  have h1 : (xd : ℚ) ≠ 0 := by exact_mod_cast hxd.ne'
  have h2 : (yn : ℚ) ≠ 0 := by exact_mod_cast hyn.ne'
  have h3 : (yd : ℚ) ≠ 0 := hyd.ne'
  cases x.sign <;> cases y.sign <;> push_cast <;> simp <;> field_simp

/-- Every finite encoding is exactly representable in its own format. -/
theorem representable_value {x : BinaryFloat f} (hx : x.isFinite = true) :
    Representable f (value x) := by
  have ht := fraction_lt x
  have hfin : x.exponent ≠ f.maxExponent := by simpa [BinaryFloat.isFinite] using hx
  have hmag : x.magnitude < 2 ^ (f.width - 1) := Nat.mod_lt _ (Nat.two_pow_pos _)
  have hmax := maxExponent_eq f
  have hexp : x.exponent < f.maxExponent := by
    have : x.exponent < 2 ^ f.exponentBits := by
      unfold BinaryFloat.exponent
      rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _), ← pow_add, ← width_sub_one]
      exact hmag
    omega
  have hmax2 : 2 ≤ f.maxExponent := by cases f <;> decide
  refine ⟨significand x, unitExp x, ?_, ?_, ?_, ?_⟩
  · unfold significand; split
    · exact lt_of_lt_of_le ht (Nat.pow_le_pow_right (by norm_num) (by omega))
    · rw [pow_succ]; omega
  · unfold minUnit unitExp; omega
  · unfold maxUnit unitExp; omega
  · rw [abs_value]; rfl

/-- A finite encoding cast to a format at least as wide keeps its exact value. -/
theorem realize_cast {g : BinaryFormat} {x : BinaryFloat f} (hx : x.isFinite = true)
    (hwide : ∀ q, Representable f q → Representable g q) :
    realize (BinaryFloat.cast g x) = some (value x) := by
  unfold BinaryFloat.cast
  rw [isNaN_eq_false_of_isFinite hx, isInfinite_eq_false_of_isFinite hx]
  simp only [Bool.false_eq_true, if_false]
  have hrep := hwide _ (representable_value hx)
  rw [← representable_abs_iff] at hrep
  have habs : |value x| = (x.ratio.1 : ℚ) / x.ratio.2 := by
    unfold value; split
    · rw [abs_neg, abs_of_nonneg (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))]
    · exact abs_of_nonneg (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
  rw [habs] at hrep
  rw [round_exact g _ _ _ (ratio_snd_pos x) hrep]
  rfl

/-- binary32 embeds in binary64. -/
theorem binary32_to_binary64 (q : ℚ) (h : Representable .binary32 q) :
    Representable .binary64 q := by
  obtain ⟨m, u, hm, h1, h2, hq⟩ := h
  refine ⟨m, u, lt_of_lt_of_le hm (by decide), le_trans (by decide) h1,
    le_trans h2 (by decide), hq⟩

theorem neg_toNat (y : BinaryFloat f) :
    (BinaryFloat.neg y).bits.toNat = (if y.sign then 0 else 2 ^ (f.width - 1)) + y.magnitude := by
  unfold BinaryFloat.neg BinaryFloat.ofNat
  rw [BitVec.toNat_ofNat]
  have hm : y.magnitude < 2 ^ (f.width - 1) := Nat.mod_lt _ (Nat.two_pow_pos _)
  have hw := two_pow_width f
  apply Nat.mod_eq_of_lt
  split <;> omega

theorem neg_magnitude (y : BinaryFloat f) : (BinaryFloat.neg y).magnitude = y.magnitude := by
  have hm : y.magnitude < 2 ^ (f.width - 1) := Nat.mod_lt _ (Nat.two_pow_pos _)
  unfold BinaryFloat.magnitude
  rw [neg_toNat]
  split
  · rw [Nat.zero_add, Nat.mod_eq_of_lt (by unfold BinaryFloat.magnitude at hm; exact hm)]; rfl
  · rw [Nat.add_mod_left, Nat.mod_eq_of_lt (by unfold BinaryFloat.magnitude at hm; exact hm)]; rfl

theorem neg_sign (y : BinaryFloat f) : (BinaryFloat.neg y).sign = !y.sign := by
  have hm : y.magnitude < 2 ^ (f.width - 1) := Nat.mod_lt _ (Nat.two_pow_pos _)
  show decide ((BinaryFloat.neg y).bits.toNat ≥ 2 ^ (f.width - 1)) = !y.sign
  rw [neg_toNat]
  cases y.sign
  · simp
  · simp only [if_true, Nat.zero_add, Bool.not_true, decide_eq_false_iff_not, ge_iff_le, not_le]
    exact hm

theorem value_neg (y : BinaryFloat f) : value (BinaryFloat.neg y) = -value y := by
  rw [value_eq_uval, value_eq_uval, neg_sign, uval_eq_of_magnitude (neg_magnitude y)]
  cases y.sign <;> simp

theorem isFinite_neg (y : BinaryFloat f) : (BinaryFloat.neg y).isFinite = y.isFinite := by
  unfold BinaryFloat.isFinite BinaryFloat.exponent; rw [neg_magnitude]

theorem realize_neg {y : BinaryFloat f} {b : ℚ} (hy : realize y = some b) :
    realize (BinaryFloat.neg y) = some (-b) := by
  obtain ⟨hyf, rfl⟩ := realize_eq_some.mp hy
  exact realize_eq_some.mpr ⟨(isFinite_neg y).trans hyf, value_neg y⟩

/-- One exact IEEE subtraction (`add x (neg y)`) is real subtraction. -/
theorem realize_sub {x y : BinaryFloat f} {a b : ℚ} (hx : realize x = some a)
    (hy : realize y = some b) (h : Representable f (a - b)) :
    realize (BinaryFloat.sub x y) = some (a - b) := by
  unfold BinaryFloat.sub
  rw [sub_eq_add_neg] at h ⊢
  exact realize_add hx (realize_neg hy) h

/-! ### Finiteness of rounding below one and the dropout thresholds -/

theorem roundQuotient_le (a b : ℕ) : BinaryFloat.roundQuotient a b ≤ a / b + 1 := by
  unfold BinaryFloat.roundQuotient
  dsimp only
  split <;> omega

theorem format_bias_bounds (g : BinaryFormat) : 2 ≤ g.bias ∧ g.bias < g.maxExponent := by
  cases g <;> decide

/-- Rounding a nonnegative quotient `n / d < 1` never overflows: the result is finite
(in every format, whatever the requested sign). -/
theorem round_isFinite_of_lt (g : BinaryFormat) (neg : Bool) (n d : ℕ) (hd : 0 < d)
    (hlt : n < d) : (BinaryFloat.round g neg n d).isFinite = true := by
  have hmax := maxExponent_pos g
  obtain ⟨hb2, hbm⟩ := format_bias_bounds g
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · have : BinaryFloat.round g neg 0 d = BinaryFloat.zero g neg := by
      simp [BinaryFloat.round, hd.ne']
    rw [this]
    unfold BinaryFloat.zero
    have := (pack_fields g neg 0 0 hmax (Nat.two_pow_pos _)).2.1
    simp only [BinaryFloat.isFinite, this, bne_iff_ne, ne_eq]
    omega
  have hdq : (0 : ℚ) < d := by exact_mod_cast hd
  obtain ⟨hL1, hL2⟩ := ratioExponent_spec hn hd
  obtain ⟨L, hL⟩ : ∃ L, L = BinaryFloat.ratioExponent n d := ⟨_, rfl⟩
  rw [← hL] at hL1 hL2
  have hLneg : L < 0 := by
    by_contra hc
    have h1 : (1 : ℚ) ≤ 2 ^ L := one_le_zpow₀ (by norm_num) (by omega)
    have h2 : (n : ℚ) / d < 1 := (div_lt_one hdq).mpr (by exact_mod_cast hlt)
    linarith
  obtain ⟨unit, hunit⟩ : ∃ unit : ℤ,
      unit = max (1 - (g.bias : ℤ) - (g.fractionBits : ℤ)) (L - (g.fractionBits : ℤ)) :=
    ⟨_, rfl⟩
  have hunit_min : 1 - (g.bias : ℤ) - g.fractionBits ≤ unit := by
    rw [hunit]; exact le_max_left _ _
  have hunit_L : L - g.fractionBits ≤ unit := by rw [hunit]; exact le_max_right _ _
  have hunit_max : unit + g.fractionBits + g.bias ≤ g.bias := by
    rw [hunit]; omega
  -- `n / d < 2 ^ (p + 1 + unit)`: the rounded significand has at most `p + 1` bits.
  have hbound : (n : ℚ) / d < 2 ^ ((g.fractionBits : ℤ) + 1 + unit) :=
    lt_of_lt_of_le hL2 (zpow_le_zpow_right₀ (by norm_num) (by omega))
  obtain ⟨q0, hq0⟩ : ∃ q0, q0 = (if unit ≥ 0 then BinaryFloat.roundQuotient n (d * 2 ^ unit.toNat)
      else BinaryFloat.roundQuotient (n * 2 ^ (-unit).toNat) d) := ⟨_, rfl⟩
  have hq0le : q0 ≤ 2 ^ (g.fractionBits + 1) := by
    rw [hq0]
    split
    · rename_i hu0
      have hlt' : n < d * 2 ^ unit.toNat * 2 ^ (g.fractionBits + 1) := by
        have : (n : ℚ) < d * 2 ^ unit.toNat * 2 ^ (g.fractionBits + 1) := by
          rw [div_lt_iff₀ hdq] at hbound
          rw [two_pow_nat_eq_zpow, two_pow_nat_eq_zpow, Int.toNat_of_nonneg hu0, mul_assoc,
            ← zpow_add₀ two_ne_zero]
          push_cast
          calc (n : ℚ) < 2 ^ ((g.fractionBits : ℤ) + 1 + unit) * d := hbound
            _ = d * 2 ^ (unit + ((g.fractionBits : ℤ) + 1)) := by rw [mul_comm]; ring_nf
        exact_mod_cast this
      have hdiv : n / (d * 2 ^ unit.toNat) < 2 ^ (g.fractionBits + 1) :=
        (Nat.div_lt_iff_lt_mul (by positivity)).mpr
          (by rw [Nat.mul_comm (2 ^ _)]; linarith [hlt'])
      have := roundQuotient_le n (d * 2 ^ unit.toNat)
      omega
    · rename_i hu0
      have hlt' : n * 2 ^ (-unit).toNat < d * 2 ^ (g.fractionBits + 1) := by
        have : (n : ℚ) * 2 ^ (-unit).toNat < d * 2 ^ (g.fractionBits + 1) := by
          rw [div_lt_iff₀ hdq] at hbound
          rw [two_pow_nat_eq_zpow, two_pow_nat_eq_zpow, Int.toNat_of_nonneg (by omega)]
          have hp : (0 : ℚ) < 2 ^ (-unit) := two_zpow_pos _
          calc (n : ℚ) * 2 ^ (-unit) < 2 ^ ((g.fractionBits : ℤ) + 1 + unit) * d * 2 ^ (-unit) :=
                mul_lt_mul_of_pos_right hbound hp
            _ = d * 2 ^ (((g.fractionBits + 1 : ℕ) : ℤ)) := by
                rw [mul_comm _ (d : ℚ), mul_assoc, ← zpow_add₀ two_ne_zero]; push_cast; ring_nf
        exact_mod_cast this
      have hdiv : n * 2 ^ (-unit).toNat / d < 2 ^ (g.fractionBits + 1) :=
        (Nat.div_lt_iff_lt_mul hd).mpr (by rw [Nat.mul_comm (2 ^ _)]; exact hlt')
      have := roundQuotient_le (n * 2 ^ (-unit).toNat) d
      omega
  have hround : BinaryFloat.round g neg n d =
      (let carry := q0 ≥ 2 ^ (g.fractionBits + 1)
       let q := if carry then q0 / 2 else q0
       let unit := if carry then unit + 1 else unit
       let e := if q < 2 ^ g.fractionBits then 0 else (unit + g.fractionBits + g.bias).toNat
       if e ≥ g.maxExponent then BinaryFloat.infinity g neg
       else BinaryFloat.pack g neg e (if e == 0 then q else q - 2 ^ g.fractionBits)) := by
    unfold BinaryFloat.round
    simp only [beq_iff_eq, hd.ne', hn.ne', if_false, ← hL, ← hunit, ← hq0]
  rw [hround]
  dsimp only
  have hp := Nat.two_pow_pos g.fractionBits
  have hpow : 2 ^ (g.fractionBits + 1) = 2 * 2 ^ g.fractionBits := by rw [pow_succ]; ring
  by_cases hc : q0 ≥ 2 ^ (g.fractionBits + 1)
  · have hq0eq : q0 = 2 ^ (g.fractionBits + 1) := le_antisymm hq0le hc
    have hq2 : q0 / 2 = 2 ^ g.fractionBits := by rw [hq0eq, hpow]; omega
    simp only [hc, if_true, hq2, lt_irrefl, if_false]
    have he : (unit + 1 + g.fractionBits + g.bias).toNat < g.maxExponent := by omega
    have hne : ((unit + 1 + g.fractionBits + g.bias).toNat == 0) = false := by
      simp only [beq_eq_false_iff_ne]; omega
    rw [if_neg (by omega), hne]
    simp only [Bool.false_eq_true, if_false, Nat.sub_self]
    have := (pack_fields g neg _ 0 he hp).2.1
    simp only [BinaryFloat.isFinite, this, bne_iff_ne, ne_eq]
    omega
  · simp only [hc, if_false]
    by_cases hq : q0 < 2 ^ g.fractionBits
    · simp only [hq, if_true]
      have : ¬ (0 ≥ g.maxExponent) := by omega
      simp only [this, if_false, beq_self_eq_true, if_true]
      have := (pack_fields g neg 0 q0 hmax hq).2.1
      simp only [BinaryFloat.isFinite, this, bne_iff_ne, ne_eq]
      omega
    · simp only [hq, if_false]
      have he : (unit + g.fractionBits + g.bias).toNat < g.maxExponent := by omega
      have hne : ((unit + g.fractionBits + g.bias).toNat == 0) = false := by
        simp only [beq_eq_false_iff_ne]; omega
      rw [if_neg (by omega), hne]
      simp only [Bool.false_eq_true, if_false]
      have := (pack_fields g neg _ (q0 - 2 ^ g.fractionBits) he (by omega)).2.1
      simp only [BinaryFloat.isFinite, this, bne_iff_ne, ne_eq]
      omega

/-- The unsigned ratio of an encoding is the absolute value of its finite interpretation. -/
theorem ratio_div_eq_abs (x : BinaryFloat f) : (x.ratio.1 : ℚ) / x.ratio.2 = |value x| := by
  unfold value
  have h0 : (0 : ℚ) ≤ (x.ratio.1 : ℚ) / x.ratio.2 :=
    div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
  split
  · rw [abs_neg, abs_of_nonneg h0]
  · rw [abs_of_nonneg h0]

/-- **Finiteness of narrowing below one.** Casting a finite value of magnitude below `1`
to any format yields a finite encoding (the rounding cannot overflow). -/
theorem cast_isFinite_of_abs_lt_one (g : BinaryFormat) {x : BinaryFloat f} {a : ℚ}
    (hx : realize x = some a) (h1 : |a| < 1) : (BinaryFloat.cast g x).isFinite = true := by
  obtain ⟨hxf, rfl⟩ := realize_eq_some.mp hx
  unfold BinaryFloat.cast
  rw [isNaN_eq_false_of_isFinite hxf, isInfinite_eq_false_of_isFinite hxf]
  simp only [Bool.false_eq_true, if_false]
  have hd := ratio_snd_pos x
  have hlt : x.ratio.1 < x.ratio.2 := by
    have : (x.ratio.1 : ℚ) / x.ratio.2 < 1 := by rw [ratio_div_eq_abs]; exact h1
    rw [div_lt_one (by exact_mod_cast hd)] at this
    exact_mod_cast this
  exact round_isFinite_of_lt g _ _ _ hd hlt

/-- The Rust `u as f32` of a binary64 unit `u ∈ (-1, 1)` is finite. -/
theorem toF32_isFinite {u : F64} {a : ℚ} (hu : realize u = some a) (h1 : |a| < 1) :
    (F64.toF32 u).isFinite = true :=
  cast_isFinite_of_abs_lt_one .binary32 hu h1

/-- **Non-f64 dropout threshold.** Rust compares `f64::from(u as f32)` with the wide rate
`w`; for a unit of magnitude below `1` and a finite `w` of exact value `r` this is the exact
comparison of the *f32-rounded* unit with `r` (not of the unit itself). -/
theorem nonf64_lt_eq {u w : F64} {a r : ℚ} (hu : realize u = some a) (h1 : |a| < 1)
    (hw : realize w = some r) :
    F64.lt (F32.toF64 (F64.toF32 u)) w = decide (value (F64.toF32 u) < r) := by
  have hfin := toF32_isFinite hu h1
  have hc : realize (F32.toF64 (F64.toF32 u)) = some (value (F64.toF32 u)) :=
    realize_cast hfin binary32_to_binary64
  obtain ⟨hcf, hcv⟩ := realize_eq_some.mp hc
  obtain ⟨hwf, rfl⟩ := realize_eq_some.mp hw
  unfold F64.lt
  rw [lt_eq_decide hcf hwf, hcv]

/-- f64 dropout threshold: the exact comparison of the unit itself with the rate. -/
theorem f64_lt_eq {u w : F64} {a r : ℚ} (hu : realize u = some a) (hw : realize w = some r) :
    F64.lt u w = decide (a < r) := by
  obtain ⟨huf, rfl⟩ := realize_eq_some.mp hu
  obtain ⟨hwf, rfl⟩ := realize_eq_some.mp hw
  exact lt_eq_decide huf hwf

end ChelisBridge.Protocol
