import ChelisBridge.Protocol.FloatOrder
import Aeneas.Std.Half
import Mathlib.Tactic.IntervalCases

/-!
# Exactness of `f64::from(bf16)` in the `half` crate model

`f64::from(bf16)` is modelled bit for bit by `Half.bf16ToF64` (the crate's software algorithm).
Every finite bf16 encoding widens to an `f64` encoding with the same exact value. The proof is
an exhaustive kernel evaluation over all `2^16` encodings, split into 64 blocks of 1024.
-/

namespace ChelisBridge.Protocol

open Aeneas.Std ChelisBridge

/-- The widening of the bf16 encoding `n` is exact (or `n` is not finite). -/
def bf16WidenOk (n : Nat) : Bool :=
  let x : BF16 := BinaryFloat.ofNat _ n
  (not x.isFinite) || realize (BinaryFloat.ofNat .binary64 (Half.bf16ToF64 n) : F64) == some (value x)

theorem bf16_widen_block_0 : ∀ m < 1024, bf16WidenOk (0 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_1 : ∀ m < 1024, bf16WidenOk (1 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_2 : ∀ m < 1024, bf16WidenOk (2 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_3 : ∀ m < 1024, bf16WidenOk (3 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_4 : ∀ m < 1024, bf16WidenOk (4 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_5 : ∀ m < 1024, bf16WidenOk (5 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_6 : ∀ m < 1024, bf16WidenOk (6 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_7 : ∀ m < 1024, bf16WidenOk (7 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_8 : ∀ m < 1024, bf16WidenOk (8 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_9 : ∀ m < 1024, bf16WidenOk (9 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_10 : ∀ m < 1024, bf16WidenOk (10 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_11 : ∀ m < 1024, bf16WidenOk (11 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_12 : ∀ m < 1024, bf16WidenOk (12 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_13 : ∀ m < 1024, bf16WidenOk (13 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_14 : ∀ m < 1024, bf16WidenOk (14 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_15 : ∀ m < 1024, bf16WidenOk (15 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_16 : ∀ m < 1024, bf16WidenOk (16 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_17 : ∀ m < 1024, bf16WidenOk (17 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_18 : ∀ m < 1024, bf16WidenOk (18 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_19 : ∀ m < 1024, bf16WidenOk (19 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_20 : ∀ m < 1024, bf16WidenOk (20 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_21 : ∀ m < 1024, bf16WidenOk (21 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_22 : ∀ m < 1024, bf16WidenOk (22 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_23 : ∀ m < 1024, bf16WidenOk (23 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_24 : ∀ m < 1024, bf16WidenOk (24 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_25 : ∀ m < 1024, bf16WidenOk (25 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_26 : ∀ m < 1024, bf16WidenOk (26 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_27 : ∀ m < 1024, bf16WidenOk (27 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_28 : ∀ m < 1024, bf16WidenOk (28 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_29 : ∀ m < 1024, bf16WidenOk (29 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_30 : ∀ m < 1024, bf16WidenOk (30 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_31 : ∀ m < 1024, bf16WidenOk (31 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_32 : ∀ m < 1024, bf16WidenOk (32 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_33 : ∀ m < 1024, bf16WidenOk (33 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_34 : ∀ m < 1024, bf16WidenOk (34 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_35 : ∀ m < 1024, bf16WidenOk (35 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_36 : ∀ m < 1024, bf16WidenOk (36 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_37 : ∀ m < 1024, bf16WidenOk (37 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_38 : ∀ m < 1024, bf16WidenOk (38 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_39 : ∀ m < 1024, bf16WidenOk (39 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_40 : ∀ m < 1024, bf16WidenOk (40 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_41 : ∀ m < 1024, bf16WidenOk (41 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_42 : ∀ m < 1024, bf16WidenOk (42 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_43 : ∀ m < 1024, bf16WidenOk (43 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_44 : ∀ m < 1024, bf16WidenOk (44 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_45 : ∀ m < 1024, bf16WidenOk (45 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_46 : ∀ m < 1024, bf16WidenOk (46 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_47 : ∀ m < 1024, bf16WidenOk (47 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_48 : ∀ m < 1024, bf16WidenOk (48 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_49 : ∀ m < 1024, bf16WidenOk (49 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_50 : ∀ m < 1024, bf16WidenOk (50 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_51 : ∀ m < 1024, bf16WidenOk (51 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_52 : ∀ m < 1024, bf16WidenOk (52 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_53 : ∀ m < 1024, bf16WidenOk (53 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_54 : ∀ m < 1024, bf16WidenOk (54 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_55 : ∀ m < 1024, bf16WidenOk (55 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_56 : ∀ m < 1024, bf16WidenOk (56 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_57 : ∀ m < 1024, bf16WidenOk (57 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_58 : ∀ m < 1024, bf16WidenOk (58 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_59 : ∀ m < 1024, bf16WidenOk (59 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_60 : ∀ m < 1024, bf16WidenOk (60 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_61 : ∀ m < 1024, bf16WidenOk (61 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_62 : ∀ m < 1024, bf16WidenOk (62 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_block_63 : ∀ m < 1024, bf16WidenOk (63 * 1024 + m) = true := by
  decide +kernel

theorem bf16_widen_all (n : Nat) (hn : n < 65536) : bf16WidenOk n = true := by
  have hc : n / 1024 < 64 := by omega
  have hm : n % 1024 < 1024 := Nat.mod_lt _ (by decide)
  have hn' : n = n / 1024 * 1024 + n % 1024 := by omega
  rw [hn']
  generalize n / 1024 = c at hc
  generalize n % 1024 = m at hm
  interval_cases c
  · exact bf16_widen_block_0 m hm
  · exact bf16_widen_block_1 m hm
  · exact bf16_widen_block_2 m hm
  · exact bf16_widen_block_3 m hm
  · exact bf16_widen_block_4 m hm
  · exact bf16_widen_block_5 m hm
  · exact bf16_widen_block_6 m hm
  · exact bf16_widen_block_7 m hm
  · exact bf16_widen_block_8 m hm
  · exact bf16_widen_block_9 m hm
  · exact bf16_widen_block_10 m hm
  · exact bf16_widen_block_11 m hm
  · exact bf16_widen_block_12 m hm
  · exact bf16_widen_block_13 m hm
  · exact bf16_widen_block_14 m hm
  · exact bf16_widen_block_15 m hm
  · exact bf16_widen_block_16 m hm
  · exact bf16_widen_block_17 m hm
  · exact bf16_widen_block_18 m hm
  · exact bf16_widen_block_19 m hm
  · exact bf16_widen_block_20 m hm
  · exact bf16_widen_block_21 m hm
  · exact bf16_widen_block_22 m hm
  · exact bf16_widen_block_23 m hm
  · exact bf16_widen_block_24 m hm
  · exact bf16_widen_block_25 m hm
  · exact bf16_widen_block_26 m hm
  · exact bf16_widen_block_27 m hm
  · exact bf16_widen_block_28 m hm
  · exact bf16_widen_block_29 m hm
  · exact bf16_widen_block_30 m hm
  · exact bf16_widen_block_31 m hm
  · exact bf16_widen_block_32 m hm
  · exact bf16_widen_block_33 m hm
  · exact bf16_widen_block_34 m hm
  · exact bf16_widen_block_35 m hm
  · exact bf16_widen_block_36 m hm
  · exact bf16_widen_block_37 m hm
  · exact bf16_widen_block_38 m hm
  · exact bf16_widen_block_39 m hm
  · exact bf16_widen_block_40 m hm
  · exact bf16_widen_block_41 m hm
  · exact bf16_widen_block_42 m hm
  · exact bf16_widen_block_43 m hm
  · exact bf16_widen_block_44 m hm
  · exact bf16_widen_block_45 m hm
  · exact bf16_widen_block_46 m hm
  · exact bf16_widen_block_47 m hm
  · exact bf16_widen_block_48 m hm
  · exact bf16_widen_block_49 m hm
  · exact bf16_widen_block_50 m hm
  · exact bf16_widen_block_51 m hm
  · exact bf16_widen_block_52 m hm
  · exact bf16_widen_block_53 m hm
  · exact bf16_widen_block_54 m hm
  · exact bf16_widen_block_55 m hm
  · exact bf16_widen_block_56 m hm
  · exact bf16_widen_block_57 m hm
  · exact bf16_widen_block_58 m hm
  · exact bf16_widen_block_59 m hm
  · exact bf16_widen_block_60 m hm
  · exact bf16_widen_block_61 m hm
  · exact bf16_widen_block_62 m hm
  · exact bf16_widen_block_63 m hm

/-- `f64::from(bf16)` is exact on finite encodings. -/
theorem realize_bf16_to_f64 {w : BF16} (hw : w.isFinite = true) :
    realize (BinaryFloat.ofNat .binary64 (Half.bf16ToF64 w.bits.toNat) : F64) = some (value w) := by
  have h := bf16_widen_all w.bits.toNat w.bits.isLt
  simp only [bf16WidenOk, ofNat_bits, hw, Bool.not_true, Bool.false_or, beq_iff_eq] at h
  exact h

end ChelisBridge.Protocol
