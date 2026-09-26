import ChelisBridge.Model
import ChelisBridge.Protocol.FloatOrder
import ChelisBridge.Protocol.Iter
import LaCaDiLECore.Protocol

open Aeneas.Std LaCaDiLECore ChelisBridge ChelisBridge.Protocol
open_chelis_model

namespace ChelisBridge.Random

/-- The Aeneas `u64` carrying the same bits as a Lean `UInt64`. -/
def ofUInt64 (x : UInt64) : U64 := ⟨x.toBitVec⟩

theorem splitmix_eq (v : U64) :
    chelis_types.dtype_semantics.dropout_splitmix64 v =
      .ok (ofUInt64 (DrawKey.splitmix (UInt64.ofBitVec v.bv))) := by
  rfl

/-- Rust `((word >> 11) as f64) / ((1u64 << 53) as f64)` in the Aeneas float model. -/
def unitOfWord (w : UInt64) : F64 :=
  F64.div (F64.ofUScalar (ofUInt64 (w >>> 11))) (F64.ofUScalar (ofUInt64 ((1 : UInt64) <<< 53)))

theorem ofNat_val (v : U64) : UInt64.ofNat v.val = UInt64.ofBitVec v.bv := by
  apply UInt64.toBitVec_inj.mp
  apply BitVec.eq_of_toNat_eq
  show (BitVec.ofNat 64 v.bv.toNat).toNat = v.bv.toNat
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt v.bv.isLt]

theorem shr11 (w : UInt64) :
    (ofUInt64 w >>> 11#i32 : Result U64) = .ok (ofUInt64 (w >>> 11)) := by
  simp only [HShiftRight.hShiftRight, UScalar.shiftRight_IScalar, UScalar.shiftRight]
  simp
  rfl

theorem shl53 : (1#u64 <<< 53#i32 : Result U64) = .ok (ofUInt64 ((1 : UInt64) <<< 53)) := by
  simp only [HShiftLeft.hShiftLeft, UScalar.shiftLeft_IScalar, UScalar.shiftLeft]
  simp
  rfl

/-- **Word exactness.** The extracted `dropout_random_unit` computes the reference integer
word `DrawKey.word` (splitmix64 of seed, rotated ordinal hash and rotated index hash) and
turns it into a unit by the Rust expression `unitOfWord`. -/
theorem word_eq_reference (seed ordinal index : U64) :
    chelis_types.dtype_semantics.dropout_random_unit seed ordinal index =
      .ok (unitOfWord (DrawKey.word ⟨UInt64.ofBitVec seed.bv, ordinal.val⟩ index.val)) := by
  unfold chelis_types.dtype_semantics.dropout_random_unit
  simp only [splitmix_eq, bind_tc_ok, lift]
  have hw : ofUInt64 (DrawKey.splitmix (UInt64.ofBitVec
      ((seed ^^^ core.num.U64.rotate_left
          (ofUInt64 (DrawKey.splitmix (UInt64.ofBitVec ordinal.bv))) 17#u32) ^^^
        core.num.U64.rotate_left
          (ofUInt64 (DrawKey.splitmix (UInt64.ofBitVec index.bv))) 41#u32).bv)) =
      ofUInt64 (DrawKey.word ⟨UInt64.ofBitVec seed.bv, ordinal.val⟩ index.val) := by
    unfold DrawKey.word
    rw [ofNat_val, ofNat_val]
    congr 2
  rw [hw, shr11, shl53]
  rfl

theorem ofUInt64_val (w : UInt64) : (ofUInt64 w).val = w.toNat := rfl

/-- An exactly representable `u64` converts to binary64 without rounding. -/
theorem realize_ofUScalar (x : U64) (n : ℕ) (hn : x.val = n)
    (h : Representable .binary64 (n : ℚ)) :
    realize (F64.ofUScalar x) = some (n : ℚ) := by
  unfold F64.ofUScalar
  rw [hn, round_exact .binary64 false n 1 Nat.one_pos (by simpa using h)]
  simp

theorem realize_unitOfWord (w : UInt64) :
    realize (unitOfWord w) = some (((w.toNat / 2048 : ℕ) : ℚ) / 9007199254740992) := by
  have h1 : (ofUInt64 (w >>> 11)).val = w.toNat / 2048 := by
    rw [ofUInt64_val, UInt64.toNat_shiftRight, Nat.shiftRight_eq_div_pow]; rfl
  have h2 : (ofUInt64 ((1 : UInt64) <<< 53)).val = 9007199254740992 := by decide
  have hw : w.toNat / 2048 < 2 ^ 53 :=
    (Nat.div_lt_iff_lt_mul (by decide)).mpr (lt_of_lt_of_eq w.toNat_lt_size (by decide))
  unfold unitOfWord F64.div
  refine realize_div (realize_ofUScalar _ _ h1 ⟨_, 0, hw, by decide, by decide, by simp⟩)
    (realize_ofUScalar _ _ h2 ⟨1, 53, by decide, by decide, by decide, by norm_num⟩)
    (by norm_num) ?_
  refine ⟨w.toNat / 2048, -53, hw, by decide, by decide, ?_⟩
  rw [abs_of_nonneg (by positivity)]
  norm_num [zpow_neg]
  ring

/-- The unit of every word is exactly the reference `unitValue`. -/
theorem realize_unit (k : DrawKey) (i : ℕ) :
    realize (unitOfWord (k.word i)) = some (k.unitValue i) := by
  rw [realize_unitOfWord]; rfl

/-- Seed chosen by inverting splitmix64: at ordinal 0, index 0 its word is
`0x7FFFFFFFFFFFF800`, so the exact unit is `(2^52 - 1) / 2^53 = 1/2 - 2^-53`. -/
def cexSeed : U64 := 3167153300553317085#u64

theorem cex_word :
    DrawKey.word ⟨UInt64.ofBitVec cexSeed.bv, (0#u64 : U64).val⟩ (0#u64 : U64).val =
      0x7FFFFFFFFFFFF800 := by decide +kernel

theorem cex_unit_value :
    DrawKey.unitValue ⟨UInt64.ofBitVec cexSeed.bv, (0#u64 : U64).val⟩ (0#u64 : U64).val =
      (2 ^ 52 - 1) / 2 ^ 53 := by
  unfold DrawKey.unitValue; rw [cex_word]; decide +kernel

/-- The f32 rounding used on every non-f64 path sends this unit to exactly `1/2`. -/
theorem cex_f32_round :
    realize (F32.toF64 (F64.toF32 (unitOfWord 0x7FFFFFFFFFFFF800))) = some (1 / 2) := by
  decide +kernel

/-- **Unit exactness (f64 path).** The extracted unit is finite and its exact rational value
is the reference `DrawKey.unitValue`: the `u64 >> 11` conversion and the division by `2^53`
are exact binary64 operations for every seed, ordinal and index. -/
theorem unit_exact_f64 (seed ordinal index : U64) :
    ∃ u, chelis_types.dtype_semantics.dropout_random_unit seed ordinal index = .ok u ∧
      u.isFinite = true ∧
      realize u = some (DrawKey.unitValue ⟨UInt64.ofBitVec seed.bv, ordinal.val⟩ index.val) := by
  refine ⟨_, word_eq_reference seed ordinal index, ?_, ?_⟩
  · exact (realize_eq_some.mp (realize_unitOfWord _)).1
  · rw [realize_unitOfWord]; rfl

abbrev Frame := chelis_ir.evaluation.ExecutionFrame

/-- The current counter of scope `s`: the inherited context stream for the root scope,
the frame-local counter otherwise. -/
def counterAt (F : Frame) (s : chelis_ir.evaluation.ScopeId) : Option U64 :=
  if s = 0#usize then some F.context.state.counter else F.counters.val[s.val]?

/-- Lookup of a node's random site through the extracted `UnordMap::get`. -/
abbrev siteOf (F : Frame) (node : chelis_ir.dag.NodeId) :=
  chelis_unord.UnordMap.get chelis_ir.dag.NodeId.Insts.CoreCmpOrd
    (core.borrow.BorrowSelf chelis_ir.dag.NodeId) chelis_ir.dag.NodeId.Insts.CoreCmpOrd
    F.metadata.sites node

/-- The frame after a forward draw at scope `s` with counter `c`. -/
def drawnFrame (F : Frame) (draw : chelis_ir.evaluation.DrawId) (s : chelis_ir.evaluation.ScopeId)
    (seed c : U64) : Frame :=
  { F with
    counters := if s = 0#usize then F.counters
      else alloc.vec.Vec.set F.counters s (core.num.U64.wrapping_add c 1#u64),
    keys := alloc.vec.Vec.set F.keys draw (some ⟨seed, c⟩),
    context := if s = 0#usize then
        { state := { F.context.state with counter := core.num.U64.wrapping_add c 1#u64 } }
      else F.context }

theorem enter_forward_eq (F : Frame) (node : chelis_ir.dag.NodeId) (seed : U64)
    (draw : chelis_ir.evaluation.DrawId) (s : chelis_ir.evaluation.ScopeId) (c : U64)
    (hsite : siteOf F node = .ok (some (.Forward draw s)))
    (hlast : F.scopes.val.getLast? = some s)
    (hseed : F.metadata.scopes.val[s.val]? = some ⟨some seed⟩)
    (hc : counterAt F s = some c)
    (hkey : F.keys.val[draw.val]? = some none) :
    ∃ back, chelis_ir.evaluation.ExecutionFrame.enter F node seed =
      .ok (.Ok ⟨seed, c⟩, drawnFrame F draw s seed c, back) := by
  unfold chelis_ir.evaluation.ExecutionFrame.enter
  simp only [siteOf] at hsite
  simp only [hsite, bind_tc_ok, core.option.Option.ok_or, core.result.TryResult.branch,
    core.slice.Slice.last, alloc.vec.Vec.deref, hlast,
    core.option.Option.Insts.CoreCmpPartialEqOption.eq,
    core.cmp.impls.PartialEqShared.eq, chelis_ir.evaluation.ScopeId.Insts.CoreCmpPartialEqScopeId.eq,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize,
    core.option.Option.replace, core.option.Option.is_some, lift]
  simp [hseed, hkey]
  by_cases hs : s = 0#usize
  · subst hs
    have hc' : F.context.state.counter = c := by simpa [counterAt] using hc
    subst hc'
    exact ⟨_, rfl⟩
  · have hc' : F.counters.val[s.val]? = some c := by simpa [counterAt, hs] using hc
    simp only [drawnFrame, hs, if_false, hc', bind_tc_ok]
    exact ⟨_, rfl⟩

theorem counterAt_drawn_self (F : Frame) (draw : chelis_ir.evaluation.DrawId)
    (s : chelis_ir.evaluation.ScopeId) (seed c : U64) (hc : counterAt F s = some c) :
    counterAt (drawnFrame F draw s seed c) s = some (core.num.U64.wrapping_add c 1#u64) := by
  by_cases hs : s = 0#usize
  · subst hs; simp [counterAt, drawnFrame]
  · have hc' : F.counters.val[s.val]? = some c := by simpa [counterAt, hs] using hc
    have hlt : s.val < F.counters.val.length := by
      rcases List.getElem?_eq_some_iff.mp hc' with ⟨h, _⟩; exact h
    simp [counterAt, drawnFrame, hs, alloc.vec.Vec.set_val_eq, hlt]

theorem counterAt_drawn_other (F : Frame) (draw : chelis_ir.evaluation.DrawId)
    (s t : chelis_ir.evaluation.ScopeId) (seed c : U64) (ht : t ≠ s) :
    counterAt (drawnFrame F draw s seed c) t = counterAt F t := by
  have hv : s.val ≠ t.val := fun e => ht (UScalar.eq_of_val_eq e).symm
  by_cases hs : s = 0#usize <;> by_cases htz : t = 0#usize <;>
    simp_all [counterAt, drawnFrame, alloc.vec.Vec.set_val_eq]

/-- Reference key of scope `s`: its handler seed and the `Nat` image of its counter. -/
def keyOf (F : Frame) (s : chelis_ir.evaluation.ScopeId) : Option DrawKey :=
  match F.metadata.scopes.val[s.val]?, counterAt F s with
  | some ⟨some seed⟩, some c => some ⟨UInt64.ofBitVec seed.bv, c.val⟩
  | _, _ => none

/-- Reference Random state of a frame: the innermost entered scope is active and the
enclosing scopes are saved innermost first. -/
def absState (F : Frame) : RandomState :=
  match F.scopes.val.reverse with
  | [] => ⟨none, []⟩
  | top :: rest => ⟨keyOf F top, rest.map (keyOf F)⟩

theorem keyOf_drawn_other (F : Frame) (draw : chelis_ir.evaluation.DrawId)
    (s t : chelis_ir.evaluation.ScopeId) (seed c : U64) (ht : t ≠ s) :
    keyOf (drawnFrame F draw s seed c) t = keyOf F t := by
  unfold keyOf
  rw [counterAt_drawn_other F draw s t seed c ht]
  rfl

theorem wrapping_add_one_val (c : U64) (h : c.val + 1 < 2 ^ 64) :
    (core.num.U64.wrapping_add c 1#u64).val = c.val + 1 := by
  rw [core.num.U64.wrapping_add_val_eq]
  have h1 : (1#u64 : U64).val = 1 := rfl
  have h2 : UScalar.size .U64 = 2 ^ 64 := by rw [UScalar.size_def]; rfl
  rw [h1, h2]; exact Nat.mod_eq_of_lt h

theorem getLast?_of_reverse {α : Type} {l : List α} {s : α} {rest : List α}
    (h : l.reverse = s :: rest) : l.getLast? = some s := by
  rw [← List.head?_reverse, h]; rfl

/-- **Counter relation (no overflow).** A forward draw at the innermost scope returns the
current counter as ordinal and advances it by one; without `u64` overflow the reference
`RandomState.draw` of the abstracted state returns the same key and the abstraction of the
successor frame. -/
theorem counter_relation_no_overflow (F : Frame) (node : chelis_ir.dag.NodeId) (seed : U64)
    (draw : chelis_ir.evaluation.DrawId) (s : chelis_ir.evaluation.ScopeId) (c : U64)
    (rest : List chelis_ir.evaluation.ScopeId)
    (hsite : siteOf F node = .ok (some (.Forward draw s)))
    (hstack : F.scopes.val.reverse = s :: rest) (hfresh : s ∉ rest)
    (hseed : F.metadata.scopes.val[s.val]? = some ⟨some seed⟩)
    (hc : counterAt F s = some c)
    (hkey : F.keys.val[draw.val]? = some none)
    (hno : c.val + 1 < 2 ^ 64) :
    ∃ F' back, chelis_ir.evaluation.ExecutionFrame.enter F node seed =
        .ok (.Ok ⟨seed, c⟩, F', back) ∧
      (absState F).draw = some (⟨UInt64.ofBitVec seed.bv, c.val⟩, absState F') := by
  obtain ⟨back, he⟩ :=
    enter_forward_eq F node seed draw s c hsite (getLast?_of_reverse hstack) hseed hc hkey
  refine ⟨_, back, he, ?_⟩
  have hk : keyOf F s = some ⟨UInt64.ofBitVec seed.bv, c.val⟩ := by
    unfold keyOf; rw [hseed, hc]
  have hk' : keyOf (drawnFrame F draw s seed c) s =
      some ⟨UInt64.ofBitVec seed.bv, c.val + 1⟩ := by
    have hm : (drawnFrame F draw s seed c).metadata = F.metadata := rfl
    unfold keyOf
    rw [counterAt_drawn_self F draw s seed c hc, hm, hseed]
    simp only [wrapping_add_one_val c hno]
  have hrest : rest.map (keyOf (drawnFrame F draw s seed c)) = rest.map (keyOf F) := by
    apply List.map_congr_left
    intro t ht
    exact keyOf_drawn_other F draw s t seed c (fun e => hfresh (e ▸ ht))
  have hsc : (drawnFrame F draw s seed c).scopes = F.scopes := rfl
  simp only [absState, hstack, hsc, hk, hk', hrest, RandomState.draw]
  rfl

theorem wrapping_add_one_max (c : U64) (h : c.val = 2 ^ 64 - 1) :
    (core.num.U64.wrapping_add c 1#u64).val = 0 := by
  rw [core.num.U64.wrapping_add_val_eq, UScalar.size_def, h]
  rfl

/-- **Counter wrap counterexample.** At counter `u64::MAX` the extracted entry returns ordinal
`2^64 - 1` and wraps the stored counter to `0`, whereas the reference draw advances the active
ordinal to `2^64`: the abstracted successor frame is not the reference successor state. -/
theorem counter_wrap_counterexample (F : Frame) (node : chelis_ir.dag.NodeId) (seed : U64)
    (draw : chelis_ir.evaluation.DrawId) (s : chelis_ir.evaluation.ScopeId) (c : U64)
    (rest : List chelis_ir.evaluation.ScopeId)
    (hsite : siteOf F node = .ok (some (.Forward draw s)))
    (hstack : F.scopes.val.reverse = s :: rest)
    (hseed : F.metadata.scopes.val[s.val]? = some ⟨some seed⟩)
    (hc : counterAt F s = some c) (hmax : c.val = 2 ^ 64 - 1)
    (hkey : F.keys.val[draw.val]? = some none) :
    ∃ F' back c', chelis_ir.evaluation.ExecutionFrame.enter F node seed =
        .ok (.Ok ⟨seed, c⟩, F', back) ∧
      counterAt F' s = some c' ∧ c'.val = 0 ∧
      (absState F).draw = some (⟨UInt64.ofBitVec seed.bv, 2 ^ 64 - 1⟩,
        { absState F with active := some ⟨UInt64.ofBitVec seed.bv, 2 ^ 64⟩ }) ∧
      (absState F').active = some ⟨UInt64.ofBitVec seed.bv, 0⟩ ∧
      (absState F').active ≠ some ⟨UInt64.ofBitVec seed.bv, 2 ^ 64⟩ := by
  obtain ⟨back, he⟩ :=
    enter_forward_eq F node seed draw s c hsite (getLast?_of_reverse hstack) hseed hc hkey
  have hself := counterAt_drawn_self F draw s seed c hc
  have h0 := wrapping_add_one_max c hmax
  have hact : (absState (drawnFrame F draw s seed c)).active =
      some ⟨UInt64.ofBitVec seed.bv, 0⟩ := by
    have hm : (drawnFrame F draw s seed c).metadata = F.metadata := rfl
    have hsc : (drawnFrame F draw s seed c).scopes = F.scopes := rfl
    simp only [absState, hsc, hstack]
    unfold keyOf
    rw [hself, hm, hseed]
    simp [h0]
  refine ⟨_, back, _, he, hself, h0, ?_, hact, ?_⟩
  · have hk : keyOf F s = some ⟨UInt64.ofBitVec seed.bv, 2 ^ 64 - 1⟩ := by
      unfold keyOf; rw [hseed, hc]; simp only [hmax]
    simp only [absState, hstack, hk, RandomState.draw]
    norm_num
  · rw [hact]; simp

/-- **Replay.** A replay site returns the recorded forward key and leaves the frame (every
counter, key and the inherited stream) unchanged: it is not another draw. -/
theorem enter_replay (F : Frame) (node : chelis_ir.dag.NodeId) (seed : U64)
    (draw : chelis_ir.evaluation.DrawId) (k : chelis_ir.evaluation.DrawKey)
    (hsite : siteOf F node = .ok (some (.Replay draw)))
    (hkey : F.keys.val[draw.val]? = some (some k)) :
    ∃ back, chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k, F, back) := by
  unfold chelis_ir.evaluation.ExecutionFrame.enter
  simp only [siteOf] at hsite
  simp only [hsite, bind_tc_ok, core.option.Option.ok_or, core.result.TryResult.branch,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
    core.option.Option.ok_or_else]
  simp [hkey]

/-- A replay whose forward draw has not been realized is an error, never an ordinal-0 key. -/
theorem enter_replay_unrealized (F : Frame) (node : chelis_ir.dag.NodeId) (seed : U64)
    (draw : chelis_ir.evaluation.DrawId)
    (hsite : siteOf F node = .ok (some (.Replay draw)))
    (hkey : F.keys.val[draw.val]? = some none) :
    ∃ e back, chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Err e, F, back) := by
  unfold chelis_ir.evaluation.ExecutionFrame.enter
  simp only [siteOf] at hsite
  simp only [hsite, bind_tc_ok, core.option.Option.ok_or, core.result.TryResult.branch,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize,
    core.option.Option.ok_or_else]
  simp [hkey,
    chelis_ir.evaluation.ExecutionFrame.enter.closure.Insts.CoreOpsFunctionFnOnceTupleString.call_once]
  exact ⟨_, _, rfl⟩

theorem control_enter (F : Frame) (scope : chelis_ir.evaluation.ScopeId) (payload : U64)
    (hlt : scope.val < F.counters.val.length) (hlen : F.scopes.val.length < Usize.max) :
    ∃ F' back, chelis_ir.evaluation.ExecutionFrame.control F (.Enter scope payload) =
        .ok (.Ok (), F', back) ∧
      F'.metadata = F.metadata ∧ F'.keys = F.keys ∧ F'.context = F.context ∧
      F'.scopes.val = F.scopes.val ++ [scope] ∧
      F'.counters.val = F.counters.val.set scope.val 0#u64 := by
  have hc : F.counters.val[scope.val]? = some F.counters.val[scope.val] :=
    List.getElem?_eq_getElem hlt
  unfold chelis_ir.evaluation.ExecutionFrame.control
  have hpush : ∃ v, alloc.vec.Vec.push F.scopes scope = .ok v ∧
      v.val = F.scopes.val ++ [scope] := by
    unfold alloc.vec.Vec.push
    have : F.scopes.val.length + 1 ≤ Usize.max := hlen
    rw [dif_pos (by simp [this])]
    exact ⟨_, rfl, List.concat_eq_append⟩
  obtain ⟨v, hv, hvv⟩ := hpush
  simp only [alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize,
    alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq, hc, hv, bind_tc_ok]
  exact ⟨_, _, rfl, rfl, rfl, rfl, hvv, alloc.vec.Vec.set_val_eq _ _ _⟩

/-- **Nested scope entry.** Entering a fresh non-root scope whose handler seed is `sd`
starts its key at ordinal `0` and saves every enclosing key unchanged: the abstraction
follows `RandomState.enter`. The payload seed of `Control::Enter` is ignored by the frame
(keys read the handler seed `metadata.scopes[scope].seed`). -/
theorem scope_relation (F : Frame) (scope : chelis_ir.evaluation.ScopeId) (payload sd : U64)
    (hlt : scope.val < F.counters.val.length) (hlen : F.scopes.val.length < Usize.max)
    (hroot : scope ≠ 0#usize) (hfresh : scope ∉ F.scopes.val) (hne : F.scopes.val ≠ [])
    (hseed : F.metadata.scopes.val[scope.val]? = some ⟨some sd⟩) :
    ∃ F' back, chelis_ir.evaluation.ExecutionFrame.control F (.Enter scope payload) =
        .ok (.Ok (), F', back) ∧
      absState F' = (absState F).enter (UInt64.ofBitVec sd.bv) := by
  obtain ⟨F', back, he, hm, hk, hctx, hsc, hcnt⟩ := control_enter F scope payload hlt hlen
  refine ⟨F', back, he, ?_⟩
  have hcounter : ∀ t, counterAt F' t =
      if t = scope then some 0#u64 else counterAt F t := by
    intro t
    by_cases hts : t = scope
    · subst hts
      simp [counterAt, hroot, hcnt, hlt]
    · have hv : scope.val ≠ t.val := fun e => hts (UScalar.eq_of_val_eq e).symm
      by_cases ht0 : t = 0#usize
      · subst ht0
        have h0s : (0#usize : chelis_ir.evaluation.ScopeId) ≠ scope := fun e => hroot e.symm
        simp [counterAt, hctx, h0s]
      · simp [counterAt, ht0, hcnt, hts, hv]
  have hkey_other : ∀ t, t ≠ scope → keyOf F' t = keyOf F t := by
    intro t ht
    unfold keyOf; rw [hm, hcounter t, if_neg ht]
  have hkey_new : keyOf F' scope = some ⟨UInt64.ofBitVec sd.bv, 0⟩ := by
    unfold keyOf; rw [hm, hcounter scope, if_pos rfl, hseed]; rfl
  obtain ⟨top, rest, hrev⟩ : ∃ top rest, F.scopes.val.reverse = top :: rest := by
    cases h : F.scopes.val.reverse with
    | nil => exact absurd (List.reverse_eq_nil_iff.mp h) hne
    | cons a l => exact ⟨a, l, rfl⟩
  have hrev' : F'.scopes.val.reverse = scope :: top :: rest := by
    rw [hsc, List.reverse_append, hrev]; rfl
  have hmem : ∀ t ∈ top :: rest, t ≠ scope := by
    intro t ht e; subst e
    exact hfresh (List.mem_reverse.mp (hrev ▸ ht))
  simp only [absState, hrev, hrev', hkey_new, RandomState.enter]
  congr 1
  rw [List.map_cons, hkey_other top (hmem top List.mem_cons_self),
    List.map_congr_left (fun t ht => hkey_other t (hmem t (List.mem_cons_of_mem _ ht)))]

/-- `Control::Leave` whose popped id is the left scope, with an enclosing scope below it:
the frame pops the scope and changes nothing else. -/
theorem control_leave (F : Frame) (scope : chelis_ir.evaluation.ScopeId)
    (pre : List chelis_ir.evaluation.ScopeId) (hsc : F.scopes.val = pre ++ [scope])
    (hpre : pre ≠ []) :
    ∃ v back, chelis_ir.evaluation.ExecutionFrame.control F (.Leave scope) =
        .ok (.Ok (), { F with scopes := v }, back) ∧ v.val = pre := by
  unfold chelis_ir.evaluation.ExecutionFrame.control
  have hlen : ¬ (alloc.vec.Vec.len F.scopes ≤ 1#usize) := by
    have : 2 ≤ F.scopes.val.length := by
      rw [hsc, List.length_append]
      have := List.length_pos_of_ne_nil hpre
      simp only [List.length_singleton]; omega
    intro h
    have h' := (UScalar.le_equiv _ _).mp h
    simp only [alloc.vec.Vec.len_val, alloc.vec.Vec.length] at h'
    have h1 : (1#usize : Usize).val = 1 := rfl
    omega
  simp only [hlen, if_false, alloc.vec.Vec.pop, hsc, List.getLast?_append_of_ne_nil _ (List.cons_ne_nil _ _),
    List.getLast?_singleton, bind_tc_ok, core.option.Option.Insts.CoreCmpPartialEqOption.eq,
    chelis_ir.evaluation.ScopeId.Insts.CoreCmpPartialEqScopeId.eq]
  simp

/-- Changing only the scope stack changes no key: keys read metadata, counters and the
inherited stream. -/
theorem keyOf_scopes (F : Frame) (v : alloc.vec.Vec chelis_ir.evaluation.ScopeId)
    (t : chelis_ir.evaluation.ScopeId) : keyOf { F with scopes := v } t = keyOf F t := rfl

theorem reverse_cons_of_ne_nil {α : Type} {l : List α} (h : l ≠ []) :
    ∃ top rest, l.reverse = top :: rest := by
  cases hr : l.reverse with
  | nil => exact absurd (List.reverse_eq_nil_iff.mp hr) h
  | cons a r => exact ⟨a, r, rfl⟩

/-- Success of a model computation (for kernel-evaluated facts about concrete literals). -/
def isOkR {α : Type} : Result α → Bool
  | .ok _ => true
  | _ => false

theorem ok_of_isOkR {α : Type} {r : Result α} (h : isOkR r = true) : ∃ a, r = .ok a := by
  cases r with
  | ok a => exact ⟨a, rfl⟩
  | fail _ => cases h
  | div => cases h

/-- The `String` of the frame's unmatched-scope diagnostic. -/
def unmatchedMsg : String :=
  match alloc.string.FromStrString.from_ (toStr "execution left an unmatched seed scope") with
  | .ok s => s
  | _ => ""

/-- The unmatched-scope message is valid UTF-8, so building the Rust `String` succeeds. -/
theorem unmatched_msg (h) :
    alloc.string.FromStrString.from_ (toStr "execution left an unmatched seed scope" h) =
      .ok unmatchedMsg := by
  have hk : isOkR (alloc.string.FromStrString.from_
      (toStr "execution left an unmatched seed scope")) = true := by decide +kernel
  obtain ⟨a, ha⟩ := ok_of_isOkR hk
  unfold unmatchedMsg
  rw [ha]

/-- **Scope leave.** Leaving the innermost scope (the popped id is the left scope, an
enclosing scope remains) succeeds, and the abstraction follows `RandomState.leave`: the saved
key of the enclosing scope becomes active again. -/
theorem leave_relation (F : Frame) (scope : chelis_ir.evaluation.ScopeId)
    (pre : List chelis_ir.evaluation.ScopeId) (hsc : F.scopes.val = pre ++ [scope])
    (hpre : pre ≠ []) :
    ∃ F' back, chelis_ir.evaluation.ExecutionFrame.control F (.Leave scope) =
        .ok (.Ok (), F', back) ∧ F'.scopes.val = pre ∧
      (absState F).leave = some (absState F') := by
  obtain ⟨v, back, he, hv⟩ := control_leave F scope pre hsc hpre
  refine ⟨_, back, he, hv, ?_⟩
  obtain ⟨top, rest, hrev⟩ := reverse_cons_of_ne_nil hpre
  have h1 : F.scopes.val.reverse = scope :: top :: rest := by
    rw [hsc, List.reverse_append, hrev]; rfl
  simp only [absState, h1, hv, hrev, RandomState.leave, List.map_cons]
  rfl

/-- **Popped-id check.** Rust `Leave` rejects a scope that is not the innermost one (after
popping the stack), whereas the reference `RandomState.leave` has no scope id and restores the
saved key: the two disagree on mismatched leaves. -/
theorem leave_mismatch (F : Frame) (scope t : chelis_ir.evaluation.ScopeId)
    (pre : List chelis_ir.evaluation.ScopeId) (hsc : F.scopes.val = pre ++ [t])
    (hpre : pre ≠ []) (ht : t ≠ scope) :
    ∃ e v back, chelis_ir.evaluation.ExecutionFrame.control F (.Leave scope) =
        .ok (.Err e, { F with scopes := v }, back) ∧ v.val = pre ∧
      (absState F).leave = some (absState { F with scopes := v }) := by
  unfold chelis_ir.evaluation.ExecutionFrame.control
  have hlen : ¬ (alloc.vec.Vec.len F.scopes ≤ 1#usize) := by
    have : 2 ≤ F.scopes.val.length := by
      rw [hsc, List.length_append]
      have := List.length_pos_of_ne_nil hpre
      simp only [List.length_singleton]; omega
    intro h
    have h' := (UScalar.le_equiv _ _).mp h
    simp only [alloc.vec.Vec.len_val, alloc.vec.Vec.length] at h'
    have h1 : (1#usize : Usize).val = 1 := rfl
    omega
  obtain ⟨top, rest, hrev⟩ := reverse_cons_of_ne_nil hpre
  have h1 : F.scopes.val.reverse = t :: top :: rest := by
    rw [hsc, List.reverse_append, hrev]; rfl
  have htv : ¬ (t.val = scope.val) := fun e => ht (UScalar.eq_of_val_eq e)
  simp only [hlen, if_false, alloc.vec.Vec.pop, hsc,
    List.getLast?_append_of_ne_nil _ (List.cons_ne_nil _ _),
    List.getLast?_singleton, bind_tc_ok, core.option.Option.Insts.CoreCmpPartialEqOption.eq,
    chelis_ir.evaluation.ScopeId.Insts.CoreCmpPartialEqScopeId.eq]
  simp [ht, core.convert.IntoFrom.into, unmatched_msg]
  simp only [absState, h1, hrev, RandomState.leave, List.map_cons]
  rfl

/-- Leaving the root scope (at most one entered scope) is an error in Rust, and the reference
`leave` of the abstracted state also fails. -/
theorem leave_root (F : Frame) (scope : chelis_ir.evaluation.ScopeId)
    (hlen : F.scopes.val.length ≤ 1) :
    (∃ e back, chelis_ir.evaluation.ExecutionFrame.control F (.Leave scope) =
        .ok (.Err e, F, back)) ∧ (absState F).leave = none := by
  constructor
  · unfold chelis_ir.evaluation.ExecutionFrame.control
    have hle : alloc.vec.Vec.len F.scopes ≤ 1#usize := by
      apply (UScalar.le_equiv _ _).mpr
      simp only [alloc.vec.Vec.len_val, alloc.vec.Vec.length]
      exact hlen
    simp only [hle, if_true, core.convert.IntoFrom.into, unmatched_msg, bind_tc_ok]
    exact ⟨_, _, rfl⟩
  · unfold absState
    match h : F.scopes.val.reverse with
    | [] => rfl
    | [_] => rfl
    | _ :: _ :: _ =>
      have := congrArg List.length h
      simp only [List.length_reverse, List.length_cons] at this
      omega

/-- **Enter then leave (fresh scope).** Entering a fresh scope and leaving it again gives back
the abstract state of the enclosing frame (`RandomState.leave_enter`). -/
theorem enter_leave_restores (F : Frame) (scope : chelis_ir.evaluation.ScopeId)
    (payload sd : U64)
    (hlt : scope.val < F.counters.val.length) (hlen : F.scopes.val.length < Usize.max)
    (hroot : scope ≠ 0#usize) (hfresh : scope ∉ F.scopes.val) (hne : F.scopes.val ≠ [])
    (hseed : F.metadata.scopes.val[scope.val]? = some ⟨some sd⟩) :
    ∃ F' F'' back back', chelis_ir.evaluation.ExecutionFrame.control F (.Enter scope payload) =
        .ok (.Ok (), F', back) ∧
      chelis_ir.evaluation.ExecutionFrame.control F' (.Leave scope) = .ok (.Ok (), F'', back') ∧
      F''.scopes.val = F.scopes.val ∧ absState F'' = absState F := by
  obtain ⟨F', back, he, _, _, _, hsc, _⟩ := control_enter F scope payload hlt hlen
  obtain ⟨F'₁, back₁, he₁, hrel⟩ :=
    scope_relation F scope payload sd hlt hlen hroot hfresh hne hseed
  rw [he] at he₁
  cases he₁
  obtain ⟨F'', back', hl, hsc'', hleave⟩ := leave_relation F' scope F.scopes.val hsc hne
  refine ⟨F', F'', back, back', he, hl, hsc'', ?_⟩
  rw [hrel, RandomState.leave_enter] at hleave
  exact (Option.some.inj hleave).symm

/-- **Re-entrancy counterexample.** Re-entering the innermost scope while it is still on the
stack resets its (shared) counter slot, so after the matching leave the outer occurrence resumes
at ordinal `0` instead of its saved ordinal `c`: the abstraction no longer follows the reference,
for which `(state.enter seed).leave = some state`. Hence `scope ∉ scopes` is necessary in
`scope_relation`. -/
theorem reentry_counterexample (F : Frame) (scope : chelis_ir.evaluation.ScopeId)
    (payload sd c : U64) (pre : List chelis_ir.evaluation.ScopeId)
    (hsc : F.scopes.val = pre ++ [scope]) (hlen : F.scopes.val.length < Usize.max)
    (hroot : scope ≠ 0#usize)
    (hseed : F.metadata.scopes.val[scope.val]? = some ⟨some sd⟩)
    (hc : F.counters.val[scope.val]? = some c) (hc0 : c.val ≠ 0) :
    ∃ F' F'' back back', chelis_ir.evaluation.ExecutionFrame.control F (.Enter scope payload) =
        .ok (.Ok (), F', back) ∧
      chelis_ir.evaluation.ExecutionFrame.control F' (.Leave scope) = .ok (.Ok (), F'', back') ∧
      F''.scopes.val = F.scopes.val ∧
      (absState F).active = some ⟨UInt64.ofBitVec sd.bv, c.val⟩ ∧
      (absState F'').active = some ⟨UInt64.ofBitVec sd.bv, 0⟩ ∧
      absState F'' ≠ absState F ∧
      ((absState F).enter (UInt64.ofBitVec sd.bv)).leave = some (absState F) := by
  have hlt : scope.val < F.counters.val.length := (List.getElem?_eq_some_iff.mp hc).1
  obtain ⟨F', back, he, hm, _, hctx, hsc', hcnt⟩ := control_enter F scope payload hlt hlen
  have hne' : F.scopes.val ≠ [] := by rw [hsc]; simp
  obtain ⟨F'', back', hl, hsc'', hleave⟩ := leave_relation F' scope F.scopes.val hsc' hne'
  have h2 : F.scopes.val.reverse = scope :: pre.reverse := by rw [hsc]; simp
  have hF : (absState F).active = some ⟨UInt64.ofBitVec sd.bv, c.val⟩ := by
    simp only [absState, h2, keyOf, counterAt, hroot, if_false, hseed, hc]
  have hF'' : (absState F'').active = some ⟨UInt64.ofBitVec sd.bv, 0⟩ := by
    -- `F'` has counter `0` at `scope`; the leave restores the saved key of the outer `scope`.
    have h1 : F'.scopes.val.reverse = scope :: scope :: pre.reverse := by rw [hsc', hsc]; simp
    have hk : keyOf F' scope = some ⟨UInt64.ofBitVec sd.bv, 0⟩ := by
      simp [keyOf, counterAt, hroot, hm, hseed, hcnt, hlt]
    simp only [absState, h1, RandomState.leave, List.map_cons, hk] at hleave
    have e := Option.some.inj hleave
    unfold absState
    rw [← e]
  refine ⟨F', F'', back, back', he, hl, hsc'', hF, hF'', ?_, RandomState.leave_enter _ _⟩
  intro heq
  have hact := congrArg RandomState.active heq
  rw [hF, hF''] at hact
  simp only [Option.some.injEq, DrawKey.mk.injEq] at hact
  exact hc0 hact.2.symm

end ChelisBridge.Random
