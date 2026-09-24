import Aeneas

/-!
# Loop and iterator steps of the Aeneas standard-library models

Facts about library models only (no generated declaration): one step of
`(start..end).next()` over `usize`, and collecting `slice.iter().map(f)` into a `Vec`
with a closure whose captured state never changes.
-/

namespace ChelisBridge.Protocol

open Aeneas.Std

/-- `(k..n).next()` with `k < n` yields `k` and advances the start by one. -/
theorem range_next_lt (k n : Usize) (h : k.val < n.val) :
    ∃ k' : Usize, k'.val = k.val + 1 ∧
      core.iter.range.IteratorRange.next core.iter.range.StepUsize ⟨k, n⟩ =
        .ok (some k, ⟨k', n⟩) := by
  have hmax : k.val + (1#usize : Usize).val ≤ Usize.max := by
    have h1 : (1#usize : Usize).val = 1 := rfl; rw [h1]; scalar_tac
  unfold core.iter.range.IteratorRange.next
  have hlt : core.cmp.impls.PartialOrdUsize.lt k n = true := by
    simp [core.cmp.impls.PartialOrdUsize.lt, h]
  have hspec := Usize.checked_add_bv_spec k 1#usize
  cases hc : Usize.checked_add k 1#usize with
  | none => rw [hc] at hspec; simp only at hspec; omega
  | some z =>
    rw [hc] at hspec
    refine ⟨z, by simpa using hspec.2.1, ?_⟩
    simp [core.iter.range.StepUsize.forward_checked, hlt, hc,
      core.clone.impls.CloneUsize.clone]

/-- An exhausted range `(k..n)`, `n ≤ k`, yields nothing and is unchanged. -/
theorem range_next_ge (k n : Usize) (h : n.val ≤ k.val) :
    core.iter.range.IteratorRange.next core.iter.range.StepUsize ⟨k, n⟩ = .ok (none, ⟨k, n⟩) := by
  unfold core.iter.range.IteratorRange.next
  have hlt : core.cmp.impls.PartialOrdUsize.lt k n = false := by
    simp [core.cmp.impls.PartialOrdUsize.lt]; omega
  simp [hlt]

section IterMap

variable {T B F : Type} (fnInst : core.ops.function.FnMut F T B) (c : F) (f : T → Result B)

theorem mapSlice_next_some (hf : ∀ x, fnInst.call_mut c x = (do let y ← f x; .ok (y, c)))
    (s : Slice T) (k : ℕ) (hk : k < s.val.length) :
    (core.iter.traits.iterator.IteratorMap (core.iter.traits.iterator.IteratorSliceIter T)
        fnInst).next ⟨{ slice := s, i := k, back := 0 }, c⟩ =
      (do let y ← f s.val[k]; .ok (some y, ⟨{ slice := s, i := k + 1, back := 0 }, c⟩)) := by
  simp only [core.iter.adapters.map.IteratorMap.next, core.slice.iter.IteratorSliceIter.next]
  have hk' : k < s.length - 0 := by simp only [Slice.length]; omega
  simp only [hk', dite_true, bind_tc_ok, hf, Slice.getElem_Nat_eq]
  cases f s.val[k] <;> rfl

theorem mapSlice_next_none (s : Slice T) (k : ℕ) (hk : s.val.length ≤ k) :
    (core.iter.traits.iterator.IteratorMap (core.iter.traits.iterator.IteratorSliceIter T)
        fnInst).next ⟨{ slice := s, i := k, back := 0 }, c⟩ =
      .ok (none, ⟨{ slice := s, i := k, back := 0 }, c⟩) := by
  simp only [core.iter.adapters.map.IteratorMap.next, core.slice.iter.IteratorSliceIter.next]
  have hk' : ¬ (k < s.length - 0) := by simp only [Slice.length]; omega
  simp only [hk', dite_false, bind_tc_ok]

/-- Collecting `slice.iter().map(f)` for a closure whose state never changes: a successful
collection is `mapM` of the underlying function over the remaining elements. -/
theorem iterToList_map_slice (hf : ∀ x, fnInst.call_mut c x = (do let y ← f x; .ok (y, c)))
    (s : Slice T) :
    ∀ (n k : ℕ) (acc : List B) (l : List B) it', s.val.length - k = n →
      alloc.vec.FromIteratorVec.iterToList
          (core.iter.traits.iterator.IteratorMap (core.iter.traits.iterator.IteratorSliceIter T)
            fnInst)
          ⟨{ slice := s, i := k, back := 0 }, c⟩ acc = .ok (l, it') →
      ∃ ys, (s.val.drop k).mapM f = .ok ys ∧ l = acc.reverse ++ ys := by
  intro n
  induction n with
  | zero =>
    intro k acc l it' hn h
    rw [alloc.vec.FromIteratorVec.iterToList.eq_def,
      mapSlice_next_none fnInst c s k (by omega)] at h
    have hd : s.val.drop k = [] := List.drop_eq_nil_of_le (by omega)
    refine ⟨[], by rw [hd]; rfl, ?_⟩
    simp only [bind_tc_ok, Result.ok.injEq, Prod.mk.injEq] at h
    rw [← h.1]; simp
  | succ n ih =>
    intro k acc l it' hn h
    have hk : k < s.val.length := by omega
    rw [alloc.vec.FromIteratorVec.iterToList.eq_def,
      mapSlice_next_some fnInst c f hf s k hk] at h
    rw [List.drop_eq_getElem_cons hk]
    cases hfx : f s.val[k] with
    | ok y =>
      simp only [hfx, bind_tc_ok] at h
      obtain ⟨ys, hys, hl⟩ := ih (k + 1) (y :: acc) l it' (by omega) h
      refine ⟨y :: ys, ?_, ?_⟩
      · simp only [List.mapM_cons, hfx, hys, bind_tc_ok]; rfl
      · rw [hl]; simp
    | fail e => simp [hfx] at h
    | div => simp [hfx] at h

end IterMap

end ChelisBridge.Protocol
