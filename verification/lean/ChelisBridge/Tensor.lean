import ChelisBridge.Float
import Aeneas.Std.Vec
import LaCaDiLECore.TensorLiteral
import LaCaDiLECore.GraphSyntax

/-!
# Flat row-major storage as reference tensors

`decodeTensor s xs` reads a flat buffer of IEEE elements with a concrete shape `s` into
`LaCaDiLECore.Tensor ℚ s`, coordinate `i` being element `Index.flat i` (row-major). It
fails, never defaults, when `xs.length ≠ s.prod` or when any element is NaN/infinite.

The same flat buffer reinterpreted with a singleton axis inserted is exactly
`LaCaDiLECore.insert` on that axis (`decodeTensor_reshape_singleton`): this is the
value-preservation fact behind decoding the AD pass's generated `Reshape` as `insert`.
-/

namespace ChelisBridge

open Aeneas.Std LaCaDiLECore

section MapM

variable {α β : Type}

theorem mapM_eq_some_iff (g : α → Option β) :
    ∀ (xs : List α) (ys : List β),
      xs.mapM g = some ys ↔ List.Forall₂ (fun x y => g x = some y) xs ys
  | [], ys => by cases ys <;> simp
  | x :: xs, [] => by
      rw [List.mapM_cons]
      cases g x <;> cases xs.mapM g <;> simp
  | x :: xs, y :: ys => by
      rw [List.mapM_cons, List.forall₂_cons, ← mapM_eq_some_iff g xs ys]
      cases g x <;> cases xs.mapM g <;> simp

theorem mapM_eq_none_of_mem (g : α → Option β) (xs : List α) (x : α) (hx : x ∈ xs)
    (hg : g x = none) : xs.mapM g = none := by
  induction xs with
  | nil => simp at hx
  | cons y ys ih =>
      rcases List.mem_cons.mp hx with rfl | hmem
      · simp [List.mapM_cons, hg]
      · cases hy : g y <;> simp [List.mapM_cons, hy, ih hmem]

end MapM

variable {f : BinaryFormat}

/-- Realize every element; fails on any NaN or infinity. -/
def realizeList (xs : List (BinaryFloat f)) : Option (List ℚ) := xs.mapM realize

/-- Decode flat row-major storage with concrete shape `s`. Fails on length mismatch
(`xs.length ≠ s.prod`) and on any non-finite element; never fills a default. -/
def decodeTensor (s : Shape) (xs : List (BinaryFloat f)) : Option (Tensor ℚ s) := do
  let qs ← realizeList xs
  Tensor.ofList s qs

/-- Decode an extracted Rust `Vec` buffer. -/
def decodeVec (s : Shape) (v : alloc.vec.Vec (BinaryFloat f)) : Option (Tensor ℚ s) :=
  decodeTensor s v.val

theorem realizeList_eq_some {xs : List (BinaryFloat f)} {qs : List ℚ} :
    realizeList xs = some qs ↔ List.Forall₂ (fun x q => realize x = some q) xs qs :=
  mapM_eq_some_iff realize xs qs

theorem realizeList_eq_none_of_nonfinite {xs : List (BinaryFloat f)} {x : BinaryFloat f}
    (hx : x ∈ xs) (hnf : x.isFinite = false) : realizeList xs = none :=
  mapM_eq_none_of_mem realize xs x hx (by simp [realize, hnf])

theorem ofList_apply {C : Type} {s : Shape} {xs : List C} {t : Tensor C s}
    (h : Tensor.ofList s xs = some t) (i : Index s) :
    ∃ hi : i.flat < xs.length, t i = xs[i.flat] := by
  unfold Tensor.ofList at h
  split at h
  · rename_i hn
    cases Option.some.inj h
    have hlt : i.flat < xs.length := by
      rw [← Index.equivFin_val, hn]; exact (Index.equivFin s i).isLt
    exact ⟨hlt, by simp only [Index.equivFin_val]⟩
  · contradiction

theorem ofList_length {C : Type} {s : Shape} {xs : List C} {t : Tensor C s}
    (h : Tensor.ofList s xs = some t) : xs.length = s.prod := by
  unfold Tensor.ofList at h
  split at h
  · assumption
  · contradiction

/-- **Row-major read-back.** Reading coordinate `i` of a decoded tensor is the realized
value of the flat element at `Index.flat i`. -/
theorem decodeTensor_get {s : Shape} {xs : List (BinaryFloat f)} {t : Tensor ℚ s}
    (h : decodeTensor s xs = some t) (i : Index s) :
    ∃ hi : i.flat < xs.length, realize xs[i.flat] = some (t i) := by
  simp only [decodeTensor, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨qs, hqs, ht⟩ := h
  have hf := realizeList_eq_some.mp hqs
  obtain ⟨hi, hti⟩ := ofList_apply ht i
  have hlen := hf.length_eq
  refine ⟨hlen ▸ hi, ?_⟩
  rw [hti]
  exact hf.get _ _

theorem decodeTensor_length {s : Shape} {xs : List (BinaryFloat f)} {t : Tensor ℚ s}
    (h : decodeTensor s xs = some t) : xs.length = s.prod := by
  simp only [decodeTensor, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨qs, hqs, ht⟩ := h
  rw [(realizeList_eq_some.mp hqs).length_eq]
  exact ofList_length ht

/-- A length mismatch is rejected. -/
theorem decodeTensor_eq_none_of_length {s : Shape} {xs : List (BinaryFloat f)}
    (h : xs.length ≠ s.prod) : decodeTensor s xs = none := by
  cases hd : decodeTensor s xs with
  | none => rfl
  | some t => exact absurd (decodeTensor_length hd) h

/-- Any NaN or infinity is rejected; it is never mapped to a number. -/
theorem decodeTensor_eq_none_of_nonfinite {s : Shape} {xs : List (BinaryFloat f)}
    {x : BinaryFloat f} (hx : x ∈ xs) (hnf : x.isFinite = false) :
    decodeTensor s xs = none := by
  simp [decodeTensor, realizeList_eq_none_of_nonfinite hx hnf]

/-- Decoding succeeds exactly for finite buffers of the right length. -/
theorem decodeTensor_isSome_iff {s : Shape} {xs : List (BinaryFloat f)} :
    (decodeTensor s xs).isSome ↔ xs.length = s.prod ∧ ∀ x ∈ xs, x.isFinite = true := by
  constructor
  · intro h
    obtain ⟨t, ht⟩ := Option.isSome_iff_exists.mp h
    refine ⟨decodeTensor_length ht, fun x hx => ?_⟩
    cases hfin : x.isFinite
    · rw [decodeTensor_eq_none_of_nonfinite hx hfin] at ht; contradiction
    · rfl
  · rintro ⟨hlen, hfin⟩
    have : ∃ qs, realizeList xs = some qs := by
      clear hlen
      induction xs with
      | nil => exact ⟨[], rfl⟩
      | cons x xs ih =>
          obtain ⟨qs, hqs⟩ := ih (fun y hy => hfin y (List.mem_cons_of_mem _ hy))
          refine ⟨value x :: qs, ?_⟩
          have hx : realize x = some (value x) := by
            simp [realize, hfin x List.mem_cons_self]
          show (x :: xs).mapM realize = some (value x :: qs)
          rw [List.mapM_cons, hx]
          simp only [realizeList] at hqs
          simp [hqs]
    obtain ⟨qs, hqs⟩ := this
    have hq := (realizeList_eq_some.mp hqs).length_eq
    simp [decodeTensor, hqs, Tensor.ofList, ← hq, hlen]

section Reshape

/-- Removing an axis of extent `n` divides the element count by `n`. -/
theorem Axis.prod_eq {s t : Shape} {n : ℕ} (a : Axis s n t) : s.prod = n * t.prod := by
  induction a with
  | here => simp
  | there a ih => simp only [List.prod_cons, ih]; ring

theorem flat_split_of_one {s t : Shape} {n : ℕ} (a : Axis s n t) :
    n = 1 → ∀ i : Index s, (a.split i).2.flat = i.flat := by
  induction a with
  | here =>
      intro hn i
      subst hn
      rcases i with ⟨j, rest⟩
      simp [Axis.split, Index.flat]
  | @there s' _ t' m a ih =>
      intro hn i
      rcases i with ⟨j, rest⟩
      have hp : s'.prod = t'.prod := by rw [Axis.prod_eq a, hn, one_mul]
      have h1 := ih hn rest
      rcases hs : a.split rest with ⟨k, ks⟩
      rw [hs] at h1
      simp only at h1
      simp only [Axis.split, hs, Index.flat]
      rw [h1, hp]

/-- Deleting a singleton axis preserves the row-major position. -/
theorem flat_split_singleton {s t : Shape} (a : Axis s 1 t) (i : Index s) :
    (a.split i).2.flat = i.flat :=
  flat_split_of_one a rfl i

/-- Row-major reinterpretation of the same flat payload with one singleton axis inserted
is `insert` on that axis. This is the value lemma for decoding a singleton-axis
`Reshape` (emitted after `Sum` by the same-rank `Expand` adjoint) as `insert`. -/
theorem ofList_reshape_singleton {C : Type} [CommSemiring C] {s t : Shape}
    (a : Axis s 1 t) {xs : List C} {x : Tensor C t} (h : Tensor.ofList t xs = some x) :
    Tensor.ofList s xs = some (LaCaDiLECore.insert a x) := by
  have hlen : xs.length = s.prod := by rw [ofList_length h, Axis.prod_eq a, one_mul]
  unfold Tensor.ofList
  rw [dif_pos hlen]
  congr 1
  funext i
  obtain ⟨hi, hxi⟩ := ofList_apply h (a.split i).2
  simp only [LaCaDiLECore.insert, hxi, Index.equivFin_val, flat_split_singleton]

theorem decodeTensor_reshape_singleton {s t : Shape} (a : Axis s 1 t)
    {xs : List (BinaryFloat f)} {x : Tensor ℚ t} (h : decodeTensor t xs = some x) :
    decodeTensor s xs = some (LaCaDiLECore.insert a x) := by
  simp only [decodeTensor, Option.bind_eq_bind, Option.bind_eq_some_iff] at h ⊢
  obtain ⟨qs, hqs, ht⟩ := h
  exact ⟨qs, hqs, ofList_reshape_singleton a ht⟩

/-- Row-major reinterpretation of a flat payload with one singleton axis deleted is `sum`
over that extent-one axis. This is the value lemma for decoding the singleton-removing
`Reshape` emitted by the Reshape adjoint as `sum`. -/
theorem ofList_remove_singleton {C : Type} [CommSemiring C] {s t : Shape}
    (a : Axis s 1 t) {xs : List C} {x : Tensor C s} (h : Tensor.ofList s xs = some x) :
    Tensor.ofList t xs = some (LaCaDiLECore.sum a x) := by
  have hlen : xs.length = t.prod := by rw [ofList_length h, Axis.prod_eq a, one_mul]
  obtain ⟨y, hy⟩ : ∃ y, Tensor.ofList t xs = some y := by
    unfold Tensor.ofList; rw [dif_pos hlen]; exact ⟨_, rfl⟩
  have hx : x = LaCaDiLECore.insert a y := by
    have := ofList_reshape_singleton a hy
    rw [h] at this
    exact Option.some.inj this
  rw [hy, hx, sum_insert_singleton]

theorem decodeTensor_remove_singleton {s t : Shape} (a : Axis s 1 t)
    {xs : List (BinaryFloat f)} {x : Tensor ℚ s} (h : decodeTensor s xs = some x) :
    decodeTensor t xs = some (LaCaDiLECore.sum a x) := by
  simp only [decodeTensor, Option.bind_eq_bind, Option.bind_eq_some_iff] at h ⊢
  obtain ⟨qs, hqs, ht⟩ := h
  exact ⟨qs, hqs, ofList_remove_singleton a ht⟩

end Reshape

section Environment

/-- Decode one flat buffer per context entry, in context order. -/
def decodeValues : (Γ : Context) → List (List (BinaryFloat f)) → Option (Values ℚ Γ)
  | [], [] => some (fun i => nomatch i)
  | s :: Γ, xs :: rest => do
      let x ← decodeTensor s xs
      let ρ ← decodeValues Γ rest
      pure (Values.cons x ρ)
  | _, _ => none

/-- Each context variable reads the tensor decoded from the buffer at its offset. -/
theorem decodeValues_get {Γ : Context} {bufs : List (List (BinaryFloat f))}
    {ρ : Values ℚ Γ} (h : decodeValues Γ bufs = some ρ) {s : Shape} (v : Var Γ s) :
    ∃ xs, bufs[v.offset]? = some xs ∧ decodeTensor s xs = some (v.get ρ) := by
  induction v generalizing bufs with
  | @here s Γ =>
      cases bufs with
      | nil => simp [decodeValues] at h
      | cons xs rest =>
          simp only [decodeValues, Option.bind_eq_bind, Option.bind_eq_some_iff,
            Option.pure_def, Option.some.injEq] at h
          obtain ⟨x, hx, ρ', _, rfl⟩ := h
          exact ⟨xs, rfl, hx⟩
  | @there Γ s t v ih =>
      cases bufs with
      | nil => simp [decodeValues] at h
      | cons xs rest =>
          simp only [decodeValues, Option.bind_eq_bind, Option.bind_eq_some_iff,
            Option.pure_def, Option.some.injEq] at h
          obtain ⟨x, _, ρ', hρ, rfl⟩ := h
          obtain ⟨ys, hys, hd⟩ := ih hρ
          exact ⟨ys, by simpa [Var.offset] using hys, hd⟩

end Environment

end ChelisBridge
