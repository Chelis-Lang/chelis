import ChelisBridge.Numeric
import ChelisBridge.Tensor

/-!
# Flat row-major kernels of the Rust evaluator versus `opEval`

The Rust evaluator stores every tensor as one flat buffer in row-major order and plans
its kernels with explicit index arithmetic (`crates/chelis-ir/src/eval.rs`):

* `linear_to_index` / `index_to_linear` — `linearToIndex` / `indexToLinear`;
* `axis_reduction_groups` — `reductionGroups`: the group of output position `k` lists,
  for `j = 0, 1, …`, the input position with `j` inserted at the reduced axis;
* `expand` — `expandPicks`: rank-increasing removes the axis coordinate, same-rank sets it
  to `0`; the storage is gathered (`reuse_gather`), never re-rounded;
* `reshape` — the buffer is moved unchanged;
* `float_tensor_binop` — elementwise `zipWith` of equal-length buffers.

Each theorem here states that the flat kernel, read back row-major with
`Tensor.ofList` (bitwise, no realization), is exactly the corresponding `opEval` clause.
These are the index-level obligations between the extracted kernels and the IEEE
reference evaluation.
-/

namespace ChelisBridge.Numeric

open Aeneas.Std LaCaDiLECore

/-! ## Coordinates and Rust's index arithmetic -/

/-- The coordinate list of an index. -/
def coordsOf : {s : Shape} → Index s → List Nat
  | [], _ => []
  | _ :: _, (i, rest) => i.val :: coordsOf rest

/-- `linear_to_index`: component `k` is `(linear / ∏ shape[k+1..]) % shape[k]`. -/
def linearToIndex : List Nat → Nat → List Nat
  | [], _ => []
  | d :: ds, lin => (lin / ds.prod) % d :: linearToIndex ds (lin % ds.prod)

/-- `index_to_linear` (Horner form). -/
def indexToLinear : List Nat → List Nat → Nat
  | _ :: ds, v :: vs => v * ds.prod + indexToLinear ds vs
  | _, _ => 0

/-- Insert `j` at position `k` (the input index built by `axis_reduction_groups`). -/
def insertAt : Nat → Nat → List Nat → List Nat
  | 0, j, l => j :: l
  | _ + 1, j, [] => [j]
  | k + 1, j, x :: l => x :: insertAt k j l

theorem flat_lt {s : Shape} (i : Index s) : i.flat < s.prod := by
  rw [← Index.equivFin_val]; exact (Index.equivFin s i).isLt

theorem linearToIndex_flat : ∀ {s : Shape} (i : Index s), linearToIndex s i.flat = coordsOf i
  | [], _ => rfl
  | d :: ds, (i, rest) => by
      have hr := flat_lt rest
      have hp : 0 < ds.prod := by omega
      simp only [Index.flat, linearToIndex, coordsOf]
      have h1 : (i.val * ds.prod + rest.flat) / ds.prod = i.val := by
        rw [Nat.add_comm, Nat.add_mul_div_right _ _ hp, Nat.div_eq_of_lt hr, Nat.zero_add]
      have h2 : (i.val * ds.prod + rest.flat) % ds.prod = rest.flat := by
        rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hr]
      rw [h1, h2, Nat.mod_eq_of_lt i.isLt, linearToIndex_flat rest]

theorem indexToLinear_coords : ∀ {s : Shape} (i : Index s), indexToLinear s (coordsOf i) = i.flat
  | [], _ => rfl
  | _ :: _, (i, rest) => by
      simp only [coordsOf, indexToLinear, Index.flat, indexToLinear_coords rest]

theorem coords_join {s t : Shape} {n : Nat} (a : Axis s n t) (j : Fin n) (i : Index t) :
    coordsOf (a.join (j, i)) = insertAt a.position j.val (coordsOf i) := by
  induction a with
  | here => rfl
  | there a ih =>
      rcases i with ⟨i0, is⟩
      simp only [Axis.join, coordsOf, Axis.position, insertAt, ih]

theorem coords_split {s t : Shape} {n : Nat} (a : Axis s n t) (i : Index s) :
    coordsOf (a.split i).2 = (coordsOf i).eraseIdx a.position := by
  induction a with
  | here => rcases i with ⟨i0, is⟩; rfl
  | there a ih =>
      rcases i with ⟨i0, is⟩
      simp only [Axis.split, coordsOf, Axis.position, List.eraseIdx_cons_succ, ih]

theorem length_coords : ∀ {s : Shape} (i : Index s), (coordsOf i).length = s.length
  | [], _ => rfl
  | _ :: _, (_, rest) => by simp [coordsOf, length_coords rest]

theorem axis_eraseIdx {s t : Shape} {n : Nat} (a : Axis s n t) :
    s.eraseIdx a.position = t ∧ s[a.position]? = some n := by
  induction a with
  | here => simp [Axis.position]
  | there a ih => simpa [Axis.position] using ih

theorem insertAt_eraseIdx : ∀ (k v : Nat) (l : List Nat), k < l.length →
    insertAt k v (l.eraseIdx k) = l.set k v
  | 0, v, x :: l, _ => rfl
  | k + 1, v, x :: l, h => by
      simp only [List.eraseIdx_cons_succ, List.set_cons_succ, insertAt,
        insertAt_eraseIdx k v l (by simpa using h)]

theorem atPosition_position (pre : Shape) (n : Nat) (post : Shape) :
    (Axis.atPosition pre n post).position = pre.length := by
  induction pre with
  | nil => rfl
  | cons _ pre ih => simp [Axis.atPosition, Axis.position, ih]

/-! ## Reading a flat buffer back row-major -/

variable {f : BinaryFormat}

/-- Element read of a flat buffer (`TensorStorage::scalar_at`; out-of-range reads panic in
Rust and never occur for planned indices). -/
def at' (xs : List (BinaryFloat f)) (k : Nat) : BinaryFloat f := xs.getD k (BinaryFloat.zero f false)

theorem ofList_eq_some_of {C : Type} {s : Shape} {xs : List C} {x : Tensor C s}
    (hlen : xs.length = s.prod) (h : ∀ i : Index s, xs[i.flat]? = some (x i)) :
    Tensor.ofList s xs = some x := by
  unfold Tensor.ofList
  rw [dif_pos hlen]
  congr 1
  funext i
  have hi := h i
  rw [List.getElem?_eq_some_iff] at hi
  obtain ⟨_, hi⟩ := hi
  simp only [Index.equivFin_val, hi]

theorem at'_flat {s : Shape} {xs : List (BinaryFloat f)} {x : Tensor (BinaryFloat f) s}
    (hx : Tensor.ofList s xs = some x) (i : Index s) : at' xs i.flat = x i := by
  obtain ⟨hi, hxi⟩ := ofList_apply hx i
  simp [at', List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, hxi]

/-- A planned gather (`reuse_gather picks`) read back row-major. -/
theorem gather_flat {s t : Shape} {xs : List (BinaryFloat f)} {x : Tensor (BinaryFloat f) s}
    (hx : Tensor.ofList s xs = some x) (picks : List Nat) (g : Index t → Index s)
    (hlen : picks.length = t.prod) (hp : ∀ i : Index t, picks[i.flat]? = some (g i).flat) :
    Tensor.ofList t (picks.map (at' xs)) = some (fun i => x (g i)) := by
  apply ofList_eq_some_of (by simp [hlen])
  intro i
  rw [List.getElem?_map, hp i]
  simp [at'_flat hx]

/-! ## Elementwise kernels -/

/-- `float_tensor_binop` on equal-shape operands (`Add` → `BinaryFloat.add`, `Mul` →
`BinaryFloat.mul`), operand order preserved. -/
theorem zipWith_flat {s : Shape} {xs ys : List (BinaryFloat f)} {x y : Tensor (BinaryFloat f) s}
    (hx : Tensor.ofList s xs = some x) (hy : Tensor.ofList s ys = some y)
    (g : BinaryFloat f → BinaryFloat f → BinaryFloat f) :
    Tensor.ofList s (List.zipWith g xs ys) = some (fun i => g (x i) (y i)) := by
  have hlx := ofList_length hx
  have hly := ofList_length hy
  apply ofList_eq_some_of (by simp [hlx, hly])
  intro i
  obtain ⟨hi, hxi⟩ := ofList_apply hx i
  obtain ⟨hj, hyi⟩ := ofList_apply hy i
  rw [List.getElem?_zipWith, List.getElem?_eq_getElem hi, List.getElem?_eq_getElem hj, ← hxi, ← hyi]

/-! ## Axis reduction -/

/-- `axis_reduction_groups input_shape axis` (groups only). -/
def reductionGroups (s : List Nat) (axis : Nat) : List (List Nat) :=
  (List.range (s.eraseIdx axis).prod).map fun k =>
    (List.range (s.getD axis 0)).map fun j =>
      indexToLinear s (insertAt axis j (linearToIndex (s.eraseIdx axis) k))

/-- The Rust typed `Sum` over a flat buffer: one pairwise fold per group. -/
def flatSum (xs : List (BinaryFloat f)) (s : List Nat) (axis : Nat) : List (BinaryFloat f) :=
  (reductionGroups s axis).map fun g => pairFold (g.map (at' xs))

theorem reductionGroups_flat {s t : Shape} {n : Nat} (a : Axis s n t) (i : Index t) :
    (reductionGroups s a.position)[i.flat]? =
      some ((List.ofFn fun j : Fin n => (a.join (j, i)).flat)) := by
  obtain ⟨he, hn⟩ := axis_eraseIdx a
  have hlt := flat_lt i
  unfold reductionGroups
  rw [he, List.getElem?_map, List.getElem?_range hlt, Option.map_some, linearToIndex_flat]
  congr 1
  have hd : s.getD a.position 0 = n := by simp [List.getD_eq_getElem?_getD, hn]
  rw [hd]
  apply List.ext_getElem (by simp)
  intro k h1 h2
  simp only [List.getElem_map, List.getElem_range, List.getElem_ofFn]
  rw [← coords_join a ⟨k, by simpa using h1⟩ i, indexToLinear_coords]

/-- **The Rust axis `Sum` is `opEval (.sum a)` row-major**: groups list the reduced axis
in increasing order and each is folded pairwise. -/
theorem flatSum_eq {s t : Shape} {n : Nat} (a : Axis s n t) {xs : List (BinaryFloat f)}
    {x : Tensor (BinaryFloat f) s} (hx : Tensor.ofList s xs = some x) :
    Tensor.ofList t (flatSum xs s a.position) = some (fun i => pairFold (sumLeaves a x i)) := by
  have hlen : (flatSum xs s a.position).length = t.prod := by
    simp [flatSum, reductionGroups, (axis_eraseIdx a).1]
  apply ofList_eq_some_of hlen
  intro i
  unfold flatSum
  rw [List.getElem?_map, reductionGroups_flat a i, Option.map_some]
  congr 2
  unfold sumLeaves
  apply List.ext_getElem (by simp)
  intro k h1 h2
  simp [at'_flat hx]

/-! ## Movement -/

/-- `expand`'s gather plan for output shape `out` and input shape `inp`. -/
def expandPicks (inp out : List Nat) (axis : Nat) : List Nat :=
  (List.range out.prod).map fun k =>
    let idx := linearToIndex out k
    indexToLinear inp (if out.length = inp.length then idx.set axis 0 else idx.eraseIdx axis)

theorem expandPicks_length (inp out : List Nat) (axis : Nat) :
    (expandPicks inp out axis).length = out.prod := by simp [expandPicks]

/-- Rank-increasing `Expand` (decoded `insert a`) gathers row-major to `opEval (.insert a)`. -/
theorem expand_insert_flat {s t : Shape} {n : Nat} (a : Axis s n t) {xs : List (BinaryFloat f)}
    {x : Tensor (BinaryFloat f) t} (hx : Tensor.ofList t xs = some x) :
    Tensor.ofList s ((expandPicks t s a.position).map (at' xs)) =
      some (fun i => x (a.split i).2) := by
  have hrank : s.length = t.length + 1 := by
    have := congrArg List.length (axis_eraseIdx a).1
    have hp : a.position < s.length := by
      have := (axis_eraseIdx a).2; exact (List.getElem?_eq_some_iff.mp this).1
    rw [List.length_eraseIdx_of_lt hp] at this; omega
  apply gather_flat hx _ _ (expandPicks_length _ _ _)
  intro i
  simp only [expandPicks, List.getElem?_map, List.getElem?_range (flat_lt i), Option.map_some,
    linearToIndex_flat, hrank, Nat.succ_ne_self, if_false, ← coords_split, indexToLinear_coords]

/-- Same-rank `Expand` (decoded `expand pre post n`) gathers coordinate `0` of the singleton
axis, which is `opEval (.expand pre post n)`. -/
theorem expand_same_rank_flat (pre post : Shape) (n : Nat) {xs : List (BinaryFloat f)}
    {x : Tensor (BinaryFloat f) (pre ++ 1 :: post)}
    (hx : Tensor.ofList (pre ++ 1 :: post) xs = some x) :
    Tensor.ofList (pre ++ n :: post)
        ((expandPicks (pre ++ 1 :: post) (pre ++ n :: post) pre.length).map (at' xs)) =
      some (fun i => x ((Axis.atPosition pre 1 post).join
        (0, ((Axis.atPosition pre n post).split i).2))) := by
  apply gather_flat hx _ _ (expandPicks_length _ _ _)
  intro i
  have hlen : (pre ++ n :: post).length = (pre ++ 1 :: post).length := by simp
  have hk : pre.length < (coordsOf i).length := by rw [length_coords]; simp
  simp only [expandPicks, List.getElem?_map, List.getElem?_range (flat_lt i), Option.map_some,
    linearToIndex_flat, hlen, if_true]
  congr 1
  rw [← indexToLinear_coords, coords_join, coords_split, atPosition_position,
    atPosition_position]
  simp only [Fin.val_zero]
  rw [insertAt_eraseIdx _ _ _ hk]

/-- `reshape` moves the buffer unchanged: inserting a singleton axis (decoded `insert a`,
`a : Axis s 1 t`) reads the same flat buffer as `opEval (.insert a)`. -/
theorem reshape_insert_flat {s t : Shape} (a : Axis s 1 t) {xs : List (BinaryFloat f)}
    {x : Tensor (BinaryFloat f) t} (hx : Tensor.ofList t xs = some x) :
    Tensor.ofList s xs = some (fun i => x (a.split i).2) := by
  have hl : xs.length = s.prod := by rw [ofList_length hx, Axis.prod_eq a, one_mul]
  apply ofList_eq_some_of hl
  intro i
  have h1 : (a.split i).2.flat = i.flat := flat_split_singleton a i
  obtain ⟨hi, hxi⟩ := ofList_apply hx (a.split i).2
  rw [hxi, List.getElem?_eq_some_iff]
  exact ⟨by rw [← h1]; exact hi, by simp only [h1]⟩

/-- `reshape` removing a singleton axis (decoded `sum a`, `a : Axis s 1 t`) reads the same
flat buffer as `opEval (.sum a)`: a one-element group is returned without an addition. -/
theorem reshape_remove_flat {s t : Shape} (a : Axis s 1 t) {xs : List (BinaryFloat f)}
    {x : Tensor (BinaryFloat f) s} (hx : Tensor.ofList s xs = some x) :
    Tensor.ofList t xs = some (fun i => pairFold (sumLeaves a x i)) := by
  have hl : xs.length = t.prod := by rw [ofList_length hx, Axis.prod_eq a, one_mul]
  apply ofList_eq_some_of hl
  intro i
  have hflat : (a.join (0, i)).flat = i.flat := by
    have := flat_split_singleton a (a.join (0, i))
    rw [Axis.split_join] at this; exact this.symm
  obtain ⟨hi, hxi⟩ := ofList_apply hx (a.join (0, i))
  have hv : pairFold (sumLeaves a x i) = x (a.join (0, i)) := by
    simp [sumLeaves, pairFold, pairFoldN]
  rw [hv, hxi, List.getElem?_eq_some_iff]
  exact ⟨by rw [← hflat]; exact hi, by simp only [hflat]⟩

end ChelisBridge.Numeric
