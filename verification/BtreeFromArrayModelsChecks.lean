import BtreeFromArrayModels

open Aeneas Aeneas.Std

namespace BtreeFromArrayModelsChecks

/-! Mirrors the native tests of `btree_from_array_models.rs`: `BTreeMap::from` of a one-pair array
    followed by a lookup and an insertion (as `verify_unit`), arrays of three pairs (sorted by
    key; of equal keys the last pair stays, key and value), the empty array, and keys compared by
    their first field only (the kept key is the last of its run). -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩

def vec {α : Type} (l : List α) : alloc.vec.Vec α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def pairs (v : alloc.vec.Vec (U32 × U32)) : List (Nat × Nat) := v.val.map fun (a, b) => (a.val, b.val)

open btree_from_array_models

#eval do
  let sets := fun (v : alloc.vec.Vec (U32 × alloc.vec.Vec U32)) =>
    v.val.map fun ((id, set) : U32 × alloc.vec.Vec U32) => (id.val, set.val.map UScalar.val)
  expect "incoming 4 2" (decode sets (incoming (u32 4) (vec [u32 3, u32 1, u32 3]) (u32 2)))
    (.ok [(2, [1, 3]), (4, [1, 3])])
  expect "incoming 4 4" (decode sets (incoming (u32 4) (vec []) (u32 4))) (.ok [(4, [])])

#eval do
  let p := fun (a b : Nat) => (u32 a, u32 b)
  expect "three sorted" (decode pairs (three (p 3 30) (p 1 10) (p 2 20)))
    (.ok [(1, 10), (2, 20), (3, 30)])
  expect "three last wins" (decode pairs (three (p 2 1) (p 1 5) (p 2 7))) (.ok [(1, 5), (2, 7)])
  expect "three equal" (decode pairs (three (p 5 1) (p 5 2) (p 5 3))) (.ok [(5, 3)])
  expect "empty" (decode UScalar.val empty) (.ok 0)

#eval do
  let k := fun (a b v : Nat) => ((u32 a, u32 b), u32 v)
  let triples := fun (v : alloc.vec.Vec (U32 × U32 × U32)) =>
    v.val.map fun ((a, b, c) : U32 × U32 × U32) => (a.val, b.val, c.val)
  expect "keys runs" (decode triples
      (keys (Array.make 4#usize [k 2 0 10, k 1 1 11, k 2 2 12, k 1 3 13])))
    (.ok [(1, 3, 13), (2, 2, 12)])
  expect "keys distinct" (decode triples
      (keys (Array.make 4#usize [k 9 0 1, k 8 1 2, k 7 2 3, k 6 3 4])))
    (.ok [(6, 3, 4), (7, 2, 3), (8, 1, 2), (9, 0, 1)])

end BtreeFromArrayModelsChecks
