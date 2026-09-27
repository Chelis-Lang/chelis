import ReverseModels

open Aeneas Aeneas.Std

namespace ReverseModelsChecks

/-! Mirrors the native tests of `reverse_models.rs`: the stable sorts keyed by `Reverse` (alone,
    last in a tuple key, from a map lookup that panics on a missing key) and the comparison
    operators of `Reverse`, including which argument `max` and `min` return on equal keys and the
    panic of `clamp` on reversed bounds. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def us (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩

def vec {α : Type} (l : List α) : alloc.vec.Vec α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

open reverse_models

def placed (point : Placement) (order owner : Nat) : Placed :=
  { point, definition_order := us order, owner := u32 owner }

def sample : alloc.vec.Vec Placed :=
  let e := fun n => Placement.OnEdge (u32 n)
  let b := fun n m => Placement.InBlock (u32 n) (u32 m)
  vec [placed (e 2) 1 10, placed (b 0 3) 4 11, placed (e 2) 4 12, placed (b 0 1) 1 13,
    placed (b 0 3) 7 14, placed (e 1) 4 15]

def owners (v : alloc.vec.Vec Placed) : List Nat := v.val.map (·.owner.val)

def map (l : List (Nat × Nat)) : alloc.collections.btree.map.BTreeMap U32 Usize :=
  ⟨l.map fun (k, v) => (u32 k, us v)⟩

#eval do
  expect "latest_first" (decode owners (latest_first sample)) (.ok [14, 11, 12, 15, 10, 13])
  expect "by_point_then_latest" (decode owners (by_point_then_latest sample))
    (.ok [13, 14, 11, 15, 12, 10])
  expect "latest_first empty" (decode owners (latest_first (vec []))) (.ok [])
  let definitions := map [(1, 5), (2, 9), (3, 5), (4, 0)]
  expect "owners_latest_first" (decode (fun (v : alloc.vec.Vec U32) => v.val.map UScalar.val)
    (owners_latest_first definitions (vec [u32 4, u32 1, u32 2, u32 3]))) (.ok [2, 1, 3, 4])
  expect "missing_definition" (decode (fun (v : alloc.vec.Vec U32) => v.val.map UScalar.val)
    (owners_latest_first (map [(1, 5)]) (vec [u32 1, u32 7]))) (.error "panic")

def ord : Ordering → Int
  | .lt => -1
  | .eq => 0
  | .gt => 1

def rel := fun ((p, a, b, c, d, o) : Option Ordering × Bool × Bool × Bool × Bool × Ordering) =>
  (p.map ord, [a, b, c, d], ord o)

#eval do
  expect "relations 3 5" (decode rel (relations (u32 3) (u32 5)))
    (.ok (some 1, [false, false, true, true], 1))
  expect "relations 5 3" (decode rel (relations (u32 5) (u32 3)))
    (.ok (some (-1), [true, true, false, false], -1))
  expect "relations 4 4" (decode rel (relations (u32 4) (u32 4)))
    (.ok (some 0, [false, true, false, true], 0))
  expect "equalities 4 4" (decode id (equalities (u32 4) (u32 4))) (.ok (true, false))
  expect "equalities 4 5" (decode id (equalities (u32 4) (u32 5))) (.ok (false, true))

#eval do
  let pairs := fun (((a, x), (b, y)) : (U32 × Bool) × (U32 × Bool)) => ((a.val, x), (b.val, y))
  expect "extremes 3 5" (decode pairs (extremes (u32 3, true) (u32 5, false)))
    (.ok ((3, true), (5, false)))
  expect "extremes 5 3" (decode pairs (extremes (u32 5, true) (u32 3, false)))
    (.ok ((3, false), (5, true)))
  expect "extremes 4 4" (decode pairs (extremes (u32 4, true) (u32 4, false)))
    (.ok ((4, false), (4, true)))
  expect "clamped 1" (decode UScalar.val (clamped (u32 1) (u32 9) (u32 5))) (.ok 5)
  expect "clamped 7" (decode UScalar.val (clamped (u32 7) (u32 9) (u32 5))) (.ok 7)
  expect "clamped 12" (decode UScalar.val (clamped (u32 12) (u32 9) (u32 5))) (.ok 9)
  expect "clamp_bounds" (decode UScalar.val (clamped (u32 7) (u32 5) (u32 9))) (.error "panic")

end ReverseModelsChecks
