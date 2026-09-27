import IteratorConsumerModels

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def vec {α : Type} (l : List α) : alloc.vec.Vec α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩
def slice {α : Type} (l : List α) : Slice α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩
def u (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def z (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩
def pairs (v : alloc.vec.Vec (U32 × U32)) : List (Nat × Nat) := v.val.map fun (a, b) => (a.val, b.val)
def vpairs (l : List (Nat × Nat)) : alloc.vec.Vec (U32 × U32) := vec (l.map fun (a, b) => (u a, u b))

open iterator_consumer_models

def definition (eager : Bool) (item_index : Nat) (label : String) : Definition :=
  { eager, item_index := u item_index, «name» := label }

#eval do
  expect "dedup_names" (decode pairs (dedup_names (vpairs [(3, 1), (1, 2), (3, 0), (1, 1), (2, 5)])))
    (.ok [(1, 1), (2, 5), (3, 0)])
  expect "dedup_names, single" (decode pairs (dedup_names (vpairs [(7, 7)]))) (.ok [(7, 7)])
  -- the key closure changes `current` and the kept `prev`; its captured count is given back
  let marking := fun ((calls, v) : U32 × alloc.vec.Vec (U32 × U32)) => (calls.val, pairs v)
  expect "dedup_marking"
    (decode marking (dedup_marking (vpairs [(11, 0), (12, 0), (25, 0), (21, 0), (30, 0)])))
    (.ok (8, [(11, 2), (25, 3), (30, 1)]))
  expect "dedup_marking, single" (decode marking (dedup_marking (vpairs [(1, 0)]))) (.ok (0, [(1, 0)]))
  let names := slice ["a", "b", "a", "c"]
  let idx := fun (o : Option Usize) => o.map fun (x : Usize) => x.val
  expect "last_match a" (decode idx (last_match names (toStr "a"))) (.ok (some 2))
  expect "last_match z" (decode idx (last_match names (toStr "z"))) (.ok none)
  let id := fun (o : Option U32) => o.map fun (x : U32) => x.val
  expect "highest_id" (decode id (highest_id (slice [vec [u 3, u 9], vec [], vec [u 4]]))) (.ok (some 10))
  expect "highest_id, empty blocks" (decode id (highest_id (slice [vec [], vec []]))) (.ok none)
  expect "max_pair"
    (decode (fun (o : Option (U32 × U32)) => o.map fun ((a, b) : U32 × U32) => (a.val, b.val))
      (max_pair (slice [(u 2, u 1), (u 0, u 9), (u 2, u 0), (u 1, u 5)])))
    (.ok (some (2, 1)))
  let near := fun (o : Option Str) => o.map fun (s : Str) =>
    match Str.toString s with | .ok v => v | _ => "<invalid>"
  expect "nearest 3"
    (decode near (nearest (slice [toStr "f32", toStr "f64", toStr "bf16", toStr "i8"]) (z 3)))
    (.ok (some "f32"))
  expect "nearest 1"
    (decode near (nearest (slice [toStr "float64", toStr "i8", toStr "u8"]) (z 1))) (.ok (some "i8"))
  expect "nearest, none" (decode near (nearest (slice []) (z 1))) (.ok none)
  let defs := slice [definition true 4 "b", definition false 0 "a", definition true 4 "a",
    definition true 9 "c"]
  expect "eager_start" (decode idx (eager_start defs (slice [z 0, z 1, z 2, z 3]))) (.ok (some 2))
  expect "eager_start, none" (decode idx (eager_start defs (slice [z 1]))) (.ok none)
  let dist := fun ((a, b) : U32 × U64) => (a.val, b.val)
  expect "distances" (decode dist (distances (u 3) (u 10) ⟨BitVec.ofNat _ (2 ^ 64 - 1)⟩ ⟨0⟩))
    (.ok (7, 2 ^ 64 - 1))
  expect "distances, reversed and equal" (decode dist (distances (u 10) (u 3) ⟨5⟩ ⟨5⟩)) (.ok (7, 0))
  expect "distances, u32::MAX" (decode dist (distances (u 0) (u (2 ^ 32 - 1)) ⟨0⟩ ⟨1⟩))
    (.ok (2 ^ 32 - 1, 1))
  expect "element_count" (decode (fun (x : Usize) => x.val) (element_count (slice [z 2, z 3, z 4]))) (.ok 24)
  expect "element_count, empty" (decode (fun (x : Usize) => x.val) (element_count (slice []))) (.ok 1)
  expect "element_count, zero" (decode (fun (x : Usize) => x.val) (element_count (slice [z 5, z 0]))) (.ok 1)
