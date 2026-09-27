import CollectionConvertModels

open Aeneas Aeneas.Std

namespace CollectionConvertModelsChecks

/-! Mirrors the native tests of `collection_convert_models.rs`, including the sets, vectors and
    counters given back through mutable borrows. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def nat {ty : UScalarTy} (x : UScalar ty) : Nat := x.val

def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def u64 (n : Nat) : U64 := ⟨BitVec.ofNat _ n⟩
def us (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩

def slice {α : Type} (l : List α) : Slice α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def vec {α : Type} (l : List α) : alloc.vec.Vec α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def vals {α β : Type} (f : α → β) (l : { l : List α // l.length ≤ Usize.max }) : List β :=
  l.val.map f

/-- A set of `u32` from its elements in ascending order. -/
def set (l : List Nat) : alloc.collections.btree.set.BTreeSet U32 :=
  ⟨⟨l.map fun n => (u32 n, ())⟩⟩

def elems (s : alloc.collections.btree.set.BTreeSet U32) : List Nat := s.map.entries.map (·.1.val)

open collection_convert_models

#eval do
  expect "drop_vertex" (decode elems (drop_vertex (set [1, 4, 7, 9]) (u32 9))) (.ok [1, 4, 7])
  let kept := fun ((s, calls, seen) : alloc.collections.btree.set.BTreeSet U32 × U32 ×
      alloc.vec.Vec U32) => (elems s, calls.val, vals nat seen)
  expect "keep_even" (decode kept (keep_even (set [2, 3, 5, 6, 8]) (u32 1) (vec [u32 0])))
    (.ok ([2, 6, 8], 6, [0, 2, 3, 5, 6, 8]))

#eval do
  let items : List Keyed := [{ key := u32 3, tag := u32 0 }, { key := u32 1, tag := u32 1 },
    { key := u32 3, tag := u32 2 }, { key := u32 1, tag := u32 3 }]
  let arr : Array Keyed 4#usize := ⟨items, by simp [items]⟩
  let pairs := fun (v : alloc.vec.Vec (U32 × U32)) => vals (fun ((k, t) : U32 × U32) => (k.val, t.val)) v
  expect "set_of" (decode pairs (set_of arr)) (.ok [(1, 3), (3, 2)])
  expect "empty_set" (decode nat empty_set) (.ok 0)

#eval do
  let entries := fun (l : List (Nat × Nat)) => slice (l.map fun (k, v) => (u32 k, u64 v))
  expect "registry_values" (decode (vals nat) (registry_values (entries [(3, 30), (1, 10), (2, 20), (1, 11)])))
    (.ok [11, 20, 30])
  expect "registry_values empty" (decode (vals nat) (registry_values (entries []))) (.ok [])
  let two := fun ((a, b) : Option U64 × Option U64) => (a.map nat, b.map nat)
  expect "first_two_values" (decode two (first_two_values (entries [(5, 50), (4, 40)])))
    (.ok (some 40, some 50))
  expect "first_two_values one" (decode two (first_two_values (entries [(5, 50)])))
    (.ok (some 50, none))

def tracked (id : Nat) : Tracked := { id := u32 id, clones := u32 0, reuses := u32 0 }

def track (t : Tracked) : Nat × Nat × Nat := (t.id.val, t.clones.val, t.reuses.val)

#eval do
  let arr := fun (r : core.result.Result (Array U32 3#usize) (alloc.vec.Vec U32)) =>
    match r with
    | .Ok a => Sum.inl (a.val.map nat)
    | .Err v => Sum.inr (vals nat v)
  expect "triple" (decode arr (triple (vec ([1, 2, 3].map u32)))) (.ok (.inl [1, 2, 3]))
  expect "triple short" (decode arr (triple (vec ([1, 2].map u32)))) (.ok (.inr [1, 2]))
  expect "triple long" (decode arr (triple (vec ([1, 2, 3, 4].map u32)))) (.ok (.inr [1, 2, 3, 4]))
  expect "owned" (decode (vals track) (owned (slice [tracked 1, tracked 2])))
    (.ok [(1, 1, 0), (2, 1, 0)])
  expect "refresh shorter" (decode (vals track)
    (refresh (slice [tracked 1, tracked 2]) (vec [tracked 7, tracked 8, tracked 9])))
    (.ok [(1, 0, 1), (2, 0, 1)])
  expect "refresh longer" (decode (vals track)
    (refresh (slice [tracked 1, tracked 2, tracked 3]) (vec [tracked 7])))
    (.ok [(1, 0, 1), (2, 1, 0), (3, 1, 0)])
  expect "borrowed_total" (decode nat (borrowed_total (vec ([4, 5, 6].map us)))) (.ok 15)

end CollectionConvertModelsChecks
