import CollectionStdModels

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def strings : alloc.collections.btree.map.BTreeMap U32 String := ⟨[(1#u32, "a"), (3#u32, "c")]⟩
def vec {α : Type} (l : List α) (h : l.length ≤ Usize.max := by scalar_tac) :
    alloc.vec.Vec α := ⟨l, h⟩
def vals (v : alloc.vec.Vec U32) : List Nat := v.val.map (·.val)
def pairs (m : alloc.collections.btree.map.BTreeMap U32 U32) : List (Nat × Nat) :=
  m.entries.map fun (k, v) => (k.val, v.val)
def members (s : alloc.collections.btree.set.BTreeSet U32) : List Nat :=
  s.map.entries.map fun (k, _) => k.val

#eval do
  expect "get_key_value, found"
    (decode (fun (o : Option (U32 × String)) => o.map fun ((k, v) : U32 × String) => (k.val, v)) (collection_std_models.entry_of strings 3#u32))
    (.ok (some (3, "c")))
  expect "get_key_value, absent"
    (decode (fun (o : Option (U32 × String)) => o.map fun ((k, v) : U32 × String) => (k.val, v)) (collection_std_models.entry_of strings 2#u32))
    (.ok none)
  expect "get_key_value in the empty map"
    (decode (fun (o : Option (U32 × String)) => o.map fun ((k, v) : U32 × String) => (k.val, v)) (collection_std_models.entry_of ⟨[]⟩ 1#u32))
    (.ok none)
  expect "clear, the given-back map is empty"
    (decode (fun ((n, m) : Usize × alloc.collections.btree.map.BTreeMap U32 U32) => (n.val, pairs m))
      (collection_std_models.cleared ⟨[(1#u32, 10#u32), (2#u32, 20#u32)]⟩)) (.ok (2, []))
  expect "clear of the empty map"
    (decode (fun ((n, m) : Usize × alloc.collections.btree.map.BTreeMap U32 U32) => (n.val, pairs m)) (collection_std_models.cleared ⟨[]⟩))
    (.ok (0, []))
  expect "BTreeSet len, then clear: the given-back set is empty"
    (decode (fun ((n, s) : Usize × alloc.collections.btree.set.BTreeSet U32) => (n.val, members s))
      (collection_std_models.set_len_then_clear ⟨⟨[(4#u32, ()), (7#u32, ())]⟩⟩)) (.ok (2, []))
  expect "retain with a pure predicate"
    (decode vals (collection_std_models.keep_even (vec [1#u32, 2#u32, 3#u32, 4#u32, 6#u32, 7#u32])))
    (.ok [2, 4, 6])
  expect "retain of nothing" (decode vals (collection_std_models.keep_even (vec []))) (.ok [])
  expect "split_off: the tail and the given-back head"
    (decode (fun ((t, h) : alloc.vec.Vec U32 × alloc.vec.Vec U32) => (vals t, vals h))
      (collection_std_models.split_tail (vec [1#u32, 2#u32, 3#u32, 4#u32]) 1#usize))
    (.ok ([2, 3, 4], [1]))
  expect "split_off at the length"
    (decode (fun ((t, h) : alloc.vec.Vec U32 × alloc.vec.Vec U32) => (vals t, vals h))
      (collection_std_models.split_tail (vec [1#u32, 2#u32]) 2#usize))
    (.ok ([], [1, 2]))
  expect "split_off past the length panics"
    (decode (fun ((t, h) : alloc.vec.Vec U32 × alloc.vec.Vec U32) => (vals t, vals h))
      (collection_std_models.split_tail (vec [1#u32]) 2#usize))
    (.error "failure")
  expect "Default for BTreeMap, then insert"
    (decode pairs (collection_std_models.default_map 3#u32)) (.ok [(3, 4)])
  expect "BTreeSet clone and equal sets"
    (decode (fun ((c, e) : alloc.collections.btree.set.BTreeSet U32 × Bool) => (members c, e))
      (collection_std_models.cloned_set_equal ⟨⟨[(1#u32, ()), (5#u32, ())]⟩⟩ ⟨⟨[(1#u32, ()), (5#u32, ())]⟩⟩))
    (.ok ([1, 5], true))
  expect "BTreeSet different elements"
    (decode (fun ((c, e) : alloc.collections.btree.set.BTreeSet U32 × Bool) => (members c, e))
      (collection_std_models.cloned_set_equal ⟨⟨[(1#u32, ()), (5#u32, ())]⟩⟩ ⟨⟨[(1#u32, ()), (6#u32, ())]⟩⟩))
    (.ok ([1, 5], false))
  expect "BTreeSet different lengths"
    (decode (fun ((c, e) : alloc.collections.btree.set.BTreeSet U32 × Bool) => (members c, e))
      (collection_std_models.cloned_set_equal ⟨⟨[(1#u32, ())]⟩⟩ ⟨⟨[(1#u32, ()), (5#u32, ())]⟩⟩))
    (.ok ([1], false))
  expect "Default for Box" (decode (fun (o : Option U32) => o.map (fun (n : U32) => n.val)) collection_std_models.default_box) (.ok none)
