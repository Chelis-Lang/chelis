import BtreeMaps

open Aeneas Aeneas.Std

def expect [DecidableEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual = expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def triples (values : alloc.vec.Vec (U32 × U32 × U32)) : List (Nat × Nat × Nat) :=
  values.val.map fun (rank, identity, value) => (rank.val, identity.val, value.val)

def pairs (values : alloc.vec.Vec (U32 × U32)) : List (Nat × Nat) :=
  values.val.map fun (key, value) => (key.val, value.val)


set_option synthInstance.maxSize 512 in
#eval do
  let original := [(1, 10, 100), (2, 20, 200), (3, 30, 300)]
  expect "iteration sorts keys" (decode triples btree_maps.ordered_insert) (.ok original)
  expect "equivalent insert returns old value and retains original key"
    (decode (fun ((old, entries) : Option U32 × alloc.vec.Vec (U32 × U32 × U32)) => (old.map UScalar.val, triples entries)) btree_maps.equivalent_replacement)
    (.ok (some 200, [(1, 10, 100), (2, 20, 250), (3, 30, 300)]))
  expect "occupied restoration changes selected value, not key or neighbors"
    (decode triples btree_maps.occupied_restoration)
    (.ok [(1, 10, 100), (2, 20, 211), (3, 30, 300)])
  expect "vacant restoration retains insertion and later reference writes"
    (decode triples btree_maps.vacant_restoration)
    (.ok [(1, 10, 100), (2, 40, 411), (3, 30, 300), (4, 50, 511)])
  expect "dropped occupied and vacant entries preserve the map"
    (decode triples btree_maps.dropped_entries) (.ok original)
  expect "heterogeneous lookup/removal preserve unrelated entries"
    (decode (fun ((found, removed, absent, missing, present, entries) :
        Option U32 × Option U32 × Option U32 × Bool × Bool × alloc.vec.Vec (U32 × U32 × U32)) =>
      ([found.map UScalar.val, removed.map UScalar.val, absent.map UScalar.val], [missing, present], triples entries))
      btree_maps.remove_and_lookup)
    (.ok ([some 200, some 200, none], [false, true], [(1, 10, 100), (3, 30, 300)]))
  expect "lookup before the first key is absent"
    (decode (Option.map UScalar.val) btree_maps.missing_lookup) (.ok none)
  expect "empty and exhausted shared iterators stay exhausted"
    (decode id btree_maps.shared_exhaustion) (.ok (true, true, true, true, true))
  expect "owned iterator returns item then stays exhausted"
    (decode (fun ((first, second, item, exhausted, again) :
        Bool × Bool × Option (U32 × U32) × Bool × Bool) =>
      (first, second, item.map (fun ((key, value) : U32 × U32) => (key.val, value.val)), exhausted, again))
      btree_maps.owned_exhaustion)
    (.ok (true, true, some (1, 10), true, true))
  expect "owned collect preserves sorted order"
    (decode pairs btree_maps.owned_collect) (.ok [(1, 10), (2, 20), (3, 30)])
  expect "shared collect preserves sorted references"
    (decode pairs btree_maps.shared_collect) (.ok [(1, 10), (2, 20), (3, 30)])
  expect "February FromIterator retains last equal key and value"
    (decode triples btree_maps.duplicate_from_iterator)
    (.ok [(1, 88, 150), (2, 99, 250), (3, 30, 300)])
  expect "empty FromIterator produces empty map"
    (decode id btree_maps.empty_from_iterator) (.ok true)
  expect "wrapped entry returns preserve both branches and lazy insertion writes"
    (decode (fun ((calls, entries) : U32 × alloc.vec.Vec (U32 × U32 × U32)) =>
      (calls.val, triples entries)) btree_maps.wrapped_entry_restoration)
    (.ok (1, [(1, 10, 100), (2, 20, 211), (3, 30, 300), (4, 40, 411), (5, 50, 513)]))
  expect "delegated collection restores conversion and iterator mutations"
    (decode (fun ((entries, polls) : alloc.vec.Vec (U32 × U32) × U32) =>
      (pairs entries, polls.val)) (btree_maps.delegated_collection_state 2#u32))
    (.ok ([(1, 10), (2, 20)], 13))
  expect "empty delegated collection still converts and polls once"
    (decode (fun ((entries, polls) : alloc.vec.Vec (U32 × U32) × U32) =>
      (pairs entries, polls.val)) (btree_maps.delegated_collection_state 0#u32))
    (.ok ([], 11))
  expect "String Borrow exposes UTF-8, not Unicode codepoints"
    (decode (fun (bytes : Str) => bytes.val.map UScalar.val) (alloc.str.BorrowString.borrow "é"))
    (.ok [195, 169])
  expect "trusted tree iterators do not evaluate an excess mapped zip item"
    (decode (fun ((sharedCount, sharedCalls, ownedCount, ownedCalls) :
        Usize × U32 × Usize × U32) =>
      (sharedCount.val, sharedCalls.val, ownedCount.val, ownedCalls.val))
      btree_maps.zipped_iterator_callbacks)
    (.ok (1, 1, 1, 1))
  IO.println "BTreeMap smoke: 18 behavioral checks passed"
