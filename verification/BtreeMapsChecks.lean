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
  expect "String Borrow exposes UTF-8, not Unicode codepoints"
    (decode (fun (bytes : Str) => bytes.val.map UScalar.val) (alloc.str.BorrowString.borrow "é"))
    (.ok [195, 169])
  expect "trusted tree iterators do not evaluate an excess mapped zip item"
    (decode (fun ((sharedCount, sharedCalls, ownedCount, ownedCalls) :
        Usize × U32 × Usize × U32) =>
      (sharedCount.val, sharedCalls.val, ownedCount.val, ownedCalls.val))
      btree_maps.zipped_iterator_callbacks)
    (.ok (1, 1, 1, 1))
  expect "keys map and collect into a map in key order"
    (decode pairs btree_maps.keys_collected) (.ok [(1, 10), (2, 20), (3, 30)])
  expect "keys and values iterate in key order with exact hints"
    (decode (fun ((ranks, values, (lower, upper), first, count) :
        alloc.vec.Vec U32 × alloc.vec.Vec U32 × (Usize × Option Usize) × Option U32 × Usize) =>
      (ranks.val.map UScalar.val, values.val.map UScalar.val, lower.val, upper.map UScalar.val,
        first.map UScalar.val, count.val))
      btree_maps.keys_and_values)
    (.ok ([2, 3], [100, 200, 300], 2, some 2, some 10, 3))
  expect "map equality compares lengths, then entries through their PartialEq"
    (decode id btree_maps.maps_equal) (.ok (true, true, false, false))
  expect "a next-only crate source is converted once and polled to its first None"
    (decode (fun ((entries, polls) : alloc.vec.Vec (U32 × U32) × U32) =>
      (pairs entries, polls.val)) (btree_maps.delegated_collection_state 2#u32))
    (.ok ([(1, 10), (2, 20)], 13))
  expect "an empty next-only crate source is polled once"
    (decode (fun ((entries, polls) : alloc.vec.Vec (U32 × U32) × U32) =>
      (pairs entries, polls.val)) (btree_maps.delegated_collection_state 0#u32))
    (.ok ([], 11))
  expect "for over &map visits entries in key order and leaves the map"
    (decode (fun ((entries, empty) : alloc.vec.Vec (U32 × U32 × U32) × Bool) =>
      (triples entries, empty)) btree_maps.borrowed_for_loop)
    (.ok ([(1, 10, 100), (2, 20, 200), (3, 30, 300)], false))
  expect "&map as a generic IntoIterator dictionary"
    (decode UScalar.val btree_maps.borrowed_generic_total) (.ok 1400)
  expect "&map as the IntoIterator dictionary of generic loops"
    (decode (fun ((total, count, empty) : U32 × U32 × Bool) => (total.val, count.val, empty))
      btree_maps.generic_loop_totals)
    (.ok (1400, 3, false))
  expect "std Entry methods run a vacant entry's closure, keep stored keys and write back"
    (decode (fun ((calls, stored, (removed, value), entries) :
        U32 × U32 × (U32 × U32) × alloc.vec.Vec (U32 × U32 × U32)) =>
      (calls.val, stored.val, (removed.val, value.val), triples entries))
      btree_maps.std_entry_methods)
    (.ok (1, 30, (60, 600), [(1, 10, 112), (2, 20, 205), (3, 30, 300), (4, 40, 4), (5, 50, 501)]))
  IO.println "BTreeMap smoke: 27 behavioral checks passed"
