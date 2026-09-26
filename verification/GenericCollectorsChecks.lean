import GenericCollectors
import Aeneas.Std.LawfulIterator

open Aeneas Aeneas.Std

namespace GenericCollectorsChecks

/-! Mirrors the native tests of `generic_collectors.rs`: generic collectors
    whose deferred iterator obligation is discharged by `vec::IntoIter` or by
    a crate iterator that defines only `next`. -/

def expect [DecidableEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual = expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def mapView (m : alloc.collections.btree.map.BTreeMap generic_collectors.Tagged U32) :
    List (Nat × Nat × Nat) :=
  m.entries.map fun (k, v) => (k.key.val, k.tag.val, v.val)

def setView (s : alloc.collections.btree.set.BTreeSet generic_collectors.Tagged) :
    List (Nat × Nat) :=
  s.map.entries.map fun (k, _) => (k.key.val, k.tag.val)

-- The instance that discharges the obligations is lawful.
example := core.iter.traits.iterator.LawfulIterator.vecIntoIter

#eval do
  expect "collect keeps the last Ord-equal entry (stable sort, keep-last dedup)"
    (decode mapView generic_collectors.map_collected)
    (.ok [(1, 11, 101), (2, 21, 201), (3, 30, 300)])
  expect "set collect keeps the last Ord-equal element"
    (decode setView generic_collectors.set_collected) (.ok [(1, 11), (2, 21), (3, 30)])
  expect "extend keeps the first key object and the last value"
    (decode mapView generic_collectors.map_extended)
    (.ok [(1, 10, 101), (2, 20, 201), (3, 30, 300)])
  expect "set extend keeps the first element"
    (decode setView generic_collectors.set_extended) (.ok [(1, 10), (2, 20), (3, 30)])
  expect "a next-only crate source keeps the last Ord-equal element"
    (decode setView generic_collectors.custom_source) (.ok [(0, 0), (1, 2)])
  expect "Vec collection of a next-only crate source keeps every item in order"
    (decode (fun (items : alloc.vec.Vec generic_collectors.Tagged) =>
      items.val.map fun (k : generic_collectors.Tagged) => (k.key.val, k.tag.val))
      generic_collectors.custom_vec)
    (.ok [(1, 3), (1, 2), (0, 1), (0, 0)])
  IO.println "generic_collectors: native and Lean agree"

end GenericCollectorsChecks
