import BtreeSets
import Aeneas.Std.BTreeSet

open Aeneas Aeneas.Std

namespace BTreeSetsChecks

open alloc.collections.btree.set

def expect [DecidableEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual = expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def natEq : core.cmp.PartialEq Nat Nat := { eq := fun a b => .ok (a == b) }
def natOrd : core.cmp.Ord Nat := {
  eqInst := { partialEqInst := natEq }
  partialOrdInst := { partialEqInst := natEq, partial_cmp := fun a b => .ok (some (compare a b)) }
  cmp := fun a b => .ok (compare a b)
}

def stream : core.iter.traits.collect.IntoIterator
    (List (Option Nat) × Nat) Nat (List (Option Nat) × Nat) := {
  iteratorInst := { next := fun (items, polls) => match items with
    | [] => .ok (none, ([], polls + 1))
    | item :: rest => .ok (item, (rest, polls + 1)) }
  into_iter := fun (items, polls) => .ok ((items, polls + 10), id)
}

def collect (ord : core.cmp.Ord Nat) (items : List (Option Nat)) :=
  FromIteratorBTreeSet.from_iter ord stream (items, 0)

def failedCmp : core.cmp.Ord Nat := { natOrd with cmp := fun _ _ => .fail .panic }
def failedLt : core.cmp.Ord Nat := {
  natOrd with partialOrdInst := {
    natOrd.partialOrdInst with lt := fun _ _ => .fail .panic }
}
def failedNe : core.cmp.Ord Nat := {
  natOrd with eqInst := { partialEqInst := { natEq with ne := fun _ _ => .fail .panic } }
}

def keys {T : Type} (set : BTreeSet T) : List T := set.map.entries.map Prod.fst

set_option synthInstance.maxSize 512 in
#eval do
  expect "stable sort retains the final equal key, not the first key"
    (decode (fun (set : BTreeSet btree_sets.Key) =>
      (keys set).map fun (key : btree_sets.Key) => (key.rank.val, key.identity.val))
      btree_sets.duplicate_from_iterator)
    (.ok [(1, 88), (2, 99), (3, 30)])
  expect "heterogeneous lookup covers both ordered boundaries and duplicates"
    (decode id btree_sets.borrowed_lookup) (.ok (false, true, false, true))
  expect "nonfused source stops at first None and retains unconsumed state"
    (decode (fun ((set, state) : BTreeSet Nat × (List (Option Nat) × Nat)) => (keys set, state))
      (collect natOrd [some 2, none, some 99])) (.ok ([2], ([some 99], 12)))
  expect "singleton collection does not invoke comparator"
    (decode (fun ((set, _) : BTreeSet Nat × (List (Option Nat) × Nat)) => keys set)
      (collect failedLt [some 2])) (.ok [2])
  expect "sorting dispatches PartialOrd.lt and propagates failure"
    (decode (fun _ => ()) (collect failedLt [some 2, some 1])) (.error "failure")
  expect "deduplication dispatches PartialEq.ne and propagates failure"
    (decode (fun _ => ()) (collect failedNe [some 2, some 2])) (.error "failure")
  let borrow : core.borrow.Borrow Nat Nat := { borrow := fun n => .ok n }
  let set : BTreeSet Nat := ⟨⟨[(2, ())]⟩⟩
  expect "contains propagates comparison failure"
    (decode id (BTreeSet.contains borrow natOrd failedCmp set 2)) (.error "failure")
  expect "contains propagates Borrow failure"
    (decode id (BTreeSet.contains { borrow := fun _ => .fail .panic } natOrd natOrd set 2))
    (.error "failure")
  let failingSource := { stream with iteratorInst := {
    stream.iteratorInst with next := fun _ => .fail .panic } }
  expect "iterator failure is not mistaken for exhaustion"
    (decode (fun _ => ()) (FromIteratorBTreeSet.from_iter natOrd failingSource ([], 0)))
    (.error "failure")
  let divergingSource := { stream with
    into_iter := fun (_ : List (Option Nat) × Nat) => .div }
  expect "conversion divergence is propagated"
    (decode (fun _ => ()) (FromIteratorBTreeSet.from_iter natOrd divergingSource ([], 0)))
    (.error "divergence")
  IO.println "BTreeSet smoke: 10 behavioral checks passed"

end BTreeSetsChecks
