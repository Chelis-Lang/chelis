import CapacityStdModels

open Aeneas Aeneas.Std

namespace CapacityStdModelsChecks

/-! Mirrors the native tests of `capacity_std_models.rs` on its extraction. -/

def expect [DecidableEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual = expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def slice {α} (l : List α) : Slice α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def pairs (l : List (Nat × Nat)) : Slice (U32 × U32) :=
  slice (l.map fun (k, t) => (⟨BitVec.ofNat 32 k⟩, ⟨BitVec.ofNat 32 t⟩))

def tags (l : List capacity_std_models.Tagged) : List (Nat × Nat) :=
  l.map fun item => (item.key.val, item.tag.val)

#eval do
  expect "dedup keeps the first of each run"
    (decode (fun v : alloc.vec.Vec _ => tags v.val) (capacity_std_models.dedup_runs
      (pairs [(1, 10), (1, 11), (2, 20), (1, 12), (1, 13), (3, 30), (3, 31)])))
    (.ok [(1, 10), (2, 20), (1, 12), (3, 30)])
  expect "dedup of nothing"
    (decode (fun v : alloc.vec.Vec _ => tags v.val) (capacity_std_models.dedup_runs (pairs [])))
    (.ok [])
  expect "dedup of one element"
    (decode (fun v : alloc.vec.Vec _ => tags v.val) (capacity_std_models.dedup_runs
      (pairs [(5, 50)]))) (.ok [(5, 50)])
  expect "dedup of one run"
    (decode (fun v : alloc.vec.Vec _ => tags v.val) (capacity_std_models.dedup_runs
      (pairs [(4, 1), (4, 2), (4, 3)]))) (.ok [(4, 1)])
  expect "boxed slices and their clones preserve elements and order"
    (decode (fun ((b, c) : Slice _ × Slice _) => (tags b.val, tags c.val))
      (capacity_std_models.boxed_clone (pairs [(1, 10), (1, 11), (0, 5)])))
    (.ok ([(1, 10), (1, 11), (0, 5)], [(1, 10), (1, 11), (0, 5)]))
  expect "boxed slice iteration is in order"
    (decode (fun (x : U32) => x.val) (capacity_std_models.boxed_digits
      (slice ([3, 1, 4, 1, 5].map fun d => (⟨BitVec.ofNat 32 d⟩ : U32))))) (.ok 31415)
  expect "empty boxed slice iteration"
    (decode (fun (x : U32) => x.val) (capacity_std_models.boxed_digits (slice []))) (.ok 0)
  let unsigned (v : I64) := decode (Option.map fun (x : U64) => x.val) (capacity_std_models.to_unsigned v)
  expect "u64::try_from(0)" (unsigned 0#i64) (.ok (some 0))
  expect "u64::try_from(i64::MAX)" (unsigned 9223372036854775807#i64)
    (.ok (some 9223372036854775807))
  expect "u64::try_from(-1)" (unsigned (-1)#i64) (.ok none)
  expect "u64::try_from(i64::MIN)" (unsigned (-9223372036854775808)#i64) (.ok none)
  let lifted (v : core.result.Result U64 U8) :=
    decode (fun (r : core.result.Result (Option U64) U8) => match r with
      | .Ok o => Sum.inl (o.map fun (x : U64) => x.val) | .Err e => Sum.inr e.val)
      (capacity_std_models.lift_some v)
  expect "Ok(7).map(Some)" (lifted (.Ok 7#u64)) (.ok (Sum.inl (some 7)))
  expect "Err(3).map(Some)" (lifted (.Err 3#u8)) (.ok (Sum.inr 3))
  expect "set insert keeps the stored equal element; remove reports presence"
    (decode (fun ((r, s) : Array Bool 6#usize × alloc.collections.btree.set.BTreeSet _) =>
      (r.val, tags (s.map.entries.map Prod.fst))) capacity_std_models.set_script)
    (.ok ([true, false, true, true, false, true], [(2, 20), (3, 30)]))
  expect "borrowed tuple visiting set"
    (decode (fun (a : Array Bool 4#usize) => a.val) (capacity_std_models.visiting_script 7#u32 2#usize))
    (.ok [true, false, true, false])
  IO.println "capacity_std_models: native and Lean agree"

end CapacityStdModelsChecks
