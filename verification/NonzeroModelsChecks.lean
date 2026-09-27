import NonzeroModels

open Aeneas Aeneas.Std

namespace NonzeroModelsChecks

/-! Mirrors the native tests of `nonzero_models.rs`, including the hasher given back through its
    mutable borrow. -/

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

open nonzero_models

#eval do
  expect "compare equal" (decode id (compare (u32 7) (u32 7))) (.ok (some (true, false)))
  expect "compare differ" (decode id (compare (u32 7) (u32 8))) (.ok (some (false, true)))
  expect "compare zero" (decode id (compare (u32 0) (u32 8))) (.ok none)

#eval do
  let logged := fun ((b, log) : Bool × Log) => (b, log.bytes.val.map (fun (b : U8) => b.val), log.writes.val)
  let log : Log := { bytes := vec [⟨BitVec.ofNat _ 9⟩], writes := u32 1 }
  expect "hash" (decode logged (hash_issue (u32 0x01020304) log)) (.ok (true, [9, 4, 3, 2, 1], 2))
  let log : Log := { bytes := vec ([9, 4, 3, 2, 1].map fun n => ⟨BitVec.ofNat _ n⟩), writes := u32 2 }
  expect "hash zero" (decode logged (hash_issue (u32 0) log)) (.ok (false, [9, 4, 3, 2, 1], 2))

#eval do
  expect "show" (decode id («show» (u32 42))) (.ok (some "IssueRef(42)|   42|IssueRef(\n    42,\n)"))
  expect "show zero" (decode id («show» (u32 0))) (.ok none)

end NonzeroModelsChecks
