import CollectionMiscModels

open Aeneas Aeneas.Std

namespace CollectionMiscModelsChecks

/-! Mirrors the native tests of `collection_misc_models.rs`. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def nat {ty : UScalarTy} (x : UScalar ty) : Nat := x.val
def int {ty : IScalarTy} (x : IScalar ty) : Int := x.val

def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def u64 (n : Nat) : U64 := ⟨BitVec.ofNat _ n⟩

def slice {α : Type} (l : List α) : Slice α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def vec {α : Type} (l : List α) : alloc.vec.Vec α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

open collection_misc_models

#eval do
  expect "arc value" (decode nat (arc_value (vec ([1, 2, 3].map u32)))) (.ok 3)
  expect "arc default" (decode nat arc_default_len) (.ok 0)
  expect "least" (decode (Option.map nat) (least (slice ([5, 3, 9, 3].map u32)))) (.ok (some 3))
  expect "least empty" (decode (Option.map nat) (least (slice []))) (.ok none)

#eval do
  expect "refresh" (decode id (refresh (some "old") (some "new"))) (.ok (some "new"))
  expect "refresh none" (decode id (refresh (some "new") none)) (.ok none)
  expect "refresh again" (decode id (refresh none (some "again"))) (.ok (some "again"))
  let two := fun ((a, b) : Option U64 × Option U64) => (a.map nat, b.map nat)
  expect "last_of 7" (decode two (last_of (some (u64 7)))) (.ok (some 7, none))
  expect "last_of none" (decode two (last_of none)) (.ok (none, none))

#eval do
  let two := fun ((a, b) : Usize × Usize) => (a.val, b.val)
  expect "empty defaults" (decode two empty_defaults) (.ok (0, 0))
  let vals := fun (v : alloc.vec.Vec U32) => v.val.map nat
  expect "extend" (decode vals (extend_from (vec [u32 1]) (slice ([2, 3].map u32)))) (.ok [1, 2, 3])
  expect "extend empty" (decode vals (extend_from (vec ([1, 2, 3].map u32)) (slice []))) (.ok [1, 2, 3])

end CollectionMiscModelsChecks
