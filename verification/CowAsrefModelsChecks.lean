import CowAsrefModels

open Aeneas Aeneas.Std

namespace CowAsrefModelsChecks

/-! Mirrors the native tests of `cow_asref_models.rs`: `as_ref()` of a borrowed and of an owned
    `Cow` of a cloneable node, of a `Cow<str>` and of a `Cow<[u32]>`. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩

def slice {α : Type} (l : List α) : Slice α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def vec {α : Type} (l : List α) : alloc.vec.Vec α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

open cow_asref_models

def expr (head : Nat) (args : List Nat) : Expr := { head := u32 head, args := vec (args.map u32) }

def view (e : Expr) : Nat × List Nat := (e.head.val, e.args.val.map (·.val))

#eval do
  let e := expr 1 [2, 3]
  let sized := fun ((n, e) : U32 × Expr) => (n.val, view e)
  expect "borrowed" (decode sized (bound_size e false)) (.ok (7, (1, [2, 3])))
  expect "owned" (decode sized (bound_size e true)) (.ok (18, (2, [2, 3, 10])))

#eval do
  let texts := fun ((n, s) : Usize × String) => (n.val, s)
  expect "text borrowed" (decode texts (text_len (toStr "héllo") false)) (.ok (6, "héllo"))
  expect "text owned" (decode texts (text_len (toStr "héllo") true)) (.ok (7, "héllo!"))
  expect "slice borrowed" (decode UScalar.val (slice_sum (slice [u32 1, u32 2, u32 3]) false)) (.ok 6)
  expect "slice owned" (decode UScalar.val (slice_sum (slice [u32 1, u32 2, u32 3]) true)) (.ok 21)

end CowAsrefModelsChecks
