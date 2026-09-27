import StringWriteModels

open Aeneas Aeneas.Std

namespace StringWriteModelsChecks

/-! Mirrors the native tests of `string_write_models.rs`, including the strings given back
    through their mutable borrows (after a failing `Display`, the text written before it). -/

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

open string_write_models

#eval do
  let units : List string_write_models.Unit := [
    { «name» := "main", kind := .Entry,
      blocks := vec [{ id := u32 0, ops := vec ["move %1", "drop %2"] },
        { id := u32 1, ops := vec [] }] },
    { «name» := "h", kind := .Helper (u32 2), blocks := vec [] }]
  expect "render" (decode id (render (vec units)))
    (.ok "unit main Entry\n  b0 (2):\n    move %1\n    drop %2\n  b1 (0):\nunit h Helper { arity: 2 }\n")
  expect "render empty" (decode id (render (vec []))) (.ok "")

#eval do
  expect "append" (decode id (append "x" (u32 7) (toStr "abc")))
    (.ok (true, "x[   7|7  |007|  abc  |ab]done"))
  expect "flaky ok" (decode id (write_flaky "s" false)) (.ok (true, "s<pre-ok>7"))
  expect "flaky fail" (decode id (write_flaky "s<pre-ok>7" true)) (.ok (false, "s<pre-ok>7<pre"))
  expect "pieces" (decode id (pieces "z")) (.ok ((true, true), "zabé"))

end StringWriteModelsChecks
