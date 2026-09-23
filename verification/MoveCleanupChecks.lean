import MoveCleanup

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode : Result (alloc.vec.Vec U32) → Except String (List Nat)
  | .ok values => .ok (values.val.map UScalar.val)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def decodePair : Result (U32 × U32) → Except String (Nat × Nat)
  | .ok (first, second) => .ok (first.val, second.val)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

#eval do
  expect "first moved payload retains its buffer and mutation"
    (decode (move_cleanup.append_selected true)) (.ok [17, 99])
  expect "second moved payload retains its buffer and mutation"
    (decode (move_cleanup.append_selected false)) (.ok [29, 99])
  expect "success cleanup drops the default, not the returned buffer"
    (decode (move_cleanup.append_or_default true)) (.ok [17, 99])
  expect "error cleanup drops the error payload, not the default buffer"
    (decode (move_cleanup.append_or_default false)) (.ok [41, 99])
  expect "last direct copy preserves an indirect reader"
    (decodePair (move_cleanup.copy_with_live_borrow core.marker.CopyU32 37#u32))
    (.ok (37, 37))
  expect "a mutable receiver retains the stored borrow after the lender is copied"
    (decodePair (move_cleanup.copy_after_borrowed_push 41#u32)) (.ok (41, 41))
  IO.println "Post-move cleanup smoke: 6 behavioral checks passed"
