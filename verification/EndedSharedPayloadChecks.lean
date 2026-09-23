import EndedSharedPayload

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

#eval do
  let node : ended_shared_payload.Node := {
    op := .Second
    values := ⟨[7#u32, 11#u32], by scalar_tac⟩
  }
  expect "unselected field survives shared join"
    (decode UScalar.val (ended_shared_payload.selected_field node false 0#u32)) (.ok 7)
  expect "selected field survives shared join"
    (decode UScalar.val (ended_shared_payload.selected_field node true 0#u32)) (.ok 11)
  expect "shared payload retains nested field"
    (decode UScalar.val (ended_shared_payload.selected_field node true 1#u32)) (.ok 7)
  expect "loop reads surviving shared payload"
    (decode UScalar.val (ended_shared_payload.selected_field node false 3#u32)) (.ok 11)
  let values : Slice U32 := ⟨[7#u32, 11#u32], by scalar_tac⟩
  expect "higher-ranked predicate borrows remain bound"
    (decode UScalar.val (ended_shared_payload.quantified_iterator values 8#u32)) (.ok 1)
  IO.println "Ended shared payload smoke: 5 behavioral checks passed"
