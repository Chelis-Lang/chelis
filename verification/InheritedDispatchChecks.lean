import InheritedDispatch

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def countResult : Result Usize → Except String Nat
  | .ok value => .ok value.val
  | .fail _ => .error "failure"
  | .div => .error "divergence"

#eval do
  expect "Inherited count preserves empty iterator termination"
    (countResult (inherited_dispatch.count_visible 0#u32)) (.ok 0)
  expect "Restricted root restores next, helper, item type, and static dependencies"
    (countResult (inherited_dispatch.count_visible 6#u32)) (.ok 3)
  expect "Inherited dispatch propagates failure from the restored method"
    (countResult (inherited_dispatch.count_visible 3#u32)) (.error "failure")
  IO.println "Inherited dispatch smoke: 3 behavioral checks passed"
