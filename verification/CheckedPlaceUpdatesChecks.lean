import CheckedPlaceUpdates

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def naturals {ty : UScalarTy} (v : alloc.vec.Vec (UScalar ty)) : List Nat :=
  v.val.map (·.val)

#eval do
  expect "window_source [1,2,3] strides [2,3] pos [1,0] leading 1"
    (decode naturals (checked_place_updates.window_source
      ⟨[1#usize, 2#usize, 3#usize], by scalar_tac⟩ ⟨[2#usize, 3#usize], by scalar_tac⟩
      ⟨[1#usize, 0#usize], by scalar_tac⟩ 1#usize))
    (.ok [1, 5, 9])
  expect "window_source [4,0] strides [1] pos [2] leading 1"
    (decode naturals (checked_place_updates.window_source
      ⟨[4#usize, 0#usize], by scalar_tac⟩ ⟨[1#usize], by scalar_tac⟩
      ⟨[2#usize], by scalar_tac⟩ 1#usize))
    (.ok [4, 2])
  expect "window_source overflow panics"
    (decode naturals (checked_place_updates.window_source
      ⟨[core.num.Usize.MAX, 0#usize], by scalar_tac⟩ ⟨[2#usize], by scalar_tac⟩
      ⟨[0#usize], by scalar_tac⟩ 0#usize))
    (.error "failure")
  expect "window_group [0,1] strides [2] windows [3]"
    (decode naturals (checked_place_updates.window_group
      ⟨[0#usize, 1#usize], by scalar_tac⟩ ⟨[2#usize], by scalar_tac⟩ ⟨[3#usize], by scalar_tac⟩))
    (.ok [2, 3, 4])
  expect "window_group [1,1] strides [1,2] windows [2,2]"
    (decode naturals (checked_place_updates.window_group
      ⟨[1#usize, 1#usize], by scalar_tac⟩ ⟨[1#usize, 2#usize], by scalar_tac⟩
      ⟨[2#usize, 2#usize], by scalar_tac⟩))
    (.ok [2, 3, 2, 3])
  expect "assign_sum 3 + 4"
    (decode naturals (checked_place_updates.assign_sum ⟨[9#u8, 9#u8], by scalar_tac⟩ 3#u8 4#u8))
    (.ok [7, 9])
  expect "assign_sum overflow panics"
    (decode naturals (checked_place_updates.assign_sum ⟨[0#u8], by scalar_tac⟩ 200#u8 100#u8))
    (.error "failure")
