import ClosureMutArgs

open Aeneas Aeneas.Std

private def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

private def outcome : Result α → Except String α
  | .ok value => .ok value
  | .fail _ => .error "failure"
  | .div => .error "divergence"

#eval do
  expect "Fn gives back its mutable argument"
    (outcome (closure_mut_args.scaled 3#u32 7#u32)) (.ok (21#u32, 21#u32))
  expect "Fn gives back its wrapped mutable argument"
    (outcome (closure_mut_args.scaled 4294967295#u32 2#u32))
    (.ok (4294967294#u32, 4294967294#u32))
  expect "FnMut gives back its argument and its mutably borrowed capture"
    (outcome (closure_mut_args.accumulate 1#u32 2#u32)) (.ok ((10#u32, 4#u32), 7#u32))
  expect "FnMut with zero increments"
    (outcome (closure_mut_args.accumulate 0#u32 0#u32)) (.ok ((0#u32, 0#u32), 0#u32))
  expect "FnMut state of the argument type is not swapped with the argument"
    (outcome (closure_mut_args.counted 2#u32 3#u32)) (.ok (30#u32, 23#u32))
  expect "FnMut state is threaded between calls"
    (outcome (closure_mut_args.counted 5#u32 0#u32)) (.ok (3#u32, 2#u32))
  expect "FnOnce gives back its argument and its mutably borrowed capture"
    (outcome (closure_mut_args.once 1#u32 2#u32)) (.ok (15#u32, 8#u32))
  expect "FnMut of a single mutable borrow, capturing a mutable borrow"
    (outcome (closure_mut_args.bumped 10#u32)) (.ok ((35#u32, 9#u32), 22#u32))
  expect "FnMut of a single mutable borrow, owning its state"
    (outcome (closure_mut_args.stepped 1#u32)) (.ok (26#u32, 21#u32))
