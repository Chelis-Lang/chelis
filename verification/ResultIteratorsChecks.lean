import ResultIterators
import Aeneas.Std.ResultIter

open Aeneas Aeneas.Std

private def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

private def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

private def resume (value : Bool × Usize × Option Usize × Usize) :
    Bool × Nat × Option Nat × Nat :=
  (value.1, value.2.1.val, value.2.2.1.map UScalar.val, value.2.2.2.val)

private def residual (value : Usize × Option Usize × Usize) : Nat × Option Nat × Nat :=
  (value.1.val, value.2.1.map UScalar.val, value.2.2.val)

private def dispatch (value : Bool × Usize × Usize × Usize) : Bool × Nat × Nat × Nat :=
  (value.1, value.2.1.val, value.2.2.1.val, value.2.2.2.val)

private def methodResult (value : Option Usize × Usize) : Option Nat × Nat :=
  (value.1.map UScalar.val, value.2.val)

#eval do
  expect "empty collection succeeds without calling the mapper"
    (decode resume (result_iterators.collect_resume 0#usize 9#usize))
    (.ok (true, 0, none, 0))
  expect "all successes retain their values and restore the callback state"
    (decode resume (result_iterators.collect_resume 4#usize 9#usize))
    (.ok (true, 10, none, 4))
  expect "first Err consumes exactly its own item and permits resumption"
    (decode resume (result_iterators.collect_resume 4#usize 0#usize))
    (.ok (false, 100, some 2, 2))
  expect "middle Err restores the source and mutable callback at the stopping point"
    (decode resume (result_iterators.collect_resume 4#usize 2#usize))
    (.ok (false, 102, some 4, 4))
  expect "an arbitrary collector may stop before a later Err without precollection"
    (decode resume result_iterators.early_consumer)
    (.ok (true, 1, some 2, 2))
  expect "polling after shunt None resumes, while residual and zero hint persist"
    (decode residual (result_iterators.repeat_after_none false))
    (.ok (11, some 5, 5))
  expect "another Err overwrites the residual without artificially fusing the shunt"
    (decode residual (result_iterators.repeat_after_none true))
    (.ok (33, some 5, 5))
  expect "shunt next selects underlying try_fold and restores a distinct IntoIterator source"
    (decode dispatch result_iterators.next_dispatch)
    (.ok (true, 1, 1, 1))
  expect "shunt fold selects underlying try_fold rather than panicking next"
    (decode dispatch (result_iterators.fold_dispatch false))
    (.ok (true, 123, 3, 1))
  expect "specialized fold stops at Err and restores the exact source state"
    (decode dispatch (result_iterators.fold_dispatch true))
    (.ok (false, 41, 2, 1))
  expect "a by_ref override retains its mutation before the next poll"
    (decode methodResult result_iterators.overridden_by_ref)
    (.ok (some 10, 11))
  expect "Result method names preserve successful monadic returns"
    (decode methodResult (result_iterators.result_method_names 5#usize true))
    (.ok (some 5, 5))
  expect "Result method names preserve the error branch and fallback"
    (decode methodResult (result_iterators.result_method_names 5#usize false))
    (.ok (none, 9))
  IO.println "Result iterator smoke: 13 behavioral checks passed"
