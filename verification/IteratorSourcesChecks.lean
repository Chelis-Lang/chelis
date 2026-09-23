import IteratorSources

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def endpoints (value : Option Usize × Option Usize × Option Usize) :
  Option Nat × Option Nat × Option Nat :=
  (value.1.map (·.val), value.2.1.map (·.val), value.2.2.map (·.val))

def partialCount (value : Option Usize × Usize) : Option Nat × Nat :=
  (value.1.map (·.val), value.2.val)

def anyResume (value : Bool × Option Usize × Usize) : Bool × Option Nat × Nat :=
  (value.1, value.2.1.map (·.val), value.2.2.val)

def nthResume (value : Option Usize × Option Usize) : Option Nat × Option Nat :=
  (value.1.map (·.val), value.2.map (·.val))

def onceResume (value : Option Usize × Option Usize × Usize) :
  Option Nat × Option Nat × Nat :=
  (value.1.map (·.val), value.2.1.map (·.val), value.2.2.val)

def foldResult (value : Usize × Usize) : Nat × Nat :=
  (value.1.val, value.2.val)

def hint (value : Usize × Option Usize) : Nat × Option Nat :=
  (value.1.val, value.2.map (·.val))

def panicked {α : Type} : Result α → Bool
  | .fail .panic | .fail .assertionFailure => true
  | _ => false

#eval do
  let maxWidth : Usize := Usize.ofNatCore Usize.max (by scalar_tac)
  expect "inclusive endpoint is yielded exactly once"
    (decode endpoints (iterator_sources.inclusive_endpoint 4#usize 5#usize))
    (.ok (some 4, some 5, none))
  expect "inclusive singleton stays exhausted"
    (decode endpoints (iterator_sources.inclusive_endpoint 5#usize 5#usize))
    (.ok (some 5, none, none))
  expect "reversed inclusive range is empty"
    (decode endpoints (iterator_sources.inclusive_endpoint 6#usize 5#usize))
    (.ok (none, none, none))
  expect "scalar maximum singleton never increments past its endpoint"
    (decode anyResume iterator_sources.inclusive_max_singleton) (.ok (true, none, 0))
  expect "scalar maximum any consumes its endpoint without incrementing"
    (decode anyResume iterator_sources.inclusive_max_any) (.ok (true, none, 1))
  expect "inclusive count observes the unconsumed endpoint"
    (decode partialCount (iterator_sources.inclusive_partial_count 3#usize 5#usize))
    (.ok (some 3, 2))
  expect "reversed inclusive count is zero"
    (decode partialCount (iterator_sources.inclusive_partial_count 6#usize 5#usize))
    (.ok (none, 0))
  expect "inclusive nth resumes before the endpoint"
    (decode nthResume (iterator_sources.inclusive_nth_resume 3#usize 5#usize 1#usize))
    (.ok (some 4, some 5))
  expect "inclusive nth consumes the endpoint"
    (decode nthResume (iterator_sources.inclusive_nth_resume 3#usize 5#usize 2#usize))
    (.ok (some 5, none))
  expect "inclusive nth overflow exhausts without yielding an item"
    (decode nthResume (iterator_sources.inclusive_nth_resume 3#usize 5#usize maxWidth))
    (.ok (none, none))
  expect "inclusive any preserves predicate state and resumes after the match"
    (decode anyResume (iterator_sources.inclusive_any_resume 3#usize 5#usize 4#usize))
    (.ok (true, some 5, 2))
  expect "inclusive any consumes the matching endpoint"
    (decode anyResume (iterator_sources.inclusive_any_resume 3#usize 5#usize 5#usize))
    (.ok (true, none, 3))
  expect "inclusive any without a match is exhausted"
    (decode anyResume (iterator_sources.inclusive_any_resume 3#usize 5#usize 9#usize))
    (.ok (false, none, 3))
  expect "reversed inclusive any never calls the predicate"
    (decode anyResume (iterator_sources.inclusive_any_resume 6#usize 5#usize 5#usize))
    (.ok (false, none, 0))
  expect "inclusive count reports overflow without enumerating the whole range"
    (panicked iterator_sources.inclusive_count_overflow) true
  expect "inclusive any propagates predicate failure"
    (panicked iterator_sources.inclusive_predicate_failure) true
  expect "once yields once and remains exhausted"
    (decode onceResume (iterator_sources.once_resume 7#usize)) (.ok (some 7, none, 0))
  expect "once count consumes its sole item"
    (decode UScalar.val (iterator_sources.once_count 7#usize)) (.ok 1)
  expect "once any consumes a matching item"
    (decode anyResume (iterator_sources.once_any_resume 7#usize 7#usize)) (.ok (true, none, 1))
  expect "once any consumes an unmatched item"
    (decode anyResume (iterator_sources.once_any_resume 7#usize 9#usize)) (.ok (false, none, 1))
  expect "exhausted once never calls the predicate"
    (decode (fun (value : Bool × Usize) => (value.1, value.2.val))
      (iterator_sources.once_exhausted_any 7#usize)) (.ok (false, 0))
  expect "once nth zero consumes the sole item"
    (decode nthResume (iterator_sources.once_nth_resume 7#usize 0#usize)) (.ok (some 7, none))
  expect "once nth past the item is exhausted"
    (decode nthResume (iterator_sources.once_nth_resume 7#usize maxWidth)) (.ok (none, none))
  expect "once any propagates predicate failure"
    (panicked iterator_sources.once_predicate_failure) true
  expect "inclusive fold retains the callback state"
    (decode foldResult (iterator_sources.inclusive_fold 3#usize 5#usize)) (.ok (13, 3))
  expect "reversed inclusive fold preserves the accumulator without calling its closure"
    (decode foldResult (iterator_sources.inclusive_fold 6#usize 5#usize)) (.ok (10, 0))
  expect "inclusive fold never advances the scalar maximum endpoint"
    (decode foldResult (iterator_sources.inclusive_fold maxWidth maxWidth)) (.ok (11, 1))
  expect "inclusive try_fold resumes immediately after the break"
    (decode onceResume (iterator_sources.inclusive_try_fold 3#usize 5#usize 4#usize))
    (.ok (none, some 5, 2))
  expect "inclusive try_fold consumes a breaking endpoint"
    (decode onceResume (iterator_sources.inclusive_try_fold 3#usize 5#usize 5#usize))
    (.ok (none, none, 3))
  expect "inclusive try_fold success exhausts and returns the accumulator"
    (decode onceResume (iterator_sources.inclusive_try_fold 3#usize 5#usize 9#usize))
    (.ok (some 3, none, 3))
  expect "once fold restores its closure's state"
    (decode foldResult (iterator_sources.once_fold 7#usize)) (.ok (17, 1))
  expect "once try_fold consumes a breaking item"
    (decode onceResume (iterator_sources.once_try_fold 7#usize 7#usize)) (.ok (none, none, 1))
  expect "once try_fold success preserves the accumulator and callback"
    (decode onceResume (iterator_sources.once_try_fold 7#usize 9#usize)) (.ok (some 17, none, 1))
  expect "inclusive advance_by zero does not consume"
    (decode nthResume (iterator_sources.inclusive_advance 3#usize 5#usize 0#usize))
    (.ok (none, some 3))
  expect "inclusive advance_by leaves the remaining endpoint"
    (decode nthResume (iterator_sources.inclusive_advance 3#usize 5#usize 2#usize))
    (.ok (none, some 5))
  expect "inclusive advance_by exact length exhausts"
    (decode nthResume (iterator_sources.inclusive_advance 3#usize 5#usize 3#usize))
    (.ok (none, none))
  expect "inclusive advance_by reports the positive remaining distance"
    (decode nthResume (iterator_sources.inclusive_advance 3#usize 5#usize 5#usize))
    (.ok (some 2, none))
  expect "once advance_by zero preserves the item"
    (decode nthResume (iterator_sources.once_advance 7#usize 0#usize)) (.ok (none, some 7))
  expect "once advance_by one consumes the item"
    (decode nthResume (iterator_sources.once_advance 7#usize 1#usize)) (.ok (none, none))
  expect "once advance_by reports unconsumed distance"
    (decode nthResume (iterator_sources.once_advance 7#usize 3#usize)) (.ok (some 2, none))
  expect "inclusive size_hint includes its unconsumed endpoint"
    (decode hint (iterator_sources.inclusive_hint 3#usize 5#usize true)) (.ok (2, some 2))
  expect "inclusive size_hint is zero after the endpoint"
    (decode hint (iterator_sources.inclusive_hint 5#usize 5#usize true)) (.ok (0, some 0))
  expect "reversed inclusive size_hint is zero"
    (decode hint (iterator_sources.inclusive_hint 6#usize 5#usize false)) (.ok (0, some 0))
  expect "inclusive size_hint saturates the lower bound but not the upper"
    (decode (fun (value : Bool × Option Usize) => (value.1, value.2.map (fun n : Usize => n.val)))
      iterator_sources.inclusive_max_hint) (.ok (true, none))
  expect "once size_hint before consumption"
    (decode hint (iterator_sources.once_hint 7#usize false)) (.ok (1, some 1))
  expect "once size_hint after consumption"
    (decode hint (iterator_sources.once_hint 7#usize true)) (.ok (0, some 0))
  IO.println "Iterator sources smoke: 46 behavioral checks passed"
