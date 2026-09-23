import FlatMapIterators
import Aeneas.Std.FlatMap

open Aeneas Aeneas.Std

private def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

private def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

private def resume (value : Option Usize × Option Usize × Option Usize × Usize) :
  Option Nat × Option Nat × Option Nat × Nat :=
  (value.1.map (·.val), value.2.1.map (·.val), value.2.2.1.map (·.val), value.2.2.2.val)

private def anyResume (value : Bool × Option Usize × Usize × Usize) :
  Bool × Option Nat × Nat × Nat :=
  (value.1, value.2.1.map (·.val), value.2.2.1.val, value.2.2.2.val)

private def triple (value : Option Usize × Option Usize × Option Usize) :
  Option Nat × Option Nat × Option Nat :=
  (value.1.map (·.val), value.2.1.map (·.val), value.2.2.map (·.val))

private def foldResume
    (value : core.result.Result Usize Usize × Option Usize × Usize × Usize) :
    Bool × Nat × Option Nat × Nat × Nat :=
  let result := match value.1 with
    | .Ok value => (true, value.val)
    | .Err value => (false, value.val)
  (result.1, result.2, value.2.1.map (·.val), value.2.2.1.val, value.2.2.2.val)

private def hintPair (value : Usize × Option Usize × Usize × Option Usize) :
    Nat × Option Nat × Nat × Option Nat :=
  (value.1.val, value.2.1.map (·.val), value.2.2.1.val, value.2.2.2.map (·.val))

-- Direct dictionary probes defend the canonical state ABI, including the state
-- of a terminal next. Native tests above cannot observe a by-value count receiver
-- after Rust has consumed it, but extracted mutable captures need that state.
private def outerInst : core.iter.traits.iterator.Iterator Nat Unit := {
  next := fun state => .ok (if state == 0 then some () else none, state + 1)
}

private def innerInst : core.iter.traits.iterator.Iterator Nat Nat := {
  next := fun state => .ok (if state < 2 then some (10 + state) else none, state + 1)
  count := fun state => .ok (5#usize, state + 10)
}

private def intoInst : core.iter.traits.collect.IntoIterator Nat Nat Nat := {
  iteratorInst := innerInst
  into_iter := fun state => .ok (state, id)
}

private def mapperCall (calls : Nat) (_ : Unit) : Result (Nat × Nat) :=
  .ok (0, calls + 1)

private def mapperInst : core.ops.function.FnMut Nat Unit Nat := {
  FnOnceInst := { call_once := mapperCall }
  call_mut := mapperCall
}

private def terminalState (fused : Bool) : Result (Option Nat × Nat × Nat × Option Nat) := do
  let outerInst := { outerInst with fused }
  let (self, restoreOuter, restoreMapper) ←
    core.iter.traits.iterator.Iterator.flat_map.default outerInst intoInst mapperInst 0 0
  let (_, self) ← core.iter.adapters.flatten.IteratorFlatMap.next outerInst intoInst mapperInst self
  let (_, self) ← core.iter.adapters.flatten.IteratorFlatMap.next outerInst intoInst mapperInst self
  let (_, self) ← core.iter.adapters.flatten.IteratorFlatMap.next outerInst intoInst mapperInst self
  let (item, self) ← core.iter.adapters.flatten.IteratorFlatMap.next outerInst intoInst mapperInst self
  .ok (item, restoreOuter self, restoreMapper self, self.inner)

private def countState : Result (Nat × Nat × Nat × Option Nat) := do
  let (self, restoreOuter, restoreMapper) ←
    core.iter.traits.iterator.Iterator.flat_map.default outerInst intoInst mapperInst 0 0
  let (_, self) ← core.iter.adapters.flatten.IteratorFlatMap.next outerInst intoInst mapperInst self
  let (count, self) ← core.iter.adapters.flatten.IteratorFlatMap.count outerInst intoInst mapperInst self
  .ok (count.val, restoreOuter self, restoreMapper self, self.inner)

#eval do
  expect "empty inners advance the mapper and partial inners resume without remapping"
    (decode resume flat_map_iterators.empty_inner_resume) (.ok (some 20, some 21, some 40, 4))
  expect "any returns updated mapper and predicate captures and resumes its live inner"
    (decode anyResume flat_map_iterators.any_resume) (.ok (true, some 12, 2, 4))
  expect "count selects each inner count override, even for empty inners"
    (decode (fun value : Usize × Usize => (value.1.val, value.2.val))
      (flat_map_iterators.count_override false)) (.ok (303, 3))
  expect "count starts at the partially consumed inner and excludes discarded inners"
    (decode (fun value : Usize × Usize => (value.1.val, value.2.val))
      (flat_map_iterators.count_override true)) (.ok (202, 3))
  expect "any does not dispatch to an inner any override"
    (decode (fun value : Bool × Option Usize => (value.1, value.2.map UScalar.val))
      flat_map_iterators.any_ignores_inner_override) (.ok (true, some 2))
  expect "the outer iterator is fused after its first None"
    (decode triple flat_map_iterators.fused_outer) (.ok (some 0, none, none))
  expect "non-fused inner iterators are discarded at their first None"
    (decode triple flat_map_iterators.discard_exhausted_inner) (.ok (some 1, some 1, none))
  for stage in [1#usize, 2#usize, 3#usize, 4#usize] do
    expect "construction invokes neither mapper, conversion, nor inner operations"
      (decode (Option.map UScalar.val) (flat_map_iterators.failure_boundary stage 0#usize)) (.ok none)
    expect "the first inner is available without evaluating a later failing inner"
      (decode (Option.map UScalar.val) (flat_map_iterators.failure_boundary stage 1#usize))
      (.ok (some 0))
  for stage in [1#usize, 2#usize, 3#usize] do
    expect "next propagates mapper, conversion, and inner-next failure"
      (decode (Option.map UScalar.val) (flat_map_iterators.failure_boundary stage 2#usize))
      (.error "failure")
  for stage in [1#usize, 2#usize, 4#usize] do
    expect "count propagates mapper, conversion, and inner-count failure"
      (decode (Option.map UScalar.val) (flat_map_iterators.failure_boundary stage 3#usize))
      (.error "failure")
  expect "count override bypasses a failing inner next"
    (decode (Option.map UScalar.val) (flat_map_iterators.failure_boundary 3#usize 3#usize))
    (.ok (some 201))
  expect "next does not invoke a failing count override"
    (decode (Option.map UScalar.val) (flat_map_iterators.failure_boundary 4#usize 2#usize))
    (.ok (some 0))
  expect "terminal next preserves real outer, mapper, and exhausted inner states"
    (decode id (terminalState false)) (.ok (none, 2, 1, some 3))
  expect "count preserves the selected override's final receiver and constructor givebacks"
    (decode id countState) (.ok (5, 2, 1, some 11))
  expect "native FusedIterator specialization still observes repeated terminal outer polls"
    (decode id (terminalState true)) (.ok (none, 3, 1, some 3))
  expect "fold selects both outer and inner overrides rather than panicking next"
    (decode (fun value : Usize × Usize × Usize => (value.1.val, value.2.1.val, value.2.2.val))
      flat_map_iterators.fold_dispatch) (.ok (11011, 2, 4))
  expect "try_fold preserves a residual, live inner, mapper, and fold captures"
    (decode foldResume flat_map_iterators.try_fold_dispatch) (.ok (false, 1, some 11, 2, 3))
  expect "nth dispatches outer try_fold and inner advance_by before selecting next"
    (decode (fun value : Option Usize × Option Usize × Usize =>
      (value.1.map UScalar.val, value.2.1.map UScalar.val, value.2.2.val))
      flat_map_iterators.nth_dispatch) (.ok (some 1, some 2, 1))
  expect "general zero advance may prefetch an inner without calling its next"
    (decode (fun value : Bool × Usize => (value.1, value.2.val))
      flat_map_iterators.zero_advance_prefetch) (.ok (true, 1))
  expect "OneShot zero advance does not map or prefetch"
    (decode (fun value : Bool × Usize => (value.1, value.2.val))
      flat_map_iterators.oneshot_zero_advance) (.ok (true, 0))
  expect "successful advance preserves the current inner's continuation"
    (decode (fun value : Usize × Option Usize × Usize =>
      (value.1.val, value.2.1.map UScalar.val, value.2.2.val))
      (flat_map_iterators.advance_remainder 2#usize)) (.ok (0, some 1, 2))
  expect "short advance returns the exact positive remainder and terminal state"
    (decode (fun value : Usize × Option Usize × Usize =>
      (value.1.val, value.2.1.map UScalar.val, value.2.2.val))
      (flat_map_iterators.advance_remainder 5#usize)) (.ok (2, none, 2))
  expect "fixed-array TrustedLen lets Zip's empty fold avoid the mapper"
    (decode (fun value : Bool × Usize => (value.1, value.2.val))
      flat_map_iterators.trusted_array_zip) (.ok (true, 0))
  expect "fixed-array hints include the partially consumed live inner"
    (decode hintPair flat_map_iterators.array_size_hint) (.ok (4, some 4, 3, some 3))
  expect "OneShot array hints track the outer after consuming an inner once"
    (decode hintPair flat_map_iterators.oneshot_size_hint) (.ok (2, some 2, 1, some 1))
  expect "an extracted local FusedIterator marker preserves repeated terminal polls"
    (decode UScalar.val flat_map_iterators.custom_fused_terminal) (.ok 2)
  expect "conditional fusion evidence follows the wrapped iterator dictionary"
    (decode UScalar.val flat_map_iterators.conditional_fused_terminal) (.ok 2)
  IO.println "FlatMap iterator smoke: 38 behavioral checks passed"
