import ProjectionIteratorState

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

set_option synthInstance.maxSize 512 in
#eval do
  expect "Map projections retain both mutable updates, variants, and shared captures"
    (decode
      (fun ((output, total, slots, source) :
          alloc.vec.Vec U32 × U32 × alloc.vec.Vec U32 × U32) =>
        (output.val.map UScalar.val, total.val, slots.val.map UScalar.val, source.val))
      projection_iterator_state.recovered)
    (.ok ([40, 1, 60, 2], 28, [5, 18, 28], 7))
  expect "Map any short-circuits and next resumes both mutable captures"
    (decode
      (fun ((found, next, exhausted, total, slots, source) :
          Bool × Option U32 × Option U32 × U32 × alloc.vec.Vec U32 × U32) =>
        (found, next.map UScalar.val, exhausted.map UScalar.val, total.val,
          slots.val.map UScalar.val, source.val))
      projection_iterator_state.resumed)
    (.ok (true, some 41, none, 41, [19, 29, 41], 7))
  let first : Slice (Option U32) := ⟨[some 2#u32, none, some 5#u32], by scalar_tac⟩
  let second : Slice (Option U32) := ⟨[none, some 9#u32], by scalar_tac⟩
  expect "Sequential mutable loops preserve both sources and untouched variants"
    (decode
      (fun (sources : Slice (Option U32) × Slice (Option U32)) =>
        (sources.1.val.map (Option.map UScalar.val),
          sources.2.val.map (Option.map UScalar.val)))
      (projection_iterator_state.sequential_projected_loops first second 3#u32))
    (.ok ([some 5, none, some 8], [none, some 12]))
  expect "Option mutable transfer preserves callee writes, caller writes, and absence"
    (decode
      (fun ((callee, caller, absent) : U32 × U32 × Bool) =>
        (callee.val, caller.val, absent))
      projection_iterator_state.optional_reference_transfer)
    (.ok (42, 50, true))
  expect "Nested optional reborrows restore both loop writes without fabricating state"
    (decode
      (fun ((value, absent) : U32 × Bool) => (value.val, absent))
      projection_iterator_state.nested_reference_restoration)
    (.ok (13, true))
  let reborrowState (result : core.result.Result U32 Unit × U32) : Except Unit Nat × Nat :=
    ((match result.1 with | .Ok value => .ok value.val | .Err _ => .error ()), result.2.val)
  let successValues : Slice U32 := ⟨[2#u32, 3#u32], by scalar_tac⟩
  let failureValues : Slice U32 := ⟨[2#u32, 0#u32, 3#u32], by scalar_tac⟩
  expect "Normal loop completion restores all optional reborrow writes"
    (decode reborrowState
      (projection_iterator_state.observe_fallible_reborrow 5#u32 successValues))
    (.ok (.ok 5, 10))
  expect "A question-mark exit restores writes completed before the error"
    (decode reborrowState
      (projection_iterator_state.observe_fallible_reborrow 5#u32 failureValues))
    (.ok (.error (), 8))
  let filterState (state : Option U32 × Option U32 × Option U32 × U32) :=
    (state.1.map UScalar.val, state.2.1.map UScalar.val,
      state.2.2.1.map UScalar.val, state.2.2.2.val)
  let filteredValues : Slice U32 := ⟨[1#u32, 2#u32, 3#u32, 4#u32, 5#u32], by scalar_tac⟩
  expect "FilterMap preserves callback writes on rejected and terminal elements"
    (decode filterState (projection_iterator_state.filtered_callback_state filteredValues))
    (.ok (some 4, some 8, none, 5))
  expect "An empty FilterMap never invokes its callback"
    (decode filterState (projection_iterator_state.filtered_callback_state ⟨[], by scalar_tac⟩))
    (.ok (none, none, none, 0))
  let axes (values : alloc.vec.Vec Usize) := values.val.map UScalar.val
  expect "Nested shared captures preserve reduction-axis filtering"
    (decode axes (projection_iterator_state.filtered_shared_capture 4#usize 1#usize))
    (.ok [0, 2, 3])
  expect "Empty iteration leaves nested shared captures untouched"
    (decode axes (projection_iterator_state.filtered_shared_capture 0#usize 1#usize))
    (.ok [])
  let mixedValues : Slice U32 := ⟨[1#u32, 2#u32, 2#u32, 4#u32], by scalar_tac⟩
  expect "Nested shared captures and rejected-element writes survive together"
    (decode (fun ((values, calls) : alloc.vec.Vec U32 × U32) =>
      (values.val.map UScalar.val, calls.val))
      (projection_iterator_state.filtered_shared_mutable_captures mixedValues 2#u32))
    (.ok ([1, 4], 4))
  IO.println "Iterator projection state smoke: 12 behavioral checks passed"
