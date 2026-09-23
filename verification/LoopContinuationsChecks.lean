import LoopContinuations

open Aeneas Aeneas.Std

def decodeReturn (value : Result (core.result.Result U32 U32)) :
    Except String (Except Nat Nat) :=
  match value with
  | .ok (.Ok n) => .ok (.ok n.val)
  | .ok (.Err n) => .ok (.error n.val)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def decodeSlice (value : Result (Slice U32)) : Except String (List Nat) :=
  match value with
  | .ok values => .ok (values.val.map (·.val))
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decodeNat (value : Result U32) : Option Nat :=
  match value with
  | .ok n => some n.val
  | _ => none

def decodeMixed (value : Result (core.result.Result U32 U32 × U32)) :
    Except String (Except Nat Nat × Nat) :=
  match value with
  | .ok (result, state) => (decodeReturn (.ok result)).map (fun result => (result, state.val))
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def decodeSearch (value : Result (Option Usize)) : Except String (Option Nat) :=
  match value with
  | .ok found => .ok (found.map (·.val))
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def decodeOptional
    (value : Result (core.result.Result Unit U32 × loop_continuations.OptionalOwner)) :
    Option (Option Nat × Option (List Nat × List Nat × Nat) × List Nat × Nat) :=
  match value with
  | .ok (result, owner) =>
    let error := match result with
      | .Ok () => none
      | .Err n => some n.val
    let payload := owner.«full».map fun state =>
      (state.steps.val.map (·.val), state.source.val.map (·.val), state.generation.val)
    some (error, payload, owner.legacy.val.map (·.val), owner.recorded.val)
  | _ => none

#eval do
  let values : alloc.vec.Vec U32 := ⟨[2#u32, 0#u32, 3#u32], by scalar_tac⟩
  let nonzero : alloc.vec.Vec U32 := ⟨[2#u32, 3#u32], by scalar_tac⟩
  expect "scoped early error" (decodeReturn (loop_continuations.scoped_return values true)) (.ok (.error 1))
  expect "scoped normal continuation" (decodeReturn (loop_continuations.scoped_return nonzero true)) (.ok (.ok 2))
  expect "skipped scope" (decodeReturn (loop_continuations.scoped_return values false)) (.ok (.ok 0))
  let bothErrors : Slice U32 := ⟨[1#u32, 0#u32], by scalar_tac⟩
  let lateError : Slice U32 := ⟨[2#u32, 1#u32], by scalar_tac⟩
  let empty : Slice U32 := ⟨[], by scalar_tac⟩
  expect "first loop error wins" (decodeReturn (loop_continuations.serial bothErrors)) (.ok (.error 10))
  expect "second loop error" (decodeReturn (loop_continuations.serial lateError)) (.ok (.error 20))
  expect "both loops complete" (decodeReturn (loop_continuations.serial nonzero)) (.ok (.ok 30))
  expect "zero loop iterations" (decodeReturn (loop_continuations.serial empty)) (.ok (.ok 30))
  let borrowed : Slice U32 := ⟨[2#u32, 0#u32, 0#u32], by scalar_tac⟩
  expect "first returned borrow updated" (decodeSlice (loop_continuations.update_selected borrowed true)) (.ok [2, 7, 0])
  expect "skipped borrow preserves input" (decodeSlice (loop_continuations.update_selected borrowed false)) (.ok [2, 0, 0])
  expect "absent borrow preserves input" (decodeSlice (loop_continuations.update_selected nonzero true)) (.ok [2, 3])
  let noBranches : Std.Array Bool 12#usize :=
    ⟨[false, false, false, false, false, false, false, false, false, false, false, false], by decide⟩
  let allBranches : Std.Array Bool 12#usize :=
    ⟨[true, true, true, true, true, true, true, true, true, true, true, true], by decide⟩
  expect "joined prefix, normal exit" (decodeNat (loop_continuations.branch_prefix 2#u32 noBranches)) (some 2)
  expect "joined prefix, early exit" (decodeNat (loop_continuations.branch_prefix 2#u32 allBranches)) (some 5)
  expect "mixed joins, all left arms retain mutation"
    (decodeMixed (loop_continuations.mixed_exit_observation 7#u32 0#u32 0#u32 0#u32))
    (.ok (.ok 12229, 12229))
  expect "mixed joins, all right arms retain mutation"
    (decodeMixed (loop_continuations.mixed_exit_observation 7#u32 1#u32 1#u32 1#u32))
    (.ok (.ok 12451, 12451))
  expect "continue reaches the next iteration and skips later error"
    (decodeMixed (loop_continuations.mixed_exit_observation 7#u32 2#u32 2#u32 2#u32))
    (.ok (.ok 10013, 10013))
  expect "break reaches the post-loop mutation without another iteration or later error"
    (decodeMixed (loop_continuations.mixed_exit_observation 7#u32 0#u32 2#u32 2#u32))
    (.ok (.ok 10038, 10038))
  expect "early error retains left-arm mutation and skips remaining mutations"
    (decodeMixed (loop_continuations.mixed_exit_observation 7#u32 0#u32 0#u32 2#u32))
    (.ok (.error 18, 18))
  expect "early error retains right-arm mutation and skips remaining mutations"
    (decodeMixed (loop_continuations.mixed_exit_observation 7#u32 1#u32 1#u32 2#u32))
    (.ok (.error 29, 29))
  expect "nested fallthrough retains both mutations"
    (decodeMixed (loop_continuations.nested_exit_observation 7#u32 0#u32 0#u32))
    (.ok (.ok 118, 118))
  expect "nested alternate arm retains both mutations"
    (decodeMixed (loop_continuations.nested_exit_observation 7#u32 0#u32 1#u32))
    (.ok (.ok 119, 119))
  expect "nested error bypasses both join continuations"
    (decodeMixed (loop_continuations.nested_exit_observation 7#u32 0#u32 2#u32))
    (.ok (.error 7, 7))
  expect "outer normal arm skips the nested error"
    (decodeMixed (loop_continuations.nested_exit_observation 7#u32 1#u32 2#u32))
    (.ok (.ok 127, 127))
  expect "outer success return bypasses fallthrough"
    (decodeMixed (loop_continuations.nested_exit_observation 7#u32 2#u32 0#u32))
    (.ok (.ok 7, 7))
  expect "nested panic is not an ordinary exit payload"
    (decodeMixed (loop_continuations.nested_exit_observation 4294967295#u32 0#u32 0#u32))
    (.error "failure")
  let scanValues : Slice U32 := ⟨[3#u32, 1#u32, 4#u32], by scalar_tac⟩
  let scanIndices : Slice Usize := ⟨[2#usize, 99#usize, 0#usize, 2#usize, 1#usize], by scalar_tac⟩
  let noIndices : Slice Usize := ⟨[], by scalar_tac⟩
  expect "nested shared iteration retains duplicates and filters missing elements"
    (decodeNat (loop_continuations.nested_shared_scan scanValues scanIndices 2#u32)) (some 3)
  expect "nested shared iteration preserves the branch predicate"
    (decodeNat (loop_continuations.nested_shared_scan scanValues scanIndices 4#u32)) (some 0)
  expect "nested shared iteration accepts an empty temporary vector"
    (decodeNat (loop_continuations.nested_shared_scan scanValues noIndices 2#u32)) (some 0)
  let routeState : Result (U32 × U32) → Option (Nat × Nat)
    | .ok (value, defaults) => some (value.val, defaults.val)
    | _ => none
  expect "an explicit arm preserves the borrowed payload without running the default"
    (routeState (loop_continuations.shared_default_route (.Selected 7#u32))) (some (14, 0))
  expect "a shared default retains both borrowed payloads for a later match"
    (routeState (loop_continuations.shared_default_route (.Pair 7#u32 11#u32))) (some (18, 1))
  expect "a shared default retains an owned payload and its original constructor"
    (routeState (loop_continuations.shared_default_route (.Owned 11#u32))) (some (11, 1))
  let searchValues : Slice U32 := ⟨[0#u32, 9#u32, 1#u32, 4#u32, 4#u32], by scalar_tac⟩
  expect "recursive loop continues after a failed branch and returns the first success"
    (decodeSearch (loop_continuations.recursive_search searchValues 0#usize 4#u32)) (.ok (some 3))
  expect "recursive loop returns early from its first branch"
    (decodeSearch (loop_continuations.recursive_search searchValues 0#usize 9#u32)) (.ok (some 1))
  expect "recursive loop exhausts all branches without a match"
    (decodeSearch (loop_continuations.recursive_search searchValues 0#usize 2#u32)) (.ok none)
  expect "recursive loop respects its increasing-value search predicate"
    (decodeSearch (loop_continuations.recursive_search searchValues 1#usize 4#u32)) (.ok none)
  expect "recursive search returns directly when its root matches"
    (decodeSearch (loop_continuations.recursive_search searchValues 3#usize 4#u32)) (.ok (some 3))
  expect "recursive search rejects a root just past the input"
    (decodeSearch (loop_continuations.recursive_search searchValues 5#usize 4#u32)) (.ok none)
  expect "recursive search accepts empty input"
    (decodeSearch (loop_continuations.recursive_search empty 0#usize 4#u32)) (.ok none)
  let optionalOwner (steps : alloc.vec.Vec Usize) : loop_continuations.OptionalOwner :=
    { «full» := some { steps, source := ⟨[13#u32, 17#u32], by scalar_tac⟩, generation := 19#u32 },
      legacy := ⟨[23#u32, 29#u32], by scalar_tac⟩, recorded := 31#usize }
  expect "optional first-iteration bound error retains the whole owner"
    (decodeOptional (loop_continuations.complete_optional
      (optionalOwner ⟨[5#usize, 1#usize], by scalar_tac⟩) 5#usize))
    (some (some 7, some ([5, 1], [13, 17], 19), [23, 29], 31))
  expect "optional later reorder error discards the candidate but retains the original steps"
    (decodeOptional (loop_continuations.complete_optional
      (optionalOwner ⟨[1#usize, 0#usize], by scalar_tac⟩) 5#usize))
    (some (some 7, some ([1, 0], [13, 17], 19), [23, 29], 31))
  expect "optional later bound error retains all original payload fields"
    (decodeOptional (loop_continuations.complete_optional
      (optionalOwner ⟨[1#usize, 5#usize], by scalar_tac⟩) 5#usize))
    (some (some 7, some ([1, 5], [13, 17], 19), [23, 29], 31))
  expect "optional success commits replacement steps and recorded count only"
    (decodeOptional (loop_continuations.complete_optional
      (optionalOwner ⟨[1#usize, 3#usize], by scalar_tac⟩) 5#usize))
    (some (none, some ([0, 1, 2, 3, 4], [13, 17], 19), [23, 29], 5))
  expect "optional empty payload remains present after successful completion"
    (decodeOptional (loop_continuations.complete_optional
      (optionalOwner ⟨[], by scalar_tac⟩) 0#usize))
    (some (none, some ([], [13, 17], 19), [23, 29], 0))
  expect "absent optional payload remains absent while completion records the limit"
    (decodeOptional (loop_continuations.complete_optional
      { «full» := none, legacy := ⟨[23#u32, 29#u32], by scalar_tac⟩, recorded := 31#usize } 5#usize))
    (some (none, none, [23, 29], 5))
  IO.println "Loop continuation smoke: 43 behavioral checks passed"
