import ProductionSequences

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def values (v : alloc.vec.Vec U32) : List Nat := v.val.map (·.val)

def failed {α : Type} : Result α → Bool
  | .fail _ => true
  | _ => false

#eval do
  expect "remove returns the item and shifts only the following elements"
    (decode (fun ((item, rest) : U32 × alloc.vec.Vec U32) => (item.val, values rest))
      (production_sequences.remove_at 1#usize)) (.ok (20, [10, 30]))
  expect "remove at length fails instead of returning a fabricated item"
    (failed (production_sequences.remove_at 3#usize)) true
  expect "clear discards old elements before extension"
    (decode values production_sequences.clear_then_extend) (.ok [7, 8])
  expect "explicit append retains the terminal next state and stops at None"
    (decode (fun ((v, calls, next) : alloc.vec.Vec U32 × Usize × Option U32) =>
      (values v, calls.val, next.map UScalar.val))
      production_sequences.append_resumes) (.ok ([8, 4], 2, some 9))
  expect "extend recovers mutable state captured by its consumed input iterator"
    (decode (fun ((v, sum) : alloc.vec.Vec U32 × U32) => (values v, sum.val))
      production_sequences.extend_callback_state) (.ok ([8, 11, 12], 3))
  expect "extend propagates iterator callback failure"
    (failed production_sequences.extend_failure) true
  expect "stable sorting preserves duplicate order and restores key callback state"
    (decode (fun ((v, called) : alloc.vec.Vec (U32 × U32) × Bool) =>
      (v.val.map (fun ((key, tag) : U32 × U32) => (key.val, tag.val)), called))
      production_sequences.stable_duplicate_keys)
    (.ok ([(1, 11), (1, 13), (2, 10), (2, 12)], true))
  expect "sort uses the element ordering"
    (decode values production_sequences.sort_values) (.ok [1, 1, 2, 3])
  expect "sort_by_key propagates callback failure"
    (failed production_sequences.sort_key_failure) true
  expect "admitted owned map/filter chains preserve callback order"
    (decode (fun ((v, trace) : alloc.vec.Vec U32 × U32) => (values v, trace.val))
      production_sequences.collect_ad_callback_order) (.ok ([11, 13], 123))
  expect "result collection stops before the later mapping callback"
    (decode (fun ((result, trace) :
        core.result.Result (alloc.vec.Vec U32) U32 × U32) =>
      (match result with
       | .Ok v => (values v, none, trace.val)
       | .Err error => ([], some error.val, trace.val)))
      production_sequences.collect_result_short_circuit) (.ok ([], some 9, 12))
  IO.println "Production sequences smoke: 11 behavioral checks passed"
