import SliceIterators

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def resume (value : Bool × Option Usize) : Bool × Option Nat :=
  (value.1, value.2.map (·.val))

def panicked {α : Type} : Result α → Bool
  | .fail .panic => true
  | _ => false

#eval do
  let values : Slice Usize := ⟨[1#usize, 2#usize, 3#usize, 4#usize], by scalar_tac⟩
  let longer : Slice Usize := ⟨[1#usize, 2#usize, 3#usize, 4#usize, 5#usize], by scalar_tac⟩
  let empty : Slice Usize := ⟨[], by scalar_tac⟩
  let maxWidth : Usize := Usize.ofNatCore Usize.max (by scalar_tac)
  expect "mixed ends meet once, stay exhausted, and restore borrowed storage"
    (decode (fun (value : Usize × Usize) => (value.1.val, value.2.val))
      (slice_iterators.mixed_end_signature longer)) (.ok (1524300, 15))
  expect "reversing preserves the remaining range and swaps both ends"
    (decode UScalar.val (slice_iterators.reversed_mixed_end_signature values)) (.ok 42300)
  expect "reverse count does not select the forward count override"
    (decode UScalar.val (slice_iterators.reversed_count 3#usize)) (.ok 3)
  expect "overlapping window contents" (decode UScalar.val (slice_iterators.window_signature values 2#usize)) (.ok 122232)
  expect "nonoverlapping chunk contents" (decode UScalar.val (slice_iterators.chunk_signature values 2#usize)) (.ok 1232)
  expect "last chunk keeps its partial length" (decode UScalar.val (slice_iterators.chunk_signature values 3#usize)) (.ok 1341)
  expect "unit windows" (decode UScalar.val (slice_iterators.window_signature values 1#usize)) (.ok 11213141)
  expect "oversized windows are empty" (decode UScalar.val (slice_iterators.window_signature values 5#usize)) (.ok 0)
  expect "maximum-width chunk contains the whole slice" (decode UScalar.val (slice_iterators.chunk_signature values maxWidth)) (.ok 14)
  expect "empty windows" (decode UScalar.val (slice_iterators.window_signature empty 1#usize)) (.ok 0)
  expect "empty chunks" (decode UScalar.val (slice_iterators.chunk_signature empty 1#usize)) (.ok 0)
  expect "overlapping window count" (decode UScalar.val (slice_iterators.window_count values 2#usize)) (.ok 3)
  expect "maximum-width window count does not overflow" (decode UScalar.val (slice_iterators.window_count values maxWidth)) (.ok 0)
  expect "partial chunk counts once" (decode UScalar.val (slice_iterators.chunk_count values 3#usize)) (.ok 2)
  expect "maximum-width chunk count does not overflow" (decode UScalar.val (slice_iterators.chunk_count values maxWidth)) (.ok 1)
  expect "empty chunk count" (decode UScalar.val (slice_iterators.chunk_count empty 1#usize)) (.ok 0)
  expect "any resumes after first match" (decode resume (slice_iterators.window_any_resumes longer 2#usize 0#usize 1#usize)) (.ok (true, some 2))
  expect "any threads mutable closure state" (decode resume (slice_iterators.window_any_resumes longer 2#usize 2#usize 3#usize)) (.ok (true, some 4))
  expect "any consumes its last match" (decode resume (slice_iterators.window_any_resumes longer 2#usize 0#usize 4#usize)) (.ok (true, none))
  expect "unmatched any exhausts windows" (decode resume (slice_iterators.window_any_resumes longer 2#usize 0#usize 9#usize)) (.ok (false, none))
  expect "empty any is false" (decode resume (slice_iterators.window_any_resumes empty 2#usize 0#usize 1#usize)) (.ok (false, none))
  expect "chunk any resumes at the partial tail" (decode resume (slice_iterators.chunk_any_resumes longer 2#usize 3#usize)) (.ok (true, some 5))
  expect "chunk any consumes its last match" (decode resume (slice_iterators.chunk_any_resumes longer 2#usize 5#usize)) (.ok (true, none))
  expect "unmatched any exhausts chunks" (decode resume (slice_iterators.chunk_any_resumes longer 2#usize 9#usize)) (.ok (false, none))
  expect "windows remain exhausted" (decode id (slice_iterators.windows_stay_exhausted longer 2#usize)) (.ok true)
  expect "oversized windows remain exhausted" (decode id (slice_iterators.windows_stay_exhausted longer 6#usize)) (.ok true)
  expect "chunks remain exhausted" (decode id (slice_iterators.chunks_stay_exhausted longer 2#usize)) (.ok true)
  expect "zero window width panics before empty iteration" (panicked (slice_iterators.window_signature empty 0#usize)) true
  expect "zero chunk width panics before empty iteration" (panicked (slice_iterators.chunk_signature empty 0#usize)) true
  expect "generic count dispatch preserves custom override" (decode UScalar.val (slice_iterators.count_override_dispatch 3#usize)) (.ok 103)
  expect "direct inherited count uses next rather than inner count override" (decode UScalar.val (slice_iterators.direct_inherited_count 3#usize)) (.ok 3)
  expect "generic inherited count uses the selected default" (decode UScalar.val (slice_iterators.generic_inherited_count 3#usize)) (.ok 3)
  expect "inherited any preserves associated-type binders and continuation"
    (decode (fun (value : Bool × Usize) => (value.1, value.2.val))
      (slice_iterators.direct_inherited_any 5#usize 2#usize)) (.ok (true, 2))
  IO.println "Slice iterator smoke: 33 behavioral checks passed"
