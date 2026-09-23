import JoinSharedProjection

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decodeNat : Result U32 → Except String Nat
  | .ok value => .ok value.val
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def decodePair : Result (U32 × U32) → Except String (Nat × Nat)
  | .ok (left, right) => .ok (left.val, right.val)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def decodeReturn : Result (core.result.Result U32 U32) → Except String (Except Nat Nat)
  | .ok (.Ok value) => .ok (.ok value.val)
  | .ok (.Err error) => .ok (.error error.val)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def decodeVec : Result (alloc.vec.Vec U32) → Except String (List Nat)
  | .ok values => .ok (values.val.map (·.val))
  | .fail _ => .error "failure"
  | .div => .error "divergence"

#eval do
  let nodes : Slice join_shared_projection.Node :=
    ⟨[{ values := ⟨[7#u32, 11#u32], by scalar_tac⟩ },
      { values := ⟨[3#u32, 5#u32], by scalar_tac⟩ }], by scalar_tac⟩
  let empty : Slice join_shared_projection.Node := ⟨[], by scalar_tac⟩
  expect "nested iterator loans retain outer shared payload"
    (decodeNat (join_shared_projection.retained_iterator_payload nodes)) (.ok 36)
  expect "empty iterator has no retained payload"
    (decodeNat (join_shared_projection.retained_iterator_payload empty)) (.ok 0)
  expect "symbolic Some shared payload survives repeated projection"
    (decodeNat (join_shared_projection.repeated_optional (some 7#u32) 4#usize)) (.ok 28)
  expect "symbolic None keeps its branch through loop matching"
    (decodeNat (join_shared_projection.repeated_optional none 4#usize)) (.ok 4)
  expect "zero iterations preserve frozen shared input"
    (decodeNat (join_shared_projection.repeated_optional (some 7#u32) 0#usize)) (.ok 0)
  expect "Result payload and shared array survive loop ownership matching"
    (decodeReturn (join_shared_projection.selected_result nodes 0#usize 3#usize)) (.ok (.ok 25))
  expect "successful zero-iteration Result preserves selected loan"
    (decodeReturn (join_shared_projection.selected_result nodes 1#usize 0#usize)) (.ok (.ok 0))
  expect "error exits without inventing a selected payload"
    (decodeReturn (join_shared_projection.selected_result nodes 2#usize 3#usize)) (.ok (.error 9))
  expect "nested shared join retains the selected left payload and both sources"
    (decodeNat (join_shared_projection.nested_shared_choice (some 7#u32) (some 11#u32) true))
    (.ok 25)
  expect "nested shared join retains the selected right payload and both sources"
    (decodeNat (join_shared_projection.nested_shared_choice (some 7#u32) (some 11#u32) false))
    (.ok 29)
  expect "nested shared join preserves an absent selected payload"
    (decodeNat (join_shared_projection.nested_shared_choice (some 7#u32) none false))
    (.ok 11)
  expect "static shared slice remains available through repeated loop matching"
    (decodeNat (join_shared_projection.static_shared_member 11#u32)) (.ok 11)
  expect "static shared slice lookup preserves an unsuccessful search"
    (decodeNat (join_shared_projection.static_shared_member 5#u32)) (.ok 0)
  expect "reserved method receiver activates after the left argument branch"
    (decodeNat (join_shared_projection.reserved_receiver_across_branch true)) (.ok 12)
  expect "reserved method receiver activates after the right argument branch"
    (decodeNat (join_shared_projection.reserved_receiver_across_branch false)) (.ok 15)
  expect "copying both original payloads preserves the selected Some view"
    (decodeNat (join_shared_projection.shared_choice_after_copies (some 7#u32) (some 11#u32) true))
    (.ok 25)
  expect "copying both original payloads preserves the selected None view"
    (decodeNat (join_shared_projection.shared_choice_after_copies (some 7#u32) none false))
    (.ok 11)
  expect "length and iteration retain simultaneous shared views"
    (decodeNat (join_shared_projection.simultaneous_shared_views
      ⟨[3#u32, 8#u32], by scalar_tac⟩)) (.ok 13)
  expect "simultaneous views of an empty vector remain empty"
    (decodeNat (join_shared_projection.simultaneous_shared_views
      ⟨[], by scalar_tac⟩)) (.ok 0)
  let left : Slice U32 := ⟨[7#u32, 11#u32], by scalar_tac⟩
  let right : Slice U32 := ⟨[3#u32, 5#u32], by scalar_tac⟩
  expect "selected mutable iterator retains the left shared slice"
    (decodeNat (join_shared_projection.selected_iterator_sum left right true)) (.ok 18)
  expect "selected mutable iterator retains the right shared slice"
    (decodeNat (join_shared_projection.selected_iterator_sum left right false)) (.ok 8)
  expect "selected mutable iterator preserves an empty selected slice"
    (decodeNat (join_shared_projection.selected_iterator_sum
      ⟨[], by scalar_tac⟩ right true)) (.ok 0)
  let caller : join_shared_projection.ObservationNode :=
    { id := 0#u32, site := .Caller 2#i32,
      inputs := ⟨[7#u32], by scalar_tac⟩, shape_deps := ⟨[], by scalar_tac⟩ }
  let claim (id : U32) (axis : I32) (inputs deps : alloc.vec.Vec U32) :
      join_shared_projection.ObservationNode :=
    { id, site := .Claim axis, inputs, shape_deps := deps }
  let observations : Slice join_shared_projection.ObservationNode :=
    ⟨[caller,
      claim 1#u32 2#i32 ⟨[7#u32], by scalar_tac⟩ ⟨[0#u32], by scalar_tac⟩,
      claim 2#u32 9#i32 ⟨[7#u32], by scalar_tac⟩ ⟨[0#u32], by scalar_tac⟩,
      claim 3#u32 2#i32 ⟨[7#u32], by scalar_tac⟩ ⟨[0#u32, 0#u32], by scalar_tac⟩,
      claim 4#u32 2#i32 ⟨[7#u32], by scalar_tac⟩ ⟨[], by scalar_tac⟩,
      claim 5#u32 2#i32 ⟨[7#u32], by scalar_tac⟩ ⟨[9#u32], by scalar_tac⟩,
      claim 6#u32 2#i32 ⟨[8#u32], by scalar_tac⟩ ⟨[0#u32], by scalar_tac⟩],
      by scalar_tac⟩
  expect "short-circuit call and drop preserve borrowed observations through loop joins"
    (decodeNat (join_shared_projection.count_missing_observations observations)) (.ok 5)
  expect "empty borrowed vector remains movable after the pattern fails"
    (decodePair (join_shared_projection.borrowed_owner_join ⟨[], by scalar_tac⟩))
    (.ok (0, 0))
  expect "borrowed payload survives a false guard"
    (decodePair (join_shared_projection.borrowed_owner_join ⟨[0#u32], by scalar_tac⟩))
    (.ok (0, 0))
  expect "borrowed payload remains readable after conditional lending ends"
    (decodePair (join_shared_projection.borrowed_owner_join ⟨[7#u32], by scalar_tac⟩))
    (.ok (1, 7))
  expect "nonmatching vector retains every referenced element"
    (decodePair (join_shared_projection.borrowed_owner_join ⟨[7#u32, 3#u32], by scalar_tac⟩))
    (.ok (0, 10))
  let tree : join_shared_projection.SharedTree :=
    .Add (.Leaf 3#u32) (.Multiply (.Leaf 5#u32) .Ignore)
  expect "recursive shared arms retain both children and accumulated output"
    (decodeVec (join_shared_projection.collect_shared_tree_values
      tree ⟨[99#u32], by scalar_tac⟩))
    (.ok [99, 3, 5])
  IO.println "Shared projection join smoke: 28 behavioral checks passed"
