import SharedSliceMetadata

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

#eval do
  let nodes : Slice shared_slice_metadata.NodeId :=
    ⟨[⟨7#usize⟩, ⟨11#usize⟩, ⟨19#usize⟩], by scalar_tac⟩
  let empty : Slice shared_slice_metadata.NodeId := ⟨[], by scalar_tac⟩
  let observeHead := fun ((count, first) : Usize × Option Usize) =>
    (count.val, first.map UScalar.val)
  let observeLengths := fun ((leftSize, rightSize, whole) : Usize × Usize × Usize) =>
    (leftSize.val, rightSize.val, whole.val)
  expect "metadata read preserves the subsequent element borrow"
    (decode observeHead (shared_slice_metadata.length_and_first nodes)) (.ok (3, some 7))
  expect "empty metadata does not dereference the slice"
    (decode observeHead (shared_slice_metadata.length_and_first empty)) (.ok (0, none))
  expect "subslice metadata describes its view"
    (decode observeLengths (shared_slice_metadata.subslice_lengths nodes 1#usize)) (.ok (1, 2, 3))
  expect "empty prefix preserves the whole suffix"
    (decode observeLengths (shared_slice_metadata.subslice_lengths nodes 0#usize)) (.ok (0, 3, 3))
  expect "empty suffix preserves the whole prefix"
    (decode observeLengths (shared_slice_metadata.subslice_lengths nodes 3#usize)) (.ok (3, 0, 3))
  let observeStr := fun (value : Str) => value.val.map UScalar.val
  expect "first literal survives return cleanup"
    (decode observeStr (shared_slice_metadata.reduction_name 0#u8))
    (.ok ("reduce_window_max".toList.map Char.toNat))
  expect "second literal survives return cleanup"
    (decode observeStr (shared_slice_metadata.reduction_name 1#u8))
    (.ok ("reduce_window_min".toList.map Char.toNat))
  expect "default literal survives return cleanup"
    (decode observeStr (shared_slice_metadata.reduction_name 2#u8))
    (.ok ("reduce_window_sum".toList.map Char.toNat))
  expect "call abstraction slots survive loop projection"
    (decode UScalar.val (shared_slice_metadata.repeat_selected nodes 1#usize 3#usize)) (.ok 33)
  expect "zero iterations retain the selected loan"
    (decode UScalar.val (shared_slice_metadata.repeat_selected nodes 0#usize 0#usize)) (.ok 0)
  expect "absent selection has no borrowed loop input"
    (decode UScalar.val (shared_slice_metadata.repeat_selected nodes 3#usize 3#usize)) (.ok 0)
  expect "zero iterations preserve the original inner alias"
    (decode UScalar.val (shared_slice_metadata.nested_shared_branches 7#usize 0#usize)) (.ok 7)
  expect "ending outer shared loans preserves inner aliases"
    (decode UScalar.val (shared_slice_metadata.nested_shared_branches 7#usize 4#usize)) (.ok 35)
  expect "higher-ranked function item borrows each argument independently"
    (decode UScalar.val (shared_slice_metadata.function_item_map nodes)) (.ok 829)
  expect "empty function-item iterator has no captured argument loan"
    (decode UScalar.val (shared_slice_metadata.function_item_map empty)) (.ok 0)
  IO.println "Shared slice metadata smoke: 15 behavioral checks passed"
