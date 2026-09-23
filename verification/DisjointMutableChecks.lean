import DisjointMutable

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

#eval do
  expect "disjoint mutable give-backs preserve both updates and unrelated fields"
    (decode (fun ((left, right, untouched, stamp) : U32 × U32 × U32 × U32) =>
      (left.val, right.val, untouched.val, stamp.val)) disjoint_mutable.recovered)
    (.ok (10, 16, 29, 41))
  expect "complete Some projection restores only its selected element"
    (decode (fun ((found, first, second) : Bool × U32 × U32) =>
      (found, first.val, second.val)) (disjoint_mutable.optional_recovered 0#usize))
    (.ok (true, 10, 29))
  expect "complete None projection retains the absent variant and original values"
    (decode (fun ((found, first, second) : Bool × U32 × U32) =>
      (found, first.val, second.val)) (disjoint_mutable.optional_recovered 2#usize))
    (.ok (false, 3, 29))
  expect "complete opaque projections retain both map updates and unrelated entries"
    (decode (fun ((first, second, untouched) : U32 × U32 × U32) =>
      (first.val, second.val, untouched.val)) disjoint_mutable.opaque_recovered)
    (.ok (17, 23, 41))
  expect "mutable IntoIterator restores every vector element"
    (decode (fun (values : alloc.vec.Vec U32) => values.val.map UScalar.val)
      (disjoint_mutable.mutate_iterated_vec ⟨[3#u32, 5#u32, 29#u32], by scalar_tac⟩))
    (.ok [10, 12, 36])
  expect "early mutable iteration restores the updated prefix and untouched suffix"
    (decode (fun (values : alloc.vec.Vec U32) => values.val.map UScalar.val)
      (disjoint_mutable.mutate_iterated_vec_prefix
        ⟨[3#u32, 5#u32, 29#u32], by scalar_tac⟩ 2#usize))
    (.ok [10, 12, 29])
  IO.println "Disjoint mutable projections smoke: 6 behavioral checks passed"
