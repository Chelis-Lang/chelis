import AssociatedTypes

open Aeneas Aeneas.Std

def decodeNat (value : Result U32) : Option Nat :=
  match value with
  | .ok n => some n.val
  | _ => none

def decodeVec (value : Result (alloc.vec.Vec U32)) : Option (List Nat) :=
  match value with
  | .ok values => some (values.val.map UScalar.val)
  | _ => none

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

#eval do
  expect "inherited method equality"
    (decodeNat (associated_types.value_default 41#u32)) (some 41)
  expect "local bound before known impl"
    (decodeNat (associated_types.known_local_bound 43#u32)) (some 43)
  expect "custom trait dispatch"
    (decodeNat (associated_types.custom_dispatch 5#u32)) (some 12)
  expect "borrowed associated type"
    (decodeNat (associated_types.borrowed_default 47#u32)) (some 47)
  expect "virtual inherited default"
    (decodeNat (associated_types.virtual_default 5#u32)) (some 6)
  expect "virtual explicit override"
    (decodeNat (associated_types.virtual_override 5#u32)) (some 15)
  expect "static inherited default"
    (decodeNat (associated_types.static_default 5#u32)) (some 7)
  expect "static explicit override"
    (decodeNat (associated_types.static_override 5#u32)) (some 25)
  expect "trait function item uses Value's Self type and dictionary"
    (decodeNat associated_types.trait_item_value) (some 31)
  expect "trait function item uses Offset's Self type and dictionary"
    (decodeNat associated_types.trait_item_offset) (some 17)
  expect "regular zero-argument item remains callable"
    (decodeNat associated_types.regular_item) (some 29)
  expect "multi-argument item unpacks its FnOnce tuple"
    (decodeNat (associated_types.pair_from_item 4#u32 7#u32)) (some 47)
  expect "higher-ranked item keeps its call lifetime beneath collection binders"
    (decodeVec (associated_types.collected_items ⟨[1#u32, 3#u32, 5#u32], by scalar_tac⟩))
    (some [2, 4, 6])
  expect "tuple constructor remains callable beside its type"
    (decodeNat (associated_types.tuple_constructor_item 53#u32)) (some 53)
  expect "variant constructor item retains argument order"
    (decodeNat (associated_types.variant_constructor_item 4#u32 7#u32)) (some 47)
  let aliases := match associated_types.option_builtin_aliases 59#u32 with
    | .ok (owned, borrowed, payload) => some (owned, borrowed, payload.val)
    | _ => none
  expect "builtin aliases preserve option observations and borrowed payload"
    aliases (some (true, true, 59))
  let vectorInequality := match associated_types.vector_inequality_dispatch with
    | .ok observations => some observations
    | _ => none
  expect "vector inequality preserves element inequality dispatch without calling element equality"
    vectorInequality (some (false, true, true))
  let boxedInequality := match associated_types.boxed_inequality_dispatch with
    | .ok observations => some observations
    | _ => none
  expect "boxed inequality preserves the selected override"
    boxedInequality (some (false, true))
  let sliceArrayComparisons := match associated_types.slice_array_comparison_dispatch with
    | .ok observations => some observations
    | _ => none
  expect "slice-array comparisons preserve dispatch and check lengths before callbacks"
    sliceArrayComparisons (some (false, true, false, true))
  IO.println "Associated type smoke: 19 behavioral checks passed"
