import SliceStrModels

open Aeneas Aeneas.Std

namespace SliceStrModelsChecks

/-! Mirrors the native tests of `slice_str_models.rs`. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩

def kids (ns : List Nat) : Slice U32 :=
  let l := ns.map u32
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

open slice_str_models

#eval do
  let off := fun (o : Option Usize) => o.map (fun (x : Usize) => x.val)
  expect "span_offset surf:a:12..17" (decode off (span_offset_from_id (toStr "surf:a:12..17"))) (.ok (some 12))
  expect "span_offset 12..17" (decode off (span_offset_from_id (toStr "12..17"))) (.ok (some 12))
  expect "span_offset é:3..4" (decode off (span_offset_from_id (toStr "é:3..4"))) (.ok (some 3))
  expect "span_offset a:b..4" (decode off (span_offset_from_id (toStr "a:b..4"))) (.ok none)
  expect "span_offset a:12" (decode off (span_offset_from_id (toStr "a:12"))) (.ok none)
  expect "last_separator a::b::c" (decode off (last_separator (toStr "a::b::c"))) (.ok (some 4))
  expect "last_separator a:::b" (decode off (last_separator (toStr "a:::b"))) (.ok (some 2))
  expect "last_separator ab" (decode off (last_separator (toStr "ab"))) (.ok none)
  expect "last_separator empty" (decode off (last_separator (toStr ""))) (.ok none)
  expect "strip_zeros 12500" (decode id (strip_zeros (toStr "12500"))) (.ok "125")
  expect "strip_zeros 000" (decode id (strip_zeros (toStr "000"))) (.ok "0")
  expect "strip_zeros 7" (decode id (strip_zeros (toStr "7"))) (.ok "7")
  expect "strip_zeros é0" (decode id (strip_zeros (toStr "é0"))) (.ok "é")
  expect "has_suffix abc bc" (decode id (has_suffix (toStr "abc") (toStr "bc"))) (.ok true)
  expect "has_suffix abc empty" (decode id (has_suffix (toStr "abc") (toStr ""))) (.ok true)
  expect "has_suffix empty empty" (decode id (has_suffix (toStr "") (toStr ""))) (.ok true)
  expect "has_suffix abc abcd" (decode id (has_suffix (toStr "abc") (toStr "abcd"))) (.ok false)
  expect "has_suffix abc ab" (decode id (has_suffix (toStr "abc") (toStr "ab"))) (.ok false)
  let parts := fun (o : Option (U32 × alloc.vec.Vec U32)) =>
    o.map fun ((r, a) : U32 × alloc.vec.Vec U32) => (r.val, a.val.map (fun (x : U32) => x.val))
  expect "fn_parts [1,2,3]" (decode parts (fn_parts (kids [1, 2, 3]))) (.ok (some (3, [1, 2])))
  expect "fn_parts [4]" (decode parts (fn_parts (kids [4]))) (.ok (some (4, [])))
  expect "fn_parts []" (decode parts (fn_parts (kids []))) (.ok none)
  let pair := fun ((n, x) : Usize × U32) => (n.val, x.val)
  expect "as_slice 9" (decode pair (as_slice (u32 9))) (.ok (1, 9))
  IO.println "slice_str_models: native and Lean agree"

end SliceStrModelsChecks
