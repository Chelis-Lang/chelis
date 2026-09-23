import ProductionScalarHelpers

open Aeneas Aeneas.Std

private def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

private def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

private def converted {α β : Type} (f : α → β) :
    core.result.Result α core.num.error.TryFromIntError → Option β
  | .Ok value => some (f value)
  | .Err _ => none

private def usize (value : Nat) : Usize := ⟨BitVec.ofNat _ value⟩

#eval do
  expect "i32 conversion accepts its upper endpoint"
    (decode (converted IScalar.val) (production_scalar_helpers.usize_to_i32 2147483647#usize))
    (.ok (some 2147483647))
  expect "i32 conversion overflow is a normal Err"
    (decode (converted IScalar.val) (production_scalar_helpers.usize_to_i32 2147483648#usize))
    (.ok none)
  expect "negative signed input is a normal Err"
    (decode (converted UScalar.val) (production_scalar_helpers.i32_to_usize (-1)#i32))
    (.ok none)
  expect "zero converts successfully"
    (decode (converted UScalar.val) (production_scalar_helpers.i32_to_usize 0#i32))
    (.ok (some 0))
  if System.Platform.numBits = 64 then
    expect "i64 conversion accepts its upper endpoint"
      (decode (converted IScalar.val)
        (production_scalar_helpers.usize_to_i64 (usize 9223372036854775807)))
      (.ok (some 9223372036854775807))
    expect "i64 conversion overflow is a normal Err"
      (decode (converted IScalar.val)
        (production_scalar_helpers.usize_to_i64 (usize 9223372036854775808)))
      (.ok none)
  let maximum := usize Usize.max
  expect "ceil division does not overflow its numerator at MAX"
    (decode UScalar.val (production_scalar_helpers.rounded_division maximum 2#usize))
    (.ok (Usize.max / 2 + 1))
  expect "exact division is not rounded again"
    (decode UScalar.val (production_scalar_helpers.rounded_division 12#usize 3#usize)) (.ok 4)
  expect "zero numerator stays zero"
    (decode UScalar.val (production_scalar_helpers.rounded_division 0#usize 3#usize)) (.ok 0)
  expect "zero divisor fails even for zero numerator"
    (decode UScalar.val (production_scalar_helpers.rounded_division 0#usize 0#usize)) (.error "failure")
  expect "signed MIN has a representable unsigned magnitude"
    (decode (fun ((negative, magnitude) : Bool × U64) => (negative, magnitude.val))
      (production_scalar_helpers.signed_magnitude (-9223372036854775808)#i64))
    (.ok (true, 9223372036854775808))
  expect "zero is not negative"
    (decode (fun ((negative, magnitude) : Bool × U64) => (negative, magnitude.val))
      (production_scalar_helpers.signed_magnitude 0#i64)) (.ok (false, 0))
  expect "borrowed multiply returns updated self"
    (decode UScalar.val (production_scalar_helpers.multiply_borrowed 7#usize 6#usize)) (.ok 42)
  expect "borrowed multiply preserves arithmetic overflow"
    (decode UScalar.val (production_scalar_helpers.multiply_borrowed maximum 2#usize)) (.error "failure")
  expect "then_some evaluates its argument when false"
    (decode (fun ((chosen, count) : Option U32 × U32) => (chosen.map UScalar.val, count.val))
      (production_scalar_helpers.eager_then_some false 4#u32)) (.ok (none, 5))
  expect "false then_some does not hide argument failure"
    (decode (fun ((chosen, count) : Option U32 × U32) => (chosen.map UScalar.val, count.val))
      (production_scalar_helpers.eager_then_some false 4294967295#u32)) (.error "failure")
  expect "Option clone dispatches to payload Clone"
    (decode (Option.map UScalar.val) (production_scalar_helpers.clone_optional (some 4#u32)))
    (.ok (some 5))
  expect "Option clone propagates payload failure"
    (decode (Option.map UScalar.val) (production_scalar_helpers.clone_optional (some 4294967295#u32)))
    (.error "failure")
  expect "None clone has no payload to evaluate"
    (decode (Option.map UScalar.val) (production_scalar_helpers.clone_optional none)) (.ok none)
  expect "owned identity helpers retain mutable borrow updates"
    (decode UScalar.val (production_scalar_helpers.owned_helpers_update 4#u32)) (.ok 5)
  expect "then_some retains a selected mutable borrow"
    (decode UScalar.val (production_scalar_helpers.then_some_borrow_update true 4#u32)) (.ok 5)
  expect "then_some releases an unselected mutable borrow"
    (decode UScalar.val (production_scalar_helpers.then_some_borrow_update false 4#u32)) (.ok 4)
  IO.println "Production scalar helper boundary checks passed"
