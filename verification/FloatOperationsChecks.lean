import FloatOperations

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

#eval do
  expect "IEEE arithmetic operators"
    (decode (fun ((a, b, c, d, e, f) : U64 × U64 × U64 × U64 × U64 × U64) => [a.val, b.val, c.val, d.val, e.val, f.val])
      (float_operations.arithmetic64 0x4018000000000000#u64 0x4000000000000000#u64))
    (.ok [0x4020000000000000, 0x4010000000000000, 0x4028000000000000,
      0x4008000000000000, 0, 0xc018000000000000])
  expect "addition breaks an exact tie toward even"
    (decode (fun (value : U64 × U64 × U64 × U64 × U64 × U64) => value.1.val)
      (float_operations.arithmetic64 0x3ff0000000000000#u64 0x3ca0000000000000#u64))
    (.ok 0x3ff0000000000000)
  expect "IEEE division by zero does not panic"
    (decode (fun ((_, _, _, quotient, _, _) : U64 × U64 × U64 × U64 × U64 × U64) => quotient.val)
      (float_operations.arithmetic64 0x3ff0000000000000#u64 0#u64))
    (.ok 0x7ff0000000000000)
  expect "signed zeros compare numerically"
    (decode (fun ((a, b, c, d, e, f) : Bool × Bool × Bool × Bool × Bool × Bool) => [a, b, c, d, e, f])
      (float_operations.compare64 0#u64 0x8000000000000000#u64))
    (.ok [true, false, false, true, false, true])
  expect "NaN operands are unordered"
    (decode (fun ((a, b, c, d, e, f) : Bool × Bool × Bool × Bool × Bool × Bool) => [a, b, c, d, e, f])
      (float_operations.compare64 0x7ff0000000000001#u64 0#u64))
    (.ok [false, true, false, false, false, false])
  expect "self equality is not folded for NaN"
    (decode id (float_operations.self_equal64 0x7ff0000000000001#u64)) (.ok false)
  expect "positive infinity saturates integer casts"
    (decode (fun ((a, b, c) : I64 × U64 × U32) => (a.val, b.val, c.val))
      (float_operations.casts64 0x7ff0000000000000#u64))
    (.ok (9223372036854775807, 18446744073709551615, 0x7f800000))
  expect "negative infinity saturates integer casts"
    (decode (fun ((a, b, c) : I64 × U64 × U32) => (a.val, b.val, c.val))
      (float_operations.casts64 0xfff0000000000000#u64))
    (.ok (-9223372036854775808, 0, 0xff800000))
  expect "NaN casts to integer zero and remains NaN when narrowed"
    (decode (fun ((a, b, c) : I64 × U64 × U32) => (a.val, b.val, BinaryFloat.isNaN (F32.ofBits c)))
      (float_operations.casts64 0x7ff0000000000001#u64))
    (.ok (0, 0, true))
  expect "width conversion rounds the exact midpoint to even"
    (decode (fun ((a, b, c) : I64 × U64 × U32) => (a.val, b.val, c.val))
      (float_operations.casts64 0x3ff0000010000000#u64)) (.ok (1, 1, 0x3f800000))
  expect "width conversion preserves the side of a midpoint"
    (decode (fun ((a, b, c) : I64 × U64 × U32) => (a.val, b.val, c.val))
      (float_operations.casts64 0x3ff0000010000001#u64)) (.ok (1, 1, 0x3f800001))
  expect "signed and unsigned integer conversion"
    (decode (fun ((a, b) : U64 × U32) => (a.val, b.val)) (float_operations.integer_casts (-1)#i128 16777217#u128))
    (.ok (0xbff0000000000000, 0x4b800000))
  expect "decimal literal conversion is exact before target rounding"
    (decode (fun ((a, b, c, d, e) : U32 × U64 × U32 × U64 × U64) => (a.val, b.val, c.val, d.val, e.val)) float_operations.decimal_literals)
    (.ok (0x3dcccccd, 0x3fb999999999999a, 1, 1, 0x7fefffffffffffff))
  IO.println "Float operation smoke: 13 behavioral checks passed"
