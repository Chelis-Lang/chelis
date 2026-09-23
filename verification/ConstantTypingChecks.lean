import ConstantTyping

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

#eval do
  expect "AD seed literal preserves its f64 encoding"
    (decode (fun (value : F64) => value.bits.toNat) constant_typing.adjoint_seed)
    (.ok 0x3ff0000000000000)
  expect "f32 literal negative zero keeps its sign"
    (decode (fun (value : F32) => value.bits.toNat) constant_typing.negative_zero32)
    (.ok 0x80000000)
  expect "f64 literal negative zero keeps its sign"
    (decode (fun (value : F64) => value.bits.toNat) constant_typing.negative_zero64)
    (.ok 0x8000000000000000)
  expect "f32 NaN payload survives bit construction"
    (decode (fun (value : F32) => value.bits.toNat) (constant_typing.nan_payload32 0x7fc00042#u32))
    (.ok 0x7fc00042)
  expect "f64 NaN payload survives bit construction"
    (decode (fun (value : F64) => value.bits.toNat) (constant_typing.nan_payload64 0x7ff8000000000042#u64))
    (.ok 0x7ff8000000000042)
  expect "static tuple field return"
    (decode (fun (value : Str) => value.val.map UScalar.val) constant_typing.static_atom)
    (.ok [97, 116, 111, 109])
  expect "static named field return"
    (decode (fun (value : Str) => value.val.map UScalar.val) constant_typing.static_hint)
    (.ok [104, 105, 110, 116])
  expect "static argument aliases survive discarded fields and both returns"
    (decode
      (fun ((atom, hint) : Str × Str) =>
        (atom.val.map UScalar.val, hint.val.map UScalar.val))
      (constant_typing.static_atom >>= constant_typing.static_shared_argument))
    (.ok ([97, 116, 111, 109], [97, 116, 111, 109]))
  expect "integer signedness is not float encoding"
    (decode (fun ((signed, unsigned) : I64 × U64) => (signed.val, unsigned.val))
      constant_typing.signed_and_unsigned)
    (.ok (-1, 18446744073709551615))
  IO.println "Constant typing smoke: 9 behavioral checks passed"
