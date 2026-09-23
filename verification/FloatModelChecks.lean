import Aeneas.Std.Float

open Aeneas Aeneas.Std

namespace FloatModelChecks

def f32 (bits : Nat) : F32 := F32.ofBits ⟨BitVec.ofNat 32 bits⟩
def f64 (bits : Nat) : F64 := F64.ofBits ⟨BitVec.ofNat 64 bits⟩
def emitFloat {f} (label : String) (result : Result (BinaryFloat f)) : IO Unit := do
  match result with
  | .ok x => IO.println s!"{label}={x.bits.toNat}"
  | .fail _ => throw (IO.userError s!"{label}: unexpected failure")
  | .div => throw (IO.userError s!"{label}: unexpected divergence")

-- Kernel reduction, not native_decide or an opaque floating primitive.
example : BinaryFloat.roundQuotient 5 2 = 2 := by decide
example : BinaryFloat.roundQuotient 7 2 = 4 := by decide
example : (F32.ofBits ⟨BitVec.ofNat 32 0xff812345⟩).bits.toNat = 0xff812345 := by decide
example (bits : U16) : (F16.from_bits bits >>= F16.to_bits) = .ok bits := by
  cases bits; rfl
example (bits : U16) : (BF16.from_bits bits >>= BF16.to_bits) = .ok bits := by
  cases bits; rfl

/-- The native oracle prints the same rows. Special arithmetic values compare
    classification plus the specified preferred NaN policy, not host payloads. -/
def checkWidth (f : BinaryFormat) : IO Unit := do
  let width := f.width
  let frac := f.fractionBits
  let sign := 2 ^ (width - 1)
  let one := f.bias * 2 ^ frac
  let inf := f.maxExponent * 2 ^ frac
  let values := [0, sign, 1, sign + 1, 2 ^ frac - 1, 2 ^ frac, one, one + 1,
    inf - 1, inf, sign + inf, inf + 1, sign + inf + 123, inf + 2 ^ (frac - 1) + 37]
  for (b, i) in values.zipIdx do
    let x := BinaryFloat.ofNat f b
    IO.println s!"bits{width}-{i}={x.bits.toNat}"
    for (label, value) in [("is_nan", x.isNaN), ("is_infinite", x.isInfinite),
        ("is_finite", x.isFinite)] do
      IO.println s!"{label}{width}-{i}={value}"
    for (label, value) in [("neg", x.neg), ("abs", x.abs)] do
      IO.println s!"{label}{width}-{i}={value.bits.toNat}"
    for (label, value) in [("trunc", x.trunc), ("fract", x.fract)] do
      if x.isFinite then IO.println s!"{label}{width}-{i}={value.bits.toNat}"
      else IO.println s!"{label}nan{width}-{i}={value.isNaN}"
    for (c, j) in [0, sign, one, inf, inf + 1].zipIdx do
      let y := BinaryFloat.ofNat f c
      for (label, value) in [("eq", BinaryFloat.eq x y), ("lt", x.lt y), ("le", x.le y)] do
        IO.println s!"{label}{width}-{i}-{j}={value}"
  let pairs := [(one, one - (frac + 1) * 2 ^ frac),
    (one + 1, one - (frac + 1) * 2 ^ frac), (1, one - 2 ^ frac),
    (3, one - 2 ^ frac), (2 ^ frac - 1, 1), (inf - 1, inf - 1),
    (one, 3 * one / 2), (0, sign), (sign, sign), (inf, sign + inf),
    (inf, 0), (one, 0), (0, 0)]
  for ((b, c), i) in pairs.zipIdx do
    let x := BinaryFloat.ofNat f b
    let y := BinaryFloat.ofNat f c
    for (label, value) in [("add", x.add y), ("sub", x.sub y),
        ("mul", x.mul y), ("div", x.div y), ("rem", x.rem y)] do
      IO.println s!"{label}{width}-{i}-nan={value.isNaN}"
      IO.println s!"{label}{width}-{i}={value.bits.toNat}"
  let signed : List Int := [0, 1, -1, 2 ^ (frac + 1) + 1, 2 ^ (frac + 1) + 3,
    -(2 ^ 63), 2 ^ 63 - 1]
  for (n, i) in signed.zipIdx do
    IO.println s!"fromi64-{width}-{i}={(BinaryFloat.ofInt f n).bits.toNat}"
  for (n, i) in [0, 2 ^ 64 - 1, 2 ^ 63 + 1].zipIdx do
    IO.println s!"fromu64-{width}-{i}={(BinaryFloat.round f false n 1).bits.toNat}"

#eval do
  checkWidth .binary32
  checkWidth .binary64
  IO.println s!"narrow-0={(F64.toF32 (f64 0)).bits.toNat}"
  IO.println s!"narrow-1={(F64.toF32 (f64 9223372036854775808)).bits.toNat}"
  IO.println s!"narrow-2={(F64.toF32 (f64 4607182419068452864)).bits.toNat}"
  IO.println s!"narrow-3={(F64.toF32 (f64 4607182419068452865)).bits.toNat}"
  IO.println s!"narrow-4={(F64.toF32 (f64 4607182419605323776)).bits.toNat}"
  IO.println s!"narrow-5={(F64.toF32 (f64 3931642474694443008)).bits.toNat}"
  IO.println s!"narrow-6={(F64.toF32 (f64 3931642474694443009)).bits.toNat}"
  IO.println s!"narrow-7={(F64.toF32 (f64 3938397874135498752)).bits.toNat}"
  IO.println s!"narrow-8={(F64.toF32 (f64 4039728865214464000)).bits.toNat}"
  IO.println s!"narrow-9={(F64.toF32 (f64 4039728865214464001)).bits.toNat}"
  IO.println s!"narrow-10={(F64.toF32 (f64 5183643170566569984)).bits.toNat}"
  IO.println s!"narrow-11={(F64.toF32 (f64 5183643170835005440)).bits.toNat}"
  IO.println s!"narrow-12={(F64.toF32 (f64 5183643170835005439)).bits.toNat}"
  IO.println s!"narrow-13={(F64.toF32 (f64 9218868437227405312)).bits.toNat}"
  IO.println s!"narrow-14={(F64.toF32 (f64 18442240474082181120)).bits.toNat}"
  IO.println s!"widen-0={(F32.toF64 (f32 0)).bits.toNat}"
  IO.println s!"widen-1={(F32.toF64 (f32 2147483648)).bits.toNat}"
  IO.println s!"widen-2={(F32.toF64 (f32 1)).bits.toNat}"
  IO.println s!"widen-3={(F32.toF64 (f32 8388607)).bits.toNat}"
  IO.println s!"widen-4={(F32.toF64 (f32 8388608)).bits.toNat}"
  IO.println s!"widen-5={(F32.toF64 (f32 1065353217)).bits.toNat}"
  IO.println s!"widen-6={(F32.toF64 (f32 2139095039)).bits.toNat}"
  IO.println s!"widen-7={(F32.toF64 (f32 2139095040)).bits.toNat}"
  IO.println s!"widen-8={(F32.toF64 (f32 4286578688)).bits.toNat}"
  let castInputs : List Nat := [0x0, 0x8000000000000000, 0x3ff8000000000000, 0xbff8000000000000, 0x43dfffffffffffff, 0x43e0000000000000, 0xc3e0000000000000, 0xc3e0000000000001, 0x43efffffffffffff, 0x43f0000000000000, 0x47e0000000000000, 0x7ff0000000000000, 0xfff0000000000000, 0x7ff0000000000001]
  for (b, i) in castInputs.zipIdx do
    IO.println s!"cast-i8-{i}={(F64.toIScalar .I8 (f64 b)).val}"
    IO.println s!"cast-i16-{i}={(F64.toIScalar .I16 (f64 b)).val}"
    IO.println s!"cast-i32-{i}={(F64.toIScalar .I32 (f64 b)).val}"
    IO.println s!"cast-i64-{i}={(F64.toIScalar .I64 (f64 b)).val}"
    IO.println s!"cast-i128-{i}={(F64.toIScalar .I128 (f64 b)).val}"
    IO.println s!"cast-u8-{i}={(F64.toUScalar .U8 (f64 b)).val}"
    IO.println s!"cast-u16-{i}={(F64.toUScalar .U16 (f64 b)).val}"
    IO.println s!"cast-u32-{i}={(F64.toUScalar .U32 (f64 b)).val}"
    IO.println s!"cast-u64-{i}={(F64.toUScalar .U64 (f64 b)).val}"
    IO.println s!"cast-u128-{i}={(F64.toUScalar .U128 (f64 b)).val}"
  emitFloat "powi-0" (F64.powi (f64 4611686018427387904) ⟨BitVec.ofInt 32 (63)⟩)
  emitFloat "powi-1" (F64.powi (f64 4611686018427387904) ⟨BitVec.ofInt 32 (-63)⟩)
  emitFloat "powi-2" (F64.powi (f64 13835058055282163712) ⟨BitVec.ofInt 32 (63)⟩)
  emitFloat "powi-3" (F64.powi (f64 0) ⟨BitVec.ofInt 32 (0)⟩)
  emitFloat "powi-4" (F64.powi (f64 9223372036854775808) ⟨BitVec.ofInt 32 (3)⟩)
  emitFloat "powi-5" (F64.powi (f64 9223372036854775808) ⟨BitVec.ofInt 32 (-3)⟩)
  emitFloat "powi-6" (F64.powi (f64 9218868437227405312) ⟨BitVec.ofInt 32 (-2)⟩)
  emitFloat "powi-7" (F64.powi (f64 9218868437227405313) ⟨BitVec.ofInt 32 (0)⟩)
  emitFloat "powi-8" (F64.powi (f64 4611686018427387904) ⟨BitVec.ofInt 32 2147483647⟩)
  emitFloat "powi-9" (F64.powi (f64 4611686018427387904) ⟨BitVec.ofInt 32 (-2147483648)⟩)
  emitFloat "powi-10" (F64.powi (f64 4607182418800017408) ⟨BitVec.ofInt 32 2147483647⟩)

end FloatModelChecks
