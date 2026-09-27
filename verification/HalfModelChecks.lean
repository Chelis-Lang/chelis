import Aeneas.Std.Half

open Aeneas Aeneas.Std

namespace HalfModelChecks

/-! Mirrors `crate_model_oracle/src/bin/half_model_oracle.rs` line for line,
    evaluating the registered `rust_fun`/`rust_const` models. -/

def fnv (h v : Nat) : Nat := ((h ^^^ v) * 0x100000001B3) % 2 ^ 64
def fnvStart : Nat := 0xCBF29CE484222325

def lcg (seed count : Nat) : List Nat := Id.run do
  let mut s := seed
  let mut out := #[]
  for _ in [0:count] do
    s := (s * 1103515245 + 12345) % 2 ^ 32
    out := out.push s
  return out.toList

def mantissas32 : List Nat :=
  let mask := 2 ^ 23 - 1
  [0, mask] ++ (List.range 23).flatMap (fun p =>
    let b := 2 ^ p
    [b, b ||| 1, b - 1, (3 * b) &&& mask, mask ^^^ b]) ++
    (lcg 0x12345678 32).map (· &&& mask)

def mantissas64Full : List Nat :=
  let mask := 2 ^ 52 - 1
  [0, mask] ++ (List.range 52).flatMap (fun p => let b := 2 ^ p; [b, b ||| 1, b - 1]) ++
    (lcg 0x0BAD5EED 16).map (fun s => ((s <<< 20) ||| s) &&& mask)

def mantissas64Small : List Nat :=
  [0, 1, 0xFFFFFFFF, 2 ^ 32, 2 ^ 51, 2 ^ 52 - 1, 2 ^ 44, 2 ^ 44 ||| 1]

def cmpOperands (inf one : Nat) : List Nat :=
  let quiet := inf ||| ((inf &&& (inf ^^^ (inf - 1))) >>> 1)
  [0, 0x8000, 1, 0x8001, inf - 1, 0x8000 ||| (inf - 1), inf, 0x8000 ||| inf, inf + 1,
    0x8000 ||| (inf + 1), quiet, 0x8000 ||| quiet, 0x7FFF, 0xFFFF, one, one + 1, 0x8000 ||| one,
    0x8000 ||| (one + 1), one - 1, 0x0400, 0x0080] ++ (lcg 0xC0FFEE 19).map (· &&& 0xFFFF)

instance : Inhabited F16 := ⟨BinaryFloat.ofNat _ 0⟩
instance : Inhabited BF16 := ⟨BinaryFloat.ofNat _ 0⟩
instance : Inhabited F32 := ⟨BinaryFloat.ofNat _ 0⟩
instance : Inhabited F64 := ⟨BinaryFloat.ofNat _ 0⟩

def ok! {α} [Inhabited α] : Result α → α
  | .ok x => x
  | _ => panic! "unexpected model failure"

def f32 (bits : Nat) : F32 := BinaryFloat.ofNat _ bits
def f64 (bits : Nat) : F64 := BinaryFloat.ofNat _ bits
def f16 (bits : Nat) : F16 := BinaryFloat.ofNat _ bits
def bf16 (bits : Nat) : BF16 := BinaryFloat.ofNat _ bits
def bool (b : Bool) : Nat := if b then 1 else 0
def ordCode : Option Ordering → Nat
  | none => 0 | some .lt => 1 | some .eq => 2 | some .gt => 3

def cmpRows {T} (label : String) (inst : core.cmp.PartialOrd T T) (mk : Nat → T)
    (inf one : Nat) : IO Unit := do
  let operands := cmpOperands inf one
  for (a, i) in operands.zipIdx do
    let x := mk a
    let mut h := fnvStart
    for b in operands do
      let y := mk b
      for v in [bool (ok! (inst.partialEqInst.eq x y)), bool (ok! (inst.partialEqInst.ne x y)),
          ordCode (ok! (inst.partial_cmp x y)), bool (ok! (inst.lt x y)),
          bool (ok! (inst.le x y)), bool (ok! (inst.gt x y)), bool (ok! (inst.ge x y))] do
        h := fnv h v
    IO.println s!"{label}-cmp-{i}={h}"

def main : IO Unit := do
  for (name, x) in [("ZERO", F16.ZERO), ("NEG_ZERO", F16.NEG_ZERO), ("ONE", F16.ONE),
      ("INFINITY", F16.INFINITY), ("NAN", F16.NAN), ("MAX", F16.MAX)] do
    IO.println s!"f16-{name}={x.bits.toNat}"
  for (name, x) in [("ZERO", BF16.ZERO), ("NEG_ZERO", BF16.NEG_ZERO), ("ONE", BF16.ONE),
      ("INFINITY", BF16.INFINITY), ("NAN", BF16.NAN), ("MAX", BF16.MAX)] do
    IO.println s!"bf16-{name}={x.bits.toNat}"
  let explicit32 : List Nat := [0x7F800001, 0xFF800001, 0x7FC00000, 0x7FBFFFFF, 0x33000000,
    0x33000001, 0x477FF000, 0x477FEFFF, 0x38800000, 0x387FC000, 0x3F801000, 0x3F803000,
    0x3F808000, 0x3F818000]
  for (b, i) in explicit32.zipIdx do
    IO.println s!"f16-from-f32-{i}={(ok! (F16.from_f32 (f32 b))).bits.toNat}"
    IO.println s!"bf16-from-f32-{i}={(ok! (BF16.from_f32 (f32 b))).bits.toNat}"
  let explicit64 : List Nat := [0x7FF0000000000001, 0xFFF0000000000001, 0x7FF8000000000000,
    0x3FF0000080000000, 0x3FF0100000000001, 0x3FF0100000000000, 0x3FF0300000000000,
    0x3800000000000000]
  for (b, i) in explicit64.zipIdx do
    IO.println s!"bf16-from-f64-{i}={(ok! (BF16.from_f64 (f64 b))).bits.toNat}"
  let explicit64F16 : List Nat := [0x3FF0020000000000, 0x3FF0020000000001,
    0x3FF0060000000000, 0x3E70000000000000, 0x3E70000000000001, 0x3E60000000000000,
    0x40EFFE0000000000, 0x40EFFDFFFFFFFFFF, 0x7FF0000000000001, 0xFFF4000000000001,
    0x7FF8000000000000, 0xFFF0000000000000]
  for (b, i) in explicit64F16.zipIdx do
    IO.println s!"f16-from-f64-{i}={(ok! (F16.from_f64 (f64 b))).bits.toNat}"
  for sign in [0, 1] do
    for exp in List.range 256 do
      let mut h := fnvStart
      let mut hb := fnvStart
      for m in mantissas32 do
        let f := f32 (sign <<< 31 ||| exp <<< 23 ||| m)
        h := fnv h (ok! (F16.from_f32 f)).bits.toNat
        hb := fnv hb (ok! (BF16.from_f32 f)).bits.toNat
      IO.println s!"f32-to-f16-{sign}-{exp}={h}"
      IO.println s!"f32-to-bf16-{sign}-{exp}={hb}"
  for sign in [0, 1] do
    for exp in List.range 2048 do
      let near := exp == 0 || exp == 2047 || (1023 - 140 ≤ exp && exp ≤ 1023 + 130)
      let mut h := fnvStart
      for m in if near then mantissas64Full else mantissas64Small do
        h := fnv h (ok! (BF16.from_f64 (f64 (sign <<< 63 ||| exp <<< 52 ||| m)))).bits.toNat
      IO.println s!"f64-to-bf16-{sign}-{exp}={h}"
  for sign in [0, 1] do
    for exp in List.range 2048 do
      let near := exp == 0 || exp == 2047 || (1023 - 30 ≤ exp && exp ≤ 1023 + 20)
      let mut h := fnvStart
      for m in if near then mantissas64Full else mantissas64Small do
        h := fnv h (ok! (F16.from_f64 (f64 (sign <<< 63 ||| exp <<< 52 ||| m)))).bits.toNat
      IO.println s!"f64-to-f16-{sign}-{exp}={h}"
  for hi in List.range 256 do
    let mut hs := List.replicate 8 fnvStart
    for lo in List.range 256 do
      let bits := hi <<< 8 ||| lo
      let x := f16 bits
      let y := bf16 bits
      let values := [(ok! (F16.to_f32 x)).bits.toNat,
        (ok! (half.binary16.FromF64F16.from_ x)).bits.toNat, bool (ok! (F16.is_nan x)),
        (ok! (BF16.to_f32 y)).bits.toNat, (ok! (half.bfloat.FromF64BF16.from_ y)).bits.toNat,
        bool (ok! (BF16.is_nan y)), bool (ok! (F16.is_finite x)), bool (ok! (BF16.is_finite y))]
      hs := (hs.zip values).map fun (h, v) => fnv h v
    for (label, h) in ["f16-to-f32", "f16-to-f64", "f16-is-nan", "bf16-to-f32", "bf16-to-f64",
        "bf16-is-nan", "f16-is-finite", "bf16-is-finite"].zip hs do
      IO.println s!"{label}-{hi}={h}"
  cmpRows "f16" half.binary16.PartialOrdF16 f16 0x7C00 0x3C00
  cmpRows "bf16" half.bfloat.PartialOrdBF16 bf16 0x7F80 0x3F80

#eval main

end HalfModelChecks
