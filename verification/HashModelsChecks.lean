import HashModels

open Aeneas Aeneas.Std

namespace HashModelsChecks

/-! Mirrors the native tests of `hash_models.rs`: the bytes each derived `Hash` writes through
the modeled `Hash` impls and the provided `Hasher::write_*` methods. -/

/-- The platform assumption of verification/trusted_base.json (`platform_assumptions`): the
model's `to_ne_bytes` is little-endian, as on the build host (the native test
`little_endian_host`). The integer writes below read it. -/
example : Platform.littleEndian = true := rfl
example (x : U32) : core.hash.neBytes x.bv =
    (let bytes := x.bv.toLEBytes.map (@UScalar.mk .U8)
     if h : bytes.length ≤ Usize.max then .ok ⟨bytes, h⟩ else .fail .panic) := by
  simp [core.hash.neBytes, Platform.littleEndian]

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def bytes (v : alloc.vec.Vec U8) : List Nat := v.val.map (fun (b : U8) => b.val)
def logged := fun ((b, w, f) : alloc.vec.Vec U8 × U32 × U64) => (bytes b, w.val, f.val)

def u8 (n : Nat) : U8 := ⟨BitVec.ofNat _ n⟩
def u16 (n : Nat) : U16 := ⟨BitVec.ofNat _ n⟩
def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def u64 (n : Nat) : U64 := ⟨BitVec.ofNat _ n⟩
def u128 (n : Nat) : U128 := ⟨BitVec.ofNat _ n⟩
def us (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩
def i8 (n : Int) : I8 := ⟨BitVec.ofInt _ n⟩
def i16 (n : Int) : I16 := ⟨BitVec.ofInt _ n⟩
def i32 (n : Int) : I32 := ⟨BitVec.ofInt _ n⟩
def i64 (n : Int) : I64 := ⟨BitVec.ofInt _ n⟩
def i128 (n : Int) : I128 := ⟨BitVec.ofInt _ n⟩
def isz (n : Int) : Isize := ⟨BitVec.ofInt _ n⟩

/-- The 8-byte little-endian discriminant of variant `d` (`write_isize`). -/
def disc (d : Nat) : List Nat := [d, 0, 0, 0, 0, 0, 0, 0]

open hash_models

#eval do
  expect "Family::Float" (decode logged (family_log .Float)) (.ok (disc 0, 1, 8001))
  expect "Family::Bool" (decode logged (family_log .Bool)) (.ok (disc 2, 1, 8001))
  expect "Named(ab)" (decode logged (key_log (.Named "ab")))
    (.ok (disc 0 ++ [97, 98, 255], 3, 11003))
  expect "Index" (decode logged (key_log (.Index (u32 0x01020304))))
    (.ok (disc 1 ++ [4, 3, 2, 1], 2, 12002))
  let first := fun ((b, _, _) : alloc.vec.Vec U8 × U32 × U64) => bytes b
  expect "Pair(-2, 258)" (decode first (key_log (.Pair (i64 (-2)) (us 258))))
    (.ok (disc 2 ++ [254, 255, 255, 255, 255, 255, 255, 255, 2, 1, 0, 0, 0, 0, 0, 0]))
  expect "Maybe(None)" (decode first (key_log (.Maybe none))) (.ok (disc 3 ++ disc 0))
  expect "Maybe(Some 7)" (decode first (key_log (.Maybe (some (u32 7)))))
    (.ok (disc 3 ++ disc 1 ++ [7, 0, 0, 0]))
  expect "Flag(true)" (decode first (key_log (.Flag true))) (.ok (disc 4 ++ [1]))
  expect "Letter(é)" (decode first (key_log (.Letter 'é'))) (.ok (disc 5 ++ [0xe9, 0, 0, 0]))
  expect "Small" (decode first (key_log (.Small (u8 0x81) (i8 (-1)) (u16 0x0102) (i16 (-2)))))
    (.ok (disc 6 ++ [0x81, 0xff, 2, 1, 0xfe, 0xff]))
  expect "Wide" (decode first (key_log (.Wide (u128 (2 ^ 120)) (i128 (-1)) (isz (-3)))))
    (.ok (disc 7 ++ (List.replicate 15 0 ++ [1]) ++ List.replicate 16 0xff ++
      [0xfd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff]))
  expect "Site(f, 3, -1)" (decode logged (site_log (toStr "f") (u64 3) (i32 (-1))))
    (.ok ([102, 255, 3, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255], 4, 14004))
  expect "str_log empty" (decode logged (str_log (toStr ""))) (.ok ([255], 2, 1002))
  expect "str_log é" (decode logged (str_log (toStr "é"))) (.ok ([0xc3, 0xa9, 255], 2, 3002))
  expect "Tagged Index" (decode bytes (key_tagged (.Index (u32 0x01020304))))
    (.ok (disc 1 ++ [0xAA, 4]))
  expect "Tagged Letter(A)" (decode bytes (key_tagged (.Letter 'A'))) (.ok (disc 5 ++ [0xAA, 65]))
  expect "Tagged Maybe(Some 9)" (decode bytes (key_tagged (.Maybe (some (u32 9)))))
    (.ok (disc 3 ++ disc 1 ++ [0xAA, 9]))
  IO.println "hash_models: native and Lean agree"

end HashModelsChecks
