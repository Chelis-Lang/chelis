import ScalarRefModels

open Aeneas Aeneas.Std

namespace ScalarRefModelsChecks

/-! Mirrors the native tests of `scalar_ref_models.rs`. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def nat {ty : UScalarTy} (x : UScalar ty) : Nat := x.val
def int {ty : IScalarTy} (x : IScalar ty) : Int := x.val

def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def u64 (n : Nat) : U64 := ⟨BitVec.ofNat _ n⟩
def u128 (n : Nat) : U128 := ⟨BitVec.ofNat _ n⟩
def us (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩
def i64 (n : Int) : I64 := ⟨BitVec.ofInt _ n⟩

def u32Max : Nat := 4294967295
def u128Max : Nat := 2 ^ 128 - 1
def i64Min : Int := -9223372036854775808
def usizeMax : Nat := 2 ^ System.Platform.numBits - 1
def u64Max : Nat := 18446744073709551615

def range (lo hi : Nat) : core.ops.range.Range U32 := { start := u32 lo, «end» := u32 hi }
def bounds (r : core.ops.range.Range U32) : Nat × Nat := (r.start.val, r.end.val)

open scalar_ref_models

#eval do
  expect "rem" (decode nat (rem_ref (u64 17) (u64 5))) (.ok 2)
  expect "rem 0" (decode nat (rem_ref (u64 17) (u64 0))) (.error "panic")
  let two := fun ((a, b) : U128 × U128) => (a.val, b.val)
  expect "masks" (decode two (masks (u128 0xF0F0) (u128 0xFF00) (u128 0x0FF0))) (.ok (0xF000, 0x00F0))
  expect "masks max" (decode two (masks (u128 u128Max) (u128 (2 ^ 127)) (u128 u128Max)))
    (.ok (2 ^ 127, u128Max))
  expect "not true" (decode id (not_ref true)) (.ok false)
  expect "not false" (decode id (not_ref false)) (.ok true)
  expect "diff" (decode int (diff_refs (i64 5) (i64 7))) (.ok (-2))
  expect "diff min" (decode int (diff_refs (i64 i64Min) (i64 (-1)))) (.ok (i64Min + 1))
  expect "diff overflow" (decode int (diff_refs (i64 i64Min) (i64 1))) (.error "panic")

#eval do
  let two := fun ((a, b) : Usize × Usize) => (a.val, b.val)
  expect "classes" (decode two (byte_classes (toStr "0x1F aG\tz\n\x0b\x0c\r"))) (.ok (4, 5))
  expect "classes empty" (decode two (byte_classes (toStr ""))) (.ok (0, 0))
  expect "classes é9" (decode two (byte_classes (toStr "é9"))) (.ok (1, 0))
  expect "widen 0" (decode (Option.map nat) (widen (us 0))) (.ok (some 0))
  expect "widen max" (decode (Option.map nat) (widen (us usizeMax))) (.ok (some u64Max))

#eval do
  expect "span" (decode nat (span (u32 3) (u32 10))) (.ok 7)
  expect "span empty" (decode nat (span (u32 10) (u32 3))) (.ok 0)
  expect "span max" (decode nat (span (u32 0) (u32 u32Max))) (.ok u32Max)
  let step := fun ((x, r) : Option U32 × core.ops.range.Range U32) => (x.map nat, bounds r)
  expect "nth 2" (decode step (nth_from (range 3 10) (us 2))) (.ok (some 5, (6, 10)))
  expect "nth 4" (decode step (nth_from (range 6 10) (us 4))) (.ok (none, (10, 10)))
  expect "nth huge" (decode step (nth_from (range 0 u32Max) (us (u32Max + 1))))
    (.ok (none, (u32Max, u32Max)))
  expect "nth_back 2" (decode step (nth_back_from (range 3 10) (us 2))) (.ok (some 7, (3, 7)))
  expect "nth_back 9" (decode step (nth_back_from (range 3 7) (us 9))) (.ok (none, (3, 3)))

end ScalarRefModelsChecks
