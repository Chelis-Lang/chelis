import SmallStdModels

open Aeneas Aeneas.Std

namespace SmallStdModelsChecks

/-! Mirrors the native tests of `small_std_models.rs`: `u32::try_from(usize)` at and past
    `u32::MAX`, `usize += &usize` (with its overflow panic), `{:x}` of `u8` (direct and through
    `&u8`), `Debug` of a triple, and `iter::empty()` through a generic `&mut impl Iterator`. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def u8 (n : Nat) : U8 := ⟨BitVec.ofNat _ n⟩
def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def us (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩

def vec {α : Type} (l : List α) : alloc.vec.Vec α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def res : core.result.Result U32 String → Except String Nat
  | .Ok v => .ok v.val
  | .Err e => .error e

open small_std_models

#eval do
  expect "from_slot 7" (decode res (from_slot (us 7))) (.ok (.ok 7))
  expect "from_slot max" (decode res (from_slot (us 4294967295))) (.ok (.ok 4294967295))
  expect "from_slot past" (decode res (from_slot (us 4294967296)))
    (.ok (.error "slot 4294967296 does not fit: TryFromIntError(())"))

#eval do
  expect "total" (decode UScalar.val (total (vec [us 1, us 2, us 3]))) (.ok 6)
  expect "total empty" (decode UScalar.val (total (vec []))) (.ok 0)
  expect "total overflow" (decode UScalar.val (total (vec [us Usize.max, us 1]))) (.error "panic")

#eval do
  expect "hex_bytes" (decode id (hex_bytes (vec [u8 0, u8 15, u8 16, u8 255])))
    (.ok "000f10ff|a|0xff|   7|0x00ab")
  expect "show_triple" (decode id (show_triple (u32 3) true (toStr "x\"y")))
    (.ok "(3, true, \"x\\\"y\")|(\n    3,\n    true,\n    \"x\\\"y\",\n)")

#eval do
  let roots := fun ((v, (lo, hi)) : alloc.vec.Vec U32 × (Usize × Option Usize)) =>
    (v.val.map UScalar.val, lo.val, hi.map UScalar.val)
  expect "empty_roots 3" (decode roots (empty_roots (us 3))) (.ok ([], 0, some 0))
  expect "empty_roots 0" (decode roots (empty_roots (us 0))) (.ok ([], 0, some 0))

end SmallStdModelsChecks
