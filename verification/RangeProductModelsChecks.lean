import RangeProductModels

open Aeneas Aeneas.Std

namespace RangeProductModelsChecks

/-! Mirrors the native tests of `range_product_models.rs` on its extraction. -/

def expect [DecidableEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual = expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def usize (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩

def shape (l : List Nat) : Slice Usize :=
  let l := l.map usize
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def zero : U64 := 0#u64
def negZero : U64 := 0x8000000000000000#u64
def half : U64 := 0x3fe0000000000000#u64
def one : U64 := 0x3ff0000000000000#u64
def negTiny : U64 := 0x8000000000000001#u64
def belowOne : U64 := 0x3fefffffffffffff#u64
def nan : U64 := 0x7ff8000000000000#u64
def inf : U64 := 0x7ff0000000000000#u64

def pow32 : Nat := 2 ^ 32
def usizeMax : Nat := 2 ^ 64 - 1

#eval do
  let rate (bits : U64) := decode id (range_product_models.dropout_rate_in_domain bits)
  expect "0.0 in 0.0..1.0" (rate zero) (.ok true)
  expect "-0.0 in 0.0..1.0" (rate negZero) (.ok true)
  expect "0.5 in 0.0..1.0" (rate half) (.ok true)
  expect "pred(1.0) in 0.0..1.0" (rate belowOne) (.ok true)
  expect "1.0 not in 0.0..1.0" (rate one) (.ok false)
  expect "-tiny not in 0.0..1.0" (rate negTiny) (.ok false)
  expect "NaN not in 0.0..1.0" (rate nan) (.ok false)
  expect "inf not in 0.0..1.0" (rate inf) (.ok false)
  let within (s e x : U64) := decode id (range_product_models.contains_f64 s e x)
  expect "NaN start" (within nan one half) (.ok false)
  expect "NaN end" (within zero nan half) (.ok false)
  expect "infinite end" (within zero inf one) (.ok true)
  expect "reversed float range" (within one zero half) (.ok false)
  let indices (s e x : Nat) :=
    decode id (range_product_models.contains_usize (usize s) (usize e) (usize x))
  expect "start included" (indices 3 5 3) (.ok true)
  expect "interior" (indices 3 5 4) (.ok true)
  expect "end excluded" (indices 3 5 5) (.ok false)
  expect "below start" (indices 3 5 2) (.ok false)
  expect "empty range" (indices 3 3 3) (.ok false)
  expect "reversed range" (indices 5 3 4) (.ok false)
  expect "near usize::MAX" (indices 0 usizeMax (usizeMax - 1)) (.ok true)
  let mixed (s e x : U32) := decode id (range_product_models.contains_mixed s e x)
  expect "mixed start included" (mixed 3#u32 7#u32 3#u32) (.ok true)
  expect "mixed interior" (mixed 3#u32 7#u32 6#u32) (.ok true)
  expect "mixed end excluded" (mixed 3#u32 7#u32 7#u32) (.ok false)
  expect "mixed reversed" (mixed 7#u32 3#u32 5#u32) (.ok false)
  expect "failed start comparison skips the end" (mixed 1#u32 99#u32 0#u32) (.ok false)
  expect "start compared through the bound" (mixed 99#u32 100#u32 5#u32) (.error "failure")
  expect "end compared through the item" (mixed 1#u32 99#u32 5#u32) (.error "failure")
  let numel (l : List Nat) := decode (fun (n : Usize) => n.val) (range_product_models.numel (shape l))
  expect "empty product" (numel []) (.ok 1)
  expect "singleton product" (numel [7]) (.ok 7)
  expect "product" (numel [2, 3, 4]) (.ok 24)
  expect "zero first" (numel [0, usizeMax, 2]) (.ok 0)
  expect "largest product" (numel [pow32, pow32 - 1]) (.ok (usizeMax - (pow32 - 1)))
  expect "first overflow panics before a later zero" (numel [usizeMax, 2, 0]) (.error "failure")
  expect "overflow panics" (numel [pow32, pow32]) (.error "failure")
  let inner (l : List Nat) := decode (fun (n : Usize) => n.val) (range_product_models.product_after_ends (shape l))
  expect "product of the remaining items" (inner [1000, 2, 3, 5, 1000]) (.ok 30)
  expect "product after exhausting a singleton" (inner [9]) (.ok 1)
  expect "product of an empty iterator" (inner []) (.ok 1)
  IO.println "range_product_models: native and Lean agree"

end RangeProductModelsChecks
