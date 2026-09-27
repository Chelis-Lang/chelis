import IntMethodModels

open Aeneas Aeneas.Std

namespace IntMethodModelsChecks

/-! Mirrors the native tests of `int_method_models.rs`. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def u64 (n : Nat) : U64 := ⟨BitVec.ofNat _ n⟩
def us (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩
def i64 (n : Int) : I64 := ⟨BitVec.ofInt _ n⟩

def sizes (ns : List Nat) : Slice Usize :=
  let l := ns.map us
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def u64Max : Nat := 18446744073709551615
def usizeMax : Nat := 2 ^ System.Platform.numBits - 1

open int_method_models

#eval do
  let pair := fun ((a, b) : U64 × U64) => (a.val, b.val)
  expect "decimal_bounds 1" (decode pair (decimal_bounds (u32 1))) (.ok (1, 9))
  expect "decimal_bounds 19" (decode pair (decimal_bounds (u32 19)))
    (.ok (1000000000000000000, 9999999999999999999))
  expect "decimal_bounds 20" (decode pair (decimal_bounds (u32 20))) (.error "panic")
  expect "decimal_bounds 0" (decode pair (decimal_bounds (u32 0))) (.error "panic")
  let val := fun (x : U64) => x.val
  expect "power 2 63" (decode val (power (u64 2) (u32 63))) (.ok (2 ^ 63))
  expect "power 2 64" (decode val (power (u64 2) (u32 64))) (.error "panic")
  expect "power 3 40" (decode val (power (u64 3) (u32 40))) (.ok 12157665459056928801)
  expect "power 3 41" (decode val (power (u64 3) (u32 41))) (.error "panic")
  expect "power 0 0" (decode val (power (u64 0) (u32 0))) (.ok 1)
  expect "power 0 MAX" (decode val (power (u64 0) (u32 4294967295))) (.ok 0)
  expect "power 1 MAX" (decode val (power (u64 1) (u32 4294967295))) (.ok 1)
  expect "power 2 MAX" (decode val (power (u64 2) (u32 4294967295))) (.error "panic")
  expect "power MAX 1" (decode val (power (u64 u64Max) (u32 1))) (.ok u64Max)
  let bytes := fun ((a, b) : Usize × Usize) => (a.val, b.val)
  expect "copy_bytes" (decode bytes (copy_bytes (sizes [2, 3, 4]) (us 8))) (.ok (24, 192))
  expect "copy_bytes []" (decode bytes (copy_bytes (sizes []) (us 8))) (.ok (1, 8))
  expect "copy_bytes saturated" (decode bytes (copy_bytes (sizes [2 ^ 32, 2 ^ 32]) (us 1)))
    (.ok (usizeMax, usizeMax))
  expect "copy_bytes saturated bytes" (decode bytes (copy_bytes (sizes [2 ^ 32, 2 ^ 31]) (us 2)))
    (.ok (2 ^ 63, usizeMax))
  expect "copy_bytes zero" (decode bytes (copy_bytes (sizes [usizeMax, 0]) (us 3))) (.ok (0, 0))
  let tz := fun (x : U32) => x.val
  expect "trailing 0" (decode tz (trailing (u64 0))) (.ok 64)
  expect "trailing 1" (decode tz (trailing (u64 1))) (.ok 0)
  expect "trailing 40" (decode tz (trailing (u64 40))) (.ok 3)
  expect "trailing 2^63" (decode tz (trailing (u64 (2 ^ 63)))) (.ok 63)
  expect "exact_at 0" (decode id (exact_at (u64 0) (u32 53) (u32 24))) (.ok true)
  expect "exact_at 2^29" (decode id (exact_at (u64 (2 ^ 29)) (u32 53) (u32 24))) (.ok true)
  expect "exact_at 2^29+2^28" (decode id (exact_at (u64 (2 ^ 29 + 2 ^ 28)) (u32 53) (u32 24))) (.ok false)
  expect "exact_at 3" (decode id (exact_at (u64 3) (u32 2) (u32 24))) (.ok true)
  let eu := fun ((q, r) : Option I64 × Option I64) => (q.map (fun (x : I64) => x.val), r.map (fun (x : I64) => x.val))
  let euclidOf := fun (a b : Int) => decode eu (euclid (i64 a) (i64 b))
  expect "euclid 7 2" (euclidOf 7 2) (.ok (some 3, some 1))
  expect "euclid -7 2" (euclidOf (-7) 2) (.ok (some (-4), some 1))
  expect "euclid 7 -2" (euclidOf 7 (-2)) (.ok (some (-3), some 1))
  expect "euclid -7 -2" (euclidOf (-7) (-2)) (.ok (some 4, some 1))
  expect "euclid 6 -3" (euclidOf 6 (-3)) (.ok (some (-2), some 0))
  expect "euclid 5 0" (euclidOf 5 0) (.ok (none, none))
  let min : Int := -9223372036854775808
  let max : Int := 9223372036854775807
  expect "euclid MIN -1" (euclidOf min (-1)) (.ok (none, none))
  expect "euclid MIN 1" (euclidOf min 1) (.ok (some min, some 0))
  expect "euclid MIN MIN" (euclidOf min min) (.ok (some 1, some 0))
  expect "euclid -1 MIN" (euclidOf (-1) min) (.ok (some 1, some max))
  expect "euclid MAX MIN" (euclidOf max min) (.ok (some 0, some max))
  IO.println "int_method_models: native and Lean agree"

end IntMethodModelsChecks
