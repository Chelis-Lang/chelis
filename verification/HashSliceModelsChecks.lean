import HashSliceModels

open Aeneas Aeneas.Std

namespace HashSliceModelsChecks

/-! Mirrors the native tests of `hash_slice_models.rs`: the bytes and the `write` count a caller's
    hasher records (given back through its mutable borrow) for slices of integers (the length
    prefix, then one `write` of all the elements' bytes), of other types (each element's own
    `hash`), of slices, and for the derived `Hash` of an enum recursive through `Box<[Self]>`. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def u8 (n : Nat) : U8 := ⟨BitVec.ofNat _ n⟩
def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def u64 (n : Nat) : U64 := ⟨BitVec.ofNat _ n⟩
def us (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩
def i16 (n : Int) : I16 := ⟨BitVec.ofInt _ n⟩

def vec {α : Type} (l : List α) : alloc.vec.Vec α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def slice {α : Type} (l : List α) : Slice α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

/-- `n.to_le_bytes()[..width]`. -/
def le (n width : Nat) : List Nat := (List.range width).map fun i => n / 256 ^ i % 256

open hash_slice_models

/-- `Log { bytes: vec![9], writes: 1 }`. -/
def log0 : Log := { bytes := vec [u8 9], writes := u32 1 }

def logged (l : Log) : List Nat × Nat := (l.bytes.val.map (·.val), l.writes.val)

#eval do
  let out := hash_ints (vec [u32 0x01020304, u32 5]) (vec [u8 7, u8 8, u8 9]) (vec [i16 (-2)])
    (vec []) log0
  expect "ints" (decode logged out)
    (.ok ([9] ++ le 2 8 ++ [4, 3, 2, 1, 5, 0, 0, 0] ++ le 3 8 ++ [7, 8, 9] ++ le 1 8 ++
      [0xfe, 0xff] ++ le 0 8, 1 + 8))

#eval do
  let out := hash_others (vec [true, false]) (vec ["ab", ""]) (vec [none, some (u32 3)]) log0
  expect "others" (decode logged out)
    (.ok ([9] ++ le 2 8 ++ [1, 0] ++ le 2 8 ++ [97, 98, 0xff, 0xff] ++ le 2 8 ++ le 0 8 ++
      le 1 8 ++ [3, 0, 0, 0], 1 + 1 + 2 + 1 + 4 + 1 + 3))

#eval do
  let out := hash_nested (vec [slice [u64 1, u64 2], slice []]) log0
  expect "nested" (decode logged out)
    (.ok ([9] ++ le 2 8 ++ le 2 8 ++ le 1 8 ++ le 2 8 ++ le 0 8, 1 + 1 + 2 + 2))

#eval do
  let value : Result Canonical := do
    let axis := Canonical.Source (Source.Axis (u32 3) (us 1))
    let dividend ← product (vec [Canonical.Literal (u64 6), axis])
    let divisor ← product (vec [Canonical.Source (Source.Scalar (u32 2))])
    quotient dividend divisor
  let out := do hash_canonical (← value) log0
  expect "canonical" (decode logged out)
    (.ok ([9] ++ le 3 8 ++ le 2 8 ++ le 2 8 ++ le 0 8 ++ le 6 8 ++ le 1 8 ++ le 0 8 ++ le 3 4 ++
      le 1 8 ++ le 2 8 ++ le 1 8 ++ le 1 8 ++ le 1 8 ++ le 2 4, 1 + 14))
  let empty := do hash_canonical (← product (vec [])) log0
  expect "canonical empty" (decode logged empty) (.ok ([9] ++ le 2 8 ++ le 0 8, 3))

end HashSliceModelsChecks
