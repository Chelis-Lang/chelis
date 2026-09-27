import CharCaseModels

open Aeneas Aeneas.Std

namespace CharCaseModelsChecks

/-! Mirrors the native tests of `char_case_models.rs`. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩

/-- `case_digest` of the native tests, computed from the model's ranges. -/
def digest (ranges : List (UInt32 × UInt32)) : Nat × Nat :=
  ranges.foldl (fun (count, h) (lo, hi) =>
    (List.range (hi.toNat - lo.toNat + 1)).foldl
      (fun (count, h) k => (count + 1, (h * 1000003 + (lo.toNat + k) + 1) % 2 ^ 64)) (count, h))
    (0, 0)

#eval do
  expect "uppercase digest" (digest core.char.methods.uppercaseRanges) (2006, 3862418619914882295)
  expect "lowercase digest" (digest core.char.methods.lowercaseRanges) (2595, 12857385794233412760)

open char_case_models

#eval do
  expect "Some" (decode id (starts_upper (toStr "Some"))) (.ok true)
  expect "some" (decode id (starts_upper (toStr "some"))) (.ok false)
  expect "empty upper" (decode id (starts_upper (toStr ""))) (.ok false)
  expect "_X" (decode id (starts_upper (toStr "_X"))) (.ok false)
  expect "Ωmega" (decode id (starts_upper (toStr "Ωmega"))) (.ok true)
  expect "ǅ" (decode id (starts_upper (toStr "ǅ"))) (.ok false)
  expect "x1" (decode id (starts_lower (toStr "x1"))) (.ok true)
  expect "X1" (decode id (starts_lower (toStr "X1"))) (.ok false)
  expect "empty lower" (decode id (starts_lower (toStr ""))) (.ok false)
  expect "ßx" (decode id (starts_lower (toStr "ßx"))) (.ok true)
  expect "ª" (decode id (starts_lower (toStr "ª"))) (.ok true)
  expect "Tensor" (decode id (starts_upper_closure (toStr "Tensor"))) (.ok true)
  expect "tensor" (decode id (starts_upper_closure (toStr "tensor"))) (.ok false)
  expect "U+1D400" (decode id (starts_upper_closure (toStr "𝐀"))) (.ok true)

def cases : List (Nat × (Bool × Bool)) := [
  (0x41, (true, false)), (0x5A, (true, false)), (0x40, (false, false)), (0x5B, (false, false)),
  (0x61, (false, true)), (0x7A, (false, true)), (0x60, (false, false)), (0x7B, (false, false)),
  (0x30, (false, false)), (0x20, (false, false)), (0x7F, (false, false)),
  (0xAA, (false, true)), (0xC0, (true, false)), (0xD7, (false, false)),
  (0xDE, (true, false)), (0xDF, (false, true)), (0xF7, (false, false)),
  (0xFF, (false, true)), (0x1C5, (false, false)), (0x3A3, (true, false)),
  (0x3C2, (false, true)), (0x2160, (true, false)), (0x2170, (false, true)),
  (0x1D400, (true, false)), (0x1E900, (true, false)), (0x1E922, (false, true)),
  (0x10FFFF, (false, false))]

#eval show IO Unit from do
  for (c, expected) in cases do
    expect s!"classify {c}" (decode id (classify (Char.ofNat c))) (.ok expected)

#eval do
  let pair := fun ((a, b) : U32 × U32) => (a.val, b.val)
  expect "tally" (decode pair (tally (toStr "AbΣσǅ9") (u32 1, u32 2))) (.ok (3, 4))
  expect "tally empty" (decode pair (tally (toStr "") (u32 0, u32 0))) (.ok (0, 0))

end CharCaseModelsChecks
