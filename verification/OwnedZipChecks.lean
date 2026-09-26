import OwnedZip

open Aeneas Aeneas.Std

namespace OwnedZipChecks

/-! Mirrors the native test of `owned_zip.rs`. -/

def expect [DecidableEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual = expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def ints (v : alloc.vec.Vec I64) : List Int := v.val.map (·.val)

#eval do
  expect "equal lengths" (decode ints owned_zip.equal_lengths) (.ok [4, 5, 6])
  expect "a longer left vector stops at the right end" (decode ints owned_zip.longer_left) (.ok [4, 5])
  expect "a longer right vector stops at the left end" (decode ints owned_zip.longer_right) (.ok [4])
  expect "an empty left vector" (decode ints owned_zip.empty_left) (.ok [])
  expect "a mapped right side runs its callback once per pair"
    (decode (fun ((v, calls) : alloc.vec.Vec I64 × U32) => (ints v, calls.val)) owned_zip.mapped_right)
    (.ok ([5, 14], 2))
  IO.println "owned_zip: native and Lean agree"

end OwnedZipChecks
