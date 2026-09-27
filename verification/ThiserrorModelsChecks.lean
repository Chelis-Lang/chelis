import ThiserrorModels

open Aeneas Aeneas.Std

namespace ThiserrorModelsChecks

/-! Mirrors the native test of `thiserror_models.rs`: the messages of the errors whose
`Display` impls format their fields through `AsDisplay::as_display`. -/

def expect [DecidableEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual = expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode : Result String → Except String String
  | .ok value => .ok value
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def us (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩

open thiserror

#eval do
  expect "missing signature"
    (decode (message (.MissingSignature "main")))
    (.ok "ownership lowering has no checked signature for `main`")
  expect "call arity (fields in the message's order, not the variant's)"
    (decode (message (.CallArityMismatch "u" "f" (us 3) (us 12))))
    (.ok "call to `f` in `u` supplies 3 arguments, expected 12")
  let zero := OwnershipError.Capacity "k" .StaticZeroDivisor
  expect "a field of a local error type, through its Display dictionary"
    (decode (message zero))
    (.ok "capacity key in `k`: exact capacity quotient has a statically zero divisor")
  expect "a tuple field of a local error type"
    (decode (message (.Capacity "" (.MalformedSource "x/0"))))
    (.ok "capacity key in ``: verified capacity source is malformed: x/0")
  expect "a boxed error of the same type (recursive Display)"
    (decode (message (.Nested "outer" (.Nested "inner" zero))))
    (.ok "in `outer`: in `inner`: capacity key in `k`: exact capacity quotient has a statically zero divisor")
  IO.println "thiserror_models: native and Lean agree"

end ThiserrorModelsChecks
