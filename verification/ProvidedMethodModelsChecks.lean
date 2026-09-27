import ProvidedMethodModels

open Aeneas Aeneas.Std

namespace ProvidedMethodModelsChecks

/-! Mirrors the native test of `provided_method_models.rs`: the dead provided-method
copies are dropped (no unknown externals), and the implementations' dictionaries supply
the inherited methods. -/

def expect [DecidableEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual = expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

/-- `Option<&dyn Error>` lives in `Type 1`: only whether it is `None` is observed. -/
def decodeNone {α : Type 1} : Result (Option α) → Except String Bool
  | .ok value => .ok value.isNone
  | .fail _ => .error "failure"
  | .div => .error "divergence"

open provided_method_models

def u (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def span (s e : Nat) : Span := { start := u s, «end» := u e }
def spanR (s : Span) : Nat × Nat := (s.start.val, s.«end».val)

#eval do
  expect "widen clones the span, then widens the clone"
    (decode spanR (widen (span 3 5) (u 4))) (.ok (3, 9))
  -- `Span`'s Clone dictionary inherits the model's `clone_from` (`*self = source.clone()`):
  -- the target given back is the clone of the source
  expect "inherited clone_from"
    (decode spanR (Span.Insts.CoreCloneClone.clone_from (span 1 2) (span 7 8))) (.ok (7, 8))
  -- `ParseError`'s Error implementation inherits `source`: no lower-level source
  let error : ParseError := { span := span 0 1, message := "bad" }
  expect "inherited source"
    (decodeNone (core.error.Error.source.default ParseError.Insts.CoreErrorError error))
    (.ok true)
  IO.println "provided_method_models: native and Lean agree"

end ProvidedMethodModelsChecks
