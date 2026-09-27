import OwnedRevModels

open Aeneas Aeneas.Std

namespace OwnedRevModelsChecks

/-! Mirrors the native tests of `owned_rev_models.rs`, including the scope given back
through `&mut Scope` after its bindings are restored in reverse order. -/

def expect [DecidableEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual = expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def vec {α : Type} (l : List α) : alloc.vec.Vec α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩
def u (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def us (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩
def vals (v : alloc.vec.Vec U32) : List Nat := v.val.map (·.val)
def opt (o : Option U32) : Option Nat := o.map (·.val)

open owned_rev_models

def scope (entries : List (String × Option String)) : Scope := { bindings := { entries } }
def scopeR : Option (Option String) × Scope → Option (Option String) × List (String × Option String)
  | (seen, s) => (seen, s.bindings.entries)

#eval do
  -- scoped_bindings_are_restored_in_reverse: `f` is bound twice (the later binding
  -- shadows), so only the reverse restoration gives back the outer binding
  let outer := scope [("f", some "outer")]
  let first := with_bindings outer (vec ["f", "g", "f"]) (toStr "f")
  expect "with_bindings f/g/f reads the inner f and restores the outer f"
    (decode scopeR first) (.ok (some (some "f"), [("f", some "outer")]))
  -- the second call runs on the scope given back by the first, as the native test does
  let restored ← match first with
    | .ok (_, s) => pure s
    | _ => throw (IO.userError "with_bindings f/g/f failed")
  expect "with_bindings g/h reads h (bound to None) and removes g and h"
    (decode scopeR (with_bindings restored (vec ["g", "h"]) (toStr "h")))
    (.ok (some none, [("f", some "outer")]))
  expect "with_bindings a/a on an empty scope leaves it empty"
    (decode scopeR (with_bindings (scope []) (vec ["a", "a"]) (toStr "a")))
    (.ok (some none, []))
  -- reversed_and_both_ends
  expect "reversed" (decode vals (reversed (vec [u 1, u 2, u 3]))) (.ok [3, 2, 1])
  expect "reversed empty" (decode vals (reversed (vec []))) (.ok [])
  let ends := fun ((b, f, rest) : Option U32 × Option U32 × alloc.vec.Vec U32) =>
    (opt b, opt f, vals rest)
  expect "both ends of 4 5 6 7" (decode ends (both_ends (vec [u 4, u 5, u 6, u 7])))
    (.ok (some 7, some 4, [5, 6]))
  expect "both ends of 9" (decode ends (both_ends (vec [u 9]))) (.ok (some 9, none, []))
  expect "both ends of nothing" (decode ends (both_ends (vec []))) (.ok (none, none, []))
  -- nth_back_and_rfold
  let nth := fun ((found, rest) : Option U32 × alloc.vec.Vec U32) => (opt found, vals rest)
  expect "nth_back 1 of 1..5" (decode nth (nth_from_back (vec [u 1, u 2, u 3, u 4, u 5]) (us 1)))
    (.ok (some 4, [1, 2, 3]))
  expect "nth_back 0" (decode nth (nth_from_back (vec [u 1, u 2, u 3]) (us 0)))
    (.ok (some 3, [1, 2]))
  expect "nth_back past the front" (decode nth (nth_from_back (vec [u 1, u 2, u 3]) (us 3)))
    (.ok (none, []))
  expect "nth_back far past the front" (decode nth (nth_from_back (vec [u 1, u 2, u 3]) (us 7)))
    (.ok (none, []))
  expect "rfold digits" (decode (fun (d : U32) => d.val) (digits_reversed (vec [u 1, u 2, u 3])))
    (.ok 321)
  expect "rfold nothing" (decode (fun (d : U32) => d.val) (digits_reversed (vec []))) (.ok 0)
  IO.println "owned_rev_models: native and Lean agree"

end OwnedRevModelsChecks
