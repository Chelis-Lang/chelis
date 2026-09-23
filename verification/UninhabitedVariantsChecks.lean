import UninhabitedVariants

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def resultValue : core.result.Result U32 U16 → Sum Nat Nat
  | .Ok value => .inl value.val
  | .Err error => .inr error.val

def nodeValue : uninhabited_variants.graph.Node U32 → Option Nat
  | .Value value => some value.val
  | .Missing => none

def keywordValue : uninhabited_variants.KeywordFields → Option Nat
  | ⟨value⟩ => value.map UScalar.val

#eval do
  let present : Slice (Option U32) := ⟨[some 2#u32, some 3#u32], by scalar_tac⟩
  let missing : Slice (Option U32) := ⟨[some 2#u32, none, some 3#u32], by scalar_tac⟩
  expect "Nested transparent wrappers preserve the inhabited variant payload"
    (decode UScalar.val (uninhabited_variants.read_present (.Present 37#u32)))
    (.ok 37)
  expect "Result success continues after the residual branch"
    (decode resultValue (uninhabited_variants.propagate_result (.Ok 41#u32)))
    (.ok (.inl 42))
  expect "Result failure retains the original error"
    (decode resultValue (uninhabited_variants.propagate_result (.Err 17#u16)))
    (.ok (.inr 17))
  expect "A local never-returning branch does not affect a present value"
    (decode UScalar.val (uninhabited_variants.value_or_panic (some 17#u32)))
    (.ok 17)
  expect "A local never-returning call propagates panic rather than falling through"
    (decode UScalar.val (uninhabited_variants.value_or_panic none))
    (.error "failure")
  expect "The standard unwrap path preserves a present value"
    (decode UScalar.val (uninhabited_variants.unwrap_value (some 23#u32)))
    (.ok 23)
  expect "The standard unwrap helper propagates panic"
    (decode UScalar.val (uninhabited_variants.unwrap_value none))
    (.error "failure")
  expect "Loops retain normal values around an impossible branch"
    (decode UScalar.val (uninhabited_variants.sum_present present))
    (.ok 5)
  expect "A never-returning call inside a loop propagates panic"
    (decode UScalar.val (uninhabited_variants.sum_present missing))
    (.error "failure")
  expect "An enum constructor cannot capture the primitive Bool result type"
    (decode id (uninhabited_variants.PrimitiveNames.is_bool .Bool)) (.ok true)
  expect "Primitive Bool remains usable on the other branch"
    (decode id (uninhabited_variants.PrimitiveNames.is_bool .Char)) (.ok false)
  expect "An enum constructor cannot capture the primitive Char result type"
    (decode id (uninhabited_variants.PrimitiveNames.marker .Bool)) (.ok 'b')
  expect "Character results preserve the other branch"
    (decode id (uninhabited_variants.PrimitiveNames.marker .Char)) (.ok 'c')
  expect "An enum constructor cannot capture the scalar library namespace"
    (decode UScalar.val (uninhabited_variants.PrimitiveNames.number .Std)) (.ok 23)
  expect "An enum constructor cannot capture a float type or primitive operation"
    (decode (fun (value : F32) => value.bits.toNat)
      (uninhabited_variants.PrimitiveNames.fraction .F32))
    (.ok 0xbfc00000)
  expect "A module-named binder preserves generic existing values"
    (decode nodeValue (uninhabited_variants.namespace_shadow (.Value 31#u32) 47#u32))
    (.ok (some 31))
  expect "A module-named binder preserves constructor and function references"
    (decode nodeValue (uninhabited_variants.namespace_shadow .Missing 47#u32))
    (.ok (some 47))
  expect "A module-named binder preserves lexical recursive calls"
    (decode UScalar.val (uninhabited_variants.graph.count_down 4#u32)) (.ok 4)
  expect "An enum constructor cannot capture an imported model type"
    (decode id (uninhabited_variants.PrimitiveNames.text .String "model namespace"))
    (.ok "model namespace")
  expect "Lean syntax tokens remain usable as Rust fields and binders"
    (decode (fun (state : Option U32 × uninhabited_variants.KeywordFields) =>
        (state.1.map UScalar.val, keywordValue state.2))
      (uninhabited_variants.update_full ⟨some 11#u32⟩ 29#u32))
    (.ok (some 11, some 29))
  IO.println "Uninhabited variant smoke: 20 behavioral checks passed"
