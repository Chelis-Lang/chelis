import AssertUnwindSafeModels

open Aeneas Aeneas.Std

namespace AssertUnwindSafeModelsChecks

/-! Mirrors the native tests of `assert_unwind_safe_models.rs`: the catch of a wrapped body moved
    whole into the catch's closure, with the runtime and the session given back after each catch
    (also after a rejected one, and through a nested catch), then the wrapper's `Deref`,
    `DerefMut` (the updated wrapper given back, and its overflow panic) and `FnOnce<()>`. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩

def vec {α : Type} (l : List α) : alloc.vec.Vec α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def nats (v : alloc.vec.Vec U32) : List Nat := v.val.map (·.val)

def res {α β : Type} (f : α → β) : core.result.Result α U32 → Except Nat β
  | .Ok v => .ok (f v)
  | .Err e => .error e.val

open assert_unwind_safe_models

def runtime (entered steps budget : Nat) : Runtime :=
  { entered := u32 entered, steps := u32 steps, budget := u32 budget }

def session (log : List Nat) (depth : Nat) : Session := { log := vec (log.map u32), depth := u32 depth }

/-- The result, the runtime's counters and the session, as the native asserts read them. -/
def observe {α β : Type} (f : α → β) (out : core.result.Result α U32 × Runtime × Session) :=
  (res f out.1, (out.2.1.entered.val, out.2.1.steps.val), (nats out.2.2.log, out.2.2.depth.val))

#eval do
  let step1 := lower_step (runtime 0 5 20) (session [9] 0) (u32 3) (u32 4)
  expect "lower_step 3 4" (decode (observe UScalar.val) step1) (.ok (.ok 7, (1, 9), ([9, 3], 1)))
  let .ok (_, rt, st) := step1 | throw (IO.userError "lower_step 3 4 failed")
  -- Rejected after the body ran: its writes stay.
  expect "lower_step 90 20" (decode (observe UScalar.val) (lower_step rt st (u32 90) (u32 20)))
    (.ok (.error 2, (2, 29), ([9, 3, 90], 2)))

#eval do
  let inner := res (fun (v : U32) => v.val)
  let nested1 := lower_nested (runtime 0 0 50) (session [] 0) (u32 5)
  expect "lower_nested 5" (decode (observe inner) nested1) (.ok (.ok (.ok 15), (2, 10), ([5, 2], 2)))
  let .ok (_, rt, st) := nested1 | throw (IO.userError "lower_nested 5 failed")
  -- The inner catch rejects, then the outer one (with the inner entry count).
  expect "lower_nested 25" (decode (observe inner) (lower_nested rt st (u32 25)))
    (.ok (.error 4, (4, 60), ([5, 2, 25, 4], 4)))

#eval do
  let pair := fun ((n, t) : Usize × U32) => (n.val, t.val)
  expect "read_through [1, 2, 3]" (decode pair (read_through (vec [u32 1, u32 2, u32 3])))
    (.ok (3, 6))
  expect "read_through []" (decode pair (read_through (vec []))) (.ok (0, 0))
  -- The returned field and the wrapper given back through `&mut`.
  expect "bump_through 5 2" (decode (fun ((r, w) : U32 × U32) => (r.val, w.val))
    (bump_through (u32 5) (u32 2))) (.ok (14, 14))
  expect "bump_through overflow" (decode (fun ((r, w) : U32 × U32) => (r.val, w.val))
    (bump_through (u32 (2 ^ 32 - 1)) (u32 1))) (.error "panic")
  expect "push_through" (decode nats (push_through (vec [u32 1]) (u32 7))) (.ok [1, 7, 8])

#eval do
  let pair := fun ((d, l) : U32 × alloc.vec.Vec U32) => (d.val, nats l)
  expect "call_wrapped 21" (decode pair (call_wrapped (u32 21) (vec [u32 1]))) (.ok (42, [1, 42]))
  expect "call_wrapped 0" (decode pair (call_wrapped (u32 0) (vec []))) (.ok (0, [0]))

end AssertUnwindSafeModelsChecks
