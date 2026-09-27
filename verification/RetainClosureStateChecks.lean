import RetainClosureState

open Aeneas Aeneas.Std

-- `BEq` of the decoded results goes through `DecidableEq` of nested tuples and lists
set_option synthInstance.maxSize 512

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def vec {α : Type} (l : List α) : alloc.vec.Vec α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩
def slice {α : Type} (l : List α) : Slice α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩
def vals (v : alloc.vec.Vec U32) : List Nat := v.val.map (·.val)
def u (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩

open retain_closure_state

def gate (operand : String) (dims : List Nat) : DeferredOperandGate :=
  { operand, dims := vec (dims.map u) }
def gateR (g : DeferredOperandGate) : String × List Nat := (g.operand, vals g.dims)
def ledgerR (s : Subst) : List (Nat × String × List Nat) × Nat :=
  (s.deferred_tensor_operands.val.map fun (tv, g) => (tv.val, gateR g), s.generation.val)

def subst : Subst :=
  { deferred_tensor_operands := vec [(u 1, gate "x" [2]), (u 2, gate "y" []),
      (u 1, gate "z" [3, 4]), (u 3, gate "w" [5])], generation := u 7 }

#eval do
  -- the closure captures `taken` mutably (read after `retain`) and `v` by shared borrow; the
  -- ledger and the generation are given back through `&mut self`
  let takeR := fun ((taken, s) : alloc.vec.Vec DeferredOperandGate × Subst) =>
    (taken.val.map gateR, ledgerR s)
  expect "take_operand_gates_on 1"
    (decode takeR (Subst.take_operand_gates_on subst (u 1)))
    (.ok ([("x", [2]), ("z", [3, 4])], ([(2, "y", []), (3, "w", [5])], 8)))
  let after := match Subst.take_operand_gates_on subst (u 1) with
    | .ok (_, s) => s | _ => subst
  expect "take_operand_gates_on 9 (nothing suspended)"
    (decode takeR (Subst.take_operand_gates_on after (u 9)))
    (.ok ([], ([(2, "y", []), (3, "w", [5])], 9)))
  -- the retain closure captures the substitution mutably and its inner `any` closure
  -- reborrows it: the substitution is given back with its resolution count, and the
  -- kept checks
  let cancelR := fun ((cs, s) : alloc.vec.Vec ShapeCheck × Subst) =>
    (cs.val.map fun (c : ShapeCheck) => vals c.arg_tys, ledgerR s)
  let checks : alloc.vec.Vec ShapeCheck :=
    vec ([[1, 2], [35], [4], [], [12, 7]].map fun (l : List Nat) => ({ arg_tys := vec (l.map u) } : ShapeCheck))
  let s0 : Subst := { deferred_tensor_operands := vec [(u 1, gate "x" [2])], generation := u 7 }
  expect "cancel_shape_checks"
    (decode cancelR (cancel_shape_checks checks s0 (slice [u 3, u 7])))
    (.ok ([[1, 2], [4], []], ([(1, "x", [2])], 13)))
  let (kept, s1) := match cancel_shape_checks checks s0 (slice [u 3, u 7]) with
    | .ok r => r | _ => (checks, s0)
  expect "cancel_shape_checks, nothing failed"
    (decode cancelR (cancel_shape_checks kept s1 (slice [])))
    (.ok ([[1, 2], [4], []], ([(1, "x", [2])], 16)))
  -- two captured counters, and the given-back vector
  let countsR := fun (((c, d), v) : (U32 × U32) × alloc.vec.Vec U32) => ((c.val, d.val), vals v)
  expect "drop_odds" (decode countsR (drop_odds (vec ([1, 2, 3, 4, 6].map u))))
    (.ok ((5, 2), [2, 4, 6]))
  expect "drop_odds, empty" (decode countsR (drop_odds (vec []))) (.ok ((0, 0), []))
  -- a captured vector read and extended by the closure
  expect "first_occurrences"
    (decode (fun ((seen, v) : alloc.vec.Vec U32 × alloc.vec.Vec U32) => (vals seen, vals v))
      (first_occurrences (vec ([3, 1, 3, 2, 1, 4].map u))))
    (.ok ([3, 1, 2, 4], [3, 1, 2, 4]))
