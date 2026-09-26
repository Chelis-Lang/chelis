import StartFromRoots

open Aeneas Aeneas.Std

namespace StartFromRootsChecks

/-! Mirrors the native test of `start_from_roots.rs`: each canonical start-from
    pattern selected its root (a missing root is an unknown constant here). -/

def expect [DecidableEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual = expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def withCounter (r : Result (U32 × start_from_roots.Counter)) : Except String (Nat × Nat) :=
  decode (fun ((out, c) : U32 × start_from_roots.Counter) => (out.val, c.value.val)) r

#eval do
  expect "inherent impl method `{impl T}::bump`"
    (withCounter (start_from_roots.Counter.bump { value := 4#u32 })) (.ok (5, 5))
  expect "trait impl method `{impl Tr<u32> for T}::scale`"
    (decode UScalar.val
      (start_from_roots.Counter.Insts.Start_from_rootsScaleU32.scale { value := 5#u32 } 3#u32))
    (.ok 15)
  expect "`_#1::{impl T}::reset` in an anonymous constant"
    (withCounter (start_from_roots.__1.Counter.reset { value := 5#u32 })) (.ok (5, 0))
  expect "closure#10's Fn::call, the last step of offsets(1)"
    (decode UScalar.val
      (start_from_roots.offsets.closure_10.Insts.CoreOpsFunctionFnTupleU32U32.call () 46#u32))
    (.ok 460)
  expect "comma-joined roots"
    (decode UScalar.val (start_from_roots.plain 3#u32), decode UScalar.val (start_from_roots.other 3#u32))
    (.ok 4, .ok 6)
  IO.println "start_from_roots: native and Lean agree"

end StartFromRootsChecks
