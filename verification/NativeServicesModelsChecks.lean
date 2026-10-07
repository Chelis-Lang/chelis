import NativeServicesModels

open Aeneas Aeneas.Std Result

namespace NativeServicesModelsChecks

/-! Mirrors the native tests of `native_services_models.rs` on its extraction, whose
primitives Aeneas replaced by the models of `Aeneas/Std/ChelisNative.lean`. A native probe on
ample stack answers `false` and one on a 64 KiB `stacker` segment answers `true`, so the
schedules below are constant; profiling marks are the clock indices when profiling is on. -/

abbrev Services := chelis_types.native.NativeServices

def expect [DecidableEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual = expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def value {α : Type} : Result α → IO α
  | .ok value => pure value
  | .fail _ => throw (IO.userError "failure")
  | .div => throw (IO.userError "divergence")

/-- `NativeServices::new(..)`: depth 0, every clock at 0, the given oracles. -/
def services (cancelled exhausted : Bool) (profiling : Bool := false) : Services := {
  cancel := fun _ => cancelled, cancelTaken := 0, stack := fun _ => exhausted, stackTaken := 0,
  depth := 0#usize, profileMark := fun n => if profiling then ⟨BitVec.ofNat _ n⟩ else 0#usize,
  profileTaken := 0 }

def depth (s : Services) : Nat := s.depth.val

#eval do
  -- frames_on_ample_stack
  let (bail, s) ← value (chelis_types.scenarios.frame_exhausted (services false false))
  expect "ample stack: no bail" bail false
  expect "ample stack: one frame" (depth s) 1
  let (bails, s) ← value (chelis_types.scenarios.descend s 4#u32)
  expect "ample stack: descend" bails 0#u32
  expect "ample stack: five frames" (depth s) 5
  let s ← value (chelis_types.scenarios.ascend s 7#u32)
  expect "exits saturate" (depth s) 0
  let (_, s) ← value (chelis_types.scenarios.descend s 3#u32)
  let s ← value (chelis_types.scenarios.enter_outermost_scope s)
  expect "the outermost scope resets the depth" (depth s) 0
  -- frames_on_a_small_segment
  let (bail, s) ← value (chelis_types.scenarios.frame_exhausted (services false true))
  expect "small segment: bail" bail true
  expect "small segment: the frame is exited" (depth s) 0
  let (bails, s) ← value (chelis_types.scenarios.descend s 3#u32)
  expect "small segment: every probe bails" bails 3#u32
  expect "small segment: no frame stays" (depth s) 0
  -- grown_entry_applies_the_closure
  let (result, s) ← value (chelis_types.scenarios.grown_entry (services true true) 2#u32)
  expect "grown entry on a small segment" result (2#u32, true)
  expect "grown entry on a small segment: depth" (depth s) 0
  let (result, s) ← value (chelis_types.scenarios.grown_entry (services false false) 2#u32)
  expect "grown entry on the production segment" result (0#u32, false)
  expect "grown entry on the production segment: depth" (depth s) 2
  -- cancellation_polls
  let (observed, _) ← value (chelis_types.scenarios.polls (services false false))
  expect "no cancellation" observed (false, false, false)
  let (observed, _) ← value (chelis_types.scenarios.polls (services true false))
  expect "cancelled" observed (true, true, true)
  -- profiling_marks
  let (marks, _) ← value (chelis_types.scenarios.profiled (services false false))
  expect "profiling off" marks (0#usize, 0#usize)
  let (marks, s) ← value (chelis_types.scenarios.profiled (services false false (profiling := true)))
  expect "profiling on" marks (0#usize, 1#usize)
  let (marks, _) ← value (chelis_types.scenarios.profiled s)
  expect "profiling on, next scopes" marks (2#usize, 3#usize)
  IO.println "native_services_models: extracted scenarios agree"

end NativeServicesModelsChecks
