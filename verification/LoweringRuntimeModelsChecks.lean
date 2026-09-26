import LoweringRuntimeModels

open Aeneas Aeneas.Std Result

namespace LoweringRuntimeModelsChecks

/-! Mirrors the native tests of `lowering_runtime_models.rs` on its extraction,
whose primitives Aeneas replaced by the models of
`Aeneas/Std/ChelisNative.lean`. -/

abbrev Runtime := chelis_ir.lowering_runtime.LoweringRuntime

def expect [DecidableEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual = expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def value {α : Type} : Result α → Except String α
  | .ok value => .ok value
  | .fail _ => .error "failure"
  | .div => .error "divergence"

/-- A fresh thread of a fresh process: both flags clear, no raise, counters at
    zero, the process supply unshared, no profiling clock, and the environment's
    `CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL` reads given by `env`. -/
def fresh (env : Nat → Bool := fun _ => false) : Runtime := {
  u := false, o := false, hook := [], builds := 0#u64, nextPreflightFrame := 0#u64,
  hostVar := fun n => ⟨BitVec.ofNat _ n⟩, hostVarsTaken := 0,
  profileEnv := env, profileEnvReads := 0, profileMarks := 0#usize }

def run {α : Type} (step : Runtime → Result (α × Runtime)) (s : Runtime) :
    Except String (α × Runtime) :=
  value (step s)

def flags (s : Runtime) : Bool × Bool := (s.u, s.o)

/-! ## Extracted scenarios -/

#eval do
  -- speculation_scopes_set_and_clear_u
  let .ok (outside, s) := run chelis_ir.speculation_flags (fresh) | throw (IO.userError "speculation")
  expect "speculation flags" outside (false, true, false)
  let nested : core.ops.function.FnOnceMutArg (Runtime → Result ((Bool × Bool × Bool) × Runtime))
      Runtime (Bool × Bool × Bool) := { call_once := fun f s => f s }
  let .ok (inside, s) :=
      run (fun s => chelis_ir.lowering_runtime.LoweringRuntime.with_suppress_unrepresentable_panic
        nested s chelis_ir.speculation_flags) s | throw (IO.userError "nested speculation")
  expect "an inner scope clears the enclosing flag" inside (true, true, false)
  expect "no flag left set" (flags s) (false, false)
  -- summary_probes_saturate
  let .ok (probes, s) := run (chelis_ir.summary_probes · 3#u32) (fresh) | throw (IO.userError "probes")
  expect "summary probes" probes (3#u64, 0#u64)
  let .ok (probes, _) := run (chelis_ir.summary_probes · 3#u32) { s with builds := 18446744073709551614#u64 }
    | throw (IO.userError "saturated probes")
  expect "summary probes saturate" probes (18446744073709551615#u64, 0#u64)
  -- preflight_frames_wrap
  let .ok (frames, s) := run chelis_ir.preflight_frames (fresh) | throw (IO.userError "frames")
  expect "preflight frames" frames (0#u64, 1#u64, 2#u64)
  let .ok (frames, s) := run chelis_ir.preflight_frames s | throw (IO.userError "frames")
  expect "preflight frames continue" frames (3#u64, 4#u64, 5#u64)
  let .ok (frames, _) := run chelis_ir.preflight_frames { s with nextPreflightFrame := 18446744073709551614#u64 }
    | throw (IO.userError "wrapped frames")
  expect "preflight frames wrap" frames (18446744073709551614#u64, 18446744073709551615#u64, 0#u64)
  -- host_inference_vars_are_consecutive
  let .ok (vars, s) := run chelis_ir.host_inference_vars (fresh) | throw (IO.userError "vars")
  expect "host inference vars" vars (0#u32, 1#u32)
  let .ok (vars, _) := run chelis_ir.host_inference_vars s | throw (IO.userError "vars")
  expect "host inference vars continue" vars (2#u32, 3#u32)
  IO.println "lowering_runtime_models: extracted scenarios agree"

def totals (l : List Nat) : Slice Usize :=
  let l := l.map fun n => (⟨BitVec.ofNat _ n⟩ : Usize)
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def usizes (t : Usize × Usize × Usize) : Nat × Nat × Nat := (t.1.val, t.2.1.val, t.2.2.val)

#eval do
  -- profile_marks_and_declarations: the environment reads disabled, enabled,
  -- enabled, then `0`.
  let s := fresh (fun n => n = 1 ∨ n = 2)
  let declarations (l : List Nat) := fun s => chelis_ir.profile_declarations s (totals l)
  let unissued (mark total : Nat) := fun s =>
    chelis_ir.profile_unissued_mark s ⟨BitVec.ofNat _ mark⟩ ⟨BitVec.ofNat _ total⟩
  let .ok (profile, s) := run (declarations [4, 9]) s | throw (IO.userError "profile")
  expect "disabled profile" (usizes profile) (0, 0, 0)
  let .ok (last, s) := run (unissued 1 8) s | throw (IO.userError "profile")
  expect "no clock before the first mark" last.val 5
  let .ok (profile, s) := run (declarations [4, 9]) s | throw (IO.userError "profile")
  expect "enabled profile" (usizes profile) (1, 3, 9)
  let .ok (profile, s) := run (declarations []) s | throw (IO.userError "profile")
  expect "enabled profile without declarations" (usizes profile) (4, 0, 0)
  let .ok (profile, s) := run (declarations [7]) s | throw (IO.userError "profile")
  expect "profile disabled by 0" (usizes profile) (0, 0, 0)
  let .ok (last, s) := run (unissued 4 8) s | throw (IO.userError "profile")
  expect "issued mark" last.val 8
  let .ok (last, s) := run (unissued 5 8) s | throw (IO.userError "profile")
  expect "unissued mark" last.val 5
  let .ok (last, _) := run (unissued 0 8) s | throw (IO.userError "profile")
  expect "mark 0" last.val 5
  IO.println "lowering_runtime_models: profile scenarios agree"

def scope (items : List Nat) (exits : Nat) : chelis_ir.Scope :=
  let items := items.map fun n => (⟨BitVec.ofNat _ n⟩ : U32)
  { items := if h : items.length ≤ Usize.max then ⟨items, h⟩ else ⟨[], by simp⟩,
    exits := ⟨BitVec.ofNat _ exits⟩ }

def contents (s : chelis_ir.Scope) : List Nat × Nat := (s.items.val.map (·.val), s.exits.val)

#eval do
  -- scoped_runs_enter_body_exit
  let .ok (length, s) := value (chelis_ir.scoped_push (scope [1] 0) 5#u32)
    | throw (IO.userError "scoped")
  expect "scoped result" length.val 3
  expect "scoped state" (contents s) ([1, 5], 2)
  let .ok (length, s) := value (chelis_ir.grown_push s 8#u32) | throw (IO.userError "grown")
  expect "grown result" (match length with | .Ok n => some n.val | .Err _ => none) (some 4)
  expect "grown state" (contents s).1 [1, 5, 8]
  IO.println "lowering_runtime_models: scope scenarios agree"

/-! ## Extracted raise and catch scenarios

The native observations of `raises_record_their_flags`, `catch_threads_state`
and `boundary_resume_skips_the_hook`, with a diagnostic reduced to its code and
`fatal` flag. -/

section RaiseCatch
open chelis_ir.lowering_runtime

/-- A result with its diagnostic reduced to `(code, fatal)`. -/
def plain : core.result.Result U32 chelis_ir.lower.LowerDiagnostic → Sum Nat (Nat × Bool)
  | .Ok v => .inl v.val
  | .Err d => .inr (d.code.val, d.fatal)

/-- A failure's diagnostic as `(code, fatal)`, `none` when unrepresentable. -/
def failure : LoweringFailure → Option (Nat × Bool)
  | .Unrepresentable => none
  | .Diagnostic d => some (d.code.val, d.fatal)

/-- The raises of one step, with its result. -/
def observe {α : Type} (step : Runtime → Result (α × Runtime)) (s : Runtime) :
    Except String (α × List (Bool × Bool) × Runtime) :=
  match step { s with hook := [] } with
  | .ok (a, s) => .ok (a, s.hook, s)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def caughtWithState (count : U32) (reject : Bool) (s : Runtime) :
    Result ((core.result.Result U32 chelis_ir.lower.LowerDiagnostic × U32) × Runtime) := do
  let (r, s, count) ← chelis_ir.caught_with_state s count reject
  ok ((r, count), s)

#eval do
  -- raises_record_their_flags
  let .ok (r, raises, s) := observe (chelis_ir.speculative_rejection · 9#u32) (fresh)
    | throw (IO.userError "speculative rejection")
  expect "speculative rejection" (plain r) (.inr (1, false))
  expect "speculative raise flags" raises [(true, true)]
  let .ok (r, raises, s) := observe (chelis_ir.rejection_after_inner_catch · 9#u32) s
    | throw (IO.userError "rejection after an inner catch")
  expect "rejection after an inner catch" (plain r) (.inr (9, false))
  expect "an inner catch clears o" raises [(false, false)]
  let .ok (r, raises, s) := observe (chelis_ir.rejection_after_inner_catch · 17#u32) s
    | throw (IO.userError "inner catch result")
  expect "inner catch result" (plain r) (.inl 17)
  expect "no raise" raises []
  let .ok (r, raises, s) := observe (chelis_ir.fatal_reraise · 4#u32) s
    | throw (IO.userError "fatal re-raise")
  expect "fatal re-raise" (plain r) (.inr (4, true))
  expect "fatal re-raise flags" raises [(true, true), (true, true)]
  let .ok (f, raises, s) := observe (chelis_ir.uncaught_rejection · 5#u32) s
    | throw (IO.userError "uncaught rejection")
  expect "uncaught rejection" (failure f) (some (5, false))
  expect "uncaught raise flags" raises [(false, false)]
  expect "no flag left set" (flags s) (false, false)
  -- catch_threads_state
  let .ok ((r, count), raises, s) := observe (caughtWithState 3#u32 false) (fresh)
    | throw (IO.userError "caught state")
  expect "caught state" (plain r, count.val) (.inl 4, 4)
  expect "no raise" raises []
  let .ok ((r, count), raises, _) := observe (caughtWithState count true) s
    | throw (IO.userError "caught rejection")
  expect "caught rejection" (plain r, count.val) (.inr (1, false), 5)
  expect "caught raise flags" raises [(false, true)]
  -- boundary_resume_skips_the_hook
  let .ok (f, raises, _) := observe (chelis_ir.uncaught_rejection · 6#u32) (fresh)
    | throw (IO.userError "boundary raise")
  expect "boundary raise flags" raises [(false, false)]
  expect "boundary diagnostic" (failure f) (some (6, false))
  expect "resuming leaves the modeled domain"
    (match LoweringFailure.resume_at_boundary f with | .fail _ => true | _ => false) true
  IO.println "lowering_runtime_models: raise and catch scenarios agree"

end RaiseCatch

end LoweringRuntimeModelsChecks
