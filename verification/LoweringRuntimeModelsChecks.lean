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

/-- A fresh thread of a fresh process: both flags clear, the hook not installed, no
    raise, counters at zero, the process supply unshared, no profiling clock, and the
    environment's `CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL` reads given by `env`. -/
def fresh (env : Nat → Bool := fun _ => false) : Runtime := {
  u := false, o := false, installed := false, hook := [], builds := 0#u64,
  nextPreflightFrame := 0#u64, hostVar := fun n => ⟨BitVec.ofNat _ n⟩, hostVarsTaken := 0,
  profileEnv := env, profileEnvReads := 0, profileMarks := 0#usize }

/-- A fresh thread after `record_raises`: that function installs the Chelis hook before the
    recorder, so the lower bound `installed` holds. -/
def hooked : Runtime := { fresh with installed := true }

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
`fatal` flag. Each recorded raise is `(u, o, installed, kind)`. The model hook entry gives
`(u, o, installed)`. The payload kind of a raise is the payload of the failure that the raise
returns (`ExtractedScopes.raise_hook_payload`). A caught failure converts to a diagnostic. The
fixture's `unrepresentable_diagnostic` has code `1`, and no scenario raises code `1`. Thus a
caught `Err` with code `1` comes from an `Unrepresentable` failure. -/

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

/-- The payload kind of a raise, as the native recorder reads it. -/
inductive Kind where
  | unrep
  | diag (code : Nat) (fatal : Bool)
  deriving DecidableEq, Repr

/-- The kind of the payload of a failure. -/
def failureKind : LoweringFailure → Kind
  | .Unrepresentable => .unrep
  | .Diagnostic d => .diag d.code.val d.fatal

/-- The kind of the payload of the raise behind a caught `Err`. -/
def caughtKind {α : Type} : core.result.Result α chelis_ir.lower.LowerDiagnostic → Kind
  | .Ok _ => .unrep
  | .Err d => if d.code.val = 1 then .unrep else .diag d.code.val d.fatal

/-- Hook entries with the payload kinds of their raises. -/
def withKinds (raises : List (Bool × Bool × Bool)) (kinds : List Kind) :
    List (Bool × Bool × Bool × Kind) :=
  (raises.zip kinds).map fun ((u, o, i), k) => (u, o, i, k)

/-- The raises of one step, with its result. -/
def observe {α : Type} (step : Runtime → Result (α × Runtime)) (s : Runtime) :
    Except String (α × List (Bool × Bool × Bool) × Runtime) :=
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
  let .ok (r, raises, s) := observe (chelis_ir.speculative_rejection · 9#u32) hooked
    | throw (IO.userError "speculative rejection")
  expect "speculative rejection" (plain r) (.inr (1, false))
  expect "speculative raise" (withKinds raises [caughtKind r]) [(true, true, true, .unrep)]
  let .ok (r, raises, s) := observe (chelis_ir.rejection_after_inner_catch · 9#u32) s
    | throw (IO.userError "rejection after an inner catch")
  expect "rejection after an inner catch" (plain r) (.inr (9, false))
  expect "an inner catch clears o" (withKinds raises [caughtKind r])
    [(false, false, true, .diag 9 false)]
  let .ok (r, raises, s) := observe (chelis_ir.rejection_after_inner_catch · 17#u32) s
    | throw (IO.userError "inner catch result")
  expect "inner catch result" (plain r) (.inl 17)
  expect "no raise" raises []
  let .ok (r, raises, s) := observe (chelis_ir.fatal_reraise · 4#u32) s
    | throw (IO.userError "fatal re-raise")
  expect "fatal re-raise" (plain r) (.inr (4, true))
  -- The re-raise raises the caught diagnostic again (`raiseFatal_spec`).
  expect "fatal re-raise" (withKinds raises [caughtKind r, caughtKind r])
    [(true, true, true, .diag 4 true), (true, true, true, .diag 4 true)]
  let .ok (f, raises, s) := observe (chelis_ir.uncaught_rejection · 5#u32) s
    | throw (IO.userError "uncaught rejection")
  expect "uncaught rejection" (failure f) (some (5, false))
  expect "uncaught raise" (withKinds raises [failureKind f]) [(false, false, true, .diag 5 false)]
  expect "no flag left set" (flags s) (false, false)
  -- catch_threads_state
  let .ok ((r, count), raises, s) := observe (caughtWithState 3#u32 false) hooked
    | throw (IO.userError "caught state")
  expect "caught state" (plain r, count.val) (.inl 4, 4)
  expect "no raise" raises []
  let .ok ((r, count), raises, _) := observe (caughtWithState count true) s
    | throw (IO.userError "caught rejection")
  expect "caught rejection" (plain r, count.val) (.inr (1, false), 5)
  expect "caught raise" (withKinds raises [caughtKind r]) [(false, true, true, .unrep)]
  -- boundary_resume_skips_the_hook
  let .ok (f, raises, _) := observe (chelis_ir.uncaught_rejection · 6#u32) hooked
    | throw (IO.userError "boundary raise")
  expect "boundary raise" (withKinds raises [failureKind f]) [(false, false, true, .diag 6 false)]
  expect "boundary diagnostic" (failure f) (some (6, false))
  expect "resuming leaves the modeled domain"
    (match LoweringFailure.resume_at_boundary f with | .fail _ => true | _ => false) true
  IO.println "lowering_runtime_models: raise and catch scenarios agree"

/-! The native observations of `fatal_raise_before_first_catch_is_visible`. A child process
runs it before any installation of the Chelis hook, so it starts from `fresh`. The previous hook
observes a raise iff the hook is not installed or both flags are clear. The lower bound
`installed` of the model is exact here: only the first catch installs the hook. -/

/-- The raises that the previous hook observes, as `(u, o, kind)`. -/
def visible (raises : List (Bool × Bool × Bool × Kind)) : List (Bool × Bool × Kind) :=
  (raises.filter fun (u, o, i, _) => !(i && (u || o))).map fun (u, o, _, k) => (u, o, k)

#eval do
  let .ok ((first, caught, after, fatal), raises, s) :=
      observe (chelis_ir.fatal_before_first_catch · 7#u32) (fresh)
    | throw (IO.userError "fatal raise before the first catch")
  expect "first fatal raise" (failure first) (some (7, true))
  expect "caught rejection" (plain caught) (.inr (8, false))
  expect "raise after the catch" (failure after) (some (9, false))
  expect "fatal raise after the catch" (failure fatal) (some (10, true))
  let raises := withKinds raises
    [failureKind first, caughtKind caught, failureKind after, failureKind fatal]
  expect "hook entries" raises
    [(false, true, false, .diag 7 true), (false, true, true, .diag 8 false),
      (false, false, true, .diag 9 false), (false, true, true, .diag 10 true)]
  expect "visible raises" (visible raises) [(false, true, .diag 7 true), (false, false, .diag 9 false)]
  expect "a fatal raise outside a catch leaves o set" (flags s) (false, true)
  expect "the catch installed the hook" s.installed true
  IO.println "lowering_runtime_models: hook installation scenario agrees"

/-! The native observations of `host_scopes_thread_the_context`: `scoped` on a
context holding mutable borrows, with a grown stack on a tuple of them. -/

def hostScopes (checked : Nat) (s : chelis_ir.Scope) (native : Nat) (reject : Bool)
    (rt : Runtime) :
    Result ((core.result.Result Usize chelis_ir.lower.LowerDiagnostic × chelis_ir.Scope × U32) ×
      Runtime) := do
  let (r, s, rt, native) ←
    chelis_ir.host_scopes ⟨BitVec.ofNat _ checked⟩ s rt ⟨BitVec.ofNat _ native⟩ reject
  ok ((r, s, native), rt)

/-- A result with its diagnostic reduced to `(code, fatal)`. -/
def plainUsize : core.result.Result Usize chelis_ir.lower.LowerDiagnostic → Sum Nat (Nat × Bool)
  | .Ok v => .inl v.val
  | .Err d => .inr (d.code.val, d.fatal)

#eval do
  -- host_scopes_thread_the_context
  let .ok ((r, s, native), raises, rt) := observe (hostScopes 7 (scope [1] 0) 4 false) hooked
    | throw (IO.userError "host scope")
  expect "host scope" (plainUsize r) (.inl 3)
  expect "host scope state" (contents s, native.val) (([1, 7], 2), 5)
  expect "host scope raises" raises []
  let .ok ((r, s, native), raises, rt) := observe (hostScopes 9 s native.val true) rt
    | throw (IO.userError "host scope rejection")
  expect "host scope rejection" (plainUsize r) (.inr (1, false))
  expect "host scope state after the rejection" (contents s, native.val) (([1, 7, 9], 5), 6)
  expect "host scope raise" (withKinds raises [caughtKind r]) [(false, true, true, .unrep)]
  expect "no flag left set" (flags rt) (false, false)
  IO.println "lowering_runtime_models: borrowing context scenarios agree"

end RaiseCatch

end LoweringRuntimeModelsChecks
