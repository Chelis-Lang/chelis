import ChelisBridge.Runtime.Scopes
import ChelisBridge.Model

/-!
# E. The scope obligations on the extracted production combinators

`Runtime/Scopes.lean` fixed the properties every scope combinator of the semantic core must
satisfy. This module discharges them against the definitions Aeneas extracted from the
production crates (strict flags; the native shell `scoped` / `catch_native_unwind` /
`observe_raise` / `resume_native_unwind` and the flag primitives are hard-opaque and replaced
by their `Aeneas/Std/ChelisNative.lean` models, which are part of the trusted base):

* lowering (`chelis_ir::lowering_runtime`):
  - `LoweringRuntime::with_suppress_unrepresentable_panic` sets the speculation flag `u` at
    entry and CLEARS it to `false` at exit (`withSuppress_spec`, `withSuppress_obligation`:
    the literal `SuppressionObligation`), even inside an enclosing speculative scope
    (`withSuppress_nested_clears`);
  - `LoweringRuntime::catch_lowering_with` runs the body with the quiet-output flag `o` set,
    threads the explicit state, CLEARS `o` at exit, keeps `Ok` and maps a lowering failure to
    exactly the diagnostic `LoweringFailure::into_diagnostic` gives (`catchLoweringWith_spec`,
    `catchLoweringWith_clears`); `catch_lowering` is the stateless instance
    (`catchLowering_spec`);
  - `raise_lowering_diagnostic` records exactly one hook observation of the current flags and
    turns a nonfatal rejection under speculation into `Unrepresentable`
    (`raiseLowering_spec`); `raise_fatal_lowering_diagnostic` explicitly RE-ENABLES quiet
    output and always raises the structured fatal diagnostic (`raiseFatal_spec`);
  - `LoweringFailure::resume_at_boundary` never returns (`resumeAtBoundary_never_ok`): a
    boundary re-raise is an unwind, never a success or an ordinary rejection;
* checking (`chelis_types::session::DiagnosticSink`):
  - `with_opacity_context` installs the policy for the body and RESTORES the enclosing policy
    (`withOpacity_spec`, `withOpacity_obligation`: the literal `InstallRestoreObligation`);
  - `with_recursive_caller` with no caller is the body itself; with a caller it attributes the
    active group to it for the body and restores the enclosing caller
    (`withRecursiveCaller_none`, `withRecursiveCaller_some`);
  - `with_recursive_inference` runs the body in the fresh frame (the value the extracted
    `RecursiveState::default` returns: no group, nothing pinned) and RESTORES the enclosing frame
    (`withRecursiveInference_spec`, `withRecursiveInference_obligation`: the literal
    `InstallRestoreObligation`).

Every statement quantifies over every closure type and `FnOnce` instance (so over every body
the production code can pass) and covers `Ok` and `Err` results alike; a failing or diverging
body (a native unwind is outside the modeled domain) makes the combinator fail too. The
`*_obligation` theorems instantiate the `Scopes` obligations literally, with bodies given as
state transformers through one canonical instance.

The host lowering scopes (`with_host_invocation`, `with_mono_probe`, `with_inlining`) are in
`Runtime/ExtractedHostScopes.lean`. Not yet in the model (not extraction roots of the bridge
model): `host::with_type_subst` (`ActiveTypeSubstGuard`, an `InstallRestoreObligation`) and
`host::with_preflight` (`PreflightObligation`).
-/

open Aeneas.Std
open_chelis_model

namespace ChelisBridge.Runtime.ExtractedScopes

open ChelisBridge.Runtime.Scopes

theorem bind_ok_inv {α β : Type} {x : Result α} {f : α → Result β} {b : β}
    (h : (x >>= f) = .ok b) : ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x <;> simp_all

/-! ## Lowering runtime -/

/-- The explicit lowering state the extracted code threads. -/
abbrev RT := Aeneas.Std.chelis_ir.lowering_runtime.LoweringRuntime

/-- The successful-exit view of an extracted `(result, state)` pair. -/
def okRS {σ γ : Type} : Result (γ × σ) → Option (σ × γ)
  | .ok (r, s) => some (s, r)
  | _ => none

/-- The successful-exit view of the `Scopes` obligations in the Aeneas monad. -/
def okResult {σ γ : Type} : Result (σ × γ) → Option (σ × γ)
  | .ok x => some x
  | _ => none

/-- The speculation flag `u`. -/
def uLens : Lens RT Bool := { get := (·.u), set := fun b s => { s with u := b } }

/-- The quiet-output flag `o`. -/
def oLens : Lens RT Bool := { get := (·.o), set := fun b s => { s with o := b } }

theorem uLens_lawful : uLens.Lawful := ⟨fun _ _ => rfl, fun _ => rfl, fun _ _ _ => rfl⟩

theorem oLens_lawful : oLens.Lawful := ⟨fun _ _ => rfl, fun _ => rfl, fun _ _ _ => rfl⟩

/-- **`with_suppress_unrepresentable_panic`, exactly.** For every closure `f` and instance:
run `f` with `u := true`, then set `u := false` whatever `f` returned. -/
theorem withSuppress_eq {R T1 : Type} (inst : core.ops.function.FnOnceMutArg T1 RT R) (s : RT)
    (f : T1) :
    chelis_ir.lowering_runtime.LoweringRuntime.with_suppress_unrepresentable_panic inst s f =
      (do
        let (r, s₁) ← inst.call_once f { s with u := true }
        .ok (r, { s₁ with u := false })) := by
  rfl

/-- **Suppression obligation, every instance.** -/
theorem withSuppress_spec {R T1 : Type} (inst : core.ops.function.FnOnceMutArg T1 RT R) :
    ScopeSpecOn (fun f s => okRS (inst.call_once f s))
      (fun f s => okRS
        (chelis_ir.lowering_runtime.LoweringRuntime.with_suppress_unrepresentable_panic inst s f))
      (uLens.set true) (clearExit uLens) id := by
  intro f s
  simp only [withSuppress_eq, uLens, clearExit]
  cases inst.call_once f { s with u := true } with
  | ok x => rfl
  | fail e => rfl
  | div => rfl

/-- The canonical instance running a state-transformer body. -/
def rtBody (γ : Type) : core.ops.function.FnOnceMutArg (RT → Result (RT × γ)) RT γ :=
  ⟨fun b s => (fun x => (x.2, x.1)) <$> b s⟩

/-- The extracted `with_suppress_unrepresentable_panic` as a `Scopes.Combinator`. -/
def suppressComb (γ : Type) : Combinator Result RT γ := fun body s =>
  (fun x => (x.2, x.1)) <$>
    chelis_ir.lowering_runtime.LoweringRuntime.with_suppress_unrepresentable_panic (rtBody γ) s body

/-- **The literal Section E `SuppressionObligation`** for the extracted
`with_suppress_unrepresentable_panic`. -/
theorem withSuppress_obligation (γ : Type) :
    SuppressionObligation okResult uLens (suppressComb γ) := by
  intro body s
  simp only [suppressComb, withSuppress_eq, rtBody, uLens, clearExit]
  cases body { s with u := true } with
  | ok x => rfl
  | fail e => rfl
  | div => rfl

/-- After every successful exit of the extracted scope, `u = false`, even when an enclosing
speculative scope had set it (clear, not restore). -/
theorem withSuppress_nested_clears {R T1 : Type} (inst : core.ops.function.FnOnceMutArg T1 RT R)
    (f : T1) (s s' : RT) (r : R)
    (h : chelis_ir.lowering_runtime.LoweringRuntime.with_suppress_unrepresentable_panic inst s f =
      .ok (r, s')) :
    s'.u = false :=
  clears_of_specOn uLens_lawful (withSuppress_spec inst) f s s' r (by simp [okRS, h])

/-- The diagnostic a lowering failure maps to is total: `into_diagnostic` never fails, and a
structured failure keeps its diagnostic. -/
theorem intoDiagnostic_ok (e : chelis_ir.lowering_runtime.LoweringFailure) :
    ∃ d, chelis_ir.lowering_runtime.LoweringFailure.into_diagnostic e = .ok d := by
  cases e with
  | Unrepresentable => exact ⟨_, rfl⟩
  | Diagnostic d => exact ⟨d, rfl⟩

theorem intoDiagnostic_diagnostic (d : chelis_ir.lower.LowerDiagnostic) :
    chelis_ir.lowering_runtime.LoweringFailure.into_diagnostic (.Diagnostic d) = .ok d := rfl

/-- **Output guard of a catch.** On every successful exit (`Ok` or mapped `Err`), the quiet
output flag is `false`; the body ran with it set and with the caller's explicit state; an `Ok`
result is the body's and an `Err` is the diagnostic of the body's failure. -/
theorem catchLoweringWith_spec {S R T2 : Type}
    (inst : core.ops.function.FnOnceMutArgMutArg T2 RT S
      (core.result.Result R chelis_ir.lowering_runtime.LoweringFailure))
    (rt rt' : RT) (st st' : S) (f : T2)
    (r' : core.result.Result R chelis_ir.lower.LowerDiagnostic)
    (h : chelis_ir.lowering_runtime.LoweringRuntime.catch_lowering_with inst rt st f =
      .ok (r', rt', st')) :
    ∃ r rt₁, inst.call_once f ({ rt with o := true }, st) = .ok (r, rt₁, st') ∧
      rt' = { rt₁ with o := false } ∧ rt'.o = false ∧
      (∀ x, r' = .Ok x ↔ r = .Ok x) ∧
      (∀ d, r' = .Err d ↔
        ∃ e, r = .Err e ∧ chelis_ir.lowering_runtime.LoweringFailure.into_diagnostic e = .ok d) := by
  obtain ⟨⟨r₁, rt₁, st₁⟩, h1, h2⟩ := bind_ok_inv h
  have h2' : Result.ok (r₁, { rt₁ with o := false }, st₁) = Result.ok (r', rt', st') := h2
  simp only [Result.ok.injEq, Prod.mk.injEq] at h2'
  obtain ⟨rfl, rfl, rfl⟩ := h2'
  obtain ⟨⟨r, rt₂, st₂⟩, hb, h3⟩ := bind_ok_inv h1
  have hb' : inst.call_once f ({ rt with o := true }, st) = .ok (r, rt₂, st₂) := hb
  cases r with
  | Ok x =>
    have h3' : Result.ok (core.result.Result.Ok x, rt₂, st₂) = Result.ok (r₁, rt₁, st₁) := h3
    simp only [Result.ok.injEq, Prod.mk.injEq] at h3'
    obtain ⟨rfl, rfl, rfl⟩ := h3'
    refine ⟨.Ok x, rt₂, hb', rfl, rfl, fun y => ?_, fun d => ?_⟩
    · constructor <;> intro e <;> cases e <;> rfl
    · constructor
      · intro e; cases e
      · rintro ⟨e, he, -⟩; cases he
  | Err e =>
    obtain ⟨d, hd⟩ := intoDiagnostic_ok e
    have h3' : Result.ok (core.result.Result.Err d, rt₂, st₂) = Result.ok (r₁, rt₁, st₁) := by
      rw [← h3]
      simp [core.result.Result.map_err, BuiltinFnOnce, hd]
    simp only [Result.ok.injEq, Prod.mk.injEq] at h3'
    obtain ⟨rfl, rfl, rfl⟩ := h3'
    refine ⟨.Err e, rt₂, hb', rfl, rfl, fun y => ?_, fun d' => ?_⟩
    · constructor <;> intro e' <;> cases e'
    · constructor
      · intro e'
        cases e'
        exact ⟨e, rfl, hd⟩
      · rintro ⟨e', he', hd'⟩
        cases he'
        rw [hd] at hd'
        cases hd'
        rfl

/-- The catch's output flag is cleared on every successful exit (clear, not restore). -/
theorem catchLoweringWith_clears {S R T2 : Type}
    (inst : core.ops.function.FnOnceMutArgMutArg T2 RT S
      (core.result.Result R chelis_ir.lowering_runtime.LoweringFailure))
    (rt rt' : RT) (st st' : S) (f : T2)
    (r' : core.result.Result R chelis_ir.lower.LowerDiagnostic)
    (h : chelis_ir.lowering_runtime.LoweringRuntime.catch_lowering_with inst rt st f =
      .ok (r', rt', st')) :
    rt'.o = false := by
  obtain ⟨-, -, -, -, h, -⟩ := catchLoweringWith_spec inst rt rt' st st' f r' h
  exact h

/-- `catch_lowering` is `catch_lowering_with` on the unit state. -/
theorem catchLowering_spec {R T1 : Type}
    (inst : core.ops.function.FnOnceMutArg T1 RT
      (core.result.Result R chelis_ir.lowering_runtime.LoweringFailure))
    (rt rt' : RT) (f : T1) (r' : core.result.Result R chelis_ir.lower.LowerDiagnostic)
    (h : chelis_ir.lowering_runtime.LoweringRuntime.catch_lowering inst rt f = .ok (r', rt')) :
    ∃ r rt₁, inst.call_once f { rt with o := true } = .ok (r, rt₁) ∧
      rt' = { rt₁ with o := false } ∧ rt'.o = false ∧
      (∀ x, r' = .Ok x ↔ r = .Ok x) ∧
      (∀ d, r' = .Err d ↔
        ∃ e, r = .Err e ∧ chelis_ir.lowering_runtime.LoweringFailure.into_diagnostic e = .ok d) := by
  unfold chelis_ir.lowering_runtime.LoweringRuntime.catch_lowering at h
  obtain ⟨⟨r'', rt'', u⟩, hw, h⟩ := bind_ok_inv h
  cases h
  obtain ⟨r, rt₁, hc, h1, h2, h3, h4⟩ := catchLoweringWith_spec _ rt _ () u f r'' hw
  obtain ⟨⟨r₀, rt₀⟩, hb, hc⟩ := bind_ok_inv hc
  have hc' : Result.ok (r₀, rt₀, ()) = Result.ok (r, rt₁, u) := hc
  simp only [Result.ok.injEq, Prod.mk.injEq] at hc'
  obtain ⟨rfl, rfl, -⟩ := hc'
  exact ⟨r₀, rt₀, hb, h1, h2, h3, h4⟩

/-- **Raising.** Every raise records exactly one hook observation of the current flags before
anything else; a nonfatal rejection under speculation becomes `Unrepresentable`, any other
rejection stays the structured diagnostic. -/
theorem raiseLowering_spec (s : RT) (d : chelis_ir.lower.LowerDiagnostic) :
    chelis_ir.lowering_runtime.LoweringRuntime.raise_lowering_diagnostic s d =
      .ok ((if !d.fatal && s.u then .Unrepresentable else .Diagnostic d),
        { s with hook := s.hook ++ [(s.u, s.o)] }) := by
  unfold chelis_ir.lowering_runtime.LoweringRuntime.raise_lowering_diagnostic
  cases hf : d.fatal <;> cases hu : s.u <;> simp [hu,
    chelis_ir.lowering_runtime.LoweringRuntime.raise_unrepresentable,
    Aeneas.Std.chelis_ir.lowering_runtime.LoweringRuntime.observe_raise,
    Aeneas.Std.chelis_ir.lowering_runtime.LoweringRuntime.unrepresentable_panic_suppressed]

/-- **Fatal re-raise re-enables quiet output.** `raise_fatal_lowering_diagnostic` sets `o := true`
(after an inner catch cleared it), makes the diagnostic fatal, observes the hook with
`o = true`, and always returns the structured diagnostic (never `Unrepresentable`, whatever
`u`). -/
theorem raiseFatal_spec (s : RT) (d : chelis_ir.lower.LowerDiagnostic) :
    chelis_ir.lowering_runtime.LoweringRuntime.raise_fatal_lowering_diagnostic s d =
      .ok (.Diagnostic { d with fatal := true },
        { s with o := true, hook := s.hook ++ [(s.u, true)] }) := by
  unfold chelis_ir.lowering_runtime.LoweringRuntime.raise_fatal_lowering_diagnostic
  cases hf : d.fatal <;>
    simp [raiseLowering_spec, chelis_ir.lower.LowerDiagnostic.impl.fatal,
      Aeneas.Std.chelis_ir.lowering_runtime.LoweringRuntime.set_output_suppressed]

/-- **A boundary re-raise never returns.** `resume_at_boundary` of any failure is a native
unwind (the model's failure outside the modeled domain), never a success. -/
theorem resumeAtBoundary_never_ok (e : chelis_ir.lowering_runtime.LoweringFailure) :
    chelis_ir.lowering_runtime.LoweringFailure.resume_at_boundary e = .fail .panic := by
  cases e <;> rfl

/-! ## Checker diagnostic sink -/

/-- The checker's explicit session state. -/
abbrev Sink := chelis_types.session.DiagnosticSink

/-- The successful-exit view of an extracted sink scope `(result, sink, backward function)`. -/
def okSink {γ : Type} : Result (γ × Sink × (Sink → Sink)) → Option (Sink × γ)
  | .ok (r, s, _) => some (s, r)
  | _ => none

/-- The opacity policy of the sink. -/
def opacityLens : Lens Sink (Option chelis_types.opacity.OpacityContextData) :=
  { get := (·.opacity), set := fun v s => { s with opacity := v } }

theorem opacityLens_lawful : opacityLens.Lawful := ⟨fun _ _ => rfl, fun _ => rfl, fun _ _ _ => rfl⟩

/-- **`with_opacity_context`: install, then restore, every instance.** -/
theorem withOpacity_spec {R T2 : Type} (inst) (ctx : chelis_types.opacity.OpacityContextData) :
    ScopeSpecOn (fun (b : T2) s => okSink (γ := R) (inst.call_once b s))
      (fun b s => okSink (chelis_types.session.DiagnosticSink.with_opacity_context inst s ctx b))
      (opacityLens.set (some ctx)) (restoreExit opacityLens) id := by
  intro b s
  unfold chelis_types.session.DiagnosticSink.with_opacity_context
  simp only [opacityLens, restoreExit]
  cases h : inst.call_once b { s with opacity := some ctx } with
  | ok x => obtain ⟨r, s₁, bk⟩ := x; simp [core.option.Option.replace, h, okSink]
  | fail e => simp [core.option.Option.replace, h, okSink]
  | div => simp [core.option.Option.replace, h, okSink]

/-- The extracted `with_opacity_context` as a `Scopes.Combinator` (state-transformer bodies). -/
def opacityComb (ctx : chelis_types.opacity.OpacityContextData) (γ : Type) :
    Combinator Result Sink γ := fun body s =>
  do
    let (r, s', _) ←
      chelis_types.session.DiagnosticSink.with_opacity_context (T2 := Sink → Result (Sink × γ))
        ⟨fun b s => do let (s', r) ← b s; .ok (r, s', fun _ => s')⟩ s ctx body
    .ok (s', r)

/-- **The literal Section E `InstallRestoreObligation`** for the extracted
`with_opacity_context`. -/
theorem withOpacity_obligation (ctx : chelis_types.opacity.OpacityContextData) (γ : Type) :
    InstallRestoreObligation okResult opacityLens (some ctx) (opacityComb ctx γ) := by
  intro body s
  unfold opacityComb chelis_types.session.DiagnosticSink.with_opacity_context
  simp only [opacityLens, restoreExit]
  cases h : body { s with opacity := some ctx } with
  | ok x => obtain ⟨s₁, r⟩ := x; simp [core.option.Option.replace, h, okResult]
  | fail e => simp [core.option.Option.replace, h, okResult]
  | div => simp [core.option.Option.replace, h, okResult]

/-- The caller of the active recursive group, if any. -/
def groupCaller (s : Sink) : Option chelis_types.infer.recursion.Caller :=
  s.recursive.group.bind (·.caller)

/-- Attribute the active recursive group (if any) to `c`. -/
def setGroupCaller (c : Option chelis_types.infer.recursion.Caller) (s : Sink) : Sink :=
  { s with recursive := { s.recursive with group := s.recursive.group.map ({ · with caller := c }) } }

/-- `RecursiveState::replace_caller`, exactly: replace the active group's caller, returning the
previous one; without an active group nothing changes and `None` is returned. -/
theorem replaceCaller_eq (r : chelis_types.infer.recursion.RecursiveState)
    (c : Option chelis_types.infer.recursion.Caller) :
    chelis_types.infer.recursion.RecursiveState.replace_caller r c =
      .ok (r.group.bind (·.caller), { r with group := r.group.map ({ · with caller := c }) }) := by
  rcases r with ⟨_ | g, p⟩ <;> rfl

/-- **`with_recursive_caller` without a caller** is the body itself. -/
theorem withRecursiveCaller_none {R T2 : Type} (inst) :
    ScopeSpecOn (fun (b : T2) s => okSink (γ := R) (inst.call_once b s))
      (fun b s => okSink
        (chelis_types.session.DiagnosticSink.with_recursive_caller inst s none b))
      id (fun _ s₁ => s₁) id := by
  intro b s
  simp only [chelis_types.session.DiagnosticSink.with_recursive_caller,
    Aeneas.Std.core.option.Option.is_none, Option.isNone_none, ite_true, id]
  cases inst.call_once b s with
  | ok x => obtain ⟨r, s₁, bk⟩ := x; rfl
  | fail e => rfl
  | div => rfl

/-- **`with_recursive_caller` with a caller**: the body runs with the active group attributed
to `c`, and at exit the active group (if any) is attributed back to the enclosing caller. -/
theorem withRecursiveCaller_some {R T2 : Type} (inst) (c : chelis_types.infer.recursion.Caller) :
    ScopeSpecOn (fun (b : T2) s => okSink (γ := R) (inst.call_once b s))
      (fun b s => okSink
        (chelis_types.session.DiagnosticSink.with_recursive_caller inst s (some c) b))
      (setGroupCaller (some c)) (fun s s₁ => setGroupCaller (groupCaller s) s₁) id := by
  intro b s
  unfold chelis_types.session.DiagnosticSink.with_recursive_caller
  simp only [setGroupCaller, groupCaller]
  cases h : inst.call_once b { s with recursive := { s.recursive with
      group := s.recursive.group.map ({ · with caller := some c }) } } with
  | ok x =>
    obtain ⟨r, s₁, bk⟩ := x
    simp [h, okSink, replaceCaller_eq, Aeneas.Std.core.option.Option.is_none]
  | fail e => simp [h, okSink, replaceCaller_eq, Aeneas.Std.core.option.Option.is_none]
  | div => simp [h, okSink, replaceCaller_eq, Aeneas.Std.core.option.Option.is_none]

/-- The sink's recursive-inference frame. -/
def recursiveLens : Lens Sink chelis_types.infer.recursion.RecursiveState :=
  { get := (·.recursive), set := fun v s => { s with recursive := v } }

theorem recursiveLens_lawful : recursiveLens.Lawful := ⟨fun _ _ => rfl, fun _ => rfl, fun _ _ _ => rfl⟩

/-- The fresh frame: no active group, nothing pinned. -/
def freshFrame : chelis_types.infer.recursion.RecursiveState :=
  { group := none, pinned := ⟨⟨⟨[]⟩⟩⟩ }

/-- The extracted `RecursiveState::default` (derived) returns the fresh frame. -/
theorem recursiveState_default :
    chelis_types.infer.recursion.RecursiveState.Insts.CoreDefaultDefault.default = .ok freshFrame :=
  rfl

/-- **`with_recursive_inference`, every instance**: the body runs in the fresh frame
(`mem::take` of the enclosing one) and the enclosing frame is restored at exit (`mem::replace`,
dropping the pass's own frame); every other effect of the body on the sink, e.g. its
diagnostics, is kept. -/
theorem withRecursiveInference_spec {R T2 : Type} (inst) :
    ScopeSpecOn (fun (b : T2) s => okSink (γ := R) (inst.call_once b s))
      (fun b s => okSink (chelis_types.session.DiagnosticSink.with_recursive_inference inst s b))
      (recursiveLens.set freshFrame) (restoreExit recursiveLens) id := by
  intro b s
  unfold chelis_types.session.DiagnosticSink.with_recursive_inference
  simp only [core.mem.take, recursiveState_default, bind_tc_ok, recursiveLens, restoreExit]
  cases h : inst.call_once b { s with recursive := freshFrame } with
  | ok x => obtain ⟨r, s₁, bk⟩ := x; simp [h, okSink]
  | fail e => simp [h, okSink]
  | div => simp [h, okSink]

/-- The extracted `with_recursive_inference` as a `Scopes.Combinator` (state-transformer
bodies). -/
def recursiveComb (γ : Type) : Combinator Result Sink γ := fun body s =>
  do
    let (r, s', _) ←
      chelis_types.session.DiagnosticSink.with_recursive_inference (T2 := Sink → Result (Sink × γ))
        ⟨fun b s => do let (s', r) ← b s; .ok (r, s', fun _ => s')⟩ s body
    .ok (s', r)

/-- **The literal Section E `InstallRestoreObligation`** (`RecursiveInferenceScope`, fresh
frame) for the extracted `with_recursive_inference`. -/
theorem withRecursiveInference_obligation (γ : Type) :
    InstallRestoreObligation okResult recursiveLens freshFrame (recursiveComb γ) := by
  intro body s
  unfold recursiveComb chelis_types.session.DiagnosticSink.with_recursive_inference
  simp only [core.mem.take, recursiveState_default, bind_tc_ok, recursiveLens, restoreExit]
  cases h : body { s with recursive := freshFrame } with
  | ok x => obtain ⟨s₁, r⟩ := x; simp [h, okResult]
  | fail e => simp [h, okResult]
  | div => simp [h, okResult]

end ChelisBridge.Runtime.ExtractedScopes
