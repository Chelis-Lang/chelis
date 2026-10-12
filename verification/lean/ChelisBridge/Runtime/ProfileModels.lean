import Aeneas

/-!
# Profiling models change only profiling state (O16(a))

The profiling primitives of `chelis_types::native::NativeServices` and
`chelis_ir::lowering_runtime::LoweringRuntime` read clocks and the environment and write stderr.
Their Lean models (`Aeneas/Std/ChelisNative.lean`) touch only the profiling fields:
`profileMark` and `profileTaken` of `NativeServices`, and `profileEnv`, `profileEnvReads` and
`profileMarks` of `LoweringRuntime`. The theorems below show that every successful step keeps
every other field, so a profiling call cannot change cancellation, stack, depth, flags, hook
trace, counters or identity supplies. The two exceptions are explicit: `profile_mark` fails
like `Vec::push` past `usize::MAX` clocks, and `detail_profile_declaration` returns `dagTotal`
or `lastDagSize` to its caller (`*last_dag_size := dag_total` when the mark names a clock).
That a whole extracted entry does not depend on the profiling oracles is O16(b), which needs
the whole-core extraction.
-/

namespace ChelisBridge.Runtime.ProfileModels

open Aeneas.Std

/-- The non-profiling fields of the Lean model of `NativeServices`. -/
def servicesCore (c : chelis_types.native.NativeServices) :=
  (c.cancel, c.cancelTaken, c.stack, c.stackTaken, c.depth)

/-- The non-profiling fields of the Lean model of `LoweringRuntime`. -/
def runtimeCore (s : chelis_ir.lowering_runtime.LoweringRuntime) :=
  (s.u, s.o, s.installed, s.hook, s.builds, s.nextPreflightFrame, s.hostVar, s.hostVarsTaken)

/-- **O16(a).** `profile_start` returns an oracle mark and keeps the non-profiling fields. -/
theorem profile_start_core (c : chelis_types.native.NativeServices) :
    ∃ m c', chelis_types.native.NativeServices.profile_start c = .ok (m, c') ∧
      servicesCore c' = servicesCore c :=
  ⟨_, _, rfl, rfl⟩

/-- **O16(a).** `profile_phase` changes nothing. -/
theorem profile_phase_core (c : chelis_types.native.NativeServices) (mark : Usize)
    (scope label : Str) :
    chelis_types.native.NativeServices.profile_phase c mark scope label = .ok c :=
  rfl

/-- **O16(a).** The private `profile_mark` keeps the non-profiling fields, and it fails only past
`usize::MAX` clocks. -/
theorem profile_mark_core (s : chelis_ir.lowering_runtime.LoweringRuntime) :
    (∀ r, chelis_ir.lowering_runtime.LoweringRuntime.profile_mark s = .ok r →
      runtimeCore r.2 = runtimeCore s) ∧
    (∀ e, chelis_ir.lowering_runtime.LoweringRuntime.profile_mark s = .fail e →
      e = .maximumSizeExceeded ∧ s.profileMarks.val = Usize.max) := by
  unfold chelis_ir.lowering_runtime.LoweringRuntime.profile_mark
  have hle : s.profileMarks.val ≤ Usize.max := by
    have := s.profileMarks.hmax
    rw [Usize.max_def, Usize.numBits_def]
    omega
  by_cases h : s.profileMarks.val < Usize.max
  · refine ⟨fun r hr => ?_, fun e he => ?_⟩
    · simp only [h, if_true, Result.ok.injEq] at hr
      subst hr
      rfl
    · simp [h] at he
  · refine ⟨fun r hr => ?_, fun e he => ?_⟩
    · simp [h] at hr
    · simp only [h, if_false, Result.fail.injEq] at he
      exact ⟨he.symm, by omega⟩

/-- **O16(a).** `detail_profile_start` keeps the non-profiling fields. It fails only when the
environment enables the profile and the thread already holds `usize::MAX` clocks. -/
theorem detail_profile_start_core (s : chelis_ir.lowering_runtime.LoweringRuntime) :
    (∀ r, chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_start s = .ok r →
      runtimeCore r.2 = runtimeCore s) ∧
    (∀ e, chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_start s = .fail e →
      e = .maximumSizeExceeded ∧ s.profileEnv s.profileEnvReads = true ∧
        s.profileMarks.val = Usize.max) := by
  unfold chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_start
  have hm := profile_mark_core { s with profileEnvReads := s.profileEnvReads + 1 }
  cases henv : s.profileEnv s.profileEnvReads with
  | false =>
    refine ⟨fun r hr => ?_, fun e he => ?_⟩
    · simp only [Bool.false_eq_true, if_false, Result.ok.injEq] at hr
      subst hr
      rfl
    · simp at he
  | true =>
    simp only [if_true]
    exact ⟨fun r hr => hm.1 r hr, fun e he => ⟨(hm.2 e he).1, trivial, (hm.2 e he).2⟩⟩

/-- **O16(a).** `detail_profile_phase` changes nothing. -/
theorem detail_profile_phase_core (s : chelis_ir.lowering_runtime.LoweringRuntime)
    (mark : Usize) (label : Str) :
    chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_phase s mark label = .ok s :=
  rfl

/-- **O16(a).** `detail_profile_declaration_start` keeps the non-profiling fields; it fails only
past `usize::MAX` clocks. -/
theorem detail_profile_declaration_start_core (s : chelis_ir.lowering_runtime.LoweringRuntime)
    (mark : Usize) :
    (∀ r, chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_declaration_start s mark =
      .ok r → runtimeCore r.2 = runtimeCore s) ∧
    (∀ e, chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_declaration_start s mark =
      .fail e → e = .maximumSizeExceeded ∧ s.profileMarks.val = Usize.max) := by
  unfold chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_declaration_start
  by_cases h : mark = 0#usize
  · refine ⟨fun r hr => ?_, fun e he => ?_⟩
    · simp only [h, if_true, Result.ok.injEq] at hr
      subst hr
      rfl
    · simp [h] at he
  · simp only [h, if_false]
    exact profile_mark_core s

/-- **O16(a).** `detail_profile_declaration` keeps the whole state. Its value, the caller's new
`last_dag_size`, is `dagTotal` when `start` names a clock of the thread and `lastDagSize`
otherwise. -/
theorem detail_profile_declaration_core (s : chelis_ir.lowering_runtime.LoweringRuntime)
    (start lastDagSize dagTotal : Usize) (label : Str) :
    chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_declaration s start lastDagSize
      dagTotal label =
      .ok (s, if start ≠ 0#usize ∧ start.val ≤ s.profileMarks.val then dagTotal
        else lastDagSize) := by
  unfold chelis_ir.lowering_runtime.LoweringRuntime.detail_profile_declaration
  split <;> rfl

/-- **O16(a).** `report_internal` (one stderr line) changes nothing. -/
theorem report_internal_core (s : chelis_ir.lowering_runtime.LoweringRuntime) (line : Str) :
    chelis_ir.lowering_runtime.LoweringRuntime.report_internal s line = .ok s :=
  rfl

/-!
## Hook installation in native primitives

These laws apply to successful receiver calls, including returned Rust errors.
Lean failure and divergence return no receiver state.
The profile laws retain the non-profile state, which includes `installed`.
Constructors receive no runtime, so these laws do not apply to constructors.
Actual catch bodies need separate proofs with their extracted definitions.
Successful profile calls need no clock bound for installation preservation.
The existing profile contracts retain their failure and clock-overflow conditions.
Primitive preservation does not discharge the actual body premise of O5.
-/

abbrev Runtime := chelis_ir.lowering_runtime.LoweringRuntime
open chelis_ir.lowering_runtime.LoweringRuntime

theorem runtimeCore_installed {before after : Runtime}
    (h : runtimeCore after = runtimeCore before) :
    after.installed = before.installed :=
  congrArg (fun core => core.2.2.1) h

theorem unrepresentable_panic_suppressed_installed (s s' : Runtime) (value : Bool)
    (h : unrepresentable_panic_suppressed s = .ok (value, s')) :
    s'.installed = s.installed := by
  cases h
  rfl

theorem set_unrepresentable_panic_suppressed_installed (s s' : Runtime) (value : Bool)
    (h : set_unrepresentable_panic_suppressed s value = .ok s') :
    s'.installed = s.installed := by
  cases h
  rfl

theorem set_output_suppressed_installed (s s' : Runtime) (value : Bool)
    (h : set_output_suppressed s value = .ok s') :
    s'.installed = s.installed := by
  cases h
  rfl

theorem observe_raise_installed {Payload : Type} (inst : core.any.Any Payload)
    (s s' : Runtime) (payload returned : Payload)

    (h : observe_raise inst s payload = .ok (returned, s')) :
    s'.installed = s.installed := by
  cases h
  rfl

theorem fresh_host_inference_var_installed (s s' : Runtime) (value : U32)
    (h : fresh_host_inference_var s = .ok (value, s')) :
    s'.installed = s.installed := by
  cases h
  rfl

theorem record_summary_probe_build_installed (s s' : Runtime)
    (h : record_summary_probe_build s = .ok s') :
    s'.installed = s.installed := by
  cases h
  rfl

theorem summary_probe_builds_installed (s s' : Runtime) (value : U64)
    (h : summary_probe_builds s = .ok (value, s')) :
    s'.installed = s.installed := by
  cases h
  rfl

theorem reset_summary_probe_builds_installed (s s' : Runtime)
    (h : reset_summary_probe_builds s = .ok s') :
    s'.installed = s.installed := by
  cases h
  rfl

theorem fresh_preflight_frame_installed (s s' : Runtime) (value : U64)
    (h : fresh_preflight_frame s = .ok (value, s')) :
    s'.installed = s.installed := by
  cases h
  rfl

theorem profile_mark_installed (s s' : Runtime) (value : Usize)
    (h : profile_mark s = .ok (value, s')) :
    s'.installed = s.installed :=
  runtimeCore_installed ((profile_mark_core s).1 _ h)

theorem detail_profile_start_installed (s s' : Runtime) (value : Usize)
    (h : detail_profile_start s = .ok (value, s')) :
    s'.installed = s.installed :=
  runtimeCore_installed ((detail_profile_start_core s).1 _ h)

theorem detail_profile_phase_installed (s s' : Runtime) (mark : Usize) (label : Str)

    (h : detail_profile_phase s mark label = .ok s') :
    s'.installed = s.installed := by
  rw [detail_profile_phase_core] at h
  cases h
  rfl

theorem detail_profile_declaration_start_installed (s s' : Runtime) (mark value : Usize)

    (h : detail_profile_declaration_start s mark = .ok (value, s')) :
    s'.installed = s.installed :=
  runtimeCore_installed ((detail_profile_declaration_start_core s mark).1 _ h)

theorem detail_profile_declaration_installed (s s' : Runtime)
    (start lastDagSize dagTotal returned : Usize) (label : Str)

    (h : detail_profile_declaration s start lastDagSize dagTotal label =
      .ok (s', returned)) :
    s'.installed = s.installed := by
  rw [detail_profile_declaration_core] at h
  cases h
  rfl

theorem report_internal_installed (s s' : Runtime) (line : Str)

    (h : report_internal s line = .ok s') :
    s'.installed = s.installed := by
  rw [report_internal_core] at h
  cases h
  rfl

end ChelisBridge.Runtime.ProfileModels
