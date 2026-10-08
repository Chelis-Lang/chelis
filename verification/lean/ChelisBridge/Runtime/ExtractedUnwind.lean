import ChelisBridge.Runtime.ExtractedScopes

/-!
# O7. The explicit-`Result` model `Unwind.new` on the extracted combinators

`Runtime/Unwind.lean` relates the old native semantics `old` to the explicit-`Result` semantics
`new` of a small language (`unwind_equivalence`). This module relates `new` to the code that
Aeneas extracted from `crates/chelis-ir/src/lowering_runtime.rs`, with the native primitives
replaced by their `Aeneas/Std/ChelisNative.lean` models.

`run c p` interprets a program `p` with the extracted functions. A raise calls the extracted
`raise_lowering_diagnostic`, `raise_fatal_lowering_diagnostic` or `raise_unrepresentable`. A
catch calls the extracted `catch_lowering_with` with the work list as its explicit state. A
speculative scope calls the extracted `with_suppress_unrepresentable_panic`. `work` appends to
the work list, and a `noop` scope runs its body. The codec `c` maps the diagnostics of the
language to `LowerDiagnostic` values and back.

`Frag` is the fragment that the extracted code covers. It excludes native panics and aborts, a
finalizer panic, session cells and other guards. For every program in `Frag`,
`run_new` shows that `run c p` returns `.ok`. Its speculation and quiet-output flags, hook
installation, hook entries, work list and result are those of `new`.
The start state of `new` is any exact model lift of the start runtime.
With `unwind_equivalence`, this also relates the old and new language models.
There is one lemma for each constructor (`run_pure` through `run_scope_speculate`).


These are fragment and primitive relation laws, not whole-core caller correspondence.
`Lifts` needs exact equality of the installation flags.
`LowerBoundLifts` retains the process installation uncertainty of the native model.
Native visibility uses `HookObservationRel`, or an exact lift when the lower bound is true.
RA-tls covers thread state only. Whole-core caller proofs still need final extraction.

-/

open Aeneas.Std
open_chelis_model

namespace ChelisBridge.Runtime.ExtractedUnwind

open ChelisBridge.Runtime.Unwind
open ChelisBridge.Runtime.ExtractedScopes

/-- The diagnostic map between the language and the extracted `LowerDiagnostic`. -/
structure Codec where
  enc : Diag → chelis_ir.lower.LowerDiagnostic
  dec : chelis_ir.lower.LowerDiagnostic → Diag
  dec_enc : ∀ d, dec (enc d) = d
  enc_fatal : ∀ d, (enc d).fatal = d.fatal
  enc_makeFatal : ∀ d, enc d.makeFatal = { enc d with fatal := true }
  dec_unrep : ∀ ld, chelis_ir.lowering_runtime.LoweringFailure.into_diagnostic .Unrepresentable =
    .ok ld → dec ld = unrepDiag

/-- The result of a program: `Ok` with its value or `Err` with a lowering failure. -/
abbrev Out := core.result.Result Nat chelis_ir.lowering_runtime.LoweringFailure

/-- The work list. -/
abbrev Works := List Nat

/-- The programs that the extracted lowering combinators cover. -/
def Frag : Prog → Prop
  | .pure _ => True
  | .work _ => True
  | .setCell _ _ => False
  | .bind p k => Frag p ∧ ∀ v, Frag (k v)
  | .scope g _ fp body => (g = .noop ∨ g = .speculate) ∧ fp = none ∧ Frag body
  | .raise _ => True
  | .raiseFatal _ => True
  | .raiseUnrep => True
  | .panic _ => False
  | .abort => False
  | .catchL body onOk onErr => Frag body ∧ (∀ v, Frag (onOk v)) ∧ ∀ d, Frag (onErr d)
  | .entry _ => False

/-- Run a program through the extracted lowering combinators. Programs outside `Frag` fail. -/
def run (c : Codec) : Prog → RT → Works → Result (Out × RT × Works)
  | .pure v, rt, ws => .ok (.Ok v, rt, ws)
  | .work e, rt, ws => .ok (.Ok 0, rt, ws ++ [e])
  | .bind p k, rt, ws => do
      let (r, rt, ws) ← run c p rt ws
      match r with
      | .Ok v => run c (k v) rt ws
      | .Err e => .ok (.Err e, rt, ws)
  | .raise d, rt, ws => do
      let (e, rt) ← chelis_ir.lowering_runtime.LoweringRuntime.raise_lowering_diagnostic rt (c.enc d)
      .ok (.Err e, rt, ws)
  | .raiseFatal d, rt, ws => do
      let (e, rt) ←
        chelis_ir.lowering_runtime.LoweringRuntime.raise_fatal_lowering_diagnostic rt (c.enc d)
      .ok (.Err e, rt, ws)
  | .raiseUnrep, rt, ws => do
      let (e, rt) ← chelis_ir.lowering_runtime.LoweringRuntime.raise_unrepresentable rt
      .ok (.Err e, rt, ws)
  | .catchL body onOk onErr, rt, ws => do
      let (r, rt, ws) ← chelis_ir.lowering_runtime.LoweringRuntime.catch_lowering_with
        (S := Works) (R := Nat) (T2 := Unit) ⟨fun _ x => run c body x.1 x.2⟩ rt ws ()
      match r with
      | .Ok v => run c (onOk v) rt ws
      | .Err ld => run c (onErr (c.dec ld)) rt ws
  | .scope .noop _ none body, rt, ws => run c body rt ws
  | .scope .speculate _ none body, rt, ws => do
      let ((r, ws), rt) ←
        chelis_ir.lowering_runtime.LoweringRuntime.with_suppress_unrepresentable_panic
          (R := Out × Works) (T1 := Unit)
          ⟨fun _ rt => do
            let (r, rt, ws) ← run c body rt ws
            .ok ((r, ws), rt)⟩ rt ()
      .ok (r, rt, ws)
  | _, _, _ => .fail .panic

/-- The hook entries `(u, o, installed)` of a trace. -/
def hookEntries : List Ev → List (Bool × Bool × Bool)
  | [] => []
  | .hook _ fl i _ :: tr => (fl.unrep, fl.out, i) :: hookEntries tr
  | _ :: tr => hookEntries tr

/-- The work events of a trace. -/
def workList : List Ev → Works
  | [] => []
  | .work _ e :: tr => e :: workList tr
  | _ :: tr => workList tr

@[simp] theorem hookEntries_append (a b : List Ev) :
    hookEntries (a ++ b) = hookEntries a ++ hookEntries b := by
  induction a with
  | nil => rfl
  | cons e a ih => cases e <;> simp [hookEntries, ih]

@[simp] theorem workList_append (a b : List Ev) : workList (a ++ b) = workList a ++ workList b := by
  induction a with
  | nil => rfl
  | cons e a ih => cases e <;> simp [workList, ih]

/-- Exact agreement between two model states. RA-tls does not supply the installation equality. -/

abbrev Lifts (t : ThreadId) (rt : RT) (s : State) : Prop :=
  s.flags t = ⟨rt.u, rt.o⟩ ∧ s.installed = rt.installed

/-- Thread flags agree. The runtime field gives only an installation lower bound. -/

abbrev LowerBoundLifts (t : ThreadId) (rt : RT) (s : State) : Prop :=
  s.flags t = ⟨rt.u, rt.o⟩ ∧ InstallationLowerBound rt.installed s.installed

/-- A true bound recovers exact installation without a false-bound absence assumption. -/

theorem lowerBoundLifts_exact_of_installed (t : ThreadId) (rt : RT) (s : State)
    (hs : LowerBoundLifts t rt s) (hi : rt.installed = true) :

    Lifts t rt s :=
  ⟨hs.1, installationLowerBound_exact_of_true rt.installed s.installed hs.2 hi⟩

/-- Hook visibility retains the process states that the installation bound allows. -/

theorem lowerBoundLifts_hook_visibility (t : ThreadId) (rt : RT) (s : State)
    (p : Payload) (hs : LowerBoundLifts t rt s) :

    HookObservationRel rt.installed ⟨rt.u, rt.o⟩ p (observed [hookEv t s p]) := by
  rw [← hs.1]
  exact observed_hook_lower_bound t s p rt.installed hs.2

/-- The result of `run` corresponds to the result of `new`. -/
def Corr (c : Codec) : Out → XRes → Prop
  | .Ok v, .ok w => v = w
  | .Err .Unrepresentable, .err .unrep => True
  | .Err (.Diagnostic ld), .err (.diag d) => ld = c.enc d
  | _, _ => False

theorem corr_ok {c : Codec} {out : Out} {v : Nat} (h : Corr c out (.ok v)) : out = .Ok v := by
  cases out with
  | Ok w =>
    have hw : w = v := h
    rw [hw]
  | Err e => cases e <;> exact h.elim

theorem corr_err {c : Codec} {out : Out} {f : Failure} (h : Corr c out (.err f)) :
    ∃ e, out = .Err e := by
  cases out with
  | Ok w => exact h.elim
  | Err e => exact ⟨e, rfl⟩

theorem corr_unwind {c : Codec} {out : Out} {p : Payload} (h : Corr c out (.unwind p)) : False := by
  cases out with
  | Ok w => exact h
  | Err e => cases e <;> exact h

theorem corr_abort {c : Codec} {out : Out} (h : Corr c out .abort) : False := by
  cases out with
  | Ok w => exact h
  | Err e => cases e <;> exact h

/-- The runtime after a run from `rt` that ends in state `s` with trace `tr`. -/
def after (t : ThreadId) (rt : RT) (s : State) (tr : List Ev) : RT :=
  { rt with
    u := (s.flags t).unrep
    o := (s.flags t).out
    installed := s.installed
    hook := rt.hook ++ hookEntries tr }

/-- `run` agrees with `new`: it returns `.ok`, its runtime is `after`, its work list grows by the
work events, and its result corresponds. -/
abbrev Agrees (c : Codec) (t : ThreadId) (rt : RT) (ws : Works) (x : Result (Out × RT × Works))
    (y : State × List Ev × XRes) : Prop :=
  ∃ out, x = .ok (out, after t rt y.1 y.2.1, ws ++ workList y.2.1) ∧ Corr c out y.2.2

theorem after_lifts (t : ThreadId) (rt : RT) (s : State) (tr : List Ev) :
    Lifts t (after t rt s tr) s := by
  refine ⟨?_, rfl⟩
  simp [after]

theorem after_nil (t : ThreadId) (rt : RT) (s : State) (h : Lifts t rt s) :
    after t rt s [] = rt := by
  obtain ⟨hf, hi⟩ := h
  cases rt
  simp [after, hf, hi, hookEntries]

theorem after_after (t : ThreadId) (rt : RT) (s1 s2 : State) (tr1 tr2 : List Ev) :
    after t (after t rt s1 tr1) s2 tr2 = after t rt s2 (tr1 ++ tr2) := by
  simp [after]

/-! ## One lemma for each constructor -/

theorem run_pure (c : Codec) (t : ThreadId) (v : Nat) (rt : RT) (ws : Works) (s : State)
    (hs : Lifts t rt s) : Agrees c t rt ws (run c (.pure v) rt ws) (new t (.pure v) s) := by
  refine ⟨.Ok v, ?_, ?_⟩
  · simp [run, new, after_nil t rt s hs, workList]
  · show v = v
    rfl

theorem run_work (c : Codec) (t : ThreadId) (e : Nat) (rt : RT) (ws : Works) (s : State)
    (hs : Lifts t rt s) : Agrees c t rt ws (run c (.work e) rt ws) (new t (.work e) s) := by
  refine ⟨.Ok 0, ?_, ?_⟩
  · have h0 : after t rt s [.work t e] = rt := by
      have := after_nil t rt s hs
      simpa [after, hookEntries] using this
    simp [run, new, h0, workList]
  · show 0 = 0
    rfl

theorem run_raise (c : Codec) (t : ThreadId) (d : Diag) (rt : RT) (ws : Works) (s : State)
    (hs : Lifts t rt s) : Agrees c t rt ws (run c (.raise d) rt ws) (new t (.raise d) s) := by
  obtain ⟨hf, hi⟩ := hs
  refine ⟨.Err (if !(c.enc d).fatal && rt.u then .Unrepresentable else .Diagnostic (c.enc d)),
    ?_, ?_⟩
  · simp only [run, raiseLowering_spec, bind_tc_ok, new]
    cases rt
    simp_all [after, hookEntries, hookEv, workList]
  · simp only [new, raiseFailure, c.enc_fatal, hf]
    split <;> simp_all [Corr]

theorem run_raiseFatal (c : Codec) (t : ThreadId) (d : Diag) (rt : RT) (ws : Works) (s : State)
    (hs : Lifts t rt s) :
    Agrees c t rt ws (run c (.raiseFatal d) rt ws) (new t (.raiseFatal d) s) := by
  obtain ⟨hf, hi⟩ := hs
  have hmf : d.makeFatal.fatal = true := rfl
  refine ⟨.Err (.Diagnostic { c.enc d with fatal := true }), ?_, ?_⟩
  · simp only [run, raiseFatal_spec, bind_tc_ok, new]
    cases rt
    simp_all [after, hookEntries, hookEv, workList, State.setOut, State.modFlags]
  · simp [new, raiseFailure, hmf, Corr, c.enc_makeFatal]

theorem run_raiseUnrep (c : Codec) (t : ThreadId) (rt : RT) (ws : Works) (s : State)
    (hs : Lifts t rt s) :
    Agrees c t rt ws (run c .raiseUnrep rt ws) (new t .raiseUnrep s) := by
  obtain ⟨hf, hi⟩ := hs
  refine ⟨.Err .Unrepresentable, ?_, trivial⟩
  have hr := (raiseUnrepresentable_hook_payload rt).1
  simp only [run, hr, bind_tc_ok, new]
  cases rt
  simp_all [after, hookEntries, hookEv, workList]

theorem run_bind (c : Codec) (t : ThreadId) (p : Prog) (k : Nat → Prog) (rt : RT) (ws : Works)
    (s : State) (hs : Lifts t rt s)
    (hp : ∀ rt ws s, Lifts t rt s → Agrees c t rt ws (run c p rt ws) (new t p s))
    (hk : ∀ v rt ws s, Lifts t rt s → Agrees c t rt ws (run c (k v) rt ws) (new t (k v) s)) :
    Agrees c t rt ws (run c (.bind p k) rt ws) (new t (.bind p k) s) := by
  obtain ⟨out, hrun, hcorr⟩ := hp rt ws s hs
  rcases hN : new t p s with ⟨s1, tr1, r⟩
  rw [hN] at hrun hcorr
  dsimp only at hrun hcorr
  cases r with
  | ok v =>
    have ho := corr_ok hcorr
    subst ho
    obtain ⟨out2, hrun2, hcorr2⟩ := hk v _ (ws ++ workList tr1) s1 (after_lifts t rt s1 tr1)
    rcases hN2 : new t (k v) s1 with ⟨s2, tr2, r2⟩
    rw [hN2] at hrun2 hcorr2
    dsimp only at hrun2 hcorr2
    refine ⟨out2, ?_, ?_⟩
    · simp only [run, hrun, bind_tc_ok]
      simp [hrun2, new, hN, hN2, after_after, List.append_assoc, workList_append]
    · simpa only [new, hN, hN2] using hcorr2
  | err f =>
    obtain ⟨e, rfl⟩ := corr_err hcorr
    refine ⟨.Err e, ?_, ?_⟩
    · simp only [run, hrun, bind_tc_ok]
      simp [new, hN]
    · simpa only [new, hN] using hcorr
  | unwind q => exact (corr_unwind hcorr).elim
  | abort => exact (corr_abort hcorr).elim

/-- `catch_lowering_with`, unfolded to the call of its body. -/
theorem catchLoweringWith_unfold {S R T2 : Type}
    (inst : core.ops.function.FnOnceMutArgMutArg T2 RT S
      (core.result.Result R chelis_ir.lowering_runtime.LoweringFailure))
    (rt : RT) (st : S) (f : T2) :
    chelis_ir.lowering_runtime.LoweringRuntime.catch_lowering_with inst rt st f =
      ((do
          let (r, a1, a2) ← inst.call_once f ({ rt with o := true, installed := true }, st)
          let r1 ← chelis_ir.core.result.Result.map_err
            (BuiltinFnOnce chelis_ir.lowering_runtime.LoweringFailure chelis_ir.lower.LowerDiagnostic)
            r chelis_ir.lowering_runtime.LoweringFailure.into_diagnostic
          .ok (r1, a1, a2)) >>= fun (result, s, state) => .ok (result, { s with o := false }, state)) :=
  rfl

/-- `catch_lowering_with` once its body returned. -/
theorem catchLoweringWith_eq {S R T2 : Type}
    (inst : core.ops.function.FnOnceMutArgMutArg T2 RT S
      (core.result.Result R chelis_ir.lowering_runtime.LoweringFailure))
    (rt rt₁ : RT) (st st₁ : S) (f : T2)
    (r : core.result.Result R chelis_ir.lowering_runtime.LoweringFailure)
    (h : inst.call_once f ({ rt with o := true, installed := true }, st) = .ok (r, rt₁, st₁)) :
    chelis_ir.lowering_runtime.LoweringRuntime.catch_lowering_with inst rt st f =
      (match r with
        | .Ok x => .ok (.Ok x, { rt₁ with o := false }, st₁)
        | .Err e => do
          let d ← chelis_ir.lowering_runtime.LoweringFailure.into_diagnostic e
          .ok (.Err d, { rt₁ with o := false }, st₁)) := by
  rw [catchLoweringWith_unfold]
  cases r with
  | Ok x => simp [h, chelis_ir.core.result.Result.map_err]
  | Err e =>
    obtain ⟨d, hd⟩ := intoDiagnostic_ok e
    simp [h, chelis_ir.core.result.Result.map_err, BuiltinFnOnce, hd]

theorem run_catchL (c : Codec) (t : ThreadId) (body : Prog) (onOk : Nat → Prog)
    (onErr : Diag → Prog) (rt : RT) (ws : Works) (s : State) (hs : Lifts t rt s)
    (hb : ∀ rt ws s, Lifts t rt s → Agrees c t rt ws (run c body rt ws) (new t body s))
    (hok : ∀ v rt ws s, Lifts t rt s → Agrees c t rt ws (run c (onOk v) rt ws) (new t (onOk v) s))
    (herr : ∀ d rt ws s, Lifts t rt s →
      Agrees c t rt ws (run c (onErr d) rt ws) (new t (onErr d) s)) :
    Agrees c t rt ws (run c (.catchL body onOk onErr) rt ws) (new t (.catchL body onOk onErr) s) := by
  have hin : Lifts t { rt with o := true, installed := true }
      ({ s with installed := true }.setOut t true) := by
    obtain ⟨hf, hi⟩ := hs
    exact ⟨by simp [State.setOut, State.modFlags, hf], rfl⟩
  obtain ⟨out, hrun, hcorr⟩ := hb _ ws _ hin
  rcases hN : new t body ({ s with installed := true }.setOut t true) with ⟨s1, tr1, r⟩
  rw [hN] at hrun hcorr
  dsimp only at hrun hcorr
  have hcatch := catchLoweringWith_eq (S := Works) (R := Nat) (T2 := Unit)
    ⟨fun _ x => run c body x.1 x.2⟩ rt _ ws _ () out hrun
  have hl : Lifts t { after t { rt with o := true, installed := true } s1 tr1 with o := false }
      (s1.setOut t false) :=
    ⟨by simp [after, State.setOut, State.modFlags], by simp [after, State.setOut, State.modFlags]⟩
  have hcat : ∀ (s2 : State) (tr2 : List Ev),
      after t { after t { rt with o := true, installed := true } s1 tr1 with o := false } s2 tr2 =
        after t rt s2 (tr1 ++ tr2) := by
    intro s2 tr2
    simp [after]
  cases r with
  | ok v =>
    have ho := corr_ok hcorr
    subst ho
    obtain ⟨out2, hrun2, hcorr2⟩ := hok v _ (ws ++ workList tr1) _ hl
    rcases hN2 : new t (onOk v) (s1.setOut t false) with ⟨s2, tr2, r2⟩
    rw [hN2] at hrun2 hcorr2
    dsimp only at hrun2 hcorr2
    refine ⟨out2, ?_, ?_⟩
    · simp only [run, hcatch, bind_tc_ok]
      simp [hrun2, new, hN, hN2, hcat, List.append_assoc, workList_append]
    · simpa only [new, hN, hN2] using hcorr2
  | err f =>
    obtain ⟨e, rfl⟩ := corr_err hcorr
    obtain ⟨ld, hld⟩ := intoDiagnostic_ok e
    have hdec : c.dec ld = f.intoDiag := by
      cases f with
      | unrep =>
        cases e with
        | Unrepresentable => exact c.dec_unrep ld hld
        | Diagnostic _ => exact hcorr.elim
      | diag d =>
        cases e with
        | Unrepresentable => exact hcorr.elim
        | Diagnostic ld' =>
          have hl' : ld' = c.enc d := hcorr
          have hld' : ld' = ld := by
            simpa [chelis_ir.lowering_runtime.LoweringFailure.into_diagnostic] using hld
          rw [← hld', hl']
          exact c.dec_enc d
    obtain ⟨out2, hrun2, hcorr2⟩ := herr f.intoDiag _ (ws ++ workList tr1) _ hl
    rcases hN2 : new t (onErr f.intoDiag) (s1.setOut t false) with ⟨s2, tr2, r2⟩
    rw [hN2] at hrun2 hcorr2
    dsimp only at hrun2 hcorr2
    refine ⟨out2, ?_, ?_⟩
    · simp only [run, hcatch, hld, bind_tc_ok]
      simp [hdec, hrun2, new, hN, hN2, hcat, List.append_assoc, workList_append]
    · simpa only [new, hN, hN2] using hcorr2
  | unwind q => exact (corr_unwind hcorr).elim
  | abort => exact (corr_abort hcorr).elim

theorem run_scope_noop (c : Codec) (t : ThreadId) (fid : Nat) (body : Prog) (rt : RT)
    (ws : Works) (s : State) (hs : Lifts t rt s)
    (hb : ∀ rt ws s, Lifts t rt s → Agrees c t rt ws (run c body rt ws) (new t body s)) :
    Agrees c t rt ws (run c (.scope .noop fid none body) rt ws)
      (new t (.scope .noop fid none body) s) := by
  obtain ⟨out, hrun, hcorr⟩ := hb rt ws s hs
  rcases hN : new t body s with ⟨s1, tr1, r⟩
  rw [hN] at hrun hcorr
  dsimp only at hrun hcorr
  have hnew : new t (.scope .noop fid none body) s = (s1, tr1 ++ [.fin t fid], r) := by
    have he : Guard.enter t .noop s = s := rfl
    simp only [new, he, hN]
    cases r with
    | ok v => rfl
    | err f => rfl
    | unwind q => exact (corr_unwind hcorr).elim
    | abort => exact (corr_abort hcorr).elim
  refine ⟨out, ?_, ?_⟩
  · rw [hnew]
    simp only [run, hrun]
    simp [after, hookEntries, workList]
  · rw [hnew]
    exact hcorr

theorem run_scope_speculate (c : Codec) (t : ThreadId) (fid : Nat) (body : Prog) (rt : RT)
    (ws : Works) (s : State) (hs : Lifts t rt s)
    (hb : ∀ rt ws s, Lifts t rt s → Agrees c t rt ws (run c body rt ws) (new t body s)) :
    Agrees c t rt ws (run c (.scope .speculate fid none body) rt ws)
      (new t (.scope .speculate fid none body) s) := by
  have hin : Lifts t { rt with u := true } (Guard.enter t .speculate s) := by
    obtain ⟨hf, hi⟩ := hs
    exact ⟨by simp [Guard.enter, State.setUnrep, State.modFlags, hf], hi⟩
  obtain ⟨out, hrun, hcorr⟩ := hb _ ws _ hin
  rcases hN : new t body (Guard.enter t .speculate s) with ⟨s1, tr1, r⟩
  rw [hN] at hrun hcorr
  dsimp only at hrun hcorr
  have hnew : new t (.scope .speculate fid none body) s =
      (Guard.exit t s.cells .speculate s1, tr1 ++ [.fin t fid], r) := by
    simp only [new, hN]
    cases r with
    | ok v => rfl
    | err f => rfl
    | unwind q => exact (corr_unwind hcorr).elim
    | abort => exact (corr_abort hcorr).elim
  refine ⟨out, ?_, ?_⟩
  · rw [hnew]
    simp only [run, withSuppress_eq, hrun, bind_tc_ok]
    simp [after, Guard.exit, State.setUnrep, State.modFlags, hookEntries, workList]
  · rw [hnew]
    exact hcorr

/-- **O7. `new` on the extracted combinators.** For every program of `Frag`, every codec and
every start runtime `rt` that a state `s` lifts on thread `t`, `run c p rt ws` returns `.ok`. Its
runtime has the flags, the hook installation and the hook entries of `new t p s`, all other
fields of `rt` stay. Its work list grows by the work events of `new`, and its result corresponds
to the result of `new`. -/
theorem run_new (c : Codec) (t : ThreadId) (p : Prog) (hp : Frag p) :
    ∀ rt ws s, Lifts t rt s → Agrees c t rt ws (run c p rt ws) (new t p s) := by
  induction p with
  | pure v => exact fun rt ws s hs => run_pure c t v rt ws s hs
  | work e => exact fun rt ws s hs => run_work c t e rt ws s hs
  | setCell k v => exact False.elim hp
  | bind p k ihp ihk =>
    obtain ⟨hp1, hp2⟩ := hp
    exact fun rt ws s hs => run_bind c t p k rt ws s hs (ihp hp1) (fun v => ihk v (hp2 v))
  | scope g fid fp body ih =>
    obtain ⟨hg, rfl, hb⟩ := hp
    rcases hg with rfl | rfl
    · exact fun rt ws s hs => run_scope_noop c t fid body rt ws s hs (ih hb)
    · exact fun rt ws s hs => run_scope_speculate c t fid body rt ws s hs (ih hb)
  | raise d => exact fun rt ws s hs => run_raise c t d rt ws s hs
  | raiseFatal d => exact fun rt ws s hs => run_raiseFatal c t d rt ws s hs
  | raiseUnrep => exact fun rt ws s hs => run_raiseUnrep c t rt ws s hs
  | panic q => exact False.elim hp
  | abort => exact False.elim hp
  | catchL body onOk onErr ih ihOk ihErr =>
    obtain ⟨hb, hok, herr⟩ := hp
    exact fun rt ws s hs => run_catchL c t body onOk onErr rt ws s hs (ih hb)
      (fun v => ihOk v (hok v)) (fun d => ihErr d (herr d))
  | entry body ih => exact False.elim hp

/-- **O7 with the hook installed.** The lift of the O7 statement in NATIVE_SHELL_SCOPE.md: a
runtime whose flags are those of `s` on thread `t` and whose hook is installed, with
`s.installed = true`. -/
theorem run_new_installed (c : Codec) (t : ThreadId) (p : Prog) (hp : Frag p) (rt : RT)
    (ws : Works) (s : State) (hf : s.flags t = ⟨rt.u, rt.o⟩) (hi : s.installed = true)
    (hrt : rt.installed = true) :
    ∃ out, run c p rt ws = .ok (out, after t rt (new t p s).1 (new t p s).2.1,
        ws ++ workList (new t p s).2.1) ∧ Corr c out (new t p s).2.2 :=
  run_new c t p hp rt ws s ⟨hf, hi.trans hrt.symm⟩

/-! ## A codec -/

/-- The span of a diagnostic of the language, kept in `span_id` as a string of that length. -/
def spanText (n : Nat) : String := String.ofList (List.replicate n ' ')

/-- The message of the fixed unrepresentable diagnostic, through the `Str` model. -/
theorem unrepresentable_message :
    Str.toString (toStr
      "program uses a form that is not supported by IR evaluation yet; use `chelis build --target c` instead") =
      .ok "program uses a form that is not supported by IR evaluation yet; use `chelis build --target c` instead" := by
  have h : (match Str.toString (toStr
      "program uses a form that is not supported by IR evaluation yet; use `chelis build --target c` instead") with
      | .ok s => decide (s =
        "program uses a form that is not supported by IR evaluation yet; use `chelis build --target c` instead")
      | _ => false) = true := by
    decide +kernel
  revert h
  cases Str.toString (toStr
      "program uses a form that is not supported by IR evaluation yet; use `chelis build --target c` instead") <;>
    simp

/-- The canonical codec: the message and the `fatal` flag directly, the span through
`span_id`. -/
def canonical : Codec where
  enc d :=
    { message := d.msg
      span := none
      span_id := d.span.map spanText
      fatal := d.fatal
      unsupported := none }
  dec ld := ⟨ld.message, ld.span_id.map String.length, ld.fatal⟩
  dec_enc d := by
    cases d with
    | mk msg span fatal =>
      cases span <;> simp [spanText, String.length_ofList]
  enc_fatal _ := rfl
  enc_makeFatal _ := rfl
  dec_unrep ld h := by
    simp only [chelis_ir.lowering_runtime.LoweringFailure.into_diagnostic,
      chelis_ir.lowering_runtime.unrepresentable_diagnostic, chelis_ir.lower.LowerDiagnostic.new,
      core.convert.IntoFrom.into, alloc.string.FromStrString.from_, unrepresentable_message, bind_tc_ok,
      Result.ok.injEq] at h
    subst h
    rfl

/-- `Frag` and the codec laws can hold together: `run_new` is not vacuous. -/
theorem run_new_canonical :
    Frag (.catchL (.scope .speculate 0 none (.raise ⟨"r", some 2, false⟩)) .pure
      (fun d => .raiseFatal d)) ∧
    ∃ out rt', run canonical (.catchL (.scope .speculate 0 none (.raise ⟨"r", some 2, false⟩))
        .pure (fun d => .raiseFatal d))
        { u := false, o := false, installed := false, hook := [], builds := 0#u64,
          nextPreflightFrame := 0#u64, hostVar := fun _ => 0#u32, hostVarsTaken := 0,
          profileEnv := fun _ => false, profileEnvReads := 0, profileMarks := 0#usize } [] =
      .ok (out, rt', []) ∧ rt'.hook = [(true, true, true), (false, true, true)] := by
  have hf : Frag (.catchL (.scope .speculate 0 none (.raise ⟨"r", some 2, false⟩)) .pure
      (fun d => .raiseFatal d)) :=
    ⟨⟨Or.inr rfl, rfl, trivial⟩, fun _ => trivial, fun _ => trivial⟩
  refine ⟨hf, ?_⟩
  obtain ⟨out, hrun, -⟩ := run_new canonical 0 _ hf
    { u := false, o := false, installed := false, hook := [], builds := 0#u64,
      nextPreflightFrame := 0#u64, hostVar := fun _ => 0#u32, hostVarsTaken := 0,
      profileEnv := fun _ => false, profileEnvReads := 0, profileMarks := 0#usize } [] s0
    ⟨rfl, rfl⟩
  exact ⟨out, _, hrun, rfl⟩

end ChelisBridge.Runtime.ExtractedUnwind
