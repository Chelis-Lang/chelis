import ChelisBridge.Protocol.Dropout

/-!
# Accepted runs of the extracted execution frame simulate the reference Random protocol

The single-step relations of `ChelisBridge.Protocol.Random` assume the facts that the extracted
`ExecutionFrame::{control, enter}` check at run time. The inversion lemmas below derive those facts
from an accepted (`Ok`) step, so a whole accepted run (source controls, forward random sites and
replays in any nesting) is related to `runProtocol` by induction. The explicit domain
restrictions are the ones shown necessary by counterexamples: an entered scope is fresh
(`reentry_counterexample`) and carries a handler seed, and no forward counter is at `u64::MAX`
(`counter_wrap_counterexample`).
-/

open Aeneas.Std LaCaDiLECore ChelisBridge ChelisBridge.Protocol ChelisBridge.Random
open_chelis_model

namespace ChelisBridge.Random

/-- An `Err` built from a computed message is never an `Ok` result. -/
theorem err_bind_ne_ok {α ε σ : Type} (x : Result ε) (g : ε → σ) (k : α) (t : σ) :
    (x >>= fun s => Result.ok ((core.result.Result.Err s : core.result.Result α ε), g s)) ≠
      Result.ok (core.result.Result.Ok k, t) := by
  cases x <;> simp

theorem enter_forward_inv (F : Frame) (node : chelis_ir.dag.NodeId) (seed : U64)
    (draw : chelis_ir.evaluation.DrawId) (s : chelis_ir.evaluation.ScopeId)
    (k : chelis_ir.evaluation.DrawKey) (F' : Frame) (back : Frame → Frame)
    (hsite : siteOf F node = .ok (some (.Forward draw s)))
    (h : chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k, F', back)) :
    ∃ c, F.scopes.val.getLast? = some s ∧ F.metadata.scopes.val[s.val]? = some ⟨some seed⟩ ∧
      counterAt F s = some c ∧ F.keys.val[draw.val]? = some none ∧ k = ⟨seed, c⟩ ∧
      F' = drawnFrame F draw s seed c := by
  unfold chelis_ir.evaluation.ExecutionFrame.enter at h
  simp only [siteOf] at hsite
  simp only [hsite, bind_tc_ok, core.option.Option.ok_or, core.result.TryResult.branch,
    core.slice.Slice.last, alloc.vec.Vec.deref,
    core.option.Option.Insts.CoreCmpPartialEqOption.eq,
    core.cmp.impls.PartialEqShared.eq, chelis_ir.evaluation.ScopeId.Insts.CoreCmpPartialEqScopeId.eq,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
    alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize, alloc.vec.Vec.getElem?_Nat_eq,
    core.option.Option.replace, core.option.Option.is_some, lift] at h
  rcases hl : F.scopes.val.getLast? with _ | t
  · simp only [hl, bind_tc_ok, Bool.not_false, if_true] at h
    exact absurd h (err_bind_ne_ok _ _ _ _)
  by_cases hts : t = s
  swap
  · simp only [hl, hts, decide_false, bind_tc_ok, Bool.not_false, if_true] at h
    exact absurd h (err_bind_ne_ok _ _ _ _)
  subst hts
  simp only [hl, decide_true, bind_tc_ok, Bool.not_true, Bool.false_eq_true, if_false] at h
  rcases hm : F.metadata.scopes.val[t.val]? with _ | ⟨_ | sd⟩
  · simp [hm] at h
  · simp only [hm, bind_tc_ok, core.result.FromResidualResult.from_residual, bind_assoc_eq] at h
    exact absurd h (err_bind_ne_ok _ _ _ _)
  by_cases hsd : sd = seed
  swap
  · simp only [hm, bind_tc_ok, bne_iff_ne, ne_eq, hsd, not_false_eq_true, if_true] at h
    exact absurd h (err_bind_ne_ok _ _ _ _)
  subst hsd
  simp only [hm, bind_tc_ok, bne_self_eq_false, Bool.false_eq_true, if_false] at h
  by_cases ht0 : t = 0#usize
  · subst ht0
    simp only [if_true, bind_tc_ok] at h
    rcases hkd : F.keys.val[draw.val]? with _ | (_ | k0)
    · simp [hkd] at h
    · simp only [hkd, bind_tc_ok] at h
      obtain ⟨rfl, rfl, -⟩ := h
      refine ⟨F.context.state.counter, rfl, rfl, by simp [counterAt], rfl, rfl, ?_⟩
      simp [drawnFrame]
    · simp [hkd] at h
      exact absurd h (err_bind_ne_ok _ _ _ _)
  · rcases hc : F.counters.val[t.val]? with _ | c
    · simp [ht0, hc] at h
    · simp only [ht0, if_false, hc, bind_tc_ok] at h
      rcases hkd : F.keys.val[draw.val]? with _ | (_ | k0)
      · simp [hkd] at h
      · simp only [hkd, bind_tc_ok] at h
        obtain ⟨rfl, rfl, -⟩ := h
        refine ⟨c, rfl, rfl, by simp [counterAt, ht0, hc], rfl, rfl, ?_⟩
        simp [drawnFrame, ht0]
      · simp [hkd] at h
        exact absurd h (err_bind_ne_ok _ _ _ _)

theorem enter_replay_inv (F : Frame) (node : chelis_ir.dag.NodeId) (seed : U64)
    (draw : chelis_ir.evaluation.DrawId) (k : chelis_ir.evaluation.DrawKey) (F' : Frame)
    (back : Frame → Frame)
    (hsite : siteOf F node = .ok (some (.Replay draw)))
    (h : chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k, F', back)) :
    F' = F ∧ F.keys.val[draw.val]? = some (some k) := by
  unfold chelis_ir.evaluation.ExecutionFrame.enter at h
  simp only [siteOf] at hsite
  simp only [hsite, bind_tc_ok, core.option.Option.ok_or, core.result.TryResult.branch,
    alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq,
    chelis_ir.core.option.Option.ok_or_else] at h
  rcases hkd : F.keys.val[draw.val]? with _ | (_ | k0)
  · simp [hkd] at h
  · simp [hkd,
      chelis_ir.evaluation.ExecutionFrame.enter.closure.Insts.CoreOpsFunctionFnOnceTupleString.call_once] at h
    exact absurd h (err_bind_ne_ok _ _ _ _)
  · simp [hkd] at h
    obtain ⟨rfl, rfl, -⟩ := h
    exact ⟨rfl, rfl⟩

theorem control_enter_inv (F : Frame) (scope : chelis_ir.evaluation.ScopeId) (payload : U64)
    (F' : Frame) (back : Frame → Frame)
    (h : chelis_ir.evaluation.ExecutionFrame.control F (.Enter scope payload) =
      .ok (.Ok (), F', back)) :
    scope.val < F.counters.val.length ∧ F.scopes.val.length < Usize.max := by
  unfold chelis_ir.evaluation.ExecutionFrame.control at h
  simp only [alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize,
    alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq] at h
  rcases hc : F.counters.val[scope.val]? with _ | c
  · simp [hc] at h
  · refine ⟨(List.getElem?_eq_some_iff.mp hc).1, ?_⟩
    by_contra hlen
    have hU : U32.max ≤ Usize.max := by
      rcases Usize.bounds_eq with e | e <;> simp [e, U32.max_eq, U64.max_eq]
    simp only [hc, bind_tc_ok, alloc.vec.Vec.push] at h
    rw [dif_neg (by simp; omega)] at h
    simp at h

theorem control_leave_inv (F : Frame) (scope : chelis_ir.evaluation.ScopeId)
    (F' : Frame) (back : Frame → Frame)
    (h : chelis_ir.evaluation.ExecutionFrame.control F (.Leave scope) = .ok (.Ok (), F', back)) :
    ∃ (pre : List chelis_ir.evaluation.ScopeId) (v : alloc.vec.Vec chelis_ir.evaluation.ScopeId),
      F.scopes.val = pre ++ [scope] ∧ pre ≠ [] ∧ v.val = pre ∧ F' = { F with scopes := v } := by
  unfold chelis_ir.evaluation.ExecutionFrame.control at h
  by_cases hle : alloc.vec.Vec.len F.scopes ≤ 1#usize
  · simp only [hle, if_true] at h
    exact absurd h (err_bind_ne_ok _ _ _ _)
  simp only [hle, if_false, alloc.vec.Vec.pop,
    core.option.Option.Insts.CoreCmpPartialEqOption.eq,
    chelis_ir.evaluation.ScopeId.Insts.CoreCmpPartialEqScopeId.eq] at h
  have h2 : 2 ≤ F.scopes.val.length := by
    have h' : ¬ (F.scopes.val.length ≤ 1) := by
      intro e; apply hle
      apply (UScalar.le_equiv _ _).mpr
      simpa only [alloc.vec.Vec.len_val, alloc.vec.Vec.length] using e
    omega
  rcases hl : F.scopes.val.getLast? with _ | t
  · simp [hl] at h
    exact absurd h (err_bind_ne_ok _ _ _ _)
  by_cases hts : t = scope
  swap
  · simp [hl, hts] at h
    exact absurd h (err_bind_ne_ok _ _ _ _)
  subst hts
  simp [hl] at h
  obtain ⟨rfl, -⟩ := h
  refine ⟨F.scopes.val.dropLast, _, (List.dropLast_append_getLast? t hl).symm, ?_, rfl, rfl⟩
  intro e
  have := congrArg List.length e
  simp only [List.length_dropLast, List.length_nil] at this
  omega

/-- One operation of an execution frame: a source control or a random site (dropout or uniform
key) at `node` with the plan's raw seed. -/
inductive FrameOp where
  | control (c : chelis_ir.execution_spine.Control)
  | site (node : chelis_ir.dag.NodeId) (seed : U64)

/-- `AcceptedRun F ops evs F'`: the extracted `control`/`enter` accept every operation of `ops` in
order (each returns `Ok`) and lead from `F` to `F'`; `evs` are the reference protocol events of
the run: `enter` of the handler seed for `Control::Enter`, `leave` for `Control::Leave`, `draw` of
the returned key for a forward site, and nothing for a replay. Premises beyond acceptance are the
explicit domain restrictions: an entered scope has a handler seed and is not on the stack, and a
forward ordinal is below `u64::MAX`. -/
inductive AcceptedRun : Frame → List FrameOp → List ProtocolEvent → Frame → Prop
  | nil (F : Frame) : AcceptedRun F [] [] F
  | enter (F F' F'' : Frame) (scope : chelis_ir.evaluation.ScopeId) (payload sd : U64)
      (back : Frame → Frame) (ops : List FrameOp) (evs : List ProtocolEvent)
      (h : chelis_ir.evaluation.ExecutionFrame.control F (.Enter scope payload) =
        .ok (.Ok (), F', back))
      (hseed : F.metadata.scopes.val[scope.val]? = some ⟨some sd⟩)
      (hfresh : scope ∉ F.scopes.val)
      (rest : AcceptedRun F' ops evs F'') :
      AcceptedRun F (.control (.Enter scope payload) :: ops)
        (.enter (UInt64.ofBitVec sd.bv) :: evs) F''
  | leave (F F' F'' : Frame) (scope : chelis_ir.evaluation.ScopeId) (back : Frame → Frame)
      (ops : List FrameOp) (evs : List ProtocolEvent)
      (h : chelis_ir.evaluation.ExecutionFrame.control F (.Leave scope) = .ok (.Ok (), F', back))
      (rest : AcceptedRun F' ops evs F'') :
      AcceptedRun F (.control (.Leave scope) :: ops) (.leave :: evs) F''
  | forward (F F' F'' : Frame) (node : chelis_ir.dag.NodeId) (seed : U64)
      (draw : chelis_ir.evaluation.DrawId) (s : chelis_ir.evaluation.ScopeId)
      (k : chelis_ir.evaluation.DrawKey) (back : Frame → Frame) (ops : List FrameOp)
      (evs : List ProtocolEvent)
      (hsite : siteOf F node = .ok (some (.Forward draw s)))
      (h : chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k, F', back))
      (hno : k.ordinal.val + 1 < 2 ^ 64)
      (rest : AcceptedRun F' ops evs F'') :
      AcceptedRun F (.site node seed :: ops) (.draw (keyOfDraw k.seed k.ordinal) :: evs) F''
  | replay (F F' F'' : Frame) (node : chelis_ir.dag.NodeId) (seed : U64)
      (draw : chelis_ir.evaluation.DrawId) (k : chelis_ir.evaluation.DrawKey)
      (back : Frame → Frame) (ops : List FrameOp) (evs : List ProtocolEvent)
      (hsite : siteOf F node = .ok (some (.Replay draw)))
      (h : chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k, F', back))
      (rest : AcceptedRun F' ops evs F'') :
      AcceptedRun F (.site node seed :: ops) evs F''

/-- Scope-stack well-formedness: the root scope `0` is at the bottom (where
`EvaluationPlan::frame` puts it and where `Leave` can never pop it) and no scope occurs twice. -/
def StackWF (F : Frame) : Prop := F.scopes.val.Nodup ∧ F.scopes.val.head? = some 0#usize

theorem stackWF_root (F : Frame) (h : F.scopes.val = [0#usize]) : StackWF F := by
  unfold StackWF; rw [h]; simp

/-- **Accepted runs simulate the reference protocol.** From a well-formed stack, every accepted
run of the extracted frame, with fresh seeded scopes and no forward counter overflow, is a run
of the reference `runProtocol` (for every target: the events contain no `require`) between the
abstracted states, and keeps the stack well formed. Forward sites draw exactly the reference key,
nested `Enter`/`Leave` push and restore the saved keys, the root scope draws from the inherited
context stream, and replays emit no event and leave the state unchanged. -/
theorem accepted_run_simulates (target : Target) {F : Frame} {ops : List FrameOp}
    {evs : List ProtocolEvent} {F' : Frame} (h : AcceptedRun F ops evs F') (hwf : StackWF F) :
    runProtocol target evs (absState F) = some (absState F') ∧ StackWF F' := by
  induction h with
  | nil F => exact ⟨rfl, hwf⟩
  | enter F F' F'' scope payload sd back ops evs h hseed hfresh rest ih =>
    obtain ⟨hlt, hlen⟩ := control_enter_inv F scope payload F' back h
    have hne : F.scopes.val ≠ [] := by
      intro e; have := hwf.2; rw [e] at this; simp at this
    have hroot : scope ≠ 0#usize := by
      intro e; subst e; exact hfresh (List.mem_of_mem_head? hwf.2)
    obtain ⟨F₁, back₁, he₁, hrel⟩ := scope_relation F scope payload sd hlt hlen hroot hfresh hne hseed
    obtain ⟨F₂, back₂, he₂, -, -, -, hsc, -⟩ := control_enter F scope payload hlt hlen
    rw [h] at he₁ he₂
    simp only [Result.ok.injEq, Prod.mk.injEq] at he₁ he₂
    obtain ⟨-, rfl, -⟩ := he₁
    obtain ⟨-, rfl, -⟩ := he₂
    have hwf' : StackWF F' := by
      refine ⟨?_, ?_⟩
      · rw [hsc, List.nodup_append]
        exact ⟨hwf.1, List.nodup_singleton _, fun a ha b hb e => by
          simp only [List.mem_singleton] at hb; subst hb; subst e; exact hfresh ha⟩
      · rw [hsc, List.head?_append_of_ne_nil _ hne]; exact hwf.2
    obtain ⟨ih1, ih2⟩ := ih hwf'
    refine ⟨?_, ih2⟩
    show runProtocol target evs ((absState F).enter (UInt64.ofBitVec sd.bv)) = _
    rw [← hrel]; exact ih1
  | leave F F' F'' scope back ops evs h rest ih =>
    obtain ⟨pre, v, hsc, hpre, hv, hF⟩ := control_leave_inv F scope F' back h
    obtain ⟨F₁, back₁, he₁, hsc₁, hrel⟩ := leave_relation F scope pre hsc hpre
    rw [h] at he₁
    simp only [Result.ok.injEq, Prod.mk.injEq] at he₁
    obtain ⟨-, rfl, -⟩ := he₁
    have hwf' : StackWF F' := by
      refine ⟨?_, ?_⟩
      · have hnd := hwf.1
        rw [hsc] at hnd
        rw [hsc₁]
        exact hnd.sublist (List.sublist_append_left _ _)
      · have hh := hwf.2
        rw [hsc, List.head?_append_of_ne_nil _ hpre] at hh
        rw [hsc₁]; exact hh
    obtain ⟨ih1, ih2⟩ := ih hwf'
    refine ⟨?_, ih2⟩
    show ((absState F).leave).bind (runProtocol target evs) = _
    rw [hrel]; exact ih1
  | forward F F' F'' node seed draw s k back ops evs hsite h hno rest ih =>
    obtain ⟨c, hlast, hseedm, hc, hkey, rfl, hF⟩ :=
      enter_forward_inv F node seed draw s k F' back hsite h
    obtain ⟨rest', hrev⟩ : ∃ rest', F.scopes.val.reverse = s :: rest' := by
      rw [← List.head?_reverse] at hlast
      cases hr : F.scopes.val.reverse with
      | nil => simp [hr] at hlast
      | cons a r =>
        simp only [hr, List.head?_cons, Option.some.injEq] at hlast
        exact ⟨r, by rw [hlast]⟩
    have hfresh : s ∉ rest' := by
      have hnd := hwf.1
      rw [← List.nodup_reverse, hrev] at hnd
      exact (List.nodup_cons.mp hnd).1
    obtain ⟨F₁, back₁, he₁, hrel⟩ :=
      counter_relation_no_overflow F node seed draw s c rest' hsite hrev hfresh hseedm hc hkey hno
    rw [h] at he₁
    simp only [Result.ok.injEq, Prod.mk.injEq] at he₁
    obtain ⟨-, rfl, -⟩ := he₁
    have hwf' : StackWF F' := by rw [hF]; exact hwf
    obtain ⟨ih1, ih2⟩ := ih hwf'
    refine ⟨?_, ih2⟩
    simp [runProtocol, ProtocolEvent.step, hrel, keyOfDraw, ih1]
  | replay F F' F'' node seed draw k back ops evs hsite h rest ih =>
    obtain ⟨rfl, -⟩ := enter_replay_inv F node seed draw k F' back hsite h
    exact ih hwf

/-- **Protocol acceptance.** An accepted run from a root frame (`scopes = [0]`, the frame
`EvaluationPlan::frame` builds) back to a root frame passes the reference `checkProtocol`. -/
theorem accepted_run_checks (target : Target) {F : Frame} {ops : List FrameOp}
    {evs : List ProtocolEvent} {F' : Frame} (h : AcceptedRun F ops evs F')
    (h0 : F.scopes.val = [0#usize]) (h1 : F'.scopes.val = [0#usize]) :
    checkProtocol target evs (absState F) = true := by
  obtain ⟨hrun, -⟩ := accepted_run_simulates target h (stackWF_root F h0)
  rw [checkProtocol_of_run target evs _ _ hrun]
  simp [absState, h0, h1]

/-- An accepted forward site records its returned key in the frame. -/
theorem forward_records_key (F : Frame) (node : chelis_ir.dag.NodeId) (seed : U64)
    (draw : chelis_ir.evaluation.DrawId) (s : chelis_ir.evaluation.ScopeId)
    (k : chelis_ir.evaluation.DrawKey) (F' : Frame) (back : Frame → Frame)
    (hsite : siteOf F node = .ok (some (.Forward draw s)))
    (h : chelis_ir.evaluation.ExecutionFrame.enter F node seed = .ok (.Ok k, F', back)) :
    F'.keys.val[draw.val]? = some (some k) := by
  obtain ⟨c, -, -, -, hkey, rfl, rfl⟩ := enter_forward_inv F node seed draw s k F' back hsite h
  have hlt : draw.val < F.keys.val.length := (List.getElem?_eq_some_iff.mp hkey).1
  simp [drawnFrame, alloc.vec.Vec.set_val_eq, hlt]

/-- Recorded keys persist through every accepted operation (a second forward draw of the same
id is rejected, controls and replays do not touch keys). -/
theorem accepted_run_keys_persist {F : Frame} {ops : List FrameOp} {evs : List ProtocolEvent}
    {F' : Frame} (h : AcceptedRun F ops evs F') (d : chelis_ir.evaluation.DrawId)
    (k : chelis_ir.evaluation.DrawKey) (hk : F.keys.val[d.val]? = some (some k)) :
    F'.keys.val[d.val]? = some (some k) := by
  induction h with
  | nil F => exact hk
  | enter F F' F'' scope payload sd back ops evs h hseed hfresh rest ih =>
    obtain ⟨hlt, hlen⟩ := control_enter_inv F scope payload F' back h
    obtain ⟨F₂, back₂, he₂, -, hkeys, -, -, -⟩ := control_enter F scope payload hlt hlen
    rw [h] at he₂
    simp only [Result.ok.injEq, Prod.mk.injEq] at he₂
    obtain ⟨-, rfl, -⟩ := he₂
    exact ih (by rw [hkeys]; exact hk)
  | leave F F' F'' scope back ops evs h rest ih =>
    obtain ⟨pre, v, -, -, -, rfl⟩ := control_leave_inv F scope F' back h
    exact ih hk
  | forward F F' F'' node seed draw s k' back ops evs hsite h hno rest ih =>
    obtain ⟨c, -, -, -, hkey, -, rfl⟩ := enter_forward_inv F node seed draw s k' F' back hsite h
    apply ih
    have hne : draw.val ≠ d.val := by
      intro e; rw [e, hk] at hkey; simp at hkey
    simp only [drawnFrame, alloc.vec.Vec.set_val_eq]
    rw [List.getElem?_set_ne hne]; exact hk
  | replay F F' F'' node seed draw k' back ops evs hsite h rest ih =>
    obtain ⟨rfl, -⟩ := enter_replay_inv F node seed draw k' F' back hsite h
    exact ih hk

/-- **Replay returns the forward key.** After an accepted forward site of draw `d` returned `k`,
any accepted run followed by a replay site of `d` succeeds with exactly `k` and leaves the frame
unchanged: replay is not another draw. -/
theorem replay_after_forward (F₀ F₁ F₂ : Frame) (n₁ n₂ : chelis_ir.dag.NodeId) (seed₁ seed₂ : U64)
    (d : chelis_ir.evaluation.DrawId) (s : chelis_ir.evaluation.ScopeId)
    (k : chelis_ir.evaluation.DrawKey) (b₁ : Frame → Frame) (ops : List FrameOp)
    (evs : List ProtocolEvent)
    (hsite₁ : siteOf F₀ n₁ = .ok (some (.Forward d s)))
    (h₁ : chelis_ir.evaluation.ExecutionFrame.enter F₀ n₁ seed₁ = .ok (.Ok k, F₁, b₁))
    (hrun : AcceptedRun F₁ ops evs F₂)
    (hsite₂ : siteOf F₂ n₂ = .ok (some (.Replay d))) :
    ∃ b₂, chelis_ir.evaluation.ExecutionFrame.enter F₂ n₂ seed₂ = .ok (.Ok k, F₂, b₂) :=
  enter_replay F₂ n₂ seed₂ d k hsite₂
    (accepted_run_keys_persist hrun d k (forward_records_key F₀ n₁ seed₁ d s k F₁ b₁ hsite₁ h₁))

end ChelisBridge.Random
