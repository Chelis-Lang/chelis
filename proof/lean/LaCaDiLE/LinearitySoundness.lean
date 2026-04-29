-- LaCaDiLE/LinearitySoundness.lean — linearity soundness theorem (Phase 2 proof).
--
-- WS2.8 target: no well-typed program accesses a deallocated location.
-- Structural form: `Step` preserves `StoreWf` and the invariant that the
-- live locations in the store correspond to linear bindings in Γ.
-- Phase 1 skeleton: stub.

import LaCaDiLE.Syntax
import LaCaDiLE.Store
import LaCaDiLE.Typing
import LaCaDiLE.Operational

namespace LaCaDiLE

/-- Generalized Linearity soundness over arbitrary configurations,
    suitable for structural induction on Step. The concrete
    `linearity_soundness` below is a trivial specialization. -/
private theorem linearity_soundness_aux
    (Sigma : StoreTyp)
    (c1 c2 : Config)
    (h_wf : StoreWf c1.store Sigma)
    (h_step : Step c1 c2) :
    ∃ Sigma', StoreWf c2.store Sigma' := by
  induction h_step with
  | beta => exact ⟨Sigma, h_wf⟩
  | letBind => exact ⟨Sigma, h_wf⟩
  | letpair => exact ⟨Sigma, h_wf⟩
  | fst => exact ⟨Sigma, h_wf⟩
  | snd => exact ⟨Sigma, h_wf⟩
  | handleRet => exact ⟨Sigma, h_wf⟩
  | handleOpDirect => exact ⟨Sigma, h_wf⟩
  | handleOpCtx => exact ⟨Sigma, h_wf⟩
  | handleOpCtxs => exact ⟨Sigma, h_wf⟩
  | tgrad => exact ⟨Sigma, h_wf⟩
  | tvmap => exact ⟨Sigma, h_wf⟩
  | tconst s v ds ell hell =>
      refine ⟨storeTypExtend Sigma ell (Typ.tensor ds), ?_⟩
      exact StoreWf.extend_fresh ell ⟨ds, v⟩ (Typ.tensor ds) h_wf
  | copy s ell ellNew w _hlook hfresh =>
      refine ⟨storeTypExtend Sigma ellNew (Typ.tensor w.shape), ?_⟩
      exact StoreWf.extend_fresh ellNew w (Typ.tensor w.shape) h_wf
  | tadd s ell1 ell2 ellOut w1 w2 _h1 _h2 _hfresh =>
      refine ⟨storeTypExtend
                (storeTypRemove (storeTypRemove Sigma ell1) ell2)
                ellOut (Typ.tensor w1.shape), ?_⟩
      have h_wf1 := StoreWf.remove ell1 h_wf
      have h_wf2 := StoreWf.remove ell2 h_wf1
      exact StoreWf.extend_fresh ellOut
        (tensorOpPlaceholder w1 w2) (Typ.tensor w1.shape) h_wf2
  | tmul s ell1 ell2 ellOut w1 w2 _h1 _h2 _hfresh =>
      refine ⟨storeTypExtend
                (storeTypRemove (storeTypRemove Sigma ell1) ell2)
                ellOut (Typ.tensor w1.shape), ?_⟩
      have h_wf1 := StoreWf.remove ell1 h_wf
      have h_wf2 := StoreWf.remove ell2 h_wf1
      exact StoreWf.extend_fresh ellOut
        (tensorOpPlaceholder w1 w2) (Typ.tensor w1.shape) h_wf2
  | tsum s ell ellOut w d hlook hfresh =>
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor (rem w.shape d)), ?_⟩
      have h_isSome : (storeLookup s ell).isSome := by
        rw [hlook]; rfl
      have hne : ell ≠ ellOut := by
        rw [hfresh]; exact storeFreshLoc_ne s ell h_isSome
      exact StoreWf.remove_extend ell ellOut
        { shape := rem w.shape d, data := w.data }
        (Typ.tensor (rem w.shape d)) h_wf hfresh hne
  | texpand s ell ellOut w d hlook hfresh =>
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor (ins w.shape d)), ?_⟩
      have h_isSome : (storeLookup s ell).isSome := by
        rw [hlook]; rfl
      have hne : ell ≠ ellOut := by
        rw [hfresh]; exact storeFreshLoc_ne s ell h_isSome
      exact StoreWf.remove_extend ell ellOut
        { shape := ins w.shape d, data := w.data }
        (Typ.tensor (ins w.shape d)) h_wf hfresh hne
  | tuniformLike s ell ellOut w lo hi hlook hfresh =>
      refine ⟨storeTypExtend (storeTypRemove Sigma ell) ellOut
                (Typ.tensor w.shape), ?_⟩
      have h_isSome : (storeLookup s ell).isSome := by
        rw [hlook]; rfl
      have hne : ell ≠ ellOut := by
        rw [hfresh]; exact storeFreshLoc_ne s ell h_isSome
      exact StoreWf.remove_extend ell ellOut
        { shape := w.shape, data := lo }
        (Typ.tensor w.shape) h_wf hfresh hne
  | ctx _sig _sig' _E _e _e' _h_inner ih =>
      -- E-Ctx: ih is the inner step's witness, which uses the same
      -- store-side states since plugging an evaluation context doesn't
      -- touch the store further.
      exact ih h_wf

/-- Runtime-safety cases that remain open after the store-aware
    handler/substitution redesign. These are exactly the steps that
    still require a dedicated substitution/context proof or depend on
    the unresolved AD surface. This is the honest public residual
    surface for the runtime track. -/
inductive RuntimeSafeDebt : Config → Config → Prop
  | beta
      (sigma : Store) (x : String) (t : Typ) (e v : Term)
      (hv : IsValue v) :
      RuntimeSafeDebt
        ⟨sigma, Term.app (Term.abs x t e) v⟩
        ⟨sigma, subst e v x⟩
  | letBind
      (sigma : Store) (x : String) (v e : Term)
      (hv : IsValue v) :
      RuntimeSafeDebt
        ⟨sigma, Term.letBind x v e⟩
        ⟨sigma, subst e v x⟩
  | letpair
      (sigma : Store) (x y : String) (v1 v2 e : Term)
      (hv1 : IsValue v1) (hv2 : IsValue v2) :
      RuntimeSafeDebt
        ⟨sigma, Term.letpair x y (Term.pair v1 v2) e⟩
        ⟨sigma, subst (subst e v1 x) v2 y⟩
  | handleOpDirect
      (sigma : Store) (op : EffectLabel) (v : Term)
      (epsH : EffectRow)
      (clauses : List (EffectLabel × String × String × Term))
      (x k : String) (handlerBody : Term) (tRet : Typ)
      (hv : IsValue v)
      (hsig : ∃ tArg, OpSigMatch op tArg tRet)
      (hmem : (op, x, k, handlerBody) ∈ clauses) :
      RuntimeSafeDebt
        ⟨sigma, Term.handle epsH (Term.perform op v) clauses⟩
        ⟨sigma,
          subst (subst handlerBody v x)
            (directIdCont epsH op v clauses tRet) k⟩
  | handleOpCtx
      (sigma : Store) (op : EffectLabel) (v : Term)
      (epsH : EffectRow) (E : EvalCtx)
      (clauses : List (EffectLabel × String × String × Term))
      (xVar kVar : String) (hb : Term) (tRet : Typ)
      (hv : IsValue v)
      (hsig : ∃ tArg, OpSigMatch op tArg tRet)
      (hmem : (op, xVar, kVar, hb) ∈ clauses)
      (hop : op ∈ epsH)
      (hE : EvalCtx.noHandleFor op E) :
      RuntimeSafeDebt
        ⟨sigma, Term.handle epsH (plug E (Term.perform op v)) clauses⟩
        ⟨sigma,
          subst (subst hb v xVar)
            (Term.abs (capturedContName
              (Term.handle epsH (plug E (Term.perform op v)) clauses)) tRet
              (Term.handle epsH
                (plug E (Term.var (capturedContName
                  (Term.handle epsH (plug E (Term.perform op v)) clauses))))
                clauses)) kVar⟩
  | handleOpCtxs
      (sigma : Store) (op : EffectLabel) (v : Term)
      (epsH : EffectRow) (Es : EvalCtxChain)
      (clauses : List (EffectLabel × String × String × Term))
      (xVar kVar : String) (hb : Term) (tRet : Typ)
      (hv : IsValue v)
      (hsig : ∃ tArg, OpSigMatch op tArg tRet)
      (hmem : (op, xVar, kVar, hb) ∈ clauses)
      (hop : op ∈ epsH)
      (hEs : EvalCtxChain.noHandleFor op Es) :
      RuntimeSafeDebt
        ⟨sigma, Term.handle epsH (multiPlug Es (Term.perform op v)) clauses⟩
        ⟨sigma,
          subst (subst hb v xVar)
            (Term.abs (capturedContName
              (Term.handle epsH (multiPlug Es (Term.perform op v)) clauses)) tRet
              (Term.handle epsH
                (multiPlug Es (Term.var (capturedContName
                  (Term.handle epsH (multiPlug Es (Term.perform op v)) clauses))))
                clauses)) kVar⟩
  | tgrad
      (sigma : Store) (x : String) (tv tOut : Typ) (body : Term)
      (hSupp : AdjointSupported body) :
      RuntimeSafeDebt
        ⟨sigma, Term.grad x tv tOut body⟩
        ⟨sigma,
          Term.abs x tv
            (Term.letpair (gradPrimalName x body) x
              (Term.copy (Term.var x))
              (Term.abs (gradSeedName x body) tOut
                (Term.letBind (gradResultName x body)
                  (Term.handle [EffectLabel.accum]
                    (adjointTypedFrom body tOut x (Term.var (gradSeedName x body))
                      (gradAdjointCounter x (gradSeedName x body) body))
                    [(EffectLabel.accum, "p", "k", Term.app (Term.var "k") (Term.var "p"))])
                  (Term.var (gradPrimalName x body)))))⟩
  | ctx
      (sigma sigma' : Store) (E : EvalCtx) (e e' : Term)
      (h_inner : Step ⟨sigma, e⟩ ⟨sigma', e'⟩) :
      RuntimeSafeDebt
        ⟨sigma, plug E e⟩
        ⟨sigma', plug E e'⟩

/-- Every handler-aware term still has a duplicate-free active
    footprint at its root. -/
private theorem handlerAwareRuntimeLinear_active
    {e : Term}
    (h : HandlerAwareRuntimeLinear e) :
    ActiveRuntimeLinear e := by
  cases e <;>
    simp [HandlerAwareRuntimeLinear, ActiveRuntimeLinear,
      activeLocRefs, activeLocRefsClauses] at h ⊢
  all_goals
    first
    | exact h.1
    | exact h
/-- One-step preservation boundary for the stronger handler-aware
    runtime invariant. The purely runtime/store-structural head rules
    are closed here; the remaining cases are reported explicitly as
    `RuntimeSafeDebt` rather than being silently folded into the
    theorem statement. -/
private theorem handlerAwareRuntimeLinear_step_or_debt
    (c1 c2 : Config)
    (h_step : Step c1 c2) :
    HandlerAwareRuntimeLinear c1.term →
      HandlerAwareRuntimeLinear c2.term ∨ RuntimeSafeDebt c1 c2 := by
  induction h_step with
  | beta sigma x t e v hv =>
      intro _h
      exact Or.inr (RuntimeSafeDebt.beta sigma x t e v hv)
  | letBind sigma x v e hv =>
      intro _h
      exact Or.inr (RuntimeSafeDebt.letBind sigma x v e hv)
  | letpair sigma x y v1 v2 e hv1 hv2 =>
      intro _h
      exact Or.inr (RuntimeSafeDebt.letpair sigma x y v1 v2 e hv1 hv2)
  | fst sigma tRight v1 v2 hv1 hv2 =>
      intro h
      rcases (by simpa [HandlerAwareRuntimeLinear] using h) with
        ⟨_hAct, hPair⟩
      rcases (by simpa [HandlerAwareRuntimeLinear] using hPair) with
        ⟨_hPairAct, hv1Aware, _hv2Aware, _hsep12, _hsep21⟩
      exact Or.inl hv1Aware
  | snd sigma tLeft v1 v2 hv1 hv2 =>
      intro h
      rcases (by simpa [HandlerAwareRuntimeLinear] using h) with
        ⟨_hAct, hPair⟩
      rcases (by simpa [HandlerAwareRuntimeLinear] using hPair) with
        ⟨_hPairAct, _hv1Aware, hv2Aware, _hsep12, _hsep21⟩
      exact Or.inl hv2Aware
  | tconst sigma v ds ell hell =>
      intro _h
      subst ell
      simp [HandlerAwareRuntimeLinear]
  | copy sigma ell ellNew w hlook hfresh =>
      intro _h
      have hsome : (storeLookup sigma ell).isSome := by
        rw [hlook]
        rfl
      have hne : ell ≠ ellNew := by
        rw [hfresh]
        exact storeFreshLoc_ne sigma ell hsome
      have hne' : ellNew ≠ ell := by
        intro hEq
        exact hne hEq.symm
      exact Or.inl <| by
        refine ⟨?_, by simp [HandlerAwareRuntimeLinear],
          by simp [HandlerAwareRuntimeLinear], ?_, ?_⟩
        · simp [ActiveRuntimeLinear, activeLocRefs, hne, hne']
        · simp [LocRefsDisjoint, LocRefsSeparated, StepLocRefs, activeLocRefs, hne, hne']
        · simp [LocRefsDisjoint, LocRefsSeparated, StepLocRefs, activeLocRefs, hne, hne']
  | tadd sigma ell1 ell2 ellOut w1 w2 h1 h2 hfresh =>
      intro _h
      subst ellOut
      simp [HandlerAwareRuntimeLinear]
  | tmul sigma ell1 ell2 ellOut w1 w2 h1 h2 hfresh =>
      intro _h
      subst ellOut
      simp [HandlerAwareRuntimeLinear]
  | tsum sigma ell ellOut w d hlook hfresh =>
      intro _h
      subst ellOut
      simp [HandlerAwareRuntimeLinear]
  | texpand sigma ell ellOut w d hlook hfresh =>
      intro _h
      subst ellOut
      simp [HandlerAwareRuntimeLinear]
  | tuniformLike sigma ell ellOut w lo hi hlook hfresh =>
      intro _h
      subst ellOut
      simp [HandlerAwareRuntimeLinear]
  | handleRet sigma epsH v clauses hv =>
      intro h
      rcases (by simpa [HandlerAwareRuntimeLinear] using h) with
        ⟨_hAct, hvAware, _hClauses, _hSep⟩
      exact Or.inl hvAware
  | handleOpDirect sigma op v epsH clauses x k handlerBody tRet hv hsig hmem =>
      intro _h
      exact Or.inr
        (RuntimeSafeDebt.handleOpDirect sigma op v epsH clauses x k handlerBody tRet
          hv hsig hmem)
  | handleOpCtx sigma op v epsH E clauses xVar kVar hb tRet hv hsig hmem hop hE =>
      intro _h
      exact Or.inr
        (RuntimeSafeDebt.handleOpCtx sigma op v epsH E clauses xVar kVar hb tRet
          hv hsig hmem hop hE)
  | handleOpCtxs sigma op v epsH Es clauses xVar kVar hb tRet hv hsig hmem hop hEs =>
      intro _h
      exact Or.inr
        (RuntimeSafeDebt.handleOpCtxs sigma op v epsH Es clauses xVar kVar hb tRet
          hv hsig hmem hop hEs)
  | tgrad sigma x tv tOut body hSupp =>
      intro _h
      exact Or.inr (RuntimeSafeDebt.tgrad sigma x tv tOut body hSupp)
  | tvmap sigma x tv d body =>
      intro h
      rcases (by simpa [HandlerAwareRuntimeLinear, ActiveRuntimeLinear, activeLocRefs] using h) with
        ⟨hActBody, hBody⟩
      exact Or.inl <| by
        refine ⟨?_, handlerAwareRuntimeLinear_addDimTerm d hBody⟩
        simpa [ActiveRuntimeLinear, activeLocRefs, activeLocRefs_addDimTerm (d := d) (e := body)]
          using hActBody
  | ctx sigma sigma' E e e' h_inner ih =>
      intro h
      cases E with
      | hole =>
          have hInner : HandlerAwareRuntimeLinear e := by
            simpa [plug] using h
          rcases ih hInner with hInner' | hDebt
          · exact Or.inl (by simpa [plug] using hInner')
          · exact Or.inr hDebt
      | appL e2 =>
          exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.appL e2) e e' h_inner)
      | appR v1 =>
          exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.appR v1) e e' h_inner)
      | letBind x e2 =>
          exact Or.inr
            (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.letBind x e2) e e' h_inner)
      | copy =>
          rcases (by simpa [HandlerAwareRuntimeLinear, plug] using h) with
            ⟨_hAct, hInner⟩
          rcases ih hInner with hInner' | _hDebt
          · exact Or.inl <| by
              simpa [HandlerAwareRuntimeLinear, plug, ActiveRuntimeLinear, activeLocRefs] using
                (show ActiveRuntimeLinear e' ∧ HandlerAwareRuntimeLinear e' from
                  ⟨handlerAwareRuntimeLinear_active hInner', hInner'⟩)
          · exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' EvalCtx.copy e e' h_inner)
      | letpair x y e2 =>
          exact Or.inr
            (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.letpair x y e2) e e' h_inner)
      | pairL e2 =>
          exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.pairL e2) e e' h_inner)
      | pairR v1 =>
          exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.pairR v1) e e' h_inner)
      | fst tRight =>
          rcases (by simpa [HandlerAwareRuntimeLinear, plug] using h) with
            ⟨_hAct, hInner⟩
          rcases ih hInner with hInner' | _hDebt
          · exact Or.inl <| by
              simpa [HandlerAwareRuntimeLinear, plug, ActiveRuntimeLinear, activeLocRefs] using
                (show ActiveRuntimeLinear e' ∧ HandlerAwareRuntimeLinear e' from
                  ⟨handlerAwareRuntimeLinear_active hInner', hInner'⟩)
          · exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.fst tRight) e e' h_inner)
      | snd tLeft =>
          rcases (by simpa [HandlerAwareRuntimeLinear, plug] using h) with
            ⟨_hAct, hInner⟩
          rcases ih hInner with hInner' | _hDebt
          · exact Or.inl <| by
              simpa [HandlerAwareRuntimeLinear, plug, ActiveRuntimeLinear, activeLocRefs] using
                (show ActiveRuntimeLinear e' ∧ HandlerAwareRuntimeLinear e' from
                  ⟨handlerAwareRuntimeLinear_active hInner', hInner'⟩)
          · exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.snd tLeft) e e' h_inner)
      | addL e2 =>
          exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.addL e2) e e' h_inner)
      | addR v1 =>
          exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.addR v1) e e' h_inner)
      | mulL e2 =>
          exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.mulL e2) e e' h_inner)
      | mulR v1 =>
          exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.mulR v1) e e' h_inner)
      | sum d =>
          rcases (by simpa [HandlerAwareRuntimeLinear, plug] using h) with
            ⟨_hAct, hInner⟩
          rcases ih hInner with hInner' | _hDebt
          · exact Or.inl <| by
              simpa [HandlerAwareRuntimeLinear, plug, ActiveRuntimeLinear, activeLocRefs] using
                (show ActiveRuntimeLinear e' ∧ HandlerAwareRuntimeLinear e' from
                  ⟨handlerAwareRuntimeLinear_active hInner', hInner'⟩)
          · exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.sum d) e e' h_inner)
      | expand d =>
          rcases (by simpa [HandlerAwareRuntimeLinear, plug] using h) with
            ⟨_hAct, hInner⟩
          rcases ih hInner with hInner' | _hDebt
          · exact Or.inl <| by
              simpa [HandlerAwareRuntimeLinear, plug, ActiveRuntimeLinear, activeLocRefs] using
                (show ActiveRuntimeLinear e' ∧ HandlerAwareRuntimeLinear e' from
                  ⟨handlerAwareRuntimeLinear_active hInner', hInner'⟩)
          · exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.expand d) e e' h_inner)
      | uniformLike lo hi =>
          rcases (by simpa [HandlerAwareRuntimeLinear, plug] using h) with
            ⟨_hAct, hInner⟩
          rcases ih hInner with hInner' | _hDebt
          · exact Or.inl <| by
              simpa [HandlerAwareRuntimeLinear, plug, ActiveRuntimeLinear, activeLocRefs] using
                (show ActiveRuntimeLinear e' ∧ HandlerAwareRuntimeLinear e' from
                  ⟨handlerAwareRuntimeLinear_active hInner', hInner'⟩)
          · exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma'
                (EvalCtx.uniformLike lo hi) e e' h_inner)
      | handle epsH clauses =>
          exact Or.inr
            (RuntimeSafeDebt.ctx sigma sigma'
              (EvalCtx.handle epsH clauses) e e' h_inner)
      | perform op =>
          rcases (by simpa [HandlerAwareRuntimeLinear, plug] using h) with
            ⟨_hAct, hInner⟩
          rcases ih hInner with hInner' | _hDebt
          · exact Or.inl <| by
              simpa [HandlerAwareRuntimeLinear, plug, ActiveRuntimeLinear, activeLocRefs] using
                (show ActiveRuntimeLinear e' ∧ HandlerAwareRuntimeLinear e' from
                  ⟨handlerAwareRuntimeLinear_active hInner', hInner'⟩)
          · exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.perform op) e e' h_inner)

/-- Full-step entry point for the current runtime-invariant track.
    A globally runtime-linear source term is strong enough to enter the
    newer handler-aware surface in one step; the only remaining gaps are
    the explicitly classified debt constructors. -/
private theorem runtimeLinear_step_or_runtimeSafeDebt
    (c1 c2 : Config)
    (h_step : Step c1 c2) :
    RuntimeLinear c1.term →
      HandlerAwareRuntimeLinear c2.term ∨ RuntimeSafeDebt c1 c2 := by
  intro hRuntime
  exact handlerAwareRuntimeLinear_step_or_debt c1 c2 h_step
    (handlerAwareRuntimeLinear_of_runtimeLinear hRuntime)

private theorem runtimeSafeConfig_loc
    {sigma : Store} {ell : Loc} {w : TensorVal}
    (hlook : storeLookup sigma ell = some w) :
    RuntimeSafeConfig ⟨sigma, Term.loc ell⟩ := by
  refine ⟨?_, ?_⟩
  · simp [SubstAwareHandlerRuntimeLinear, HandlerAwareRuntimeLinear,
      SubstAwareRuntimeLinear]
  · intro ell' hmem
    simp [locRefs] at hmem
    subst ell'
    rw [hlook]
    rfl

private theorem runtimeSafeConfig_pair_left
    {sigma : Store} {e1 e2 : Term}
    (h : RuntimeSafeConfig ⟨sigma, Term.pair e1 e2⟩) :
    RuntimeSafeConfig ⟨sigma, e1⟩ := by
  simpa [plug] using
    (runtimeSafeConfig_plug (sigma := sigma) (E := EvalCtx.pairL e2) (e := e1)
      (by simpa [plug] using h)).1

private theorem runtimeSafeConfig_pair_right
    {sigma : Store} {e1 e2 : Term}
    (h : RuntimeSafeConfig ⟨sigma, Term.pair e1 e2⟩) :
    RuntimeSafeConfig ⟨sigma, e2⟩ := by
  simpa [plug] using
    (runtimeSafeConfig_plug (sigma := sigma) (E := EvalCtx.pairR e1) (e := e2)
      (by simpa [plug] using h)).1

/-- First honest config-level one-step theorem for the runtime-safety
    sidecar. This closes the store-safe head rules and the unary
    context frames, while leaving the remaining substitution, handler,
    AD, and sibling-interaction cases as explicit `RuntimeSafeDebt`.
    The unresolved `ctx` surface is now known to include dormant frame
    locations that are invisible to the active/step footprints but
    still required by `StoreLiveCtxLocRefs`, so those cases cannot be
    folded back into the theorem without a stronger frame premise. -/
theorem runtimeSafeConfig_step_or_debt
    (c1 c2 : Config)
    (h_step : Step c1 c2) :
    RuntimeSafeConfig c1 →
      RuntimeSafeConfig c2 ∨ RuntimeSafeDebt c1 c2 := by
  induction h_step with
  | beta sigma x t e v hv =>
      intro _h
      exact Or.inr (RuntimeSafeDebt.beta sigma x t e v hv)
  | letBind sigma x v e hv =>
      intro _h
      exact Or.inr (RuntimeSafeDebt.letBind sigma x v e hv)
  | letpair sigma x y v1 v2 e hv1 hv2 =>
      intro _h
      exact Or.inr (RuntimeSafeDebt.letpair sigma x y v1 v2 e hv1 hv2)
  | fst sigma tRight v1 v2 hv1 hv2 =>
      intro h
      exact Or.inl <| runtimeSafeConfig_pair_left <|
        (runtimeSafeConfig_plug (sigma := sigma) (E := EvalCtx.fst tRight)
          (e := Term.pair v1 v2) (by simpa [plug] using h)).1
  | snd sigma tLeft v1 v2 hv1 hv2 =>
      intro h
      exact Or.inl <| runtimeSafeConfig_pair_right <|
        (runtimeSafeConfig_plug (sigma := sigma) (E := EvalCtx.snd tLeft)
          (e := Term.pair v1 v2) (by simpa [plug] using h)).1
  | tconst sigma v ds ell hell =>
      intro _h
      subst ell
      have hlook :
          storeLookup (storeExtend sigma (storeFreshLoc sigma) ⟨ds, v⟩) (storeFreshLoc sigma) =
            some ⟨ds, v⟩ := by
        simp [storeLookup, storeExtend]
      exact Or.inl <| runtimeSafeConfig_loc hlook
  | copy sigma ell ellNew w hlook hfresh =>
      intro _h
      have hne : ell ≠ ellNew := by
        rw [hfresh]
        exact storeFreshLoc_ne sigma ell (by rw [hlook]; rfl)
      have hne' : ellNew ≠ ell := by
        intro hEq
        exact hne hEq.symm
      have hInner : RuntimeSafeConfig ⟨storeExtend sigma ellNew w, Term.loc ell⟩ := by
        have hkeep : storeLookup (storeExtend sigma ellNew w) ell = some w := by
          unfold storeLookup storeExtend
          simp only [List.find?, hne', decide_false, Bool.false_eq_true, ite_false,
            Option.map]
          exact hlook
        exact runtimeSafeConfig_loc hkeep
      have hCtx : RuntimeSafeCtx (storeExtend sigma ellNew w) (EvalCtx.pairL (Term.loc ellNew)) := by
        have hnew : storeLookup (storeExtend sigma ellNew w) ellNew = some w := by
          simp [storeLookup, storeExtend]
        constructor
        · simpa [DeepActiveCtx, DeepActiveRuntimeLinear, ActiveRuntimeLinear,
            activeLocRefs]
        · intro ell' hmem
          simp [ctxLocRefs, locRefs] at hmem
          subst ell'
          rw [hnew]
          rfl
      have hPlug :
          SubstAwareHandlerRuntimeLinear
            (Term.pair (Term.loc ell) (Term.loc ellNew)) := by
        simp [SubstAwareHandlerRuntimeLinear, HandlerAwareRuntimeLinear,
          HandlerAwareRuntimeLinearClauses, SubstAwareRuntimeLinear,
          SubstAwareRuntimeLinearClauses, ActiveRuntimeLinear,
          activeLocRefs, activeLocRefsClauses,
          StepLocRefs, StepLocRefsClauses, AppFunLocRefs,
          activeVarRefs, StepVarRefs, StepVarRefsClauses, AppFunVarRefs,
          LocRefsDisjoint, LocRefsSeparated, VarRefsDisjoint, VarRefsSeparated,
          hne, hne']
      exact Or.inl <| runtimeSafeConfig_ctx hCtx hInner hPlug
  | tadd sigma ell1 ell2 ellOut w1 w2 h1 h2 hfresh =>
      intro _h
      have hlook : storeLookup
          (storeExtend (storeRemove (storeRemove sigma ell1) ell2) ellOut
            (tensorOpPlaceholder w1 w2)) ellOut =
          some (tensorOpPlaceholder w1 w2) := by
        simp [storeLookup, storeExtend]
      exact Or.inl (runtimeSafeConfig_loc hlook)
  | tmul sigma ell1 ell2 ellOut w1 w2 h1 h2 hfresh =>
      intro _h
      have hlook : storeLookup
          (storeExtend (storeRemove (storeRemove sigma ell1) ell2) ellOut
            (tensorOpPlaceholder w1 w2)) ellOut =
          some (tensorOpPlaceholder w1 w2) := by
        simp [storeLookup, storeExtend]
      exact Or.inl (runtimeSafeConfig_loc hlook)
  | tsum sigma ell ellOut w d hlook hfresh =>
      intro _h
      have hlook' : storeLookup
          (storeExtend (storeRemove sigma ell) ellOut { shape := rem w.shape d, data := w.data })
          ellOut =
          some { shape := rem w.shape d, data := w.data } := by
        simp [storeLookup, storeExtend]
      exact Or.inl (runtimeSafeConfig_loc hlook')
  | texpand sigma ell ellOut w d hlook hfresh =>
      intro _h
      have hlook' : storeLookup
          (storeExtend (storeRemove sigma ell) ellOut { shape := ins w.shape d, data := w.data })
          ellOut =
          some { shape := ins w.shape d, data := w.data } := by
        simp [storeLookup, storeExtend]
      exact Or.inl (runtimeSafeConfig_loc hlook')
  | tuniformLike sigma ell ellOut w lo hi hlook hfresh =>
      intro _h
      have hlook' : storeLookup
          (storeExtend (storeRemove sigma ell) ellOut { shape := w.shape, data := lo })
          ellOut =
          some { shape := w.shape, data := lo } := by
        simp [storeLookup, storeExtend]
      exact Or.inl (runtimeSafeConfig_loc hlook')
  | handleRet sigma epsH v clauses hv =>
      intro h
      exact Or.inl <|
        (runtimeSafeConfig_plug (sigma := sigma) (E := EvalCtx.handle epsH clauses)
          (e := v) (by simpa [plug] using h)).1
  | handleOpDirect sigma op v epsH clauses x k handlerBody tRet hv hsig hmem =>
      intro _h
      exact Or.inr
        (RuntimeSafeDebt.handleOpDirect sigma op v epsH clauses x k handlerBody tRet
          hv hsig hmem)
  | handleOpCtx sigma op v epsH E clauses xVar kVar hb tRet hv hsig hmem hop hE =>
      intro _h
      exact Or.inr
        (RuntimeSafeDebt.handleOpCtx sigma op v epsH E clauses xVar kVar hb tRet
          hv hsig hmem hop hE)
  | handleOpCtxs sigma op v epsH Es clauses xVar kVar hb tRet hv hsig hmem hop hEs =>
      intro _h
      exact Or.inr
        (RuntimeSafeDebt.handleOpCtxs sigma op v epsH Es clauses xVar kVar hb tRet
          hv hsig hmem hop hEs)
  | tgrad sigma x tv tOut body hSupp =>
      intro _h
      exact Or.inr (RuntimeSafeDebt.tgrad sigma x tv tOut body hSupp)
  | tvmap sigma x tv d body =>
      intro h
      rcases h with ⟨hSafe, hLive⟩
      rcases hSafe with ⟨hHandler, hSubst⟩
      rcases (by simpa [HandlerAwareRuntimeLinear, ActiveRuntimeLinear, activeLocRefs] using hHandler)
        with ⟨hActLocBody, hBodyHandler⟩
      rcases (by simpa [SubstAwareRuntimeLinear, activeVarRefs] using hSubst)
        with ⟨hActVarBody, hBodySubst⟩
      have hHandler' :
          HandlerAwareRuntimeLinear (Term.abs x (addDim d tv) (addDimTerm d body)) := by
        refine ⟨?_, handlerAwareRuntimeLinear_addDimTerm d hBodyHandler⟩
        simpa [ActiveRuntimeLinear, activeLocRefs, activeLocRefs_addDimTerm (d := d) (e := body)]
          using hActLocBody
      have hSubst' :
          SubstAwareRuntimeLinear (Term.abs x (addDim d tv) (addDimTerm d body)) := by
        refine ⟨?_, substAwareRuntimeLinear_addDimTerm d hBodySubst⟩
        simpa [SubstAwareRuntimeLinear, activeVarRefs,
          activeVarRefs_addDimTerm (d := d) (e := body)] using hActVarBody
      have hLive' :
          StoreLiveLocRefs sigma (Term.abs x (addDim d tv) (addDimTerm d body)) := by
        intro ell hmem
        exact hLive ell (by
          simpa [locRefs, locRefs_addDimTerm (d := d) (e := body)] using hmem)
      exact Or.inl ⟨⟨hHandler', hSubst'⟩, hLive'⟩
  | ctx sigma sigma' E e e' h_inner ih =>
      intro h
      rcases runtimeSafeConfig_plug (sigma := sigma) (E := E) (e := e) h with
        ⟨hInner, hCtx⟩
      rcases ih hInner with hInner' | hDebt
      · cases E with
        | hole =>
            exact Or.inl (by simpa [plug] using hInner')
        | appL e2 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.appL e2) e e' h_inner)
        | appR v1 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.appR v1) e e' h_inner)
        | letBind x e2 =>
            exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.letBind x e2) e e' h_inner)
        | copy =>
            have hPlug : SubstAwareHandlerRuntimeLinear (plug EvalCtx.copy e') := by
              rcases hInner'.1 with ⟨hHandler, hSubst⟩
              exact ⟨by
                  simpa [plug, HandlerAwareRuntimeLinear, ActiveRuntimeLinear, activeLocRefs] using
                    (show ActiveRuntimeLinear e' ∧ HandlerAwareRuntimeLinear e' from
                      ⟨handlerAwareRuntimeLinear_active hHandler, hHandler⟩),
                by
                  simpa [plug, SubstAwareRuntimeLinear, activeVarRefs] using
                    (show (activeVarRefs e').Nodup ∧ SubstAwareRuntimeLinear e' from
                      ⟨substAwareRuntimeLinear_activeNodup hSubst, hSubst⟩)⟩
            have hCtx' : RuntimeSafeCtx sigma' EvalCtx.copy := by
              simp [RuntimeSafeCtx, DeepActiveCtx, StoreLiveCtxLocRefs, ctxLocRefs]
            exact Or.inl (runtimeSafeConfig_ctx hCtx' hInner' hPlug)
        | letpair x y e2 =>
            exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.letpair x y e2) e e' h_inner)
        | pairL e2 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.pairL e2) e e' h_inner)
        | pairR v1 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.pairR v1) e e' h_inner)
        | fst tRight =>
            have hPlug : SubstAwareHandlerRuntimeLinear (plug (EvalCtx.fst tRight) e') := by
              rcases hInner'.1 with ⟨hHandler, hSubst⟩
              exact ⟨by
                  simpa [plug, HandlerAwareRuntimeLinear, ActiveRuntimeLinear, activeLocRefs] using
                    (show ActiveRuntimeLinear e' ∧ HandlerAwareRuntimeLinear e' from
                      ⟨handlerAwareRuntimeLinear_active hHandler, hHandler⟩),
                by
                  simpa [plug, SubstAwareRuntimeLinear, activeVarRefs] using
                    (show (activeVarRefs e').Nodup ∧ SubstAwareRuntimeLinear e' from
                      ⟨substAwareRuntimeLinear_activeNodup hSubst, hSubst⟩)⟩
            have hCtx' : RuntimeSafeCtx sigma' (EvalCtx.fst tRight) := by
              simp [RuntimeSafeCtx, DeepActiveCtx, StoreLiveCtxLocRefs, ctxLocRefs]
            exact Or.inl (runtimeSafeConfig_ctx hCtx' hInner' hPlug)
        | snd tLeft =>
            have hPlug : SubstAwareHandlerRuntimeLinear (plug (EvalCtx.snd tLeft) e') := by
              rcases hInner'.1 with ⟨hHandler, hSubst⟩
              exact ⟨by
                  simpa [plug, HandlerAwareRuntimeLinear, ActiveRuntimeLinear, activeLocRefs] using
                    (show ActiveRuntimeLinear e' ∧ HandlerAwareRuntimeLinear e' from
                      ⟨handlerAwareRuntimeLinear_active hHandler, hHandler⟩),
                by
                  simpa [plug, SubstAwareRuntimeLinear, activeVarRefs] using
                    (show (activeVarRefs e').Nodup ∧ SubstAwareRuntimeLinear e' from
                      ⟨substAwareRuntimeLinear_activeNodup hSubst, hSubst⟩)⟩
            have hCtx' : RuntimeSafeCtx sigma' (EvalCtx.snd tLeft) := by
              simp [RuntimeSafeCtx, DeepActiveCtx, StoreLiveCtxLocRefs, ctxLocRefs]
            exact Or.inl (runtimeSafeConfig_ctx hCtx' hInner' hPlug)
        | addL e2 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.addL e2) e e' h_inner)
        | addR v1 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.addR v1) e e' h_inner)
        | mulL e2 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.mulL e2) e e' h_inner)
        | mulR v1 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.mulR v1) e e' h_inner)
        | sum d =>
            have hPlug : SubstAwareHandlerRuntimeLinear (plug (EvalCtx.sum d) e') := by
              rcases hInner'.1 with ⟨hHandler, hSubst⟩
              exact ⟨by
                  simpa [plug, HandlerAwareRuntimeLinear, ActiveRuntimeLinear, activeLocRefs] using
                    (show ActiveRuntimeLinear e' ∧ HandlerAwareRuntimeLinear e' from
                      ⟨handlerAwareRuntimeLinear_active hHandler, hHandler⟩),
                by
                  simpa [plug, SubstAwareRuntimeLinear, activeVarRefs] using
                    (show (activeVarRefs e').Nodup ∧ SubstAwareRuntimeLinear e' from
                      ⟨substAwareRuntimeLinear_activeNodup hSubst, hSubst⟩)⟩
            have hCtx' : RuntimeSafeCtx sigma' (EvalCtx.sum d) := by
              simp [RuntimeSafeCtx, DeepActiveCtx, StoreLiveCtxLocRefs, ctxLocRefs]
            exact Or.inl (runtimeSafeConfig_ctx hCtx' hInner' hPlug)
        | expand d =>
            have hPlug : SubstAwareHandlerRuntimeLinear (plug (EvalCtx.expand d) e') := by
              rcases hInner'.1 with ⟨hHandler, hSubst⟩
              exact ⟨by
                  simpa [plug, HandlerAwareRuntimeLinear, ActiveRuntimeLinear, activeLocRefs] using
                    (show ActiveRuntimeLinear e' ∧ HandlerAwareRuntimeLinear e' from
                      ⟨handlerAwareRuntimeLinear_active hHandler, hHandler⟩),
                by
                  simpa [plug, SubstAwareRuntimeLinear, activeVarRefs] using
                    (show (activeVarRefs e').Nodup ∧ SubstAwareRuntimeLinear e' from
                      ⟨substAwareRuntimeLinear_activeNodup hSubst, hSubst⟩)⟩
            have hCtx' : RuntimeSafeCtx sigma' (EvalCtx.expand d) := by
              simp [RuntimeSafeCtx, DeepActiveCtx, StoreLiveCtxLocRefs, ctxLocRefs]
            exact Or.inl (runtimeSafeConfig_ctx hCtx' hInner' hPlug)
        | uniformLike lo hi =>
            have hPlug : SubstAwareHandlerRuntimeLinear (plug (EvalCtx.uniformLike lo hi) e') := by
              rcases hInner'.1 with ⟨hHandler, hSubst⟩
              exact ⟨by
                  simpa [plug, HandlerAwareRuntimeLinear, ActiveRuntimeLinear, activeLocRefs] using
                    (show ActiveRuntimeLinear e' ∧ HandlerAwareRuntimeLinear e' from
                      ⟨handlerAwareRuntimeLinear_active hHandler, hHandler⟩),
                by
                  simpa [plug, SubstAwareRuntimeLinear, activeVarRefs] using
                    (show (activeVarRefs e').Nodup ∧ SubstAwareRuntimeLinear e' from
                      ⟨substAwareRuntimeLinear_activeNodup hSubst, hSubst⟩)⟩
            have hCtx' : RuntimeSafeCtx sigma' (EvalCtx.uniformLike lo hi) := by
              simp [RuntimeSafeCtx, DeepActiveCtx, StoreLiveCtxLocRefs, ctxLocRefs]
            exact Or.inl (runtimeSafeConfig_ctx hCtx' hInner' hPlug)
        | handle epsH clauses =>
            exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.handle epsH clauses) e e' h_inner)
        | perform op =>
            have hPlug : SubstAwareHandlerRuntimeLinear (plug (EvalCtx.perform op) e') := by
              rcases hInner'.1 with ⟨hHandler, hSubst⟩
              exact ⟨by
                  simpa [plug, HandlerAwareRuntimeLinear, ActiveRuntimeLinear, activeLocRefs] using
                    (show ActiveRuntimeLinear e' ∧ HandlerAwareRuntimeLinear e' from
                      ⟨handlerAwareRuntimeLinear_active hHandler, hHandler⟩),
                by
                  simpa [plug, SubstAwareRuntimeLinear, activeVarRefs] using
                    (show (activeVarRefs e').Nodup ∧ SubstAwareRuntimeLinear e' from
                      ⟨substAwareRuntimeLinear_activeNodup hSubst, hSubst⟩)⟩
            have hCtx' : RuntimeSafeCtx sigma' (EvalCtx.perform op) := by
              simp [RuntimeSafeCtx, DeepActiveCtx, StoreLiveCtxLocRefs, ctxLocRefs]
            exact Or.inl (runtimeSafeConfig_ctx hCtx' hInner' hPlug)
      · cases E with
        | hole =>
            exact Or.inr hDebt
        | appL e2 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.appL e2) e e' h_inner)
        | appR v1 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.appR v1) e e' h_inner)
        | letBind x e2 =>
            exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.letBind x e2) e e' h_inner)
        | copy =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' EvalCtx.copy e e' h_inner)
        | letpair x y e2 =>
            exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.letpair x y e2) e e' h_inner)
        | pairL e2 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.pairL e2) e e' h_inner)
        | pairR v1 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.pairR v1) e e' h_inner)
        | fst tRight =>
            exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.fst tRight) e e' h_inner)
        | snd tLeft =>
            exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.snd tLeft) e e' h_inner)
        | addL e2 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.addL e2) e e' h_inner)
        | addR v1 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.addR v1) e e' h_inner)
        | mulL e2 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.mulL e2) e e' h_inner)
        | mulR v1 =>
            exact Or.inr (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.mulR v1) e e' h_inner)
        | sum d =>
            exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.sum d) e e' h_inner)
        | expand d =>
            exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.expand d) e e' h_inner)
        | uniformLike lo hi =>
            exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma'
                (EvalCtx.uniformLike lo hi) e e' h_inner)
        | handle epsH clauses =>
            exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma'
                (EvalCtx.handle epsH clauses) e e' h_inner)
        | perform op =>
            exact Or.inr
              (RuntimeSafeDebt.ctx sigma sigma' (EvalCtx.perform op) e e' h_inner)

theorem linearity_soundness
    (sigma sigma' : Store) (Sigma : StoreTyp)
    (e e' : Term)
    (h_wf : StoreWf sigma Sigma)
    (h_step : Step ⟨sigma, e⟩ ⟨sigma', e'⟩) :
    ∃ Sigma', StoreWf sigma' Sigma' :=
  linearity_soundness_aux Sigma ⟨sigma, e⟩ ⟨sigma', e'⟩ h_wf h_step

end LaCaDiLE
