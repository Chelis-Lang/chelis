-- LaCaDiLE/Operational.lean — Step inductive relation.
--
-- Encodes every reduction rule from proof/paper/figures/opsem.tex as a
-- constructor of the `Step` relation. Phase 1 T8: structural skeleton.
--
-- Phase 1 skeleton simplification: this file encodes only the **head
-- reductions** (redex-at-the-outermost-position). The full small-step
-- relation is the congruence closure of these rules via an evaluation
-- context E[·] (see opsem.tex E-Ctx). Phase 2 will add the congruence
-- closure as either a single `ctx` constructor parameterized by an
-- `EvalCtx` inductive, or as inline per-position congruence constructors.
--
-- Substitution is naive (no alpha-renaming). For Phase 1 the skeleton
-- uses `partial def` to avoid termination proofs; Phase 2 WS2.1 will
-- prove a total substitution lemma with capture avoidance.

import LaCaDiLE.Syntax
import LaCaDiLE.StringHelpers

namespace LaCaDiLE

-- Naive capture-unaware substitution: replace every free occurrence of
-- `x` in `target` with `v`. Stops descending into scopes that shadow
-- `x`. Wave 1 makes this a total `def` (no `partial`) via mutual
-- recursion with `substClauses` so Lean's structural checker accepts
-- the handler-clause case without `List.map`, and equation lemmas are
-- available for unfolding in proofs.
mutual

def subst (target : Term) (v : Term) (x : String) : Term :=
  match target with
  | Term.var y => if y = x then v else Term.var y
  | Term.abs y t body =>
      if y = x then Term.abs y t body else Term.abs y t (subst body v x)
  | Term.app e1 e2 => Term.app (subst e1 v x) (subst e2 v x)
  | Term.letBind y e1 e2 =>
      Term.letBind y (subst e1 v x) (if y = x then e2 else subst e2 v x)
  | Term.copy e => Term.copy (subst e v x)
  | Term.letpair y z e1 e2 =>
      Term.letpair y z (subst e1 v x)
        (if y = x ∨ z = x then e2 else subst e2 v x)
  | Term.pair e1 e2 => Term.pair (subst e1 v x) (subst e2 v x)
  | Term.fst e => Term.fst (subst e v x)
  | Term.snd e => Term.snd (subst e v x)
  | Term.unit => Term.unit
  | Term.const c ds => Term.const c ds
  | Term.add e1 e2 => Term.add (subst e1 v x) (subst e2 v x)
  | Term.mul e1 e2 => Term.mul (subst e1 v x) (subst e2 v x)
  | Term.sum e d => Term.sum (subst e v x) d
  | Term.expand e d => Term.expand (subst e v x) d
  | Term.uniformLike e lo hi => Term.uniformLike (subst e v x) lo hi
  | Term.grad y t tOut body =>
      if y = x then Term.grad y t tOut body
      else Term.grad y t tOut (subst body v x)
  | Term.vmap y t body =>
      if y = x then Term.vmap y t body
      else Term.vmap y t (subst body v x)
  | Term.handle epsH body clauses =>
      Term.handle epsH (subst body v x) (substClauses clauses v x)
  | Term.perform op e => Term.perform op (subst e v x)
  | Term.loc ell => Term.loc ell

def substClauses
    (clauses : List (EffectLabel × String × String × Term))
    (v : Term) (x : String) :
    List (EffectLabel × String × String × Term) :=
  match clauses with
  | [] => []
  | (op, y, k, body) :: rest =>
      let body' := if y = x ∨ k = x then body else subst body v x
      (op, y, k, body') :: substClauses rest v x

end

/-- Capture a tensor value at location `ell` into a `TensorVal`. Used in
    placeholders for primitive reduction rules; the actual pointwise
    operations (`oplus`, `odot`, etc.) are not exercised by Phase 1
    metatheory. -/
def tensorOpPlaceholder (_l1 _l2 : TensorVal) : TensorVal :=
  { shape := DimList.empty, data := 0.0 }

/-! ## Evaluation contexts (Wave 0.5)

Call-by-value left-to-right evaluation contexts. `EvalCtx` enumerates
the positions where reduction can happen under a compound term; each
constructor names one such position, and `plug E e` produces the term
with `e` substituted at the hole. The `Step.ctx` congruence rule
reduces `plug E e` to `plug E e'` whenever `e` reduces to `e'`. This
is the standard workaround for needing one structural reduction rule
per (term-former × sub-position) combination. -/

inductive EvalCtx where
  | hole                                            : EvalCtx
  | appL       (e2 : Term)                          : EvalCtx
  | appR       (v1 : Term)                          : EvalCtx
  | letBind    (x : String) (e2 : Term)             : EvalCtx
  | copy                                            : EvalCtx
  | letpair    (x y : String) (e2 : Term)           : EvalCtx
  | pairL      (e2 : Term)                          : EvalCtx
  | pairR      (v1 : Term)                          : EvalCtx
  | fst                                             : EvalCtx
  | snd                                             : EvalCtx
  | addL       (e2 : Term)                          : EvalCtx
  | addR       (v1 : Term)                          : EvalCtx
  | mulL       (e2 : Term)                          : EvalCtx
  | mulR       (v1 : Term)                          : EvalCtx
  | sum        (d : Dim)                            : EvalCtx
  | expand     (d : Dim)                            : EvalCtx
  | uniformLike (lo hi : Float)                     : EvalCtx
  | handle     (epsH : EffectRow)
               (clauses : List (EffectLabel × String × String × Term))
                                                    : EvalCtx
  | perform    (op : EffectLabel)                   : EvalCtx
  deriving Repr

/-- Plug a term into an evaluation context, producing the compound
    term with the hole replaced by `e`. -/
def plug : EvalCtx → Term → Term
  | EvalCtx.hole, e                => e
  | EvalCtx.appL e2, e             => Term.app e e2
  | EvalCtx.appR v1, e             => Term.app v1 e
  | EvalCtx.letBind x e2, e        => Term.letBind x e e2
  | EvalCtx.copy, e                => Term.copy e
  | EvalCtx.letpair x y e2, e      => Term.letpair x y e e2
  | EvalCtx.pairL e2, e            => Term.pair e e2
  | EvalCtx.pairR v1, e            => Term.pair v1 e
  | EvalCtx.fst, e                 => Term.fst e
  | EvalCtx.snd, e                 => Term.snd e
  | EvalCtx.addL e2, e             => Term.add e e2
  | EvalCtx.addR v1, e             => Term.add v1 e
  | EvalCtx.mulL e2, e             => Term.mul e e2
  | EvalCtx.mulR v1, e             => Term.mul v1 e
  | EvalCtx.sum d, e               => Term.sum e d
  | EvalCtx.expand d, e            => Term.expand e d
  | EvalCtx.uniformLike lo hi, e   => Term.uniformLike e lo hi
  | EvalCtx.handle epsH clauses, e => Term.handle epsH e clauses
  | EvalCtx.perform op, e          => Term.perform op e

/-- Runtime locations mentioned by the outer frame of a one-step
    evaluation context, excluding the hole term itself. -/
def ctxLocRefs : EvalCtx → List Loc
  | EvalCtx.hole => []
  | EvalCtx.appL e2 => locRefs e2
  | EvalCtx.appR v1 => locRefs v1
  | EvalCtx.letBind _ e2 => locRefs e2
  | EvalCtx.copy => []
  | EvalCtx.letpair _ _ e2 => locRefs e2
  | EvalCtx.pairL e2 => locRefs e2
  | EvalCtx.pairR v1 => locRefs v1
  | EvalCtx.fst => []
  | EvalCtx.snd => []
  | EvalCtx.addL e2 => locRefs e2
  | EvalCtx.addR v1 => locRefs v1
  | EvalCtx.mulL e2 => locRefs e2
  | EvalCtx.mulR v1 => locRefs v1
  | EvalCtx.sum _ => []
  | EvalCtx.expand _ => []
  | EvalCtx.uniformLike _ _ => []
  | EvalCtx.handle _ clauses => locRefsClauses clauses
  | EvalCtx.perform _ => []

/-- Active runtime locations mentioned by the outer frame of a one-step
    evaluation context, excluding the hole term itself. Handler clauses
    are dormant and therefore contribute nothing to the active
    footprint. -/
def activeCtxLocRefs : EvalCtx → List Loc
  | EvalCtx.hole => []
  | EvalCtx.appL e2 => activeLocRefs e2
  | EvalCtx.appR v1 => activeLocRefs v1
  | EvalCtx.letBind _ e2 => activeLocRefs e2
  | EvalCtx.copy => []
  | EvalCtx.letpair _ _ e2 => activeLocRefs e2
  | EvalCtx.pairL e2 => activeLocRefs e2
  | EvalCtx.pairR v1 => activeLocRefs v1
  | EvalCtx.fst => []
  | EvalCtx.snd => []
  | EvalCtx.addL e2 => activeLocRefs e2
  | EvalCtx.addR v1 => activeLocRefs v1
  | EvalCtx.mulL e2 => activeLocRefs e2
  | EvalCtx.mulR v1 => activeLocRefs v1
  | EvalCtx.sum _ => []
  | EvalCtx.expand _ => []
  | EvalCtx.uniformLike _ _ => []
  | EvalCtx.handle _ clauses => activeLocRefsClauses clauses
  | EvalCtx.perform _ => []

/-- Separation of runtime-location mentions between two lists. -/
def LocRefsSeparated (xs ys : List Loc) : Prop :=
  ∀ ell, ell ∈ xs → ell ∉ ys

private theorem locRefsSeparated_left_of_nodup_append
    {xs ys : List Loc}
    (h : (xs ++ ys).Nodup) :
    LocRefsSeparated xs ys := by
  rcases List.nodup_append.mp h with ⟨_hxs, _hys, hxy⟩
  intro ell hx hy
  exact hxy ell hx ell hy rfl

private theorem locRefsSeparated_right_of_nodup_append
    {xs ys : List Loc}
    (h : (xs ++ ys).Nodup) :
    LocRefsSeparated ys xs := by
  rcases List.nodup_append.mp h with ⟨_hxs, _hys, hxy⟩
  intro ell hy hx
  exact hxy ell hx ell hy rfl

private theorem locRefsSeparated_nil_left
    {xs : List Loc} :
    LocRefsSeparated [] xs := by
  intro ell hmem _hloc
  cases hmem

private theorem locRefsSeparated_nil_right
    {xs : List Loc} :
    LocRefsSeparated xs [] := by
  intro ell _hloc hmem
  cases hmem

/-- A location appears in a plugged term iff it appears either in the
    outer frame or in the hole term. -/
theorem mem_locRefs_plug
    (E : EvalCtx) (e : Term) (ell : Loc) :
    ell ∈ locRefs (plug E e) ↔ ell ∈ ctxLocRefs E ∨ ell ∈ locRefs e := by
  cases E <;> simp [plug, ctxLocRefs, locRefs, or_comm]

theorem mem_activeLocRefs_plug
    (E : EvalCtx) (e : Term) (ell : Loc) :
    ell ∈ activeLocRefs (plug E e) ↔ ell ∈ activeCtxLocRefs E ∨ ell ∈ activeLocRefs e := by
  cases E <;> simp [plug, activeCtxLocRefs, activeLocRefs, activeLocRefsClauses, or_comm]

/-! ### Multi-frame evaluation context chains (Wave 1 P2)

A single `EvalCtx` is a one-hole context of depth one. But the
handle-op rule needs to reach a `perform` sitting under an arbitrary
*stack* of frames, e.g.
`letBind x (letBind y (perform op unit) unit) unit`, whose hole sits
two `letBind` frames deep. An `EvalCtxChain` is a list of frames,
composed outside-in, so that

    multiPlug [E1, E2, E3] e = plug E1 (plug E2 (plug E3 e))

`Es.noHandleFor op` holds iff every frame in the chain is not a
`handle` catching `op`. This lets the new `Step.handleOpCtxs` rule
match `handle epsH (multiPlug Es (perform op v)) clauses` in full
generality while still keeping the handler-shadowing condition. -/

abbrev EvalCtxChain := List EvalCtx

/-- Fold `plug` through a chain of evaluation-context frames. -/
def multiPlug : EvalCtxChain → Term → Term
  | [],      e => e
  | E :: Es, e => plug E (multiPlug Es e)

/-- Runtime locations mentioned by the outer frames of an evaluation
    context chain, excluding the hole term itself. -/
def chainLocRefs : EvalCtxChain → List Loc
  | [] => []
  | E :: Es => ctxLocRefs E ++ chainLocRefs Es

/-- Active runtime locations mentioned by the outer frames of an
    evaluation context chain, excluding the hole term itself. -/
def activeChainLocRefs : EvalCtxChain → List Loc
  | [] => []
  | E :: Es => activeCtxLocRefs E ++ activeChainLocRefs Es

/-- Recursive closure of `DeepActiveRuntimeLinear` for a one-step
    evaluation frame: any sibling terms or dormant handler clauses
    carried by the frame must themselves satisfy the deep active
    invariant. -/
def DeepActiveCtx : EvalCtx → Prop
  | EvalCtx.hole => True
  | EvalCtx.appL e2 => DeepActiveRuntimeLinear e2
  | EvalCtx.appR v1 => DeepActiveRuntimeLinear v1
  | EvalCtx.letBind _ e2 => DeepActiveRuntimeLinear e2
  | EvalCtx.copy => True
  | EvalCtx.letpair _ _ e2 => DeepActiveRuntimeLinear e2
  | EvalCtx.pairL e2 => DeepActiveRuntimeLinear e2
  | EvalCtx.pairR v1 => DeepActiveRuntimeLinear v1
  | EvalCtx.fst => True
  | EvalCtx.snd => True
  | EvalCtx.addL e2 => DeepActiveRuntimeLinear e2
  | EvalCtx.addR v1 => DeepActiveRuntimeLinear v1
  | EvalCtx.mulL e2 => DeepActiveRuntimeLinear e2
  | EvalCtx.mulR v1 => DeepActiveRuntimeLinear v1
  | EvalCtx.sum _ => True
  | EvalCtx.expand _ => True
  | EvalCtx.uniformLike _ _ => True
  | EvalCtx.handle _ clauses => DeepActiveRuntimeLinearClauses clauses
  | EvalCtx.perform _ => True

/-- Chain version of `DeepActiveCtx`. -/
def DeepActiveChain : EvalCtxChain → Prop
  | [] => True
  | E :: Es => DeepActiveCtx E ∧ DeepActiveChain Es

theorem mem_locRefs_multiPlug
    (Es : EvalCtxChain) (e : Term) (ell : Loc) :
    ell ∈ locRefs (multiPlug Es e) ↔ ell ∈ chainLocRefs Es ∨ ell ∈ locRefs e := by
  induction Es with
  | nil =>
      simp [multiPlug, chainLocRefs]
  | cons E Es ih =>
      simp [multiPlug, chainLocRefs, mem_locRefs_plug, ih, or_left_comm, or_assoc]

theorem mem_activeLocRefs_multiPlug
    (Es : EvalCtxChain) (e : Term) (ell : Loc) :
    ell ∈ activeLocRefs (multiPlug Es e) ↔
      ell ∈ activeChainLocRefs Es ∨ ell ∈ activeLocRefs e := by
  induction Es with
  | nil =>
      simp [multiPlug, activeChainLocRefs]
  | cons E Es ih =>
      simp [multiPlug, activeChainLocRefs, mem_activeLocRefs_plug, ih, or_left_comm, or_assoc]

/-- Every deep-active term has a duplicate-free active footprint at its
    root. -/
theorem deepActiveRuntimeLinear_active
    {e : Term}
    (h : DeepActiveRuntimeLinear e) :
    ActiveRuntimeLinear e := by
  cases e <;>
    simp [DeepActiveRuntimeLinear, ActiveRuntimeLinear,
      activeLocRefs, activeLocRefsClauses] at h ⊢
  all_goals
    first
    | exact h.1
    | exact h

theorem deepActiveCtx_activeNodup
    {E : EvalCtx}
    (h : DeepActiveCtx E) :
    (activeCtxLocRefs E).Nodup := by
  cases E <;> simp [DeepActiveCtx, activeCtxLocRefs] at h ⊢
  all_goals exact deepActiveRuntimeLinear_active h

/-- Runtime linearity of a plugged term forces the hole term to remain
    runtime-linear and disjoint from the frame's explicit location
    references. -/
theorem runtimeLinear_plug
    {E : EvalCtx} {e : Term}
    (h : RuntimeLinear (plug E e)) :
    RuntimeLinear e ∧ LocRefsSeparated (ctxLocRefs E) (locRefs e) := by
  cases E with
  | hole =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h
  | appL e2 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using h) with
        ⟨he, _he2, _hsep⟩
      exact ⟨he, locRefsSeparated_right_of_nodup_append (by simpa [plug, locRefs] using h)⟩
  | appR v1 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using h) with
        ⟨_hv1, he, _hsep⟩
      exact ⟨he, locRefsSeparated_left_of_nodup_append (by simpa [plug, locRefs] using h)⟩
  | letBind x e2 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using h) with
        ⟨he, _he2, _hsep⟩
      exact ⟨he, locRefsSeparated_right_of_nodup_append (by simpa [plug, locRefs] using h)⟩
  | copy =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h
  | letpair x y e2 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using h) with
        ⟨he, _he2, _hsep⟩
      exact ⟨he, locRefsSeparated_right_of_nodup_append (by simpa [plug, locRefs] using h)⟩
  | pairL e2 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using h) with
        ⟨he, _he2, _hsep⟩
      exact ⟨he, locRefsSeparated_right_of_nodup_append (by simpa [plug, locRefs] using h)⟩
  | pairR v1 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using h) with
        ⟨_hv1, he, _hsep⟩
      exact ⟨he, locRefsSeparated_left_of_nodup_append (by simpa [plug, locRefs] using h)⟩
  | fst =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h
  | snd =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h
  | addL e2 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using h) with
        ⟨he, _he2, _hsep⟩
      exact ⟨he, locRefsSeparated_right_of_nodup_append (by simpa [plug, locRefs] using h)⟩
  | addR v1 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using h) with
        ⟨_hv1, he, _hsep⟩
      exact ⟨he, locRefsSeparated_left_of_nodup_append (by simpa [plug, locRefs] using h)⟩
  | mulL e2 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using h) with
        ⟨he, _he2, _hsep⟩
      exact ⟨he, locRefsSeparated_right_of_nodup_append (by simpa [plug, locRefs] using h)⟩
  | mulR v1 =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using h) with
        ⟨_hv1, he, _hsep⟩
      exact ⟨he, locRefsSeparated_left_of_nodup_append (by simpa [plug, locRefs] using h)⟩
  | sum d =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h
  | expand d =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h
  | uniformLike lo hi =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h
  | handle epsH clauses =>
      rcases List.nodup_append.mp (by simpa [RuntimeLinear, plug, locRefs] using h) with
        ⟨he, _hcls, _hsep⟩
      exact ⟨he, locRefsSeparated_right_of_nodup_append (by simpa [plug, locRefs] using h)⟩
  | perform op =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h

/-- Chain version of `runtimeLinear_plug`. -/
theorem runtimeLinear_multiPlug
    {Es : EvalCtxChain} {e : Term}
    (h : RuntimeLinear (multiPlug Es e)) :
    RuntimeLinear e ∧ LocRefsSeparated (chainLocRefs Es) (locRefs e) := by
  induction Es with
  | nil =>
      simpa [multiPlug, chainLocRefs, RuntimeLinear, LocRefsSeparated] using h
  | cons E Es ih =>
      have hOuter := runtimeLinear_plug (E := E) (e := multiPlug Es e) h
      have hInner := ih hOuter.1
      constructor
      · exact hInner.1
      · intro ell hmem
        rcases List.mem_append.mp hmem with hmemE | hmemEs
        · intro hLocE
          have hLocMulti :
              ell ∈ locRefs (multiPlug Es e) := by
            exact (mem_locRefs_multiPlug Es e ell).2 (Or.inr hLocE)
          exact hOuter.2 ell hmemE hLocMulti
        · intro hLocE
          exact hInner.2 ell hmemEs hLocE

/-- Active-footprint version of `runtimeLinear_plug`. -/
theorem activeRuntimeLinear_plug
    {E : EvalCtx} {e : Term}
    (h : ActiveRuntimeLinear (plug E e)) :
    ActiveRuntimeLinear e ∧ LocRefsSeparated (activeCtxLocRefs E) (activeLocRefs e) := by
  cases E with
  | hole =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h
  | appL e2 =>
      rcases List.nodup_append.mp (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using h) with
        ⟨he, _he2, _hsep⟩
      exact ⟨he, locRefsSeparated_right_of_nodup_append (by simpa [plug, activeLocRefs] using h)⟩
  | appR v1 =>
      rcases List.nodup_append.mp (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using h) with
        ⟨_hv1, he, _hsep⟩
      exact ⟨he, locRefsSeparated_left_of_nodup_append (by simpa [plug, activeLocRefs] using h)⟩
  | letBind x e2 =>
      rcases List.nodup_append.mp (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using h) with
        ⟨he, _he2, _hsep⟩
      exact ⟨he, locRefsSeparated_right_of_nodup_append (by simpa [plug, activeLocRefs] using h)⟩
  | copy =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h
  | letpair x y e2 =>
      rcases List.nodup_append.mp (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using h) with
        ⟨he, _he2, _hsep⟩
      exact ⟨he, locRefsSeparated_right_of_nodup_append (by simpa [plug, activeLocRefs] using h)⟩
  | pairL e2 =>
      rcases List.nodup_append.mp (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using h) with
        ⟨he, _he2, _hsep⟩
      exact ⟨he, locRefsSeparated_right_of_nodup_append (by simpa [plug, activeLocRefs] using h)⟩
  | pairR v1 =>
      rcases List.nodup_append.mp (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using h) with
        ⟨_hv1, he, _hsep⟩
      exact ⟨he, locRefsSeparated_left_of_nodup_append (by simpa [plug, activeLocRefs] using h)⟩
  | fst =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h
  | snd =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h
  | addL e2 =>
      rcases List.nodup_append.mp (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using h) with
        ⟨he, _he2, _hsep⟩
      exact ⟨he, locRefsSeparated_right_of_nodup_append (by simpa [plug, activeLocRefs] using h)⟩
  | addR v1 =>
      rcases List.nodup_append.mp (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using h) with
        ⟨_hv1, he, _hsep⟩
      exact ⟨he, locRefsSeparated_left_of_nodup_append (by simpa [plug, activeLocRefs] using h)⟩
  | mulL e2 =>
      rcases List.nodup_append.mp (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using h) with
        ⟨he, _he2, _hsep⟩
      exact ⟨he, locRefsSeparated_right_of_nodup_append (by simpa [plug, activeLocRefs] using h)⟩
  | mulR v1 =>
      rcases List.nodup_append.mp (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using h) with
        ⟨_hv1, he, _hsep⟩
      exact ⟨he, locRefsSeparated_left_of_nodup_append (by simpa [plug, activeLocRefs] using h)⟩
  | sum d =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h
  | expand d =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h
  | uniformLike lo hi =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h
  | handle epsH clauses =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs,
        activeLocRefs, activeLocRefsClauses] using h
  | perform op =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h

/-- Chain version of `activeRuntimeLinear_plug`. -/
theorem activeRuntimeLinear_multiPlug
    {Es : EvalCtxChain} {e : Term}
    (h : ActiveRuntimeLinear (multiPlug Es e)) :
    ActiveRuntimeLinear e ∧ LocRefsSeparated (activeChainLocRefs Es) (activeLocRefs e) := by
  induction Es with
  | nil =>
      simpa [multiPlug, activeChainLocRefs, ActiveRuntimeLinear, LocRefsSeparated] using h
  | cons E Es ih =>
      have hOuter := activeRuntimeLinear_plug (E := E) (e := multiPlug Es e) h
      have hInner := ih hOuter.1
      constructor
      · exact hInner.1
      · intro ell hmem
        rcases List.mem_append.mp hmem with hmemE | hmemEs
        · intro hLocE
          have hLocMulti :
              ell ∈ activeLocRefs (multiPlug Es e) := by
            exact (mem_activeLocRefs_multiPlug Es e ell).2 (Or.inr hLocE)
          exact hOuter.2 ell hmemE hLocMulti
        · intro hLocE
          exact hInner.2 ell hmemEs hLocE

/-- Deep-active version of `activeRuntimeLinear_plug`: plugging a term
    into a deep-active frame lets us recover the hole term, the frame's
    own recursive invariant, and separation of their active
    footprints. -/
theorem deepActiveRuntimeLinear_plug
    {E : EvalCtx} {e : Term}
    (h : DeepActiveRuntimeLinear (plug E e)) :
    DeepActiveRuntimeLinear e ∧ DeepActiveCtx E ∧
      LocRefsSeparated (activeCtxLocRefs E) (activeLocRefs e) ∧
      LocRefsSeparated (activeLocRefs e) (activeCtxLocRefs E) := by
  cases E with
  | hole =>
      simpa [DeepActiveRuntimeLinear, DeepActiveCtx, LocRefsSeparated,
        plug, activeCtxLocRefs, activeLocRefs] using h
  | appL e2 =>
      rcases h with ⟨hAct, he, he2⟩
      exact ⟨he, he2,
        locRefsSeparated_right_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct),
        locRefsSeparated_left_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct)⟩
  | appR v1 =>
      rcases h with ⟨hAct, hv1, he⟩
      exact ⟨he, hv1,
        locRefsSeparated_left_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct),
        locRefsSeparated_right_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct)⟩
  | letBind x e2 =>
      rcases h with ⟨hAct, he, he2⟩
      exact ⟨he, he2,
        locRefsSeparated_right_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct),
        locRefsSeparated_left_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct)⟩
  | copy =>
      rcases h with ⟨_hAct, he⟩
      exact ⟨he, ⟨trivial, ⟨locRefsSeparated_nil_left, locRefsSeparated_nil_right⟩⟩⟩
  | letpair x y e2 =>
      rcases h with ⟨hAct, he, he2⟩
      exact ⟨he, he2,
        locRefsSeparated_right_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct),
        locRefsSeparated_left_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct)⟩
  | pairL e2 =>
      rcases h with ⟨hAct, he, he2⟩
      exact ⟨he, he2,
        locRefsSeparated_right_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct),
        locRefsSeparated_left_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct)⟩
  | pairR v1 =>
      rcases h with ⟨hAct, hv1, he⟩
      exact ⟨he, hv1,
        locRefsSeparated_left_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct),
        locRefsSeparated_right_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct)⟩
  | fst =>
      rcases h with ⟨_hAct, he⟩
      exact ⟨he, ⟨trivial, ⟨locRefsSeparated_nil_left, locRefsSeparated_nil_right⟩⟩⟩
  | snd =>
      rcases h with ⟨_hAct, he⟩
      exact ⟨he, ⟨trivial, ⟨locRefsSeparated_nil_left, locRefsSeparated_nil_right⟩⟩⟩
  | addL e2 =>
      rcases h with ⟨hAct, he, he2⟩
      exact ⟨he, he2,
        locRefsSeparated_right_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct),
        locRefsSeparated_left_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct)⟩
  | addR v1 =>
      rcases h with ⟨hAct, hv1, he⟩
      exact ⟨he, hv1,
        locRefsSeparated_left_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct),
        locRefsSeparated_right_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct)⟩
  | mulL e2 =>
      rcases h with ⟨hAct, he, he2⟩
      exact ⟨he, he2,
        locRefsSeparated_right_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct),
        locRefsSeparated_left_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct)⟩
  | mulR v1 =>
      rcases h with ⟨hAct, hv1, he⟩
      exact ⟨he, hv1,
        locRefsSeparated_left_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct),
        locRefsSeparated_right_of_nodup_append
          (by simpa [ActiveRuntimeLinear, plug, activeLocRefs] using hAct)⟩
  | sum d =>
      rcases h with ⟨_hAct, he⟩
      exact ⟨he, ⟨trivial, ⟨locRefsSeparated_nil_left, locRefsSeparated_nil_right⟩⟩⟩
  | expand d =>
      rcases h with ⟨_hAct, he⟩
      exact ⟨he, ⟨trivial, ⟨locRefsSeparated_nil_left, locRefsSeparated_nil_right⟩⟩⟩
  | uniformLike lo hi =>
      rcases h with ⟨_hAct, he⟩
      exact ⟨he, ⟨trivial, ⟨locRefsSeparated_nil_left, locRefsSeparated_nil_right⟩⟩⟩
  | handle epsH clauses =>
      rcases h with ⟨_hAct, he, hclauses⟩
      refine ⟨he, ⟨hclauses, ⟨?_, ?_⟩⟩⟩
      · simpa [activeCtxLocRefs, activeLocRefsClauses] using
          (locRefsSeparated_nil_left (xs := activeLocRefs e))
      · simpa [activeCtxLocRefs, activeLocRefsClauses] using
          (locRefsSeparated_nil_right (xs := activeLocRefs e))
  | perform op =>
      rcases h with ⟨_hAct, he⟩
      exact ⟨he, ⟨trivial, ⟨locRefsSeparated_nil_left, locRefsSeparated_nil_right⟩⟩⟩

/-- Chain version of `deepActiveRuntimeLinear_plug`. -/
theorem deepActiveRuntimeLinear_multiPlug
    {Es : EvalCtxChain} {e : Term}
    (h : DeepActiveRuntimeLinear (multiPlug Es e)) :
    DeepActiveRuntimeLinear e ∧ DeepActiveChain Es ∧
      LocRefsSeparated (activeChainLocRefs Es) (activeLocRefs e) ∧
      LocRefsSeparated (activeLocRefs e) (activeChainLocRefs Es) := by
  induction Es with
  | nil =>
      exact ⟨h, ⟨trivial, ⟨locRefsSeparated_nil_left, locRefsSeparated_nil_right⟩⟩⟩
  | cons E Es ih =>
      rcases deepActiveRuntimeLinear_plug (E := E) (e := multiPlug Es e) h with
        ⟨hInner, hE, hSepOuter, hSepOuterSymm⟩
      rcases ih hInner with ⟨he, hEs, hSepInner, hSepInnerSymm⟩
      refine ⟨he, ⟨hE, hEs⟩, ?_, ?_⟩
      · intro ell hmem hLoc
        rcases List.mem_append.mp hmem with hmemE | hmemEs
        · have hLocMulti :
            ell ∈ activeLocRefs (multiPlug Es e) := by
            exact (mem_activeLocRefs_multiPlug Es e ell).2 (Or.inr hLoc)
          exact hSepOuter ell hmemE hLocMulti
        · exact hSepInner ell hmemEs hLoc
      · intro ell hLoc hmem
        rcases List.mem_append.mp hmem with hmemE | hmemEs
        · have hLocMulti :
            ell ∈ activeLocRefs (multiPlug Es e) := by
            exact (mem_activeLocRefs_multiPlug Es e ell).2 (Or.inr hLoc)
          exact hSepOuterSymm ell hLocMulti hmemE
        · exact hSepInnerSymm ell hLoc hmemEs

/-- Converse direction of `runtimeLinear_plug`: plugging preserves
    runtime linearity when the frame's own location references are
    duplicate-free and separated from the hole term in both directions. -/
theorem runtimeLinear_plug_of
    {E : EvalCtx} {e : Term}
    (hCtx : (ctxLocRefs E).Nodup)
    (h : RuntimeLinear e)
    (hSep : LocRefsSeparated (ctxLocRefs E) (locRefs e))
    (hSepSymm : LocRefsSeparated (locRefs e) (ctxLocRefs E)) :
    RuntimeLinear (plug E e) := by
  cases E with
  | hole =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h
  | appL e2 =>
      exact List.nodup_append.mpr ⟨h, hCtx, by
        intro ell hLoc ell' hLocE2 hEq
        subst ell'
        exact hSepSymm ell hLoc hLocE2⟩
  | appR v1 =>
      exact List.nodup_append.mpr ⟨hCtx, h, by
        intro ell hLocV1 ell' hLoc hEq
        subst ell'
        exact hSep ell hLocV1 hLoc⟩
  | letBind x e2 =>
      exact List.nodup_append.mpr ⟨h, hCtx, by
        intro ell hLoc ell' hLocE2 hEq
        subst ell'
        exact hSepSymm ell hLoc hLocE2⟩
  | copy =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h
  | letpair x y e2 =>
      exact List.nodup_append.mpr ⟨h, hCtx, by
        intro ell hLoc ell' hLocE2 hEq
        subst ell'
        exact hSepSymm ell hLoc hLocE2⟩
  | pairL e2 =>
      exact List.nodup_append.mpr ⟨h, hCtx, by
        intro ell hLoc ell' hLocE2 hEq
        subst ell'
        exact hSepSymm ell hLoc hLocE2⟩
  | pairR v1 =>
      exact List.nodup_append.mpr ⟨hCtx, h, by
        intro ell hLocV1 ell' hLoc hEq
        subst ell'
        exact hSep ell hLocV1 hLoc⟩
  | fst =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h
  | snd =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h
  | addL e2 =>
      exact List.nodup_append.mpr ⟨h, hCtx, by
        intro ell hLoc ell' hLocE2 hEq
        subst ell'
        exact hSepSymm ell hLoc hLocE2⟩
  | addR v1 =>
      exact List.nodup_append.mpr ⟨hCtx, h, by
        intro ell hLocV1 ell' hLoc hEq
        subst ell'
        exact hSep ell hLocV1 hLoc⟩
  | mulL e2 =>
      exact List.nodup_append.mpr ⟨h, hCtx, by
        intro ell hLoc ell' hLocE2 hEq
        subst ell'
        exact hSepSymm ell hLoc hLocE2⟩
  | mulR v1 =>
      exact List.nodup_append.mpr ⟨hCtx, h, by
        intro ell hLocV1 ell' hLoc hEq
        subst ell'
        exact hSep ell hLocV1 hLoc⟩
  | sum d =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h
  | expand d =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h
  | uniformLike lo hi =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h
  | handle epsH clauses =>
      exact List.nodup_append.mpr ⟨h, hCtx, by
        intro ell hLoc ell' hLocCls hEq
        subst ell'
        exact hSepSymm ell hLoc hLocCls⟩
  | perform op =>
      simpa [RuntimeLinear, LocRefsSeparated, plug, ctxLocRefs, locRefs] using h

/-- Converse direction of `runtimeLinear_multiPlug`: plugging a whole
    chain preserves runtime linearity when the chain's explicit
    location references are duplicate-free and separated from the hole
    term in both directions. -/
theorem runtimeLinear_multiPlug_of
    {Es : EvalCtxChain} {e : Term}
    (hChain : (chainLocRefs Es).Nodup)
    (h : RuntimeLinear e)
    (hSep : LocRefsSeparated (chainLocRefs Es) (locRefs e))
    (hSepSymm : LocRefsSeparated (locRefs e) (chainLocRefs Es)) :
    RuntimeLinear (multiPlug Es e) := by
  induction Es generalizing e with
  | nil =>
      simpa [multiPlug, chainLocRefs, RuntimeLinear, LocRefsSeparated] using h
  | cons E Es ih =>
      rcases List.nodup_append.mp hChain with ⟨hCtx, hEs, _hCross⟩
      apply runtimeLinear_plug_of hCtx
      · exact ih hEs h
          (fun ell hmemEs hLocE => hSep ell (List.mem_append.mpr <| Or.inr hmemEs) hLocE)
          (fun ell hLocE hmemEs => hSepSymm ell hLocE (List.mem_append.mpr <| Or.inr hmemEs))
      · intro ell hmemE hLocMulti
        rcases (mem_locRefs_multiPlug Es e ell).1 hLocMulti with hmemEs | hLocE
        · exact locRefsSeparated_left_of_nodup_append hChain ell hmemE hmemEs
        · exact hSep ell (List.mem_append.mpr <| Or.inl hmemE) hLocE
      · intro ell hLocMulti hmemE
        rcases (mem_locRefs_multiPlug Es e ell).1 hLocMulti with hmemEs | hLocE
        · exact locRefsSeparated_right_of_nodup_append hChain ell hmemEs hmemE
        · exact hSepSymm ell hLocE (List.mem_append.mpr <| Or.inl hmemE)

/-- Converse direction of `activeRuntimeLinear_plug`: plugging preserves
    the active runtime-linearity footprint when the frame's active
    location references are duplicate-free and separated from the hole
    term in both directions. -/
theorem activeRuntimeLinear_plug_of
    {E : EvalCtx} {e : Term}
    (hCtx : (activeCtxLocRefs E).Nodup)
    (h : ActiveRuntimeLinear e)
    (hSep : LocRefsSeparated (activeCtxLocRefs E) (activeLocRefs e))
    (hSepSymm : LocRefsSeparated (activeLocRefs e) (activeCtxLocRefs E)) :
    ActiveRuntimeLinear (plug E e) := by
  cases E with
  | hole =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h
  | appL e2 =>
      exact List.nodup_append.mpr ⟨h, hCtx, by
        intro ell hLoc ell' hLocE2 hEq
        subst ell'
        exact hSepSymm ell hLoc hLocE2⟩
  | appR v1 =>
      exact List.nodup_append.mpr ⟨hCtx, h, by
        intro ell hLocV1 ell' hLoc hEq
        subst ell'
        exact hSep ell hLocV1 hLoc⟩
  | letBind x e2 =>
      exact List.nodup_append.mpr ⟨h, hCtx, by
        intro ell hLoc ell' hLocE2 hEq
        subst ell'
        exact hSepSymm ell hLoc hLocE2⟩
  | copy =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h
  | letpair x y e2 =>
      exact List.nodup_append.mpr ⟨h, hCtx, by
        intro ell hLoc ell' hLocE2 hEq
        subst ell'
        exact hSepSymm ell hLoc hLocE2⟩
  | pairL e2 =>
      exact List.nodup_append.mpr ⟨h, hCtx, by
        intro ell hLoc ell' hLocE2 hEq
        subst ell'
        exact hSepSymm ell hLoc hLocE2⟩
  | pairR v1 =>
      exact List.nodup_append.mpr ⟨hCtx, h, by
        intro ell hLocV1 ell' hLoc hEq
        subst ell'
        exact hSep ell hLocV1 hLoc⟩
  | fst =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h
  | snd =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h
  | addL e2 =>
      exact List.nodup_append.mpr ⟨h, hCtx, by
        intro ell hLoc ell' hLocE2 hEq
        subst ell'
        exact hSepSymm ell hLoc hLocE2⟩
  | addR v1 =>
      exact List.nodup_append.mpr ⟨hCtx, h, by
        intro ell hLocV1 ell' hLoc hEq
        subst ell'
        exact hSep ell hLocV1 hLoc⟩
  | mulL e2 =>
      exact List.nodup_append.mpr ⟨h, hCtx, by
        intro ell hLoc ell' hLocE2 hEq
        subst ell'
        exact hSepSymm ell hLoc hLocE2⟩
  | mulR v1 =>
      exact List.nodup_append.mpr ⟨hCtx, h, by
        intro ell hLocV1 ell' hLoc hEq
        subst ell'
        exact hSep ell hLocV1 hLoc⟩
  | sum d =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h
  | expand d =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h
  | uniformLike lo hi =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h
  | handle epsH clauses =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs,
        activeLocRefs, activeLocRefsClauses] using h
  | perform op =>
      simpa [ActiveRuntimeLinear, LocRefsSeparated, plug, activeCtxLocRefs, activeLocRefs] using h

/-- Converse direction of `activeRuntimeLinear_multiPlug`: plugging a
    whole chain preserves the active runtime-linearity footprint when
    the chain's active location references are duplicate-free and
    separated from the hole term in both directions. -/
theorem activeRuntimeLinear_multiPlug_of
    {Es : EvalCtxChain} {e : Term}
    (hChain : (activeChainLocRefs Es).Nodup)
    (h : ActiveRuntimeLinear e)
    (hSep : LocRefsSeparated (activeChainLocRefs Es) (activeLocRefs e))
    (hSepSymm : LocRefsSeparated (activeLocRefs e) (activeChainLocRefs Es)) :
    ActiveRuntimeLinear (multiPlug Es e) := by
  induction Es generalizing e with
  | nil =>
      simpa [multiPlug, activeChainLocRefs, ActiveRuntimeLinear, LocRefsSeparated] using h
  | cons E Es ih =>
      rcases List.nodup_append.mp hChain with ⟨hCtx, hEs, _hCross⟩
      apply activeRuntimeLinear_plug_of hCtx
      · exact ih hEs h
          (fun ell hmemEs hLocE =>
            hSep ell (List.mem_append.mpr <| Or.inr hmemEs) hLocE)
          (fun ell hLocE hmemEs =>
            hSepSymm ell hLocE (List.mem_append.mpr <| Or.inr hmemEs))
      · intro ell hmemE hLocMulti
        rcases (mem_activeLocRefs_multiPlug Es e ell).1 hLocMulti with hmemEs | hLocE
        · exact locRefsSeparated_left_of_nodup_append hChain ell hmemE hmemEs
        · exact hSep ell (List.mem_append.mpr <| Or.inl hmemE) hLocE
      · intro ell hLocMulti hmemE
        rcases (mem_activeLocRefs_multiPlug Es e ell).1 hLocMulti with hmemEs | hLocE
        · exact locRefsSeparated_right_of_nodup_append hChain ell hmemEs hmemE
        · exact hSepSymm ell hLocE (List.mem_append.mpr <| Or.inl hmemE)

/-- Converse direction of `deepActiveRuntimeLinear_plug`: plugging
    preserves the recursive active invariant when the frame itself is
    deep-active and its active footprint is separated from the hole
    term in both directions. -/
theorem deepActiveRuntimeLinear_plug_of
    {E : EvalCtx} {e : Term}
    (hCtx : DeepActiveCtx E)
    (h : DeepActiveRuntimeLinear e)
    (hSep : LocRefsSeparated (activeCtxLocRefs E) (activeLocRefs e))
    (hSepSymm : LocRefsSeparated (activeLocRefs e) (activeCtxLocRefs E)) :
    DeepActiveRuntimeLinear (plug E e) := by
  cases E with
  | hole =>
      simpa [DeepActiveRuntimeLinear, DeepActiveCtx, LocRefsSeparated,
        plug, activeCtxLocRefs, activeLocRefs] using h
  | appL e2 =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        h, hCtx⟩
  | appR v1 =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        hCtx, h⟩
  | letBind x e2 =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        h, hCtx⟩
  | copy =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        h⟩
  | letpair x y e2 =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        h, hCtx⟩
  | pairL e2 =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        h, hCtx⟩
  | pairR v1 =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        hCtx, h⟩
  | fst =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        h⟩
  | snd =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        h⟩
  | addL e2 =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        h, hCtx⟩
  | addR v1 =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        hCtx, h⟩
  | mulL e2 =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        h, hCtx⟩
  | mulR v1 =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        hCtx, h⟩
  | sum d =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        h⟩
  | expand d =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        h⟩
  | uniformLike lo hi =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        h⟩
  | handle epsH clauses =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        h, hCtx⟩
  | perform op =>
      exact ⟨activeRuntimeLinear_plug_of
          (deepActiveCtx_activeNodup hCtx)
          (deepActiveRuntimeLinear_active h) hSep hSepSymm,
        h⟩

/-- Converse direction of `deepActiveRuntimeLinear_multiPlug`: plugging
    a whole chain preserves the recursive active invariant when the
    chain's active footprint is duplicate-free, the chain itself is
    deep-active, and the chain/hole footprints are separated in both
    directions. -/
theorem deepActiveRuntimeLinear_multiPlug_of
    {Es : EvalCtxChain} {e : Term}
    (hChain : (activeChainLocRefs Es).Nodup)
    (hEs : DeepActiveChain Es)
    (h : DeepActiveRuntimeLinear e)
    (hSep : LocRefsSeparated (activeChainLocRefs Es) (activeLocRefs e))
    (hSepSymm : LocRefsSeparated (activeLocRefs e) (activeChainLocRefs Es)) :
    DeepActiveRuntimeLinear (multiPlug Es e) := by
  induction Es generalizing e with
  | nil =>
      simpa [multiPlug, DeepActiveChain, activeChainLocRefs, LocRefsSeparated] using h
  | cons E Es ih =>
      rcases hEs with ⟨hE, hEs⟩
      rcases List.nodup_append.mp hChain with ⟨hCtx, hEsNodup, _hCross⟩
      apply deepActiveRuntimeLinear_plug_of hE
      · exact ih hEsNodup hEs h
          (fun ell hmemEs hLocE =>
            hSep ell (List.mem_append.mpr <| Or.inr hmemEs) hLocE)
          (fun ell hLocE hmemEs =>
            hSepSymm ell hLocE (List.mem_append.mpr <| Or.inr hmemEs))
      · intro ell hmemE hLocMulti
        rcases (mem_activeLocRefs_multiPlug Es e ell).1 hLocMulti with hmemEs | hLocE
        · exact locRefsSeparated_left_of_nodup_append hChain ell hmemE hmemEs
        · exact hSep ell (List.mem_append.mpr <| Or.inl hmemE) hLocE
      · intro ell hLocMulti hmemE
        rcases (mem_activeLocRefs_multiPlug Es e ell).1 hLocMulti with hmemEs | hLocE
        · exact locRefsSeparated_right_of_nodup_append hChain ell hmemEs hmemE
        · exact hSepSymm ell hLocE (List.mem_append.mpr <| Or.inl hmemE)

/-- `EvalCtx.noHandleFor op E` holds when the single-step evaluation
    context `E` is not itself a `handle` whose effect row catches `op`.
    Because `EvalCtx` is a one-step (non-recursive) context, this is a
    simple case split: only the `handle` constructor can catch an
    operation, and it does so exactly when `op ∈ epsH`. All other
    contexts trivially do not catch anything at their own level. -/
def EvalCtx.noHandleFor (op : EffectLabel) : EvalCtx → Prop
  | EvalCtx.handle epsH _ => op ∉ epsH
  | _                     => True

/-- Every frame in the chain has `noHandleFor op`. Recursively folded
    over the list; the empty chain trivially satisfies the predicate. -/
def EvalCtxChain.noHandleFor (op : EffectLabel) : EvalCtxChain → Prop
  | []      => True
  | E :: Es => EvalCtx.noHandleFor op E ∧ EvalCtxChain.noHandleFor op Es

/-- Maximum string length in a finite list of names. -/
def maxStringLength : List String → Nat
  | [] => 0
  | s :: rest => Nat.max s.toList.length (maxStringLength rest)

theorem mem_maxStringLength
    {used : List String} {s : String}
    (hmem : s ∈ used) :
    s.toList.length ≤ maxStringLength used := by
  induction used with
  | nil =>
      cases hmem
  | cons hd tl ih =>
      rcases List.mem_cons.mp hmem with rfl | htl
      · exact Nat.le_max_left _ _
      · exact Nat.le_trans (ih htl) (Nat.le_max_right _ _)

/-- A mechanically fresh name for a finite avoid-set, built from the
    shared `freshName` infrastructure. The counter is chosen so the
    resulting string is strictly longer than every string in `used`. -/
def freshNameAvoiding (used : List String) : String :=
  freshName "" (maxStringLength used + 1)

theorem freshNameAvoiding_not_mem
    (used : List String) :
    freshNameAvoiding used ∉ used := by
  intro hmem
  have hle : (freshNameAvoiding used).toList.length ≤ maxStringLength used :=
    mem_maxStringLength hmem
  have hlen : (freshNameAvoiding used).toList.length = maxStringLength used + 2 := by
    unfold freshNameAvoiding
    rw [freshName_toList]
    simp
  have hgt : maxStringLength used < (freshNameAvoiding used).toList.length := by
    rw [hlen]
    exact Nat.lt_succ_of_le (Nat.le_succ _)
  exact Nat.not_lt_of_ge hle hgt

/-- Continuation binder minted for captured-handler operational steps. -/
def capturedContName (e : Term) : String :=
  freshNameAvoiding (freeVars e ++ boundVars e)

theorem capturedContName_freshInTerm
    (e : Term) :
    freshInTerm (capturedContName e) e := by
  unfold freshInTerm capturedContName
  refine ⟨?_, ?_⟩
  · intro hmem
    exact freshNameAvoiding_not_mem _ (List.mem_append.mpr (Or.inl hmem))
  · intro hmem
    exact freshNameAvoiding_not_mem _ (List.mem_append.mpr (Or.inr hmem))

/-- The small-step reduction relation. Constructors cover the head
    reductions (redex at top position); `Step.ctx` provides the
    congruence closure via evaluation contexts. -/
inductive Step : Config → Config → Prop

  /- ## Core λ-calculus redexes -/

  -- E-Beta: (λx:t.e) v  ↦  e[v/x]   when v is a value
  | beta
      (sigma : Store) (x : String) (t : Typ) (e v : Term) :
      IsValue v →
      Step ⟨sigma, Term.app (Term.abs x t e) v⟩
           ⟨sigma, subst e v x⟩

  -- E-Let: let x = v in e  ↦  e[v/x]   when v is a value
  | letBind
      (sigma : Store) (x : String) (v e : Term) :
      IsValue v →
      Step ⟨sigma, Term.letBind x v e⟩
           ⟨sigma, subst e v x⟩

  -- E-LetPair: let (x, y) = (v1, v2) in e  ↦  e[v1/x, v2/y]
  | letpair
      (sigma : Store) (x y : String) (v1 v2 e : Term) :
      IsValue v1 → IsValue v2 →
      Step ⟨sigma, Term.letpair x y (Term.pair v1 v2) e⟩
           ⟨sigma, subst (subst e v1 x) v2 y⟩

  -- E-Fst: fst((v1, v2))  ↦  v1
  | fst
      (sigma : Store) (v1 v2 : Term) :
      IsValue v1 → IsValue v2 →
      Step ⟨sigma, Term.fst (Term.pair v1 v2)⟩
           ⟨sigma, v1⟩

  -- E-Snd: snd((v1, v2))  ↦  v2
  | snd
      (sigma : Store) (v1 v2 : Term) :
      IsValue v1 → IsValue v2 →
      Step ⟨sigma, Term.snd (Term.pair v1 v2)⟩
           ⟨sigma, v2⟩

  /- ## Store-allocating primitives -/

  -- E-Const: const(v, ds)  ↦  fresh location with the literal stored
  | tconst
      (sigma : Store) (v : Float) (ds : DimList) (ell : Loc) :
      ell = storeFreshLoc sigma →
      Step ⟨sigma, Term.const v ds⟩
           ⟨storeExtend sigma ell ⟨ds, v⟩, Term.loc ell⟩

  -- E-Copy: copy(ℓ)  ↦  (ℓ, ℓ')   with fresh ℓ' holding a physical copy
  -- The original location ℓ is retained (first component of the pair).
  | copy
      (sigma : Store) (ell ellNew : Loc) (w : TensorVal) :
      storeLookup sigma ell = some w →
      ellNew = storeFreshLoc sigma →
      Step ⟨sigma, Term.copy (Term.loc ell)⟩
           ⟨storeExtend sigma ellNew w,
            Term.pair (Term.loc ell) (Term.loc ellNew)⟩

  -- E-Add: add(ℓ1, ℓ2)  ↦  fresh ℓ with the elementwise sum; operands freed
  | tadd
      (sigma : Store) (ell1 ell2 ellOut : Loc) (w1 w2 : TensorVal) :
      storeLookup sigma ell1 = some w1 →
      storeLookup sigma ell2 = some w2 →
      ellOut = storeFreshLoc sigma →
      Step ⟨sigma, Term.add (Term.loc ell1) (Term.loc ell2)⟩
           ⟨storeExtend (storeRemove (storeRemove sigma ell1) ell2)
                        ellOut (tensorOpPlaceholder w1 w2),
            Term.loc ellOut⟩

  -- E-Mul: like E-Add.
  | tmul
      (sigma : Store) (ell1 ell2 ellOut : Loc) (w1 w2 : TensorVal) :
      storeLookup sigma ell1 = some w1 →
      storeLookup sigma ell2 = some w2 →
      ellOut = storeFreshLoc sigma →
      Step ⟨sigma, Term.mul (Term.loc ell1) (Term.loc ell2)⟩
           ⟨storeExtend (storeRemove (storeRemove sigma ell1) ell2)
                        ellOut (tensorOpPlaceholder w1 w2),
            Term.loc ellOut⟩

  -- E-Sum: sum(ℓ, d)  ↦  fresh ℓ' with the reduced tensor; operand freed
  | tsum
      (sigma : Store) (ell ellOut : Loc) (w : TensorVal) (d : Dim) :
      storeLookup sigma ell = some w →
      ellOut = storeFreshLoc sigma →
      Step ⟨sigma, Term.sum (Term.loc ell) d⟩
           ⟨storeExtend (storeRemove sigma ell) ellOut
                        { shape := rem w.shape d, data := w.data },
            Term.loc ellOut⟩

  -- E-Expand: expand(ℓ, d)  ↦  fresh ℓ' with the dim-inserted tensor
  | texpand
      (sigma : Store) (ell ellOut : Loc) (w : TensorVal) (d : Dim) :
      storeLookup sigma ell = some w →
      ellOut = storeFreshLoc sigma →
      Step ⟨sigma, Term.expand (Term.loc ell) d⟩
           ⟨storeExtend (storeRemove sigma ell) ellOut
                        { shape := ins w.shape d, data := w.data },
            Term.loc ellOut⟩

  -- E-UniformLike: uniform_like(ℓ, lo, hi)  ↦  fresh ℓ' with random data
  -- **Round-2 fix**: the template location ℓ IS consumed, matching
  -- T-UniformLike's implicit consumption of its template.
  | tuniformLike
      (sigma : Store) (ell ellOut : Loc) (w : TensorVal) (lo hi : Float) :
      storeLookup sigma ell = some w →
      ellOut = storeFreshLoc sigma →
      Step ⟨sigma, Term.uniformLike (Term.loc ell) lo hi⟩
           ⟨storeExtend (storeRemove sigma ell) ellOut
                        { shape := w.shape, data := lo },
            Term.loc ellOut⟩

  /- ## Effect handlers -/

  -- E-Handle-Ret: handle[ε_h] v with h  ↦  v
  | handleRet
      (sigma : Store) (epsH : EffectRow) (v : Term)
      (clauses : List (EffectLabel × String × String × Term)) :
      IsValue v →
      Step ⟨sigma, Term.handle epsH v clauses⟩ ⟨sigma, v⟩

  -- E-Handle-Op (multi-clause direct form, Wave 0 P5 + W0 audit fix).
  -- When the body is a top-level `perform op v`, the reduction looks
  -- up the matching clause `(op, x, k, handlerBody)` in the clause
  -- list and substitutes `v` for `x` and an identity continuation
  -- `λy:tRet. y` for `k`. Multiple clauses are supported: the
  -- membership premise picks whichever clause's op matches.
  --
  -- `handleOpDirect` is the `E = hole` special case of the full
  -- E-Handle-Op rule (see `handleOpCtx` below). It is kept as a
  -- standalone constructor because it uses the identity continuation
  -- `λy:tRet. y` rather than the general reified context, and
  -- Preservation's hand-closed cases refer to it by name.
  -- Wave 1 P1 update: the Phase 1 "top-of-body only" limitation that
  -- this rule used to document is now lifted by the companion
  -- `handleOpCtx` constructor, which fires when `perform` sits inside
  -- a non-trivial evaluation context.
  -- Wave 0 W0-audit M-W0-3 fix: the rule now matches any clause list
  -- in which the matching clause appears, not just a singleton list.
  | handleOpDirect
      (sigma : Store) (op : EffectLabel) (v : Term)
      (epsH : EffectRow)
      (clauses : List (EffectLabel × String × String × Term))
      (x k : String) (handlerBody : Term) (tRet : Typ) :
      IsValue v →
      (∃ tArg, OpSigMatch op tArg tRet) →
      (op, x, k, handlerBody) ∈ clauses →
      Step ⟨sigma,
            Term.handle epsH (Term.perform op v) clauses⟩
           ⟨sigma,
            subst (subst handlerBody v x)
                  (Term.abs "y" tRet (Term.var "y")) k⟩

  -- E-Handle-Op (paper-accurate, captured-context form). When the
  -- handle body has the shape `plug E (perform op v)` with `E` an
  -- evaluation context that does NOT itself catch `op`, the step
  -- picks the matching clause `(op, xVar, kVar, hb)` from the clause
  -- list and reduces to
  --   `hb[v / xVar][λ kFresh : tRet. handle epsH (plug E (var kFresh))
  --                    clauses / kVar]`.
  -- The reified continuation re-enters the same handle around the
  -- captured context `E`, which is the standard delimited-control
  -- semantics for algebraic effects. `tRet` is the return type of
  -- the operation's signature (threaded in as a parameter, matching
  -- `handleOpDirect`'s convention). The continuation binder is chosen
  -- by `capturedContName` so it is fresh for the entire handled term
  -- by construction.
  | handleOpCtx
      (sigma : Store) (op : EffectLabel) (v : Term)
      (epsH : EffectRow) (E : EvalCtx)
      (clauses : List (EffectLabel × String × String × Term))
      (xVar kVar : String) (hb : Term) (tRet : Typ) :
      IsValue v →
      (∃ tArg, OpSigMatch op tArg tRet) →
      (op, xVar, kVar, hb) ∈ clauses →
      op ∈ epsH →
      EvalCtx.noHandleFor op E →
      Step ⟨sigma,
            Term.handle epsH (plug E (Term.perform op v)) clauses⟩
           ⟨sigma,
            subst (subst hb v xVar)
                  (Term.abs (capturedContName
                    (Term.handle epsH (plug E (Term.perform op v)) clauses)) tRet
                    (Term.handle epsH
                      (plug E (Term.var (capturedContName
                        (Term.handle epsH (plug E (Term.perform op v)) clauses))))
                      clauses)) kVar⟩

  -- E-Handle-Op (multi-frame captured-context form, Wave 1 P2).
  --
  -- Generalizes `handleOpCtx` from a one-frame `EvalCtx` to a chain
  -- of frames `Es : EvalCtxChain`, so `perform op v` can sit at
  -- arbitrary depth under the handle body. The `Es.noHandleFor op`
  -- premise demands that no intervening frame is itself a `handle`
  -- catching `op`, which is required for the reduction to be the
  -- innermost matching handler.
  --
  -- Counterexample this rule unblocks (informal trace):
  --   handle [op]
  --     (letBind x (letBind y (perform op unit) unit) unit)
  --     [(op, p, k, var p)]
  --
  -- Take `Es = [EvalCtx.letBind x unit, EvalCtx.letBind y unit]`,
  -- `v = unit`. Then
  --   multiPlug Es (perform op unit)
  --     = plug (letBind x unit) (plug (letBind y unit) (perform op unit))
  --     = plug (letBind x unit) (letBind y (perform op unit) unit)
  --     = letBind x (letBind y (perform op unit) unit) unit
  -- which matches the redex shape, so the handler clause body `var p`
  -- is reached with `p ↦ unit`. Single-frame `handleOpCtx` cannot
  -- match this term because its `E` would have to contain another
  -- compound `letBind` inside the hole.
  | handleOpCtxs
      (sigma : Store) (op : EffectLabel) (v : Term)
      (epsH : EffectRow) (Es : EvalCtxChain)
      (clauses : List (EffectLabel × String × String × Term))
      (xVar kVar : String) (hb : Term) (tRet : Typ) :
      IsValue v →
      (∃ tArg, OpSigMatch op tArg tRet) →
      (op, xVar, kVar, hb) ∈ clauses →
      op ∈ epsH →
      EvalCtxChain.noHandleFor op Es →
      Step ⟨sigma,
            Term.handle epsH (multiPlug Es (Term.perform op v)) clauses⟩
           ⟨sigma,
            subst (subst hb v xVar)
                  (Term.abs (capturedContName
                    (Term.handle epsH (multiPlug Es (Term.perform op v)) clauses)) tRet
                    (Term.handle epsH
                      (multiPlug Es (Term.var (capturedContName
                        (Term.handle epsH (multiPlug Es (Term.perform op v)) clauses))))
                      clauses)) kVar⟩

  /- ## AD and vectorization transforms -/

  -- E-Grad: grad(λx:t.e)  ↦  λx:t. λgs:tOut. handle[{Accum}]
  --                              (adjoint(e, x, gs))
  --                              with {accum(p, k) → k(p)}
  --
  -- The constructor takes `tOut` as an extra parameter so preservation can
  -- pick the output type of `e` from the typing derivation. The paper's
  -- E-Grad reduction is a typed operational rule — `tOut` is implicit in
  -- the typing context — but in Lean we need the type concretely or
  -- preservation for this rule cannot close.
  -- Round-3 fix (H3): the seed parameter type is `tOut`, read from
  -- the `Term.grad` constructor's new `tOut` field (Phase 2 Wave 0 P4).
  -- The handler clause body `app (var k) (var p)` is still a round-3
  -- skeleton placeholder for the real `update_origin_buffer(p)` routing.
  | tgrad
      (sigma : Store) (x : String) (t tOut : Typ) (e : Term) :
      Step ⟨sigma, Term.grad x t tOut e⟩
           ⟨sigma,
            Term.abs x t
              (Term.abs "gs" tOut
                (Term.handle [EffectLabel.accum]
                  (adjoint e x (Term.var "gs"))
                  [(EffectLabel.accum, "p", "k",
                    Term.app (Term.var "k") (Term.var "p"))]))⟩

  -- E-Vmap: vmap(λx:t.e)  ↦  λx:addDim(d, t). addDimTerm(d, e)
  -- Wave 0 P6: body is now lifted via `addDimTerm` (defined in
  -- `Syntax.lean`) rather than passing `e` through unchanged.
  | tvmap
      (sigma : Store) (x : String) (t : Typ) (e : Term) (d : Dim) :
      Step ⟨sigma, Term.vmap x t e⟩
           ⟨sigma, Term.abs x (addDim d t) (addDimTerm d e)⟩

  -- E-Ctx: congruence closure via evaluation contexts (Wave 0.5).
  -- If `e` steps to `e'`, then `plug E e` steps to `plug E e'` for
  -- any evaluation context `E`. This collapses what would otherwise
  -- be one congruence constructor per (term-former × position) pair
  -- into a single parameterized rule.
  | ctx
      (sigma sigma' : Store) (E : EvalCtx) (e e' : Term) :
      Step ⟨sigma, e⟩ ⟨sigma', e'⟩ →
      Step ⟨sigma, plug E e⟩ ⟨sigma', plug E e'⟩

end LaCaDiLE
