import ChelisBridge.Runtime.Unwind

/-!
# E. Scope combinators on the extracted core: statement shapes

After the FCIS refactor the scope guards of the semantic core are ordinary explicit-state
functions that Aeneas extracts (mono probe snapshot/rollback, host invocation restore, inlining
scope, tensor-helper preflight scope, type-substitution scope, opacity scope, recursive inference
scope, suppression scope clear-to-false). This module fixes, BEFORE those definitions exist, the
exact properties each extracted combinator must satisfy, parameterized over an abstract
state-transformer interface:

* `M` is the extracted monad (Aeneas `Result`), observed only through a successful-exit view `ok : M (σ × γ) → Option (σ × γ)` (for Aeneas `fun | .ok a => some a | _ => none`) (`some` exactly on
  `ok`; `fail`/`div` are not successful exits);
* `σ` is the explicit state the combinator threads (`&mut HostInvocationState`, `&mut
  DiagnosticSink`, `&mut LoweringRuntime`, …), `γ` the body's return value, which in the
  extracted code is the Rust `Result<R, E>` itself — so quantifying over every `γ` value covers
  both the `Ok` and the `Err` result;
* a body is any state transformer `σ → M (σ × γ)` (the extracted closure's `call_once` applied to
  its captured trait instance), and every obligation quantifies over all bodies.

Instantiation (once the extracted names exist, after `open <root>`): take `comb := fun body s =>
<crate>.<path>.with_scope s body` (adapting argument order / the backward-function pair
convention) and a `Lens` onto the relevant field, then prove e.g.
`MonoProbeObligation okView monoLens comb` against the extracted definition.

`scope_obligations_are_satisfiable` shows the obligations are not vacuous, `restore_ne_clear`
records why restore and clear-to-false must stay separate obligations, and the `model_*`
theorems check that the guards of the `Unwind` model meet them on both `Ok` and `Err`.
-/

namespace ChelisBridge.Runtime.Scopes

/-- A lens onto one component of the extracted state. -/
structure Lens (σ β : Type) where
  get : σ → β
  set : β → σ → σ

/-- Lens laws (satisfied by every record-field lens). -/
structure Lens.Lawful {σ β : Type} (L : Lens σ β) : Prop where
  get_set : ∀ b s, L.get (L.set b s) = b
  set_get : ∀ s, L.set (L.get s) s = s
  set_set : ∀ b c s, L.set c (L.set b s) = L.set c s

/-- The shape of an extracted scope combinator. -/
abbrev Combinator (M : Type → Type) (σ γ : Type) := (σ → M (σ × γ)) → σ → M (σ × γ)

/-- Full functional scope specification: on every successful exit the combinator ran the body
exactly on the entered state, returns the body's result unchanged, and its final state is the
required exit transformation of (entry state, body's final state); it succeeds iff the body
does. -/
def ScopeSpec {M : Type → Type} {σ γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (enter : σ → σ)
    (exit : σ → σ → σ) (comb : Combinator M σ γ) : Prop :=
  ∀ body s, ok (comb body s) = (ok (body (enter s))).map (fun x => (exit s x.1, x.2))

/-- Restore-previous exit on the lens component: other components keep the body's effects. -/
def restoreExit {σ β : Type} (L : Lens σ β) : σ → σ → σ := fun s s₁ => L.set (L.get s) s₁

/-- Clear-to-false exit (NOT restore). -/
def clearExit {σ : Type} (L : Lens σ Bool) : σ → σ → σ := fun _ s₁ => L.set false s₁

/-! ## The E properties: state after exit on both `Ok` and `Err` -/

/-- After every successful exit (any body, any result — `Ok` or `Err`) the component equals its
value at entry. -/
def RestoresOnExit {M : Type → Type} {σ β γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (L : Lens σ β)
    (comb : Combinator M σ γ) : Prop :=
  ∀ body s s' r, ok (comb body s) = some (s', r) → L.get s' = L.get s

/-- After every successful exit (any body, any result) the flag is `false`, whatever its value
at entry. -/
def ClearsOnExit {M : Type → Type} {σ γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (L : Lens σ Bool)
    (comb : Combinator M σ γ) : Prop :=
  ∀ body s s' r, ok (comb body s) = some (s', r) → L.get s' = false

/-- The body observes the component set to `v` (fresh/empty frame, replacing substitution). -/
def BodySeesInstalled {M : Type → Type} {σ β γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (L : Lens σ β) (v : β)
    (comb : Combinator M σ γ) : Prop :=
  ∀ body s s' r, ok (comb body s) = some (s', r) →
    ∃ s₁, ok (body (L.set v s)) = some (s₁, r)

/-! ## Per-combinator obligations -/

/-- `MonoProbeGuard`: snapshot the complete specialization state at entry, roll back at exit. -/
def MonoProbeObligation {M : Type → Type} {σ β γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (mono : Lens σ β)
    (comb : Combinator M σ γ) : Prop :=
  ScopeSpec ok id (restoreExit mono) comb

/-- `HostInvocationGuard`: start the whole-program call with the empty invocation state and
restore the caller's state at exit. -/
def HostInvocationObligation {M : Type → Type} {σ β γ : Type} (ok : M (σ × γ) → Option (σ × γ))
    (inv : Lens σ β) (empty : β) (comb : Combinator M σ γ) : Prop :=
  ScopeSpec ok (inv.set empty) (restoreExit inv) comb

/-- Install-then-restore scopes: `ActiveTypeSubstGuard` (nested bindings replace the outer map),
`OpacityScope` (policy), `RecursiveInferenceScope` (fresh frame). -/
def InstallRestoreObligation {M : Type → Type} {σ β γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (L : Lens σ β)
    (v : β) (comb : Combinator M σ γ) : Prop :=
  ScopeSpec ok (L.set v) (restoreExit L) comb

/-- `InliningGuard`: insert `name`; at exit remove it only if this scope inserted it
("a repeated name belongs to the outer scope; only its owner removes it"). -/
def InliningObligation {M : Type → Type} {σ S α γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (L : Lens σ S)
    (mem : α → S → Bool) (ins del : α → S → S) (name : α) (comb : Combinator M σ γ) : Prop :=
  ScopeSpec ok (fun s => L.set (ins name (L.get s)) s)
    (fun s s₁ => if mem name (L.get s) then s₁ else L.set (del name (L.get s₁)) s₁) comb

/-- `TensorHelperPreflightGuard`: push this body's facts, pop at exit. -/
def PreflightObligation {M : Type → Type} {σ F γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (L : Lens σ (List F))
    (facts : F) (comb : Combinator M σ γ) : Prop :=
  ScopeSpec ok (fun s => L.set (facts :: L.get s) s) (fun _ s₁ => L.set (L.get s₁).tail s₁) comb

/-- Suppression scopes (`with_suppress_unrepresentable_panic`, `catch_lowering`'s output guard):
set at entry, CLEAR to `false` at exit. -/
def SuppressionObligation {M : Type → Type} {σ γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (flag : Lens σ Bool)
    (comb : Combinator M σ γ) : Prop :=
  ScopeSpec ok (flag.set true) (clearExit flag) comb

/-! ## The obligations imply the E properties -/

/-- A restore specification yields restoration on every successful exit (Ok and Err). -/
theorem restores_of_spec {M : Type → Type} {σ β γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (L : Lens σ β)
    (hL : L.Lawful) (enter : σ → σ) (comb : Combinator M σ γ)
    (h : ScopeSpec ok enter (restoreExit L) comb) : RestoresOnExit ok L comb := by
  intro body s s' r hc
  rw [h body s] at hc
  cases hb : ok (body (enter s)) with
  | none => rw [hb] at hc; cases hc
  | some x =>
    rw [hb] at hc
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at hc
    rw [← hc.1]
    exact hL.get_set _ _

/-- A clear specification yields `false` on every successful exit (Ok and Err). -/
theorem clears_of_spec {M : Type → Type} {σ γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (L : Lens σ Bool)
    (hL : L.Lawful) (enter : σ → σ) (comb : Combinator M σ γ)
    (h : ScopeSpec ok enter (clearExit L) comb) : ClearsOnExit ok L comb := by
  intro body s s' r hc
  rw [h body s] at hc
  cases hb : ok (body (enter s)) with
  | none => rw [hb] at hc; cases hc
  | some x =>
    rw [hb] at hc
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at hc
    rw [← hc.1]
    exact hL.get_set _ _

/-- An install specification makes the body run on the installed value. -/
theorem installed_of_spec {M : Type → Type} {σ β γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (L : Lens σ β) (v : β)
    (comb : Combinator M σ γ) (h : InstallRestoreObligation ok L v comb) :
    BodySeesInstalled ok L v comb := by
  intro body s s' r hc
  rw [h body s] at hc
  cases hb : ok (body (L.set v s)) with
  | none => rw [hb] at hc; cases hc
  | some x =>
    rw [hb] at hc
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at hc
    exact ⟨x.1, by rw [← hc.2]⟩

/-- The mono probe rolls back on every successful exit. -/
theorem monoProbe_restores {M : Type → Type} {σ β γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (mono : Lens σ β)
    (hL : mono.Lawful) (comb : Combinator M σ γ) (h : MonoProbeObligation ok mono comb) :
    RestoresOnExit ok mono comb :=
  restores_of_spec ok mono hL id comb h

/-- The host invocation scope restores the caller's state on every successful exit. -/
theorem hostInvocation_restores {M : Type → Type} {σ β γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (inv : Lens σ β)
    (hL : inv.Lawful) (empty : β) (comb : Combinator M σ γ)
    (h : HostInvocationObligation ok inv empty comb) : RestoresOnExit ok inv comb :=
  restores_of_spec ok inv hL _ comb h

/-- A suppression scope clears on every successful exit. -/
theorem suppression_clears {M : Type → Type} {σ γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (flag : Lens σ Bool)
    (hL : flag.Lawful) (comb : Combinator M σ γ) (h : SuppressionObligation ok flag comb) :
    ClearsOnExit ok flag comb :=
  clears_of_spec ok flag hL _ comb h

/-! ## Non-vacuity and the restore/clear distinction -/

/-- The reference implementation of a scope in the `Option` monad (successful = `some`). -/
def refComb {σ γ : Type} (enter : σ → σ) (exit : σ → σ → σ) : Combinator Option σ γ :=
  fun body s => (body (enter s)).map (fun x => (exit s x.1, x.2))

/-- Every obligation shape is satisfiable (by the reference implementation). -/
theorem scope_obligations_are_satisfiable {σ γ : Type} (enter : σ → σ) (exit : σ → σ → σ) :
    ScopeSpec (M := Option) (fun x => x) enter exit (refComb (γ := γ) enter exit) :=
  fun _ _ => rfl

/-- The identity lens on `Bool`. -/
def boolLens : Lens Bool Bool := { get := id, set := fun b _ => b }

/-- Restore-previous and clear-to-false are different obligations: inside an enclosing scope
whose flag is `true`, a restoring scope leaves `true`, a clearing one leaves `false`
(ARCHITECTURE.md: "A uniform restore-previous model would be wrong"). -/
theorem restore_ne_clear :
    let body : Bool → Option (Bool × Unit) := fun b => some (b, ())
    refComb (boolLens.set true) (restoreExit boolLens) body true = some (true, ()) ∧
    refComb (boolLens.set true) (clearExit boolLens) body true = some (false, ()) ∧
    ¬ ClearsOnExit (M := Option) (fun x => x) boolLens
        (refComb (γ := Unit) (boolLens.set true) (restoreExit boolLens)) := by
  refine ⟨rfl, rfl, ?_⟩
  intro h
  have := h (fun b => some (b, ())) true true () rfl
  cases this

/-! ## The `Unwind` model's guards meet the obligations on `Ok` and `Err` -/

open ChelisBridge.Runtime.Unwind in
/-- In the new semantics the speculative suppression scope clears `suppress_unrepresentable` on
both `Ok` and `Err` exits, whatever the body. -/
theorem model_speculate_clears (t : ThreadId) (fid : Nat) (body : Prog) (s : State)
    (h : ∀ p, (new t (.scope .speculate fid none body) s).2.2 ≠ .unwind p)
    (ha : (new t (.scope .speculate fid none body) s).2.2 ≠ .abort) :
    ((new t (.scope .speculate fid none body) s).1.flags t).unrep = false := by
  rcases hN : new t body (Guard.enter t .speculate s) with ⟨s1, tr1, r⟩
  simp only [new, hN] at h ha ⊢
  cases r with
  | ok v => simp [newScopeExit, Guard.exit, State.setUnrep, State.modFlags]
  | err f => simp [newScopeExit, Guard.exit, State.setUnrep, State.modFlags]
  | unwind p => simp only [newScopeExit] at h; exact absurd rfl (h p)
  | abort => simp only [newScopeExit] at ha; exact absurd rfl ha

open ChelisBridge.Runtime.Unwind in
/-- In the new semantics a restore guard gives back the entry value of its cell on both `Ok`
and `Err` exits, whatever the body. -/
theorem model_restore_restores (t : ThreadId) (fid k : Nat) (v : Int) (body : Prog) (s : State)
    (h : ∀ p, (new t (.scope (.restore k v) fid none body) s).2.2 ≠ .unwind p)
    (ha : (new t (.scope (.restore k v) fid none body) s).2.2 ≠ .abort) :
    (new t (.scope (.restore k v) fid none body) s).1.cells k = s.cells k := by
  rcases hN : new t body (Guard.enter t (.restore k v) s) with ⟨s1, tr1, r⟩
  simp only [new, hN] at h ha ⊢
  cases r with
  | ok v => simp [newScopeExit, Guard.exit, State.setCell]
  | err f => simp [newScopeExit, Guard.exit, State.setCell]
  | unwind p => simp only [newScopeExit] at h; exact absurd rfl (h p)
  | abort => simp only [newScopeExit] at ha; exact absurd rfl ha

open ChelisBridge.Runtime.Unwind in
/-- In the new semantics a rollback guard restores all cells on both `Ok` and `Err` exits. -/
theorem model_rollback_restores (t : ThreadId) (fid : Nat) (body : Prog) (s : State)
    (h : ∀ p, (new t (.scope .rollback fid none body) s).2.2 ≠ .unwind p)
    (ha : (new t (.scope .rollback fid none body) s).2.2 ≠ .abort) :
    (new t (.scope .rollback fid none body) s).1.cells = s.cells := by
  rcases hN : new t body (Guard.enter t .rollback s) with ⟨s1, tr1, r⟩
  simp only [new, hN] at h ha ⊢
  cases r with
  | ok v => simp [newScopeExit, Guard.exit]
  | err f => simp [newScopeExit, Guard.exit]
  | unwind p => simp only [newScopeExit] at h; exact absurd rfl (h p)
  | abort => simp only [newScopeExit] at ha; exact absurd rfl ha

end ChelisBridge.Runtime.Scopes
