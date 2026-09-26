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

Instantiation: `Runtime/ExtractedScopes.lean` takes `comb := fun body s => <crate>.<path>.with_scope
s body` (adapting argument order / the backward-function pair convention through
`ScopeSpecOn` below) with a `Lens` onto the relevant field, and proves the obligations against
the extracted definitions: the suppression and install/restore obligations for the lowering
runtime and the checker's diagnostic sink; `Runtime/ExtractedHostScopes.lean` those of the host
scopes (`with_host_invocation`, `with_mono_probe`, `with_type_subst`, and `with_inlining` in the
restricted form `InliningObligationUnder`, with a counterexample to the unrestricted one).

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

/-- `ScopeSpec` restricted to entry states satisfying an invariant `Inv` and to bodies that
re-establish it on every successful exit. For a representation invariant the model's
collections need but cannot carry in their types (a sorted entry list, strings that fit in
memory), this is the honest form of a scope specification: `ScopeSpec.under` shows it is
implied by the unrestricted one, and an extracted scope that meets only this form must come with
a counterexample to the unrestricted one. -/
def ScopeSpecUnder {M : Type → Type} {σ γ : Type} (ok : M (σ × γ) → Option (σ × γ))
    (Inv : σ → Prop) (enter : σ → σ) (exit : σ → σ → σ) (comb : Combinator M σ γ) : Prop :=
  ∀ body s, Inv s → (∀ s₁ r, ok (body (enter s)) = some (s₁, r) → Inv s₁) →
    ok (comb body s) = (ok (body (enter s))).map (fun x => (exit s x.1, x.2))

theorem ScopeSpec.under {M : Type → Type} {σ γ : Type} {ok : M (σ × γ) → Option (σ × γ)}
    {enter : σ → σ} {exit : σ → σ → σ} {comb : Combinator M σ γ} (Inv : σ → Prop)
    (h : ScopeSpec ok enter exit comb) : ScopeSpecUnder ok Inv enter exit comb :=
  fun body s _ _ => h body s

/-- `InliningObligation` on the states satisfying `Inv`, for bodies that re-establish it. -/
def InliningObligationUnder {M : Type → Type} {σ S α γ : Type} (ok : M (σ × γ) → Option (σ × γ))
    (Inv : σ → Prop) (L : Lens σ S) (mem : α → S → Bool) (ins del : α → S → S) (name : α)
    (comb : Combinator M σ γ) : Prop :=
  ScopeSpecUnder ok Inv (fun s => L.set (ins name (L.get s)) s)
    (fun s s₁ => if mem name (L.get s) then s₁ else L.set (del name (L.get s₁)) s₁) comb

/-- `TensorHelperPreflightGuard`: push this body's facts, pop at exit. -/
def PreflightObligation {M : Type → Type} {σ F γ : Type} (ok : M (σ × γ) → Option (σ × γ)) (L : Lens σ (List F))
    (facts : F) (comb : Combinator M σ γ) : Prop :=
  ScopeSpec ok (fun s => L.set (facts :: L.get s) s) (fun _ s₁ => L.set (L.get s₁).tail s₁) comb

/-- `PreflightObligation` with the frame computed at entry, on bodies that leave a frame. An
entry step `pre` gives the state the scope pushes on and the frame it pushes (`none`: the entry
failed, and then so does the scope); the exit pops the top frame of the body's final state,
which the body must leave (`L.get s₁ ≠ []`). This is the target statement for the production
`with_preflight` (not extracted yet, E30; `Runtime/ExtractedHostScopes.lean`): in the Rust
source its entry allocates the frame id from the runtime's supply and fills the program's
summary cache, so it changes more than the stack, and its pop panics when the body leaves no
frame, so the literal obligation, the special case of a constant entry step
(`PreflightObligation.from_const`), is too strong for it. -/
def PreflightObligationFrom {M : Type → Type} {σ F γ : Type} (ok : M (σ × γ) → Option (σ × γ))
    (L : Lens σ (List F)) (pre : σ → Option (σ × F)) (comb : Combinator M σ γ) : Prop :=
  ∀ body s s₀ fr, pre s = some (s₀, fr) →
    (∀ s₁ r, ok (body (L.set (fr :: L.get s₀) s₀)) = some (s₁, r) → L.get s₁ ≠ []) →
    ok (comb body s) =
      (ok (body (L.set (fr :: L.get s₀) s₀))).map (fun x => (L.set (L.get x.1).tail x.1, x.2))

/-- The literal `PreflightObligation` is `PreflightObligationFrom` with the constant entry step
that pushes `facts` on the unchanged state. -/
theorem PreflightObligation.from_const {M : Type → Type} {σ F γ : Type}
    {ok : M (σ × γ) → Option (σ × γ)} {L : Lens σ (List F)} {facts : F} {comb : Combinator M σ γ}
    (h : PreflightObligation ok L facts comb) :
    PreflightObligationFrom ok L (fun s => some (s, facts)) comb := by
  intro body s s₀ fr hpre _
  simp only [Option.some.injEq, Prod.mk.injEq] at hpre
  obtain ⟨rfl, rfl⟩ := hpre
  exact h body s

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

/-! ## Extracted interfaces

An extracted combinator does not take a Lean state transformer: it takes a closure value
`b : B` together with its `FnOnce` instance, its successful result may carry more than
`(state, result)` (the backward functions of nested borrows), and a catch maps the body's
result (`catch_lowering_with` maps a lowering failure to its diagnostic). `ScopeSpecOn`
states the specification over such an interface: `run b s` is the successful-exit view of the
body `b` run from `s`, `comb b s` the one of the combinator, and `post` the result map (`id`
for every scope but a catch). `ScopeSpec` is its instance for state-transformer bodies
(`scopeSpec_iff_on`), so an extracted combinator meeting `ScopeSpecOn` for every closure
instance meets the corresponding obligation above for every body. -/

/-- Scope specification over an extracted interface (see the section header). -/
def ScopeSpecOn {B σ γ δ : Type} (run : B → σ → Option (σ × γ)) (comb : B → σ → Option (σ × δ))
    (enter : σ → σ) (exit : σ → σ → σ) (post : γ → δ) : Prop :=
  ∀ b s, comb b s = (run b (enter s)).map (fun x => (exit s x.1, post x.2))

theorem scopeSpec_iff_on {M : Type → Type} {σ γ : Type} (ok : M (σ × γ) → Option (σ × γ))
    (enter : σ → σ) (exit : σ → σ → σ) (comb : Combinator M σ γ) :
    ScopeSpec ok enter exit comb ↔
      ScopeSpecOn (fun body s => ok (body s)) (fun body s => ok (comb body s)) enter exit id :=
  Iff.rfl

/-- A restore specification over an extracted interface restores on every successful exit. -/
theorem restores_of_specOn {B σ β γ δ : Type} {run : B → σ → Option (σ × γ)}
    {comb : B → σ → Option (σ × δ)} {L : Lens σ β} (hL : L.Lawful) {enter : σ → σ}
    {post : γ → δ} (h : ScopeSpecOn run comb enter (restoreExit L) post) :
    ∀ b s s' r, comb b s = some (s', r) → L.get s' = L.get s := by
  intro b s s' r hc
  rw [h b s] at hc
  cases hb : run b (enter s) with
  | none => rw [hb] at hc; cases hc
  | some x =>
    rw [hb] at hc
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at hc
    rw [← hc.1]
    exact hL.get_set _ _

/-- A clear specification over an extracted interface clears on every successful exit. -/
theorem clears_of_specOn {B σ γ δ : Type} {run : B → σ → Option (σ × γ)}
    {comb : B → σ → Option (σ × δ)} {L : Lens σ Bool} (hL : L.Lawful) {enter : σ → σ}
    {post : γ → δ} (h : ScopeSpecOn run comb enter (clearExit L) post) :
    ∀ b s s' r, comb b s = some (s', r) → L.get s' = false := by
  intro b s s' r hc
  rw [h b s] at hc
  cases hb : run b (enter s) with
  | none => rw [hb] at hc; cases hc
  | some x =>
    rw [hb] at hc
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at hc
    rw [← hc.1]
    exact hL.get_set _ _

/-- Under an install specification the body ran on the installed value, and the combinator
returned the body's (mapped) result with the exit of the body's final state. -/
theorem installed_of_specOn {B σ β γ δ : Type} {run : B → σ → Option (σ × γ)}
    {comb : B → σ → Option (σ × δ)} (L : Lens σ β) (v : β) {exit : σ → σ → σ}
    {post : γ → δ} (h : ScopeSpecOn run comb (L.set v) exit post) :
    ∀ b s s' r, comb b s = some (s', r) →
      ∃ s₁ r₁, run b (L.set v s) = some (s₁, r₁) ∧ s' = exit s s₁ ∧ r = post r₁ := by
  intro b s s' r hc
  rw [h b s] at hc
  cases hb : run b (L.set v s) with
  | none => rw [hb] at hc; cases hc
  | some x =>
    rw [hb] at hc
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at hc
    exact ⟨x.1, x.2, rfl, hc.1.symm, hc.2.symm⟩

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
