import ChelisBridge.Runtime.ExtractedScopes
import ChelisBridge.Runtime.CloneId
import ChelisBridge.StrOrder

/-!
# E. The host lowering scopes on the extracted production combinators

The host lowering scopes (`chelis_ir::host`) change the invocation state of a
`HostCx` (`program.state.invocation`) through the native shell `scoped` (hard-opaque, modeled by
`Aeneas/Std/ChelisNative.lean`: `enter; body; exit`, a failing body propagates). Their `HostCx`
state holds mutable borrows, so the extracted instance threads the backward function of those
borrows beside the state; the successful-exit view `okHost` keeps the state and the result.

* `with_host_invocation` starts the body on the EMPTY invocation state (the value the extracted
  `HostInvocationState::default` returns) and RESTORES the caller's invocation state at exit
  (`withHostInvocation_spec`, `withHostInvocation_obligation`: the literal
  `HostInvocationObligation`);
* `with_mono_probe` runs the body on the unchanged state and ROLLS BACK the specialization
  state to its entry snapshot at exit (`withMonoProbe_spec`, `withMonoProbe_obligation`: the
  literal `MonoProbeObligation`); the snapshot is `MonoSpecializationState::clone`, the identity
  (`CloneId.cid_host_MonoSpecializationState`);
* `with_type_subst(subst)` runs the body with `subst` as the active type substitution,
  replacing (not extending) the caller's, and RESTORES the caller's substitution at exit
  (`withTypeSubst_spec`, `withTypeSubst_obligation`: the literal `InstallRestoreObligation` of
  `ActiveTypeSubstGuard`);
* `with_inlining(name)` inserts `name` into the inlining set at entry and, at exit, removes it
  only if this scope inserted it (`withInlining_spec`, `withInlining_obligation`:
  `InliningObligationUnder`); without a name it is the body itself (`withInlining_none`);
* `with_preflight(expr)` takes a frame id from the runtime's supply, fills the program's summary
  cache, analyzes `expr` against it and pushes the frame `{id, facts}` on the tensor-helper
  preflight stack; the body receives the frame root `{frame := id, node := 0}`, and the exit pops
  the top frame of the body's final context, failing (the Rust `expect` panic) when the body left
  none (`withPreflight_spec`, `withPreflight_obligation`: `PreflightObligationFrom`).

`with_inlining` meets `InliningObligation` only on states whose inlining set satisfies the
model's representation invariant (`InliningInv`: String-ordered, the `BTreeMap.Ordered`
invariant of the collection model, and every key has a `str` view, i.e. at most `usize::MAX`
UTF-8 bytes; every native set satisfies both) for bodies that keep it. The unrestricted
obligation is FALSE in the model, whatever set operations it is stated with:
`withInlining_obligation_false` gives, at every state not holding the name, a body (it stores a
`String` of `usize::MAX + 1` bytes, which no native `String` can be) after which the extracted
exit's `remove` fails while the body succeeded.

Every statement quantifies over every closure type and `FnOnce` instance (every body the
production code can pass), covers `Ok` and `Err` results alike (the result type is arbitrary),
and a failing or diverging body makes the combinator fail too.

`with_preflight` meets `PreflightObligationFrom`, not the literal `PreflightObligation`: its entry
changes more than the stack (the frame-id supply and the summary cache) and computes the frame it
pushes (`preflightEntry`), and its exit fails when the body leaves no frame. The stack is the
`Vec` of the invocation state, whose last element is the top; `preflightLens` views it top first.
The entry fails when the stack already holds `usize::MAX` frames (`Vec::push`).
-/

open Aeneas.Std
open_chelis_model

namespace ChelisBridge.Runtime.ExtractedHostScopes

open ChelisBridge.Runtime.Scopes ChelisBridge.Runtime.ExtractedScopes

open Lean Elab Tactic Meta in
/-- Unfold, in the goal, every definition whose name carries a partial-monomorphization digest
(`…_specialized_<digest>`): the `scoped` instance over `HostCx` and the closure dictionaries.
The digests differ between the combined and whole-core layouts, so bridge modules never spell
these names (`ChelisBridge/Model.lean`). -/
elab "unfold_specialized" : tactic => do
  for _ in [0:16] do
    let goal ← getMainGoal
    let env ← getEnv
    let target ← instantiateMVars (← goal.getType)
    let names := target.getUsedConstants.filter fun n =>
      (n.toString.splitOn "_specialized_").length > 1 &&
        (env.find? n matches some (.defnInfo _))
    if names.isEmpty then return
    replaceMainGoal [← goal.deltaTarget (names.contains ·)]

/-- The host lowering context the extracted scopes thread. -/
abbrev HCx := chelis_ir.host.HostCx

/-- The successful-exit view of an extracted host scope `(result, context, backward function)`. -/
def okHost {γ : Type} : Result (γ × HCx × (HCx → HCx)) → Option (HCx × γ)
  | .ok (r, s, _) => some (s, r)
  | _ => none

/-- The invocation state of the context. -/
def invocationLens : Lens HCx chelis_ir.host.HostInvocationState :=
  { get := (·.state.invocation), set := fun v s => { s with state := { s.state with invocation := v } } }

/-- The specialization state of the context. -/
def monoLens : Lens HCx chelis_ir.host.MonoSpecializationState :=
  { get := (·.state.invocation.mono),
    set := fun v s => { s with state := { s.state with invocation := { s.state.invocation with mono := v } } } }

/-- The inlining set of the context. -/
def inliningLens : Lens HCx (chelis_unord.UnordSet String) :=
  { get := (·.state.invocation.inlining),
    set := fun v s => { s with state := { s.state with invocation := { s.state.invocation with inlining := v } } } }

theorem invocationLens_lawful : invocationLens.Lawful := ⟨fun _ _ => rfl, fun _ => rfl, fun _ _ _ => rfl⟩

theorem monoLens_lawful : monoLens.Lawful := ⟨fun _ _ => rfl, fun _ => rfl, fun _ _ _ => rfl⟩

theorem inliningLens_lawful : inliningLens.Lawful := ⟨fun _ _ => rfl, fun _ => rfl, fun _ _ _ => rfl⟩

/-- The `FnOnce` dictionary of a state-transformer body (`body s = (s', r)`), as the extracted
`(result, context, backward function)` triple: the body the `Scopes.Combinator` views pass. -/
def stCall {γ : Type} (b : HCx → Result (HCx × γ)) (s : HCx) : Result (γ × HCx × (HCx → HCx)) := do
  let (s', r) ← b s
  .ok (r, s', fun _ => s')

theorem okHost_stCall {γ : Type} (b : HCx → Result (HCx × γ)) (s : HCx) :
    okHost (stCall b s) = okResult (b s) := by
  cases h : b s with
  | ok x => obtain ⟨s', r⟩ := x; simp [stCall, h, okHost, okResult]
  | fail e => simp [stCall, h, okHost, okResult]
  | div => simp [stCall, h, okHost, okResult]

/-- The `Scopes.Combinator` view of an extracted host scope `F` (applied to the body and the
context): run it on a state-transformer body and keep the context and the result. -/
def asComb {γ : Type} (F : (HCx → Result (HCx × γ)) → HCx → Result (γ × HCx × (HCx → HCx))) :
    Combinator Result HCx γ := fun body s => do
  let (r, s', _) ← F body s
  .ok (s', r)

theorem okResult_asComb {γ : Type}
    (F : (HCx → Result (HCx × γ)) → HCx → Result (γ × HCx × (HCx → HCx))) body s :
    okResult (asComb F body s) = okHost (F body s) := by
  cases h : F body s with
  | ok x => obtain ⟨r, s', bk⟩ := x; simp [asComb, h, okHost, okResult]
  | fail e => simp [asComb, h, okHost, okResult]
  | div => simp [asComb, h, okHost, okResult]

/-- An extracted scope meeting `ScopeSpecOn` on state-transformer bodies meets `ScopeSpec` as a
`Scopes.Combinator`. -/
theorem scopeSpec_asComb {γ : Type}
    (F : (HCx → Result (HCx × γ)) → HCx → Result (γ × HCx × (HCx → HCx)))
    {enter : HCx → HCx} {exit : HCx → HCx → HCx}
    (h : ScopeSpecOn (fun b s => okHost (stCall b s)) (fun b s => okHost (F b s)) enter exit id) :
    ScopeSpec okResult enter exit (asComb F) := by
  intro body s
  rw [okResult_asComb]
  refine (h body s).trans ?_
  simp only [okHost_stCall, id]

/-! ## `with_host_invocation` -/

/-- The empty specialization state. -/
def emptyMono : chelis_ir.host.MonoSpecializationState :=
  { memo := ⟨⟨[]⟩⟩, symbol_keys := ⟨⟨[]⟩⟩, functions := alloc.vec.Vec.new _,
    in_progress := alloc.vec.Vec.new _ }

/-- The empty invocation state: the value the extracted `HostInvocationState::default` returns. -/
def emptyInvocation : chelis_ir.host.HostInvocationState :=
  { inlining := ⟨⟨⟨[]⟩⟩⟩, preflight := alloc.vec.Vec.new _, type_subst := ⟨⟨[]⟩⟩, mono := emptyMono }

theorem hostInvocationState_default :
    chelis_ir.host.HostInvocationState.Insts.CoreDefaultDefault.default = .ok emptyInvocation := rfl

/-- **`with_host_invocation`, every instance**: the body runs on the empty invocation state and
the caller's invocation state is restored at exit (the body's other effects are kept). -/
theorem withHostInvocation_spec {R T1 : Type} (inst) :
    ScopeSpecOn (fun (b : T1) s => okHost (γ := R) (inst.call_once b s))
      (fun b s => okHost (chelis_ir.host.with_host_invocation inst s b))
      (invocationLens.set emptyInvocation) (restoreExit invocationLens) id := by
  intro b s
  simp only [chelis_ir.host.with_host_invocation]
  unfold_specialized
  simp only [Aeneas.Std.chelis_ir.lowering_runtime.scoped, core.ops.function.FnOnceMutArg.thread,
    core.ops.function.FnOnceMutArgPair.thread, core.mem.take, hostInvocationState_default,
    bind_tc_ok, invocationLens, restoreExit]
  cases h : inst.call_once b { s with state := { s.state with invocation := emptyInvocation } } with
  | ok x => obtain ⟨r, s₁, bk⟩ := x; simp [h, okHost]
  | fail e => simp [h, okHost]
  | div => simp [h, okHost]

/-- The extracted `with_host_invocation` as a `Scopes.Combinator` (state-transformer bodies). -/
def hostInvocationComb (γ : Type) : Combinator Result HCx γ :=
  asComb fun body s =>
    chelis_ir.host.with_host_invocation (R := γ) (T1 := HCx → Result (HCx × γ)) ⟨stCall⟩ s body

/-- **The literal Section E `HostInvocationObligation`** for the extracted
`with_host_invocation`. -/
theorem withHostInvocation_obligation (γ : Type) :
    HostInvocationObligation okResult invocationLens emptyInvocation (hostInvocationComb γ) :=
  scopeSpec_asComb _ (withHostInvocation_spec _)

/-! ## `with_mono_probe` -/

/-- **`with_mono_probe`, every instance**: the body runs on the unchanged state and the
specialization state is rolled back to its entry value at exit. -/
theorem withMonoProbe_spec {R T1 : Type} (inst) :
    ScopeSpecOn (fun (b : T1) s => okHost (γ := R) (inst.call_once b s))
      (fun b s => okHost (chelis_ir.host.with_mono_probe inst s b))
      id (restoreExit monoLens) id := by
  intro b s
  simp only [chelis_ir.host.with_mono_probe]
  unfold_specialized
  simp only [Aeneas.Std.chelis_ir.lowering_runtime.scoped, core.ops.function.FnOnceMutArg.thread,
    core.ops.function.FnOnceMutArgPair.thread, CloneId.cid_host_MonoSpecializationState,
    bind_tc_ok, monoLens, restoreExit, id]
  cases h : inst.call_once b s with
  | ok x => obtain ⟨r, s₁, bk⟩ := x; simp [h, okHost]
  | fail e => simp [h, okHost]
  | div => simp [h, okHost]

/-- The extracted `with_mono_probe` as a `Scopes.Combinator` (state-transformer bodies). -/
def monoProbeComb (γ : Type) : Combinator Result HCx γ :=
  asComb fun body s =>
    chelis_ir.host.with_mono_probe (R := γ) (T1 := HCx → Result (HCx × γ)) ⟨stCall⟩ s body

/-- **The literal Section E `MonoProbeObligation`** for the extracted `with_mono_probe`. -/
theorem withMonoProbe_obligation (γ : Type) :
    MonoProbeObligation okResult monoLens (monoProbeComb γ) :=
  scopeSpec_asComb _ (withMonoProbe_spec _)

/-- The probe rolls back the specialization state on every successful exit. -/
theorem withMonoProbe_restores {R T1 : Type} (inst) (b : T1) (s s' : HCx) (r : R)
    (h : okHost (chelis_ir.host.with_mono_probe inst s b) = some (s', r)) :
    s'.state.invocation.mono = s.state.invocation.mono :=
  restores_of_specOn monoLens_lawful (withMonoProbe_spec inst) b s s' r h

/-! ## `with_type_subst` -/

/-- The active type substitution of the context. -/
def typeSubstLens : Lens HCx (chelis_unord.UnordMap String chelis_ir.host_type_state.HostTypeTerm) :=
  { get := (·.state.invocation.type_subst),
    set := fun v s => { s with state := { s.state with invocation := { s.state.invocation with type_subst := v } } } }

theorem typeSubstLens_lawful : typeSubstLens.Lawful := ⟨fun _ _ => rfl, fun _ => rfl, fun _ _ _ => rfl⟩

/-- **`with_type_subst(subst)`, every instance**: the body runs with `subst` as the active
substitution, replacing (not extending) the caller's, and the caller's substitution is restored
at exit (the body's other effects are kept). -/
theorem withTypeSubst_spec {R T1 : Type} (inst)
    (subst : chelis_unord.UnordMap String chelis_ir.host_type_state.HostTypeTerm) :
    ScopeSpecOn (fun (b : T1) s => okHost (γ := R) (inst.call_once b s))
      (fun b s => okHost (chelis_ir.host.with_type_subst inst subst s b))
      (typeSubstLens.set subst) (restoreExit typeSubstLens) id := by
  intro b s
  simp only [chelis_ir.host.with_type_subst]
  unfold_specialized
  simp only [Aeneas.Std.chelis_ir.lowering_runtime.scoped, core.ops.function.FnOnceMutArg.thread,
    core.ops.function.FnOnceMutArgPair.thread, core.mem.replace, typeSubstLens, restoreExit]
  cases h : inst.call_once b
      { s with state := { s.state with invocation := { s.state.invocation with type_subst := subst } } } with
  | ok x => obtain ⟨r, s₁, bk⟩ := x; simp [h, okHost]
  | fail e => simp [h, okHost]
  | div => simp [h, okHost]

/-- The extracted `with_type_subst` as a `Scopes.Combinator` (state-transformer bodies). -/
def typeSubstComb (subst : chelis_unord.UnordMap String chelis_ir.host_type_state.HostTypeTerm)
    (γ : Type) : Combinator Result HCx γ :=
  asComb fun body s =>
    chelis_ir.host.with_type_subst (R := γ) (T1 := HCx → Result (HCx × γ)) ⟨stCall⟩ subst s body

/-- **The literal Section E `InstallRestoreObligation`** (`ActiveTypeSubstGuard`) for the
extracted `with_type_subst`. -/
theorem withTypeSubst_obligation
    (subst : chelis_unord.UnordMap String chelis_ir.host_type_state.HostTypeTerm) (γ : Type) :
    InstallRestoreObligation okResult typeSubstLens subst (typeSubstComb subst γ) :=
  scopeSpec_asComb _ (withTypeSubst_spec _ _)

/-- The caller's substitution is restored on every successful exit (`Ok` and `Err`). -/
theorem withTypeSubst_restores {R T1 : Type} (inst)
    (subst : chelis_unord.UnordMap String chelis_ir.host_type_state.HostTypeTerm) (b : T1)
    (s s' : HCx) (r : R) (h : okHost (chelis_ir.host.with_type_subst inst subst s b) = some (s', r)) :
    s'.state.invocation.type_subst = s.state.invocation.type_subst :=
  restores_of_specOn typeSubstLens_lawful (withTypeSubst_spec inst subst) b s s' r h

/-! ## `with_inlining` -/

open ChelisBridge.StrOrder

/-- The names of an inlining set. -/
def setKeys (S : chelis_unord.UnordSet String) : List String := keys S.inner.map.entries

/-- Membership in an inlining set. -/
def memS (m : String) (S : chelis_unord.UnordSet String) : Bool := decide (m ∈ setKeys S)

/-- Insertion into an inlining set (in order; a present name leaves it unchanged). -/
def insS (m : String) (S : chelis_unord.UnordSet String) : chelis_unord.UnordSet String :=
  ⟨⟨⟨insList m S.inner.map.entries⟩⟩⟩

/-- Removal of a name from an inlining set. -/
def delS (m : String) (S : chelis_unord.UnordSet String) : chelis_unord.UnordSet String :=
  ⟨⟨⟨delList m S.inner.map.entries⟩⟩⟩

/-- The representation invariant of an inlining set: String-ordered (the collection model's
`BTreeMap.Ordered`, `StrOrder.ordered_iff`) and every name has a `str` view (at most
`usize::MAX` UTF-8 bytes). Every native set satisfies both: a B-tree is ordered and no native
`String` exceeds `isize::MAX` bytes. -/
def SetInv (S : chelis_unord.UnordSet String) : Prop :=
  S.inner.map.entries.Pairwise (fun a b => a.1 < b.1) ∧
    ∀ k ∈ setKeys S, k.toByteArray.size ≤ Usize.max

/-- The representation invariant of the context's inlining set. -/
def InliningInv (s : HCx) : Prop := SetInv (inliningLens.get s)

/-- Entry of an inlining scope for `m`: insert `m`. -/
def inliningEnter (m : String) (s : HCx) : HCx := inliningLens.set (insS m (inliningLens.get s)) s

/-- Exit of an inlining scope for `m`: remove `m` only if the entry inserted it. -/
def inliningExit (m : String) (s s₁ : HCx) : HCx :=
  if memS m (inliningLens.get s) then s₁ else inliningLens.set (delS m (inliningLens.get s₁)) s₁

/-- Insertion keeps the invariant's order. -/
theorem insS_pairwise (m : String) (S : chelis_unord.UnordSet String) (h : SetInv S) :
    (insS m S).inner.map.entries.Pairwise (fun a b => a.1 < b.1) :=
  insList_pairwise m _ h.1

/-- **`with_inlining` without a name** is the body itself. -/
theorem withInlining_none {R T1 : Type} (inst) :
    ScopeSpecOn (fun (b : T1) s => okHost (γ := R) (inst.call_once b s))
      (fun b s => okHost (chelis_ir.host.with_inlining inst none s b)) id (fun _ s₁ => s₁) id := by
  intro b s
  simp only [chelis_ir.host.with_inlining]
  unfold_specialized
  simp only [Aeneas.Std.chelis_ir.lowering_runtime.scoped, core.ops.function.FnOnceMutArg.thread,
    core.ops.function.FnOnceMutArgPair.thread, bind_tc_ok, id]
  cases h : inst.call_once b s with
  | ok x => obtain ⟨r, s₁, bk⟩ := x; simp [h, okHost]
  | fail e => simp [h, okHost]
  | div => simp [h, okHost]

/-- **`with_inlining(Some(name))`, every instance, on the invariant**: for a name whose `str`
view is `n` (Rust `&str` is the UTF-8 of a `String`), the body runs with `name` inserted and at
exit `name` is removed exactly when the entry inserted it (a repeated name belongs to the outer
scope); every other effect of the body is kept. Hypotheses: the entry set satisfies `SetInv` and
the body's final set does (see `withInlining_obligation_false` for why). -/
theorem withInlining_spec {R T1 : Type} (inst) (m : String) (n : Str) (hm : Str.ofString m = .ok n)
    (b : T1) (s : HCx) (hs : InliningInv s)
    (hb : ∀ s₁ r, okHost (γ := R) (inst.call_once b (inliningEnter m s)) = some (s₁, r) →
      InliningInv s₁) :
    okHost (chelis_ir.host.with_inlining inst (some n) s b) =
      (okHost (inst.call_once b (inliningEnter m s))).map
        (fun x => (inliningExit m s x.1, x.2)) := by
  simp only [chelis_ir.host.with_inlining]
  unfold_specialized
  have hs1 : s.state.invocation.inlining.inner.map.entries.Pairwise (fun a b => a.1 < b.1) := hs.1
  simp only [Aeneas.Std.chelis_ir.lowering_runtime.scoped, core.ops.function.FnOnceMutArg.thread,
    core.ops.function.FnOnceMutArgPair.thread, bind_tc_ok,
    chelis_ir.host.with_inlining.closure.closure.Insts.CoreOpsFunctionFnOnceTupleSharedStrBool.call_once,
    to_string_ofString m n hm, chelis_unord.UnordSet.insert, btreeSet_insert m _ hs1]
  have henter : inliningEnter m s = inliningLens.set (insS m (inliningLens.get s)) s := rfl
  rw [henter] at hb ⊢
  simp only [inliningLens, insS] at hb ⊢
  cases h : inst.call_once b { s with state := { s.state with invocation :=
      { s.state.invocation with inlining := ⟨⟨⟨insList m s.state.invocation.inlining.inner.map.entries⟩⟩⟩ } } } with
  | ok x =>
    obtain ⟨r, s₁, bk⟩ := x
    have hs₁ : InliningInv s₁ := hb s₁ r (by simp [h, okHost])
    by_cases hmem : m ∈ setKeys s.state.invocation.inlining
    · have : ¬ (m ∉ keys s.state.invocation.inlining.inner.map.entries) := by
        simpa [setKeys] using hmem
      simp [h, okHost, inliningExit, memS, inliningLens, this, hmem]
    · have hmem' : m ∉ keys s.state.invocation.inlining.inner.map.entries := by
        simpa [setKeys] using hmem
      have hr1 : s₁.state.invocation.inlining.inner.map.entries.Pairwise (fun a b => a.1 < b.1) :=
        hs₁.1
      have hr2 : ∀ k ∈ keys s₁.state.invocation.inlining.inner.map.entries,
          k.toByteArray.size ≤ Usize.max := hs₁.2
      simp [h, okHost, inliningExit, memS, inliningLens, hmem, hmem', chelis_unord.UnordSet.remove,
        btreeSet_remove m n hm _ hr1 hr2, delS]
  | fail e => simp [h, okHost]
  | div => simp [h, okHost]

/-- The extracted `with_inlining` as a `Scopes.Combinator` (state-transformer bodies). -/
def inliningComb (nm : Option Str) (γ : Type) : Combinator Result HCx γ :=
  asComb fun body s =>
    chelis_ir.host.with_inlining (R := γ) (T1 := HCx → Result (HCx × γ)) ⟨stCall⟩ nm s body

/-- **Section E `InliningObligation` on the invariant** (`InliningObligationUnder`) for the
extracted `with_inlining(Some(name))`, with membership, ordered insertion and removal. -/
theorem withInlining_obligation (m : String) (n : Str) (hm : Str.ofString m = .ok n) (γ : Type) :
    InliningObligationUnder okResult InliningInv inliningLens memS insS delS m
      (inliningComb (some n) γ) := by
  intro body s hs hb
  have := withInlining_spec (R := γ) (T1 := HCx → Result (HCx × γ)) ⟨stCall⟩ m n hm body s hs
    (fun s₁ r h => hb s₁ r (by simpa only [okHost_stCall] using h))
  simp only [okHost_stCall] at this
  simp only [inliningComb, okResult_asComb]
  rw [this]
  rfl

/-- A body that leaves the inlining set as it found it: the scope gives back the entry set when
it inserted the name (and the entry set itself when the name belonged to an outer scope). -/
theorem withInlining_restores_untouched (m : String) (n : Str) (hm : Str.ofString m = .ok n)
    (γ : Type) (body : HCx → Result (HCx × γ)) (s s' : HCx) (r : γ) (hs : InliningInv s)
    (hb : ∀ s₁ r, okResult (body (inliningEnter m s)) = some (s₁, r) →
      inliningLens.get s₁ = inliningLens.get (inliningEnter m s))
    (h : okResult (inliningComb (some n) γ body s) = some (s', r)) :
    inliningLens.get s' = inliningLens.get s := by
  have hmr : m.toByteArray.size ≤ Usize.max := by
    by_contra hc
    rw [String.size_toByteArray] at hc
    simp [Str.ofString, hc] at hm
  have hinv : ∀ s₁ r, okResult (body (inliningEnter m s)) = some (s₁, r) → InliningInv s₁ := by
    intro s₁ r' h₁
    have heq := hb s₁ r' h₁
    refine ⟨?_, fun k hk => ?_⟩
    · rw [heq]; exact insS_pairwise m _ hs
    · rw [heq] at hk
      rcases mem_keys_insList m k _ hk with rfl | hk
      · exact hmr
      · exact hs.2 k hk
  have hob := withInlining_obligation m n hm γ body s hs hinv
  have h' : (okResult (body (inliningEnter m s))).map (fun x => (inliningExit m s x.1, x.2)) =
      some (s', r) := by
    rw [hob] at h; exact h
  cases hB : okResult (body (inliningEnter m s)) with
  | none => rw [hB] at h'; cases h'
  | some x =>
    obtain ⟨s₁, r₁⟩ := x
    rw [hB] at h'
    simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at h'
    obtain ⟨rfl, -⟩ := h'
    have heq := hb s₁ r₁ hB
    rcases hS : inliningLens.get s with ⟨⟨⟨l⟩⟩⟩
    have hl1 : l.Pairwise (fun a b => a.1 < b.1) := by
      have := hs.1; rw [hS] at this; exact this
    simp only [inliningEnter, inliningLens_lawful.get_set, hS, insS] at heq
    simp only [inliningExit, memS, setKeys, hS]
    by_cases hmem : m ∈ keys l
    · have hl : insList m l = l := insList_of_mem m l hl1 hmem
      simp only [hmem, decide_true, if_true, heq, hl]
    · simp only [hmem, decide_false, if_false, Bool.false_eq_true, inliningLens_lawful.get_set, heq,
        delS, delList_insList m l hmem]

/-! ## The unrestricted inlining obligation is false in the model -/

/-- A context whose inlining set is `{oversized}` (`StrOrder.oversized`: `usize::MAX + 1`
bytes, more than any native `String`). -/
def withOversized (s : HCx) : HCx := inliningLens.set ⟨⟨⟨[(oversized, ())]⟩⟩⟩ s

/-- **Counterexample.** At every context whose inlining set does not hold `name`, the body that
stores `{oversized}` succeeds while the extracted scope fails: its exit removes `name` (the entry
inserted it) and that removal borrows the stored key as `str`, which has no model value past
`usize::MAX` bytes. -/
theorem withInlining_oversized_fails (m : String) (n : Str) (hm : Str.ofString m = .ok n)
    (γ : Type) (r : γ) (s : HCx) (hs : m ∉ setKeys (inliningLens.get s)) :
    okResult (inliningComb (some n) γ (fun t => .ok (withOversized t, r)) s) = none := by
  have habs : alloc.collections.btree.map.insertList core.cmp.OrdString m ()
      s.state.invocation.inlining.inner.map.entries =
      .ok (none, insList m s.state.invocation.inlining.inner.map.entries) :=
    insertList_absent m _ (by simpa [setKeys, inliningLens] using hs)
  simp only [inliningComb, okResult_asComb, chelis_ir.host.with_inlining]
  unfold_specialized
  simp [Aeneas.Std.chelis_ir.lowering_runtime.scoped, core.ops.function.FnOnceMutArg.thread,
    core.ops.function.FnOnceMutArgPair.thread, stCall,
    chelis_ir.host.with_inlining.closure.closure.Insts.CoreOpsFunctionFnOnceTupleSharedStrBool.call_once,
    to_string_ofString m n hm,
    chelis_unord.UnordSet.insert, alloc.collections.btree.set.BTreeSet.insert,
    alloc.collections.btree.map.BTreeMap.insert, habs, withOversized, inliningLens,
    chelis_unord.UnordSet.remove, alloc.collections.btree.set.BTreeSet.remove,
    alloc.collections.btree.map.BTreeMap.remove, removeList_oversized, okHost]

/-- **The unrestricted `InliningObligation` is false** for the extracted `with_inlining`,
whatever membership, insertion and removal it is stated with: at any context not holding
`name`, the `withInlining_oversized_fails` body refutes it. -/
theorem withInlining_obligation_false (m : String) (n : Str) (hm : Str.ofString m = .ok n)
    (γ : Type) (r : γ) (mem : String → chelis_unord.UnordSet String → Bool)
    (ins del : String → chelis_unord.UnordSet String → chelis_unord.UnordSet String) (s : HCx)
    (hs : m ∉ setKeys (inliningLens.get s)) :
    ¬ InliningObligation okResult inliningLens mem ins del m (inliningComb (some n) γ) := by
  intro h
  have := h (fun t => .ok (withOversized t, r)) s
  rw [withInlining_oversized_fails m n hm γ r s hs] at this
  simp [okResult] at this

/-! ## `with_preflight` -/

/-- The tensor-helper preflight frames of the context, top first: the `Vec`'s last element is the
top (`Vec::push` appends, `Vec::pop` removes the last element). `set` stores a stack of at most
`usize::MAX` frames and leaves the context unchanged otherwise (no `Vec` holds more). -/
def preflightLens : Lens HCx (List chelis_ir.host.PreflightFrame) where
  get s := s.state.invocation.preflight.val.reverse
  set l s := if h : l.length ≤ Usize.max then
      { s with state := { s.state with invocation := { s.state.invocation with
          preflight := ⟨l.reverse, by simpa using h⟩ } } }
    else s

/-- The entry step of the extracted `with_preflight expr`: take the frame id from the runtime's
supply (`fresh_preflight_frame`), fill the program's summary cache
(`cached_dynamic_to_tensor_def_summaries`) and analyze `expr` against those summaries; the scope
pushes the frame `{id, facts}` on the resulting context. `none` when a step fails or diverges, or
when the stack already holds `usize::MAX` frames (`Vec::push` fails). -/
def preflightEntry (expr : chelis_deep.ast.Expr) (s : HCx) :
    Option (HCx × chelis_ir.host.PreflightFrame) :=
  match (do
      let (i, lr) ← Aeneas.Std.chelis_ir.lowering_runtime.LoweringRuntime.fresh_preflight_frame
        s.runtime
      let (summaries, s₀, _) ←
        chelis_ir.host.cached_dynamic_to_tensor_def_summaries { s with runtime := lr }
      let facts ← chelis_ir.host.PreflightFacts.new
      let (_, facts1) ← chelis_ir.host.analyze_tensor_helper_preflight expr
        (Aeneas.Std.alloc.sync.DerefArc.deref summaries) facts
      Result.ok (s₀, ({ id := i, facts := facts1 } : chelis_ir.host.PreflightFrame)) :
      Result (HCx × chelis_ir.host.PreflightFrame)) with
  | .ok (s₀, fr) => if s₀.state.invocation.preflight.val.length < Usize.max then some (s₀, fr) else none
  | _ => none

/-- `Vec::push` succeeds exactly below `usize::MAX` elements, appending. -/
theorem vec_push_eq {α : Type} (v : alloc.vec.Vec α) (x : α) :
    v.push x = if h : v.val.length < Usize.max then
      .ok ⟨v.val ++ [x], by simp; omega⟩ else .fail .maximumSizeExceeded := by
  unfold alloc.vec.Vec.push
  by_cases h : v.val.length < Usize.max
  · simp [h]
  · have h1 : ¬ (v.val.length + 1 ≤ Usize.max) := by omega
    have h2 : ¬ (v.val.length + 1 ≤ U32.max) := by
      rcases Usize.bounds_eq with h3 | h3 <;> simp only [U32.max_eq, U64.max_eq] at h3 ⊢ <;> omega
    simp [h, h1, h2]

/-- Pushing a frame on a stack that has room is the lens update. -/
theorem preflightLens_set_push (s : HCx) (fr : chelis_ir.host.PreflightFrame)
    (h : s.state.invocation.preflight.val.length < Usize.max) :
    preflightLens.set (fr :: preflightLens.get s) s =
      { s with state := { s.state with invocation := { s.state.invocation with
          preflight := ⟨s.state.invocation.preflight.val ++ [fr], by simp; omega⟩ } } } := by
  have hl : (fr :: s.state.invocation.preflight.val.reverse).length ≤ Usize.max := by simp; omega
  simp only [preflightLens, hl, _root_.dite_true, List.reverse_cons, List.reverse_reverse]

/-- Popping the top frame of a stack is the lens update. -/
theorem preflightLens_set_pop (s : HCx) :
    preflightLens.set (preflightLens.get s).tail s =
      { s with state := { s.state with invocation := { s.state.invocation with
          preflight := ⟨s.state.invocation.preflight.val.dropLast, by
            have := s.state.invocation.preflight.property; simp; omega⟩ } } } := by
  have hl : s.state.invocation.preflight.val.dropLast.reverse.length ≤ Usize.max := by
    have := s.state.invocation.preflight.property; simp; omega
  simp only [preflightLens, List.tail_reverse, List.reverse_reverse, dif_pos hl]

/-- **`with_preflight expr`, every instance**: on a successful entry (`preflightEntry`: fresh
frame id, filled summary cache, analyzed facts, room on the stack) the body runs on the context
with the frame `{id, facts}` pushed and receives the frame's root node `{frame := id, node := 0}`;
at exit the scope pops the top frame of the body's final context, failing (the Rust `expect`
panic) when the body left no frame; the body's result is returned unchanged. A failing entry or
body makes the scope fail. -/
theorem withPreflight_spec {R T1 : Type} (inst) (expr : chelis_deep.ast.Expr) (b : T1) (s : HCx) :
    okHost (chelis_ir.host.with_preflight (R := R) (T1 := T1) inst expr s b) =
      match preflightEntry expr s with
      | none => none
      | some (s₀, fr) =>
        match okHost (inst.call_once b (preflightLens.set (fr :: preflightLens.get s₀) s₀,
            ({ frame := fr.id, node := 0#usize } : chelis_ir.host.PreflightNode))) with
        | none => none
        | some (s₁, r) => if preflightLens.get s₁ = [] then none
            else some (preflightLens.set (preflightLens.get s₁).tail s₁, r) := by
  simp only [chelis_ir.host.with_preflight]
  unfold_specialized
  simp only [Aeneas.Std.chelis_ir.lowering_runtime.scoped, core.ops.function.FnOnceMutArg.thread,
    core.ops.function.FnOnceMutArgPair.thread, chelis_ir.host.push_preflight_frame,
    chelis_ir.host.pop_preflight_frame, preflightEntry,
    Aeneas.Std.chelis_ir.lowering_runtime.LoweringRuntime.fresh_preflight_frame, bind_tc_ok,
    Function.uncurry_apply_pair, lift]
  generalize ({ s.runtime with nextPreflightFrame := _ } :
    Aeneas.Std.chelis_ir.lowering_runtime.LoweringRuntime) = lr
  generalize s.runtime.nextPreflightFrame = i
  generalize ({ checked := s.checked, state := s.state, runtime := lr, native := s.native } : HCx) = s'
  cases hc : chelis_ir.host.cached_dynamic_to_tensor_def_summaries s' with
  | fail e => simp [okHost]
  | div => simp [okHost]
  | ok x =>
  obtain ⟨summaries, s₀, bk⟩ := x
  simp only [bind_tc_ok, Function.uncurry_apply_pair]
  cases hf : chelis_ir.host.PreflightFacts.new with
  | fail e => simp [okHost]
  | div => simp [okHost]
  | ok facts =>
  simp only [bind_tc_ok]
  cases ha : chelis_ir.host.analyze_tensor_helper_preflight expr
      (Aeneas.Std.alloc.sync.DerefArc.deref summaries) facts with
  | fail e => simp [okHost]
  | div => simp [okHost]
  | ok y =>
  obtain ⟨u, facts1⟩ := y
  simp only [bind_tc_ok, Function.uncurry_apply_pair, vec_push_eq]
  by_cases hlen : s₀.state.invocation.preflight.val.length < Usize.max
  · cases hb : inst.call_once b (preflightLens.set ({ id := i, facts := facts1 } :: preflightLens.get s₀) s₀,
        { frame := i, node := 0#usize }) with
    | fail e =>
      simp only [preflightLens_set_push _ _ hlen] at hb
      simp [hlen, hb, okHost, preflightLens_set_push _ _ hlen]
    | div =>
      simp only [preflightLens_set_push _ _ hlen] at hb
      simp [hlen, hb, okHost, preflightLens_set_push _ _ hlen]
    | ok z =>
    obtain ⟨r, s₁, cb⟩ := z
    simp only [preflightLens_set_push _ _ hlen] at hb
    simp only [hlen, _root_.dite_true, _root_.if_true, bind_tc_ok, Function.uncurry_apply_pair,
      preflightLens_set_push _ _ hlen, hb, okHost, alloc.vec.Vec.pop]
    cases hl : s₁.state.invocation.preflight.val.getLast? with
    | none =>
      have h0 : s₁.state.invocation.preflight.val = [] := List.getLast?_eq_none_iff.mp hl
      simp [h0, preflightLens, core.option.Option.expect, Result.ofOption]
    | some x =>
      have hne : preflightLens.get s₁ ≠ [] := by
        intro h
        simp only [preflightLens, List.reverse_eq_nil_iff] at h
        simp [h] at hl
      rw [if_neg hne, preflightLens_set_pop]
      simp [core.option.Option.expect, Result.ofOption]
  · simp [hlen, okHost]

/-- The extracted `with_preflight expr` as a `Scopes.Combinator`: state-transformer bodies that
ignore the frame root they receive. -/
def preflightComb (expr : chelis_deep.ast.Expr) (γ : Type) : Combinator Result HCx γ :=
  asComb fun body s =>
    chelis_ir.host.with_preflight (R := γ) (T1 := HCx → Result (HCx × γ))
      ⟨fun b a => stCall b a.1⟩ expr s body

/-- **`Scopes.PreflightObligationFrom`** for the extracted `with_preflight expr`, with the entry
step `preflightEntry expr`: on bodies that leave a frame, the scope runs the body on the entry's
context with the computed frame pushed and pops the top frame of the body's final context. -/
theorem withPreflight_obligation (expr : chelis_deep.ast.Expr) (γ : Type) :
    PreflightObligationFrom okResult preflightLens (preflightEntry expr) (preflightComb expr γ) := by
  intro body s s₀ fr hpre hleave
  simp only [preflightComb, okResult_asComb, withPreflight_spec, hpre]
  show (match okHost (stCall body (preflightLens.set (fr :: preflightLens.get s₀) s₀)) with
      | none => none
      | some (s₁, r) => if preflightLens.get s₁ = [] then none
          else some (preflightLens.set (preflightLens.get s₁).tail s₁, r)) = _
  rw [okHost_stCall]
  cases hb : okResult (body (preflightLens.set (fr :: preflightLens.get s₀) s₀)) with
  | none => rfl
  | some x =>
    obtain ⟨s₁, r⟩ := x
    simp [hleave s₁ r hb]

end ChelisBridge.Runtime.ExtractedHostScopes
