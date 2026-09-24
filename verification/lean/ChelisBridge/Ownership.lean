import ChelisBridge.Decode
import LaCaDiLECore.GradientValidation

/-!
# Decoding the actual final ownership plan

The Rust DAG ownership plan (`chelis_ir::ownership::DagOwnershipPlan`) is a list of
`DagDirective`s: one per DAG node in node order, then one per root in root order, then the
implicit scope-exit drops. `Directive` mirrors that enum constructor for constructor; the
`OwnershipPlan` class is the only link to the generated carrier (the extracted AD slice
does not contain the ownership lowering yet, so the instance for the generated
`DagOwnershipProgram` is one constructor-by-constructor map once it is extracted).

`decodeOwned` reads the plan into reference owned syntax `GraphSyntax.Owned ℚ`:

* `BorrowLoad` binds a requested entry (identified through the plan DAG's `Load` ids): entry
  `j` is the borrowed owner `|Γ| - 1 - j`.
* `Produce` (with exactly the borrows Rust records: inputs, then new shape dependencies)
  and `Clone` (for `Copy`) emit `bind` of the decoded operation, a fresh local owner.
* `BorrowedDrop`/`OwnedDrop` (for `Drop` nodes) emit `drop`; the Rust mode must agree with
  whether the source is an entry (borrowed) or a local.
* `Root` emits `yield`; the same source may not be transferred twice.
* `RootClone` is expanded into an explicit `bind (copy …)` of a fresh owner and `yield`
  of that copy.
* `ScopeDrop` emits `drop`.
* `Realize`, `Store`, `StoreRoot`, mismatched or leftover directives are rejected.

The decoder grants no permission: `OwnedCode.check` from the initial state (entries
borrowed) is the independent ownership judgment, run on exactly the decoded instructions
(`decodeOwned_sound`). Physical allocation and buffer reuse are not covered.
-/

namespace ChelisBridge.Ownership

open Aeneas.Std LaCaDiLECore ChelisBridge

/-- Mirror of Rust `DagDirective` (node ids as naturals). -/
inductive Directive where
  | borrowLoad (node : Nat)
  | produce (node : Nat) (borrows : List Nat)
  | clone (node source : Nat)
  | cloneProduce (node source : Nat)
  | moveProduce (node source : Nat)
  | cloneStore (node source : Nat)
  | moveStore (node source : Nat)
  | storeRoot (node : Nat)
  | borrowedDrop (node source : Nat)
  | ownedDrop (node source : Nat)
  | root (source : Nat)
  | rootClone (source : Nat)
  | scopeDrop (source : Nat)
  deriving DecidableEq, Repr

/-- The source of an owned root transfer, if the directive is one. -/
def Directive.transfer? : Directive → Option Nat
  | .root source => some source
  | _ => none

/-- Interface to the final ownership-lowered program: its DAG payload and directives. -/
class OwnershipPlan (β : Type) where
  dag : β → Model.Dag
  directives : β → List Directive

/-- A plan given directly by its payload and directive list. -/
structure PlanMirror where
  dag : Model.Dag
  directives : List Directive

instance : OwnershipPlan PlanMirror := ⟨PlanMirror.dag, PlanMirror.directives⟩

/-- Flat owned instructions, folded into `GraphSyntax.Owned`. -/
inductive Instr where
  | bind (op : GraphSyntax.Operation ℚ)
  | drop (shape : Shape) (input : Nat)
  | yield (shape : Shape) (input : Nat)
  deriving DecidableEq, Repr

def Instr.toOwned : List Instr → GraphSyntax.Owned ℚ
  | [] => .done
  | .bind op :: rest => .bind op (toOwned rest)
  | .drop s i :: rest => .drop s i (toOwned rest)
  | .yield s i :: rest => .yield s i (toOwned rest)

/-- Decoder state: the value-node state of `Decode` plus the emitted instructions. -/
structure OState where
  st : State
  instrs : List Instr

/-- The borrows Rust records for a produced node: its inputs, then each shape dependency
not already present. -/
def expectedBorrows (node : Model.DagNode) : List Nat :=
  (node.shape_deps.val.map (·.val)).foldl
    (fun acc d => if acc.contains d then acc else acc ++ [d]) (node.inputs.val.map (·.val))

/-- Bind a decoded operation as a fresh local owner for node `idx`. -/
def emitBind (os : OState) (idx : Nat) (out : Shape) (o : GraphSyntax.Operation ℚ) : OState :=
  ⟨⟨os.st.len + 1, (idx, ⟨os.st.len, out⟩) :: os.st.env, os.st.ops ++ [o]⟩,
    os.instrs ++ [.bind o]⟩

/-- A `Drop` node: `borrowed` is the Rust mode, which must match the source's kind. -/
def dropStep (Γ : Context) (os : OState) (node : Model.DagNode) (src : Nat) (borrowed : Bool) :
    Option OState := do
  guard (node.inputs.val.map (·.val) = [src])
  let sl ← os.st.slot src
  let i ← os.st.offset src
  guard (decide (sl.owner < Γ.length) = borrowed)
  pure ⟨os.st, os.instrs ++ [.drop sl.shape i]⟩

/-- A value node (`Load`, `Copy` or a produced fragment operation) with its directive. -/
def valueStep (dt : DType) (entries : List (Option Nat)) (Γ : Context) (os : OState) (idx : Nat)
    (node : Model.DagNode) (d : Directive) : Option OState := do
  guard (DType.ofPrim node.output_type.precision = some dt)
  let out ← resolveShape node.output_type.dims
  match node.op, d with
  | .Load _, .borrowLoad n => do
      guard (n = idx ∧ node.inputs.val = [])
      let j ← entries.findIdx? (· == some idx)
      guard (Γ[j]? = some out)
      pure ⟨{ os.st with env := (idx, ⟨Γ.length - 1 - j, out⟩) :: os.st.env }, os.instrs⟩
  | .Copy, .clone n src => do
      guard (n = idx ∧ node.inputs.val.map (·.val) = [src])
      let o ← decodeOperation dt os.st .Copy node.inputs.val out
      pure (emitBind os idx out o)
  | .Copy, _ => none
  | op, .produce n borrows => do
      guard (n = idx ∧ borrows = expectedBorrows node)
      let o ← decodeOperation dt os.st op node.inputs.val out
      pure (emitBind os idx out o)
  | _, _ => none

/-- One DAG node with its directive, at append position `idx`. -/
def nodeStep (dt : DType) (entries : List (Option Nat)) (Γ : Context) (os : OState) (idx : Nat)
    (node : Model.DagNode) (d : Directive) : Option OState :=
  if node.id.val = idx then
    match node.op, d with
    | .Drop, .borrowedDrop n src => if n = idx then dropStep Γ os node src true else none
    | .Drop, .ownedDrop n src => if n = idx then dropStep Γ os node src false else none
    | .Drop, _ => none
    | _, _ => valueStep dt entries Γ os idx node d
  else none

def runNodes (dt : DType) (entries : List (Option Nat)) (Γ : Context) :
    Nat → OState → List (Model.DagNode × Directive) → Option OState
  | _, os, [] => some os
  | i, os, (node, d) :: rest => do
      let os' ← nodeStep dt entries Γ os i node d
      runNodes dt entries Γ (i + 1) os' rest

/-- One root with its directive. `Root` transfers an owner (never twice); `RootClone` is
expanded into an explicit copy of a fresh owner followed by its transfer. -/
def rootStep (os : OState) (transferred : List Nat) (r : Nat) :
    Directive → Option (OState × List Nat × Shape)
  | .root src => do
      guard (src = r ∧ src ∉ transferred)
      let sl ← os.st.slot r
      let i ← os.st.offset r
      pure (⟨os.st, os.instrs ++ [.yield sl.shape i]⟩, r :: transferred, sl.shape)
  | .rootClone src => do
      guard (src = r)
      let sl ← os.st.slot r
      let i ← os.st.offset r
      let copy : GraphSyntax.Operation ℚ := .copy sl.shape i
      pure (⟨⟨os.st.len + 1, os.st.env, os.st.ops ++ [copy]⟩,
        os.instrs ++ [.bind copy, .yield sl.shape 0]⟩, transferred, sl.shape)
  | _ => none

def runRoots : OState → List Nat → List (Nat × Directive) → Option (OState × List Nat × Context)
  | os, tr, [] => some (os, tr, [])
  | os, tr, (r, d) :: rest => do
      let (os', tr', s) ← rootStep os tr r d
      let (os'', tr'', Ω) ← runRoots os' tr' rest
      pure (os'', tr'', s :: Ω)

def runScopeDrops (os : OState) : List Directive → Option OState
  | [] => some os
  | .scopeDrop src :: rest => do
      let sl ← os.st.slot src
      let i ← os.st.offset src
      runScopeDrops ⟨os.st, os.instrs ++ [.drop sl.shape i]⟩ rest
  | _ :: _ => none

/-- The flat instructions and root shapes of a plan: node directives, root directives,
then scope drops, consuming every directive exactly once. -/
def planInstrs (dt : DType) (entries : List (Option Nat)) (Γ : Context) (dag : Model.Dag)
    (ds : List Directive) : Option (List Instr × Context) := do
  guard (entries.reduceOption.Nodup ∧ entries.length = Γ.length)
  let nodes := dag.nodes.val
  let roots := dag.roots.val.map (·.val)
  guard (nodes.length + roots.length ≤ ds.length)
  let os ← runNodes dt entries Γ 0 ⟨State.init Γ, []⟩ (nodes.zip (ds.take nodes.length))
  let (os, _, Ω) ← runRoots os [] (roots.zip ((ds.drop nodes.length).take roots.length))
  let os ← runScopeDrops os (ds.drop (nodes.length + roots.length))
  pure (os.instrs, Ω)

variable {β : Type} [OwnershipPlan β]

/-- The owned syntax decoded from the actual plan over the entry context `Γ`; `entries[j]`
is the plan DAG's `Load` node of entry `j`, if any (each `Load` must be one of them, with
the entry's shape). -/
def ownedSyntax (dt : DType) (entries : List (Option Nat)) (Γ : Context) (plan : β) :
    Option (GraphSyntax.Owned ℚ × Context) := do
  let (is, Ω) ← planInstrs dt entries Γ (OwnershipPlan.dag plan) (OwnershipPlan.directives plan)
  pure (Instr.toOwned is, Ω)

/-- The typed final plan `OwnedCode ℚ Γ Ω`. -/
def decodeOwned (dt : DType) (entries : List (Option Nat)) (Γ Ω : Context) (plan : β) :
    Option (OwnedCode ℚ Γ Ω) := do
  let (syn, _) ← ownedSyntax dt entries Γ plan
  syn.decode Γ Ω

/-! ## Theorems -/

theorem guard_some {p : Prop} [Decidable p] {u : Unit} (h : (guard p : Option Unit) = some u) :
    p := by
  by_contra hn
  simp [hn] at h

/-- Typed owned code re-encodes to exactly the syntax it was decoded from. -/
theorem Owned.encode_of_decode :
    ∀ {Γ Ω : Context} (syn : GraphSyntax.Owned ℚ) (c : OwnedCode ℚ Γ Ω),
      syn.decode Γ Ω = some c → c.encode = syn
  | Γ, Ω, .done, c, h => by
      simp only [GraphSyntax.Owned.decode] at h
      split at h
      · rename_i hΩ
        subst hΩ
        cases Option.some.inj h
        rfl
      · contradiction
  | Γ, Ω, .bind op body, c, h => by
      simp only [GraphSyntax.Owned.decode, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at h
      obtain ⟨⟨s, op'⟩, hop, body', hbody, rfl⟩ := h
      simp [OwnedCode.encode, Operation.encode_of_decode op ⟨s, op'⟩ hop,
        Owned.encode_of_decode body body' hbody]
  | Γ, Ω, .drop s i body, c, h => by
      simp only [GraphSyntax.Owned.decode, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at h
      obtain ⟨v, hv, body', hbody, rfl⟩ := h
      simp [OwnedCode.encode, Var.offset_of_decode hv, Owned.encode_of_decode body body' hbody]
  | Γ, [], .yield s i body, c, h => by simp [GraphSyntax.Owned.decode] at h
  | Γ, t :: Ω, .yield s i body, c, h => by
      simp only [GraphSyntax.Owned.decode] at h
      split at h
      · rename_i hst
        subst hst
        simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
          Option.some.injEq] at h
        obtain ⟨v, hv, body', hbody, rfl⟩ := h
        simp [OwnedCode.encode, Var.offset_of_decode hv, Owned.encode_of_decode body body' hbody]
      · contradiction

/-- `RootClone` expands to a copy of the root into a fresh owner (offset `0` afterwards)
followed by the transfer of that copy; the original owner is not transferred. -/
theorem rootStep_rootClone {os : OState} {tr : List Nat} {r : Nat} {res : OState × List Nat × Shape}
    (h : rootStep os tr r (.rootClone r) = some res) :
    ∃ sl i, os.st.slot r = some sl ∧ os.st.offset r = some i ∧ res.2.1 = tr ∧
      res.1.instrs = os.instrs ++ [.bind (.copy sl.shape i), .yield sl.shape 0] := by
  simp only [rootStep, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
    Option.some.injEq] at h
  obtain ⟨_, _, sl, hsl, i, hi, rfl⟩ := h
  exact ⟨sl, i, hsl, hi, rfl, rfl⟩

theorem rootStep_transferred {os : OState} {tr : List Nat} {r : Nat} {d : Directive}
    {res : OState × List Nat × Shape} (h : rootStep os tr r d = some res) :
    res.2.1 = (d.transfer?.toList) ++ tr ∧ ∀ x ∈ d.transfer?.toList, x ∉ tr := by
  cases d <;> simp only [rootStep, reduceCtorEq] at h
  case root src =>
    simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨_, hg, sl, _, i, _, rfl⟩ := h
    obtain ⟨rfl, hn⟩ := guard_some hg
    simpa [Directive.transfer?] using hn
  case rootClone src =>
    simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
      Option.some.injEq] at h
    obtain ⟨_, _, sl, _, i, _, rfl⟩ := h
    simp [Directive.transfer?]

/-- The transferred owners of a successful root phase are pairwise distinct. -/
theorem runRoots_transferred :
    ∀ (os : OState) (tr : List Nat) (items : List (Nat × Directive))
      (res : OState × List Nat × Context), tr.Nodup → runRoots os tr items = some res →
      res.2.1.Nodup ∧ res.2.1.Perm ((items.map (·.2)).filterMap Directive.transfer? ++ tr)
  | os, tr, [], res, hnd, h => by
      simp only [runRoots, Option.some.injEq] at h
      subst h
      exact ⟨hnd, by simp⟩
  | os, tr, (r, d) :: rest, res, hnd, h => by
      simp only [runRoots, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
        Option.some.injEq] at h
      obtain ⟨⟨os', tr', s⟩, h1, ⟨os'', tr'', Ω⟩, h2, rfl⟩ := h
      obtain ⟨htr, hfresh⟩ := rootStep_transferred h1
      simp only at htr
      have hnd' : tr'.Nodup := by
        rw [htr]
        cases hd : d.transfer? with
        | none => simpa using hnd
        | some x =>
            simp only [Option.toList_some, List.singleton_append, List.nodup_cons]
            exact ⟨hfresh x (by simp [hd]), hnd⟩
      obtain ⟨hn2, hp2⟩ := runRoots_transferred os' tr' rest _ hnd' h2
      refine ⟨hn2, ?_⟩
      simp only at hp2 ⊢
      rw [htr] at hp2
      refine hp2.trans ?_
      cases hd : d.transfer? with
      | none => simp [hd]
      | some x =>
          simp only [hd, Option.toList_some, List.map_cons, List.filterMap_cons,
            List.cons_append]
          exact List.perm_middle

/-- A successful node pass accepts only node directives (no root transfer). -/
theorem nodeStep_transfer {dt : DType} {entries : List (Option Nat)} {Γ : Context} {os os' : OState}
    {i : Nat} {node : Model.DagNode} {d : Directive}
    (h : nodeStep dt entries Γ os i node d = some os') : d.transfer? = none := by
  cases d with
  | root src =>
      unfold nodeStep at h
      split at h
      · split at h <;> first
          | contradiction
          | (simp only [valueStep, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
             obtain ⟨_, _, _, _, h⟩ := h
             split at h <;> simp_all)
      · contradiction
  | _ => rfl

theorem runNodes_transfer {dt : DType} {entries : List (Option Nat)} {Γ : Context} :
    ∀ (i : Nat) (os os' : OState) (items : List (Model.DagNode × Directive)),
      runNodes dt entries Γ i os items = some os' →
      (items.map (·.2)).filterMap Directive.transfer? = []
  | _, _, _, [], _ => rfl
  | i, os, os', (node, d) :: rest, h => by
      simp only [runNodes, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
      obtain ⟨os1, h1, h2⟩ := h
      simp [nodeStep_transfer h1, runNodes_transfer (i + 1) os1 os' rest h2]

theorem runScopeDrops_transfer :
    ∀ (os os' : OState) (ds : List Directive), runScopeDrops os ds = some os' →
      ds.filterMap Directive.transfer? = []
  | _, _, [], _ => rfl
  | os, os', d :: rest, h => by
      cases d with
      | scopeDrop src =>
          simp only [runScopeDrops, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
          obtain ⟨sl, _, i, _, h⟩ := h
          rw [List.filterMap_cons_none rfl]
          exact runScopeDrops_transfer _ os' rest h
      | _ => simp [runScopeDrops] at h

/-- **No repeated owned-root transfer.** Every `Root` directive of an accepted plan names a
different owner (Rust `DagDuplicateTerminal`); repeated roots are admitted only through
`RootClone`, i.e. as copies. -/
theorem planInstrs_transfers_nodup {dt : DType} {entries : List (Option Nat)} {Γ : Context}
    {dag : Model.Dag} {ds : List Directive} {res : List Instr × Context}
    (h : planInstrs dt entries Γ dag ds = some res) :
    (ds.filterMap Directive.transfer?).Nodup := by
  simp only [planInstrs, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
    Option.some.injEq] at h
  obtain ⟨_, _, _, _, os, hn, ⟨os1, tr, Ω⟩, hr, os2, hs, _⟩ := h
  have h1 := runNodes_transfer 0 _ os _ hn
  have h3 := runScopeDrops_transfer os1 os2 _ hs
  obtain ⟨hnd, hperm⟩ := runRoots_transferred os [] _ _ List.nodup_nil hr
  have hz : ((dag.nodes.val.zip (ds.take dag.nodes.val.length)).map (·.2)) =
      ds.take dag.nodes.val.length :=
    List.map_snd_zip (by simp)
  have hz2 : (((dag.roots.val.map (·.val)).zip
      ((ds.drop dag.nodes.val.length).take (dag.roots.val.map (·.val)).length)).map (·.2)) =
      (ds.drop dag.nodes.val.length).take (dag.roots.val.map (·.val)).length :=
    List.map_snd_zip (by simp)
  rw [hz] at h1
  rw [hz2, List.append_nil] at hperm
  have hsplit : ds = ds.take dag.nodes.val.length ++
      ((ds.drop dag.nodes.val.length).take (dag.roots.val.map (·.val)).length ++
        ds.drop (dag.nodes.val.length + (dag.roots.val.map (·.val)).length)) := by
    rw [← List.drop_drop, List.take_append_drop, List.take_append_drop]
  rw [hsplit, List.filterMap_append, List.filterMap_append, h1, h3, List.nil_append,
    List.append_nil]
  exact hperm.nodup_iff.mp hnd

/-- **Decoded-plan soundness.** A decoded final plan is exactly the typed form of the
decoded Rust instructions (with `RootClone` expanded, entries borrowed, fresh local
owners), transfers no owner twice, and — if the independent reference ownership check
accepts it from the initial state — executes with every access permitted, returns its
value semantics, never terminates a borrowed entry and terminates every local owner
exactly once. -/
theorem decodeOwned_sound {dt : DType} {entries : List (Option Nat)} {Γ Ω : Context} {plan : β}
    {final : OwnedCode ℚ Γ Ω} (h : decodeOwned dt entries Γ Ω plan = some final) :
    (∃ is Ω', planInstrs dt entries Γ (OwnershipPlan.dag plan) (OwnershipPlan.directives plan) =
        some (is, Ω') ∧ final.encode = Instr.toOwned is) ∧
    ((OwnershipPlan.directives plan).filterMap Directive.transfer?).Nodup ∧
    (final.check (Ownership.initial Γ.length) = true →
      ∀ {R : Type} [CommSemiring R] [Algebra ℚ R] (ρ : Values R Γ),
        ∃ ended, final.run ρ (Ownership.initial Γ.length) = some (final.eval ρ, ended) ∧
          ended.Valid ∧ ended.live = ∅ ∧ ended.borrowed = Finset.range Γ.length ∧
          (∀ i, Γ.length ≤ i → i < ended.next → ended.ended.count i = 1) ∧
          (∀ i, i < Γ.length → ended.ended.count i = 0)) := by
  simp only [decodeOwned, ownedSyntax, Option.bind_eq_bind, Option.bind_eq_some_iff,
    Option.pure_def, Option.some.injEq] at h
  obtain ⟨⟨syn, Ω'⟩, ⟨⟨is, Ω''⟩, hp, hsyn⟩, hdec⟩ := h
  simp only [Prod.mk.injEq] at hsyn
  obtain ⟨rfl, rfl⟩ := hsyn
  refine ⟨⟨is, Ω'', hp, Owned.encode_of_decode _ final hdec⟩,
    planInstrs_transfers_nodup hp, ?_⟩
  intro hc R _ _ ρ
  exact final.checked_execution ρ hc

end ChelisBridge.Ownership
