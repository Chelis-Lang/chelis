import ChelisBridge.Validation

/-!
# The extracted production ownership lowering, decoded

`chelis_ir::ownership::lower_dag_ownership` (extracted by Aeneas: `Model.lowerDagOwnership`)
consumes a DAG and returns a `DagOwnershipProgram`: the payload DAG together with the private
`DagOwnershipPlan` whose directive vector is the final ownership plan. `loweredPlan` reads
that actual returned value — the payload DAG and every directive, constructor by
constructor, `RootClone` kept as returned — into the `Ownership.OwnershipPlan` interface, so
`Ownership.decodeOwned` decodes exactly what production returned. Nothing here re-derives
or repairs a plan.

Domain (every theorem below): logical ownership only — borrowed entries, fresh local owners,
each local terminated exactly once, no entry terminated, no owned root transferred twice.
Physical allocation, buffer reuse, the capacity scope drawn from the `CapacityScopeSupply`,
and cross-plan capacity identities are not claimed. The supply is the Aeneas oracle model of
the process-global identity sequence; theorems quantify over every supply.
-/

namespace ChelisBridge.ExtractedOwnership

open Aeneas.Std LaCaDiLECore ChelisBridge ChelisBridge.Ownership

/-- The Rust `DagDirective`, constructor by constructor (node ids as naturals). -/
def Directive.ofModel : Model.DagDirective → Directive
  | .BorrowLoad node => .borrowLoad node.val
  | .Produce node borrows => .produce node.val (borrows.val.map (·.val))
  | .Clone node source => .clone node.val source.val
  | .CloneProduce node source => .cloneProduce node.val source.val
  | .MoveProduce node source => .moveProduce node.val source.val
  | .CloneStore node source => .cloneStore node.val source.val
  | .MoveStore node source => .moveStore node.val source.val
  | .StoreRoot node => .storeRoot node.val
  | .BorrowedDrop node source => .borrowedDrop node.val source.val
  | .OwnedDrop node source => .ownedDrop node.val source.val
  | .Root source => .root source.val
  | .RootClone source => .rootClone source.val
  | .ScopeDrop source => .scopeDrop source.val

/-- A lowered DAG program's payload DAG and its private DAG ownership plan. -/
structure LoweredPlan where
  dag : Model.Dag
  plan : Model.DagOwnershipPlan

/-- The final plan is the payload DAG with the plan's directive vector, in order. -/
instance : OwnershipPlan LoweredPlan where
  dag p := p.dag
  directives p := (Model.planDirectives p.plan).map Directive.ofModel

/-- The actual final plan returned by one call of the extracted lowering, if it succeeded
(with a DAG proof, as `lower_dag_ownership` always builds). -/
def loweredPlan (dag : Model.Dag) (supply : Model.CapacityScopeSupply) : Option LoweredPlan :=
  match Model.lowerDagOwnership dag supply with
  | .ok (.Ok prog, _) => (Model.dagProgramParts prog).map fun (d, p) => ⟨d, p⟩
  | _ => none

/-- `loweredPlan` reads exactly a successful call: its plan is the returned program's DAG
plan, and its DAG is the lowered DAG itself (the program's payload). -/
theorem loweredPlan_spec {dag : Model.Dag} {supply : Model.CapacityScopeSupply}
    {pl : LoweredPlan} (h : loweredPlan dag supply = some pl) :
    pl.dag = dag ∧
    ∃ prog supply', Model.lowerDagOwnership dag supply = .ok (.Ok prog, supply') ∧
      Model.dagProgramParts prog = some (pl.dag, pl.plan) := by
  unfold loweredPlan at h
  split at h
  · rename_i prog supply' heq
    obtain ⟨⟨d, p⟩, hparts, rfl⟩ := Option.map_eq_some_iff.mp h
    obtain ⟨plan, hplan⟩ := Model.lowerDagOwnership_ok heq
    rw [hparts] at hplan
    cases hplan
    exact ⟨rfl, prog, supply', heq, hparts⟩
  · contradiction

/-- Every successful extracted lowering has a plan to decode: acceptance can only fail on the
decoded plan itself, never because `loweredPlan` discarded a returned program. -/
theorem loweredPlan_of_ok {dag : Model.Dag} {supply supply' : Model.CapacityScopeSupply}
    {prog : Model.DagOwnershipProgram}
    (h : Model.lowerDagOwnership dag supply = .ok (.Ok prog, supply')) :
    ∃ plan, loweredPlan dag supply = some ⟨dag, plan⟩ ∧
      Model.dagProgramParts prog = some (dag, plan) := by
  obtain ⟨plan, hplan⟩ := Model.lowerDagOwnership_ok h
  exact ⟨plan, by simp [loweredPlan, h, hplan], hplan⟩

/-- **Extracted ownership acceptance** (bridge contract item 5, logical ownership only).
If the extracted production lowering returns a program for `dag`, its actual final plan
decodes to `final` (entries borrowed through the payload's `Load` nodes, fresh local
owners, `RootClone` expanded into an explicit copy and its transfer), and the independent
reference ownership check accepts `final` from the initial state, then: `final` is exactly
the typed form of the decoded Rust directives; no owned root is transferred twice by the
returned directives; and `final` executes under the reference ownership machine with every
access permitted, returns its value semantics, terminates every local owner exactly once
and never terminates a borrowed entry. -/
theorem lowerDagOwnership_sound {dag : Model.Dag} {supply : Model.CapacityScopeSupply}
    {pl : LoweredPlan} {dt : DType} {entries : List (Option Nat)} {Γ Ω : Context}
    {final : OwnedCode ℚ Γ Ω}
    (hl : loweredPlan dag supply = some pl)
    (hd : decodeOwned dt entries Γ Ω pl = some final) :
    (pl.dag = dag ∧ ∃ prog supply', Model.lowerDagOwnership dag supply = .ok (.Ok prog, supply') ∧
      Model.dagProgramParts prog = some (pl.dag, pl.plan)) ∧
    (∃ is Ω', planInstrs dt entries Γ pl.dag
        ((Model.planDirectives pl.plan).map Directive.ofModel) = some (is, Ω') ∧
      final.encode = Instr.toOwned is) ∧
    (((Model.planDirectives pl.plan).map Directive.ofModel).filterMap
      Directive.transfer?).Nodup ∧
    (final.check (LaCaDiLECore.Ownership.initial Γ.length) = true →
      ∀ {R : Type} [CommSemiring R] [Algebra ℚ R] (ρ : Values R Γ),
        ∃ ended, final.run ρ (LaCaDiLECore.Ownership.initial Γ.length) =
            some (final.eval ρ, ended) ∧
          ended.Valid ∧ ended.live = ∅ ∧ ended.borrowed = Finset.range Γ.length ∧
          (∀ i, Γ.length ≤ i → i < ended.next → ended.ended.count i = 1) ∧
          (∀ i, i < Γ.length → ended.ended.count i = 0)) :=
  ⟨loweredPlan_spec hl, decodeOwned_sound hd⟩

/-- Proof-producing acceptance of the actual production pipeline: run the extracted
`grad_dag_checked forward output wrt`, run the extracted `lower_dag_ownership` on the
returned post-AD DAG (as production does), then decode the source, the post-AD artifact
and the returned final plan independently and require `Validation.acceptPlan` (reference
ownership check of the original plan, gradient correspondence, copy-normalized value
correspondence of the final plan). -/
def acceptLowered (dt : DType) (forward : Model.Dag) (output : Model.NodeId)
    (wrt : Slice Model.NodeId) (supply : Model.CapacityScopeSupply)
    (adWitness normalizationWitness : List (Option Nat)) : Bool :=
  match extractedResult (Model.gradDagChecked forward output wrt) with
  | none => false
  | some r =>
      match loweredPlan r.dag supply with
      | none => false
      | some pl => Validation.acceptPlan dt forward (wrt.val.map (·.val)) output.val r pl
          adWitness normalizationWitness

/-- **Composed pipeline theorem for the actual returned artifacts.** If `acceptLowered`
accepts, then the extracted AD returned `r`, the extracted ownership lowering of `r.dag`
returned a program whose plan `pl` is over `r.dag` itself, and for the independently
decoded source loss `p`, post-AD circuit `actual` and final plan `final` (over the artifact's own roots `Ω`): no owned
root is transferred twice; `final` executes under the reference ownership machine from
borrowed entries, returning its value semantics and terminating each local owner exactly
once and no entry; the real derivative of the source loss is the artifact's value, which is
the reference gradient; the final plan computes the artifact's own roots; and every final
root that is the returned gradient of entry `u` equals the reference gradient at `u`.
Logical ownership only (see the module header). -/
theorem acceptLowered_sound {dt : DType} {forward : Model.Dag} {output : Model.NodeId}
    {wrt : Slice Model.NodeId} {supply : Model.CapacityScopeSupply}
    {adWitness normalizationWitness : List (Option Nat)}
    (h : acceptLowered dt forward output wrt supply adWitness normalizationWitness = true) :
    ∃ (r : Model.GradResult) (pl : LoweredPlan) (prog : Model.DagOwnershipProgram)
      (supply' : Model.CapacityScopeSupply)
      (Γ : Context) (p : Program ℚ Γ []) (actual : Circuit ℚ Γ Γ) (Ω : Context)
      (rc : Circuit ℚ Γ Ω) (final : OwnedCode ℚ Γ Ω),
      Model.gradDagChecked forward output wrt = .ok (.Ok r) ∧
      Model.lowerDagOwnership r.dag supply = .ok (.Ok prog, supply') ∧
      pl.dag = r.dag ∧ Model.dagProgramParts prog = some (pl.dag, pl.plan) ∧
      decodeSource dt forward (wrt.val.map (·.val)) output.val = some ⟨Γ, p⟩ ∧
      decodeArtifact dt forward (wrt.val.map (·.val)) p r = some actual ∧
      decodeArtifactRoots dt forward (wrt.val.map (·.val)) p r = some ⟨Ω, rc⟩ ∧
      (∃ ids, Validation.planEntryIds forward (wrt.val.map (·.val)) pl = some ids ∧
        decodeOwned dt ids Γ Ω pl = some final) ∧
      (((Model.planDirectives pl.plan).map Directive.ofModel).filterMap
        Directive.transfer?).Nodup ∧
      ∀ ρ : Values ℝ Γ,
        (∃ ended, final.run ρ (LaCaDiLECore.Ownership.initial Γ.length) =
            some (final.eval ρ, ended) ∧
          ended.Valid ∧ ended.live = ∅ ∧ ended.borrowed = Finset.range Γ.length ∧
          (∀ i, Γ.length ≤ i → i < ended.next → ended.ended.count i = 1) ∧
          (∀ i, i < Γ.length → ended.ended.count i = 0)) ∧
        (∀ dρ : Values ℝ Γ, fderiv ℝ p.eval ρ dρ () = Values.dot dρ (actual.eval ρ)) ∧
        actual.eval ρ = p.grad ρ ∧
        final.eval ρ = rc.eval ρ ∧
        (∀ {s : Shape} (u : Var Γ s) (w : Var Ω s) (e g : Nat),
          (wrt.val.map (·.val))[u.offset]? = some e →
          (GradArtifact.grad r e).join = some g →
          (artifactRootIds r)[w.offset]? = some g →
          w.get (final.eval ρ) = u.get (p.grad ρ)) := by
  unfold acceptLowered at h
  split at h
  · contradiction
  · rename_i r hr
    split at h
    · contradiction
    · rename_i pl hpl
      have hgrad : Model.gradDagChecked forward output wrt = .ok (.Ok r) := by
        unfold extractedResult at hr
        split at hr
        · cases Option.some.inj hr; assumption
        · contradiction
      obtain ⟨hdag, prog, supply', hlow, hparts⟩ := loweredPlan_spec hpl
      obtain ⟨Γ, p, actual, Ω, rc, final, hs, ha, hroots, ⟨ids, hids, hf⟩, hall⟩ :=
        Validation.checkGradient_bridge h
      exact ⟨r, pl, prog, supply', Γ, p, actual, Ω, rc, final, hgrad, hlow, hdag, hparts, hs, ha,
        hroots, ⟨ids, hids, hf⟩, (decodeOwned_sound hf).2.1, hall⟩

/-! ## Relation to the Rust independent verifier

Production seals a DAG plan with `verify_ownership`, which (per the Rust source; not yet
checked against extracted code) for a DAG payload runs the
private `DagOwnershipPlan::verify`: it re-derives, node by node and root by root, the one
directive the lowering would emit (exact `DagDirective` equality), the owner origins
(`BTreeMap` equality) and the scope drops in ascending node order, and rejects any
leftover directive or unterminated owned producer.

That entry is not in the extracted model: admitting it needs the collection obligation of the
`SuccessorEdges` iterator's `size_hint` (Aeneas `-emit-collection-obligations`), which no bridge
theorem discharges yet. `acceptLowered` therefore requires only what is extracted: the
actual plan returned by `lower_dag_ownership`, decoded and accepted by the independent
reference check `OwnedCode.check`. That check is never replaced by the Rust verifier, and the
Rust verifier's acceptance is not assumed anywhere. The Rust verifier is strictly more
demanding on order: it fixes the scope-drop order and the owner-origin map, which the
reference check ignores. The converse question — whether the reference check accepts every
decodable plan the Rust verifier accepts — is open until the verifier is extracted. -/

end ChelisBridge.ExtractedOwnership
