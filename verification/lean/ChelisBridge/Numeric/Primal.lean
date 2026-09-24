import ChelisBridge.Validation

/-!
# The primal root of the post-AD artifact

`grad_dag_checked` returns the combined forward + backward graph together with
`output_node`, the remapped forward output inside it; the artifact's Rust roots are the
primal output followed by the returned gradients. `checkGradient_bridge` relates the
gradient roots to `p.grad` but says nothing about the primal root.

`primalCheck` closes that gap with the same untrusting mechanism as the gradient check:
the source program `p` is read as a one-root circuit (`progCircuit`), the artifact circuit
is restricted to the root holding `output_node` (`selectRoot`), and
`GraphCorrespondence.check` must accept the pair under a supplied witness. A witness only
names actual artifact bindings; it cannot assert equality of values.

`acceptPrimal_sound`: acceptance on an actual `GradResult` implies that the artifact
root holding `output_node` evaluates, for every commutative `ℚ`-algebra of inputs (in
particular `ℚ` and `ℝ`), to the source loss `p.eval`.
-/

namespace ChelisBridge.Numeric

open Aeneas.Std LaCaDiLECore ChelisBridge

variable {C : Type}

/-- A program as a circuit with its single result as the only root. -/
def progCircuit {Γ : Context} {s : Shape} : Program C Γ s → Circuit C Γ [s]
  | .ret v => ⟨Γ, .nil, Renaming.cons v (fun {_} w => nomatch w : Renaming [] Γ)⟩
  | .letBind op body =>
      let c := progCircuit body
      ⟨c.context, .bind op c.code, c.result⟩

theorem progCircuit_eval [CommSemiring C] {R : Type} [CommSemiring R] [Algebra C R]
    {Γ : Context} {s : Shape} (p : Program C Γ s) (ρ : Values R Γ) :
    Var.here.get ((progCircuit p).eval ρ) = p.eval ρ := by
  induction p with
  | ret v => rfl
  | letBind op body ih =>
      simp only [progCircuit, Circuit.eval, Prefix.eval, Program.eval]
      exact ih _

/-- The circuit restricted to one of its roots. -/
def selectRoot {Γ Ω : Context} {s : Shape} (c : Circuit C Γ Ω) (w : Var Ω s) : Circuit C Γ [s] :=
  ⟨c.context, c.code, Renaming.cons (c.result w) (fun {_} v => nomatch v : Renaming [] c.context)⟩

theorem selectRoot_eval [CommSemiring C] {R : Type} [CommSemiring R] [Algebra C R]
    {Γ Ω : Context} {s : Shape} (c : Circuit C Γ Ω) (w : Var Ω s) (ρ : Values R Γ) :
    Var.here.get ((selectRoot c w).eval ρ) = w.get (c.eval ρ) := by
  simp only [selectRoot, Circuit.eval, Renaming.get_read]
  rfl

/-- The primal check: root `k` of the artifact circuit corresponds to the source program. -/
def primalCheck [DecidableEq C] {Γ Ω : Context} (p : Program C Γ []) (rc : Circuit C Γ Ω)
    (k : Nat) (witness : List (Option Nat)) : Bool :=
  match Var.decode Ω [] k with
  | none => false
  | some w => GraphCorrespondence.check (progCircuit p) (selectRoot rc w) witness

theorem primalCheck_sound [CommSemiring C] [DecidableEq C] {Γ Ω : Context}
    {p : Program C Γ []} {rc : Circuit C Γ Ω} {k : Nat} {witness : List (Option Nat)}
    (h : primalCheck p rc k witness = true) :
    ∃ w : Var Ω [], w.offset = k ∧
      ∀ {R : Type} [CommSemiring R] [Algebra C R] (ρ : Values R Γ), w.get (rc.eval ρ) = p.eval ρ := by
  unfold primalCheck at h
  split at h
  · contradiction
  · rename_i w hw
    refine ⟨w, Var.offset_of_decode hw, fun ρ => ?_⟩
    have hc := GraphCorrespondence.check_sound _ _ witness h ρ
    rw [← selectRoot_eval, ← hc, progCircuit_eval]

/-- Acceptance of the primal root of an actual `GradResult`: the root listing
`output_node` is checked against the decoded source program. -/
def acceptPrimal (dt : DType) (source : Model.Dag) (entries : List Nat) (loss : Nat)
    (r : Model.GradResult) (witness : List (Option Nat)) : Bool :=
  match decodeSource dt source entries loss with
  | none => false
  | some ⟨_, p⟩ =>
      match decodeArtifactRoots dt source entries p r with
      | none => false
      | some ⟨_, rc⟩ =>
          match (artifactRootIds r).idxOf? r.output_node.val with
          | none => false
          | some k => primalCheck p rc k witness

/-- **Primal root of the extracted AD result.** If `acceptPrimal` accepts, the artifact
root holding Rust's `output_node` evaluates to the source loss for every input algebra. -/
theorem acceptPrimal_sound {dt : DType} {source : Model.Dag} {entries : List Nat} {loss : Nat}
    {r : Model.GradResult} {witness : List (Option Nat)}
    (h : acceptPrimal dt source entries loss r witness = true) :
    ∃ (Γ : Context) (p : Program ℚ Γ []) (Ω : Context) (rc : Circuit ℚ Γ Ω) (w : Var Ω []),
      decodeSource dt source entries loss = some ⟨Γ, p⟩ ∧
      decodeArtifactRoots dt source entries p r = some ⟨Ω, rc⟩ ∧
      (artifactRootIds r)[w.offset]? = some r.output_node.val ∧
      ∀ {R : Type} [CommSemiring R] [Algebra ℚ R] (ρ : Values R Γ), w.get (rc.eval ρ) = p.eval ρ := by
  unfold acceptPrimal at h
  split at h
  · contradiction
  · rename_i Γ p hsrc
    split at h
    · contradiction
    · rename_i Ω rc hrc
      split at h
      · contradiction
      · rename_i k hk
        obtain ⟨w, hwk, hw⟩ := primalCheck_sound h
        refine ⟨Γ, p, Ω, rc, w, hsrc, hrc, ?_, fun ρ => hw ρ⟩
        rw [hwk]
        exact List.idxOf?_eq_some_iff.mp hk |>.2.1 ▸ (List.getElem?_eq_getElem _)
