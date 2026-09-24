import ChelisBridge.Numeric
import LaCaDiLECore.GraphCorrespondence

/-!
# Every binding of an IEEE run holds the IEEE value of its initializer

The IEEE analogue of `GraphCorrespondence.bindings_sound`: after running a prefix, each
binding's variable (in the final context) holds `opEval` of its initializer renamed into
the final context. This is the per-node fact a simulation of the Rust node loop consumes:
the Rust value of a node is computed from the values of its input nodes, which are
bindings of the same final context.
-/

namespace ChelisBridge.Numeric

open Aeneas.Std LaCaDiLECore

variable {f : BinaryFormat}

theorem opEval_rename {Γ Δ : Context} {s : Shape} (σ : Renaming Γ Δ)
    (op : Op (BinaryFloat f) Γ s) (ρ : Values (BinaryFloat f) Δ) :
    opEval (op.rename σ) ρ = opEval op (Renaming.read σ ρ) := by
  cases op <;> simp [Op.rename, opEval, sumLeaves]

theorem prefixEval_embedding {Γ Δ : Context} (p : Prefix (BinaryFloat f) Γ Δ)
    (ρ : Values (BinaryFloat f) Γ) : Renaming.read p.embedding (prefixEval p ρ) = ρ := by
  induction p with
  | nil => exact Renaming.read_id ρ
  | bind op rest ih => simp [Prefix.embedding, prefixEval, ih]

theorem get_embedding_ieee {Γ Δ : Context} {s : Shape} (p : Prefix (BinaryFloat f) Γ Δ)
    (ρ : Values (BinaryFloat f) Γ) (v : Var Γ s) :
    (p.embedding v).get (prefixEval p ρ) = v.get ρ := by
  rw [← Renaming.get_read, prefixEval_embedding]

theorem bindings_ieee {Γ Δ : Context} (p : Prefix (BinaryFloat f) Γ Δ)
    (ρ : Values (BinaryFloat f) Γ) (s : Shape) (op : Op (BinaryFloat f) Δ s) (v : Var Δ s)
    (h : ⟨s, op, v⟩ ∈ GraphCorrespondence.bindings p) :
    opEval op (prefixEval p ρ) = v.get (prefixEval p ρ) := by
  induction p with
  | nil => simp [GraphCorrespondence.bindings] at h
  | bind init rest ih =>
      simp only [GraphCorrespondence.bindings, List.mem_cons] at h
      rcases h with h | h
      · cases h
        simp [prefixEval, opEval_rename, Renaming.read_comp, prefixEval_embedding]
        rw [get_embedding_ieee]
        rfl
      · exact ih (Values.cons (opEval init ρ) ρ) op v h

end ChelisBridge.Numeric
