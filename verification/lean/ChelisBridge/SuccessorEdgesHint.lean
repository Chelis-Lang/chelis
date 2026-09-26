import ChelisBridge.Model

/-!
# The `SuccessorEdges` collection obligation

`chelis_ir::ownership::ir::SuccessorEdges` (the successor-edge iterator of the host ownership
IR, collected into `Vec`s by the ownership verifier and scheduler) overrides `size_hint`.
Aeneas admits those collections (`-emit-collection-obligations`) only with the obligation
that the extracted `size_hint` is total whenever the inner arms iterator's is: the native
collectors call it for capacity, and the collected values do not depend on it. This module
proves that obligation with exactly the emitted proposition as its statement, so the closure
audit lists it under `discharged`.
-/

open Aeneas Aeneas.Std Result
open_chelis_model

namespace ChelisBridge.SuccessorEdgesHint

theorem successorEdges_size_hint_total :
    chelis_ir.ownership.ir.SuccessorEdges.Insts.CoreIterTraitsIteratorIterator.size_hint_total := by
  intro E A inst h s
  cases s with
  | Arms arms =>
    obtain ⟨r, hr⟩ := h arms
    exact ⟨r, by
      simp [chelis_ir.ownership.ir.SuccessorEdges.Insts.CoreIterTraitsIteratorIterator.size_hint,
        hr]⟩
  | Fixed first second =>
    have bit : ∀ b : Bool, (core.convert.num.FromUsizeBool.from b).val ≤ 1 := by
      intro b; cases b <;> simp [core.convert.num.FromUsizeBool.from]
    have := bit first.isSome
    have := bit second.isSome
    have hmax : (core.convert.num.FromUsizeBool.from first.isSome).val +
        (core.convert.num.FromUsizeBool.from second.isSome).val ≤ UScalar.max .Usize := by
      scalar_tac
    obtain ⟨z, hz, -⟩ := WP.spec_imp_exists (UScalar.add_spec hmax)
    exact ⟨(z, some z), by
      simp [chelis_ir.ownership.ir.SuccessorEdges.Insts.CoreIterTraitsIteratorIterator.size_hint,
        core.option.Option.is_some, lift, hz]⟩

end ChelisBridge.SuccessorEdgesHint
