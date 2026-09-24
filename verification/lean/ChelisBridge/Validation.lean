import ChelisBridge.Ownership
import LaCaDiLECore.GraphCorrespondence

/-!
# Bridge theorems: decoded Rust graphs against the LaCaDiLE reference

* `needs p k` is reachability of context offset `k` from the result of a decoded source
  program through *value ports* only. Shape witnesses (the extra `Expand` inputs) are
  never ports of the decoded program, so they cannot make an entry "reachable".
  `grad_get_of_disconnected`: the reference gradient of an unreachable entry is zero.
* `totalize` turns Rust's sparse gradient map (requested entry ↦ gradient node) into
  total roots: a present entry uses the returned node; a correctly shaped zero constant is
  emitted only for an entry proved disconnected; a missing reachable gradient makes
  totalization fail (`totalize_eq_none_of_missing`) — never a default.
* `checkedGrad_bridge`: for any source `Dag` and any post-AD artifact that decode to
  `p : Program ℚ Γ []` and `actual : Circuit ℚ Γ Γ`, a successful
  `GraphCorrespondence.check p.compileGrad actual witness` yields the real-derivative
  theorem for the *actual* circuit.

All statements are parametric in the artifact type through `GradArtifact`; the instance
for the extracted `GradResult` reads `grad_nodes` with the extracted `UnordMap::get`
(`Model.NodeMap.get`). A lookup that does not return is an error, never an absence.

Hypotheses not discharged here (see the bridge contract): the zero-filled entries are
exactly the unreachable ones only by the decoder's definition of reachability; no claim is
made about the Rust evaluator's numerical output (that needs the exact-execution predicate
of `ChelisBridge.Float` per executed step and a proof about the extracted evaluator).
-/

namespace ChelisBridge

open Aeneas.Std LaCaDiLECore

section Disconnection

variable {C : Type}

/-- Whether an operation reads context offset `k` through one of its value ports. -/
def opReads {Γ : Context} {s : Shape} : Op C Γ s → Nat → Bool
  | .const _, _ => false
  | .copy v, k => v.offset == k
  | .add v w, k => v.offset == k || w.offset == k
  | .mul v w, k => v.offset == k || w.offset == k
  | .sum _ v, k => v.offset == k
  | .insert _ v, k => v.offset == k
  | .expand _ _ _ v, k => v.offset == k

/-- `opReads` is membership in the operation's value ports (`ports`, order-free). -/
theorem opReads_iff {Γ : Context} {s : Shape} (op : Op C Γ s) (k : Nat) :
    opReads op k = true ↔ k ∈ ports op.encode := by
  cases op <;> (simp [opReads, ports, Op.encode]; try omega)

/-- Reachability of context offset `k` from the result through value ports. -/
def needs {Γ : Context} {s : Shape} : Program C Γ s → Nat → Bool
  | .ret v, k => v.offset == k
  | .letBind op body, k => needs body (k + 1) || (needs body 0 && opReads op k)

section Algebra

variable {R : Type} [CommSemiring R]

theorem Var.get_zero {Γ : Context} {s : Shape} (v : Var Γ s) :
    v.get (0 : Values R Γ) = 0 := rfl

theorem Var.get_add {Γ : Context} {s : Shape} (v : Var Γ s) (ρ σ : Values R Γ) :
    v.get (ρ + σ) = v.get ρ + v.get σ := rfl

theorem Var.get_scatter_of_ne {Γ : Context} {t : Shape} (w : Var Γ t) (g : Tensor R t) :
    ∀ {s : Shape} (v : Var Γ s), v.offset ≠ w.offset → v.get (w.scatter g) = 0 := by
  induction w with
  | here =>
      intro s v hv
      cases v with
      | here => exact absurd rfl hv
      | there v => exact Var.get_zero v
  | there w ih =>
      intro s v hv
      cases v with
      | here => rfl
      | there v => exact ih g v (by simp only [Var.offset] at hv; omega)

theorem Op.vjp_get_of_not_reads {Γ : Context} {s t : Shape} (op : Op C Γ s)
    (ρ : Values R Γ) (g : Tensor R s) (v : Var Γ t) (h : opReads op v.offset = false) :
    v.get (op.vjp ρ g) = 0 := by
  cases op with
  | const x => rfl
  | copy w =>
      simp only [opReads, beq_eq_false_iff_ne, ne_eq] at h
      exact Var.get_scatter_of_ne w g v (fun e => h e.symm)
  | add w₁ w₂ =>
      simp only [opReads, Bool.or_eq_false_iff, beq_eq_false_iff_ne, ne_eq] at h
      simp only [Op.vjp, Var.get_add, Var.get_scatter_of_ne w₁ _ v (fun e => h.1 e.symm),
        Var.get_scatter_of_ne w₂ _ v (fun e => h.2 e.symm), add_zero]
  | mul w₁ w₂ =>
      simp only [opReads, Bool.or_eq_false_iff, beq_eq_false_iff_ne, ne_eq] at h
      simp only [Op.vjp, Var.get_add, Var.get_scatter_of_ne w₁ _ v (fun e => h.1 e.symm),
        Var.get_scatter_of_ne w₂ _ v (fun e => h.2 e.symm), add_zero]
  | sum a w =>
      simp only [opReads, beq_eq_false_iff_ne, ne_eq] at h
      exact Var.get_scatter_of_ne w _ v (fun e => h e.symm)
  | insert a w =>
      simp only [opReads, beq_eq_false_iff_ne, ne_eq] at h
      exact Var.get_scatter_of_ne w _ v (fun e => h e.symm)
  | expand pre post n w =>
      simp only [opReads, beq_eq_false_iff_ne, ne_eq] at h
      exact Var.get_scatter_of_ne w _ v (fun e => h e.symm)

/-- Reverse evaluation assigns zero cotangent to every binding the result does not reach
through value ports. -/
theorem vjp_get_of_not_needs [CommSemiring C] [Algebra C R] {Γ : Context} {s : Shape}
    (p : Program C Γ s) :
    ∀ (ρ : Values R Γ) (g : Tensor R s) {t : Shape} (v : Var Γ t),
      needs p v.offset = false → v.get (p.vjp ρ g) = 0 := by
  induction p with
  | ret w =>
      intro ρ g t v h
      simp only [needs, beq_eq_false_iff_ne, ne_eq] at h
      exact Var.get_scatter_of_ne w g v (fun e => h e.symm)
  | letBind op body ih =>
      intro ρ g t v h
      simp only [needs, Bool.or_eq_false_iff, Bool.and_eq_false_iff] at h
      obtain ⟨h1, h2⟩ := h
      simp only [Program.vjp, Var.get_add]
      have htail : v.get (Values.tail (body.vjp (Values.cons (op.eval ρ) ρ) g)) = 0 :=
        ih _ g (Var.there v) (by simpa [Var.offset] using h1)
      rw [htail, add_zero]
      rcases h2 with h2 | h2
      · have hhead : Values.head (body.vjp (Values.cons (op.eval ρ) ρ) g) = 0 :=
          ih _ g Var.here (by simpa [Var.offset] using h2)
        rw [hhead, Op.vjp_zero]
        rfl
      · exact Op.vjp_get_of_not_reads op ρ _ v h2

/-- **Zero filling is forced.** The reference all-entry gradient vanishes at every entry
that the loss does not reach through value ports. -/
theorem grad_get_of_disconnected [CommSemiring C] [Algebra C R] {Γ : Context}
    (p : Program C Γ []) (ρ : Values R Γ) {t : Shape} (v : Var Γ t)
    (h : needs p v.offset = false) : v.get (p.grad ρ) = 0 :=
  vjp_get_of_not_needs p ρ _ v h

end Algebra

end Disconnection

section Totalize

/-- Where a total root comes from. -/
inductive RootSource where
  | present (node : Nat)
  | zero
  deriving DecidableEq, Repr

/-- The source of one requested entry's root: the returned gradient if present; a zero
only for a disconnected entry; otherwise an error. -/
def rootSource (lookup : Nat → Option Nat) (disconnected : Nat → Bool) (e j : Nat) :
    Option RootSource :=
  match lookup e with
  | some g => some (.present g)
  | none => if disconnected j then some .zero else none

/-- Classify each requested entry (in order, starting at position `j`): a present
gradient, or a zero only for a proved-disconnected entry. A missing gradient for any
other entry is an error. -/
def classifyRoots (lookup : Nat → Option Nat) (disconnected : Nat → Bool) :
    List Nat → Nat → Option (List RootSource)
  | [], _ => some []
  | e :: rest, j => (rootSource lookup disconnected e j).bind fun r =>
      (classifyRoots lookup disconnected rest (j + 1)).map (r :: ·)

theorem rootSource_present {lookup : Nat → Option Nat} {disconnected : Nat → Bool}
    {e j g : Nat} (hg : lookup e = some g) :
    rootSource lookup disconnected e j = some (.present g) := by
  simp [rootSource, hg]

theorem rootSource_missing {lookup : Nat → Option Nat} {disconnected : Nat → Bool}
    {e j : Nat} {r : RootSource} (hg : lookup e = none)
    (h : rootSource lookup disconnected e j = some r) :
    disconnected j = true ∧ r = .zero := by
  simp only [rootSource, hg] at h
  cases hd : disconnected j
  · simp [hd] at h
  · simp only [hd, if_true, Option.some.injEq] at h
    exact ⟨rfl, h.symm⟩

theorem classifyRoots_eq_none (lookup : Nat → Option Nat) (disconnected : Nat → Bool) :
    ∀ (entries : List Nat) (j0 j e : Nat), entries[j]? = some e → lookup e = none →
      disconnected (j0 + j) = false → classifyRoots lookup disconnected entries j0 = none
  | [], _, _, _, he, _, _ => by simp at he
  | e' :: rest, j0, 0, e, he, hmiss, hd => by
      simp only [List.getElem?_cons_zero, Option.some.injEq] at he
      subst he
      have : rootSource lookup disconnected e' j0 = none := by
        simp [rootSource, hmiss, show disconnected j0 = false by simpa using hd]
      simp [classifyRoots, this]
  | e' :: rest, j0, j + 1, e, he, hmiss, hd => by
      simp only [List.getElem?_cons_succ] at he
      have := classifyRoots_eq_none lookup disconnected rest (j0 + 1) j e he hmiss
        (by rw [show j0 + 1 + j = j0 + (j + 1) by omega]; exact hd)
      simp only [classifyRoots, this]
      cases rootSource lookup disconnected e' j0 <;> simp

theorem classifyRoots_spec (lookup : Nat → Option Nat) (disconnected : Nat → Bool) :
    ∀ (entries : List Nat) (j0 : Nat) (rs : List RootSource),
      classifyRoots lookup disconnected entries j0 = some rs →
        rs.length = entries.length ∧
        ∀ (j e : Nat), entries[j]? = some e →
          (∀ g, lookup e = some g → rs[j]? = some (RootSource.present g)) ∧
          (lookup e = none → disconnected (j0 + j) = true ∧ rs[j]? = some RootSource.zero)
  | [], _, rs, h => by
      simp only [classifyRoots, Option.some.injEq] at h; subst h; simp
  | e' :: rest, j0, rs, h => by
      simp only [classifyRoots, Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
      obtain ⟨r, hr, rs', hrs', rfl⟩ := h
      obtain ⟨hlen, hspec⟩ := classifyRoots_spec lookup disconnected rest (j0 + 1) rs' hrs'
      refine ⟨by simp [hlen], ?_⟩
      intro j e he
      cases j with
      | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at he
          subst he
          refine ⟨fun g hg => ?_, fun hg => ?_⟩
          · rw [rootSource_present hg] at hr; cases Option.some.inj hr; rfl
          · obtain ⟨hd, rfl⟩ := rootSource_missing hg hr
            exact ⟨by simpa using hd, rfl⟩
      | succ j =>
          simp only [List.getElem?_cons_succ] at he ⊢
          obtain ⟨h1, h2⟩ := hspec j e he
          refine ⟨h1, fun hg => ?_⟩
          obtain ⟨hd, hz⟩ := h2 hg
          exact ⟨by rw [show j0 + (j + 1) = j0 + 1 + j by omega]; exact hd, hz⟩

/-- The zero constant emitted for a disconnected entry of shape `s`. -/
def zeroConst (s : Shape) : GraphSyntax.Operation ℚ := .const s (List.replicate s.prod 0)

/-- Assign an owner to every root: the Rust gradient node's owner, or a fresh zero
constant appended after the decoded bindings. -/
def totalizeAux (st : State) :
    List (Shape × RootSource) → List (GraphSyntax.Operation ℚ) →
      Option (List (GraphSyntax.Operation ℚ) × List Nat)
  | [], zs => some (zs, [])
  | (_, .present g) :: rest, zs => do
      let sl ← st.slot g
      if sl.owner < st.len then
        let (zs', owners) ← totalizeAux st rest zs
        pure (zs', sl.owner :: owners)
      else none
  | (s, .zero) :: rest, zs => do
      let (zs', owners) ← totalizeAux st rest (zs ++ [zeroConst s])
      pure (zs', (st.len + zs.length) :: owners)

/-- The target of a totalization: the decoded operations, then the appended zero
constants; each root is the offset of its owner in the final context. -/
def totalTarget (st : State) (zs : List (GraphSyntax.Operation ℚ)) (owners : List Nat) :
    GraphSyntax.Target ℚ :=
  ⟨st.ops ++ zs, owners.map (fun o => st.len + zs.length - 1 - o)⟩

/-- Appended zero constants and root owners of a totalization. -/
def totalizeParts (Γ : Context) (lookup : Nat → Option Nat) (disconnected : Nat → Bool)
    (entries : List Nat) (st : State) :
    Option (List (GraphSyntax.Operation ℚ) × List Nat) :=
  (classifyRoots lookup disconnected entries 0).bind fun srcs => totalizeAux st (Γ.zip srcs) []

/-- Total gradient roots over the ordered entries `Γ` from a sparse map. -/
def totalize (Γ : Context) (lookup : Nat → Option Nat) (disconnected : Nat → Bool)
    (entries : List Nat) (st : State) : Option (GraphSyntax.Target ℚ) :=
  (totalizeParts Γ lookup disconnected entries st).map fun r => totalTarget st r.1 r.2

theorem totalizeAux_spec (st : State) :
    ∀ (items : List (Shape × RootSource)) (zs zs' : List (GraphSyntax.Operation ℚ))
      (owners : List Nat), totalizeAux st items zs = some (zs', owners) →
      owners.length = items.length ∧ zs <+: zs' ∧
      ∀ (j : Nat) (s : Shape) (src : RootSource), items[j]? = some (s, src) →
        (∀ g, src = RootSource.present g → ∃ sl, st.slot g = some sl ∧ sl.owner < st.len ∧
          owners[j]? = some sl.owner) ∧
        (src = RootSource.zero → ∃ t, owners[j]? = some (st.len + t) ∧
          zs'[t]? = some (zeroConst s))
  | [], zs, zs', owners, h => by
      simp only [totalizeAux, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨rfl, List.prefix_refl _, by simp⟩
  | (s0, .present g0) :: rest, zs, zs', owners, h => by
      simp only [totalizeAux, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
      obtain ⟨sl, hsl, h⟩ := h
      split at h
      · rename_i hlt
        simp only [Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at h
        obtain ⟨⟨zs1, os1⟩, hrec, h⟩ := h
        simp only [Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        obtain ⟨hlen, hpre, hspec⟩ := totalizeAux_spec st rest zs zs1 os1 hrec
        refine ⟨by simp [hlen], hpre, ?_⟩
        intro j s src hj
        cases j with
        | zero =>
            simp only [List.getElem?_cons_zero, Option.some.injEq, Prod.mk.injEq] at hj
            obtain ⟨rfl, rfl⟩ := hj
            refine ⟨fun g hg => ?_, fun hz => ?_⟩
            · cases hg; exact ⟨sl, hsl, hlt, rfl⟩
            · cases hz
        | succ j =>
            simp only [List.getElem?_cons_succ] at hj ⊢
            exact hspec j s src hj
      · contradiction
  | (s0, .zero) :: rest, zs, zs', owners, h => by
      simp only [totalizeAux, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at h
      obtain ⟨⟨zs1, os1⟩, hrec, h⟩ := h
      simp only [Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      obtain ⟨hlen, hpre, hspec⟩ := totalizeAux_spec st rest _ zs1 os1 hrec
      refine ⟨by simp [hlen], (List.prefix_append _ _).trans hpre, ?_⟩
      intro j s src hj
      cases j with
      | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq, Prod.mk.injEq] at hj
          obtain ⟨rfl, rfl⟩ := hj
          refine ⟨fun g hg => ?_, fun _ => ⟨zs.length, rfl, ?_⟩⟩
          · cases hg
          · obtain ⟨u, rfl⟩ := hpre
            simp
      | succ j =>
          simp only [List.getElem?_cons_succ] at hj ⊢
          exact hspec j s src hj

theorem totalizeParts_eq_none_of_missing {Γ : Context} {lookup : Nat → Option Nat}
    {disconnected : Nat → Bool} {entries : List Nat} {st : State} {j e : Nat}
    (hj : entries[j]? = some e) (hmiss : lookup e = none) (hreach : disconnected j = false) :
    totalizeParts Γ lookup disconnected entries st = none := by
  have := classifyRoots_eq_none lookup disconnected entries 0 j e hj hmiss (by simpa using hreach)
  simp [totalizeParts, this]

/-- **A missing reachable gradient is an error, never a default.** -/
theorem totalize_eq_none_of_missing {Γ : Context} {lookup : Nat → Option Nat}
    {disconnected : Nat → Bool} {entries : List Nat} {st : State} {j e : Nat}
    (hj : entries[j]? = some e) (hmiss : lookup e = none) (hreach : disconnected j = false) :
    totalize Γ lookup disconnected entries st = none := by
  simp [totalize, totalizeParts_eq_none_of_missing (Γ := Γ) (st := st) hj hmiss hreach]

/-- A present gradient's root owner is the owner of the returned Rust node. -/
theorem totalizeParts_present {Γ : Context} {lookup : Nat → Option Nat}
    {disconnected : Nat → Bool} {entries : List Nat} {st : State}
    {zs : List (GraphSyntax.Operation ℚ)} {owners : List Nat}
    (h : totalizeParts Γ lookup disconnected entries st = some (zs, owners))
    {j e g : Nat} {s : Shape} (he : entries[j]? = some e) (hs : Γ[j]? = some s)
    (hg : lookup e = some g) :
    ∃ sl, st.slot g = some sl ∧ sl.owner < st.len ∧ owners[j]? = some sl.owner := by
  simp only [totalizeParts, Option.bind_eq_some_iff] at h
  obtain ⟨srcs, hsrcs, haux⟩ := h
  obtain ⟨_, hcl⟩ := classifyRoots_spec lookup disconnected entries 0 srcs hsrcs
  obtain ⟨_, _, hspec⟩ := totalizeAux_spec st (Γ.zip srcs) [] zs owners haux
  have hitem : (Γ.zip srcs)[j]? = some (s, .present g) := by
    simp [List.getElem?_zip_eq_some, hs, (hcl j e he).1 g hg]
  exact (hspec j s _ hitem).1 g rfl

/-- On success, every missing gradient belongs to a proved-disconnected entry and its root
is a zero constant of the entry's shape appended after the decoded bindings; every present
gradient's root is the returned Rust node, with `owner + offset + 1` equal to the final
context length. -/
theorem totalize_spec {Γ : Context} {lookup : Nat → Option Nat} {disconnected : Nat → Bool}
    {entries : List Nat} {st : State} {tgt : GraphSyntax.Target ℚ}
    (h : totalize Γ lookup disconnected entries st = some tgt)
    {j e : Nat} {s : Shape} (he : entries[j]? = some e) (hs : Γ[j]? = some s) :
    ∃ zs, tgt.operations = st.ops ++ zs ∧
      (∀ g, lookup e = some g → ∃ sl i, st.slot g = some sl ∧ tgt.roots[j]? = some i ∧
        sl.owner + i + 1 = st.len + zs.length) ∧
      (lookup e = none → disconnected j = true ∧ ∃ t i, tgt.roots[j]? = some i ∧
        (st.len + t) + i + 1 = st.len + zs.length ∧ zs[t]? = some (zeroConst s)) := by
  simp only [totalize, totalizeParts, Option.map_eq_some_iff, Option.bind_eq_some_iff] at h
  obtain ⟨⟨zs, owners⟩, ⟨srcs, hsrcs, haux⟩, rfl⟩ := h
  obtain ⟨hlen, hcl⟩ := classifyRoots_spec lookup disconnected entries 0 srcs hsrcs
  obtain ⟨holen, _, hspec⟩ := totalizeAux_spec st (Γ.zip srcs) [] zs owners haux
  obtain ⟨hpres, hzero⟩ := hcl j e he
  refine ⟨zs, rfl, ?_, ?_⟩
  · intro g hg
    have hsrc := hpres g hg
    have hitem : (Γ.zip srcs)[j]? = some (s, .present g) := by
      simp [List.getElem?_zip_eq_some, hs, hsrc]
    obtain ⟨sl, hsl, hlt, ho⟩ := (hspec j s _ hitem).1 g rfl
    exact ⟨sl, st.len + zs.length - 1 - sl.owner, hsl,
      by simp [totalTarget, List.getElem?_map, ho],
      by omega⟩
  · intro hg
    obtain ⟨hd, hsrc⟩ := hzero hg
    have hitem : (Γ.zip srcs)[j]? = some (s, .zero) := by
      simp [List.getElem?_zip_eq_some, hs, hsrc]
    obtain ⟨t, ho, hz⟩ := (hspec j s _ hitem).2 rfl
    have htl : t < zs.length := by
      rcases Nat.lt_or_ge t zs.length with h | h
      · exact h
      · rw [List.getElem?_eq_none h] at hz; contradiction
    exact ⟨by simpa using hd, t, st.len + zs.length - 1 - (st.len + t),
      by simp [totalTarget, List.getElem?_map, ho], by omega, hz⟩

end Totalize

section Artifact

/-- Interface to the value returned by the extracted AD function: the post-AD graph and
the sparse map from requested entry node ids to gradient node ids. `grad a e` is
`some (some g)` for a returned gradient, `some none` for a key reported absent, and `none`
when the lookup itself does not return a value (so it can never be read as absence). -/
class GradArtifact (α : Type) where
  dag : α → Model.Dag
  grad : α → Nat → Option (Option Nat)

/-- Look up a gradient with the extracted `UnordMap::get` itself. -/
def gradLookup (r : Model.GradResult) (e : Nat) : Option (Option Nat) :=
  if h : e < 2 ^ UScalarTy.Usize.numBits then
    match Model.NodeMap.get r.grad_nodes (Usize.ofNatCore e h) with
    | .ok o => some (o.map (·.val))
    | _ => none
  else none

instance : GradArtifact Model.GradResult where
  dag r := r.dag
  grad := gradLookup

/-- The successful value of the extracted `grad_dag_checked`, if any. -/
def extractedResult :
    Result (core.result.Result Model.GradResult Model.AdError) → Option Model.GradResult
  | .ok (.Ok r) => some r
  | _ => none

variable {α : Type} [GradArtifact α]

/-- The input names of the ordered entries of a graph (every entry is a `Load`). -/
def entryNames (dag : Model.Dag) (entries : List Nat) : Option (List Model.LoadStoreName) :=
  entries.mapM fun e => (dag.nodes.val[e]?).bind fun n => Model.loadName? n.op

/-- The `Load` node of each named entry in another graph (`none` when that graph has no
`Load` of the name, e.g. after AD pruned an unused input). Inputs are bound by name, so the
graph's `Load` names must be distinct. -/
def entryIdsIn (dag : Model.Dag) (names : List Model.LoadStoreName) :
    Option (List (Option Nat)) :=
  let loads := dag.nodes.val.filterMap fun n => (Model.loadName? n.op).map fun l => (l, n.id.val)
  if (loads.map (·.1)).Nodup then some (names.map fun l => loads.lookup l) else none

/-- The decoded artifact: every lookup must return; the artifact graph is decoded over the
source's entry context, its `Load`s identified with the source entries by input name; its
sparse gradients are totalized against the decoded source (`needs`). Result: the node
state, the appended zero constants and the root owners. -/
def artifactParts (dt : DType) (source : Model.Dag) (entries : List Nat) {Γ : Context}
    (p : Program ℚ Γ []) (art : α) :
    Option (State × List (GraphSyntax.Operation ℚ) × List Nat) :=
  if entries.all (fun e => (GradArtifact.grad art e).isSome) then
    ((entryNames source entries).bind (entryIdsIn (GradArtifact.dag art))).bind fun ids =>
      (decodeGraphOver dt (GradArtifact.dag art) ids Γ).bind fun st =>
        (totalizeParts Γ (fun e => (GradArtifact.grad art e).join) (fun j => !needs p j)
          entries st).map fun zo => (st, zo.1, zo.2)
  else none

/-- Total post-AD target (one root per entry). -/
def artifactTarget (dt : DType) (source : Model.Dag) (entries : List Nat) {Γ : Context} (p : Program ℚ Γ [])
    (art : α) : Option (GraphSyntax.Target ℚ) :=
  (artifactParts dt source entries p art).map fun x => totalTarget x.1 x.2.1 x.2.2

/-- The decoded post-AD circuit `Circuit ℚ Γ Γ` (one total gradient root per entry). -/
def decodeArtifact (dt : DType) (source : Model.Dag) (entries : List Nat) {Γ : Context} (p : Program ℚ Γ [])
    (art : α) : Option (Circuit ℚ Γ Γ) :=
  (artifactTarget dt source entries p art).bind fun tgt => tgt.decode Γ Γ

/-- The artifact's own ordered Rust roots (the primal output, then returned gradients). -/
def artifactRootIds (art : α) : List Nat := (GradArtifact.dag art).roots.val.map (·.val)

/-- Offsets of Rust nodes in the context extended by `extra` appended bindings. -/
def rootOffsets (st : State) (extra : Nat) (roots : List Nat) : Option (List Nat) :=
  roots.mapM fun r => (st.slot r).bind fun sl =>
    if sl.owner < st.len then some (st.len + extra - 1 - sl.owner) else none

/-- The same decoded operations (including the appended zero constants) with the artifact's
own ordered Rust roots, so both views share one prefix. -/
def artifactRoots (dt : DType) (source : Model.Dag) (entries : List Nat) {Γ : Context} (p : Program ℚ Γ [])
    (art : α) : Option (Context × GraphSyntax.Target ℚ) :=
  (artifactParts dt source entries p art).bind fun x =>
    ((artifactRootIds art).mapM x.1.shapeOf).bind fun Ω =>
      (rootOffsets x.1 x.2.1.length (artifactRootIds art)).map fun offs =>
        (Ω, ⟨x.1.ops ++ x.2.1, offs⟩)

/-- The decoded post-AD circuit over the artifact's own roots, `Circuit ℚ Γ Ω`. -/
def decodeArtifactRoots (dt : DType) (source : Model.Dag) (entries : List Nat) {Γ : Context} (p : Program ℚ Γ [])
    (art : α) : Option (Sigma (Circuit ℚ Γ)) :=
  (artifactRoots dt source entries p art).bind fun x => (x.2.decode Γ x.1).map fun c => ⟨x.1, c⟩

theorem artifactParts_spec {dt : DType} {source : Model.Dag} {entries : List Nat}
    {Γ : Context} {p : Program ℚ Γ []} {art : α} {st : State}
    {zs : List (GraphSyntax.Operation ℚ)} {owners : List Nat}
    (h : artifactParts dt source entries p art = some (st, zs, owners)) :
    totalizeParts Γ (fun e => (GradArtifact.grad art e).join) (fun j => !needs p j)
      entries st = some (zs, owners) := by
  unfold artifactParts at h
  split at h
  · cases hi : (entryNames source entries).bind (entryIdsIn (GradArtifact.dag art)) with
    | none => rw [hi] at h; contradiction
    | some ids =>
        rw [hi, Option.bind_some] at h
        cases hg : decodeGraphOver dt (GradArtifact.dag art) ids Γ with
        | none => rw [hg] at h; contradiction
        | some st' =>
            rw [hg, Option.bind_some] at h
            obtain ⟨⟨zs', owners'⟩, ht, he⟩ := Option.map_eq_some_iff.mp h
            simp only [Prod.mk.injEq] at he
            obtain ⟨rfl, rfl, rfl⟩ := he
            exact ht
  · contradiction

/-- **Missing reachable gradients are rejected.** If requested entry `j` reaches the loss
through value ports of the decoded source and the artifact returns no gradient for it
(absent or lookup failure), the artifact does not decode. -/
theorem artifactParts_eq_none_of_missing {dt : DType} {source : Model.Dag}
    {entries : List Nat} {Γ : Context}
    {p : Program ℚ Γ []} {art : α} {j e : Nat} (he : entries[j]? = some e)
    (hmiss : (GradArtifact.grad art e).join = none) (hreach : needs p j = true) :
    artifactParts dt source entries p art = none := by
  unfold artifactParts
  split
  · cases (entryNames source entries).bind (entryIdsIn (GradArtifact.dag art)) with
    | none => rfl
    | some ids =>
        rw [Option.bind_some]
        cases decodeGraphOver dt (GradArtifact.dag art) ids Γ with
        | none => rfl
        | some st =>
            rw [Option.bind_some, totalizeParts_eq_none_of_missing he hmiss (by simp [hreach])]
            rfl
  · rfl

theorem decodeArtifact_eq_none_of_missing {dt : DType} {source : Model.Dag} {entries : List Nat} {Γ : Context}
    {p : Program ℚ Γ []} {art : α} {j e : Nat} (he : entries[j]? = some e)
    (hmiss : (GradArtifact.grad art e).join = none) (hreach : needs p j = true) :
    decodeArtifact dt source entries p art = none := by
  simp [decodeArtifact, artifactTarget, artifactParts_eq_none_of_missing he hmiss hreach]

/-- A lookup that does not return a value rejects the artifact, even for a disconnected
entry: failure is never read as absence. -/
theorem decodeArtifact_eq_none_of_lookup_failure {dt : DType} {source : Model.Dag} {entries : List Nat}
    {Γ : Context} {p : Program ℚ Γ []} {art : α} {e : Nat} (he : e ∈ entries)
    (hfail : GradArtifact.grad art e = none) :
    decodeArtifact dt source entries p art = none := by
  have hall : (entries.all fun e => (GradArtifact.grad art e).isSome) = false := by
    rw [List.all_eq_false]; exact ⟨e, he, by simp [hfail]⟩
  have ht : artifactParts dt source entries p art = none := by
    unfold artifactParts; rw [hall]; rfl
  simp [decodeArtifact, artifactTarget, ht]

/-- Decoding succeeds only if every missing gradient is for a disconnected entry. -/
theorem decodeArtifact_missing_disconnected {dt : DType} {source : Model.Dag} {entries : List Nat} {Γ : Context}
    {p : Program ℚ Γ []} {art : α} {actual : Circuit ℚ Γ Γ}
    (h : decodeArtifact dt source entries p art = some actual) {j e : Nat}
    (he : entries[j]? = some e) (hmiss : (GradArtifact.grad art e).join = none) :
    needs p j = false := by
  cases hn : needs p j
  · rfl
  · rw [decodeArtifact_eq_none_of_missing he hmiss hn] at h; contradiction

/-- **Bridge theorem (a).** For a source graph and a post-AD artifact that decode to
`p : Program ℚ Γ []` and `actual : Circuit ℚ Γ Γ`, a successful reference correspondence
check yields the real derivative of the source loss as the value of the *actual*
circuit, its equality with the reference all-entry gradient, and zero at every
disconnected entry (where `totalize` zero-filled). -/
theorem checkedGrad_bridge {dt : DType} {source : Model.Dag} {entries : List Nat}
    {loss : Nat} {art : α} {Γ : Context} {p : Program ℚ Γ []} {actual : Circuit ℚ Γ Γ}
    {witness : List (Option Nat)}
    (hsrc : decodeSource dt source entries loss = some ⟨Γ, p⟩)
    (hart : decodeArtifact dt source entries p art = some actual)
    (hcheck : GraphCorrespondence.check p.compileGrad actual witness = true) :
    (∀ ρ dρ : Values ℝ Γ, fderiv ℝ p.eval ρ dρ () = Values.dot dρ (actual.eval ρ)) ∧
    (∀ ρ : Values ℝ Γ, actual.eval ρ = p.grad ρ) ∧
    (∀ (ρ : Values ℝ Γ) {s : Shape} (v : Var Γ s), needs p v.offset = false →
      v.get (actual.eval ρ) = 0) := by
  have _ := hsrc
  have _ := hart
  have heval : ∀ ρ : Values ℝ Γ, actual.eval ρ = p.grad ρ := fun ρ => by
    rw [← GraphCorrespondence.check_sound p.compileGrad actual witness hcheck ρ,
      p.compileGrad_correct]
  refine ⟨Program.checkedGrad_correct p actual witness hcheck, heval, ?_⟩
  intro ρ s v hv
  rw [heval]
  exact grad_get_of_disconnected p ρ v hv

/-- One-shot acceptance: decode source and artifact from the Rust values and run the
reference correspondence check. -/
def acceptGradient (dt : DType) (source : Model.Dag) (entries : List Nat) (loss : Nat)
    (art : α) (witness : List (Option Nat)) : Bool :=
  match decodeSource dt source entries loss with
  | none => false
  | some ⟨_, p⟩ =>
      match decodeArtifact dt source entries p art with
      | none => false
      | some actual => GraphCorrespondence.check p.compileGrad actual witness

theorem acceptGradient_sound {dt : DType} {source : Model.Dag} {entries : List Nat}
    {loss : Nat} {art : α} {witness : List (Option Nat)}
    (h : acceptGradient dt source entries loss art witness = true) :
    ∃ (Γ : Context) (p : Program ℚ Γ []) (actual : Circuit ℚ Γ Γ),
      decodeSource dt source entries loss = some ⟨Γ, p⟩ ∧
      decodeArtifact dt source entries p art = some actual ∧
      (∀ ρ dρ : Values ℝ Γ, fderiv ℝ p.eval ρ dρ () = Values.dot dρ (actual.eval ρ)) ∧
      (∀ ρ : Values ℝ Γ, actual.eval ρ = p.grad ρ) := by
  unfold acceptGradient at h
  split at h
  · contradiction
  · rename_i Γ p hs
    split at h
    · contradiction
    · rename_i actual ha
      obtain ⟨h1, h2, _⟩ := checkedGrad_bridge hs ha h
      exact ⟨Γ, p, actual, hs, ha, h1, h2⟩

/-- Acceptance of the extracted production AD entry point's actual return value:
`grad_dag_checked forward output wrt`, with the requested `wrt` ids as ordered entries. -/
def acceptExtracted (dt : DType) (forward : Model.Dag) (output : Model.NodeId)
    (wrt : Slice Model.NodeId) (witness : List (Option Nat)) : Bool :=
  match extractedResult (Model.gradDagChecked forward output wrt) with
  | none => false
  | some r => acceptGradient dt forward (wrt.val.map (·.val)) output.val r witness

theorem acceptExtracted_sound {dt : DType} {forward : Model.Dag} {output : Model.NodeId}
    {wrt : Slice Model.NodeId} {witness : List (Option Nat)}
    (h : acceptExtracted dt forward output wrt witness = true) :
    ∃ (r : Model.GradResult) (Γ : Context) (p : Program ℚ Γ []) (actual : Circuit ℚ Γ Γ),
      Model.gradDagChecked forward output wrt = .ok (.Ok r) ∧
      decodeSource dt forward (wrt.val.map (·.val)) output.val = some ⟨Γ, p⟩ ∧
      decodeArtifact dt forward (wrt.val.map (·.val)) p r = some actual ∧
      (∀ ρ dρ : Values ℝ Γ, fderiv ℝ p.eval ρ dρ () = Values.dot dρ (actual.eval ρ)) ∧
      (∀ ρ : Values ℝ Γ, actual.eval ρ = p.grad ρ) := by
  unfold acceptExtracted at h
  split at h
  · contradiction
  · rename_i r hr
    obtain ⟨Γ, p, actual, hs, ha, h1, h2⟩ := acceptGradient_sound h
    refine ⟨r, Γ, p, actual, ?_, hs, ha, h1, h2⟩
    unfold extractedResult at hr
    split at hr
    · rename_i r' heq; cases Option.some.inj hr; exact heq
    · contradiction

/-! ### The artifact's own roots and the totalized entry gradients share one prefix -/

theorem Var.getElem_offset : ∀ {Γ : Context} {s : Shape} (v : Var Γ s), Γ[v.offset]? = some s
  | _ :: _, _, .here => rfl
  | _ :: _, _, .there v => by simp [Var.offset, Var.getElem_offset v]

theorem forall₂_getElem? {α' β' : Type} {R : α' → β' → Prop} :
    ∀ {l₁ : List α'} {l₂ : List β'}, List.Forall₂ R l₁ l₂ →
      ∀ {k : Nat} {a : α'}, l₁[k]? = some a → ∃ b, l₂[k]? = some b ∧ R a b
  | [], [], .nil, _, _, h => by simp at h
  | _ :: _, b' :: _, .cons hab _, 0, _, h => by
      simp only [List.getElem?_cons_zero, Option.some.injEq] at h
      subst h
      exact ⟨b', rfl, hab⟩
  | _ :: _, _ :: _, .cons _ hrest, k + 1, _, h => by
      simp only [List.getElem?_cons_succ] at h ⊢
      exact forall₂_getElem? hrest h

/-- Each root variable is the variable decoded at its listed offset. -/
theorem decodeRoots_get {Δ : Context} :
    ∀ {Ω : Context} {offs : List Nat} {σ : Renaming Ω Δ},
      GraphSyntax.decodeRoots Δ Ω offs = some (α := Renaming Ω Δ) σ →
      ∀ {s : Shape} (w : Var Ω s) {i : Nat}, offs[w.offset]? = some i →
        Var.decode Δ s i = some (σ w)
  | [], _, _, _, _, w, _, _ => nomatch w
  | _ :: _, [], _, h, _, _, _, _ => by simp [GraphSyntax.decodeRoots] at h
  | _ :: _, j :: rest, σ, h, _, w, i, hi => by
      simp only [GraphSyntax.decodeRoots, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at h
      obtain ⟨v, hv, tail, htail, hσ⟩ := h
      have hw := congrArg (fun (τ : Renaming _ Δ) => τ w) hσ
      simp only at hw
      rw [← hw]
      cases w with
      | here =>
          simp only [Var.offset, List.getElem?_cons_zero, Option.some.injEq] at hi
          subst hi
          exact hv
      | there w =>
          simp only [Var.offset, List.getElem?_cons_succ] at hi
          exact decodeRoots_get htail w hi

theorem Target.decode_spec {C : Type} {Γ Ω : Context} {tgt : GraphSyntax.Target C}
    {c : Circuit C Γ Ω} (h : tgt.decode Γ Ω = some c) :
    ∃ (Δ : Context) (code : Prefix C Γ Δ) (σ : Renaming Ω Δ),
      GraphSyntax.decodePrefix Γ tgt.operations = some ⟨Δ, code⟩ ∧
      GraphSyntax.decodeRoots Δ Ω tgt.roots = some (α := Renaming Ω Δ) σ ∧
        c = ⟨Δ, code, σ⟩ := by
  unfold GraphSyntax.Target.decode at h
  cases hp : GraphSyntax.decodePrefix Γ tgt.operations with
  | none => simp [hp] at h
  | some pr =>
      obtain ⟨Δ, code⟩ := pr
      simp only [hp, Option.bind_eq_bind, Option.bind_some] at h
      cases hr : GraphSyntax.decodeRoots Δ Ω tgt.roots with
      | none => simp [hr] at h
      | some σ =>
          simp only [hr, Option.bind_some, Option.pure_def, Option.some.injEq] at h
          exact ⟨Δ, code, σ, rfl, hr, h.symm⟩

theorem rootOffsets_get {st : State} {extra : Nat} {roots offs : List Nat}
    (h : rootOffsets st extra roots = some offs) {k g : Nat} (hk : roots[k]? = some g)
    {sl : Slot} (hsl : st.slot g = some sl) (hlt : sl.owner < st.len) :
    offs[k]? = some (st.len + extra - 1 - sl.owner) := by
  obtain ⟨b, hb, hR⟩ := forall₂_getElem? ((mapM_eq_some_iff _ _ _).mp h) hk
  simp only [hsl, Option.bind_some, hlt, if_true, Option.some.injEq] at hR
  rw [hb, hR]

theorem decodeArtifact_spec {dt : DType} {source : Model.Dag} {entries : List Nat} {Γ : Context}
    {p : Program ℚ Γ []} {art : α} {actual : Circuit ℚ Γ Γ}
    (h : decodeArtifact dt source entries p art = some actual) :
    ∃ st zs owners, artifactParts dt source entries p art = some (st, zs, owners) ∧
      (totalTarget st zs owners).decode Γ Γ = some actual := by
  unfold decodeArtifact artifactTarget at h
  cases hp : artifactParts dt source entries p art with
  | none => simp [hp] at h
  | some x =>
      obtain ⟨st, zs, owners⟩ := x
      simp only [hp, Option.map_some, Option.bind_some] at h
      exact ⟨st, zs, owners, rfl, h⟩

theorem decodeArtifactRoots_spec {dt : DType} {source : Model.Dag} {entries : List Nat} {Γ : Context}
    {p : Program ℚ Γ []} {art : α} {Ω : Context} {rc : Circuit ℚ Γ Ω}
    (h : decodeArtifactRoots dt source entries p art = some ⟨Ω, rc⟩) :
    ∃ st zs owners offs, artifactParts dt source entries p art = some (st, zs, owners) ∧
      rootOffsets st zs.length (artifactRootIds art) = some offs ∧
      (⟨st.ops ++ zs, offs⟩ : GraphSyntax.Target ℚ).decode Γ Ω = some rc := by
  unfold decodeArtifactRoots artifactRoots at h
  cases hp : artifactParts dt source entries p art with
  | none => simp [hp] at h
  | some x =>
      obtain ⟨st, zs, owners⟩ := x
      simp only [hp, Option.bind_some] at h
      cases hΩ : (artifactRootIds art).mapM st.shapeOf with
      | none => simp [hΩ] at h
      | some Ω' =>
          simp only [hΩ, Option.bind_some] at h
          cases ho : rootOffsets st zs.length (artifactRootIds art) with
          | none => simp [ho] at h
          | some offs =>
              simp only [ho, Option.map_some, Option.bind_some] at h
              cases hd : GraphSyntax.Target.decode Γ Ω' ⟨st.ops ++ zs, offs⟩ with
              | none => simp [hd] at h
              | some c =>
                  simp only [hd, Option.map_some, Option.some.injEq] at h
                  cases h
                  exact ⟨st, zs, owners, offs, rfl, ho, hd⟩

/-- **Root link.** A Rust root of the artifact that is the returned gradient node of
requested entry `u` denotes exactly the totalized entry gradient: both views decode the
same operations and resolve the node to the same final-context variable. -/
theorem artifact_root_link {dt : DType} {source : Model.Dag} {entries : List Nat} {Γ : Context}
    {p : Program ℚ Γ []} {art : α} {actual : Circuit ℚ Γ Γ} {Ω : Context}
    {rc : Circuit ℚ Γ Ω}
    (hA : decodeArtifact dt source entries p art = some actual)
    (hR : decodeArtifactRoots dt source entries p art = some ⟨Ω, rc⟩)
    {s : Shape} (u : Var Γ s) (w : Var Ω s) {e g : Nat}
    (he : entries[u.offset]? = some e) (hg : (GradArtifact.grad art e).join = some g)
    (hr : (artifactRootIds art)[w.offset]? = some g)
    {R : Type} [CommSemiring R] [Algebra ℚ R] (ρ : Values R Γ) :
    w.get (rc.eval ρ) = u.get (actual.eval ρ) := by
  obtain ⟨st, zs, owners, hp, hdA⟩ := decodeArtifact_spec hA
  obtain ⟨st', zs', owners', offs, hp', ho, hdR⟩ := decodeArtifactRoots_spec hR
  rw [hp] at hp'
  simp only [Option.some.injEq, Prod.mk.injEq] at hp'
  obtain ⟨rfl, rfl, rfl⟩ := hp'
  have htot := artifactParts_spec hp
  obtain ⟨sl, hsl, hlt, hown⟩ := totalizeParts_present htot he (Var.getElem_offset u) hg
  obtain ⟨Δ, code, σA, hpA, hrA, rfl⟩ := Target.decode_spec hdA
  obtain ⟨Δ', code', σR, hpR, hrR, rfl⟩ := Target.decode_spec hdR
  simp only [totalTarget] at hpA hrA
  simp only at hpR hrR
  rw [hpA] at hpR
  cases hpR
  have hiA : (owners.map (fun o => st.len + zs.length - 1 - o))[u.offset]? =
      some (st.len + zs.length - 1 - sl.owner) := by
    simp [List.getElem?_map, hown]
  have hiR := rootOffsets_get ho hr hsl hlt
  have hvA := decodeRoots_get hrA u hiA
  have hvR := decodeRoots_get hrR w hiR
  rw [hvA] at hvR
  have heq := Option.some.inj hvR
  simp only [Circuit.eval, Renaming.get_read, heq]

end Artifact

end ChelisBridge

/-! ## Combined acceptance of (source, post-AD artifact, final ownership plan) -/

namespace ChelisBridge.Validation

open Aeneas.Std LaCaDiLECore ChelisBridge

variable {α β : Type} [GradArtifact α] [Ownership.OwnershipPlan β]

/-- The plan DAG's `Load` node of each source entry, identified by input name. -/
def planEntryIds (source : Model.Dag) (entries : List Nat) (plan : β) :
    Option (List (Option Nat)) :=
  (entryNames source entries).bind (entryIdsIn (Ownership.OwnershipPlan.dag plan))

/-- Proof-producing acceptance per the bridge contract (item 5): independently decode the
source loss program, the post-AD artifact (totalized over the entries, and over its own
Rust roots), and the actual final ownership plan over those roots; then require the
reference ownership check of the original plan, the gradient correspondence of the
totalized artifact, and the copy-normalized value correspondence of the final plan. -/
def acceptPlan (dt : DType) (source : Model.Dag) (entries : List Nat) (loss : Nat) (art : α)
    (plan : β) (adWitness normalizationWitness : List (Option Nat)) : Bool :=
  match decodeSource dt source entries loss with
  | none => false
  | some ⟨Γ, p⟩ =>
      match decodeArtifact dt source entries p art with
      | none => false
      | some actual =>
          match decodeArtifactRoots dt source entries p art with
          | none => false
          | some ⟨Ω, rc⟩ =>
              match (planEntryIds source entries plan).bind
                  (fun ids => Ownership.decodeOwned dt ids Γ Ω plan) with
              | none => false
              | some final =>
                  final.check (LaCaDiLECore.Ownership.initial Γ.length) &&
                    GraphCorrespondence.check p.compileGrad actual adWitness &&
                    GraphCorrespondence.checkCopies rc final.circuit normalizationWitness

/-- **Combined bridge theorem.** An accepted (source, artifact, plan) triple yields: the
final plan executes under the reference ownership machine from borrowed entries, returns
its value semantics, and terminates each local owner exactly once and no entry; the
real derivative of the source loss is the totalized artifact's value, which is the
reference all-entry gradient; the final plan computes the artifact's own roots; and every
final root that is the returned gradient node of entry `u` equals the reference
gradient at `u`. -/
theorem checkGradient_bridge {dt : DType} {source : Model.Dag} {entries : List Nat}
    {loss : Nat} {art : α} {plan : β} {adWitness normalizationWitness : List (Option Nat)}
    (h : acceptPlan dt source entries loss art plan adWitness normalizationWitness = true) :
    ∃ (Γ : Context) (p : Program ℚ Γ []) (actual : Circuit ℚ Γ Γ) (Ω : Context)
      (rc : Circuit ℚ Γ Ω) (final : OwnedCode ℚ Γ Ω),
      decodeSource dt source entries loss = some ⟨Γ, p⟩ ∧
      decodeArtifact dt source entries p art = some actual ∧
      decodeArtifactRoots dt source entries p art = some ⟨Ω, rc⟩ ∧
      (∃ ids, planEntryIds source entries plan = some ids ∧
        Ownership.decodeOwned dt ids Γ Ω plan = some final) ∧
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
          entries[u.offset]? = some e → (GradArtifact.grad art e).join = some g →
          (artifactRootIds art)[w.offset]? = some g →
          w.get (final.eval ρ) = u.get (p.grad ρ)) := by
  unfold acceptPlan at h
  split at h
  next => contradiction
  next Γ p hs =>
    split at h
    next => contradiction
    next actual ha =>
      split at h
      next => contradiction
      next Ω rc hr =>
        split at h
        next => contradiction
        next final hf =>
          simp only [Bool.and_eq_true] at h
          obtain ⟨⟨hown, hck⟩, hcp⟩ := h
          obtain ⟨ids, hids, hf⟩ := Option.bind_eq_some_iff.mp hf
          refine ⟨Γ, p, actual, Ω, rc, final, hs, ha, hr, ⟨ids, hids, hf⟩, fun ρ => ?_⟩
          have heval : actual.eval ρ = p.grad ρ := by
            rw [← GraphCorrespondence.check_sound p.compileGrad actual adWitness hck ρ,
              p.compileGrad_correct]
          have hfin : final.eval ρ = rc.eval ρ := by
            rw [← final.circuit_eval,
              ← GraphCorrespondence.checkCopies_sound rc final.circuit normalizationWitness hcp ρ]
          obtain ⟨_, _, hrun⟩ := Ownership.decodeOwned_sound hf
          refine ⟨hrun hown ρ, fun dρ => Program.checkedGrad_correct p actual adWitness hck ρ dρ,
            heval, hfin, ?_⟩
          intro s u w e g he hg hrr
          rw [hfin, ← heval]
          exact artifact_root_link ha hr u w he hg hrr ρ

end ChelisBridge.Validation
