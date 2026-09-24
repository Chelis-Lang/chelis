import ChelisBridge.Numeric.Bindings
import ChelisBridge.Numeric.Flat
import ChelisBridge.Numeric.Decode
import ChelisBridge.Validation

/-!
# A flat-buffer specification of the Rust node loop, simulated by `ieeeEval`

`specRun` is the fragment of the node loop of `eval_tensor_internal_with_result_claims`
(`crates/chelis-ir/src/eval.rs`) on flat row-major buffers: nodes in append order, dead
nodes (outside the roots' live mask) skipped, `Load` bound by input name, and one Rust
kernel per admitted operation (`nodeKernel`).

`specRun_simulates`: on a graph admitted by the float decoder, with every entry's input
buffer decoding row-major to the corresponding input tensor, the loop succeeds and every
live node's buffer decodes row-major (bitwise) to the IEEE value of its binding in the
decoded program.
-/

namespace ChelisBridge.Numeric

open Aeneas.Std LaCaDiLECore ChelisBridge

/-! ## Offsets through embeddings and renamings -/

section Offsets

variable {C : Type}

theorem prefix_length_le {Γ Δ : Context} (p : Prefix C Γ Δ) : Γ.length ≤ Δ.length := by
  induction p with
  | nil => exact le_refl _
  | bind _ _ ih => simp at ih; omega

theorem embedding_offset {Γ Δ : Context} (p : Prefix C Γ Δ) {s : Shape} (v : Var Γ s) :
    (p.embedding v).offset = v.offset + (Δ.length - Γ.length) := by
  induction p with
  | nil => simp [Prefix.embedding, Renaming.id]
  | bind op rest ih =>
      have := prefix_length_le rest
      simp only [List.length_cons] at this
      simp only [Prefix.embedding, Renaming.comp, Renaming.shift, ih, Var.offset,
        List.length_cons] at this ⊢
      omega

/-- Add `d` to every value port. -/
def shiftPorts (d : Nat) : GraphSyntax.Operation C → GraphSyntax.Operation C
  | .const s xs => .const s xs
  | .copy s i => .copy s (i + d)
  | .add s i j => .add s (i + d) (j + d)
  | .mul s i j => .mul s (i + d) (j + d)
  | .sum s n t k i => .sum s n t k (i + d)
  | .insert s n t k i => .insert s n t k (i + d)
  | .expand pre post n i => .expand pre post n (i + d)

theorem rename_encode {Γ Δ : Context} {s : Shape} (σ : Renaming Γ Δ) (d : Nat)
    (hσ : ∀ {t : Shape} (v : Var Γ t), (σ v).offset = v.offset + d) (op : Op C Γ s) :
    (op.rename σ).encode = shiftPorts d op.encode := by
  cases op <;> simp [Op.rename, Op.encode, shiftPorts, hσ]

/-- The `k`-th binding of a decoded prefix is the `k`-th operation with its ports shifted
into the final context, bound at offset `|Δ| - 1 - (|Γ| + k)`. -/
theorem bindings_decode :
    ∀ {Γ Δ : Context} (ops : List (GraphSyntax.Operation C)) (p : Prefix C Γ Δ),
      GraphSyntax.decodePrefix Γ ops = some ⟨Δ, p⟩ → ∀ k (hk : k < ops.length),
        ∃ (s : Shape) (op : Op C Δ s) (v : Var Δ s),
          (GraphCorrespondence.bindings p)[k]? = some ⟨s, op, v⟩ ∧
          op.encode = shiftPorts (Δ.length - (Γ.length + k)) ops[k] ∧
          v.offset + (Γ.length + k) + 1 = Δ.length
  | Γ, Δ, [], p, _, k, hk => by simp at hk
  | Γ, Δ, o :: rest, p, h, k, hk => by
      simp only [GraphSyntax.decodePrefix, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at h
      obtain ⟨⟨s, op⟩, hop, ⟨Δ', rest'⟩, hrest, h⟩ := h
      cases h
      have hlen := prefix_length_le rest'
      simp only [List.length_cons] at hlen
      cases k with
      | zero =>
          refine ⟨s, op.rename (Renaming.comp rest'.embedding Renaming.shift), rest'.embedding .here,
            rfl, ?_, ?_⟩
          · rw [rename_encode _ (Δ'.length - Γ.length)
              (fun v => by simp only [Renaming.comp, Renaming.shift, embedding_offset, Var.offset,
                List.length_cons]; omega),
              Operation.encode_of_decode o ⟨s, op⟩ hop]
            simp
          · simp only [embedding_offset, Var.offset, List.length_cons]; omega
      | succ k =>
          obtain ⟨t, op', v, hb, he, hv⟩ := bindings_decode rest rest' hrest k (by simpa using hk)
          refine ⟨t, op', v, by simpa [GraphCorrespondence.bindings] using hb, ?_, ?_⟩
          · rw [he]; simp only [List.length_cons, List.getElem_cons_succ]
            congr 1; omega
          · dsimp only at hv ⊢; simp only [List.length_cons] at hv; omega

end Offsets

/-! ## The Rust node loop on flat buffers -/

/-- A Rust tensor value: its shape and flat row-major storage. -/
structure FlatValue (f : BinaryFormat) where
  shape : Shape
  data : List (BinaryFloat f)

/-- One node of the fragment, as the Rust loop computes it from the values of its input
nodes (`values[&node.inputs[k]]`): `Load` by input name, constants materialized from their
payload, `Copy`/`Reshape` moving the buffer, `Add`/`Mul` elementwise in port order, `Sum`
through the typed pairwise reduction, `Expand` through its gather plan. -/
def nodeKernel (dt : DType) (inputs : Model.LoadStoreName → Option (List (BinaryFloat dt.format)))
    (vals : Nat → Option (FlatValue dt.format)) (node : Model.DagNode) (out : Shape) :
    Option (List (BinaryFloat dt.format)) :=
  match node.op with
  | .Load label => inputs label
  | .Const v => payload dt (.Const v) out
  | .ConstTensor d => payload dt (.ConstTensor d) out
  | .Copy => ((node.inputs.val[0]?).bind fun x => vals x.val).map (·.data)
  | .Add => ((node.inputs.val[0]?).bind fun x => vals x.val).bind fun a =>
      ((node.inputs.val[1]?).bind fun y => vals y.val).map fun b =>
        List.zipWith BinaryFloat.add a.data b.data
  | .Mul => ((node.inputs.val[0]?).bind fun x => vals x.val).bind fun a =>
      ((node.inputs.val[1]?).bind fun y => vals y.val).map fun b =>
        List.zipWith BinaryFloat.mul a.data b.data
  | .Sum axis _ => ((node.inputs.val[0]?).bind fun x => vals x.val).map fun a =>
      flatSum a.data a.shape axis.val
  | .Expand axis _ => ((node.inputs.val[0]?).bind fun x => vals x.val).map fun a =>
      (expandPicks a.shape out axis.val).map (at' a.data)
  | .Reshape _ => ((node.inputs.val[0]?).bind fun x => vals x.val).map (·.data)
  | _ => none

/-- The node loop: append order, dead nodes skipped, each live node's value recorded. -/
def specRun (dt : DType) (inputs : Model.LoadStoreName → Option (List (BinaryFloat dt.format)))
    (live : Nat → Bool) :
    List Model.DagNode → (Nat → Option (FlatValue dt.format)) → Option (Nat → Option (FlatValue dt.format))
  | [], vals => some vals
  | node :: rest, vals =>
      if live node.id.val then
        (resolveShape node.output_type.dims).bind fun out =>
          (nodeKernel dt inputs vals node out).bind fun d =>
            specRun dt inputs live rest (fun x => if x = node.id.val then some ⟨out, d⟩ else vals x)
      else specRun dt inputs live rest vals

/-! ## The simulation invariant -/

section Simulation

variable {dt : DType} {Δ : Context}

/-- The flat buffer `d` is, row-major and bitwise, the IEEE value of the binding with Lean
owner `o` in the final context. -/
def OwnerOK (V : Values (BinaryFloat dt.format) Δ) (o : Nat) (sh : Shape)
    (d : List (BinaryFloat dt.format)) : Prop :=
  (∃ w : Var Δ sh, w.offset + o + 1 = Δ.length) ∧
    ∀ w : Var Δ sh, w.offset + o + 1 = Δ.length → Tensor.ofList sh d = some (w.get V)

/-- Every decoded, live node has a recorded buffer that is the IEEE value of its binding. -/
def Inv (live : Nat → Bool) (V : Values (BinaryFloat dt.format) Δ) (st : State)
    (vals : Nat → Option (FlatValue dt.format)) : Prop :=
  ∀ x sl, st.slot x = some sl → live x = true →
    ∃ d, vals x = some ⟨sl.shape, d⟩ ∧ OwnerOK V sl.owner sl.shape d

theorem arg_value {live : Nat → Bool} {V : Values (BinaryFloat dt.format) Δ} {st : State}
    {vals : Nat → Option (FlatValue dt.format)} (hinv : Inv live V st vals) {x i d : Nat}
    {s : Shape} (hlive : live x = true) (hoff : st.offset x = some i)
    (hshape : st.shapeOf x = some s) (hd : st.len + d = Δ.length) (v : Var Δ s)
    (hv : v.offset = i + d) :
    ∃ dx, vals x = some ⟨s, dx⟩ ∧ Tensor.ofList s dx = some (v.get V) := by
  obtain ⟨sl, hsl, hrel⟩ := State.offset_spec hoff
  have hss : sl.shape = s := by
    simp only [State.shapeOf, hsl, Option.map_some, Option.some.injEq] at hshape; exact hshape
  obtain ⟨dx, hvx, hok⟩ := hinv x sl hsl hlive
  rw [hss] at hvx hok
  exact ⟨dx, hvx, hok.2 v (by omega)⟩

theorem mapM_offsets_one {st : State} {x : Usize} {offs : List Nat}
    (h : [x].mapM (fun y => st.offset y.val) = some offs) :
    ∃ i, offs = [i] ∧ st.offset x.val = some i := by
  have hf := (mapM_eq_some_iff _ _ _).mp h
  rcases hf with _ | ⟨hi, hf⟩
  rcases hf
  exact ⟨_, rfl, hi⟩

theorem mapM_offsets_two {st : State} {x y : Usize} {offs : List Nat}
    (h : [x, y].mapM (fun y => st.offset y.val) = some offs) :
    ∃ i j, offs = [i, j] ∧ st.offset x.val = some i ∧ st.offset y.val = some j := by
  have hf := (mapM_eq_some_iff _ _ _).mp h
  rcases hf with _ | ⟨hi, hf⟩
  rcases hf with _ | ⟨hj, hf⟩
  rcases hf
  exact ⟨_, _, rfl, hi, hj⟩

end Simulation

section Kernel

variable {dt : DType} {Δ : Context}

theorem classify_copy {st : State} {inputs : List Usize} {out : Shape} {k : Kind}
    (h : classify dt st .Copy inputs out = some k) :
    ∃ x, inputs = [x] ∧ st.shapeOf x.val = some out ∧ k = .copy out := by
  simp only [classify, classifyCopy] at h
  split at h
  · rename_i x
    split at h
    · rename_i hs; exact ⟨x, rfl, hs, (Option.some.inj h).symm⟩
    · cases h
  · cases h

theorem classify_sum {st : State} {axis : Usize} {acc : Model.Prim} {inputs : List Usize}
    {out : Shape} {k : Kind} (h : classify dt st (.Sum axis acc) inputs out = some k) :
    ∃ x s n, inputs = [x] ∧ st.shapeOf x.val = some s ∧ s[axis.val]? = some n ∧
      k = .sum s n out axis.val := by
  simp only [classify, classifySum] at h
  split at h
  · rename_i x
    split at h
    · simp only [Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
      obtain ⟨s, hs, n, hn, rfl⟩ := h
      exact ⟨x, s, n, rfl, hs, hn, rfl⟩
    · cases h
  · cases h

theorem shape_split {s : Shape} {a : Nat} (h : s[a]? = some 1) :
    s = s.take a ++ 1 :: s.drop (a + 1) ∧ (s.take a).length = a := by
  obtain ⟨ha, h1⟩ := List.getElem?_eq_some_iff.mp h
  refine ⟨?_, by simp; omega⟩
  conv_lhs => rw [← List.take_append_drop a s]
  rw [List.drop_eq_getElem_cons ha, h1]

/-- **One Rust kernel against the IEEE reference.** For a node the bridge decoder admits
(non-`Load`), whose inputs are live and satisfy the invariant, the Rust kernel's buffer is
row-major and bitwise the IEEE value of the node's binding. -/
theorem kernel_correct {live : Nat → Bool} {V : Values (BinaryFloat dt.format) Δ} {st : State}
    {vals : Nat → Option (FlatValue dt.format)}
    {inputs : Model.LoadStoreName → Option (List (BinaryFloat dt.format))}
    {node : Model.DagNode} {out : Shape} {oq : GraphSyntax.Operation ℚ}
    {of : GraphSyntax.Operation (BinaryFloat dt.format)} {t : Shape}
    {op' : Op (BinaryFloat dt.format) Δ t} {d : Nat}
    (hinv : Inv live V st vals) (hlive : ∀ x ∈ node.inputs.val, live x.val = true)
    (hnl : ∀ l, node.op ≠ .Load l)
    (hdec : decodeOperation dt st node.op node.inputs.val out = some oq)
    (hlift : liftOp oq (payload dt node.op out) = some of)
    (hop : GraphSyntax.Operation.decode Δ (shiftPorts d of) = some ⟨t, op'⟩)
    (hd : st.len + d = Δ.length) :
    ∃ data, nodeKernel dt inputs vals node out = some data ∧ t = out ∧
      Tensor.ofList t data = some (opEval op' V) := by
  simp only [decodeOperation, Option.bind_eq_bind, Option.bind_eq_some_iff] at hdec
  obtain ⟨k, hk, offs, hoffs, hb⟩ := hdec
  cases hnode : node.op with
  | Load l => exact absurd hnode (hnl l)
  | Const v =>
      rw [hnode] at hk hoffs hlift
      obtain ⟨q, hin, hq, rfl⟩ := classify_const hk
      simp only [valueInputs, hin, List.mapM_nil, Option.pure_def, Option.some.injEq] at hoffs
      subst hoffs
      simp only [Kind.build, Option.some.injEq] at hb
      subst hb
      cases hp : payload dt (.Const v) out with
      | none => rw [hp] at hlift; simp [liftOp] at hlift
      | some xs =>
          rw [hp] at hlift
          simp only [liftOp] at hlift
          split at hlift
          · cases hlift
            simp only [shiftPorts, GraphSyntax.Operation.decode] at hop
            obtain ⟨tx, htx, hop⟩ := Option.map_eq_some_iff.mp hop
            cases hop
            exact ⟨xs, by simp [nodeKernel, hnode, hp], rfl, by simpa [opEval] using htx⟩
          · cases hlift
  | ConstTensor data =>
      rw [hnode] at hk hoffs hlift
      obtain ⟨qs, hin, hq, rfl⟩ := classify_constTensor hk
      simp only [valueInputs, hin, List.mapM_nil, Option.pure_def, Option.some.injEq] at hoffs
      subst hoffs
      simp only [Kind.build, Option.some.injEq] at hb
      subst hb
      cases hp : payload dt (.ConstTensor data) out with
      | none => rw [hp] at hlift; simp [liftOp] at hlift
      | some xs =>
          rw [hp] at hlift
          simp only [liftOp] at hlift
          split at hlift
          · cases hlift
            simp only [shiftPorts, GraphSyntax.Operation.decode] at hop
            obtain ⟨tx, htx, hop⟩ := Option.map_eq_some_iff.mp hop
            cases hop
            exact ⟨xs, by simp [nodeKernel, hnode, hp], rfl, by simpa [opEval] using htx⟩
          · cases hlift
  | Copy =>
      rw [hnode] at hk hoffs hlift
      obtain ⟨x, hin, hsx, rfl⟩ := classify_copy hk
      rw [hin] at hoffs
      obtain ⟨i, rfl, hi⟩ := mapM_offsets_one (by simpa [valueInputs] using hoffs)
      simp only [Kind.build, Option.some.injEq] at hb
      subst hb
      simp only [payload, liftOp, Option.some.injEq] at hlift
      subst hlift
      simp only [shiftPorts, GraphSyntax.Operation.decode] at hop
      obtain ⟨v, hv, hop⟩ := Option.map_eq_some_iff.mp hop
      cases hop
      have hlx : live x.val = true := hlive x (by rw [hin]; simp)
      obtain ⟨dx, hvx, hx⟩ := arg_value hinv hlx hi hsx hd v (Var.offset_of_decode hv)
      exact ⟨dx, by simp [nodeKernel, hnode, hin, hvx], rfl, by simpa [opEval] using hx⟩
  | Add =>
      rw [hnode] at hk hoffs hlift
      obtain ⟨x, y, hin, hsx, hsy, rfl⟩ := classify_add hk
      rw [hin] at hoffs
      obtain ⟨i, j, rfl, hi, hj⟩ := mapM_offsets_two (by simpa [valueInputs] using hoffs)
      simp only [Kind.build, Option.some.injEq] at hb
      subst hb
      simp only [payload, liftOp, Option.some.injEq] at hlift
      subst hlift
      simp only [shiftPorts, GraphSyntax.Operation.decode, Option.bind_eq_bind,
        Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hop
      obtain ⟨v, hv, w, hw, hop⟩ := hop
      cases hop
      have hlx : live x.val = true := hlive x (by rw [hin]; simp)
      have hly : live y.val = true := hlive y (by rw [hin]; simp)
      obtain ⟨dx, hvx, hx⟩ := arg_value hinv hlx hi hsx hd v (Var.offset_of_decode hv)
      obtain ⟨dy, hvy, hy⟩ := arg_value hinv hly hj hsy hd w (Var.offset_of_decode hw)
      exact ⟨List.zipWith BinaryFloat.add dx dy, by simp [nodeKernel, hnode, hin, hvx, hvy], rfl,
        by simpa [opEval] using zipWith_flat hx hy BinaryFloat.add⟩
  | Mul =>
      rw [hnode] at hk hoffs hlift
      obtain ⟨x, y, hin, hsx, hsy, rfl⟩ := classify_mul hk
      rw [hin] at hoffs
      obtain ⟨i, j, rfl, hi, hj⟩ := mapM_offsets_two (by simpa [valueInputs] using hoffs)
      simp only [Kind.build, Option.some.injEq] at hb
      subst hb
      simp only [payload, liftOp, Option.some.injEq] at hlift
      subst hlift
      simp only [shiftPorts, GraphSyntax.Operation.decode, Option.bind_eq_bind,
        Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hop
      obtain ⟨v, hv, w, hw, hop⟩ := hop
      cases hop
      have hlx : live x.val = true := hlive x (by rw [hin]; simp)
      have hly : live y.val = true := hlive y (by rw [hin]; simp)
      obtain ⟨dx, hvx, hx⟩ := arg_value hinv hlx hi hsx hd v (Var.offset_of_decode hv)
      obtain ⟨dy, hvy, hy⟩ := arg_value hinv hly hj hsy hd w (Var.offset_of_decode hw)
      exact ⟨List.zipWith BinaryFloat.mul dx dy, by simp [nodeKernel, hnode, hin, hvx, hvy], rfl,
        by simpa [opEval] using zipWith_flat hx hy BinaryFloat.mul⟩
  | Sum axis acc =>
      rw [hnode] at hk hoffs hlift
      obtain ⟨x, s, n, hin, hsx, hn, rfl⟩ := classify_sum hk
      rw [hin] at hoffs
      obtain ⟨i, rfl, hi⟩ := mapM_offsets_one (by simpa [valueInputs] using hoffs)
      simp only [Kind.build, Option.some.injEq] at hb
      subst hb
      simp only [payload, liftOp, Option.some.injEq] at hlift
      subst hlift
      simp only [shiftPorts, GraphSyntax.Operation.decode, Option.bind_eq_bind,
        Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hop
      obtain ⟨a, ha, v, hv, hop⟩ := hop
      cases hop
      have hlx : live x.val = true := hlive x (by rw [hin]; simp)
      obtain ⟨dx, hvx, hx⟩ := arg_value hinv hlx hi hsx hd v (Var.offset_of_decode hv)
      have hpos := Axis.position_of_decode ha
      refine ⟨flatSum dx s axis.val, by simp [nodeKernel, hnode, hin, hvx], rfl, ?_⟩
      rw [← hpos]
      simpa [opEval] using flatSum_eq a hx
  | Expand axis size =>
      rw [hnode] at hk hoffs hlift
      obtain ⟨x, witness, s, n, hin, hsx, hn, _, hk⟩ := classify_expand hk
      rw [hin] at hoffs
      obtain ⟨i, rfl, hi⟩ := mapM_offsets_one (by simpa [valueInputs] using hoffs)
      have hlx : live x.val = true := hlive x (by rw [hin]; simp)
      rcases hk with ⟨_, rfl⟩ | ⟨_, h1, hout, rfl⟩
      · simp only [Kind.build, Option.some.injEq] at hb
        subst hb
        simp only [payload, liftOp, Option.some.injEq] at hlift
        subst hlift
        simp only [shiftPorts, GraphSyntax.Operation.decode, Option.bind_eq_bind,
          Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hop
        obtain ⟨a, ha, v, hv, hop⟩ := hop
        cases hop
        obtain ⟨dx, hvx, hx⟩ := arg_value hinv hlx hi hsx hd v (Var.offset_of_decode hv)
        have hpos := Axis.position_of_decode ha
        refine ⟨(expandPicks s out axis.val).map (at' dx), by simp [nodeKernel, hnode, hin, hvx], rfl, ?_⟩
        rw [← hpos]
        simpa [opEval] using expand_insert_flat a hx
      · simp only [Kind.build, Option.some.injEq] at hb
        subst hb
        simp only [payload, liftOp, Option.some.injEq] at hlift
        subst hlift
        simp only [shiftPorts, GraphSyntax.Operation.decode] at hop
        obtain ⟨v, hv, hop⟩ := Option.map_eq_some_iff.mp hop
        cases hop
        subst hout
        obtain ⟨hs, hlen⟩ := shape_split h1
        rw [hs] at hsx
        obtain ⟨dx, hvx, hx⟩ := arg_value hinv hlx hi hsx hd v (Var.offset_of_decode hv)
        refine ⟨(expandPicks (s.take axis.val ++ 1 :: s.drop (axis.val + 1))
          (s.take axis.val ++ n :: s.drop (axis.val + 1)) axis.val).map (at' dx),
          by simp [nodeKernel, hnode, hin, hvx], rfl, ?_⟩
        have := expand_same_rank_flat (s.take axis.val) (s.drop (axis.val + 1)) n hx
        rw [hlen] at this
        simpa [opEval] using this
  | Reshape ns =>
      rw [hnode] at hk hoffs hlift
      obtain ⟨x, s, p, hin, hsx, _, hk⟩ := classify_reshape hk
      rw [hin] at hoffs
      obtain ⟨i, rfl, hi⟩ := mapM_offsets_one (by simpa [valueInputs] using hoffs)
      have hlx : live x.val = true := hlive x (by rw [hin]; simp)
      rcases hk with ⟨_, _, rfl⟩ | ⟨_, _, rfl⟩
      · simp only [Kind.build, Option.some.injEq] at hb
        subst hb
        simp only [payload, liftOp, Option.some.injEq] at hlift
        subst hlift
        simp only [shiftPorts, GraphSyntax.Operation.decode, Option.bind_eq_bind,
          Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hop
        obtain ⟨a, _, v, hv, hop⟩ := hop
        cases hop
        obtain ⟨dx, hvx, hx⟩ := arg_value hinv hlx hi hsx hd v (Var.offset_of_decode hv)
        exact ⟨dx, by simp [nodeKernel, hnode, hin, hvx], rfl,
          by simpa [opEval] using reshape_insert_flat a hx⟩
      · simp only [Kind.build, Option.some.injEq] at hb
        subst hb
        simp only [payload, liftOp, Option.some.injEq] at hlift
        subst hlift
        simp only [shiftPorts, GraphSyntax.Operation.decode, Option.bind_eq_bind,
          Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq] at hop
        obtain ⟨a, _, v, hv, hop⟩ := hop
        cases hop
        obtain ⟨dx, hvx, hx⟩ := arg_value hinv hlx hi hsx hd v (Var.offset_of_decode hv)
        exact ⟨dx, by simp [nodeKernel, hnode, hin, hvx], rfl,
          by simpa [opEval] using reshape_remove_flat a hx⟩
  | _ => rw [hnode] at hk; simp [classify] at hk

end Kernel

section Steps

theorem guard_prop {c : Prop} [Decidable c] {w : Unit}
    (h : (if c then pure () else failure : Option Unit) = some w) : c := by
  split at h
  · assumption
  · cases h

theorem step_load {dt : DType} {entries : List (Option Nat)} {Γ : Context} {st st' : State}
    {idx : Nat} {node : Model.DagNode} {l : Model.LoadStoreName}
    (h : step dt entries Γ st idx node = some st') (hl : node.op = .Load l) :
    ∃ j out, resolveShape node.output_type.dims = some out ∧ node.id.val = idx ∧
      entries.findIdx? (· == some idx) = some j ∧ Γ[j]? = some out ∧
      st' = { st with env := (idx, ⟨Γ.length - 1 - j, out⟩) :: st.env } := by
  unfold step at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨_, hid, _, _, out, hout, h⟩ := h
  rw [hl] at h
  simp only [Option.bind_eq_some_iff] at h
  obtain ⟨_, _, j, hj, _, hg, h⟩ := h
  cases Option.some.inj h
  exact ⟨j, out, hout, guard_prop hid, hj, guard_prop hg, rfl⟩

theorem step_nonload {dt : DType} {entries : List (Option Nat)} {Γ : Context} {st st' : State}
    {idx : Nat} {node : Model.DagNode} (h : step dt entries Γ st idx node = some st')
    (hne : ∀ l, node.op ≠ .Load l) :
    ∃ out o, resolveShape node.output_type.dims = some out ∧ node.id.val = idx ∧
      decodeOperation dt st node.op node.inputs.val out = some o ∧
      st' = { len := st.len + 1, env := (idx, ⟨st.len, out⟩) :: st.env, ops := st.ops ++ [o] } := by
  unfold step at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨_, hid, _, _, out, hout, h⟩ := h
  have hid' : node.id.val = idx := guard_prop hid
  obtain ⟨o, ho, h⟩ := h
  cases Option.some.inj h
  exact ⟨out, o, hout, hid', ho, rfl⟩

theorem liftStep_nonload {dt : DType} {st' : State} {node : Model.DagNode}
    {acc acc' : List (GraphSyntax.Operation (BinaryFloat dt.format))} {out : Shape}
    {st : State} {o : GraphSyntax.Operation ℚ}
    (hne : ∀ l, node.op ≠ .Load l) (hout : resolveShape node.output_type.dims = some out)
    (hops : st'.ops = st.ops ++ [o]) (h : liftStep dt st' node acc = some acc') :
    ∃ of, liftOp o (payload dt node.op out) = some of ∧ acc' = acc ++ [of] := by
  unfold liftStep at h
  split at h
  · rename_i l hl; exact absurd hl (hne l)
  · simp only [hout, hops, Option.bind_some, List.getLast?_append_of_ne_nil _ (List.cons_ne_nil _ _),
      List.getLast?_singleton, Option.map_eq_some_iff] at h
    obtain ⟨of, hof, rfl⟩ := h
    exact ⟨of, hof, rfl⟩

theorem runF_prefix {dt : DType} {entries : List (Option Nat)} {Γ : Context} :
    ∀ {i : Nat} {st : State} {acc : List (GraphSyntax.Operation (BinaryFloat dt.format))}
      {nodes : List Model.DagNode} {stF : State} {opsF},
      runF dt entries Γ i st acc nodes = some (stF, opsF) → acc <+: opsF
  | _, st, acc, [], stF, opsF, h => by
      simp only [runF, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨_, rfl⟩ := h
      exact List.prefix_refl _
  | i, st, acc, node :: rest, stF, opsF, h => by
      simp only [runF, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
      obtain ⟨st1, _, acc1, hacc1, h2⟩ := h
      have hp := runF_prefix h2
      have : acc <+: acc1 := by
        unfold liftStep at hacc1
        split at hacc1
        · cases Option.some.inj hacc1; exact List.prefix_refl _
        · simp only [Option.bind_eq_some_iff, Option.map_eq_some_iff] at hacc1
          obtain ⟨_, _, _, _, _, _, rfl⟩ := hacc1
          exact List.prefix_append _ _
      exact this.trans hp

theorem var_decode_of_getElem? : ∀ {Γ : Context} {s : Shape} {j : Nat}, Γ[j]? = some s →
    ∃ u : Var Γ s, u.offset = j
  | [], _, _, h => by simp at h
  | t :: Γ, s, 0, h => by
      simp only [List.getElem?_cons_zero, Option.some.injEq] at h
      subst h; exact ⟨.here, rfl⟩
  | t :: Γ, s, j + 1, h => by
      simp only [List.getElem?_cons_succ] at h
      obtain ⟨u, hu⟩ := var_decode_of_getElem? h
      exact ⟨.there u, by simp [Var.offset, hu]⟩

end Steps

section Main

variable {dt : DType} {Γ Δ : Context}

/-- Input binding by name: the `Load` node of entry `j` names an input whose buffer decodes
row-major (bitwise) to the entry value. -/
def InputsOK (dt : DType) (dag : Model.Dag) (entries : List Nat)
    (inputs : Model.LoadStoreName → Option (List (BinaryFloat dt.format)))
    (ρ : Values (BinaryFloat dt.format) Γ) : Prop :=
  ∀ (j : Nat) (n : Model.DagNode) (label : Model.LoadStoreName) {s : Shape} (u : Var Γ s),
    n ∈ dag.nodes.val → n.op = .Load label → entries[j]? = some n.id.val → u.offset = j →
      ∃ xs, inputs label = some xs ∧ Tensor.ofList s xs = some (u.get ρ)

/-- The live mask is closed under inputs (as `live_mask_for_roots` makes it). -/
def LiveClosed (dag : Model.Dag) (live : Nat → Bool) : Prop :=
  ∀ n ∈ dag.nodes.val, live n.id.val = true → ∀ x ∈ n.inputs.val, live x.val = true

theorem Inv.update_dead {live : Nat → Bool} {V : Values (BinaryFloat dt.format) Δ} {st : State}
    {vals : Nat → Option (FlatValue dt.format)} (h : Inv live V st vals) {idx : Nat} {sl0 : Slot}
    (hdead : live idx = false) {len : Nat} {ops : List (GraphSyntax.Operation ℚ)} :
    Inv live V ⟨len, (idx, sl0) :: st.env, ops⟩ vals := by
  intro x sl hsl hlive
  by_cases hx : x = idx
  · subst hx; rw [hdead] at hlive; cases hlive
  · have hb : (x == idx) = false := by simpa using hx
    have : st.slot x = some sl := by
      simpa [State.slot, List.lookup, hb] using hsl
    exact h x sl this hlive

theorem Inv.update_live {live : Nat → Bool} {V : Values (BinaryFloat dt.format) Δ} {st : State}
    {vals : Nat → Option (FlatValue dt.format)} (h : Inv live V st vals) {idx : Nat} {sl0 : Slot}
    {d : List (BinaryFloat dt.format)} (hok : OwnerOK V sl0.owner sl0.shape d) {len : Nat}
    {ops : List (GraphSyntax.Operation ℚ)} :
    Inv live V ⟨len, (idx, sl0) :: st.env, ops⟩
      (fun x => if x = idx then some ⟨sl0.shape, d⟩ else vals x) := by
  intro x sl hsl hlive
  by_cases hx : x = idx
  · subst hx
    simp only [State.slot, List.lookup, beq_self_eq_true, Option.some.injEq] at hsl
    subst hsl
    exact ⟨d, by simp, hok⟩
  · have hb : (x == idx) = false := by simpa using hx
    have : st.slot x = some sl := by
      simpa [State.slot, List.lookup, hb] using hsl
    obtain ⟨d', hd', hok'⟩ := h x sl this hlive
    exact ⟨d', by simp [hx, hd'], hok'⟩

theorem runF_simulates {entries : List Nat} {dag : Model.Dag} {live : Nat → Bool}
    {inputs : Model.LoadStoreName → Option (List (BinaryFloat dt.format))}
    {ρ : Values (BinaryFloat dt.format) Γ} {opsF : List (GraphSyntax.Operation (BinaryFloat dt.format))}
    {p : Prefix (BinaryFloat dt.format) Γ Δ}
    (hdp : GraphSyntax.decodePrefix Γ opsF = some ⟨Δ, p⟩)
    (hinputs : InputsOK dt dag entries inputs ρ) (hclosed : LiveClosed dag live) :
    ∀ {nodes : List Model.DagNode} {i : Nat} {st : State}
      {acc : List (GraphSyntax.Operation (BinaryFloat dt.format))} {stF : State}
      {vals : Nat → Option (FlatValue dt.format)},
      (∀ n ∈ nodes, n ∈ dag.nodes.val) →
      runF dt (entries.map some) Γ i st acc nodes = some (stF, opsF) →
      acc.map (mapData value) = st.ops → st.len = Γ.length + acc.length →
      Inv live (prefixEval p ρ) st vals →
        ∃ vals', specRun dt inputs live nodes vals = some vals' ∧
          Inv live (prefixEval p ρ) stF vals'
  | [], i, st, acc, stF, vals, _, h, _, _, hinv => by
      simp only [runF, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨vals, rfl, hinv⟩
  | node :: rest, i, st, acc, stF, vals, hsub, h, hacc, hlen, hinv => by
      have hmem : node ∈ dag.nodes.val := hsub node (by simp)
      have hsub' : ∀ n ∈ rest, n ∈ dag.nodes.val := fun n hn => hsub n (by simp [hn])
      simp only [runF, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
      obtain ⟨st1, h1, acc1, hacc1, h2⟩ := h
      have hΔ := (decodePrefix_spec opsF ⟨Δ, p⟩ hdp).1
      by_cases hl : ∃ l, node.op = .Load l
      · obtain ⟨l, hl⟩ := hl
        obtain ⟨j, out, hout, hid, hfind, hΓj, rfl⟩ := step_load h1 hl
        subst hid
        have hacc1' : acc1 = acc := by
          unfold liftStep at hacc1; rw [hl] at hacc1; exact (Option.some.inj hacc1).symm
        subst hacc1'
        by_cases hlv : live node.id.val = true
        · obtain ⟨u, hu⟩ := var_decode_of_getElem? hΓj
          have hj : entries[j]? = some node.id.val := by
            obtain ⟨hjlt, hpj, _⟩ := List.findIdx?_eq_some_iff_getElem.mp hfind
            simp only [List.getElem_map, beq_iff_eq, Option.some.injEq] at hpj
            simp [List.getElem?_eq_getElem (by simpa using hjlt), hpj]
          obtain ⟨xs, hxs, hdec⟩ := hinputs j node l u hmem hl hj hu
          have hjlt : j < Γ.length := (List.getElem?_eq_some_iff.mp hΓj).1
          have hle := prefix_length_le p
          have hok : OwnerOK (prefixEval p ρ) (Γ.length - 1 - j) out xs := by
            have hoff : (p.embedding u).offset + (Γ.length - 1 - j) + 1 = Δ.length := by
              rw [embedding_offset, hu]; omega
            refine ⟨⟨p.embedding u, hoff⟩, fun w hw => ?_⟩
            have : w = p.embedding u := Var.offset_injective (by omega)
            subst this
            rw [get_embedding_ieee]; exact hdec
          obtain ⟨vals', hrun, hinv'⟩ := runF_simulates hdp hinputs hclosed hsub' h2 hacc hlen
            (Inv.update_live (sl0 := ⟨Γ.length - 1 - j, out⟩) hinv hok)
          refine ⟨vals', ?_, hinv'⟩
          simp only [specRun, hlv, if_true, hout, Option.bind_some, nodeKernel, hl, hxs]
          exact hrun
        · have hdead : live node.id.val = false := by simpa using hlv
          obtain ⟨vals', hrun, hinv'⟩ := runF_simulates hdp hinputs hclosed hsub' h2 hacc hlen
            (Inv.update_dead (sl0 := ⟨Γ.length - 1 - j, out⟩) hinv hdead)
          exact ⟨vals', by simpa [specRun, hdead] using hrun, hinv'⟩
      · have hne : ∀ l, node.op ≠ .Load l := fun l h => hl ⟨l, h⟩
        obtain ⟨out, oq, hout, hid, hdec, rfl⟩ := step_nonload h1 hne
        subst hid
        obtain ⟨of, hlift, rfl⟩ := liftStep_nonload hne hout rfl hacc1
        have hacc' : (acc ++ [of]).map (mapData value) = st.ops ++ [oq] := by
          rw [List.map_append, hacc, List.map_singleton, liftOp_sound hlift]
        have hlen' : st.len + 1 = Γ.length + (acc ++ [of]).length := by
          simp only [hlen, List.length_append, List.length_singleton]; omega
        have hpre := runF_prefix h2
        have hk : acc.length < opsF.length := by
          have := hpre.length_le; simp at this; omega
        have hkof : opsF[acc.length] = of := by
          obtain ⟨tl, htl⟩ := hpre
          simp [← htl]
        obtain ⟨t, op', v, hb, henc, hv⟩ := bindings_decode opsF p hdp acc.length hk
        rw [hkof] at henc
        have hop : GraphSyntax.Operation.decode Δ
            (shiftPorts (Δ.length - (Γ.length + acc.length)) of) = some ⟨t, op'⟩ := by
          rw [← henc, Op.decode_encode]
        have hd : st.len + (Δ.length - (Γ.length + acc.length)) = Δ.length := by omega
        by_cases hlv : live node.id.val = true
        · obtain ⟨data, hker, rfl, hdata⟩ := kernel_correct (inputs := inputs) hinv
            (hclosed node hmem hlv) hne hdec hlift hop hd
          have hbind := bindings_ieee p ρ _ op' v (List.mem_of_getElem? hb)
          have hok : OwnerOK (prefixEval p ρ) st.len t data := by
            refine ⟨⟨v, by omega⟩, fun w hw => ?_⟩
            have : w = v := Var.offset_injective (by omega)
            subst this
            rw [← hbind]; exact hdata
          obtain ⟨vals', hrun, hinv'⟩ := runF_simulates hdp hinputs hclosed hsub' h2 hacc' hlen'
            (Inv.update_live (sl0 := ⟨st.len, t⟩) hinv hok)
          refine ⟨vals', ?_, hinv'⟩
          simp only [specRun, hlv, if_true, hout, Option.bind_some, hker]
          exact hrun
        · have hdead : live node.id.val = false := by simpa using hlv
          obtain ⟨vals', hrun, hinv'⟩ := runF_simulates hdp hinputs hclosed hsub' h2 hacc' hlen'
            (Inv.update_dead (sl0 := ⟨st.len, out⟩) hinv hdead)
          exact ⟨vals', by simpa [specRun, hdead] using hrun, hinv'⟩

end Main

section Circuit

variable {dt : DType}

theorem Inv.init {live : Nat → Bool} {Γ Δ : Context} {V : Values (BinaryFloat dt.format) Δ} :
    Inv live V (State.init Γ) (fun _ => none) := by
  intro x sl hsl
  simp [State.slot, State.init] at hsl

theorem runF_len {dtt : DType} {entries : List (Option Nat)} {Γ : Context} {i : Nat} {st : State}
    {acc : List (GraphSyntax.Operation (BinaryFloat dtt.format))} {nodes : List Model.DagNode}
    {stF : State} {opsF} (hacc : acc.map (mapData value) = st.ops)
    (hlen : st.len = Γ.length + acc.length)
    (h : runF dtt entries Γ i st acc nodes = some (stF, opsF)) :
    stF.len = Γ.length + opsF.length := by
  obtain ⟨hrun, hops⟩ := runF_sound hacc h
  have := (run_len hrun).1
  have h1 : acc.length = st.ops.length := by rw [← hacc, List.length_map]
  have h2 : opsF.length = stF.ops.length := by rw [← hops, List.length_map]
  omega

/-- **The flat-buffer node loop computes the IEEE reference evaluation.** For a graph the
float decoder admits, inputs bound by name, and a live mask closed under inputs containing
every root, the loop succeeds and each root's buffer is, row-major and bitwise, the
corresponding root of `ieeeEval`. -/
theorem specRun_simulates {dag : Model.Dag} {entries : List Nat} {Ω Γ : Context}
    {cf : Circuit (BinaryFloat dt.format) Γ Ω} (h : decodeCircuitF dt dag entries Ω = some ⟨Γ, cf⟩)
    {live : Nat → Bool} {inputs : Model.LoadStoreName → Option (List (BinaryFloat dt.format))}
    {ρ : Values (BinaryFloat dt.format) Γ} (hinputs : InputsOK dt dag entries inputs ρ)
    (hclosed : LiveClosed dag live) (hroots : ∀ r ∈ dag.roots.val, live r.val = true) :
    ∃ vals, specRun dt inputs live dag.nodes.val (fun _ => none) = some vals ∧
      ∀ {s : Shape} (w : Var Ω s) (r : Usize), dag.roots.val[w.offset]? = some r →
        ∃ d, vals r.val = some ⟨s, d⟩ ∧ Tensor.ofList s d = some (w.get (ieeeEval cf ρ)) := by
  simp only [decodeCircuitF, decodeGraphF, Option.bind_eq_bind, Option.bind_eq_some_iff,
    Option.pure_def, Option.some.injEq] at h
  obtain ⟨⟨Γ', st, ops⟩, ⟨u, hnd, Γ0, hΓ, ⟨st0, ops0⟩, hrunF, hq⟩, offs, hoffs, c, hc, h⟩ := h
  simp only [Prod.mk.injEq] at hq
  obtain ⟨rfl, rfl, rfl⟩ := hq
  cases h
  obtain ⟨Δ, code, σ, hdp, hσ, rfl⟩ := Target.decode_spec hc
  have hlenF := runF_len (by rfl) (by simp [State.init]) hrunF
  have hΔ := (decodePrefix_spec _ ⟨Δ, code⟩ hdp).1
  obtain ⟨vals, hrun, hinv⟩ := runF_simulates (inputs := inputs) (live := live) (ρ := ρ) hdp
    hinputs hclosed (fun n hn => hn) hrunF (by rfl) (by simp [State.init]) Inv.init
  refine ⟨vals, hrun, fun {s} w r hr => ?_⟩
  obtain ⟨i, hi, sl, hsl, hrel⟩ := forall₂_getElem? (decodeRootOffsets_spec hoffs) hr
  have hvar := decodeRoots_get hσ w hi
  obtain ⟨d, hvd, hok⟩ := hinv r.val sl hsl (hroots r (List.mem_of_getElem? hr))
  obtain ⟨w0, hw0⟩ := hok.1
  have hoff := Var.offset_of_decode hvar
  dsimp only at hrel hlenF hΔ
  have hs : s = sl.shape := by
    have h1 := Var.getElem_offset (σ w)
    have h2 := Var.getElem_offset w0
    rw [hoff] at h1
    have : w0.offset = i := by omega
    rw [this, h1] at h2
    exact Option.some.inj h2
  subst hs
  refine ⟨d, hvd, ?_⟩
  simp only [ieeeEval, Renaming.get_read]
  exact hok.2 (σ w) (by omega)

theorem decodeTensor_of_ofList {f : BinaryFormat} {s : Shape} {d : List (BinaryFloat f)}
    {x : Tensor (BinaryFloat f) s} {y : Tensor ℚ s} (hx : Tensor.ofList s d = some x)
    (hr : Realizes x y) : decodeTensor s d = some y := by
  have hfin : ∀ e ∈ d, realize e = some (value e) := by
    intro e he
    obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem he
    have hlen := ofList_length hx
    obtain ⟨hi, hxi⟩ := ofList_apply hx ((Index.equivFin s).symm ⟨k, by omega⟩)
    have hflat : ((Index.equivFin s).symm ⟨k, by omega⟩).flat = k := by
      rw [← Index.equivFin_val]; simp
    have := hr ((Index.equivFin s).symm ⟨k, by omega⟩)
    rw [hxi] at this
    simp only [hflat] at this
    exact realize_eq_some.mpr ⟨(realize_eq_some.mp this).1, rfl⟩
  have hrl : realizeList d = some (d.map value) := by
    rw [realizeList_eq_some, List.forall₂_map_right_iff]
    exact List.forall₂_same.mpr hfin
  have hmap := ofList_map value (s := s) d
  rw [hx, Option.map_some] at hmap
  simp only [decodeTensor, Option.bind_eq_bind, hrl, Option.bind_some, hmap]
  congr 1
  funext i
  exact (realize_eq_some.mp (hr i)).2

/-- **Numerical observation of the Rust node loop.** For a graph the bridge decoder admits
(rational circuit `c`), under inputs bound by name that realize to `ρq`, a live mask closed
under inputs containing the roots, and an exact run of the graph's IEEE program: the loop
succeeds and every root buffer realizes, row-major, to the reference evaluation
`c.eval ρq` at that root. -/
theorem specRun_realize {dag : Model.Dag} {entries : List Nat} {Ω Γ : Context}
    {c : Circuit ℚ Γ Ω} (h : decodeCircuit dt dag entries Ω = some ⟨Γ, c⟩) :
    ∃ cf, decodeCircuitF dt dag entries Ω = some ⟨Γ, cf⟩ ∧ ratCircuit cf = c ∧
      ∀ {live : Nat → Bool} {inputs : Model.LoadStoreName → Option (List (BinaryFloat dt.format))}
        {ρf : Values (BinaryFloat dt.format) Γ} {ρq : Values ℚ Γ},
        InputsOK dt dag entries inputs ρf → Realizes ρf ρq → ExactRun cf ρf →
        LiveClosed dag live → (∀ r ∈ dag.roots.val, live r.val = true) →
        ∃ vals, specRun dt inputs live dag.nodes.val (fun _ => none) = some vals ∧
          ∀ {s : Shape} (w : Var Ω s) (r : Usize), dag.roots.val[w.offset]? = some r →
            ∃ d, vals r.val = some ⟨s, d⟩ ∧ decodeTensor s d = some (w.get (c.eval ρq)) := by
  obtain ⟨cf, hcf, hrat, hreal⟩ := decoded_graph_realize h
  refine ⟨cf, hcf, hrat, fun hin hρ hex hcl hro => ?_⟩
  obtain ⟨vals, hrun, hroot⟩ := specRun_simulates hcf hin hcl hro
  refine ⟨vals, hrun, fun w r hr => ?_⟩
  obtain ⟨d, hvd, hd⟩ := hroot w r hr
  exact ⟨d, hvd, decodeTensor_of_ofList hd ((hreal hρ hex).get w)⟩

end Circuit

end ChelisBridge.Numeric
