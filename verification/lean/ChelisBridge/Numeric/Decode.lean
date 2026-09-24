import ChelisBridge.Numeric
import ChelisBridge.Decode

/-!
# Float-level decoding: the IEEE program of a Rust graph

`decodeCircuitF dt dag entries Ω` reads the same generated `Dag` as `decodeCircuit`, but
keeps each constant as the actual IEEE encodings of its Rust payload (a scalar `Const`
splatted to the node's shape, a `ConstTensor` buffer as stored) instead of their exact
rationals. It runs the bridge decoder `step` itself on every node, and for every emitted
operation takes the node's payload with a checked lift (`liftOp`): a payload is accepted
only if its realized values are the decoded rationals.

`decodeCircuitF_sound`: the rational image of the float program is exactly the decoded
reference circuit, `ratCircuit cf`. So `ieeeEval cf` is an IEEE evaluation *of the decoded
program* whose constants are bit-for-bit the Rust payloads, including signed zeros.
-/

namespace ChelisBridge.Numeric

open Aeneas.Std LaCaDiLECore ChelisBridge

/-! ## Mapping constant data through the reference syntax -/

section MapData

variable {C D : Type} (g : C → D)

def mapData : GraphSyntax.Operation C → GraphSyntax.Operation D
  | .const s xs => .const s (xs.map g)
  | .copy s i => .copy s i
  | .add s i j => .add s i j
  | .mul s i j => .mul s i j
  | .sum s n t k i => .sum s n t k i
  | .insert s n t k i => .insert s n t k i
  | .expand pre post n i => .expand pre post n i

def mapOp {Γ : Context} {s : Shape} : Op C Γ s → Op D Γ s
  | .const x => .const (fun i => g (x i))
  | .copy v => .copy v
  | .add v w => .add v w
  | .mul v w => .mul v w
  | .sum a v => .sum a v
  | .insert a v => .insert a v
  | .expand pre post n v => .expand pre post n v

def mapPrefix {Γ Δ : Context} : Prefix C Γ Δ → Prefix D Γ Δ
  | .nil => .nil
  | .bind op rest => .bind (mapOp g op) (mapPrefix rest)

theorem ofList_map {s : Shape} (xs : List C) :
    Tensor.ofList s (xs.map g) = (Tensor.ofList s xs).map (fun x i => g (x i)) := by
  unfold Tensor.ofList
  by_cases h : xs.length = s.prod
  · rw [dif_pos (by simpa using h), dif_pos h]
    simp
  · rw [dif_neg (by simpa using h), dif_neg h]
    rfl

theorem decode_mapData {Γ : Context} (o : GraphSyntax.Operation C) :
    (mapData g o).decode Γ = (o.decode Γ).map (fun p => ⟨p.1, mapOp g p.2⟩) := by
  cases o with
  | const s xs =>
      simp only [mapData, GraphSyntax.Operation.decode, ofList_map]
      cases Tensor.ofList s xs <;> rfl
  | copy s i =>
      simp only [mapData, GraphSyntax.Operation.decode]
      cases Var.decode Γ s i <;> rfl
  | add s i j =>
      simp only [mapData, GraphSyntax.Operation.decode]
      cases Var.decode Γ s i <;> cases Var.decode Γ s j <;> rfl
  | mul s i j =>
      simp only [mapData, GraphSyntax.Operation.decode]
      cases Var.decode Γ s i <;> cases Var.decode Γ s j <;> rfl
  | sum s n t k i =>
      simp only [mapData, GraphSyntax.Operation.decode]
      cases Axis.decode s n t k <;> cases Var.decode Γ s i <;> rfl
  | insert s n t k i =>
      simp only [mapData, GraphSyntax.Operation.decode]
      cases Axis.decode s n t k <;> cases Var.decode Γ t i <;> rfl
  | expand pre post n i =>
      simp only [mapData, GraphSyntax.Operation.decode]
      cases Var.decode Γ (pre ++ 1 :: post) i <;> rfl

theorem decodePrefix_mapData :
    ∀ {Γ : Context} (ops : List (GraphSyntax.Operation C)),
      GraphSyntax.decodePrefix Γ (ops.map (mapData g)) =
        (GraphSyntax.decodePrefix Γ ops).map (fun p => ⟨p.1, mapPrefix g p.2⟩)
  | Γ, [] => rfl
  | Γ, o :: rest => by
      simp only [List.map_cons, GraphSyntax.decodePrefix, decode_mapData]
      cases h : o.decode Γ with
      | none => rfl
      | some p =>
          obtain ⟨s, op⟩ := p
          simp only [Option.map_some, Option.bind_eq_bind, Option.bind_some]
          rw [decodePrefix_mapData rest]
          cases GraphSyntax.decodePrefix (s :: Γ) rest <;> rfl

theorem targetDecode_mapData {Γ Ω : Context} (tgt : GraphSyntax.Target C) :
    (⟨tgt.operations.map (mapData g), tgt.roots⟩ : GraphSyntax.Target D).decode Γ Ω =
      (tgt.decode Γ Ω).map (fun c => ⟨c.context, mapPrefix g c.code, c.result⟩) := by
  simp only [GraphSyntax.Target.decode, decodePrefix_mapData]
  cases GraphSyntax.decodePrefix Γ tgt.operations with
  | none => rfl
  | some p =>
      obtain ⟨Δ, code⟩ := p
      simp only [Option.map_some, Option.bind_eq_bind, Option.bind_some]
      cases GraphSyntax.decodeRoots Δ Ω tgt.roots <;> rfl

end MapData

variable {f : BinaryFormat}

theorem ratOp_eq_mapOp {Γ : Context} {s : Shape} (op : Op (BinaryFloat f) Γ s) :
    ratOp op = mapOp value op := by
  cases op <;> rfl

theorem ratPrefix_eq_mapPrefix {Γ Δ : Context} (p : Prefix (BinaryFloat f) Γ Δ) :
    ratPrefix p = mapPrefix value p := by
  induction p with
  | nil => rfl
  | bind op rest ih => simp only [ratPrefix, mapPrefix, ih, ratOp_eq_mapOp]

/-! ## Rust payloads -/

/-- The IEEE format of the graph dtype. -/
def _root_.ChelisBridge.DType.format : DType → BinaryFormat
  | .f32 => .binary32
  | .f64 => .binary64

def scalarBits : (dt : DType) → Model.ScalarValue → Option (BinaryFloat dt.format)
  | .f64, ⟨.F64 x⟩ => some x
  | .f32, ⟨.F32 x⟩ => some x
  | _, _ => none

def bufBits : (dt : DType) → Model.Buf → Option (List (BinaryFloat dt.format))
  | .f64, .F64 v => some v.val
  | .f32, .F32 v => some v.val
  | _, _ => none

/-- The IEEE payload of a constant node (a scalar `Const` splatted to the node shape, as
the Rust evaluator materializes it); `none` for every other node. -/
def payload (dt : DType) (op : Model.RiscOp) (out : Shape) : Option (List (BinaryFloat dt.format)) :=
  match op with
  | .Const v => (scalarBits dt v).map fun x => List.replicate out.prod x
  | .ConstTensor data => bufBits dt data.buf
  | _ => none

/-- Checked lift of a decoded rational operation to the float syntax. -/
def liftOp : GraphSyntax.Operation ℚ → Option (List (BinaryFloat f)) →
    Option (GraphSyntax.Operation (BinaryFloat f))
  | .const s d, some xs => if xs.map value = d then some (.const s xs) else none
  | .copy s i, none => some (.copy s i)
  | .add s i j, none => some (.add s i j)
  | .mul s i j, none => some (.mul s i j)
  | .sum s n t k i, none => some (.sum s n t k i)
  | .insert s n t k i, none => some (.insert s n t k i)
  | .expand pre post n i, none => some (.expand pre post n i)
  | _, _ => none

theorem liftOp_sound {o : GraphSyntax.Operation ℚ} {pl : Option (List (BinaryFloat f))}
    {o' : GraphSyntax.Operation (BinaryFloat f)} (h : liftOp o pl = some o') :
    mapData value o' = o := by
  cases o <;> cases pl <;> simp [liftOp] at h
  all_goals (obtain ⟨rfl, rfl⟩ := h; rfl)

/-- Lift the operation a successful `step` emitted for `node` (nothing for a `Load`). -/
def liftStep (dt : DType) (st' : State) (node : Model.DagNode)
    (acc : List (GraphSyntax.Operation (BinaryFloat dt.format))) :
    Option (List (GraphSyntax.Operation (BinaryFloat dt.format))) :=
  match node.op with
  | .Load _ => some acc
  | op =>
      (resolveShape node.output_type.dims).bind fun out =>
        st'.ops.getLast?.bind fun o =>
          (liftOp o (payload dt op out)).map fun o' => acc ++ [o']

/-- The bridge decoder's node pass, additionally lifting each emitted operation with the
node's IEEE payload. -/
def runF (dt : DType) (entries : List (Option Nat)) (Γ : Context) :
    Nat → State → List (GraphSyntax.Operation (BinaryFloat dt.format)) → List Model.DagNode →
      Option (State × List (GraphSyntax.Operation (BinaryFloat dt.format)))
  | _, st, acc, [] => some (st, acc)
  | i, st, acc, node :: rest => do
      let st' ← step dt entries Γ st i node
      let acc' ← liftStep dt st' node acc
      runF dt entries Γ (i + 1) st' acc' rest

theorem step_ops {dt : DType} {entries : List (Option Nat)} {Γ : Context} {st st' : State}
    {idx : Nat} {node : Model.DagNode} (h : step dt entries Γ st idx node = some st') :
    (∃ l, node.op = .Load l ∧ st'.ops = st.ops) ∨
      ((∀ l, node.op ≠ .Load l) ∧ ∃ o, st'.ops = st.ops ++ [o]) := by
  unfold step at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨_, _, _, _, out, _, h⟩ := h
  split at h
  · rename_i l hl
    simp only [Option.bind_eq_some_iff] at h
    obtain ⟨_, _, j, _, _, _, h⟩ := h
    cases Option.some.inj h
    exact Or.inl ⟨l, hl, rfl⟩
  · rename_i hne
    simp only [Option.bind_eq_some_iff] at h
    obtain ⟨o, _, h⟩ := h
    cases Option.some.inj h
    exact Or.inr ⟨fun l hl => hne l hl, o, rfl⟩

theorem runF_sound {dt : DType} {entries : List (Option Nat)} {Γ : Context} :
    ∀ {i : Nat} {st : State} {acc : List (GraphSyntax.Operation (BinaryFloat dt.format))}
      {nodes : List Model.DagNode} {st' : State} {acc'},
      acc.map (mapData value) = st.ops →
      runF dt entries Γ i st acc nodes = some (st', acc') →
        run dt entries Γ i st nodes = some st' ∧ acc'.map (mapData value) = st'.ops
  | _, st, acc, [], st', acc', hacc, h => by
      simp only [runF, Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨rfl, hacc⟩
  | i, st, acc, node :: rest, st', acc', hacc, h => by
      simp only [runF, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
      obtain ⟨st1, h1, acc1, hacc1, h2⟩ := h
      have hinv : acc1.map (mapData value) = st1.ops := by
        unfold liftStep at hacc1
        rcases step_ops h1 with ⟨l, hl, hops⟩ | ⟨hne, o, hops⟩
        · rw [hl] at hacc1
          cases Option.some.inj hacc1
          rw [hops]; exact hacc
        · split at hacc1
          · rename_i l hl; exact absurd hl (hne l)
          · simp only [Option.bind_eq_some_iff, Option.map_eq_some_iff] at hacc1
            obtain ⟨_, _, o1, ho1, o', hlift, rfl⟩ := hacc1
            rw [hops, List.getLast?_append_of_ne_nil _ (List.cons_ne_nil _ _)] at ho1
            simp only [List.getLast?_singleton, Option.some.injEq] at ho1
            subst ho1
            rw [hops, List.map_append, hacc, List.map_singleton, liftOp_sound hlift]
      obtain ⟨hrun, hfin⟩ := runF_sound hinv h2
      refine ⟨?_, hfin⟩
      simp only [run, Option.bind_eq_bind, h1, Option.bind_some]
      exact hrun

/-- Float-level `decodeGraph`. -/
def decodeGraphF (dt : DType) (dag : Model.Dag) (entries : List Nat) :
    Option (Context × State × List (GraphSyntax.Operation (BinaryFloat dt.format))) := do
  guard entries.Nodup
  let Γ ← entryContext dt dag entries
  let (st, ops) ← runF dt (entries.map some) Γ 0 (State.init Γ) [] dag.nodes.val
  pure (Γ, st, ops)

/-- **The IEEE program of a Rust graph** with its own ordered roots. -/
def decodeCircuitF (dt : DType) (dag : Model.Dag) (entries : List Nat) (Ω : Context) :
    Option (Sigma fun Γ : Context => Circuit (BinaryFloat dt.format) Γ Ω) := do
  let (Γ, st, ops) ← decodeGraphF dt dag entries
  let roots ← decodeRootOffsets st dag.roots.val
  let c ← (⟨ops, roots⟩ : GraphSyntax.Target (BinaryFloat dt.format)).decode Γ Ω
  pure ⟨Γ, c⟩

theorem decodeGraphF_sound {dt : DType} {dag : Model.Dag} {entries : List Nat} {Γ : Context}
    {st : State} {ops : List (GraphSyntax.Operation (BinaryFloat dt.format))}
    (h : decodeGraphF dt dag entries = some (Γ, st, ops)) :
    decodeGraph dt dag entries = some (Γ, st) ∧ ops.map (mapData value) = st.ops := by
  simp only [decodeGraphF, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
    Option.some.injEq] at h
  obtain ⟨u, hg, Γ', hΓ, ⟨st1, ops1⟩, hr, h⟩ := h
  simp only [Prod.mk.injEq] at h
  obtain ⟨rfl, rfl, rfl⟩ := h
  obtain ⟨hrun, hops⟩ := runF_sound (by rfl) hr
  refine ⟨?_, hops⟩
  simp only [decodeGraph, Option.bind_eq_bind, hg, Option.bind_some, hΓ, hrun, Option.pure_def]

/-- **Soundness of the float decoder**: the exact rational image of the IEEE program is
the decoded reference circuit. -/
theorem decodeCircuitF_sound {dt : DType} {dag : Model.Dag} {entries : List Nat} {Ω : Context}
    {Γ : Context} {cf : Circuit (BinaryFloat dt.format) Γ Ω}
    (h : decodeCircuitF dt dag entries Ω = some ⟨Γ, cf⟩) :
    decodeCircuit dt dag entries Ω = some ⟨Γ, ratCircuit cf⟩ := by
  simp only [decodeCircuitF, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
    Option.some.injEq] at h
  obtain ⟨⟨Γ', st, ops⟩, hg, roots, hroots, c, hc, h⟩ := h
  obtain ⟨hgraph, hops⟩ := decodeGraphF_sound hg
  have hdec := targetDecode_mapData (Γ := Γ') (Ω := Ω) value
    (⟨ops, roots⟩ : GraphSyntax.Target (BinaryFloat dt.format))
  rw [hc] at hdec
  simp only [hops, Option.map_some] at hdec
  cases h
  simp only [decodeCircuit, graphTarget, Option.bind_eq_bind, hgraph, Option.bind_some,
    hroots, Option.pure_def, hdec]
  simp only [ratCircuit, ratPrefix_eq_mapPrefix]

/-! ## Completeness: every decoded graph has its IEEE program -/

theorem step_ops_nonload {dt : DType} {entries : List (Option Nat)} {Γ : Context} {st st' : State}
    {idx : Nat} {node : Model.DagNode} (h : step dt entries Γ st idx node = some st')
    (hne : ∀ l, node.op ≠ .Load l) :
    ∃ out o, resolveShape node.output_type.dims = some out ∧
      decodeOperation dt st node.op node.inputs.val out = some o ∧ st'.ops = st.ops ++ [o] := by
  unfold step at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨_, _, _, _, out, hout, h⟩ := h
  obtain ⟨o, ho, h⟩ := h
  cases Option.some.inj h
  exact ⟨out, o, hout, ho, rfl⟩

theorem Kind.build_nonconst {k : Kind} {offs : List Nat} {o : GraphSyntax.Operation ℚ}
    (h : k.build offs = some o) (hk : ∀ s d, k ≠ .const s d) : ∀ s d, o ≠ .const s d := by
  cases k <;> rcases offs with _ | ⟨i, _ | ⟨j, _ | ⟨_, _⟩⟩⟩ <;>
    simp [Kind.build] at h <;> subst h <;> simp_all

theorem classify_nonconst {dt : DType} {st : State} {op : Model.RiscOp} {inputs : List Usize}
    {out : Shape} {k : Kind} (h : classify dt st op inputs out = some k)
    (hc : ∀ v, op ≠ .Const v) (ht : ∀ d, op ≠ .ConstTensor d) : ∀ s d, k ≠ .const s d := by
  intro s d hk
  subst hk
  cases op with
  | Const v => exact hc v rfl
  | ConstTensor data => exact ht data rfl
  | Copy =>
      simp only [classify, classifyCopy] at h
      split at h
      · split at h <;> cases h
      · cases h
  | Add => obtain ⟨_, _, _, _, _, h⟩ := classify_add h; cases h
  | Mul => obtain ⟨_, _, _, _, _, h⟩ := classify_mul h; cases h
  | Sum axis acc =>
      simp only [classify, classifySum] at h
      split at h
      · split at h
        · simp only [Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
          obtain ⟨_, _, _, _, h⟩ := h
          cases h
        · cases h
      · cases h
  | Expand axis size =>
      obtain ⟨_, _, _, _, _, _, _, _, h⟩ := classify_expand h
      rcases h with ⟨_, h⟩ | ⟨_, _, _, h⟩ <;> cases h
  | Reshape ns =>
      obtain ⟨_, _, _, _, _, _, h⟩ := classify_reshape h
      rcases h with ⟨_, _, h⟩ | ⟨_, _, h⟩ <;> cases h
  | _ => simp [classify] at h

theorem scalarBits_of_decodeScalar {dt : DType} {v : Model.ScalarValue} {q : ℚ}
    (h : decodeScalar dt v = some q) : ∃ x, scalarBits dt v = some x ∧ value x = q := by
  rcases v with ⟨b⟩
  cases dt <;> cases b <;> simp [decodeScalar] at h <;>
    exact ⟨_, rfl, (realize_eq_some.mp h).2⟩

theorem map_value_of_forall₂ {f : BinaryFormat} :
    ∀ {xs : List (BinaryFloat f)} {qs : List ℚ},
      List.Forall₂ (fun x q => realize x = some q) xs qs → xs.map value = qs
  | _, _, .nil => rfl
  | _, _, .cons hx h => by
      simp only [List.map_cons, (realize_eq_some.mp hx).2, map_value_of_forall₂ h]

theorem map_value_of_realizeList {f : BinaryFormat} {xs : List (BinaryFloat f)} {qs : List ℚ}
    (h : realizeList xs = some qs) : xs.map value = qs :=
  map_value_of_forall₂ (realizeList_eq_some.mp h)

theorem bufBits_of_decodeBuf {dt : DType} {buf : Model.Buf} {qs : List ℚ}
    (h : decodeBuf dt buf = some qs) : ∃ xs, bufBits dt buf = some xs ∧ xs.map value = qs := by
  cases dt <;> cases buf <;> simp [decodeBuf] at h <;>
    exact ⟨_, rfl, map_value_of_realizeList h⟩

theorem payload_none {dt : DType} {op : Model.RiscOp} {out : Shape}
    (hc : ∀ v, op ≠ .Const v) (ht : ∀ d, op ≠ .ConstTensor d) : payload dt op out = none := by
  unfold payload
  split
  · rename_i v; exact absurd rfl (hc v)
  · rename_i d; exact absurd rfl (ht d)
  · rfl

theorem liftOp_nonconst {f : BinaryFormat} {o : GraphSyntax.Operation ℚ}
    (h : ∀ s d, o ≠ .const s d) : ∃ o', liftOp (f := f) o none = some o' := by
  cases o with
  | const s d => exact absurd rfl (h s d)
  | _ => exact ⟨_, rfl⟩

/-- Every operation the bridge decoder emits has a checked IEEE lift. -/
theorem liftOp_decodeOperation {dt : DType} {st : State} {op : Model.RiscOp}
    {inputs : List Usize} {out : Shape} {o : GraphSyntax.Operation ℚ}
    (h : decodeOperation dt st op inputs out = some o) :
    ∃ o', liftOp o (payload dt op out) = some o' := by
  simp only [decodeOperation, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨k, hk, offs, hoffs, hb⟩ := h
  by_cases hc : ∃ v, op = .Const v
  · obtain ⟨v, rfl⟩ := hc
    obtain ⟨q, rfl, hq, rfl⟩ := classify_const hk
    simp only [valueInputs, List.mapM_nil, Option.pure_def, Option.some.injEq] at hoffs
    subst hoffs
    simp only [Kind.build, Option.some.injEq] at hb
    subst hb
    obtain ⟨x, hx, rfl⟩ := scalarBits_of_decodeScalar hq
    exact ⟨.const out (List.replicate out.prod x), by simp [payload, hx, liftOp, List.map_replicate]⟩
  by_cases ht : ∃ d, op = .ConstTensor d
  · obtain ⟨d, rfl⟩ := ht
    obtain ⟨qs, rfl, hq, rfl⟩ := classify_constTensor hk
    simp only [valueInputs, List.mapM_nil, Option.pure_def, Option.some.injEq] at hoffs
    subst hoffs
    simp only [Kind.build, Option.some.injEq] at hb
    subst hb
    obtain ⟨xs, hx, rfl⟩ := bufBits_of_decodeBuf hq
    exact ⟨.const out xs, by simp [payload, hx, liftOp]⟩
  have hc' : ∀ v, op ≠ .Const v := fun v hv => hc ⟨v, hv⟩
  have ht' : ∀ d, op ≠ .ConstTensor d := fun d hd => ht ⟨d, hd⟩
  rw [payload_none hc' ht']
  exact liftOp_nonconst (Kind.build_nonconst hb (classify_nonconst hk hc' ht'))

theorem runF_complete {dt : DType} {entries : List (Option Nat)} {Γ : Context} :
    ∀ {i : Nat} {st : State} (acc : List (GraphSyntax.Operation (BinaryFloat dt.format)))
      {nodes : List Model.DagNode} {st' : State},
      run dt entries Γ i st nodes = some st' → ∃ acc', runF dt entries Γ i st acc nodes = some (st', acc')
  | _, st, acc, [], st', h => by
      simp only [run, Option.some.injEq] at h
      subst h
      exact ⟨acc, rfl⟩
  | i, st, acc, node :: rest, st', h => by
      simp only [run, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
      obtain ⟨st1, h1, h2⟩ := h
      obtain ⟨acc1, hacc1⟩ : ∃ acc1, liftStep dt st1 node acc = some acc1 := by
        unfold liftStep
        split
        · exact ⟨acc, rfl⟩
        · rename_i hne
          obtain ⟨out, o, hout, ho, hops⟩ := step_ops_nonload h1 hne
          obtain ⟨o', hlift⟩ := liftOp_decodeOperation ho
          refine ⟨acc ++ [o'], ?_⟩
          simp [hout, hops, hlift]
      obtain ⟨acc', h3⟩ := runF_complete acc1 h2
      exact ⟨acc', by simp only [runF, Option.bind_eq_bind, h1, Option.bind_some, hacc1, h3]⟩

/-- **Completeness of the float decoder**: whenever the bridge decoder admits a graph, its
IEEE program exists, and its exact rational image is the decoded circuit. -/
theorem decodeCircuitF_complete {dt : DType} {dag : Model.Dag} {entries : List Nat}
    {Ω Γ : Context} {c : Circuit ℚ Γ Ω}
    (h : decodeCircuit dt dag entries Ω = some ⟨Γ, c⟩) :
    ∃ cf, decodeCircuitF dt dag entries Ω = some ⟨Γ, cf⟩ ∧ ratCircuit cf = c := by
  simp only [decodeCircuit, graphTarget, Option.bind_eq_bind, Option.bind_eq_some_iff,
    Option.pure_def, Option.some.injEq] at h
  obtain ⟨⟨Γ', tgt⟩, ⟨⟨Γ'', st⟩, hg, roots, hroots, htgt⟩, c', hc, h⟩ := h
  simp only [Prod.mk.injEq] at htgt
  obtain ⟨rfl, rfl⟩ := htgt
  cases h
  have hg' := hg
  simp only [decodeGraph, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
    Option.some.injEq, Prod.mk.injEq] at hg'
  obtain ⟨u, hnd, Γ0, hΓ, st0, hrun, rfl, rfl⟩ := hg'
  obtain ⟨ops, hrunF⟩ := runF_complete [] hrun
  have hops := (runF_sound (by rfl) hrunF).2
  have hdec := targetDecode_mapData (Γ := Γ0) (Ω := Ω) value
    (⟨ops, roots⟩ : GraphSyntax.Target (BinaryFloat dt.format))
  simp only [hops] at hdec
  rw [hc] at hdec
  obtain ⟨cf, hcf, hmap⟩ := Option.map_eq_some_iff.mp hdec.symm
  refine ⟨cf, ?_, ?_⟩
  · simp only [decodeCircuitF, decodeGraphF, Option.bind_eq_bind, hnd, Option.bind_some, hΓ,
      hrunF, Option.pure_def, hroots, hcf]
  · rw [← hmap]; simp only [ratCircuit, ratPrefix_eq_mapPrefix]

/-- **Numerical observation for a decoded Rust graph** (IEEE reference side of bridge
item 3): an admitted graph has an IEEE program with the Rust constant bits, and on any
finite inputs where that program runs exactly, realizing every root is the decoded
reference circuit's evaluation on the realized inputs. -/
theorem decoded_graph_realize {dt : DType} {dag : Model.Dag} {entries : List Nat}
    {Ω Γ : Context} {c : Circuit ℚ Γ Ω}
    (h : decodeCircuit dt dag entries Ω = some ⟨Γ, c⟩) :
    ∃ cf, decodeCircuitF dt dag entries Ω = some ⟨Γ, cf⟩ ∧ ratCircuit cf = c ∧
      ∀ {ρf : Values (BinaryFloat dt.format) Γ} {ρq : Values ℚ Γ}, Realizes ρf ρq →
        ExactRun cf ρf → Realizes (ieeeEval cf ρf) (c.eval ρq) := by
  obtain ⟨cf, hcf, hrat⟩ := decodeCircuitF_complete h
  refine ⟨cf, hcf, hrat, fun hρ hex => ?_⟩
  rw [← hrat]
  exact ieeeEval_realize cf hρ hex

end ChelisBridge.Numeric
