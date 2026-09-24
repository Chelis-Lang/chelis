import ChelisBridge.Tensor
import ChelisBridge.Model
import LaCaDiLECore.GraphSyntax

/-!
# Decoding the generated Rust `Dag` into the reference graph syntax

The decoder reads the actual generated `Dag` (initializers, ports and ordered roots); it
never consults reference-compiler output. It admits exactly the seven-operation fragment:

* `Load` nodes are the requested entries: entry `j` of the ordered entry list is context
  position `j` (reverse-context offset `j`, owner `|Γ| - 1 - j`); a `Load` that is not a
  requested entry is rejected.
* `Const`/`ConstTensor` with a `ScalarValue`/`Buf` of the graph dtype become `.const` with
  the exact binary rational of every element; NaN/infinity are rejected. A scalar `Const`
  is splatted to the node's shape, as the Rust evaluator does.
* `Add`/`Mul` require both operand shapes to equal the output shape: Rust's rank-zero
  broadcasting is rejected.
* `Sum axis acc` requires `acc` to be the graph dtype (homogeneous accumulator).
* `Expand axis size`: output rank = input rank + 1 is `insert` at `axis`; same rank
  requires input extent 1 at `axis` and all other extents unchanged and is
  `expand pre post n`. `size` must be `Lit n` (no other input) or a shape-witness
  `InputAxis {tensor := 1, axis := Lit a}` whose witness extent is `n`; the witness is
  a shape-only edge, never a value port.
* `Copy` is `copy`. `Reshape` is admitted only as a singleton-axis insertion (all target
  dims `Lit` and equal to the output shape), decoded as `insert` of extent 1.
* Everything else (including `Cast`, rank-0 broadcasts, dtype mixtures) is `none`.

Shapes are the node's `output_type` dims resolved to literal extents (`Lit n` or a named
dim with a known size); every node must carry the graph dtype; node ids must equal
their position (append-index) and references must point backwards. Node ids are mapped
to owners, and a reference is emitted as the de Bruijn offset
`len - 1 - owner`, so `owner + offset + 1 = len` (`State.offset_spec`).
-/

namespace ChelisBridge

open Aeneas.Std LaCaDiLECore

section ReferenceInverse

variable {C : Type}

theorem Var.offset_of_decode :
    ∀ {Γ : Context} {s : Shape} {i : Nat} {v : Var Γ s}, Var.decode Γ s i = some v → v.offset = i
  | [], _, _, _, h => by simp [Var.decode] at h
  | t :: Γ, s, 0, v, h => by
      simp only [Var.decode] at h
      split at h
      · rename_i hts
        subst hts
        cases Option.some.inj h
        rfl
      · contradiction
  | t :: Γ, s, k + 1, v, h => by
      simp only [Var.decode] at h
      obtain ⟨w, hw, rfl⟩ := Option.map_eq_some_iff.mp h
      simp [Var.offset, Var.offset_of_decode hw]

theorem Axis.position_of_decode :
    ∀ {s : Shape} {n : Nat} {t : Shape} {k : Nat} {a : Axis s n t},
      Axis.decode s n t k = some a → a.position = k
  | [], _, _, _, _, h => by simp [Axis.decode] at h
  | m :: s, n, t, 0, a, h => by
      simp only [Axis.decode] at h
      split at h
      · rename_i hmn
        split at h
        · rename_i hst
          subst hmn; subst hst
          cases Option.some.inj h
          rfl
        · contradiction
      · contradiction
  | m :: s, n, [], k + 1, a, h => by simp [Axis.decode] at h
  | m :: s, n, j :: t, k + 1, a, h => by
      simp only [Axis.decode] at h
      split at h
      · rename_i hmj
        subst hmj
        obtain ⟨b, hb, rfl⟩ := Option.map_eq_some_iff.mp h
        simp [Axis.position, Axis.position_of_decode hb]
      · contradiction

/-- A successfully decoded typed operation re-encodes to exactly the decoded syntax:
shapes, axis positions, constants and every port offset are retained. -/
theorem Operation.encode_of_decode {Γ : Context} (o : GraphSyntax.Operation C)
    (p : Sigma (Op C Γ)) (h : o.decode Γ = some p) : p.2.encode = o := by
  cases o with
  | const s xs =>
      simp only [GraphSyntax.Operation.decode] at h
      obtain ⟨x, hx, rfl⟩ := Option.map_eq_some_iff.mp h
      simp [Op.encode, Tensor.toList_ofList s xs x hx]
  | copy s i =>
      simp only [GraphSyntax.Operation.decode] at h
      obtain ⟨v, hv, rfl⟩ := Option.map_eq_some_iff.mp h
      simp [Op.encode, Var.offset_of_decode hv]
  | add s i j =>
      simp only [GraphSyntax.Operation.decode, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at h
      obtain ⟨v, hv, w, hw, rfl⟩ := h
      simp [Op.encode, Var.offset_of_decode hv, Var.offset_of_decode hw]
  | mul s i j =>
      simp only [GraphSyntax.Operation.decode, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at h
      obtain ⟨v, hv, w, hw, rfl⟩ := h
      simp [Op.encode, Var.offset_of_decode hv, Var.offset_of_decode hw]
  | sum s n t k i =>
      simp only [GraphSyntax.Operation.decode, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at h
      obtain ⟨a, ha, v, hv, rfl⟩ := h
      simp [Op.encode, Var.offset_of_decode hv, Axis.position_of_decode ha]
  | insert s n t k i =>
      simp only [GraphSyntax.Operation.decode, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at h
      obtain ⟨a, ha, v, hv, rfl⟩ := h
      simp [Op.encode, Var.offset_of_decode hv, Axis.position_of_decode ha]
  | expand pre post n i =>
      simp only [GraphSyntax.Operation.decode] at h
      obtain ⟨v, hv, rfl⟩ := Option.map_eq_some_iff.mp h
      simp [Op.encode, Var.offset_of_decode hv]

/-- Decoding a straight-line prefix extends the context by one binding per operation and
retains the operation list exactly. -/
theorem decodePrefix_spec :
    ∀ {Γ : Context} (ops : List (GraphSyntax.Operation C)) (p : Sigma (Prefix C Γ)),
      GraphSyntax.decodePrefix Γ ops = some p →
        p.1.length = Γ.length + ops.length ∧ p.2.encode = ops
  | Γ, [], p, h => by
      simp only [GraphSyntax.decodePrefix, Option.some.injEq] at h
      subst h
      simp [Prefix.encode]
  | Γ, op :: rest, p, h => by
      simp only [GraphSyntax.decodePrefix, Option.bind_eq_bind, Option.bind_eq_some_iff,
        Option.pure_def, Option.some.injEq] at h
      obtain ⟨⟨s, op'⟩, hop, ⟨Δ, rest'⟩, hrest, rfl⟩ := h
      obtain ⟨hlen, henc⟩ := decodePrefix_spec rest ⟨Δ, rest'⟩ hrest
      refine ⟨?_, ?_⟩
      · simp only [List.length_cons] at hlen ⊢; omega
      · simp only [Prefix.encode, henc, Operation.encode_of_decode op ⟨s, op'⟩ hop]

/-- The Lean owner of a variable is fixed by its offset and the context length. -/
theorem Var.owner_eq_of_offset {Γ : Context} {s : Shape} (v : Var Γ s) {o i : Nat}
    (hi : v.offset = i) (h : o + i + 1 = Γ.length) : v.owner = o := by
  have := v.owner_add_offset
  omega

end ReferenceInverse

/-! ## Ports of the reference syntax -/

/-- Value ports of an operation, in operand order, with repetitions. -/
def ports {C : Type} : GraphSyntax.Operation C → List Nat
  | .const _ _ => []
  | .copy _ i => [i]
  | .add _ i j => [i, j]
  | .mul _ i j => [i, j]
  | .sum _ _ _ _ i => [i]
  | .insert _ _ _ _ i => [i]
  | .expand _ _ _ i => [i]

/-- The typed operand variables of a reference operation, in operand order. -/
def portVars {C : Type} {Γ : Context} {s : Shape} : Op C Γ s → List (Sigma (Var Γ))
  | .const _ => []
  | .copy v => [⟨_, v⟩]
  | .add v w => [⟨_, v⟩, ⟨_, w⟩]
  | .mul v w => [⟨_, v⟩, ⟨_, w⟩]
  | .sum _ v => [⟨_, v⟩]
  | .insert _ v => [⟨_, v⟩]
  | .expand _ _ _ v => [⟨_, v⟩]

theorem portVars_offsets {C : Type} {Γ : Context} {s : Shape} (op : Op C Γ s) :
    (portVars op).map (fun v => v.2.offset) = ports op.encode := by
  cases op <;> rfl

/-! ## The decoder -/

/-- The homogeneous floating dtype of an admitted graph. -/
inductive DType where
  | f32
  | f64
  deriving DecidableEq, Repr

def DType.ofPrim : Model.Prim → Option DType
  | .F32 => some .f32
  | .F64 => some .f64
  | _ => none

/-- A resolved literal extent. -/
def resolveDim : Model.DimInfo → Option Nat
  | .Lit n => some n.val
  | .Named _ (some n) => some n.val
  | .Named _ none => none

def resolveShape (dims : alloc.vec.Vec Model.DimInfo) : Option Shape :=
  dims.val.mapM resolveDim

def litDim : Model.RtDim → Option Nat
  | .Lit n => some n.val
  | _ => none

/-- Exact value of a scalar constant of the graph dtype. -/
def decodeScalar : DType → Model.ScalarValue → Option ℚ
  | .f64, ⟨.F64 x⟩ => realize x
  | .f32, ⟨.F32 x⟩ => realize x
  | _, _ => none

/-- Exact values of a constant buffer of the graph dtype. -/
def decodeBuf : DType → Model.Buf → Option (List ℚ)
  | .f64, .F64 v => realizeList v.val
  | .f32, .F32 v => realizeList v.val
  | _, _ => none

/-- A retained Rust value: its stable Lean owner and its resolved shape. -/
structure Slot where
  owner : Nat
  shape : Shape
  deriving DecidableEq, Repr

/-- Decoder state: current context length, node-id bindings (newest first) and the
operations emitted so far (in order). -/
structure State where
  len : Nat
  env : List (Nat × Slot)
  ops : List (GraphSyntax.Operation ℚ)

def State.init (Γ : Context) : State := ⟨Γ.length, [], []⟩

def State.slot (st : State) (id : Nat) : Option Slot := st.env.lookup id

def State.shapeOf (st : State) (id : Nat) : Option Shape := (st.slot id).map Slot.shape

/-- De Bruijn offset of a node id in the current context: `len - 1 - owner`. -/
def State.offset (st : State) (id : Nat) : Option Nat := do
  let sl ← st.slot id
  if sl.owner < st.len then some (st.len - 1 - sl.owner) else none

/-- The offset relation of the bridge contract: `owner + offset + 1 = context.length`. -/
theorem State.offset_spec {st : State} {id i : Nat} (h : st.offset id = some i) :
    ∃ sl, st.slot id = some sl ∧ sl.owner + i + 1 = st.len := by
  simp only [State.offset, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨sl, hsl, h⟩ := h
  split at h
  · cases Option.some.inj h; exact ⟨sl, hsl, by omega⟩
  · contradiction

/-- Shape-only extent argument of `Expand`: a literal, or the extent of a witness input. -/
def sizeAdmissible (st : State) : Model.RtDim → List Usize → Nat → Bool
  | .Lit k, [], n => k.val == n
  | .InputAxis tensor (.Lit a), [w], n =>
      tensor.val == 1 && decide (0 ≤ a.val) &&
        ((st.shapeOf w.val).bind (fun ws => ws[a.val.toNat]?) == some n)
  | _, _, _ => false

/-- Operation kind determined from the node and the shapes of its inputs, before ports. -/
inductive Kind where
  | const (s : Shape) (data : List ℚ)
  | copy (s : Shape)
  | add (s : Shape)
  | mul (s : Shape)
  | sum (input : Shape) (extent : Nat) (output : Shape) (axis : Nat)
  | insert (output : Shape) (extent : Nat) (input : Shape) (axis : Nat)
  | expand (pre post : Shape) (extent : Nat)

def Kind.build : Kind → List Nat → Option (GraphSyntax.Operation ℚ)
  | .const s d, [] => some (.const s d)
  | .copy s, [i] => some (.copy s i)
  | .add s, [i, j] => some (.add s i j)
  | .mul s, [i, j] => some (.mul s i j)
  | .sum s n t k, [i] => some (.sum s n t k i)
  | .insert t n s k, [i] => some (.insert t n s k i)
  | .expand pre post n, [i] => some (.expand pre post n i)
  | _, _ => none

theorem Kind.ports_build {k : Kind} {offs : List Nat} {o : GraphSyntax.Operation ℚ}
    (h : k.build offs = some o) : ports o = offs := by
  cases k <;> rcases offs with _ | ⟨i, _ | ⟨j, _ | ⟨_, _⟩⟩⟩ <;>
    simp [Kind.build] at h <;> subst h <;> rfl

/-- The value operands of a node, in order: all inputs, except that `Expand` has exactly one
value operand (`inputs[0]`); any further `Expand` input is a shape witness. -/
def valueInputs (op : Model.RiscOp) (inputs : List Usize) : List Usize :=
  match op with
  | .Expand _ _ => inputs.take 1
  | _ => inputs

/-- `Const`: a scalar of the graph dtype, splatted to the node's shape. -/
def classifyConst (dt : DType) (v : Model.ScalarValue) (inputs : List Usize) (out : Shape) :
    Option Kind :=
  match inputs with
  | [] => (fun q => .const out (List.replicate out.prod q)) <$> decodeScalar dt v
  | _ => none

/-- `ConstTensor`: a flat buffer of the graph dtype. -/
def classifyConstTensor (dt : DType) (buf : Model.Buf) (inputs : List Usize) (out : Shape) :
    Option Kind :=
  match inputs with
  | [] => (fun qs => .const out qs) <$> decodeBuf dt buf
  | _ => none

/-- `Copy`: one operand of the output shape. -/
def classifyCopy (st : State) (inputs : List Usize) (out : Shape) : Option Kind :=
  match inputs with
  | [x] => if st.shapeOf x.val = some out then some (.copy out) else none
  | _ => none

/-- `Add`/`Mul`: two operands, both of exactly the output shape (no broadcasting). -/
def classifyBinary (st : State) (inputs : List Usize) (out : Shape) (mk : Shape → Kind) :
    Option Kind :=
  match inputs with
  | [x, y] =>
      if st.shapeOf x.val = some out ∧ st.shapeOf y.val = some out then some (mk out) else none
  | _ => none

/-- `Sum axis acc`: accumulator of the graph dtype; the axis is removed. -/
def classifySum (dt : DType) (st : State) (axis : Usize) (acc : Model.Prim)
    (inputs : List Usize) (out : Shape) : Option Kind :=
  match inputs with
  | [x] =>
      if DType.ofPrim acc = some dt then
        (st.shapeOf x.val).bind fun s => (s[axis.val]?).map fun n => .sum s n out axis.val
      else none
  | _ => none

/-- `Expand axis size`: `insert` when rank-increasing, `expand` when same-rank. -/
def classifyExpand (st : State) (axis : Usize) (size : Model.RtDim) (inputs : List Usize)
    (out : Shape) : Option Kind :=
  match inputs with
  | x :: witness => do
      let s ← st.shapeOf x.val
      let n ← out[axis.val]?
      if sizeAdmissible st size witness n = true then
        if out.length = s.length + 1 then
          pure (.insert out n s axis.val)
        else if s[axis.val]? = some 1 ∧ out = s.take axis.val ++ n :: s.drop (axis.val + 1) then
          pure (.expand (s.take axis.val) (s.drop (axis.val + 1)) n)
        else none
      else none
  | [] => none

/-- `Reshape`: only an insertion of one singleton axis into the operand shape (decoded as
`insert` of extent one) or a removal of one singleton axis from it (decoded as `sum` over
an extent-one axis, the form the Reshape adjoint emits). The first valid position is
chosen; all valid positions denote the same row-major reinterpretation. -/
def classifyReshape (st : State) (newShape : alloc.vec.Vec Model.RtDim) (inputs : List Usize)
    (out : Shape) : Option Kind :=
  match inputs with
  | [x] => do
      let s ← st.shapeOf x.val
      if newShape.val.mapM litDim = some out then
        match (List.range out.length).find? (fun p => (Axis.decode out 1 s p).isSome) with
        | some p => pure (.insert out 1 s p)
        | none =>
            let p ← (List.range s.length).find? (fun p => (Axis.decode s 1 out p).isSome)
            pure (.sum s 1 out p)
      else none
  | _ => none

/-- Admission of one non-`Load` node of the fragment; everything else is rejected. -/
def classify (dt : DType) (st : State) (op : Model.RiscOp) (inputs : List Usize)
    (out : Shape) : Option Kind :=
  match op with
  | .Const v => classifyConst dt v inputs out
  | .ConstTensor data => classifyConstTensor dt data.buf inputs out
  | .Copy => classifyCopy st inputs out
  | .Add => classifyBinary st inputs out .add
  | .Mul => classifyBinary st inputs out .mul
  | .Sum axis acc => classifySum dt st axis acc inputs out
  | .Expand axis size => classifyExpand st axis size inputs out
  | .Reshape newShape => classifyReshape st newShape inputs out
  | _ => none

/-- Decode one non-`Load` node: admission, then ports as offsets of the value operands. -/
def decodeOperation (dt : DType) (st : State) (op : Model.RiscOp) (inputs : List Usize)
    (out : Shape) : Option (GraphSyntax.Operation ℚ) := do
  let k ← classify dt st op inputs out
  let offs ← (valueInputs op inputs).mapM (fun x => st.offset x.val)
  k.build offs

/-- One node, at append position `idx`. `entries[j] = some id` names the `Load` node that
is context entry `j` of this graph (`none`: the entry has no `Load` in this graph). -/
def step (dt : DType) (entries : List (Option Nat)) (Γ : Context) (st : State) (idx : Nat)
    (node : Model.DagNode) : Option State := do
  guard (node.id.val = idx)
  guard (DType.ofPrim node.output_type.precision = some dt)
  let out ← resolveShape node.output_type.dims
  match node.op with
  | .Load _ => do
      guard (node.inputs.val = [])
      let j ← entries.findIdx? (· == some idx)
      guard (Γ[j]? = some out)
      pure { st with env := (idx, ⟨Γ.length - 1 - j, out⟩) :: st.env }
  | op => do
      let o ← decodeOperation dt st op node.inputs.val out
      pure { len := st.len + 1, env := (idx, ⟨st.len, out⟩) :: st.env, ops := st.ops ++ [o] }

def run (dt : DType) (entries : List (Option Nat)) (Γ : Context) :
    Nat → State → List Model.DagNode → Option State
  | _, st, [] => some st
  | i, st, node :: rest => do
      let st' ← step dt entries Γ st i node
      run dt entries Γ (i + 1) st' rest

/-- The ordered entry context: the shapes of the requested `Load` nodes. -/
def entryContext (dt : DType) (dag : Model.Dag) (entries : List Nat) : Option Context :=
  entries.mapM fun e => do
    let node ← dag.nodes.val[e]?
    guard (node.id.val = e)
    match node.op with
    | .Load _ => do
        guard (DType.ofPrim node.output_type.precision = some dt)
        resolveShape node.output_type.dims
    | _ => none

/-- Decode every node of a graph over the ordered entries. -/
def decodeGraph (dt : DType) (dag : Model.Dag) (entries : List Nat) :
    Option (Context × State) := do
  guard entries.Nodup
  let Γ ← entryContext dt dag entries
  let st ← run dt (entries.map some) Γ 0 (State.init Γ) dag.nodes.val
  pure (Γ, st)

/-- Decode every node of a graph over a given entry context `Γ`, where `ids[j]` is the
`Load` node of entry `j` in this graph, if it still has one (a pruned entry has none). -/
def decodeGraphOver (dt : DType) (dag : Model.Dag) (ids : List (Option Nat)) (Γ : Context) :
    Option State := do
  guard (ids.reduceOption.Nodup ∧ ids.length = Γ.length)
  run dt ids Γ 0 (State.init Γ) dag.nodes.val

/-- Ordered roots as offsets in the final context; duplicates are retained. -/
def decodeRootOffsets (st : State) (roots : List Usize) : Option (List Nat) :=
  roots.mapM (fun r => st.offset r.val)

/-- Source syntax of a scalar loss program: the decoded bindings, then `ret` of `loss`. -/
def sourceSyntax (dt : DType) (dag : Model.Dag) (entries : List Nat) (loss : Nat) :
    Option (Context × GraphSyntax.Source ℚ) := do
  let (Γ, st) ← decodeGraph dt dag entries
  let r ← st.offset loss
  pure (Γ, st.ops.foldr GraphSyntax.Source.bind (.ret [] r))

/-- The decoded source program `Program ℚ Γ []` with its ordered entry context. -/
def decodeSource (dt : DType) (dag : Model.Dag) (entries : List Nat) (loss : Nat) :
    Option (Sigma fun Γ : Context => Program ℚ Γ []) := do
  let (Γ, src) ← sourceSyntax dt dag entries loss
  let p ← src.decode Γ []
  pure ⟨Γ, p⟩

/-- A graph with its own ordered roots as a reference target circuit. -/
def graphTarget (dt : DType) (dag : Model.Dag) (entries : List Nat) :
    Option (Context × GraphSyntax.Target ℚ) := do
  let (Γ, st) ← decodeGraph dt dag entries
  let roots ← decodeRootOffsets st dag.roots.val
  pure (Γ, ⟨st.ops, roots⟩)

def decodeCircuit (dt : DType) (dag : Model.Dag) (entries : List Nat) (Ω : Context) :
    Option (Sigma fun Γ : Context => Circuit ℚ Γ Ω) := do
  let (Γ, tgt) ← graphTarget dt dag entries
  let c ← tgt.decode Γ Ω
  pure ⟨Γ, c⟩

/-! ## Decoder theorems -/

/-- **Ports keep order and multiplicity.** The ports of a decoded operation are, position
by position, the offsets of the node's value operands (e.g. `mul(x, x)` has two equal
ports). Shape witnesses of `Expand` are not ports. -/
theorem decodeOperation_ports {dt : DType} {st : State} {op : Model.RiscOp}
    {inputs : List Usize} {out : Shape} {o : GraphSyntax.Operation ℚ}
    (h : decodeOperation dt st op inputs out = some o) :
    List.Forall₂ (fun (x : Usize) i => st.offset x.val = some i) (valueInputs op inputs)
      (ports o) := by
  simp only [decodeOperation, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨k, _, offs, hoffs, hb⟩ := h
  rw [Kind.ports_build hb]
  exact (mapM_eq_some_iff _ _ _).mp hoffs

/-- **Typed ports.** If the decoded syntax typechecks in a context `Γ` of the current
length, its operand variables are exactly the node's value operands in order, each with
Lean owner equal to the owner assigned to that Rust node and
`owner + offset + 1 = Γ.length`. -/
theorem decodeOperation_typed_ports {dt : DType} {st : State} {op : Model.RiscOp}
    {inputs : List Usize} {out : Shape} {o : GraphSyntax.Operation ℚ} {Γ : Context}
    (hlen : Γ.length = st.len) (h : decodeOperation dt st op inputs out = some o)
    {p : Sigma (Op ℚ Γ)} (ht : o.decode Γ = some p) :
    List.Forall₂ (fun (x : Usize) (v : Sigma (Var Γ)) =>
        ∃ sl, st.slot x.val = some sl ∧ v.2.owner = sl.owner ∧
          sl.owner + v.2.offset + 1 = Γ.length)
      (valueInputs op inputs) (portVars p.2) := by
  have hp := decodeOperation_ports h
  rw [← Operation.encode_of_decode o p ht, ← portVars_offsets, List.forall₂_map_right_iff] at hp
  refine hp.imp ?_
  intro x v hx
  obtain ⟨sl, hsl, hrel⟩ := State.offset_spec hx
  exact ⟨sl, hsl, Var.owner_eq_of_offset v.2 rfl (by omega), by omega⟩

theorem classifyBinary_spec {st : State} {inputs : List Usize} {out : Shape}
    {mk : Shape → Kind} {k : Kind} (h : classifyBinary st inputs out mk = some k) :
    ∃ x y, inputs = [x, y] ∧ st.shapeOf x.val = some out ∧ st.shapeOf y.val = some out ∧
      k = mk out := by
  unfold classifyBinary at h
  split at h
  · rename_i x y
    split at h
    · rename_i hs; exact ⟨x, y, rfl, hs.1, hs.2, (Option.some.inj h).symm⟩
    · contradiction
  · contradiction

/-- Add is admitted only when both operands have exactly the output shape (Rust's
rank-zero broadcasting is rejected). -/
theorem classify_add {dt : DType} {st : State} {inputs : List Usize} {out : Shape}
    {k : Kind} (h : classify dt st .Add inputs out = some k) :
    ∃ x y, inputs = [x, y] ∧ st.shapeOf x.val = some out ∧ st.shapeOf y.val = some out ∧
      k = .add out :=
  classifyBinary_spec h

/-- Mul is admitted only when both operands have exactly the output shape. -/
theorem classify_mul {dt : DType} {st : State} {inputs : List Usize} {out : Shape}
    {k : Kind} (h : classify dt st .Mul inputs out = some k) :
    ∃ x y, inputs = [x, y] ∧ st.shapeOf x.val = some out ∧ st.shapeOf y.val = some out ∧
      k = .mul out :=
  classifyBinary_spec h

/-- The two `Expand` interpretations: rank-increasing is `insert`; same-rank requires
source extent one at `axis` and unchanged other extents and is `expand`. -/
theorem classify_expand {dt : DType} {st : State} {axis : Usize} {size : Model.RtDim}
    {inputs : List Usize} {out : Shape} {k : Kind}
    (h : classify dt st (.Expand axis size) inputs out = some k) :
    ∃ x witness s n, inputs = x :: witness ∧ st.shapeOf x.val = some s ∧
      out[axis.val]? = some n ∧ sizeAdmissible st size witness n = true ∧
      ((out.length = s.length + 1 ∧ k = .insert out n s axis.val) ∨
       (out.length ≠ s.length + 1 ∧ s[axis.val]? = some 1 ∧
         out = s.take axis.val ++ n :: s.drop (axis.val + 1) ∧
         k = .expand (s.take axis.val) (s.drop (axis.val + 1)) n)) := by
  change classifyExpand st axis size inputs out = some k at h
  unfold classifyExpand at h
  split at h
  · rename_i x witness
    simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
    obtain ⟨s, hs, n, hn, h⟩ := h
    refine ⟨x, witness, s, n, rfl, hs, hn, ?_⟩
    split at h
    · rename_i hsz
      refine ⟨hsz, ?_⟩
      split at h
      · exact Or.inl ⟨by assumption, (Option.some.inj h).symm⟩
      · split at h
        · rename_i hne hsame
          exact Or.inr ⟨hne, hsame.1, hsame.2, (Option.some.inj h).symm⟩
        · contradiction
    · contradiction
  · contradiction

/-- `Reshape` is admitted only as insertion or removal of one singleton axis: all target
dims are literals equal to the output shape, and either the output removing extent `1` at
`p` is the operand shape (`insert`), or the operand removing extent `1` at `p` is the
output shape (`sum` over the extent-one axis). -/
theorem classify_reshape {dt : DType} {st : State} {newShape : alloc.vec.Vec Model.RtDim}
    {inputs : List Usize} {out : Shape} {k : Kind}
    (h : classify dt st (.Reshape newShape) inputs out = some k) :
    ∃ x s p, inputs = [x] ∧ st.shapeOf x.val = some s ∧
      newShape.val.mapM litDim = some out ∧
      ((∃ a, Axis.decode out 1 s p = some a ∧ k = .insert out 1 s p) ∨
       (∃ a, Axis.decode s 1 out p = some a ∧ k = .sum s 1 out p)) := by
  change classifyReshape st newShape inputs out = some k at h
  unfold classifyReshape at h
  split at h
  · rename_i x
    simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
    obtain ⟨s, hs, h⟩ := h
    split at h
    · rename_i hlit
      split at h
      · rename_i p hp
        have hsome := List.find?_some hp
        obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp (by simpa using hsome)
        exact ⟨x, s, p, rfl, hs, hlit, Or.inl ⟨a, ha, (Option.some.inj h).symm⟩⟩
      · simp only [Option.bind_eq_some_iff] at h
        obtain ⟨p, hp, hk⟩ := h
        have hsome := List.find?_some hp
        obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp (by simpa using hsome)
        exact ⟨x, s, p, rfl, hs, hlit, Or.inr ⟨a, ha, (Option.some.inj hk).symm⟩⟩
    · contradiction
  · contradiction

/-- **Generated `Reshape` value preservation.** The Rust evaluator moves the flat buffer
unchanged under the new shape; for an admitted reshape that reinterpretation is exactly
the decoded `insert` of the singleton axis. -/
theorem reshape_value {s out : Shape} {p : Nat} {a : Axis out 1 s}
    (ha : Axis.decode out 1 s p = some a) {f : Aeneas.Std.BinaryFormat}
    {xs : List (Aeneas.Std.BinaryFloat f)} {x : Tensor ℚ s} (hx : decodeTensor s xs = some x) :
    decodeTensor out xs = some (LaCaDiLECore.insert a x) ∧ a.position = p :=
  ⟨decodeTensor_reshape_singleton a hx, Axis.position_of_decode ha⟩

/-- **Singleton-removing `Reshape` value preservation.** Reinterpreting the flat buffer
of the operand (shape `s`) with the extent-one axis at `p` deleted is exactly the decoded
`sum` over that extent-one axis. -/
theorem reshape_remove_singleton_value {s out : Shape} {p : Nat} {a : Axis s 1 out}
    (ha : Axis.decode s 1 out p = some a) {f : Aeneas.Std.BinaryFormat}
    {xs : List (Aeneas.Std.BinaryFloat f)} {x : Tensor ℚ s} (hx : decodeTensor s xs = some x) :
    decodeTensor out xs = some (LaCaDiLECore.sum a x) ∧ a.position = p :=
  ⟨decodeTensor_remove_singleton a hx, Axis.position_of_decode ha⟩

/-- A `ConstTensor` of the graph dtype decodes to the exact realized buffer, which the
typed decoder reads back row-major with `Tensor.ofList` (no printed-decimal rounding). -/
theorem classify_constTensor {dt : DType} {inputs : List Usize} {out : Shape} {k : Kind}
    {buf : Model.Buf}
    (hdata : classifyConstTensor dt buf inputs out = some k) :
    ∃ qs, inputs = [] ∧ decodeBuf dt buf = some qs ∧ k = .const out qs := by
  unfold classifyConstTensor at hdata
  split at hdata
  · obtain ⟨qs, hqs, rfl⟩ := Option.map_eq_some_iff.mp hdata
    exact ⟨qs, rfl, hqs, rfl⟩
  · contradiction

/-- A scalar `Const` of the graph dtype is splatted exactly to the node's shape. -/
theorem classify_const {dt : DType} {v : Model.ScalarValue} {inputs : List Usize}
    {out : Shape} {k : Kind} (h : classifyConst dt v inputs out = some k) :
    ∃ q, inputs = [] ∧ decodeScalar dt v = some q ∧
      k = .const out (List.replicate out.prod q) := by
  unfold classifyConst at h
  split at h
  · obtain ⟨q, hq, rfl⟩ := Option.map_eq_some_iff.mp h
    exact ⟨q, rfl, hq, rfl⟩
  · contradiction

/-- A decoded `ConstTensor` payload is exactly the row-major decoding of the Rust buffer. -/
theorem decodeBuf_tensor {dt : DType} {buf : Model.Buf} {qs : List ℚ} {out : Shape}
    {x : Tensor ℚ out} (hb : decodeBuf dt buf = some qs) (hx : Tensor.ofList out qs = some x) :
    (∃ v, buf = .F64 v ∧ decodeVec out v = some x) ∨
      (∃ v, buf = .F32 v ∧ decodeVec out v = some x) := by
  unfold decodeBuf at hb
  split at hb
  · rename_i v
    exact Or.inl ⟨v, rfl, by simp [decodeVec, decodeTensor, hb, hx]⟩
  · rename_i v
    exact Or.inr ⟨v, rfl, by simp [decodeVec, decodeTensor, hb, hx]⟩
  · contradiction

/-! ### Global invariants of the node pass -/

theorem step_len {dt : DType} {entries : List (Option Nat)} {Γ : Context} {st st' : State}
    {idx : Nat} {node : Model.DagNode} (h : step dt entries Γ st idx node = some st') :
    st'.len + st.ops.length = st.len + st'.ops.length ∧ st.ops <+: st'.ops := by
  unfold step at h
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨_, _, _, _, out, _, h⟩ := h
  split at h
  · simp only [Option.bind_eq_some_iff] at h
    obtain ⟨_, _, j, _, _, _, h⟩ := h
    cases Option.some.inj h
    exact ⟨rfl, List.prefix_refl _⟩
  · simp only [Option.bind_eq_some_iff] at h
    obtain ⟨o, _, h⟩ := h
    cases Option.some.inj h
    exact ⟨by simp; omega, List.prefix_append _ _⟩

theorem run_len {dt : DType} {entries : List (Option Nat)} {Γ : Context} :
    ∀ {i : Nat} {st st' : State} {nodes : List Model.DagNode},
      run dt entries Γ i st nodes = some st' →
        st'.len + st.ops.length = st.len + st'.ops.length ∧ st.ops <+: st'.ops
  | _, st, st', [], h => by
      simp only [run, Option.some.injEq] at h; subst h; exact ⟨rfl, List.prefix_refl _⟩
  | i, st, st', node :: rest, h => by
      simp only [run, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
      obtain ⟨st1, h1, h2⟩ := h
      obtain ⟨a1, p1⟩ := step_len h1
      obtain ⟨a2, p2⟩ := run_len h2
      exact ⟨by omega, p1.trans p2⟩

/-- After decoding, the context length is the entry count plus one binding per emitted
operation, so the final context of `decodePrefix Γ st.ops` has length `st.len`. -/
theorem decodeGraph_len {dt : DType} {dag : Model.Dag} {entries : List Nat} {Γ : Context}
    {st : State} (h : decodeGraph dt dag entries = some (Γ, st)) :
    st.len = Γ.length + st.ops.length ∧ entries.Nodup ∧ Γ.length = entries.length := by
  simp only [decodeGraph, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def,
    Option.some.injEq, Prod.mk.injEq] at h
  obtain ⟨_, hnd, Γ', hΓ, st', hr, rfl, rfl⟩ := h
  have hnodup : entries.Nodup := by
    by_contra hn
    simp [hn] at hnd
  have := (run_len hr).1
  simp only [State.init, List.length_nil, Nat.add_zero] at this
  exact ⟨by omega, hnodup, ((mapM_eq_some_iff _ _ _).mp hΓ).length_eq.symm⟩

/-- Roots decode positionally: order and duplicates are retained. -/
theorem decodeRootOffsets_spec {st : State} {roots : List Usize} {offs : List Nat}
    (h : decodeRootOffsets st roots = some offs) :
    List.Forall₂ (fun (r : Usize) i => ∃ sl, st.slot r.val = some sl ∧
      sl.owner + i + 1 = st.len) roots offs :=
  ((mapM_eq_some_iff _ _ _).mp h).imp fun _ _ hr => State.offset_spec hr

/-- Entry `j` of the ordered entry context is the variable at offset `j` of `Γ`, whose
owner is `|Γ| - 1 - j`, the owner assigned to that entry's `Load`. -/
theorem entry_owner {Γ : Context} {s : Shape} (v : Var Γ s) {j : Nat} (hj : v.offset = j) :
    v.owner = Γ.length - 1 - j := by
  have := v.owner_add_offset
  omega

end ChelisBridge
