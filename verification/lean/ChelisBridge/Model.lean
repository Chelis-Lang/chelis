import ChelisIr

/-!
# The only contact point with the generated model's module and root namespace

Aeneas prefixes every generated item with the **root crate** namespace of the extraction
and keeps the defining crate as the next segment for foreign items:

| extraction root          | `Dag`                                   | `Buf`                                                  |
| ------------------------ | --------------------------------------- | ------------------------------------------------------ |
| `chelis_ir` (AD slice)   | `chelis_ir.dag.Dag`                     | `chelis_ir.chelis_types.dtype_semantics.Buf`           |
| `chelis_pipeline_core`   | `chelis_pipeline_core.chelis_ir.dag.Dag` | `chelis_pipeline_core.chelis_types.dtype_semantics.Buf` |

With the root namespace opened, `<defining_crate>.<path>` (`chelis_ir.dag.Dag`,
`chelis_types.dtype_semantics.Buf`) resolves in every model. Bridge modules therefore
refer to generated items only in that form, after `open_chelis_model` (or through the
abbreviations below). Swapping the model changes exactly two lines here: the `import`
(the model's root module, matching the lakefile's `modelRoots`) and the namespace opened
by `open_chelis_model`.
-/

-- `open_chelis_model` opens the configured extraction root namespace (the swap point).
set_option hygiene false in
macro "open_chelis_model" : command => `(open chelis_ir)

open_chelis_model

namespace ChelisBridge.Model

abbrev Dag := chelis_ir.dag.Dag
abbrev DagNode := chelis_ir.dag.DagNode
abbrev RiscOp := chelis_ir.dag.RiscOp
abbrev TensorType := chelis_ir.dag.TensorType
abbrev DimInfo := chelis_ir.dag.DimInfo
abbrev RtDim := chelis_ir.dag.RtDim
abbrev RtAxis := chelis_ir.dag.RtAxis
abbrev NodeId := chelis_ir.dag.NodeId
abbrev Prim := chelis_types.types.Prim
abbrev Bits := chelis_types.dtype_semantics.Bits
abbrev ScalarValue := chelis_types.dtype_semantics.ScalarValue
abbrev Buf := chelis_types.dtype_semantics.Buf
abbrev TensorStorage := chelis_types.dtype_semantics.TensorStorage
abbrev NodeMap := chelis_unord.UnordMap NodeId NodeId
abbrev GradResult := chelis_ir.grad.GradResult
abbrev AdError := chelis_ir.grad.AdError
abbrev LoadStoreName := chelis_ir.load_store_name.LoadStoreName

instance : DecidableEq LoadStoreName := inferInstanceAs (DecidableEq String)

/-- The input name of a `Load` node (inputs are bound by name at evaluation). -/
def loadName? : RiscOp → Option LoadStoreName
  | .Load label => some label
  | _ => none

/-- The extracted production AD entry point `chelis_ir::grad::grad_dag_checked`. -/
abbrev gradDagChecked : Dag → NodeId → Aeneas.Std.Slice NodeId →
    Aeneas.Std.Result (Aeneas.Std.core.result.Result GradResult AdError) :=
  chelis_ir.grad.grad_dag_checked

/-- The extracted `UnordMap::<NodeId, NodeId>::get`, with the same trait instances the
generated production code passes (`NodeId: Ord`, `Borrow<NodeId> for NodeId`). -/
def NodeMap.get (m : NodeMap) (key : NodeId) : Aeneas.Std.Result (Option NodeId) :=
  chelis_unord.UnordMap.get chelis_ir.dag.NodeId.Insts.CoreCmpOrd
    (Aeneas.Std.core.borrow.BorrowSelf chelis_ir.dag.NodeId) chelis_ir.dag.NodeId.Insts.CoreCmpOrd
    m key

/-- The Aeneas oracle model of the process-global capacity-scope identity sequence. -/
abbrev CapacityScopeSupply := Aeneas.Std.chelis_ir.capacity_key.CapacityScopeSupply
abbrev OwnershipError := chelis_ir.ownership.error.OwnershipError
abbrev DagDirective := chelis_ir.ownership.DagDirective
abbrev DagOwnershipPlan := chelis_ir.ownership.DagOwnershipPlan
abbrev DagOwnershipProgram :=
  chelis_ir.ownership.OwnershipProgram chelis_ir.ownership.DagEmissionPayload

/-- The extracted production DAG ownership lowering `chelis_ir::ownership::lower_dag_ownership`
(the supply is threaded back as the `&mut CapacityScopeSupply` result). -/
abbrev lowerDagOwnership : Dag → CapacityScopeSupply →
    Aeneas.Std.Result
      (Aeneas.Std.core.result.Result DagOwnershipProgram OwnershipError × CapacityScopeSupply) :=
  chelis_ir.ownership.lower_dag_ownership

/-- The directive vector of a DAG ownership plan, in order. -/
def planDirectives (p : DagOwnershipPlan) : List DagDirective := p.directives.val

/-- The payload DAG and private DAG plan of a lowered program (`none` for a host proof). -/
def dagProgramParts (p : DagOwnershipProgram) : Option (Dag × DagOwnershipPlan) :=
  match p.proof with
  | .Dag plan => some (p.payload.dag, plan)
  | _ => none

open Aeneas.Std in
/-- A successful extracted lowering returns the consumed input DAG itself as the payload and
a DAG plan as the private proof (`OwnershipProof::Dag`), never a host proof. -/
theorem lowerDagOwnership_ok {dag : Dag} {supply supply' : CapacityScopeSupply}
    {prog : DagOwnershipProgram}
    (h : lowerDagOwnership dag supply = .ok (.Ok prog, supply')) :
    ∃ plan, dagProgramParts prog = some (dag, plan) := by
  unfold lowerDagOwnership chelis_ir.ownership.lower_dag_ownership at h
  cases hl : chelis_ir.ownership.DagOwnershipPlan.lower dag supply with
  | ok res =>
      obtain ⟨r, s1⟩ := res
      rw [hl] at h
      cases r with
      | Ok plan =>
          simp only [core.result.TryResult.branch, bind_tc_ok] at h
          obtain ⟨rfl, rfl⟩ := h
          exact ⟨plan, rfl⟩
      | Err e =>
          simp only [core.result.TryResult.branch, core.result.FromResidualResult.from_residual,
            bind_tc_ok] at h
          cases hf : (core.convert.FromSame OwnershipError).from_ e <;>
            simp_all
  | fail e => rw [hl] at h; simp at h
  | div => rw [hl] at h; simp at h

/-! ### Builders for concrete generated values (tests only)

Structure layouts of generated types are named only here, so executable examples do not
depend on how a particular extraction lays them out. -/

open Aeneas.Std in
def vec {α : Type} (l : List α) : alloc.vec.Vec α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

open Aeneas.Std in
def slice {α : Type} (l : List α) : Slice α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

open Aeneas.Std in
/-- A `usize` literal (`0` if the platform cannot represent it). -/
def usize (n : Nat) : Usize :=
  if h : n < 2 ^ UScalarTy.Usize.numBits then Usize.ofNatCore n h
  else Usize.ofNatCore 0 (Nat.two_pow_pos _)

/-- A `Load` of the named input. -/
def load (label : String) : RiscOp := .Load label

/-- A literal-shaped tensor type. -/
def tensorType (prim : Prim) (dims : List Nat) : TensorType :=
  { dims := vec (dims.map fun n => .Lit (usize n)), precision := prim }

/-- A node with no spans, reuse hints, shape dependencies or result claims. -/
def mkNode (id : Nat) (op : RiscOp) (inputs : List Nat) (ty : TensorType) : DagNode :=
  { id := usize id, op, inputs := vec (inputs.map usize), output_type := ty,
    reusable_input := none, span_id := none, merged_spans := vec [], shape_deps := vec [],
    result_claim_deps := vec [] }

def mkDag (nodes : List DagNode) (roots : List Nat) : Dag :=
  { nodes := vec nodes, roots := vec (roots.map usize) }

def constTensorF64 (xs : List Aeneas.Std.F64) : RiscOp := .ConstTensor ⟨.F64 (vec xs)⟩

open Aeneas.Std in
/-- A `u64` literal (`0` if out of range). -/
def u64 (n : Nat) : U64 :=
  if h : n < 2 ^ UScalarTy.U64.numBits then UScalar.ofNatCore n h
  else UScalar.ofNatCore 0 (Nat.two_pow_pos _)

/-- A capacity-scope supply handing out `draw 0, draw 1, …` (`none`: native exhaustion). -/
def supplyOf (draw : Nat → Option Aeneas.Std.U64) : CapacityScopeSupply := { draw, taken := 0 }

end ChelisBridge.Model
