import ChelisBridge.ExtractedOwnership

/-!
# Executable smoke checks on concrete generated values

These `#guard`s run the decoders, the extracted production AD and the reference checkers
on concrete values of the generated Rust types. They are regression tests evaluated by the
interpreter, not accepted theorems (`Usize` depends on `System.Platform.numBits`). Every
generated structure is built through the `ChelisBridge.Model` builders, so a model swap only
touches `Model.lean`.
-/

namespace ChelisBridge.Examples

open Aeneas.Std LaCaDiLECore ChelisBridge
open ChelisBridge.Model (usize slice vec load mkNode mkDag tensorType constTensorF64)
open ChelisBridge.Ownership (Directive PlanMirror OwnershipPlan)

def node (id : Nat) (op : Model.RiscOp) (inputs : List Nat) (dims : List Nat) :
    Model.DagNode :=
  mkNode id op inputs (tensorType .F64 dims)

def f64 (n : Int) : F64 := BinaryFloat.ofInt .binary64 n

/-! ## Decoder -/

/-- `loss = sum₀ (x * x)` with `x : f64[3]`: the port list of `mul` is `[0, 0]`. -/
def square : Model.Dag :=
  mkDag [node 0 (load "x") [] [3], node 1 .Mul [0, 0] [3],
    node 2 (.Sum (usize 0) .F64) [1] []] [2]

-- `mul(x, x)` keeps both (equal) ports.
#guard (sourceSyntax .f64 square [0] 2).map (fun r => r.2) ==
  some (.bind (.mul [3] 0 0) (.bind (.sum [3] 3 [] 0 0) (.ret [] 0)))
#guard (decodeSource .f64 square [0] 2).isSome
-- A `Load` that is not a requested entry is rejected.
#guard (decodeSource .f64 square [] 2).isNone
-- Entry ordering is significant and duplicates are rejected.
#guard (decodeSource .f64 square [0, 0] 2).isNone

/-- Rust rank-zero broadcasting is outside the reference `add`. -/
def broadcast : Model.Dag :=
  mkDag [node 0 (load "x") [] [3], node 1 (load "c") [] [], node 2 .Add [0, 1] [3],
    node 3 (.Sum (usize 0) .F64) [2] []] [3]

#guard (decodeSource .f64 broadcast [0, 1] 3).isNone

/-- A constant with an infinite element is rejected; a finite one is exact. -/
def constant (x : F64) : Model.Dag :=
  mkDag [node 0 (load "x") [] [2], node 1 (constTensorF64 [f64 3, x]) [] [2],
    node 2 .Mul [0, 1] [2], node 3 (.Sum (usize 0) .F64) [2] []] [3]

#guard (sourceSyntax .f64 (constant (f64 (-5))) [0] 3).map (fun r => r.2) ==
  some (.bind (.const [2] [3, -5]) (.bind (.mul [2] 1 0)
    (.bind (.sum [2] 2 [] 0 0) (.ret [] 0))))
#guard (decodeSource .f64 (constant (BinaryFloat.infinity .binary64 false)) [0] 3).isNone
-- A mixed-dtype accumulator is rejected.
#guard (decodeSource .f64
  (mkDag [node 0 (load "x") [] [2], node 1 (.Sum (usize 0) .F32) [0] []] [1]) [0] 1).isNone

/-- Rank-increasing `Expand` is `insert`; same-rank `Expand` of a singleton axis is
`expand`; a singleton-inserting `Reshape` is `insert` and a singleton-removing one is
`sum` over an extent-one axis. -/
def movement : Model.Dag :=
  mkDag [node 0 (load "x") [] [2], node 1 (.Expand (usize 1) (.Lit (usize 3))) [0] [2, 3],
    node 2 (.Sum (usize 1) .F64) [1] [2],
    node 3 (.Reshape (vec [.Lit (usize 2), .Lit (usize 1)])) [2] [2, 1],
    node 4 (.Expand (usize 1) (.InputAxis (usize 1) (.Lit 1#i32))) [3, 1] [2, 3],
    node 5 (.Sum (usize 1) .F64) [4] [2], node 6 (.Reshape (vec [.Lit (usize 2)])) [3] [2],
    node 7 .Mul [5, 6] [2], node 8 (.Sum (usize 0) .F64) [7] []] [8]

#guard (sourceSyntax .f64 movement [0] 8).map (fun r => r.2) ==
  some (.bind (.insert [2, 3] 3 [2] 1 0) (.bind (.sum [2, 3] 3 [2] 1 0)
    (.bind (.insert [2, 1] 1 [2] 1 0) (.bind (.expand [2] [] 3 0)
      (.bind (.sum [2, 3] 3 [2] 1 0) (.bind (.sum [2, 1] 1 [2] 1 2)
        (.bind (.mul [2] 1 0) (.bind (.sum [2] 2 [] 0 0) (.ret [] 0)))))))))

-- A reshape that is neither a singleton insertion nor a singleton removal is rejected.
#guard (decodeSource .f64
  (mkDag [node 0 (load "x") [] [2, 3], node 1 (.Reshape (vec [.Lit (usize 6)])) [0] [6],
    node 2 (.Sum (usize 0) .F64) [1] []] [2]) [0] 2).isNone

-- Ordered roots keep duplicates.
#guard (graphTarget .f64 (mkDag [node 0 (load "x") [] [2], node 1 .Copy [0] [2]] [1, 0, 1])
  [0]).map (fun r => r.2.roots) == some [0, 1, 0]

/-! ## Untrusted witness search

A witness only proposes, for each reference binding, an actual binding; `check` re-verifies
every proposal, so the search is outside the trusted base. Reference bindings that no root
reaches through value ports are pruned (`none`); for the others the oldest not-yet-used
actual binding with an identical rebased encoding is proposed. -/

/-- Liveness of each reference binding (in binding order) with respect to its roots. -/
def liveFlags {C : Type} {Γ Δ : Context} (p : Prefix C Γ Δ) (roots : List Nat) : List Bool :=
  let bs := GraphCorrespondence.bindings p
  let live := bs.reverse.foldl (fun (L : List Nat) (b : GraphCorrespondence.Binding C Δ) =>
    if L.contains b.2.2.offset then L ++ ports b.2.1.encode else L) roots
  bs.map fun b => live.contains b.2.2.offset

def findWitness {C : Type} [DecidableEq C] {Ξ : Context}
    (bs : List (GraphCorrespondence.Binding C Ξ)) (n : Nat) :
    {Γ Δ : Context} → Prefix C Γ Δ → PartialRenaming Γ Ξ → List Nat → List Bool →
      List (Option Nat)
  | _, _, .nil, _, _, _ => []
  | _, _, .bind (s := s) op rest, σ, used, flags =>
      let live := flags.headD true
      let hits := if !live then [] else match op.rebase σ with
        | none => []
        | some expected => (List.range n).reverse.filterMap fun i =>
            (GraphCorrespondence.lookup s i bs).bind fun (actual, v) =>
              if expected.encode = actual.encode then some (i, v) else none
      match hits.find? (fun h => !used.contains h.1) <|> hits.head? with
      | some (i, v) =>
          some i :: findWitness bs n rest (PartialRenaming.cons (some v) σ) (i :: used) flags.tail
      | none => none :: findWitness bs n rest (PartialRenaming.cons none σ) used flags.tail

def witnessBetween {C : Type} [DecidableEq C] {Γ Ω : Context} (reference actual : Circuit C Γ Ω) :
    List (Option Nat) :=
  findWitness (GraphCorrespondence.bindings actual.code) actual.context.length reference.code
    (fun v => some (actual.code.embedding v)) []
    (liveFlags reference.code (Renaming.encode reference.result))

/-! ## End to end: the extracted production AD, decoded and checked -/

/-- Run the extracted `chelis_ir::grad::grad_dag_checked`, decode its actual return value
and search a witness; `acceptExtracted` then re-checks everything. -/
def extractedWitness (forward : Model.Dag) (output : Nat) (wrt : List Nat) :
    Option (List (Option Nat)) := do
  let r ← extractedResult (Model.gradDagChecked forward (usize output) (slice (wrt.map usize)))
  let ⟨_, p⟩ ← decodeSource .f64 forward wrt output
  let actual ← decodeArtifact .f64 forward wrt p r
  pure (witnessBetween p.compileGrad actual)

def acceptsExtracted (forward : Model.Dag) (output : Nat) (wrt : List Nat) : Bool :=
  match extractedWitness forward output wrt with
  | some w => acceptExtracted .f64 forward (usize output) (slice (wrt.map usize)) w
  | none => false

-- The production AD output for `sum (x * x)` (duplicate port) is accepted: by
-- `acceptExtracted_sound` this instance carries the real-derivative theorem for the actual
-- Rust circuit (interpreter evaluation, not a kernel proof).
#guard acceptsExtracted square 2 [0]

/-- Same-rank `Expand` of a singleton axis, squared and reduced. Its production adjoint
uses a rank-increasing `Expand` with an `InputAxis` shape witness (decoded as `insert`),
`Sum`, and the generated singleton-inserting `Reshape` (decoded as `insert`). -/
def sameRank : Model.Dag :=
  mkDag [node 0 (load "x") [] [2, 1], node 1 (.Expand (usize 1) (.Lit (usize 3))) [0] [2, 3],
    node 2 .Mul [1, 1] [2, 3], node 3 (.Sum (usize 1) .F64) [2] [2],
    node 4 (.Sum (usize 0) .F64) [3] []] [4]

#guard acceptsExtracted sameRank 4 [0]

/-- `sum (c * x + x)`: Rust omits the dead cotangent of the constant `c`, which the
reference computes and the witness prunes. -/
def withConst : Model.Dag :=
  mkDag [node 0 (load "x") [] [2], node 1 (constTensorF64 [f64 3, f64 (-5)]) [] [2],
    node 2 .Mul [0, 1] [2], node 3 .Add [2, 0] [2], node 4 (.Sum (usize 0) .F64) [3] []] [4]

#guard acceptsExtracted withConst 4 [0]

-- Forward singleton insertion and removal: their adjoints are the opposite `Reshape`s.
#guard acceptsExtracted movement 8 [0]

/-- `y` is requested but disconnected from the loss: Rust returns no gradient for it, the
decoder proves it unreachable through value ports and zero-fills it. -/
def disconnected : Model.Dag :=
  mkDag [node 0 (load "x") [] [2], node 1 (load "y") [] [3], node 2 .Mul [0, 0] [2],
    node 3 (.Sum (usize 0) .F64) [2] []] [3]

#guard (do
  let r ← extractedResult (Model.gradDagChecked disconnected (usize 3) (slice [usize 0, usize 1]))
  let ⟨_, p⟩ ← decodeSource .f64 disconnected [0, 1] 3
  pure ((GradArtifact.grad r 1).join.isNone && !needs p 1 && needs p 0)) == some true
#guard acceptsExtracted disconnected 3 [0, 1]

/-! ## Ownership plans returned by the extracted production lowering

`lowered` runs the extracted `chelis_ir::ownership::lower_dag_ownership` (with an oracle
supply of capacity identities) and reads its actual returned plan; `decodeOwned` and
`OwnedCode.check` are the checked path. -/

/-- Capacity identities `0, 1, 2, …` (the lowering draws one per plan). -/
def supply : Model.CapacityScopeSupply := Model.supplyOf fun n => some (Model.u64 n)

def lowered (dag : Model.Dag) : Option ExtractedOwnership.LoweredPlan :=
  ExtractedOwnership.loweredPlan dag supply

def loweredDirectives (dag : Model.Dag) : Option (List Directive) :=
  (lowered dag).map OwnershipPlan.directives

/-- Combined acceptance of the actual production pipeline (`acceptLowered_sound`): the
extracted AD, the extracted ownership lowering of its returned DAG, the extracted
`verify_ownership` sealing the returned program, the decoders and the reference checks; the
witnesses are searched outside the trusted base. -/
def acceptsLowered (forward : Model.Dag) (output : Nat) (wrt : List Nat) : Bool :=
  match extractedResult (Model.gradDagChecked forward (usize output) (slice (wrt.map usize))),
      decodeSource .f64 forward wrt output with
  | some r, some ⟨Γ, p⟩ =>
      match lowered r.dag, decodeArtifact .f64 forward wrt p r,
          decodeArtifactRoots .f64 forward wrt p r with
      | some plan, some actual, some ⟨Ω, rc⟩ =>
          match (Validation.planEntryIds forward wrt plan).bind
              (fun ids => Ownership.decodeOwned .f64 ids Γ Ω plan) with
          | some final =>
              ExtractedOwnership.acceptLowered .f64 forward (usize output)
                (slice (wrt.map usize)) supply
                (witnessBetween p.compileGrad actual)
                (witnessBetween rc.inlineCopies final.circuit.inlineCopies)
          | none => false
      | _, _, _ => false
  | _, _ => false

#guard acceptsLowered square 2 [0]
#guard acceptsLowered withConst 4 [0]
#guard acceptsLowered disconnected 3 [0, 1]

-- `mul(x, x)`: both (equal) ports are borrowed by the product, which the root moves.
#guard loweredDirectives (mkDag [node 0 (load "x") [] [2], node 1 .Mul [0, 0] [2]] [1]) ==
  some [.borrowLoad 0, .produce 1 [0, 0], .root 1]

-- A repeated owned root is rejected by the production lowering (`DagDuplicateTerminal`).
#guard match Model.lowerDagOwnership
    (mkDag [node 0 (load "x") [] [2], node 1 .Mul [0, 0] [2]] [1, 1]) supply with
  | .ok (.Err (.DagDuplicateTerminal owner), _) => owner.val == 1
  | _ => false

/-- A graph whose roots are the borrowed entry itself (twice) and an owned value. -/
def rootsDag : Model.Dag :=
  mkDag [node 0 (load "x") [] [2], node 1 .Mul [0, 0] [2]] [0, 1, 0]

-- `RootClone` of a borrowed entry is expanded into an explicit copy and its transfer, and
-- the independent reference ownership check accepts the original plan.
#guard loweredDirectives rootsDag ==
  some [.borrowLoad 0, .produce 1 [0, 0], .rootClone 0, .root 1, .rootClone 0]
#guard ((lowered rootsDag).bind (Ownership.ownedSyntax .f64 [some 0] [[2]])).map (·.1) ==
  some (.bind (.mul [2] 0 0) (.bind (.copy [2] 1) (.yield [2] 0 (.yield [2] 1
    (.bind (.copy [2] 2) (.yield [2] 0 .done))))))
#guard ((lowered rootsDag).bind (Ownership.decodeOwned .f64 [some 0] [[2]] [[2], [2], [2]])).map
  (fun c => c.check (LaCaDiLECore.Ownership.initial 1)) == some true

/-- A borrowed `Drop` is a non-consuming logical discard (Rust `BorrowedDrop`); the entry is
still readable afterwards. -/
def discardDag : Model.Dag :=
  mkDag [node 0 (load "x") [] [2], node 1 .Drop [0] [2], node 2 .Copy [0] [2]] [2]

#guard loweredDirectives discardDag ==
  some [.borrowLoad 0, .borrowedDrop 1 0, .clone 2 0, .root 2]
#guard ((lowered discardDag).bind (Ownership.decodeOwned .f64 [some 0] [[2]] [[2]])).map
  (fun c => c.check (LaCaDiLECore.Ownership.initial 1)) == some true

-- Hand-written plans exercise the decoder's own rejections. A repeated owned-root transfer
-- is rejected by the decoder.
#guard (Ownership.decodeOwned .f64 [some 0] [[2]] [[2], [2]]
  (⟨mkDag [node 0 (load "x") [] [2], node 1 .Mul [0, 0] [2]] [1, 1],
    [.borrowLoad 0, .produce 1 [0, 0], .root 1, .root 1]⟩ : PlanMirror)).isNone
-- A plan that drops an owned value before reading it again decodes, but the independent
-- ownership check rejects it.
#guard (Ownership.decodeOwned .f64 [some 0] [[2]] [[2]]
  (⟨mkDag [node 0 (load "x") [] [2], node 1 .Mul [0, 0] [2], node 2 .Drop [1] [2],
      node 3 .Copy [1] [2]] [3],
    [.borrowLoad 0, .produce 1 [0, 0], .ownedDrop 2 1, .clone 3 1, .root 3]⟩ :
      PlanMirror)).map (fun c => c.check (LaCaDiLECore.Ownership.initial 1)) == some false
-- A `Drop` whose Rust mode disagrees with the source's kind is rejected.
#guard (Ownership.decodeOwned .f64 [some 0] [[2]] []
  (⟨mkDag [node 0 (load "x") [] [2], node 1 .Drop [0] [2]] [],
    [.borrowLoad 0, .ownedDrop 1 0]⟩ : PlanMirror)).isNone

/-- Whether the extracted production verifier `DagOwnershipPlan::verify` accepts a plan
against its DAG. -/
def verifies (pl : ExtractedOwnership.LoweredPlan) : Bool :=
  match Model.dagPlanVerify pl.plan pl.dag with
  | .ok (.Ok ()) => true
  | _ => false

-- The extracted verifier accepts the plans the extracted lowering returns.
#guard (lowered rootsDag).map verifies == some true
#guard (lowered discardDag).map verifies == some true
#guard (lowered (mkDag [node 0 (load "x") [] [2], node 1 .Mul [0, 0] [2]] [1])).map verifies ==
  some true

/-- Two owned producers no root consumes: the lowered plan ends with their scope drops, in
ascending node order. -/
def twoDrops : Model.Dag :=
  mkDag [node 0 (load "x") [] [2], node 1 .Mul [0, 0] [2], node 2 .Mul [0, 0] [2]] [0]

#guard loweredDirectives twoDrops ==
  some [.borrowLoad 0, .produce 1 [0, 0], .produce 2 [0, 0], .rootClone 0, .scopeDrop 1,
    .scopeDrop 2]
#guard (lowered twoDrops).map verifies == some true
#guard ((lowered twoDrops).bind (Ownership.decodeOwned .f64 [some 0] [[2]] [[2]])).map
  (fun c => c.check (LaCaDiLECore.Ownership.initial 1)) == some true

/-- The lowered plan of `twoDrops` with its two scope drops swapped (owner map unchanged). -/
def scopeDropSwapped : Option ExtractedOwnership.LoweredPlan :=
  (lowered twoDrops).map fun pl =>
    let ds := pl.plan.directives.val
    { pl with plan := { pl.plan with directives := ⟨ds.take 4 ++ (ds.drop 4).reverse, by
        have h : (ds.take 4 ++ (ds.drop 4).reverse).length = ds.length := by simp; omega
        rw [h]
        exact pl.plan.directives.property⟩ } }

-- Counterexample (reference acceptance does not imply Rust acceptance): the swapped plan
-- decodes and the independent reference check accepts it, but the extracted production
-- verifier rejects it (it fixes the scope-drop order).
#guard scopeDropSwapped.map OwnershipPlan.directives ==
  some [.borrowLoad 0, .produce 1 [0, 0], .produce 2 [0, 0], .rootClone 0, .scopeDrop 2,
    .scopeDrop 1]
#guard (scopeDropSwapped.bind (Ownership.decodeOwned .f64 [some 0] [[2]] [[2]])).map
  (fun c => c.check (LaCaDiLECore.Ownership.initial 1)) == some true
#guard scopeDropSwapped.map verifies == some false

/-- Whether the extracted public `verify_ownership` seals the DAG program of a plan. -/
def seals (pl : ExtractedOwnership.LoweredPlan) : Bool :=
  match Model.verifyOwnership (Model.dagProgram pl.dag pl.plan) with
  | .ok (.Ok _) => true
  | _ => false

-- The extracted `verify_ownership` seals the lowered program and refuses the swapped one.
#guard (lowered twoDrops).map seals == some true
#guard (lowered rootsDag).map seals == some true
#guard scopeDropSwapped.map seals == some false

end ChelisBridge.Examples
