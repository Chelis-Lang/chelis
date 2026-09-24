import ChelisBridge.Examples
import ChelisBridge.Numeric.Primal
import ChelisBridge.Numeric.Decode
import ChelisBridge.Numeric.Loop
import ChelisBridge.Numeric.Gradient

/-!
# Executable checks for the numeric bridge

Interpreter-evaluated regression checks (not accepted theorems):

* the primal root of the extracted `grad_dag_checked` result is accepted by
  `acceptPrimal` (untrusted witness search, re-checked);
* the float decoder keeps constant bits (a negative zero stays negative) and the IEEE
  reference evaluation of a decoded graph follows the Rust pairwise sum order.
-/

namespace ChelisBridge.Numeric.Examples

open Aeneas.Std LaCaDiLECore ChelisBridge ChelisBridge.Examples ChelisBridge.Numeric
open ChelisBridge.Model (usize slice vec load mkNode mkDag tensorType constTensorF64)

/-- Untrusted witness for the primal check of the extracted AD result. -/
def primalWitness (forward : Model.Dag) (output : Nat) (wrt : List Nat) :
    Option (Model.GradResult × List (Option Nat)) := do
  let r ← extractedResult (Model.gradDagChecked forward (usize output) (slice (wrt.map usize)))
  let ⟨_, p⟩ ← decodeSource .f64 forward wrt output
  let ⟨Ω, rc⟩ ← decodeArtifactRoots .f64 forward wrt p r
  let k ← (artifactRootIds r).idxOf? r.output_node.val
  let w ← Var.decode Ω [] k
  pure (r, witnessBetween (progCircuit p) (selectRoot rc w))

def acceptsPrimal (forward : Model.Dag) (output : Nat) (wrt : List Nat) : Bool :=
  match primalWitness forward output wrt with
  | some (r, w) => acceptPrimal .f64 forward wrt output r w
  | none => false

#guard acceptsPrimal square 2 [0]
#guard acceptsPrimal sameRank 4 [0]
#guard acceptsPrimal withConst 4 [0]
#guard acceptsPrimal movement 8 [0]
#guard acceptsPrimal disconnected 3 [0, 1]
-- The primal root is the first Rust root.
#guard (extractedResult (Model.gradDagChecked square (usize 2) (slice [usize 0]))).map
  (fun r => (artifactRootIds r).head? == some r.output_node.val) == some true

/-- `x * c` with a constant holding a negative zero. -/
def negZero : Model.Dag :=
  mkDag [node 0 (load "x") [] [2], node 1 (constTensorF64 [f64 3, BinaryFloat.zero .binary64 true]) [] [2],
    node 2 .Mul [0, 1] [2]] [2]

-- The float decoder keeps the Rust bits: the second constant element stays `-0`, which the
-- rational decoder maps to `0`.
#guard (decodeGraphF .f64 negZero [0]).map (fun r => r.2.2) ==
  some [.const [2] [f64 3, BinaryFloat.zero .binary64 true], .mul [2] 1 0]

/-- `sum [1e16, -1e16, 1]` decoded from a Rust graph: the IEEE reference evaluates the
Rust pairwise order `(1e16 + -1e16) + 1 = 1`. -/
def sumDag : Model.Dag :=
  mkDag [node 0 (constTensorF64 [c16, cneg16, cone]) [] [3], node 1 (.Sum (usize 0) .F64) [0] []] [1]

def sumDagValue : Option (Option ℚ) :=
  match decodeCircuitF .f64 sumDag [] [[]] with
  | some ⟨[], c⟩ => some (realize (ieeeEval c (fun x => nomatch x) (Sum.inl ())))
  | _ => none

#guard sumDagValue == some (some 1)

-- The flat node loop on `sum₀ (x * x)` with `x = [1, 2, 3]` (binary64): the root is `14`.
def squareRoot : Option (Option ℚ) :=
  (specRun .f64 (fun l => if l = "x" then some [f64 1, f64 2, f64 3] else none) (fun _ => true)
      square.nodes.val (fun _ => none)).map fun vals =>
    (vals 2).bind fun v => v.data.head?.bind realize

#guard squareRoot == some (some 14)

-- The loop and the IEEE reference agree on the pairwise order of `sum [1e16, -1e16, 1]`.
#guard ((specRun .f64 (fun _ => none) (fun _ => true) sumDag.nodes.val (fun _ => none)).map
  fun vals => (vals 1).bind fun v => v.data.head?.bind realize) == some (some 1)

-- The flat loop on the post-AD graph actually returned by the extracted `grad_dag_checked`
-- for `sum₀ (x * x)`, `x = [1, 2, 3]`: its Rust roots are the primal `14` and the gradient
-- `[2, 4, 6]`.
def gradRun : Option (List (Option (List ℚ))) := do
  let r ← extractedResult (Model.gradDagChecked square (usize 2) (slice [usize 0]))
  let vals ← specRun .f64 (fun l => if l = "x" then some [f64 1, f64 2, f64 3] else none)
    (fun _ => true) r.dag.nodes.val (fun _ => none)
  pure ((artifactRootIds r).map fun g => (vals g).bind fun v => realizeList v.data)

#guard gradRun == some [some [14], some [2, 4, 6]]

end ChelisBridge.Numeric.Examples
