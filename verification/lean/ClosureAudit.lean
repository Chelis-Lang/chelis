import Lean
import ChelisBridge.Audit
import Lean.Compiler.ExternAttr
import Lean.Compiler.ImplementedByAttr

/-!
# Transitive assumption inventory of the accepted bridge theorems

`lake env lean --run ClosureAudit.lean <scope> <chain> <contract.json> <output.json>` selects the built scope and feature chain.
It follows every statement and proof dependency from `ChelisBridge.Audit.acceptedFor`.
The report contains these fields:

* `axioms`: axiom declarations in the closure. Nonstandard axioms need a trust entry.
* `opaque`: constants whose definitions are not visible to proofs. Each constant needs a trust entry.
* `extern`: constants with `extern` or `implemented_by` code. Kernel proofs use their definitions, not the compiled code.
* `modules`: per-module-root counts of reachable declarations.
* `accepted_names`: exact accepted theorem names. `accepted` gives their count.
* `scope` and `chain`: the compiled layout scope and feature chain.
* `entry_statements`: exact caller propositions checked against theorem types by the kernel.
* `discharged`: the map from proposition constants to accepted theorems with exactly those statements.
  Aeneas collection obligations use proposition constants. Each obligation needs an entry in this map.

The acceptance script compares the result with `verification/trusted_base.json`.
-/

open Lean

namespace ChelisBridge.ClosureAudit

/-- Constants mentioned by a declaration, including the parts the kernel checks. -/
def dependencies (info : ConstantInfo) : Array Name :=
  match info with
  | .axiomInfo v => v.type.getUsedConstants
  | .defnInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .thmInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .opaqueInfo v => v.type.getUsedConstants ++ v.value.getUsedConstants
  | .quotInfo v => v.type.getUsedConstants
  | .inductInfo v => v.type.getUsedConstants ++ v.ctors.toArray ++ v.all.toArray
  | .ctorInfo v => v.type.getUsedConstants.push v.induct
  | .recInfo v =>
      v.rules.foldl (init := v.type.getUsedConstants ++ v.all.toArray)
        fun acc rule => acc ++ rule.rhs.getUsedConstants

/-- Every constant reachable from `roots`, with the declaration through which each was
first reached; fails on a name missing from the environment. -/
partial def closure (env : Environment) (roots : List Name) : IO (NameSet × NameMap Name) := do
  let mut seen : NameSet := {}
  let mut parent : NameMap Name := {}
  let mut stack : Array (Name × Option Name) := roots.toArray.map (·, none)
  while !stack.isEmpty do
    let (decl, via) := stack.back!
    stack := stack.pop
    if seen.contains decl then continue
    seen := seen.insert decl
    if let some p := via then parent := parent.insert decl p
    match env.find? decl with
    | none => throw <| IO.userError s!"missing declaration in closure: {decl}"
    | some info =>
      for dep in dependencies info do
        unless seen.contains dep do stack := stack.push (dep, some decl)
  return (seen, parent)

/-- The discovery path from an accepted theorem to `decl`. -/
partial def witness (parent : NameMap Name) (decl : Name) (acc : List String := []) : List String :=
  let acc := decl.toString :: acc
  match parent.find? decl with
  | some p => witness parent p acc
  | none => acc

def moduleRoot (env : Environment) (decl : Name) : String :=
  match env.getModuleIdxFor? decl with
  | none => "<current>"
  | some idx =>
    let module := env.header.moduleNames[idx.toNat]!
    match module.components with
    | root :: _ => root.toString
    | [] => "<anonymous>"

def sortedStrings (decls : Array Name) : Array String :=
  (decls.map (·.toString)).qsort (· < ·)

/-- The proposition constant an accepted theorem states, if its statement is exactly one
(`theorem t : P` with `def P : Prop`); `P ∧ Q`, implications or applied constants do not
count, so a discharge can never prove a weaker or different statement. -/
def dischargedProposition (env : Environment) (thm : Name) : Option Name := do
  let .thmInfo info ← env.find? thm | none
  let .const proposition _ := info.type.consumeMData | none
  let .defnInfo defn ← env.find? proposition | none
  guard (defn.type.consumeMData == .sort .zero)
  return proposition

def verifyEntryStatement (env : Environment) (theoremName : Name)
    (statement : String) (inner boundary : Name) : IO Unit := do

  let stx ← match Parser.runParserCategory env `term statement with
    | .ok stx => pure stx
    | .error message => throw <| IO.userError message

  let action : CoreM Unit := do
    let (expected, _) ← Meta.MetaM.run' <| Elab.Term.TermElabM.run do

      let expected ← Elab.Term.elabType stx
      Elab.Term.synthesizeSyntheticMVarsNoPostponing
      let expected ← instantiateMVars expected

      if expected.hasMVar || expected.hasLevelMVar then
        throwError "caller statement contains unresolved variables"
      return expected

    unless expected.getUsedConstants.contains inner do
      throwError "caller statement does not use extracted inner function {inner}"

    let some (.defnInfo _) := env.find? inner
      | throwError "extracted inner function is not a definition: {inner}"

    unless expected.getUsedConstants.contains boundary do
      throwError "caller statement does not use its boundary definition: {boundary}"

    match env.find? boundary with
    | some (.defnInfo _) | some (.inductInfo _) => pure ()
    | _ => throwError "caller boundary is not a definition or inductive relation: {boundary}"


    let some (.thmInfo info) := env.find? theoremName
      | throwError "caller evidence is not a theorem: {theoremName}"

    unless info.type.getUsedConstants.contains inner do
      throwError "caller theorem states a general law instead of the extracted caller: {theoremName}"

    unless info.type.getUsedConstants.contains boundary do
      throwError "caller theorem does not state its boundary relation: {theoremName}"


    match Kernel.isDefEq env {} expected (mkConst ``True) with
    | .ok true => throwError "caller statement is True: {theoremName}"

    | .error _ => throwError "invalid caller statement"
    | .ok false => pure ()

    match Kernel.isDefEq env {} expected info.type with
    | .ok true => pure ()

    | .ok false => throwError "caller theorem has a different statement: {theoremName}"
    | .error _ => throwError "invalid caller theorem statement"

  discard <| action.toIO
    { fileName := "public_entries.json", fileMap := default }
    { env := env }

def checkEntryStatements (env : Environment) (chain : String)
    (contractPath : System.FilePath) (roots : List Name) : IO Json := do

  let contract ← IO.ofExcept <| Json.parse (← IO.FS.readFile contractPath)
  let entries ← IO.ofExcept <| contract.getObjValAs? (Array Json) "entries"

  unless entries.size == 32 do
    throw <| IO.userError "public-entry contract does not contain all 32 entries"

  let mut checked : List (String × Json) := []
  for entry in entries do
    let chains ← IO.ofExcept <| entry.getObjValAs? (Array String) "chains"

    unless chains.contains chain do continue
    let theoremName ← IO.ofExcept <| entry.getObjValAs? String "theorem"
    let statement ← IO.ofExcept <| entry.getObjValAs? String "statement"

    let inner ← IO.ofExcept <| entry.getObjValAs? String "lean_inner"
    let kind ← IO.ofExcept <| entry.getObjValAs? String "boundary_kind"
    let boundary ← IO.ofExcept <| entry.getObjValAs? String "lean_boundary"

    let expectedBoundary ← match kind with
      | "resume" => pure "ChelisBridge.Runtime.Entries.BoundaryRelation"

      | "catch" => pure "ChelisBridge.Runtime.Entries.CatchBoundaryRelation"

      | "panic_on_rejection" => pure "ChelisBridge.Runtime.Entries.PanicBoundaryRelation"

      | _ => throw <| IO.userError s!"unknown caller boundary kind: {kind}"

    unless boundary == expectedBoundary do
      throw <| IO.userError s!"caller boundary differs from its kind: {theoremName}"


    unless roots.contains theoremName.toName do
      throw <| IO.userError s!"caller theorem is not an accepted root: {theoremName}"

    verifyEntryStatement env theoremName.toName statement inner.toName boundary.toName
    checked := (theoremName, Json.mkObj [
      ("statement", toJson statement),
      ("lean_inner", toJson inner),
      ("lean_boundary", toJson boundary),
      ("boundary_kind", toJson kind)]) :: checked

  let expectedCount := if chain == "default" then 27 else 32
  unless checked.length == expectedCount do
    throw <| IO.userError "public-entry contract uses the wrong chain inventory"

  return Json.mkObj checked


def main (args : List String) : IO UInt32 := do
  let [scope, chain, contract, output] := args
    | IO.eprintln "usage: ClosureAudit.lean <scope> <chain> <public_entries.json> <output.json>"

      return 2

  unless scope == chelisAcceptanceScope && chain == chelisAcceptanceChain do
    throw <| IO.userError "closure scope or chain differs from the compiled bridge"


  unless chain == "default" || chain == "lowering-trace" do
    throw <| IO.userError "closure inventory needs a production chain"

  initSearchPath (← findSysroot)
  let env ← importModules #[{ module := `ChelisBridge.Audit }] {} (trustLevel := 0)

  let some roots := ChelisBridge.Audit.acceptedFor scope chain
    | throw <| IO.userError "unknown acceptance scope or chain"


  let mut uniqueRoots : NameSet := {}
  for root in roots do
    if uniqueRoots.contains root then
      throw <| IO.userError s!"duplicate accepted theorem: {root}"
    uniqueRoots := uniqueRoots.insert root
    match env.find? root with
    | some (.thmInfo _) => pure ()
    | _ => throw <| IO.userError s!"accepted name is not a theorem: {root}"
  let entryStatements ← if scope == "whole-core" then
    checkEntryStatements env chain contract roots
    else pure <| Json.mkObj []

  let (reachable, parent) ← closure env roots
  let mut axioms : Array Name := #[]
  let mut opaques : Array Name := #[]
  let mut externs : Array Name := #[]
  let mut implemented : Array Name := #[]
  let mut modules : Std.HashMap String Nat := {}
  for decl in reachable.toList do
    let some info := env.find? decl | continue
    match info with
    | .axiomInfo _ => axioms := axioms.push decl
    | .opaqueInfo _ => opaques := opaques.push decl
    | _ => pure ()
    if isExtern env decl then externs := externs.push decl
    if (Compiler.implementedByAttr.getParam? env decl).isSome then
      implemented := implemented.push decl
    let root := moduleRoot env decl
    modules := modules.insert root (modules.getD root 0 + 1)
  let moduleCounts := modules.toList.toArray.qsort (fun a b => a.1 < b.1)
  let discharged := roots.filterMap fun root =>
    (dischargedProposition env root).map fun proposition => (proposition.toString, toJson root.toString)
  let report := Json.mkObj [
    ("scope", toJson scope),
    ("chain", toJson chain),
    ("entry_statements", entryStatements),
    ("accepted", toJson roots.length),
    ("accepted_names", toJson (sortedStrings roots.toArray)),
    ("reachable", toJson reachable.size),
    ("axioms", toJson (sortedStrings axioms)),
    ("opaque", toJson (sortedStrings opaques)),
    ("extern", toJson (sortedStrings (externs ++ implemented))),
    ("modules", Json.mkObj (moduleCounts.toList.map fun (root, count) => (root, toJson count))),
    ("discharged", Json.mkObj discharged),
    ("witnesses", Json.mkObj ((axioms ++ opaques ++ externs ++ implemented).toList.map fun decl =>
      (decl.toString, toJson (witness parent decl))))
  ]
  IO.FS.writeFile output report.pretty
  IO.println s!"{roots.length} accepted theorems, {reachable.size} reachable declarations, \
    {axioms.size} axioms, {opaques.size} opaque, {externs.size + implemented.size} extern/implemented_by"
  return 0

end ChelisBridge.ClosureAudit

def main (args : List String) : IO UInt32 := ChelisBridge.ClosureAudit.main args
