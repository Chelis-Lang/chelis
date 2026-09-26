import ChelisBridge.Audit
import Lean.Compiler.ExternAttr
import Lean.Compiler.ImplementedByAttr

/-!
# Transitive assumption inventory of the accepted bridge theorems

`lake env lean --run ClosureAudit.lean <output.json>` loads the built bridge, walks every
constant reachable from the statement or proof of each theorem in
`ChelisBridge.Audit.accepted` (types, values, constructors, recursor rules), and writes:

* `axioms`: axiom declarations in that closure (anything beyond the three standard
  axioms is a trusted assumption that must be justified);
* `opaque`: `opaque` constants, whose definitions proofs cannot see (a theorem about an
  opaque model holds for every implementation, so each one must be justified);
* `extern` / `implemented_by`: constants whose compiled code differs from their kernel
  definition. Kernel proofs do not use it; it matters only for evaluation-based evidence
  such as `#guard`, which is therefore never counted as proof;
* per-module-root counts of the reachable declarations;
* `discharged`: for every accepted theorem whose statement is exactly a proposition
  constant (`def P : Prop`), the map `P ↦ theorem`. Aeneas states collection
  obligations this way (e.g. `<Impl>.size_hint_total`), and the acceptance script
  requires each emitted obligation to appear here.

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
  | some p => if acc.length > 64 then acc else witness parent p acc
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

def main (args : List String) : IO UInt32 := do
  let [output] := args
    | IO.eprintln "usage: lake env lean --run ClosureAudit.lean <output.json>"; return 2
  initSearchPath (← findSysroot)
  let env ← importModules #[{ module := `ChelisBridge.Audit }] {} (trustLevel := 0)
  let roots := ChelisBridge.Audit.accepted
  for root in roots do
    match env.find? root with
    | some (.thmInfo _) => pure ()
    | _ => throw <| IO.userError s!"accepted name is not a theorem: {root}"
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
    ("accepted", toJson roots.length),
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
