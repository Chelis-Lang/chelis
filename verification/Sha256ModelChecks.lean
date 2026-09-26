import Aeneas.Std.ChelisNative

open Aeneas Aeneas.Std

namespace Sha256ModelChecks

/-! Mirrors `crate_model_oracle/src/bin/sha256_model_oracle.rs`, evaluating the
    registered `chelis_types::native::sha256_concat` model. -/

def slice {α} (l : List α) : Slice α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def bytes (l : List Nat) : Slice U8 := slice (l.map fun b => (⟨BitVec.ofNat 8 b⟩ : U8))
def str (s : String) : List Nat := s.toUTF8.toList.map (·.toNat)

def digest (parts : List (List Nat)) : List Nat :=
  match chelis_types.native.sha256_concat (slice (parts.map bytes)) with
  | .ok d => d.val.map (·.val)
  | _ => []

def main : IO Unit := do
  let pre := str "chelis-library-proof-v1"
  let canonical := str ("(def {} answer (lit {type: (t-prim {} i64)} 42))\n\n" ++
    "(def {} twice (fn {} (params {} x) (app {} (var {} add) (var {} x) (var {} x))))\n")
  let length := (List.range 8).map fun i => canonical.length / 2 ^ (8 * i) % 256
  let base := digest [pre, [0], length, canonical]
  let chained := digest [pre, [1], base, length, canonical]
  let long := (List.range 1000).map fun i => (i * 7 + 3) % 256
  let cases : List (List (List Nat)) := [
    [], [[]], [str "abc"], [str "a", [], str "bc"],
    [str "abcdbcdecdefdefgefghfghighijhijk", str "ijkljklmklmnlmnomnopnopq"],
    [long.take 1, (long.drop 1).take 54, (long.drop 55).take 9, long.drop 64],
    [long.take 55], [long.take 56], [long.take 64]]
  for (parts, i) in cases.zipIdx do
    IO.println s!"sha256-{i}={digest parts}"
  IO.println s!"library-proof-base={base}"
  IO.println s!"library-proof-chained={chained}"

#eval main

end Sha256ModelChecks
