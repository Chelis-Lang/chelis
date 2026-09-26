import OptionCollectors

/-! The generated model of `verification/option_collectors.rs` on the inputs of
    its native test module: every `#guard` states the native result asserted
    there. -/
namespace option_collectors.checks
open Aeneas Aeneas.Std Result

def u (x : Nat) : U32 := ⟨BitVec.ofNat _ x⟩
def sl (xs : List Nat) (h : xs.length ≤ Usize.max := by scalar_tac) : Slice U32 :=
  ⟨xs.map u, by simpa using h⟩
def nats (v : alloc.vec.Vec U32) : List Nat := v.val.map (·.val)
def hints (v : alloc.vec.Vec (Usize × Option Usize)) : List (Nat × Option Nat) :=
  v.val.map fun (l, h) => (l.val, h.map (·.val))
def vecs (r : Result (Option (alloc.vec.Vec U32))) : Option (Option (List Nat)) :=
  match r with | ok v => some (v.map nats) | _ => none

-- collectors (native `collectors`)
#guard vecs (doubled (sl [1, 2, 3])) == some (some [2, 4, 6])
#guard vecs (doubled (sl [1, 200, 3])) == some none
#guard vecs (doubled (sl [])) == some (some [])
#guard vecs (leaves (sl [2, 4])) == some (some [1, 2])
#guard vecs (leaves (sl [2, 3, 4])) == some none
#guard (match calls_until_none (sl [1, 0, 2, 0]) with
  | ok (v, n) => some (v.map nats, n.val) | _ => none) == some (none, 2)
#guard (match calls_until_none (sl [1, 2]) with
  | ok (v, n) => some (v.map nats, n.val) | _ => none) == some (some [1, 2], 2)
#guard (match resume_after_none (sl [1, 0, 2, 3]) with
  | ok (v, x) => some (v.map nats, x.map (·.val)) | _ => none) == some (none, some 2)
#guard (match resume_after_none (sl [1, 2]) with
  | ok (v, x) => some (v.map nats, x.map (·.val)) | _ => none) == some (some [1, 2], none)
#guard (match option_set (sl [3, 1, 3]) with
  | ok s => some (s.map fun s => s.map.entries.map (·.1.val)) | _ => none) == some (some [1, 3])
#guard (match option_set (sl [3, 0]) with
  | ok s => some (s.map fun s => s.map.entries.map (·.1.val)) | _ => none) == some none

-- size hints the inner collector sees (native `hints`)
#guard (match shunt_hints (sl [1, 2]) with | ok v => some (v.map hints) | _ => none)
  == some (some [(0, some 2), (0, some 1), (0, some 0)])
#guard (match shunt_hints (sl [1, 0, 2]) with | ok v => some (v.map hints) | _ => none)
  == some none
#guard (match after_none (sl [4, 5, 6]) with
  | ok (v, n, x) => some (v.map (fun (h, items) => (hints h, nats items)), n.val, x.map (·.val))
  | _ => none)
  == some (some ([(0, some 2), (0, some 1), (0, some 0)], [4, 5, 6]), 3, none)
-- The shunt is not fused: after the None it polls the source again.
#guard (match after_none (sl [4, 0, 6, 7, 8]) with
  | ok (v, n, x) => some (v.map (fun (h, items) => (hints h, nats items)), n.val, x.map (·.val))
  | _ => none)
  == some (none, 3, some 7)

end option_collectors.checks
