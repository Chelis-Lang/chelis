import FlattenIterators
import Aeneas.Std.FlatMap

/-! The generated model of `verification/flatten_iterators.rs` on the inputs of
    its native test module: every `#guard` states the native result asserted
    there. -/
namespace flatten_iterators.checks
open Aeneas Aeneas.Std Result

def u (x : Nat) : U32 := ⟨BitVec.ofNat _ x⟩
def vec {α} (xs : List α) (h : xs.length ≤ Usize.max := by scalar_tac) : alloc.vec.Vec α := ⟨xs, h⟩
def uv (xs : List Nat) (h : xs.length ≤ Usize.max := by scalar_tac) : alloc.vec.Vec U32 :=
  ⟨xs.map u, by simpa using h⟩
def sl {α} (xs : List α) (h : xs.length ≤ Usize.max := by scalar_tac) : Slice α := ⟨xs, h⟩
def arr3 (x y z : Nat) : Array U32 3#usize := ⟨[u x, u y, u z], by simp⟩
def nats (r : Result (alloc.vec.Vec U32)) : Option (List Nat) :=
  match r with | ok v => some (v.val.map (·.val)) | _ => none
/-- A set as the native BTreeSet holds it: sorted and without duplicates. -/
def set (xs : List Nat) : alloc.collections.btree.set.BTreeSet U32 :=
  ⟨⟨xs.map fun x => (u x, ())⟩⟩
def successors (entries : List (Nat × List Nat)) :
    alloc.collections.btree.map.BTreeMap U32 (alloc.collections.btree.set.BTreeSet U32) :=
  ⟨entries.map fun (k, vs) => (u k, set vs)⟩
def hint (h : Usize × Option Usize) : Nat × Option Nat := (h.1.val, h.2.map (·.val))
def hints (r : Result ((Usize × Option Usize) × (Usize × Option Usize) × (Usize × Option Usize))) :
    Option ((Nat × Option Nat) × (Nat × Option Nat) × (Nat × Option Nat)) :=
  match r with | ok (a, b, c) => some (hint a, hint b, hint c) | _ => none
def n (r : Result Usize) : Option Nat := match r with | ok x => some x.val | _ => none
def on (x : Option Usize) : Option Nat := x.map (·.val)

-- collectors (native `collectors`); the native map is built from [3, 2, 3]
def succ := successors [(1, [2, 3]), (2, [])]
#guard nats (extend_successors (uv [9]) succ (u 1)) == some [9, 2, 3]
#guard nats (extend_successors (uv [9, 2, 3]) succ (u 2)) == some [9, 2, 3]
#guard nats (extend_successors (uv [9, 2, 3]) succ (u 5)) == some [9, 2, 3]
#guard nats (collect_nested (sl [uv [1, 2], uv [], uv [3]])) == some [1, 2, 3]
#guard nats (collect_nested (sl [])) == some []
#guard nats (collect_options (vec [none, some (u 1), none, some (u 3)])) == some [1, 3]
#guard nats (collect_arrays (vec [arr3 1 2 3, arr3 4 5 6])) == some [1, 2, 3, 4, 5, 6]
#guard (match option_into_iter (some (u 7)) with
  | ok (a, b, l, h) => some (a.map (·.val), b.map (·.val), l.val, on h) | _ => none)
  == some (some 7, none, 1, some 1)
#guard (match option_into_iter none with
  | ok (a, b, l, h) => some (a.map (·.val), b.map (·.val), l.val, on h) | _ => none)
  == some (none, none, 0, some 0)

-- size hints (native `hints`)
#guard hints (nested_hints (sl [uv [1, 2, 3]])) == some ((0, none), (2, some 2), (1, some 1))
#guard hints (nested_hints (sl [uv [1, 2], uv [3]])) == some ((0, none), (1, none), (0, none))
#guard hints (nested_hints (sl [uv [], uv [1]])) == some ((0, none), (0, some 0), (0, some 0))
#guard hints (nested_hints (sl [])) == some ((0, some 0), (0, some 0), (0, some 0))
#guard hints (optional_hints (some (set [4, 5]))) == some ((0, none), (1, some 1), (0, some 0))
#guard hints (optional_hints none) == some ((0, some 0), (0, some 0), (0, some 0))
#guard hints (oneshot_hints (vec [some (u 1), none, some (u 3), none]))
  == some ((0, some 4), (0, some 3), (0, some 1))
#guard hints (array_hints (vec [arr3 1 2 3, arr3 4 5 6]))
  == some ((6, some 6), (5, some 5), (4, some 4))

-- dispatch (native `dispatch`)
#guard n (count_override false) == some 303
#guard n (count_override true) == some 302
#guard (match fold_dispatch with | ok (a, b) => some (a.val, b.val) | _ => none) == some (11011, 4)
#guard (match try_fold_dispatch with
  | ok (.Err e, next, folds) => some (false, e.val, on next, folds.val)
  | ok (.Ok v, next, folds) => some (true, v.val, on next, folds.val)
  | _ => none) == some (false, 1, some 11, 3)
#guard (match nth_dispatch with | ok (a, b) => some (on a, on b) | _ => none)
  == some (some 11, some 12)
#guard (match advance_remainder 2#usize with | ok (a, b) => some (a.val, on b) | _ => none)
  == some (0, some 11)
#guard (match advance_remainder 5#usize with | ok (a, b) => some (a.val, on b) | _ => none)
  == some (2, none)
#guard (match any_resumes with | ok (a, b, c) => some (a, on b, c.val) | _ => none)
  == some (true, some 2, 2)
#guard (match fused_outer with | ok (a, b, c) => some (on a, on b, on c) | _ => none)
  == some (some 1, none, none)

end flatten_iterators.checks
