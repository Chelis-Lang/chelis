import RangeInclusiveCollectors

/-! The generated model of `verification/range_inclusive_collectors.rs` on the
    inputs of its native test module: every `#guard` states the native result
    asserted there. -/
namespace range_inclusive_collectors.checks
open Aeneas Aeneas.Std Result

def n (x : Nat) : Usize := ⟨BitVec.ofNat _ x⟩
def i (x : Int) : I32 := ⟨BitVec.ofInt _ x⟩
def uv (xs : List Nat) (h : xs.length ≤ Usize.max := by scalar_tac) : alloc.vec.Vec Usize :=
  ⟨xs.map n, by simpa using h⟩
def nats (r : Result (alloc.vec.Vec Usize)) : Option (List Nat) :=
  match r with | ok v => some (v.val.map (·.val)) | _ => none
def ints (r : Result (alloc.vec.Vec I32)) : Option (List Int) :=
  match r with | ok v => some (v.val.map (·.val)) | _ => none
def max : Nat := core.num.Usize.MAX.val

#guard nats (row (n 3)) == some [0, 1, 2, 3]
#guard nats (row (n 0)) == some [0]
#guard ints (shifted (i (-1)) (i 2)) == some [-2, 0, 2, 4]
#guard ints (shifted (i 3) (i 2)) == some []
#guard nats (to_max (n (max - 1))) == some [max - 1, max]
#guard nats (extended (uv [9]) (n 2) (n 4)) == some [9, 2, 3, 4]
#guard nats (extended (uv [9]) (n 4) (n 2)) == some [9]
#guard (match rest_after_next (n 1) (n 3) with
  | ok (x, v) => some (x.map (·.val), v.val.map (·.val)) | _ => none) == some (some 1, [2, 3])
#guard (match rest_after_next (n 5) (n 5) with
  | ok (x, v) => some (x.map (·.val), v.val.map (·.val)) | _ => none) == some (some 5, [])
#guard (match rest_after_next (n 6) (n 5) with
  | ok (x, v) => some (x.map (·.val), v.val.map (·.val)) | _ => none) == some (none, [])

end range_inclusive_collectors.checks
