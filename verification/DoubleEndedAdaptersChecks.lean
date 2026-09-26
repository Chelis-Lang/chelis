import DoubleEndedAdapters

/-! The generated model of `verification/double_ended_adapters.rs` on the
    inputs of its native test module: every `#guard` states the native result
    asserted there. -/
namespace double_ended_adapters.checks
open Aeneas Aeneas.Std Result

def u (x : Nat) : U32 := ⟨BitVec.ofNat _ x⟩
def n (x : Nat) : Usize := ⟨BitVec.ofNat _ x⟩
def sl (xs : List Nat) (h : xs.length ≤ Usize.max := by scalar_tac) : Slice U32 :=
  ⟨xs.map u, by simpa using h⟩
def pair (p : Option (Usize × U32)) : Option (Nat × Nat) := p.map fun (i, x) => (i.val, x.val)
def upair (p : Option (Usize × Usize)) : Option (Nat × Nat) := p.map fun (i, x) => (i.val, x.val)
def opt (x : Option U32) : Option Nat := x.map (·.val)
/-- `Ok x` as `(true, x)`, `Err x` as `(false, x)`. -/
def res {α} (f : α → Nat) (r : core.result.Result α α) : Bool × Nat :=
  match r with | .Ok x => (true, f x) | .Err x => (false, f x)

-- native `enumerate_and_skip`
#guard (match last_matching (sl [7, 1, 7, 2, 7]) (n 0) (u 7) with
  | ok x => some (x.map (·.val)) | _ => none) == some (some 5)
#guard (match last_matching (sl [7, 1, 7, 2, 1]) (n 1) (u 7) with
  | ok x => some (x.map (·.val)) | _ => none) == some (some 3)
#guard (match last_matching (sl [7, 1, 7]) (n 3) (u 7) with
  | ok x => some (x.map (·.val)) | _ => none) == some none
#guard (match last_matching (sl [7, 1, 7]) (n 9) (u 7) with
  | ok x => some (x.map (·.val)) | _ => none) == some none
#guard (match enumerate_back (sl [5, 6, 7, 8]) with
  | ok (a, b, c) => some (pair a, pair b, pair c) | _ => none)
  == some (some (0, 5), some (3, 8), some (1, 6))
#guard (match enumerate_back (sl [5]) with
  | ok (a, b, c) => some (pair a, pair b, pair c) | _ => none) == some (some (0, 5), none, none)
#guard (match enumerate_rfold (sl [1, 2, 3]) with | ok x => some x.val | _ => none) == some 53
#guard (match enumerate_rfind (sl [4, 9, 4, 9]) (u 4) with
  | ok x => some (x.map (·.val)) | _ => none) == some (some 2)
#guard (match enumerate_rfind (sl [4, 9]) (u 3) with
  | ok x => some (x.map (·.val)) | _ => none) == some none
#guard (match enumerate_advance_back (sl [1, 2, 3]) (n 1) with
  | ok (r, p) => some (r.val, pair p) | _ => none) == some (0, some (1, 2))
#guard (match enumerate_advance_back (sl [1, 2, 3]) (n 5) with
  | ok (r, p) => some (r.val, pair p) | _ => none) == some (2, none)
#guard (match skip_back (sl [1, 2, 3, 4]) (n 2) with
  | ok (a, b, l) => some (opt a, opt b, l.val) | _ => none) == some (some 4, some 3, 0)
#guard (match skip_back (sl [1, 2, 3, 4]) (n 3) with
  | ok (a, b, l) => some (opt a, opt b, l.val) | _ => none) == some (some 4, none, 0)
#guard (match skip_back (sl [1, 2, 3]) (n 5) with
  | ok (a, b, l) => some (opt a, opt b, l.val) | _ => none) == some (none, none, 0)
#guard (match skip_nth_back (sl [1, 2, 3, 4, 5]) (n 1) (n 1) with
  | ok (a, b) => some (opt a, opt b) | _ => none) == some (some 4, some 2)
#guard (match skip_nth_back (sl [1, 2, 3, 4, 5]) (n 1) (n 4) with
  | ok (a, b) => some (opt a, opt b) | _ => none) == some (none, none)
#guard (match skip_nth_back (sl [1, 2, 3]) (n 3) (n 0) with
  | ok (a, b) => some (opt a, opt b) | _ => none) == some (none, none)
#guard (match skip_rfold (sl [1, 2, 3, 4]) (n 1) with | ok x => some x.val | _ => none) == some 432
#guard (match skip_rfold (sl [1, 2]) (n 2) with | ok x => some x.val | _ => none) == some 0
#guard (match skip_rfind (sl [1, 5, 2, 6]) (n 1) (u 3) with
  | ok (a, b) => some (opt a, opt b) | _ => none) == some (some 2, some 5)
#guard (match skip_rfind (sl [1, 5, 2, 6]) (n 1) (u 1) with
  | ok (a, b) => some (opt a, opt b) | _ => none) == some (none, none)
#guard (match skip_advance_back (sl [1, 2, 3, 4]) (n 1) (n 2) with
  | ok (r, a, b) => some (r.val, opt a, opt b) | _ => none) == some (0, some 2, none)
#guard (match skip_advance_back (sl [1, 2, 3, 4]) (n 1) (n 5) with
  | ok (r, a, b) => some (r.val, opt a, opt b) | _ => none) == some (2, none, none)
#guard (match skip_enumerate_back (sl [10, 11, 12]) (n 1) with
  | ok v => some (v.val.map fun (i, x) => (i.val, x.val)) | _ => none) == some [(2, 12), (1, 11)]
#guard (match filter_map_rfold (sl [1, 2, 3, 5]) with | ok x => some x.val | _ => none)
  == some 503010
#guard (match filter_map_rfold (sl [2, 4]) with | ok x => some x.val | _ => none) == some 0
#guard (match map_rfind (sl [1, 2, 5, 7]) with
  | ok (a, b) => some (opt a, opt b) | _ => none) == some (some 6, some 3)
#guard (match map_rfind (sl [1, 3]) with
  | ok (a, b) => some (opt a, opt b) | _ => none) == some (none, none)
#guard (match filter_nth_back (sl [1, 2, 3, 1, 4]) (n 1) with
  | ok (a, b) => some (opt a, opt b) | _ => none) == some (some 3, some 2)
#guard (match filter_nth_back (sl [1, 2, 3]) (n 2) with
  | ok (a, b) => some (opt a, opt b) | _ => none) == some (none, none)
#guard (match filter_try_rfold (sl [1, 0, 2, 3]) with
  | ok r => some (res (·.val) r) | _ => none) == some (true, 321)
#guard (match filter_try_rfold (sl [1, 9, 0, 2]) with
  | ok r => some (res (·.val) r) | _ => none) == some (false, 2)

-- native `dispatch`: the back folds of Skip, Enumerate, Filter and Map use the
-- overrides of the inner iterator (its next_back panics at `panic_at`)
#guard (match skip_try_rfold_dispatch (n 0) with
  | ok (r, x) => some (res (·.val) r, x.map (·.val)) | _ => none) == some ((false, 432), some 0)
#guard (match skip_try_rfold_dispatch (n 2) with
  | ok (r, x) => some (res (·.val) r, x.map (·.val)) | _ => none) == some ((true, 432), none)
#guard (match skip_try_rfold_dispatch (n 5) with
  | ok (r, x) => some (res (·.val) r, x.map (·.val)) | _ => none) == some ((true, 0), none)
#guard (match skip_rfold_dispatch (n 1) with | ok x => some x.val | _ => none) == some 321
#guard (match enumerate_rfold_dispatch with | ok x => some x.val | _ => none) == some 332211
#guard (match enumerate_advance_back_dispatch (n 1) with
  | ok (r, p) => some (r.val, upair p) | _ => none) == some (0, some (2, 2))
#guard (match enumerate_advance_back_dispatch (n 6) with
  | ok (r, p) => some (r.val, upair p) | _ => none) == some (2, none)
#guard (match filter_back_dispatch with
  | ok (a, b) => some (a.map (·.val), b.map (·.val)) | _ => none) == some (some 2, some 0)
#guard (match map_rfold_dispatch with | ok x => some x.val | _ => none) == some 9060300

-- native `index`: a missing key panics (`catch_unwind(..).is_err()`)
def map : alloc.collections.btree.map.BTreeMap U32 U32 := ⟨[(u 1, u 10), (u 3, u 30)]⟩
#guard (match lookup map (u 3) with | ok x => some x.val | _ => none) == some 30
#guard (match lookup map (u 2) with | fail .panic => true | _ => false)
def names : alloc.collections.btree.map.BTreeMap String U32 := ⟨[("a", u 1), ("b", u 2)]⟩
#guard (match lookup_str names (toStr "b") with | ok x => some x.val | _ => none) == some 2
#guard (match lookup_str names (toStr "c") with | fail .panic => true | _ => false)

end double_ended_adapters.checks
