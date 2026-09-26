import StrSplit

/-! The generated model of `verification/str_split.rs` on the inputs of its
    native test module: every `#guard` states the native result asserted
    there. -/
namespace str_split.checks
open Aeneas Aeneas.Std Result

def txt (x : Str) : Option String :=
  match Str.toString x with | ok t => some t | _ => none
def strs (r : Result (alloc.vec.Vec Str)) : Option (List String) :=
  match r with | ok v => v.val.mapM txt | _ => none
def opt (x : Option Str) : Option (Option String) :=
  match x with | some x => (txt x).map some | none => some none

-- production shape (native `production_shape`)
#guard strs (segments (toStr "Pkg__Mod__name__x")) == some ["Pkg", "Mod", "name", "x"]
#guard strs (segments (toStr "__Entry__run")) == some ["", "Entry", "run"]
#guard strs (segments (toStr "plain")) == some ["plain"]
#guard strs (segments (toStr "a____b")) == some ["a", "", "b"]

-- pieces between the leftmost non-overlapping matches (native `pieces_between_matches`)
#guard strs (pieces (toStr "") (toStr "__")) == some [""]
#guard strs (pieces (toStr "aaa") (toStr "aa")) == some ["", "a"]
#guard strs (pieces (toStr "aaaa") (toStr "aa")) == some ["", "", ""]
#guard strs (pieces (toStr "__a__b__") (toStr "__")) == some ["", "a", "b", ""]
#guard strs (pieces (toStr "ab") (toStr "abc")) == some ["ab"]
#guard strs (pieces (toStr "ab") (toStr "ab")) == some ["", ""]
#guard strs (pieces (toStr "héllo wörld") (toStr "ö")) == some ["héllo w", "rld"]
#guard strs (pieces (toStr "x→y→z") (toStr "→")) == some ["x", "y", "z"]

-- the empty needle matches at every char boundary (native `empty_needle`)
#guard strs (pieces (toStr "") (toStr "")) == some ["", ""]
#guard strs (pieces (toStr "abc") (toStr "")) == some ["", "a", "b", "c", ""]
#guard strs (pieces (toStr "aé→") (toStr "")) == some ["", "a", "é", "→", ""]

-- the iterator (native `iterator`)
#guard (match first_two (toStr "a,b") (toStr ",") with
  | ok (a, b) => some (opt a, opt b) | _ => none) == some (some (some "a"), some (some "b"))
#guard (match first_two (toStr "") (toStr ",") with
  | ok (a, b) => some (opt a, opt b) | _ => none) == some (some (some ""), some none)
#guard (match fused (toStr "a,b,,c") (toStr ",") with
  | ok (n, a, b) => some (n.val, opt a, opt b) | _ => none) == some (4, some none, some none)
#guard (match fused (toStr "") (toStr "") with
  | ok (n, a, b) => some (n.val, opt a, opt b) | _ => none) == some (2, some none, some none)
#guard (match hint (toStr "a__b") (toStr "__") with
  | ok (l, u) => some (l.val, u.map (·.val)) | _ => none) == some (0, none)

end str_split.checks
