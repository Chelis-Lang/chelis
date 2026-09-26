import StrCharPatterns

/-! The generated model of `verification/str_char_patterns.rs` on the inputs of
    its native test module: every `#guard` states the native result asserted
    there. -/
namespace str_char_patterns.checks
open Aeneas Aeneas.Std Result

def txt (x : Str) : Option String :=
  match Str.toString x with | ok t => some t | _ => none
def str (r : Result Str) : Option String :=
  match r with | ok s => txt s | _ => none
def optStr (r : Result (Option Str)) : Option (Option String) :=
  match r with | ok (some s) => (txt s).map some | ok none => some none | _ => none
def strs (r : Result (alloc.vec.Vec Str)) : Option (List String) :=
  match r with | ok v => v.val.mapM txt | _ => none
def pair (r : Result (Option (Str × Str))) : Option (Option (String × String)) :=
  match r with
  | ok none => some none
  | ok (some (a, b)) => do some (some (← txt a, ← txt b))
  | _ => none
def bool (r : Result Bool) : Option Bool :=
  match r with | ok b => some b | _ => none

-- production shape (native `production_shape`)
#guard str (terminal_segment (toStr "Pkg__opq__Demo__Types__Probability")) == some "Probability"
#guard str (terminal_segment (toStr "opq.Demo.Types")) == some "Types"
#guard str (terminal_segment (toStr "plain")) == some "plain"
#guard str (terminal_segment (toStr "a__b.c")) == some "b.c"
#guard optStr (module_part (toStr "opq__Demo__Types__Probability")) == some (some "opq__Demo__Types")
#guard optStr (module_part (toStr "__x")) == some (some "")
#guard optStr (module_part (toStr "x")) == some none
#guard strs (dot_segments (toStr "opq.Demo.Types")) == some ["opq", "Demo", "Types"]
#guard strs (dot_segments (toStr "")) == some [""]
#guard strs (dot_segments (toStr ".a.")) == some ["", "a", ""]

-- the pieces between the occurrences of a char (native `char_occurrences`)
#guard strs (char_pieces (toStr "héllo wörld") 'ö') == some ["héllo w", "rld"]
#guard strs (char_pieces (toStr "aéaé") 'é') == some ["a", "a", ""]
#guard strs (char_pieces (toStr "abc") 'x') == some ["abc"]
#guard strs (char_pieces (toStr "") 'a') == some [""]
#guard strs (char_pieces (toStr "a→b") '→') == some ["a", "b"]
#guard strs (char_pieces (toStr "𝄞x𝄞") '𝄞') == some ["", "x", ""]

-- the first match of the reverse `StrSearcher` (native `last_str_match`)
#guard pair (rsplit_str (toStr "a__b__c") (toStr "__")) == some (some ("a__b", "c"))
#guard pair (rsplit_str (toStr "aaa") (toStr "aa")) == some (some ("a", ""))
#guard pair (rsplit_str (toStr "aaaa") (toStr "aa")) == some (some ("aa", ""))
#guard pair (rsplit_str (toStr "a____b") (toStr "__")) == some (some ("a__", "b"))
#guard pair (rsplit_str (toStr "abc") (toStr "")) == some (some ("abc", ""))
#guard pair (rsplit_str (toStr "") (toStr "")) == some (some ("", ""))
#guard pair (rsplit_str (toStr "abc") (toStr "x")) == some none
#guard pair (rsplit_str (toStr "ab") (toStr "abc")) == some none
#guard pair (rsplit_str (toStr "x→y→z") (toStr "→")) == some (some ("x→y", "z"))

-- the first match of the reverse `CharSearcher` (native `last_char_match`)
#guard pair (rsplit_char (toStr "a.b.c") '.') == some (some ("a.b", "c"))
#guard pair (rsplit_char (toStr "xéyéz") 'é') == some (some ("xéy", "z"))
#guard pair (rsplit_char (toStr "abc") '.') == some none
#guard pair (rsplit_char (toStr "") 'a') == some none
#guard pair (rsplit_char (toStr "𝄞a𝄞") '𝄞') == some (some ("𝄞a", ""))

-- the char prefix test (native `char_prefix`)
#guard bool (starts_with_char (toStr "é!") 'é') == some true
#guard bool (starts_with_char (toStr "") 'a') == some false
#guard bool (starts_with_char (toStr "ab") 'b') == some false
#guard bool (starts_with_char (toStr "𝄞") '𝄞') == some true

end str_char_patterns.checks
