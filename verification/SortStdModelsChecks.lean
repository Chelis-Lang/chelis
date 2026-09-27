import SortStdModels

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def vec {α : Type} (l : List α) (h : l.length ≤ Usize.max := by scalar_tac) :
    alloc.vec.Vec α := ⟨l, h⟩

/-- The UTF-8 text of a `&str` (`Str` is its byte slice). -/
def text (s : Str) : String :=
  match Str.toString s with
  | .ok t => t
  | _ => "<invalid UTF-8>"

def slot : sort_std_models.Slot → String
  | .Free => "Free"
  | .Held a b => s!"Held({a.val}, {b})"

#eval show IO Unit from do
  expect "usizes" (decode (·.val.map (·.val))
    (sort_std_models.sort_usizes (vec [3#usize, 1#usize, 2#usize, 1#usize]))) (.ok [1, 1, 2, 3])
  expect "usizes reversed" (decode (·.val.map (·.val)) (sort_std_models.sort_usizes
    (vec [5#usize, 4#usize, 3#usize, 2#usize, 1#usize]))) (.ok [1, 2, 3, 4, 5])
  expect "empty" (decode (·.val.map (·.val)) (sort_std_models.sort_usizes (vec []))) (.ok [])
  expect "pairs" (decode (·.val.map fun ((a, b) : Usize × Usize) => (a.val, b.val))
    (sort_std_models.sort_pairs (vec [(2#usize, 1#usize), (1#usize, 9#usize), (2#usize, 0#usize)])))
    (.ok [(1, 9), (2, 0), (2, 1)])
  expect "strings" (decode (·.val) (sort_std_models.sort_strings (vec ["b", "ab", "a"])))
    (.ok ["a", "ab", "b"])
  expect "strs" (decode (·.val.map text) (sort_std_models.sort_strs
    (vec [toStr "z", toStr "", toStr "é", toStr "e"]))) (.ok ["", "e", "z", "é"])
  expect "flags" (decode (·.val.map fun ((a, c, n) : Bool × Char × I8) => (a, c, n.val))
    (sort_std_models.sort_flags (vec [(true, 'a', 1#i8), (false, 'b', (-1)#i8), (true, 'a', (-2)#i8)])))
    (.ok [(false, 'b', -1), (true, 'a', -2), (true, 'a', 1)])
  expect "derived newtype" (decode (·.val.map (·.val))
    (sort_std_models.sort_ids ⟨[7#u32, 3#u32, 5#u32, 3#u32], by scalar_tac⟩)) (.ok [3, 3, 5, 7])
  expect "derived enum" (decode (·.val.map slot) (sort_std_models.sort_slots
    (vec [.Held 2#u32 false, .Free, .Held 1#u32 true, .Held 2#u32 true])))
    (.ok ["Free", "Held(1, true)", "Held(2, false)", "Held(2, true)"])
  expect "descending" (decode (·.val.map (·.val)) (sort_std_models.sort_descending
    (vec [3#usize, 1#usize, 2#usize, 1#usize, 3#usize]))) (.ok [3, 3, 2, 1, 1])
  expect "descending reversed" (decode (·.val.map (·.val)) (sort_std_models.sort_descending
    (vec [1#usize, 2#usize, 3#usize, 4#usize]))) (.ok [4, 3, 2, 1])
  expect "descending empty" (decode (·.val.map (·.val)) (sort_std_models.sort_descending
    (vec []))) (.ok [])
  expect "ascending by" (decode (·.val.map fun ((n, c) : U8 × Char) => (n.val, c))
    (sort_std_models.sort_ascending_by (vec [(2#u8, 'b'), (1#u8, 'z'), (2#u8, 'a'), (1#u8, 'z')])))
    (.ok [(1, 'z'), (1, 'z'), (2, 'a'), (2, 'b')])
  expect "ascending by reversed" (decode (·.val.map fun ((n, c) : U8 × Char) => (n.val, c))
    (sort_std_models.sort_ascending_by (vec [(3#u8, 'c'), (2#u8, 'b'), (1#u8, 'a')])))
    (.ok [(1, 'a'), (2, 'b'), (3, 'c')])
  expect "derived descending" (decode (·.val.map (·.val))
    (sort_std_models.sort_ids_descending ⟨[3#u32, 7#u32, 3#u32, 5#u32], by scalar_tac⟩))
    (.ok [7, 5, 3, 3])
  expect "sort_by first component, stable" (decode (·.val.map fun ((k, v) : U32 × String) => (k.val, v))
    (sort_std_models.sort_by_first (vec [(2#u32, "b"), (1#u32, "z"), (2#u32, "a"), (1#u32, "y")])))
    (.ok [(1, "z"), (1, "y"), (2, "b"), (2, "a")])
  expect "sort_by first component, empty" (decode (·.val.map fun ((k, v) : U32 × String) => (k.val, v))
    (sort_std_models.sort_by_first (vec []))) (.ok [])
  expect "sort_by a tuple of borrowed fields"
    (decode (·.val.map fun ((a, c, n) : String × String × U32) => (a, c, n.val))
      (sort_std_models.sort_by_names
        (vec [("B", "x", 1#u32), ("A", "y", 2#u32), ("A", "x", 3#u32), ("B", "x", 4#u32)])))
    (.ok [("A", "x", 3), ("A", "y", 2), ("B", "x", 1), ("B", "x", 4)])
  expect "sort_by one field, descending, stable"
    (decode (·.val.map fun ((a, c, n) : String × String × U32) => (a, c, n.val))
      (sort_std_models.sort_by_name_descending
        (vec [("a", "", 1#u32), ("c", "", 2#u32), ("b", "", 3#u32), ("c", "", 4#u32)])))
    (.ok [("c", "", 2), ("c", "", 4), ("b", "", 3), ("a", "", 1)])
  expect "sort_by a tuple of copied fields"
    (decode (·.val.map fun ((a, b, c) : U32 × U8 × U8) => (a.val, b.val, c.val))
      (sort_std_models.sort_by_copied
        (vec [(1#u32, 5#u8, 2#u8), (2#u32, 0#u8, 1#u8), (3#u32, 9#u8, 2#u8), (0#u32, 0#u8, 1#u8)])))
    (.ok [(0, 0, 1), (2, 0, 1), (1, 5, 2), (3, 9, 2)])

def capacity : sort_std_models.Capacity → String
  | .Literal n => s!"L{n.val}"
  | .Product factors => s!"P{factors.val.map (·.val)}"
  | .Quotient a b => s!"Q({a.val},{b.val})"

def rawMembers : Slice (Option Usize × U32) :=
  ⟨[(some 2#usize, 10#u32), (none, 11#u32), (some 0#usize, 12#u32), (some 2#usize, 9#u32),
    (none, 8#u32)], by scalar_tac⟩

#eval show IO Unit from do
  expect "sort, derived newtype" (decode (·.val.map (·.val))
    (sort_std_models.sort_stable_ids ⟨[7#u32, 3#u32, 5#u32, 3#u32], by scalar_tac⟩)) (.ok [3, 3, 5, 7])
  expect "sort, empty" (decode (·.val.map (·.val))
    (sort_std_models.sort_stable_ids ⟨[], by scalar_tac⟩)) (.ok [])
  expect "sort, strings" (decode (·.val)
    (sort_std_models.sort_stable_strings (vec ["b", "ab", "a", "b"]))) (.ok ["a", "ab", "b", "b"])
  expect "sort, Option and Reverse"
    (decode (·.val.map fun ((o, r) : Option U32 × U8) => (o.map (·.val), r.val))
      (sort_std_models.sort_stable_options (vec [(some 2#u32, 1#u8), (none, 0#u8), (some 2#u32, 5#u8),
        (none, 3#u8), (some 1#u32, 0#u8)])))
    (.ok [(none, 3), (none, 0), (some 1, 0), (some 2, 5), (some 2, 1)])
  expect "sort, a derived Ord through boxes and boxed slices" (decode (·.val.map capacity)
    (sort_std_models.sort_capacities (vec [.Product ⟨[2#u32, 1#u32], by scalar_tac⟩, .Literal 3#u64,
      .Quotient 1#u32 2#u32, .Product ⟨[2#u32], by scalar_tac⟩, .Literal 1#u64])))
    (.ok ["L1", "L3", "P[2]", "P[2, 1]", "Q(1,2)"])
  expect "sort in place, the given-back slice"
    (decode (·.val.map fun ((n, s) : U8 × sort_std_models.Slot) => (n.val, slot s))
      (sort_std_models.sort_stable_in_place ⟨[(2#u8, .Free), (1#u8, .Held 3#u32 true),
        (2#u8, .Held 0#u32 false), (1#u8, .Free)], by scalar_tac⟩))
    (.ok [(1, "Free"), (1, "Held(3, true)"), (2, "Free"), (2, "Held(0, false)")])
  expect "sort_by_key a field, stable" (decode (·.val.map fun ((k, c) : U32 × Char) => (k.val, c))
    (sort_std_models.sort_by_key_field (vec [(2#u32, 'b'), (1#u32, 'z'), (2#u32, 'a'), (1#u32, 'y')])))
    (.ok [(1, 'z'), (1, 'y'), (2, 'b'), (2, 'a')])
  expect "sort_by_key a tuple of fields"
    (decode (·.val.map fun ((a, b, c) : U8 × U32 × Char) => (a.val, b.val, c))
      (sort_std_models.sort_by_key_fields (vec [(1#u8, 5#u32, 'a'), (0#u8, 3#u32, 'b'), (2#u8, 3#u32, 'c'),
        (0#u8, 5#u32, 'd'), (0#u8, 3#u32, 'e')])))
    (.ok [(0, 3, 'b'), (0, 3, 'e'), (2, 3, 'c'), (0, 5, 'd'), (1, 5, 'a')])
  expect "sort_by_key Reverse, stable" (decode (·.val.map fun ((k, c) : Usize × Char) => (k.val, c))
    (sort_std_models.sort_by_key_reverse (vec [(1#usize, 'a'), (3#usize, 'b'), (1#usize, 'c'),
      (2#usize, 'd'), (3#usize, 'e')])))
    (.ok [(3, 'b'), (3, 'e'), (2, 'd'), (1, 'a'), (1, 'c')])
  expect "sort_by_key as_str" (decode (·.val.map (·.val))
    (sort_std_models.sort_by_key_str (vec [("b", 1#u32), ("a", 2#u32), ("b", 3#u32), ("", 4#u32)])))
    (.ok [4, 2, 1, 3])
  expect "sort_by_key through a captured table" (decode (·.val.map (·.val))
    (sort_std_models.sort_by_key_captured (vec [0#usize, 1#usize, 2#usize, 3#usize])
      ⟨[5#u32, 7#u32, 5#u32, 9#u32], by scalar_tac⟩)) (.ok [3, 1, 0, 2])
  expect "sort_by_key one element: no key computed" (decode (·.val.map (·.val))
    (sort_std_models.sort_by_key_captured (vec [4#usize]) ⟨[1#u32], by scalar_tac⟩)) (.ok [4])
  expect "sort_by_key key out of bounds: panic" (decode (·.val.map (·.val))
    (sort_std_models.sort_by_key_captured (vec [0#usize, 4#usize]) ⟨[1#u32], by scalar_tac⟩))
    (.error "failure")
  expect "sort_by_key a method" (decode (·.val.map (·.val))
    (sort_std_models.sort_by_key_fn_item rawMembers)) (.ok [12, 9, 10, 8, 11])
  expect "sort_by_key a closure variable, by node" (decode (·.val.map (·.val))
    (sort_std_models.sort_by_key_local_closure rawMembers true)) (.ok [8, 9, 10, 11, 12])
  expect "sort_by_key a closure variable, by key" (decode (·.val.map (·.val))
    (sort_std_models.sort_by_key_local_closure rawMembers false)) (.ok [12, 9, 10, 8, 11])
  expect "sort_by_key a cloned String"
    (decode (·.val.map fun ((i, s, n) : Usize × String × U32) => (i.val, s, n.val))
      (sort_std_models.sort_by_key_cloned (vec [(1#usize, "b", 0#u32), (0#usize, "z", 1#u32),
        (1#usize, "a", 2#u32), (1#usize, "b", 3#u32)])))
    (.ok [(0, "z", 1), (1, "a", 2), (1, "b", 0), (1, "b", 3)])
  expect "sort_by_key in place, the given-back slice"
    (decode (·.val.map fun ((a, b) : U32 × U32) => (a.val, b.val))
      (sort_std_models.sort_by_key_in_place ⟨[(1#u32, 3#u32), (2#u32, 1#u32), (3#u32, 3#u32),
        (4#u32, 0#u32)], by scalar_tac⟩))
    (.ok [(4, 0), (2, 1), (1, 3), (3, 3)])
