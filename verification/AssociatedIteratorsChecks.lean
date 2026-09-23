import AssociatedIterators

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def resume (value : Option Usize × Option Usize × Usize) : Option Nat × Option Nat × Nat :=
  (value.1.map (·.val), value.2.1.map (·.val), value.2.2.val)

def anyResume (value : Bool × Option Usize) : Bool × Option Nat :=
  (value.1, value.2.map (·.val))

def filterResume (value : Option Usize × Option Usize × Option Usize) :
  Option Nat × Option Nat × Option Nat :=
  (value.1.map (·.val), value.2.1.map (·.val), value.2.2.map (·.val))

def filterMapCollected (value : alloc.vec.Vec Usize × Usize) : List Nat × Nat :=
  (value.1.val.map UScalar.val, value.2.val)

def panicked {α : Type} : Result α → Bool
  | .fail .panic | .fail .assertionFailure => true
  | _ => false

#eval do
  let values : Slice associated_iterators.DifferentClone := ⟨[
    { value := 1#usize, panic_on_clone := false },
    { value := 2#usize, panic_on_clone := false },
    { value := 3#usize, panic_on_clone := false }], by scalar_tac⟩
  let empty : Slice associated_iterators.DifferentClone := ⟨[], by scalar_tac⟩
  let latePanic : Slice associated_iterators.DifferentClone := ⟨[
    { value := 1#usize, panic_on_clone := false },
    { value := 2#usize, panic_on_clone := true }], by scalar_tac⟩
  let immediatePanic : Slice associated_iterators.DifferentClone := ⟨[
    { value := 2#usize, panic_on_clone := true }], by scalar_tac⟩
  let numbers : Slice Usize := ⟨[1#usize, 2#usize, 3#usize], by scalar_tac⟩
  let longer : Slice Usize := ⟨[1#usize, 2#usize, 3#usize, 4#usize, 5#usize], by scalar_tac⟩
  let singleton : Slice Usize := ⟨[1#usize], by scalar_tac⟩
  let noNumbers : Slice Usize := ⟨[], by scalar_tac⟩
  expect "copied never invokes custom Clone"
    (decode UScalar.val (associated_iterators.copied_signature values)) (.ok 10203)
  expect "cloned invokes selected Clone exactly once per Some"
    (decode UScalar.val (associated_iterators.cloned_signature values)) (.ok 111213)
  expect "copied preserves continuation across next and count"
    (decode resume (associated_iterators.copied_resume values)) (.ok (some 1, some 2, 1))
  expect "cloned preserves continuation across next and count"
    (decode resume (associated_iterators.cloned_resume values)) (.ok (some 11, some 12, 1))
  expect "copied any resumes after its match"
    (decode anyResume (associated_iterators.copied_any_resumes values 2#usize)) (.ok (true, some 3))
  expect "cloned any observes cloned values and resumes"
    (decode anyResume (associated_iterators.cloned_any_resumes values 12#usize)) (.ok (true, some 13))
  expect "cloned any consumes its last match"
    (decode anyResume (associated_iterators.cloned_any_resumes values 13#usize)) (.ok (true, none))
  expect "unmatched copied any exhausts the iterator"
    (decode anyResume (associated_iterators.copied_any_resumes values 99#usize)) (.ok (false, none))
  expect "empty copied loop" (decode UScalar.val (associated_iterators.copied_signature empty)) (.ok 0)
  expect "empty cloned loop" (decode UScalar.val (associated_iterators.cloned_signature empty)) (.ok 0)
  expect "empty copied remains exhausted"
    (decode resume (associated_iterators.copied_resume empty)) (.ok (none, none, 0))
  expect "empty cloned remains exhausted"
    (decode resume (associated_iterators.cloned_resume empty)) (.ok (none, none, 0))
  expect "empty cloned next does not call Clone"
    (decode (Option.map UScalar.val) (associated_iterators.cloned_first empty)) (.ok none)
  expect "empty cloned any is false"
    (decode anyResume (associated_iterators.cloned_any_resumes empty 12#usize)) (.ok (false, none))
  expect "copied does not trigger a panicking Clone"
    (decode UScalar.val (associated_iterators.copied_signature latePanic)) (.ok 102)
  expect "copied count does not trigger a panicking Clone"
    (decode UScalar.val (associated_iterators.copied_count latePanic)) (.ok 2)
  expect "cloned is lazy before the panicking element"
    (decode (Option.map UScalar.val) (associated_iterators.cloned_first latePanic)) (.ok (some 11))
  expect "cloned propagates failure on a later element"
    (panicked (associated_iterators.cloned_signature latePanic)) true
  expect "cloned count still invokes Clone"
    (panicked (associated_iterators.cloned_count latePanic)) true
  expect "cloned next propagates the selected Clone failure"
    (panicked (associated_iterators.cloned_first immediatePanic)) true
  expect "copied count selects the inner explicit override"
    (decode UScalar.val (associated_iterators.copied_count_override numbers)) (.ok 103)
  expect "cloned count uses its own default instead of the inner override"
    (decode UScalar.val (associated_iterators.cloned_count_override numbers)) (.ok 3)
  expect "copied count preserves even an empty inner override"
    (decode UScalar.val (associated_iterators.copied_count_override noNumbers)) (.ok 100)
  expect "empty cloned count follows next"
    (decode UScalar.val (associated_iterators.cloned_count_override noNumbers)) (.ok 0)
  expect "filter threads predicate state across rejected and accepted elements"
    (decode filterResume (associated_iterators.filter_resume longer)) (.ok (some 2, some 4, none))
  expect "filter exhausts when every element is rejected"
    (decode filterResume (associated_iterators.filter_resume singleton)) (.ok (none, none, none))
  expect "empty filter remains exhausted"
    (decode filterResume (associated_iterators.filter_resume noNumbers)) (.ok (none, none, none))
  expect "filter count invokes the stateful predicate once per element"
    (decode UScalar.val (associated_iterators.filter_count longer)) (.ok 2)
  expect "empty filter count" (decode UScalar.val (associated_iterators.filter_count noNumbers)) (.ok 0)
  expect "empty filter never invokes the panicking predicate"
    (decode (Option.map UScalar.val) (associated_iterators.filter_panics noNumbers)) (.ok none)
  expect "filter propagates predicate failure"
    (panicked (associated_iterators.filter_panics singleton)) true
  let owned : Aeneas.Std.Array Usize 3#usize :=
    ⟨[3#usize, 5#usize, 29#usize], by scalar_tac⟩
  let ownedEmpty : Aeneas.Std.Array Usize 0#usize := ⟨[], by scalar_tac⟩
  expect "owned array count consumes only the remaining suffix"
    (decode resume (associated_iterators.owned_array_resume owned))
    (.ok (some 3, some 5, 1))
  expect "empty owned arrays remain exhausted"
    (decode resume (associated_iterators.owned_array_resume ownedEmpty))
    (.ok (none, none, 0))
  expect "owned array any resumes immediately after its match"
    (decode anyResume (associated_iterators.owned_array_any_resume owned 5#usize))
    (.ok (true, some 29))
  expect "unmatched owned array any exhausts the iterator"
    (decode anyResume (associated_iterators.owned_array_any_resume owned 99#usize))
    (.ok (false, none))
  expect "copied nth selects the inner indexed override"
    (decode (Option.map UScalar.val) (associated_iterators.copied_nth_override numbers))
    (.ok (some 3))
  expect "map fold selects the inner fold override"
    (decode UScalar.val (associated_iterators.mapped_fold_override numbers)) (.ok 131211)
  expect "cloned fold selects the inner fold override"
    (decode UScalar.val (associated_iterators.cloned_fold_override numbers)) (.ok 30201)
  expect "borrowed fold selects try_fold instead of the consuming fold override"
    (decode (fun value : Usize × Usize × Usize =>
      (value.1.val, value.2.1.val, value.2.2.val))
      (associated_iterators.borrowed_fold_override numbers)) (.ok (10203, 3, 3))
  expect "map nth invokes mapper for skipped values and preserves mutable captures"
    (decode resume (associated_iterators.map_nth_state numbers)) (.ok (some 4, some 6, 3))
  expect "filter next selects the inner custom find, retaining callback state"
    (decode resume (associated_iterators.filter_find_override longer))
    (.ok (some 4, some 3, 3))
  expect "filter_map next selects custom find_map with a distinct output type"
    (decode resume (associated_iterators.filter_map_find_override longer))
    (.ok (some 14, some 13, 3))
  expect "nested Filter.find retains pinned inherited try_fold dispatch"
    (decode (fun value : Option Usize × Option Usize × Usize × Usize =>
      (value.1.map UScalar.val, value.2.1.map UScalar.val,
      value.2.2.1.val, value.2.2.2.val))
      (associated_iterators.nested_filter_find_dispatch longer))
    (.ok (some 3, some 4, 4, 3))
  expect "all stops at false, consumes the match, and retains mutable captures"
    (decode (fun value : Bool × Option Usize × Usize =>
      (value.1, value.2.1.map UScalar.val, value.2.2.val))
      (associated_iterators.all_state numbers)) (.ok (false, some 3, 2))
  expect "find accepts shared-item predicates and resumes after the match"
    (decode resume (associated_iterators.find_state numbers)) (.ok (some 2, some 3, 2))
  expect "find_map retains its generic output and callback state"
    (decode (fun value : Option Bool × Option Usize × Usize =>
      (value.1, value.2.1.map UScalar.val, value.2.2.val))
      (associated_iterators.find_map_state numbers)) (.ok (some true, some 3, 2))
  expect "position returns the relative index without consuming the suffix"
    (decode resume (associated_iterators.position_state numbers)) (.ok (some 1, some 3, 2))
  expect "empty all is true without invoking the callback"
    (decode (fun value : Bool × Option Usize × Usize =>
      (value.1, value.2.1.map UScalar.val, value.2.2.val))
      (associated_iterators.all_state noNumbers)) (.ok (true, none, 0))
  let unmatched : Slice Usize := ⟨[1#usize, 3#usize], by scalar_tac⟩
  expect "unmatched find exhausts with its final callback state"
    (decode resume (associated_iterators.find_state unmatched)) (.ok (none, none, 2))
  expect "find_map exhaustion preserves a rejected callback's mutation"
    (decode (fun value : Option Bool × Option Usize × Usize =>
      (value.1, value.2.1.map UScalar.val, value.2.2.val))
      (associated_iterators.find_map_state singleton)) (.ok (none, none, 1))
  expect "unmatched position returns none after exhausting"
    (decode resume (associated_iterators.position_state unmatched)) (.ok (none, none, 2))
  expect "Rev.find selects the inner rfind override"
    (decode resume (associated_iterators.reversed_find_override longer))
    (.ok (some 2, some 3, 2))
  expect "Rev.rfind selects the inner find override"
    (decode resume (associated_iterators.reversed_rfind_override longer))
    (.ok (some 4, some 3, 2))
  expect "concrete filter_map collect returns every borrowed callback mutation"
    (decode filterMapCollected (associated_iterators.borrowed_filter_map_collect longer))
    (.ok ([4, 8], 5))
  expect "concrete filter_map collect preserves rejected callback state"
    (decode filterMapCollected (associated_iterators.borrowed_filter_map_collect singleton))
    (.ok ([], 1))
  expect "empty filter_map collect never invokes its borrowed callback"
    (decode filterMapCollected (associated_iterators.borrowed_filter_map_collect noNumbers))
    (.ok ([], 0))
  IO.println "Associated iterator behavioral checks passed"
