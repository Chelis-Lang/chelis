import TupleModels

open Aeneas Aeneas.Std

private def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

private def outcome : Result α → Except String α
  | .ok value => .ok value
  | .fail _ => .error "failure"
  | .div => .error "divergence"

#eval do
  expect "eq stops before a failing second component"
    (outcome (tuple_models.pair_eq 0#u8 2#u8)) (.ok false)
  expect "ne uses the first override and short circuits"
    (outcome (tuple_models.pair_ne 1#u8 2#u8)) (.ok true)
  expect "ne uses the second override after a false prefix"
    (outcome (tuple_models.pair_ne 0#u8 1#u8)) (.ok true)
  expect "eq propagates a reached second failure"
    (outcome (tuple_models.pair_eq 1#u8 2#u8)) (.error "failure")
  expect "ne propagates a reached second failure"
    (outcome (tuple_models.pair_ne 0#u8 2#u8)) (.error "failure")
  expect "eq propagates the first failure before a false suffix"
    (outcome (tuple_models.pair_eq 2#u8 0#u8)) (.error "failure")
  expect "ne propagates the first failure before a true suffix"
    (outcome (tuple_models.pair_ne 2#u8 1#u8)) (.error "failure")
  expect "partial_cmp stops at an unequal prefix"
    (outcome (tuple_models.pair_partial_cmp 0#u8 4#u8)) (.ok (some .lt))
  expect "incomparable prefix does not evaluate the suffix"
    (outcome (tuple_models.pair_partial_cmp 3#u8 4#u8)) (.ok none)
  expect "equal prefix preserves incomparable suffix"
    (outcome (tuple_models.pair_partial_cmp 1#u8 3#u8)) (.ok none)
  expect "equal prefix delegates partial ordering to the suffix"
    (outcome (tuple_models.pair_partial_cmp 1#u8 2#u8)) (.ok (some .gt))
  expect "partial_cmp propagates prefix failure"
    (outcome (tuple_models.pair_partial_cmp 4#u8 0#u8)) (.error "failure")
  expect "partial_cmp propagates reached suffix failure"
    (outcome (tuple_models.pair_partial_cmp 1#u8 4#u8)) (.error "failure")
  expect "tuple lt preserves the last component override"
    (outcome tuple_models.pair_lt_override) (.ok true)
  expect "tuple relational operators preserve each prefix chaining override"
    (outcome (tuple_models.pair_chaining_overrides 5#u8)) (.ok (true, false, false, false))
  expect "chaining Break does not evaluate the failing suffix"
    (outcome (tuple_models.pair_chaining_overrides 6#u8)) (.ok (false, true, true, true))
  expect "Ordering preserves its signed Rust representation"
    (outcome (tuple_models.ordering_discriminant 0#u8 4#u8)) (.ok ((-1)#i8))
  expect "cmp stops at an unequal prefix"
    (outcome (tuple_models.pair_cmp 2#u8 4#u8)) (.ok .gt)
  expect "equal prefix delegates total ordering to the suffix"
    (outcome (tuple_models.pair_cmp 1#u8 0#u8)) (.ok .lt)
  expect "cmp propagates prefix failure"
    (outcome (tuple_models.pair_cmp 4#u8 0#u8)) (.error "failure")
  expect "cmp propagates reached suffix failure"
    (outcome (tuple_models.pair_cmp 1#u8 4#u8)) (.error "failure")
  expect "inherited max uses partial ordering and returns the right-hand equal value"
    (outcome tuple_models.pair_max_tie) (.ok 20#u32)
  expect "inherited min uses partial ordering and returns the left-hand equal value"
    (outcome tuple_models.pair_min_tie) (.ok 10#u32)
  expect "recursive boxed equality reaches both nested fields"
    (outcome (tuple_models.recursive_boxed_eq 3#u8 5#u8)) (.ok true)
  expect "recursive boxed equality observes a nested left difference"
    (outcome (tuple_models.recursive_boxed_eq 4#u8 5#u8)) (.ok false)
  expect "recursive boxed equality observes a nested right difference"
    (outcome (tuple_models.recursive_boxed_eq 3#u8 6#u8)) (.ok false)
  expect "recursive boxed inequality inherits negated equality"
    (outcome (tuple_models.recursive_boxed_ne 3#u8 5#u8)) (.ok false)
  expect "recursive boxed inequality observes a nested difference"
    (outcome (tuple_models.recursive_boxed_ne 3#u8 6#u8)) (.ok true)
  expect "recursive boxed equality skips a failing right subtree"
    (outcome (tuple_models.recursive_boxed_short_circuit 4#u8 2#u8)) (.ok false)
  expect "recursive boxed equality reaches a successful right subtree"
    (outcome (tuple_models.recursive_boxed_short_circuit 3#u8 1#u8)) (.ok true)
  expect "recursive boxed equality propagates a reached nested failure"
    (outcome (tuple_models.recursive_boxed_short_circuit 3#u8 2#u8)) (.error "failure")
  expect "shared boxed recursive ne overrides skip the failing right subtree"
    (outcome (tuple_models.recursive_boxed_ne_override 1#u8 2#u8)) (.ok true)
  expect "shared boxed recursive ne reaches the nested override, never eq"
    (outcome (tuple_models.recursive_boxed_ne_override 0#u8 1#u8)) (.ok true)
  expect "shared boxed recursive ne preserves a false override"
    (outcome (tuple_models.recursive_boxed_ne_override 0#u8 0#u8)) (.ok false)
  expect "shared boxed recursive ne propagates a reached nested failure"
    (outcome (tuple_models.recursive_boxed_ne_override 0#u8 2#u8)) (.error "failure")
  expect "shared boxed recursive ne propagates the first failure"
    (outcome (tuple_models.recursive_boxed_ne_override 2#u8 1#u8)) (.error "failure")
  IO.println "Tuple models: 36 behavioral checks passed"
