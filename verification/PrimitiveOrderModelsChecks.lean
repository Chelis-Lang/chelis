import PrimitiveOrderModels

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def order : Ordering → Int
  | .lt => -1
  | .eq => 0
  | .gt => 1

def four : Bool × Bool × Bool × Bool → List Bool
  | (a, b, c, d) => [a, b, c, d]

/-- `impl Ord for bool`'s `cmp` never reaches its `unreachable_unchecked` arm,
    for every input. The proof does not depend on the opaque
    `core.intrinsics.ub_checks` configuration, so the model agrees with debug
    builds (checks on) and release builds (checks off). -/
theorem bool_cmp_is_false_before_true (a b : Bool) :
    primitive_order_models.bool_cmp a b =
      .ok (if a = b then .eq else if a then .gt else .lt) := by
  cases a <;> cases b <;> rfl

#eval do
  expect "false < true" (decode order (primitive_order_models.bool_cmp false true)) (.ok (-1))
  expect "true > false" (decode order (primitive_order_models.bool_cmp true false)) (.ok 1)
  expect "true = true" (decode order (primitive_order_models.bool_cmp true true)) (.ok 0)
  expect "false = false" (decode order (primitive_order_models.bool_cmp false false)) (.ok 0)
  expect "partial_cmp false true"
    (decode (Option.map order) (primitive_order_models.bool_partial_cmp false true)) (.ok (some (-1)))
  expect "partial_cmp true true"
    (decode (Option.map order) (primitive_order_models.bool_partial_cmp true true)) (.ok (some 0))
  expect "min/max false true" (decode id (primitive_order_models.bool_min_max false true))
    (.ok (false, true))
  expect "min/max true true" (decode id (primitive_order_models.bool_min_max true true))
    (.ok (true, true))
  expect "min/max false false" (decode id (primitive_order_models.bool_min_max false false))
    (.ok (false, false))
  expect "clamp up" (decode id (primitive_order_models.bool_clamp false true true)) (.ok true)
  expect "clamp down" (decode id (primitive_order_models.bool_clamp true false false)) (.ok false)
  expect "clamp inside" (decode id (primitive_order_models.bool_clamp true false true)) (.ok true)
  expect "clamp with min > max panics"
    (decode id (primitive_order_models.bool_clamp false true false)) (.error "failure")
  expect "trait lt/le/gt/ge false true"
    (decode four (primitive_order_models.bool_trait_order false true))
    (.ok [true, true, false, false])
  expect "trait lt/le/gt/ge true false"
    (decode four (primitive_order_models.bool_trait_order true false))
    (.ok [false, false, true, true])
  expect "trait lt/le/gt/ge true true"
    (decode four (primitive_order_models.bool_trait_order true true))
    (.ok [false, true, false, true])
  expect "& | ^ false false" (decode id (primitive_order_models.bool_bits false false))
    (.ok (false, false, false))
  expect "& | ^ false true" (decode id (primitive_order_models.bool_bits false true))
    (.ok (false, true, true))
  expect "& | ^ true false" (decode id (primitive_order_models.bool_bits true false))
    (.ok (false, true, true))
  expect "& | ^ true true" (decode id (primitive_order_models.bool_bits true true))
    (.ok (true, true, false))
  expect "eager & true" (decode id (primitive_order_models.eager_and true 1#u8)) (.ok true)
  expect "eager & false" (decode id (primitive_order_models.eager_and false 1#u8)) (.ok false)
  expect "eager & evaluates its right operand"
    (decode id (primitive_order_models.eager_and false 255#u8)) (.error "failure")
  expect "a < b" (decode four (primitive_order_models.char_order 'a' 'b'))
    (.ok [true, true, false, false])
  expect "z > a" (decode four (primitive_order_models.char_order 'z' 'a'))
    (.ok [false, false, true, true])
  expect "q = q" (decode four (primitive_order_models.char_order 'q' 'q'))
    (.ok [false, true, false, true])
  expect "Z < a" (decode four (primitive_order_models.char_order 'Z' 'a'))
    (.ok [true, true, false, false])
  expect "U+10FFFF > U+E000"
    (decode four (primitive_order_models.char_order (Char.ofNat 0x10FFFF) (Char.ofNat 0xE000)))
    (.ok [false, false, true, true])
  let both := fun ((o, p) : Ordering × Option Ordering) => (order o, p.map order)
  expect "cmp a b" (decode both (primitive_order_models.char_cmp 'a' 'b')) (.ok (-1, some (-1)))
  expect "cmp U+E9 e" (decode both (primitive_order_models.char_cmp (Char.ofNat 0xE9) 'e'))
    (.ok (1, some 1))
  expect "cmp x x" (decode both (primitive_order_models.char_cmp 'x' 'x')) (.ok (0, some 0))
  expect "a starts an identifier" (decode id (primitive_order_models.is_ident_start 'a')) (.ok true)
  expect "Z starts an identifier" (decode id (primitive_order_models.is_ident_start 'Z')) (.ok true)
  expect "_ starts an identifier" (decode id (primitive_order_models.is_ident_start '_')) (.ok true)
  expect "0 does not" (decode id (primitive_order_models.is_ident_start '0')) (.ok false)
  expect "{ does not" (decode id (primitive_order_models.is_ident_start '{')) (.ok false)
  expect "U+E9 does not"
    (decode id (primitive_order_models.is_ident_start (Char.ofNat 0xE9))) (.ok false)
