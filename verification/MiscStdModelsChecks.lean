import MiscStdModels

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def naturals (v : alloc.vec.Vec U32) : List Nat := v.val.map (·.val)
def optNat (o : Option U32) : Option Nat := o.map (·.val)
def pairs (v : alloc.vec.Vec (U32 × U32)) : List (Nat × Nat) := v.val.map fun (a, b) => (a.val, b.val)

#eval do
  expect "take a Vec"
    (decode (fun ((v, n) : alloc.vec.Vec U32 × Usize) => (naturals v, n.val))
      (misc_std_models.take_vec ⟨[1#u32, 2#u32], by scalar_tac⟩)) (.ok ([1, 2], 0))
  expect "take Some"
    (decode (fun ((a, b) : Option U32 × Option U32) => (optNat a, optNat b))
      (misc_std_models.take_option (some 3#u32))) (.ok (some 3, none))
  expect "take None"
    (decode (fun ((a, b) : Option U32 × Option U32) => (optNat a, optNat b))
      (misc_std_models.take_option none)) (.ok (none, none))
  expect "Option default" (decode optNat misc_std_models.default_option) (.ok none)
  expect "upper hex 0x0a" (decode id (misc_std_models.upper_hex 0x0a#u8)) (.ok "0A|A|0xA")
  expect "upper hex 0xff" (decode id (misc_std_models.upper_hex 0xff#u8)) (.ok "FF|FF|0xFF")
  expect "upper hex 0" (decode id (misc_std_models.upper_hex 0#u8)) (.ok "00|0|0x0")
  expect "debug slice"
    (decode id (misc_std_models.debug_slice ⟨[1#u32, 2#u32, 3#u32], by scalar_tac⟩ false))
    (.ok "[1, 2, 3]")
  expect "debug empty slice"
    (decode id (misc_std_models.debug_slice ⟨[], by scalar_tac⟩ false)) (.ok "[]")
  expect "pretty debug slice"
    (decode id (misc_std_models.debug_slice ⟨[7#u32, 8#u32], by scalar_tac⟩ true))
    (.ok "[\n    7,\n    8,\n]")
  expect "same variant, other payload"
    (decode id (misc_std_models.same_kind (.Word 1#u32) (.Word 2#u32))) (.ok true)
  expect "other variants"
    (decode id (misc_std_models.same_kind (.Word 1#u32) .Comma)) (.ok false)
  expect "same unit variant" (decode id (misc_std_models.same_kind .End .End)) (.ok true)
  expect "Debug of a discriminant"
    (decode id (misc_std_models.discriminant_debug (.Word 5#u32))) (.ok "Discriminant(0)")
  expect "Debug of the last discriminant"
    (decode id (misc_std_models.discriminant_debug .End)) (.ok "Discriminant(2)")
  expect "BTreeMap clone"
    (decode (fun ((v, n) : alloc.vec.Vec (U32 × U32) × Usize) => (pairs v, n.val))
      (misc_std_models.cloned_map ⟨[(3#u32, 30#u32), (1#u32, 10#u32), (3#u32, 31#u32)], by scalar_tac⟩))
    (.ok ([(1, 10), (3, 31)], 2))
  expect "empty BTreeMap clone"
    (decode (fun ((v, n) : alloc.vec.Vec (U32 × U32) × Usize) => (pairs v, n.val))
      (misc_std_models.cloned_map ⟨[], by scalar_tac⟩)) (.ok ([], 0))
  expect "FnMut/FnOnce through &mut F"
    (decode (fun ((a, b, c) : U32 × U32 × U32) => (a.val, b.val, c.val))
      (misc_std_models.through_mut_ref 3#u32)) (.ok (3, 7, 7))
  expect "Option<&mut u32>::expect on Some (value, slot after)"
    (decode (fun ((v, s) : U32 × Option U32) => (v.val, optNat s))
      (misc_std_models.expect_slot (some 3#u32))) (.ok (4, some 4))
  expect "Option<&mut u32>::expect on None panics (expect_failed)"
    (decode (fun ((v, s) : U32 × Option U32) => (v.val, optNat s))
      (misc_std_models.expect_slot none)) (.error "failure")
  expect "Vec::drain left early removes the whole range"
    (decode (fun ((t, v) : alloc.vec.Vec U32 × alloc.vec.Vec U32) => (naturals t, naturals v))
      (misc_std_models.drain_until ⟨[1#u32, 2#u32, 3#u32, 4#u32], by scalar_tac⟩ 3#u32))
    (.ok ([2], [1, 3]))
  expect "Vec::drain exhausted"
    (decode (fun ((t, v) : alloc.vec.Vec U32 × alloc.vec.Vec U32) => (naturals t, naturals v))
      (misc_std_models.drain_until ⟨[1#u32, 2#u32], by scalar_tac⟩ 9#u32))
    (.ok ([2], [1, 9]))
  expect "Vec::drain of an empty range"
    (decode (fun ((t, v) : alloc.vec.Vec U32 × alloc.vec.Vec U32) => (naturals t, naturals v))
      (misc_std_models.drain_until ⟨[5#u32], by scalar_tac⟩ 9#u32))
    (.ok ([], [5, 9]))
