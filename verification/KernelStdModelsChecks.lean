import KernelStdModels

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def narrowed : Option I64 × Option I8 × Option I16 × Option I32 →
    Option Int × Option Int × Option Int × Option Int
  | (a, b, c, d) => (a.map (·.val), b.map (·.val), c.map (·.val), d.map (·.val))

def tensor : kernel_std_models.Tensor :=
  { values := ⟨[1#u32, 2#u32, 3#u32], by scalar_tac⟩, scale := 4#u32 }

#eval do
  expect "every conversion accepts a small value"
    (decode narrowed (kernel_std_models.narrow_all 5#i64))
    (.ok (some 5, some 5, some 5, some 5))
  expect "i8::MIN is in range"
    (decode narrowed (kernel_std_models.narrow_all (-128)#i64))
    (.ok (some (-128), some (-128), some (-128), some (-128)))
  expect "below i8::MIN"
    (decode narrowed (kernel_std_models.narrow_all (-129)#i64))
    (.ok (some (-129), none, some (-129), some (-129)))
  expect "above i16::MAX"
    (decode narrowed (kernel_std_models.narrow_all 40000#i64))
    (.ok (some 40000, none, none, some 40000))
  expect "i32::MIN is in range"
    (decode narrowed (kernel_std_models.narrow_all (-2147483648)#i64))
    (.ok (some (-2147483648), none, none, some (-2147483648)))
  expect "above i32::MAX"
    (decode narrowed (kernel_std_models.narrow_all 1099511627776#i64))
    (.ok (some 1099511627776, none, none, none))
  expect "usize::try_from(7)"
    (decode (fun (r : Option Usize) => r.map (fun (x : Usize) => x.val)) (kernel_std_models.index 7#i64)) (.ok (some 7))
  expect "usize::try_from(0)"
    (decode (fun (r : Option Usize) => r.map (fun (x : Usize) => x.val)) (kernel_std_models.index 0#i64)) (.ok (some 0))
  expect "usize::try_from(-1)"
    (decode (fun (r : Option Usize) => r.map (fun (x : Usize) => x.val)) (kernel_std_models.index (-1)#i64)) (.ok none)
  expect "usize::try_from(i64::MAX)"
    (decode (fun (r : Option Usize) => r.map (fun (x : Usize) => x.val)) (kernel_std_models.index 9223372036854775807#i64))
    (.ok (some 9223372036854775807))
  expect "f64::INFINITY bits"
    (decode (fun (bits : U64) => bits.val) (kernel_std_models.unbounded false)) (.ok 0x7ff0000000000000)
  expect "f64::NEG_INFINITY bits"
    (decode (fun (bits : U64) => bits.val) (kernel_std_models.unbounded true)) (.ok 0xfff0000000000000)
  expect "ToOwned blanket clones"
    (decode (fun (copy : kernel_std_models.Tensor) => ((copy.values.val.map (fun (x : U32) => x.val) : List Nat), copy.scale.val))
      (kernel_std_models.owned_copy tensor))
    (.ok ([1, 2, 3], 4))
  expect "Cow::Borrowed derefs to the borrowed tensor"
    (decode (fun ((scale, len) : U32 × Usize) => (scale.val, len.val))
      (kernel_std_models.cow_view tensor false))
    (.ok (4, 3))
  expect "Cow::Owned derefs through Borrow<Tensor>"
    (decode (fun ((scale, len) : U32 × Usize) => (scale.val, len.val))
      (kernel_std_models.cow_view tensor true))
    (.ok (5, 3))
  let order := fun ((a, b, c, d) : Bool × Bool × Bool × Bool) => [a, b, c, d]
  expect "false < true"
    (decode order (kernel_std_models.bool_order false true)) (.ok [true, true, false, false])
  expect "true > false"
    (decode order (kernel_std_models.bool_order true false)) (.ok [false, false, true, true])
  expect "true = true"
    (decode order (kernel_std_models.bool_order true true)) (.ok [false, true, false, true])
  expect "false = false"
    (decode order (kernel_std_models.bool_order false false)) (.ok [false, true, false, true])
