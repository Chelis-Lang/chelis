import EvalStdModels

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def ints : Option I8 × Option I16 × Option I32 × Option I64 × Option I8 × Option I16 ×
    Option I32 × Option I64 → List (Option Int)
  | (a, b, c, d, e, f, g, h) =>
    [a.map (·.val), b.map (·.val), c.map (·.val), d.map (·.val),
     e.map (·.val), f.map (·.val), g.map (·.val), h.map (·.val)]

def naturals (v : alloc.vec.Vec U32) : List Nat := v.val.map (·.val)

#eval do
  expect "checked_abs/checked_neg"
    (decode ints (eval_std_models.signed_checks (-5)#i8 7#i16 (-9)#i32 11#i64))
    (.ok [some 5, some 7, some 9, some 11, some 5, some (-7), some 9, some (-11)])
  expect "checked_abs/checked_neg of the minimum values"
    (decode ints (eval_std_models.signed_checks (-128)#i8 (-32768)#i16 (-2147483648)#i32
      (-9223372036854775808)#i64))
    (.ok [none, none, none, none, none, none, none, none])
  expect "checked_abs/checked_neg of the maximum values"
    (decode ints (eval_std_models.signed_checks 127#i8 32767#i16 2147483647#i32
      9223372036854775807#i64))
    (.ok [some 127, some 32767, some 2147483647, some 9223372036854775807, some (-127),
      some (-32767), some (-2147483647), some (-9223372036854775807)])
  expect "checked_abs/checked_neg of zero"
    (decode ints (eval_std_models.signed_checks 0#i8 0#i16 0#i32 0#i64))
    (.ok [some 0, some 0, some 0, some 0, some 0, some 0, some 0, some 0])
  expect "12 is a multiple of 4" (decode id (eval_std_models.is_multiple 12#usize 4#usize)) (.ok true)
  expect "12 is not a multiple of 5" (decode id (eval_std_models.is_multiple 12#usize 5#usize)) (.ok false)
  expect "0 is a multiple of 0" (decode id (eval_std_models.is_multiple 0#usize 0#usize)) (.ok true)
  expect "3 is not a multiple of 0" (decode id (eval_std_models.is_multiple 3#usize 0#usize)) (.ok false)
  expect "0 is a multiple of 7" (decode id (eval_std_models.is_multiple 0#usize 7#usize)) (.ok true)
  expect "2..=5 contains 5"
    (decode id (eval_std_models.range_contains 2#usize 5#usize 5#usize false)) (.ok true)
  expect "2..=5 contains 2"
    (decode id (eval_std_models.range_contains 2#usize 5#usize 2#usize false)) (.ok true)
  expect "2..=5 does not contain 6"
    (decode id (eval_std_models.range_contains 2#usize 5#usize 6#usize false)) (.ok false)
  expect "2..=5 does not contain 1"
    (decode id (eval_std_models.range_contains 2#usize 5#usize 1#usize false)) (.ok false)
  expect "5..=2 is empty"
    (decode id (eval_std_models.range_contains 5#usize 2#usize 3#usize false)) (.ok false)
  expect "exhausted 2..=5 does not contain 5"
    (decode id (eval_std_models.range_contains 2#usize 5#usize 5#usize true)) (.ok false)
  expect "exhausted 2..=5 does not contain 3"
    (decode id (eval_std_models.range_contains 2#usize 5#usize 3#usize true)) (.ok false)
  expect "exhausted 4..=4 does not contain 4"
    (decode id (eval_std_models.range_contains 4#usize 4#usize 4#usize true)) (.ok false)
  expect "split_first [3, 4, 5]"
    (decode (Option.map fun ((a, n) : U32 × Usize) => (a.val, n.val))
      (eval_std_models.first_and_rest ⟨[3#u32, 4#u32, 5#u32], by scalar_tac⟩))
    (.ok (some (3, 2)))
  expect "split_first [9]"
    (decode (Option.map fun ((a, n) : U32 × Usize) => (a.val, n.val))
      (eval_std_models.first_and_rest ⟨[9#u32], by scalar_tac⟩))
    (.ok (some (9, 0)))
  expect "split_first []"
    (decode (Option.map fun ((a, n) : U32 × Usize) => (a.val, n.val))
      (eval_std_models.first_and_rest ⟨[], by scalar_tac⟩))
    (.ok none)
  expect "append [1, 2] [3]"
    (decode (fun ((v, n) : alloc.vec.Vec U32 × Usize) => (naturals v, n.val))
      (eval_std_models.append ⟨[1#u32, 2#u32], by scalar_tac⟩ ⟨[3#u32], by scalar_tac⟩))
    (.ok ([1, 2, 3], 0))
  expect "append [] []"
    (decode (fun ((v, n) : alloc.vec.Vec U32 × Usize) => (naturals v, n.val))
      (eval_std_models.append ⟨[], by scalar_tac⟩ ⟨[], by scalar_tac⟩))
    (.ok ([], 0))
  expect "empty set" (decode id (eval_std_models.set_is_empty ⟨[], by scalar_tac⟩)) (.ok true)
  expect "non-empty set"
    (decode id (eval_std_models.set_is_empty ⟨[1#u32, 1#u32], by scalar_tac⟩)) (.ok false)
  expect "&a * b, a - &b, x += &y"
    (decode (fun ((p, d, s) : Usize × Usize × U64) => (p.val, d.val, s.val))
      (eval_std_models.ref_arith 6#usize 4#usize 0x3FF8000000000000#u64 0x4002000000000000#u64))
    (.ok (24, 2, 0x400E000000000000))
  expect "x += &y rounds a tie to even"
    (decode (fun ((p, d, s) : Usize × Usize × U64) => (p.val, d.val, s.val))
      (eval_std_models.ref_arith 6#usize 4#usize 0x3ff0000000000000#u64 0x3ca0000000000000#u64))
    (.ok (24, 2, 0x3ff0000000000000))
  expect "&a * b panics on overflow"
    (decode (fun ((p, _, _) : Usize × Usize × U64) => p.val)
      (eval_std_models.ref_arith core.num.Usize.MAX 2#usize 0#u64 0#u64))
    (.error "failure")
  expect "a - &b panics on underflow"
    (decode (fun ((p, _, _) : Usize × Usize × U64) => p.val)
      (eval_std_models.ref_arith 1#usize 2#usize 0#u64 0#u64))
    (.error "failure")
  expect "Borrow<T> for &T" (decode (fun (x : U32) => x.val) (eval_std_models.borrow_ref 42#u32)) (.ok 42)
  expect "recip 4.0f32, 0.5f64"
    (decode (fun ((a, b) : U32 × U64) => (a.val, b.val))
      (eval_std_models.recips 0x40800000#u32 0x3FE0000000000000#u64))
    (.ok (0x3E800000, 0x4000000000000000))
  expect "recip of signed zeros"
    (decode (fun ((a, b) : U32 × U64) => (a.val, b.val))
      (eval_std_models.recips 0#u32 0x8000000000000000#u64))
    (.ok (0x7F800000, 0xFFF0000000000000))
  expect "recip 3.0 rounds to nearest"
    (decode (fun ((a, b) : U32 × U64) => (a.val, b.val))
      (eval_std_models.recips 0x40400000#u32 0x4008000000000000#u64))
    (.ok (0x3eaaaaab, 0x3fd5555555555555))
  expect "recip of infinity is zero"
    (decode (fun ((a, _) : U32 × U64) => a.val)
      (eval_std_models.recips 0x7F800000#u32 0x7ff8000000000000#u64))
    (.ok 0)
  expect "try_from 3 == try_from 3, Ok(1) == Ok(1)"
    (decode id (eval_std_models.results_eq 3#i64 3#i64 (.Ok 1#u8) (.Ok 1#u8))) (.ok (true, true))
  expect "two TryFromIntErrors are equal, Ok(1) != Err(1)"
    (decode id (eval_std_models.results_eq 300#i64 (-200)#i64 (.Ok 1#u8) (.Err 1#u32)))
    (.ok (true, false))
  expect "Ok(3) != Err(_), Err(2) != Err(3)"
    (decode id (eval_std_models.results_eq 3#i64 300#i64 (.Err 2#u32) (.Err 3#u32)))
    (.ok (false, false))
  expect "Ok(3) != Ok(4), Err(7) == Err(7)"
    (decode id (eval_std_models.results_eq 3#i64 4#i64 (.Err 7#u32) (.Err 7#u32)))
    (.ok (false, true))
  expect "equal quadruples"
    (decode id (eval_std_models.quads_eq (1#u8, true, 2#u32, (-3)#i64) (1#u8, true, 2#u32, (-3)#i64)))
    (.ok true)
  expect "quadruples differing in the second component"
    (decode id (eval_std_models.quads_eq (1#u8, true, 2#u32, (-3)#i64) (1#u8, false, 2#u32, (-3)#i64)))
    (.ok false)
  expect "quadruples differing in the last component"
    (decode id (eval_std_models.quads_eq (1#u8, true, 2#u32, (-3)#i64) (1#u8, true, 2#u32, 3#i64)))
    (.ok false)
  expect "Option Eq Some" (decode (Option.map fun (x : U32) => x.val) (eval_std_models.option_total_eq (some 4#u32)))
    (.ok (some 4))
  expect "Option Eq None" (decode (Option.map fun (x : U32) => x.val) (eval_std_models.option_total_eq none))
    (.ok none)
