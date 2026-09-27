import StrPatternModels

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def txt (s : Str) : String := match Str.toString s with | .ok v => v | _ => "<invalid>"

/-- `IntErrorKind` as its Rust variant name. -/
def kindName : core.num.error.IntErrorKind → String
  | .Empty => "Empty" | .InvalidDigit => "InvalidDigit" | .PosOverflow => "PosOverflow"
  | .NegOverflow => "NegOverflow" | .Zero => "Zero"

def res {T : Type} (f : T → Int) : core.result.Result T core.num.error.IntErrorKind → Except String Int
  | .Ok v => .ok (f v)
  | .Err k => .error (kindName k)

open str_pattern_models

#eval do
  let range := fun (o : Option (Usize × Usize)) => o.map fun ((a, b) : Usize × Usize) => (a.val, b.val)
  expect "surface_range" (decode range (surface_range (toStr "surf:3..17"))) (.ok (some (3, 17)))
  expect "surface_range, bad end" (decode range (surface_range (toStr "surf:3..x"))) (.ok none)
  expect "surface_range, bad prefix" (decode range (surface_range (toStr "span:3..17"))) (.ok none)
  expect "surface_range, no separator" (decode range (surface_range (toStr "surf:3-17"))) (.ok none)
  expect "surface_range, empty start" (decode range (surface_range (toStr "surf:..17"))) (.ok none)
  let opt := fun (o : Option Str) => o.map txt
  expect "option_payload" (decode opt (option_payload (toStr "Option[u32]"))) (.ok (some "u32"))
  expect "option_payload, no ]" (decode opt (option_payload (toStr "Option[u32"))) (.ok none)
  expect "option_payload, other" (decode opt (option_payload (toStr "Vec[u32]"))) (.ok none)
  let axis := fun (o : Option U64) => o.map fun (x : U64) => x.val
  expect "axis_index d12" (decode axis (axis_index (toStr "d12"))) (.ok (some 12))
  expect "axis_index d" (decode axis (axis_index (toStr "d"))) (.ok none)
  expect "axis_index x1" (decode axis (axis_index (toStr "x1"))) (.ok none)
  expect "axis_index overflow" (decode axis (axis_index (toStr "d18446744073709551616"))) (.ok none)
  expect "hex_digits" (decode txt (hex_digits (toStr "-0x0x1f"))) (.ok "1f")
  expect "hex_digits 0X" (decode txt (hex_digits (toStr "--0X2a"))) (.ok "2a")
  expect "hex_digits 0b" (decode txt (hex_digits (toStr "0b1"))) (.ok "0b1")
  let sig := fun ((a, b) : Str × Str) => (txt a, txt b)
  expect "significant" (decode sig (significant (toStr "0012500"))) (.ok ("12500", "125"))
  expect "significant, zeros" (decode sig (significant (toStr "000"))) (.ok ("", ""))
  expect "significant, non-ASCII" (decode sig (significant (toStr "é0"))) (.ok ("é0", "é"))
  let expo := fun (o : Option (Str × core.result.Result I32 core.num.error.IntErrorKind)) =>
    o.map fun ((m, e) : Str × core.result.Result I32 core.num.error.IntErrorKind) =>
      (txt m, res (fun (x : I32) => x.val) e)
  expect "exponent 1.5e-3" (decode expo (exponent (toStr "1.5e-3"))) (.ok (some ("1.5", .ok (-3))))
  expect "exponent 2e+4" (decode expo (exponent (toStr "2e+4"))) (.ok (some ("2", .ok 4)))
  expect "exponent 2e" (decode expo (exponent (toStr "2e"))) (.ok (some ("2", .error "Empty")))
  expect "exponent 2e-" (decode expo (exponent (toStr "2e-"))) (.ok (some ("2", .error "InvalidDigit")))
  expect "exponent 2e99999999999" (decode expo (exponent (toStr "2e99999999999")))
    (.ok (some ("2", .error "PosOverflow")))
  expect "exponent 2e-99999999999" (decode expo (exponent (toStr "2e-99999999999")))
    (.ok (some ("2", .error "NegOverflow")))
  expect "exponent 25" (decode expo (exponent (toStr "25"))) (.ok none)
  expect "normalized" (decode id (normalized (toStr "  a::b-c \n"))) (.ok ("a__b-c", "a::b_c"))
  expect "normalized, Unicode whitespace" (decode id (normalized (toStr "\u3000x\u00a0")))
    (.ok ("x", "x"))
  expect "normalized, all whitespace" (decode id (normalized (toStr " \t "))) (.ok ("", ""))
  let bytes := fun ((b, n) : Option U8 × Usize) => (b.map (fun (x : U8) => x.val), n.val)
  expect "bytes_of" (decode bytes (bytes_of (toStr "é!"))) (.ok (some 0xc3, 3))
  expect "bytes_of, empty" (decode bytes (bytes_of (toStr ""))) (.ok (none, 0))
  let kinds := fun ((a, b, c) : core.result.Result Usize core.num.error.IntErrorKind ×
      core.result.Result U64 core.num.error.IntErrorKind ×
      core.result.Result I64 core.num.error.IntErrorKind) =>
    (res (fun (x : Usize) => x.val) a, res (fun (x : U64) => x.val) b, res (fun (x : I64) => x.val) c)
  expect "parse 42" (decode kinds (parse_kinds (toStr "42"))) (.ok (.ok 42, .ok 42, .ok 42))
  expect "parse +7" (decode kinds (parse_kinds (toStr "+7"))) (.ok (.ok 7, .ok 7, .ok 7))
  expect "parse -7" (decode kinds (parse_kinds (toStr "-7")))
    (.ok (.error "InvalidDigit", .error "InvalidDigit", .ok (-7)))
  expect "parse empty" (decode kinds (parse_kinds (toStr "")))
    (.ok (.error "Empty", .error "Empty", .error "Empty"))
  expect "parse -" (decode kinds (parse_kinds (toStr "-")))
    (.ok (.error "InvalidDigit", .error "InvalidDigit", .error "InvalidDigit"))
  expect "parse 2^64" (decode kinds (parse_kinds (toStr "18446744073709551616")))
    (.ok (.error "PosOverflow", .error "PosOverflow", .error "PosOverflow"))
  expect "parse 2^63" (decode kinds (parse_kinds (toStr "9223372036854775808")))
    (.ok (.ok 9223372036854775808, .ok 9223372036854775808, .error "PosOverflow"))
  expect "parse -2^63-1" (decode kinds (parse_kinds (toStr "-9223372036854775809")))
    (.ok (.error "InvalidDigit", .error "InvalidDigit", .error "NegOverflow"))
  expect "parse 1 (space)" (decode kinds (parse_kinds (toStr "1 ")))
    (.ok (.error "InvalidDigit", .error "InvalidDigit", .error "InvalidDigit"))
  let sci := fun ((d, e) : String × I32) => (d, e.val)
  expect "split_sci -1.25e-7" (decode sci (split_sci (toStr "-1.25e-7"))) (.ok ("125", -7))
  expect "split_sci 3e12" (decode sci (split_sci (toStr "3e12"))) (.ok ("3", 12))
  expect "split_sci, bad exponent panics" (decode sci (split_sci (toStr "1e+"))) (.error "failure")
  expect "split_sci, no exponent panics" (decode sci (split_sci (toStr "1.5"))) (.error "failure")
  expect "to_snake_case" (decode id (to_snake_case (toStr "TensorShapeÉ"))) (.ok "tensor_shapeÉ")
  expect "to_snake_case aB" (decode id (to_snake_case (toStr "aB"))) (.ok "a_b")
  expect "to_snake_case empty" (decode id (to_snake_case (toStr ""))) (.ok "")
  expect "split_at_digit 2" (decode id (split_at_digit (toStr "12345") 2#usize)) (.ok ("12", "345"))
  expect "split_at_digit 5" (decode id (split_at_digit (toStr "12345") 5#usize)) (.ok ("12345", ""))
  expect "split_at_digit after é" (decode id (split_at_digit (toStr "é1") 2#usize)) (.ok ("é", "1"))
  expect "split_at_digit inside é panics" (decode id (split_at_digit (toStr "é1") 1#usize))
    (.error "failure")
  expect "split_at_digit past the end panics" (decode id (split_at_digit (toStr "12") 3#usize))
    (.error "failure")
  let bits := fun {T : Type} (f : T → Nat) (r : core.result.Result T String) =>
    match r with | .Ok x => Except.ok (f x) | .Err e => .error e
  let fbits := fun ((a, b) : core.result.Result U64 String × core.result.Result U32 String) =>
    (bits (fun (x : U64) => x.val) a, bits (fun (x : U32) => x.val) b)
  expect "float_bits 0.1" (decode fbits (float_bits (toStr "0.1")))
    (.ok (.ok 0x3fb999999999999a, .ok 0x3dcccccd))
  expect "float_bits -0" (decode fbits (float_bits (toStr "-0")))
    (.ok (.ok 0x8000000000000000, .ok 0x80000000))
  expect "float_bits 1e308" (decode fbits (float_bits (toStr "1e308")))
    (.ok (.ok 0x7fe1ccf385ebc8a0, .ok 0x7f800000))
  expect "float_bits 5e-324" (decode fbits (float_bits (toStr "5e-324"))) (.ok (.ok 1, .ok 0))
  expect "float_bits 3e-324" (decode fbits (float_bits (toStr "3e-324"))) (.ok (.ok 1, .ok 0))
  expect "float_bits -Infinity" (decode fbits (float_bits (toStr "-Infinity")))
    (.ok (.ok 0xfff0000000000000, .ok 0xff800000))
  expect "float_bits nan" (decode fbits (float_bits (toStr "nan")))
    (.ok (.ok 0x7ff8000000000000, .ok 0x7fc00000))
  expect "float_bits +.5e-3" (decode fbits (float_bits (toStr "+.5e-3")))
    (.ok (.ok 0x3f40624dd2f1a9fc, .ok 0x3a03126f))
  expect "float_bits 1." (decode fbits (float_bits (toStr "1.")))
    (.ok (.ok 0x3ff0000000000000, .ok 0x3f800000))
  expect "float_bits, 30 digits" (decode fbits (float_bits (toStr "123456789012345678901234567890")))
    (.ok (.ok 0x45f8ee90ff6c373e, .ok 0x6fc77488))
  let invalid : Except String (Except String Nat × Except String Nat) :=
    .ok (.error "f64: invalid float literal", .error "invalid float literal")
  expect "float_bits 1e" (decode fbits (float_bits (toStr "1e"))) invalid
  expect "float_bits ." (decode fbits (float_bits (toStr "."))) invalid
  expect "float_bits 1e5x" (decode fbits (float_bits (toStr "1e5x"))) invalid
  expect "float_bits, empty" (decode fbits (float_bits (toStr "")))
    (.ok (.error "f64: cannot parse float from empty string",
      .error "cannot parse float from empty string"))
  expect "int_error 12" (decode id (int_error (toStr "12"))) (.ok none)
  expect "int_error -1" (decode id (int_error (toStr "-1")))
    (.ok (some "[           invalid digit found in string]"))
  expect "int_error 2^66" (decode id (int_error (toStr "99999999999999999999")))
    (.ok (some "[  number too large to fit in target type]"))
