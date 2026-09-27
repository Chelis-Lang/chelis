import FmtHexCharModels

open Aeneas Aeneas.Std

namespace FmtHexCharModelsChecks

/-! Mirrors the native tests of `fmt_hex_char_models.rs`. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def u64 (n : Nat) : U64 := ⟨BitVec.ofNat _ n⟩

open fmt_hex_char_models

#eval do
  expect "bits32 0x3f80" (decode id (bits32 (u32 0x3f80)))
    (.ok "0x00003f80|3f80|00003f80|0x3f80|  3f80|3f80  |")
  expect "bits32 0" (decode id (bits32 (u32 0))) (.ok "0x00000000|0|00000000|0x0|     0|0     |")
  expect "bits32 max" (decode id (bits32 (u32 (2 ^ 32 - 1))))
    (.ok "0xffffffff|ffffffff|ffffffff|0xffffffff|ffffffff|ffffffff|")
  expect "bits64 nan" (decode id (bits64 (u64 0x7ff8000000000000)))
    (.ok "0x7ff8000000000000|7ff8000000000000|0x7ff8000000000000")
  expect "bits64 10" (decode id (bits64 (u64 10))) (.ok "0x000000000000000a|a|0xa")
  expect "code_point 1f" (decode id (code_point (u32 0x1f))) (.ok "U+001F|0x1F|0x001F")
  expect "code_point max" (decode id (code_point (u32 0x10ffff))) (.ok "U+10FFFF|0x10FFFF|0x10FFFF")
  expect "code_point_ref" (decode id (code_point_ref (u32 0xab))) (.ok "U+00AB")

#eval do
  expect "a" (decode id (character 'a')) (.ok "'a'|a|  a|a  ||'a'")
  expect "quote" (decode id (character '\'')) (.ok "'\\''|'|  '|'  ||'\\''")
  expect "double quote" (decode id (character '"')) (.ok "'\"'|\"|  \"|\"  ||'\"'")
  expect "newline" (decode id (character '\n')) (.ok "'\\n'|\n|  \n|\n  ||'\\n'")
  expect "é" (decode id (character 'é')) (.ok "'é'|é|  é|é  ||'é'")
  expect "combining acute" (decode id (character '\u0301'))
    (.ok "'\\u{301}'|\u0301|  \u0301|\u0301  ||'\\u{301}'")
  expect "delete" (decode id (character '\x7f')) (.ok "'\\u{7f}'|\x7f|  \x7f|\x7f  ||'\\u{7f}'")
  expect "backslash" (decode id (character '\\')) (.ok "'\\\\'|\\|  \\|\\  ||'\\\\'")

#eval do
  expect "carrier" (decode id (carrier (u32 7) true 'x' false)) (.ok "Triple(7, true, 'x') Single(7)")
  expect "carrier pretty" (decode id (carrier (u32 7) false '\t' true))
    (.ok "Triple(\n    7,\n    false,\n    '\\t',\n)")

end FmtHexCharModelsChecks
