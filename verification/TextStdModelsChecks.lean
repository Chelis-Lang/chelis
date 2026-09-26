import TextStdModels

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def vals (v : alloc.vec.Vec U32) : List Nat := v.val.map (fun (x : U32) => x.val)

def byte (value : Option U8) : Option Nat := value.map (fun (x : U8) => x.val)

#eval do
  expect "flat_map over Some storage, chained with the extensions"
    (decode vals (text_std_models.entries
      { storage := some { values := ⟨[1#u32, 2#u32], by scalar_tac⟩ }, extensions := ⟨[7#u32], by scalar_tac⟩ }))
    (.ok [2, 3, 7])
  expect "flat_map over None storage"
    (decode vals (text_std_models.entries { storage := none, extensions := ⟨[7#u32, 8#u32], by scalar_tac⟩ }))
    (.ok [7, 8])
  expect "Option::iter over Some"
    (decode (fun ((a, b, n) : Option U32 × Option U32 × Usize) =>
        (a.map (fun (x : U32) => x.val), b.map (fun (x : U32) => x.val), n.val))
      (text_std_models.option_iter (some 5#u32)))
    (.ok (some 5, none, 1))
  expect "Option::iter over None"
    (decode (fun ((a, b, n) : Option U32 × Option U32 × Usize) =>
        (a.map (fun (x : U32) => x.val), b.map (fun (x : U32) => x.val), n.val))
      (text_std_models.option_iter none))
    (.ok (none, none, 0))
  let offset := fun (found : Option (Usize × U8)) =>
    found.map (fun ((i, b) : Usize × U8) => (i.val, b.val))
  expect "first control byte: tab"
    (decode offset (text_std_models.forbidden (toStr "ab\tc"))) (.ok (some (2, 9)))
  expect "first control byte: DEL"
    (decode offset (text_std_models.forbidden (toStr "abc\x7f"))) (.ok (some (3, 0x7f)))
  expect "no control byte in UTF-8 text"
    (decode offset (text_std_models.forbidden (toStr "é ok"))) (.ok none)
  let checks := fun ((lower, digit, dot, count) : Bool × Bool × Option U8 × Usize) =>
    (lower, digit, byte dot, count.val)
  expect "snake_case bytes"
    (decode checks (text_std_models.byte_checks (toStr "snake_case")))
    (.ok (true, false, none, 10))
  expect "dotted version bytes"
    (decode checks (text_std_models.byte_checks (toStr "v1.2")))
    (.ok (false, true, some 46, 4))
  expect "empty string bytes"
    (decode checks (text_std_models.byte_checks (toStr ""))) (.ok (true, false, none, 0))
  expect "a two-byte scalar counts two bytes"
    (decode checks (text_std_models.byte_checks (toStr "é"))) (.ok (false, false, none, 2))
  expect "Bytes::next yields UTF-8 bytes"
    (decode (fun ((a, b) : Option U8 × Option U8) => (byte a, byte b))
      (text_std_models.first_two (toStr "é")))
    (.ok (some 0xc3, some 0xa9))
  expect "Bytes::next after the end"
    (decode (fun ((a, b) : Option U8 × Option U8) => (byte a, byte b))
      (text_std_models.first_two (toStr "x")))
    (.ok (some 120, none))
  expect "ASCII lowercase"
    (decode id (text_std_models.lowercase (toStr "Std::IO_x9"))) (.ok "std::io_x9")
  expect "non-ASCII bytes are kept"
    (decode id (text_std_models.lowercase (toStr "ÉCOLE"))) (.ok "École")
  expect "String == &str after lowercase"
    (decode id (text_std_models.same_path (toStr "Core::Mem") (toStr "core::mem"))) (.ok true)
  expect "String != &str of a different length"
    (decode id (text_std_models.same_path (toStr "Core::Mem") (toStr "core::me"))) (.ok false)
  expect "Clone for Box<str> keeps the UTF-8 bytes"
    (decode (fun (copy : text_std_models.Diagnostic) =>
        (copy.code.val.map (fun (b : U8) => b.val), copy.line.val))
      (text_std_models.clone_diagnostic { code := toStr "E0042 é", line := 3#u32 }))
    (.ok ([0x45, 0x30, 0x30, 0x34, 0x32, 0x20, 0xc3, 0xa9], 3))
