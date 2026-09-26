import StringModels

open Aeneas Aeneas.Std

private def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

private def outcome : Result α → Except String α
  | .ok value => .ok value
  | .fail _ => .error "failure"
  | .div => .error "divergence"

private def lineText (line : Option Str) : Result (Option String) :=
  match line with
  | some line => do
    let text ← Str.toString line
    .ok (some text)
  | none => .ok none

private def firstText (s : Str) : Except String (Option String × Option String × Option String) :=
  outcome (do
    let (a, b, c) ← string_models.first_lines s
    let a ← lineText a
    let b ← lineText b
    let c ← lineText c
    .ok (a, b, c))

#eval do
  expect "Extracted literals preserve UTF-8, quotes, backslashes and controls"
    (outcome (do
      let text ← string_models.literal_text
      Str.toString text)) (.ok "§\x00\t\n\r\"\\\x1b\x7f𐀀")
  expect "long production diagnostic preserves exact text and byte length"
    (outcome (do
      let text ← string_models.extent_source_diagnostic
      let decoded ← Str.toString text
      .ok (decoded, text.val.length)))
    (.ok ("give the axis a real extent source: size the consuming operation's output from the operand that declares the extent, or declare the extent on the producing node", 160))
  expect "long Unicode production diagnostic counts UTF-8 bytes, not characters"
    (outcome (do
      let text ← string_models.deferred_dtype_diagnostic
      let decoded ← Str.toString text
      .ok (decoded, text.val.length)))
    (.ok ("reduce_sum: operand dtype `f8e4m3` is deferred per spec/04-type-system.md §1.1.1 and is not part of the active numeric primitive set", 133))
  expect "borrowed string equality preserves the anonymous-name boundary"
    (outcome (do
      let empty ← string_models.is_anonymous (toStr "")
      let wildcard ← string_models.is_anonymous (toStr "*")
      let named ← string_models.is_anonymous (toStr "axis")
      let longer ← string_models.is_anonymous (toStr "*\x00")
      .ok (empty, wildcard, named, longer))) (.ok (true, true, false, false))
  expect "owned and borrowed comparison preserve UTF-8 and embedded NUL"
    (outcome (string_models.compare_owned (toStr "é\x00𐀀") (toStr "é\x00𐀀")))
    (.ok (true, false, true, false))
  expect "comparison does not normalize Unicode"
    (outcome (string_models.compare_owned (toStr "é") (toStr "é")))
    (.ok (false, true, false, true))
  expect "equal byte lengths do not imply equality"
    (outcome (string_models.compare_owned (toStr "axis") (toStr "axes")))
    (.ok (false, true, false, true))
  expect "length mismatch makes both inequality implementations true"
    (outcome (string_models.compare_owned (toStr "a") (toStr "ab")))
    (.ok (false, true, false, true))
  expect "empty owned and borrowed strings compare equal"
    (outcome (string_models.compare_owned (toStr "") (toStr "")))
    (.ok (true, false, true, false))
  expect "UTF-8 ordering crosses encoding widths correctly"
    (outcome (string_models.compare_order (toStr "") (toStr "𐀀"))) (.ok (.lt, .lt))
  expect "a proper prefix sorts first"
    (outcome (string_models.compare_order (toStr "prefix") (toStr "prefix!"))) (.ok (.lt, .lt))
  expect "clone and borrowed views preserve nonempty text"
    (outcome (string_models.cloned_view (toStr "é\x00𐀀"))) (.ok ("é\x00𐀀", false, false))
  expect "clone and both empty checks preserve empty text"
    (outcome (string_models.cloned_view (toStr ""))) (.ok ("", true, true))
  expect "ToOwned and Deref preserve Unicode bytes"
    (outcome (string_models.dereferenced_owned (toStr "λ"))) (.ok "λ")
  expect "String to Box str preserves bytes"
    (outcome (do
      let view ← string_models.boxed_view (toStr "é\x00")
      Str.toString view)) (.ok "é\x00")
  expect "join inserts a UTF-8 separator even after an empty first part"
    (outcome (string_models.joined (toStr "") (toStr "β") (toStr "λ"))) (.ok "λβ")
  expect "empty separator concatenates exact byte sequences"
    (outcome (string_models.joined (toStr "a\x00") (toStr "b") (toStr ""))) (.ok "a\x00b")
  expect "empty join has no separator"
    (outcome (string_models.empty_join (toStr "ignored"))) (.ok "")
  expect "singleton join has no separator"
    (outcome (string_models.singleton_join (toStr "β") (toStr "ignored"))) (.ok "β")
  let malformed : Str := ⟨[0xc0#u8, 0x80#u8], by simp; scalar_tac⟩
  expect "overlong UTF-8 fails undefined rather than introducing replacement text"
    (match Str.toString malformed with | .fail .undef => true | _ => false) true
  expect "no lines in empty text" (firstText (toStr "")) (.ok (none, none, none))
  expect "an unterminated line" (firstText (toStr "a")) (.ok (some "a", none, none))
  expect "a terminated line has no empty successor" (firstText (toStr "a\n")) (.ok (some "a", none, none))
  expect "CRLF ends a line" (firstText (toStr "a\r\nb")) (.ok (some "a", some "b", none))
  expect "empty lines" (firstText (toStr "\n\n")) (.ok (some "", some "", none))
  expect "lone carriage returns stay" (firstText (toStr "a\rb\r")) (.ok (some "a\rb\r", none, none))
  expect "an empty CRLF line" (firstText (toStr "x\r\n\r\n")) (.ok (some "x", some "", none))
  expect "LF CR is not a line ending" (firstText (toStr "a\n\rb\r\n")) (.ok (some "a", some "\rb", none))
  expect "multi-byte text and a final lone CR" (firstText (toStr "é\n\r")) (.ok (some "é", some "\r", none))
  expect "a loop continues after next"
    (outcome (do
      let (a, count, bytes) ← string_models.remaining_lines (toStr "h\nab\ncde\r\n")
      let a ← lineText a
      .ok (a, count.val, bytes.val))) (.ok (some "h", 2, 5))
  expect "a loop over no lines"
    (outcome (do
      let (a, count, bytes) ← string_models.remaining_lines (toStr "")
      let a ← lineText a
      .ok (a, count.val, bytes.val))) (.ok (none, 0, 0))
  expect "collected line lengths"
    (outcome (do
      let v : alloc.vec.Vec Usize ← string_models.line_lengths (toStr "é\n\r\nxyz\r")
      .ok (v.val.map fun (n : Usize) => n.val))) (.ok [2, 0, 4])
  expect "a lone CRLF is one empty line"
    (outcome (do
      let v : alloc.vec.Vec Usize ← string_models.line_lengths (toStr "\r\n")
      .ok (v.val.map fun (n : Usize) => n.val))) (.ok [0])
  IO.println "String models: 33 behavioral checks passed"
