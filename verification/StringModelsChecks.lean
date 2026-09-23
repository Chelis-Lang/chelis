import StringModels

open Aeneas Aeneas.Std

private def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

private def outcome : Result α → Except String α
  | .ok value => .ok value
  | .fail _ => .error "failure"
  | .div => .error "divergence"

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
  IO.println "String models: 20 behavioral checks passed"
