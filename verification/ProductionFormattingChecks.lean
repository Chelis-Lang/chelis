import ProductionFormatting
import Aeneas.Std.Core.FmtWithSlice
import Aeneas.Std.FloatFmt
import Aeneas.Std.Vec

open Aeneas Aeneas.Std
open Aeneas.Std.core.fmt

namespace ProductionFormattingChecks

macro "largeLiteralBytes%" : term => do
  let even ← `(term| 17#u8)
  let odd ← `(term| 93#u8)
  let values := (List.range 512).toArray.map fun i => if i % 2 == 0 then even else odd
  `(_root_.Aeneas.Std.Array.make 512#usize [$[$values],*])

-- Large byte templates also occur inside deeply nested generated definitions.
set_option maxRecDepth 128 in
private def largeLiteralBytes : Std.Array U8 512#usize := largeLiteralBytes%

#eval show IO Unit from do
  unless largeLiteralBytes.val == (List.replicate 256 [17#u8, 93#u8]).flatten do
    throw (IO.userError "large literal template changed byte contents")

private def bytes (ns : List Nat) : List U8 := ns.map (fun n => ⟨BitVec.ofNat 8 n⟩)
private def utf8 (s : String) : List U8 := s.toByteArray.data.toList.map (fun b => ⟨b.toBitVec⟩)
private def literal (s : String) : List U8 :=
  bytes [s.utf8ByteSize] ++ utf8 s
private def le (size n : Nat) : List U8 :=
  bytes ((List.range size).map (fun i => n / 256 ^ i % 256))

private def rendered (r : Outcome) : Result String := do
  let (status, fmt) ← r
  match status with
  | .Ok () => .ok fmt.output
  | .Err _ => .fail .panic

private def check (label expected : String) (r : Result String) : IO Unit :=
  match r with
  | .ok actual => unless actual == expected do
      throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")
  | _ => throw (IO.userError (label ++ ": unexpected failure"))

private def checkFailure (label : String) (r : Result String) : IO Unit :=
  match r with
  | .fail .panic => pure ()
  | _ => throw (IO.userError (label ++ ": expected panic"))

private def checkWriterOutput (label expected : String) (failed : Bool) (r : Outcome) : IO Unit :=
  match r with
  | .ok (status, fmt) =>
    let actualFailed := match status with | .Err _ => true | .Ok () => false
    unless actualFailed == failed && fmt.output == expected do
      throw (IO.userError s!"{label}: expected ({failed}, {repr expected}), got ({actualFailed}, {repr fmt.output})")
  | _ => throw (IO.userError (label ++ ": unexpected extraction failure"))

private def boundedWriter (limit : Nat) (options : Options := {}) : Formatter := {
  options
  write := fun output text =>
    if output.utf8ByteSize + text.utf8ByteSize > limit then .ok (.Err (), output)
    else .ok (.Ok (), output ++ text)
}

private def checkWriterPrefixes (label : String) (expected : List String)
    (options : Options) (render : Callback) : IO Unit := do
  for (output, limit) in expected.zipIdx do
    checkWriterOutput s!"{label}, capacity {limit}" output (decide (limit + 1 < expected.length))
      (render (boundedWriter limit options))

private def template (code : List U8) (args : List rt.Argument) : Result String :=
  alloc.fmt.format (.template code args)

-- Exact LLBC bytes from production function 27: length 34, UTF-8 prefix,
-- default placeholder C0, length 1, ')', terminator. Literal zero bytes must
-- not be confused with the final opcode.
private def productionFailure : Result String := do
  let arg ← rt.Argument.new_display DisplayStr (toStr "bad shape")
  let code : Std.Array U8 39#usize := ⟨bytes [34] ++ utf8 "failed to construct backward DAG (" ++ bytes [192, 1, 41, 0], by decide⟩
  let args : Std.Array rt.Argument 1#usize := ⟨[arg], by scalar_tac⟩
  let arguments ← Arguments.new code args
  alloc.fmt.format arguments

private def productionNode : Result String := do
  let op ← rt.Argument.new_display DisplayStr (toStr "Argmax")
  let node ← rt.Argument.new_display DisplayUsize 17#usize
  template (literal "grad: no reverse-mode adjoint is defined for `" ++ bytes [192] ++
    literal "` (node " ++ bytes [192] ++ literal ")" ++ bytes [0]) [op, node]

private def dynamicOptions : Result String := do
  let value ← rt.Argument.new_display DisplayI32 ((-3)#i32)
  let text ← rt.Argument.new_display DisplayStr (toStr "é猫x")
  let width ← rt.Argument.from_usize 5#usize
  let precision ← rt.Argument.from_usize 2#usize
  -- Explicit flags + width; indirect width/precision; finally an explicitly
  -- selected earlier argument. This exercises all six opcode flag bits.
  let first := bytes [195] ++ le 4 (0x60000020 + 2^21 + 2^24 + 2^27) ++ le 2 8
  let second := bytes [247] ++ le 4 (0x20000020 + 2^27 + 2^28) ++ le 2 2 ++ le 2 3
  let third := bytes [219] ++ le 4 (0x60000020 + 2^27) ++ le 2 2 ++ le 2 0
  template (first ++ literal "|" ++ second ++ literal "|" ++ third ++ bytes [0])
    [value, text, width, precision]

private def frame (pretty : Bool) : Result String :=
  let pair := DebugPair DebugI32 DebugBool
  let option := DebugOption pair
  let values : alloc.vec.Vec (Option (I32 × Bool)) := ⟨[some ((-3)#i32, true), none], by scalar_tac⟩
  let fmt : Formatter := { options := { flags := 0x60000020 + if pretty then 2^23 else 0 } }
  rendered (fmt.debug_struct_field2_finish (toStr "Frame")
    (toStr "name") ⟨String, DebugString, "a\nb"⟩
    (toStr "values") ⟨alloc.vec.Vec (Option (I32 × Bool)), DebugVec option, values⟩)

private def fails : rt.Argument := .placeholder (fun fmt =>
  thenFormat (fmt.append "partial") (fun fmt => .ok (.Err (), fmt)))

#eval do
  check "production failure" "failed to construct backward DAG (bad shape)" productionFailure
  check "production node" "grad: no reverse-mode adjoint is defined for `Argmax` (node 17)" productionNode
  check "extracted production failure" "failed to construct backward DAG (bad shape)"
    (production_formatting.production_failure (toStr "bad shape"))
  check "extracted production node" "grad: no reverse-mode adjoint is defined for `Argmax` (node 17)"
    (production_formatting.production_node (toStr "Argmax") 17#usize)
  -- Extraction emits the same 175-byte Array.make that exceeded simp's default
  -- recursion depth in production; importing it checks the exact array length.
  check "extracted long production literal"
    "finalize_scalar: f8e4m3 is not in the active dtype set (spec/04-type-system.md section 1.1.1); the checker rejects it, so no numeric value can reach finalize at it (op Add)"
    (production_formatting.production_long_literal (toStr "Add"))
  check "extracted formatting options" "-0000003|   é猫|   -3"
    (production_formatting.flags ((-3)#i32) (toStr "é猫x") 5#usize 2#usize)
  check "dynamic options" "-0000003|   é猫|   -3" dynamicOptions
  check "escaped Unicode" "\"a\\n\\t\\0\\\"\\\\é\\u{301}\""
    (rendered (DebugString.fmt "a\n\t\x00\"\\é\u0301" {}))
  check "nested debug" "Frame { name: \"a\\nb\", values: [Some((-3, true)), None] }" (frame false)
  check "pretty debug"
    "Frame {\n    name: \"a\\nb\",\n    values: [\n        Some(\n            (\n                -3,\n                true,\n            ),\n        ),\n        None,\n    ],\n}" (frame true)
  check "extracted nested derived debug"
    "Frame { name: \"a\\nb\", values: [Some((-3, true)), None] }"
    (production_formatting.nested false)
  check "extracted recursive boxed Debug"
    "Pair(Child(Leaf(2)), Pair(Leaf(3), Leaf(5)))"
    (production_formatting.recursive_formatting false)
  check "extracted recursive boxed Display" "([2] + (3 + 5))"
    (production_formatting.recursive_formatting true)
  check "literal NUL" "a\x00b"
    (template (bytes [3, 97, 0, 98, 0]) [])
  check "long UTF8 literal" (repeatChar 'é' 80)
    (template (bytes [128, 160, 0] ++ utf8 (repeatChar 'é' 80) ++ bytes [0]) [])
  check "from_str" "literal" (Arguments.from_str (toStr "literal") >>= alloc.fmt.format)
  check "TryFromIntError" "TryFromIntError(())" (rendered (DebugTryFromIntError.fmt () {}))
  check "signed debug hex" "ffffffff" (rendered (DebugI32.fmt ((-1)#i32) { options := { flags := 0x60000020 + 2^25 } }))
  check "alternate padded hex" "0x002a" (rendered (num.LowerHexU16.fmt 42#u16
    { options := { flags := 0x60000020 + 2^23 + 2^24 + 2^27, width := 6 } }))

#eval do
  let code := literal "prefix " ++ bytes [192] ++ literal " suffix" ++ bytes [0]
  let result := Formatter.write_fmt {} (.template code [fails])
  match result with
  | .ok (.Err (), fmt) => unless fmt.output == "prefix partial" do
      throw (IO.userError "callback failure lost partial output or wrote suffix")
  | _ => throw (IO.userError "callback error was swallowed")
  checkFailure "format panics on callback error" (alloc.fmt.format (.template code [fails]))
  let writer : Formatter := {
    write := fun output text =>
      if output.utf8ByteSize + text.utf8ByteSize > 7 then .ok (.Err (), output)
      else .ok (.Ok (), output ++ text)
  }
  let value := rt.Argument.placeholder (DisplayU32.fmt 42#u32)
  match writer.write_fmt (.template code [value]) with
  | .ok (.Err (), fmt) => unless fmt.output == "prefix " do
      throw (IO.userError "writer failure did not stop formatting")
  | _ => throw (IO.userError "writer error was swallowed")
  let panicArg := rt.Argument.placeholder (fun _ => .fail .panic)
  checkFailure "extraction failure propagates" (template (bytes [192, 0]) [panicArg])
  checkFailure "out-of-range dynamic count" (do
    let _ ← rt.Argument.from_usize 65536#usize
    .ok "unexpected")
  let diverges := rt.Argument.placeholder (fun _ => .div)
  match Formatter.write_fmt {} (.template code [diverges]) with
  | .div => pure ()
  | _ => throw (IO.userError "callback divergence was swallowed")
  let debugFailure : Debug Unit := ⟨fun _ fmt =>
    thenFormat (fmt.append "partial") (fun fmt => .ok (.Err (), fmt))⟩
  let debugLater : Debug Unit := ⟨fun _ _ => .fail .panic⟩
  match Formatter.debug_tuple_field2_finish {} (toStr "Pair")
      ⟨Unit, DebugBox debugFailure, ()⟩ ⟨Unit, DebugShared (DebugBox debugLater), ()⟩ with
  | .ok (.Err (), fmt) => unless fmt.output == "Pair(partial" do
      throw (IO.userError "recursive debug callback error lost partial output")
  | _ => throw (IO.userError "recursive debug callback error did not short circuit")
  let debugDiverges : Debug Unit := ⟨fun _ _ => .div⟩
  match Formatter.debug_tuple_field1_finish {} (toStr "Child")
      ⟨Unit, DebugBox debugDiverges, ()⟩ with
  | .div => pure ()
  | _ => throw (IO.userError "recursive debug callback divergence was swallowed")

#eval do
  checkWriterPrefixes "right string padding" ["", " ", "  ", "   ", "   x"]
    { flags := 0x20000020 + 2^27, width := 4 } (DisplayStr.fmt (toStr "x"))
  checkWriterPrefixes "left string padding" ["", "", "ab", "ab ", "ab  "]
    { flags := 0x20 + 2^27, width := 4 } (DisplayStr.fmt (toStr "ab"))
  checkWriterPrefixes "Unicode center padding"
    ["", "", "", "猫", "猫x", "猫x", "猫x", "猫x猫", "猫x猫", "猫x猫", "猫x猫猫"]
    { flags := '猫'.toNat + 2^30 + 2^27, width := 4 } (DisplayStr.fmt (toStr "x"))
  checkWriterPrefixes "precision keeps Unicode content atomic"
    ["", " ", "  ", "  ", "  ", "  ", "  ", "  é猫"]
    { flags := 0x20000020 + 2^27 + 2^28, width := 4, precision := 2 }
    (DisplayStr.fmt (toStr "é猫x"))
  checkWriterPrefixes "numeric default right alignment" ["", " ", "  ", "  -", "  -", "  -42"]
    { flags := 0x60000020 + 2^27, width := 5 } (DisplayI32.fmt ((-42)#i32))
  checkWriterPrefixes "numeric center padding"
    ["", " ", "  ", "  -", "  -", "  -42", "  -42 ", "  -42  "]
    { flags := 0x40000020 + 2^27, width := 7 } (DisplayI32.fmt ((-42)#i32))
  checkWriterPrefixes "hex sign and prefix after padding"
    ["", " ", "  ", "   ", "   +", "   +", "   +0x", "   +0x", "   +0x2a"]
    { flags := 0x60000020 + 2^21 + 2^23 + 2^27, width := 8 }
    (LowerHexU16.fmt 42#u16)
  checkWriterPrefixes "zero fill overrides Unicode left alignment"
    ["", "+", "+", "+0x", "+0x0", "+0x00", "+0x000", "+0x000", "+0x0002a"]
    { flags := '猫'.toNat + 2^21 + 2^23 + 2^24 + 2^27, width := 8 }
    (LowerHexU16.fmt 42#u16)
  checkWriterPrefixes "sign and prefix without padding" ["", "+", "+", "+0x", "+0x", "+0x2a"]
    { flags := 0x60000020 + 2^21 + 2^23 + 2^24 + 2^27, width := 2 }
    (LowerHexU16.fmt 42#u16)
  let value := BinaryFloat.round .binary64 true 25 2
  checkWriterPrefixes "floating sign, zeros, and decimal parts"
    ["", "-", "-0", "-00", "-000", "-000", "-00012", "-00012.", "-00012.5"]
    { flags := 0x60000020 + 2^24 + 2^27 + 2^28, width := 8, precision := 1 }
    (DisplayF64.fmt value)
  checkWriterPrefixes "floating center padding"
    ["", " ", " -", " -", " -12", " -12.", " -12.5", " -12.5 ", " -12.5  "]
    { flags := 0x40000020 + 2^27 + 2^28, width := 8, precision := 1 }
    (DisplayF64.fmt value)
  checkWriterPrefixes "floating exponent parts" ["", "5", "5", "5e-", "5e-", "5e-", "5e-324"]
    {} (DebugF64.fmt (BinaryFloat.ofNat .binary64 1))
  for (limit, zeros) in ([(65, 0), (66, 64), (71, 64), (72, 70)] : List (Nat × Nat)) do
    checkWriterOutput s!"floating zero block, capacity {limit}" ("0." ++ repeatChar '0' zeros)
      (decide (limit < 72)) (DisplayF64.fmt (BinaryFloat.ofNat .binary64 0)
        (boundedWriter limit { flags := 0x60000020 + 2^28, precision := 70 }))
  checkWriterOutput "floating exact Copy buffer before zero extension" ("1." ++ repeatChar '0' 60)
    true (DisplayF64.fmt (BinaryFloat.ofNat .binary64 0x3ff0000000000000)
      (boundedWriter 70 { flags := 0x60000020 + 2^28, precision := 80 }))
  let rejectsEmpty : Formatter := {
    options := { flags := 0x60000020 + 2^21 + 2^23 }
    write := fun output text =>
      if text.isEmpty then .ok (.Err (), output) else .ok (.Ok (), output ++ text)
  }
  checkWriterOutput "alternate decimal empty prefix" "+" true (DisplayI32.fmt 42#i32 rejectsEmpty)
  checkWriterOutput "floating alternate has no integral prefix" "+1.5" false
    (DisplayF64.fmt (BinaryFloat.round .binary64 false 3 2) rejectsEmpty)
  checkWriterOutput "zero-padded float writes its empty sign" "" true
    (DisplayF64.fmt (BinaryFloat.round .binary64 false 3 2)
      { rejectsEmpty with options := { flags := 0x60000020 + 2^24 + 2^27, width := 4 } })
  let partialPrefix : Formatter := {
    options := { flags := 0x60000020 + 2^21 + 2^23 + 2^24 + 2^27, width := 8 }
    write := fun output text =>
      if output.endsWith "!" then .fail .panic
      else if text == "0x" then .ok (.Err (), output ++ "0!")
      else .ok (.Ok (), output ++ text)
  }
  checkWriterOutput "partially failing prefix stops number and suffix" "+0!" true
    (thenFormat (LowerHexU16.fmt 42#u16 partialPrefix) (fun fmt => fmt.append " suffix"))

#eval do
  check "f32 shortest" "0.1" (rendered (DebugF32.fmt (BinaryFloat.ofNat .binary32 0x3dcccccd) {}))
  check "negative zero debug" "-0.0" (rendered (DebugF64.fmt (BinaryFloat.ofNat .binary64 0x8000000000000000) {}))
  check "negative zero display" "-0" (rendered (DisplayF64.fmt (BinaryFloat.ofNat .binary64 0x8000000000000000) {}))
  check "smallest subnormal" "5e-324" (rendered (DebugF64.fmt (BinaryFloat.ofNat .binary64 1) {}))
  check "largest finite" "1.7976931348623157e308" (rendered (DebugF64.fmt (BinaryFloat.ofNat .binary64 0x7fefffffffffffff) {}))
  let exact : Formatter := { options := { flags := 0x60000020 + 2^28 } }
  check "precision ties even down" "2" (rendered (DisplayF64.fmt (BinaryFloat.round .binary64 false 5 2) exact))
  check "precision ties even up" "4" (rendered (DisplayF64.fmt (BinaryFloat.round .binary64 false 7 2) exact))
  check "NaN ignores plus" "NaN" (rendered (DisplayF64.fmt (BinaryFloat.nan .binary64)
    { options := { flags := 0x60000020 + 2^21 } }))
  check "negative infinity" "-inf" (rendered (DebugF64.fmt (BinaryFloat.infinity .binary64 true) {}))
  check "half delegates to f32" "0.33325195" (rendered (DebugF16.fmt (BinaryFloat.ofNat .binary16 0x3555) {}))
  check "extracted floating shortest roundtrip" "5e-324"
    (production_formatting.debug_float64 1#u64)
  check "extracted floating ties-even rounding" "2"
    (production_formatting.precision_float64 0x4004000000000000#u64 0#usize)
  check "bfloat delegates to f32" "0.33398438" (rendered (DebugBF16.fmt (BinaryFloat.ofNat .bfloat16 0x3eab) {}))

end ProductionFormattingChecks
