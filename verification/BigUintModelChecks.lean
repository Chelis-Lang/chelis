import Aeneas.Std.BigUint

open Aeneas Aeneas.Std num_bigint.BigUint

namespace BigUintModelChecks

/-! Mirrors `crate_model_oracle/src/bin/biguint_model_oracle.rs` line for line,
    evaluating the registered `rust_fun` models. -/

def render : Result Nat → String
  | .ok v => toString v
  | .fail _ => "panic"
  | .div => "divergence"

def ok! {α} [Inhabited α] : Result α → α
  | .ok x => x
  | _ => panic! "unexpected model failure"

def ordStr : Ordering → String
  | .lt => "lt" | .eq => "eq" | .gt => "gt"

def u8 (n : Nat) : U8 := ⟨BitVec.ofNat _ n⟩
def u16 (n : Nat) : U16 := ⟨BitVec.ofNat _ n⟩
def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def u64 (n : Nat) : U64 := ⟨BitVec.ofNat _ n⟩
def usize (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩
def u128 (n : Nat) : U128 := ⟨BitVec.ofNat _ n⟩

instance {ty} : Inhabited (core.result.Result (UScalar ty) (num_bigint.TryFromBigIntError Nat)) :=
  ⟨.Err ⟨0⟩⟩

/-- A hasher that records every byte written and counts the `write` calls (the oracle's `Log`). -/
def logHasher : core.hash.Hasher (List Nat × Nat) :=
  { finish := fun _ => .ok 0#u64
    write := fun (bytes, writes) s => .ok (bytes ++ s.val.map (·.val), writes + 1) }

def tryStr {ty} : core.result.Result (UScalar ty) (num_bigint.TryFromBigIntError Nat) → String
  | .Ok n => s!"ok {n.val}"
  | .Err e => s!"err {e.original}"

/-- Options use the pinned native formatting layout. -/
def opts (fill : Char := ' ') (align : Nat := 3) (plus alt zero lowerHex upperHex : Bool := false)
    (width : Option Nat := none) : core.fmt.Options :=
  let bit (b : Bool) (n : Nat) := if b then 2 ^ n else 0
  { flags := fill.toNat + bit plus 21 + bit alt 23 + bit zero 24 + bit lowerHex 25 +
      bit upperHex 26 + bit width.isSome 27 + align * 2 ^ 29
    width := width.getD 0 }

def run (k : core.fmt.Formatter → core.fmt.Outcome) (o : core.fmt.Options := {}) : String :=
  match k { options := o } with
  | .ok (.Ok (), f) => f.output
  | _ => "error"

def dbg (v : Nat) : core.fmt.DebugValue := ⟨Nat, num_bigint.DebugBigUint, v⟩

/-- The native oracle uses this `Capacity` enum with derived `Debug`. -/
def literal (v : Nat) : core.fmt.Formatter → core.fmt.Outcome := fun f =>
  f.debug_tuple_field1_finish (toStr "Literal") (dbg v)

def quotient (a b : Nat) : core.fmt.Formatter → core.fmt.Outcome := fun f =>
  f.debug_struct_field2_finish (toStr "Quotient") (toStr "dividend") (dbg a) (toStr "divisor") (dbg b)

-- The sign precedes zero padding. Alternate and hex flags do not change decimal output.
#guard run (DebugBigUint.fmt 42) (opts (plus := true) (zero := true) (width := some 6)) == "+00042"

#guard run (DisplayBigUint.fmt 7) (opts (alt := true) (lowerHex := true)) == "7"

#guard run (DisplayBigUint.fmt (2 ^ 64)) (opts (fill := '*') (align := 2) (width := some 23)) ==
  "*18446744073709551616**"

-- Kernel reduction of the models: exact arithmetic and the zero-divisor panic.
example : RemShared.rem (2 ^ 64 * 3) 7 = .ok ((2 ^ 64 * 3) % 7) := rfl
example : DivShared.div 5 0 = .fail .panic := rfl

def main : IO Unit := do
  let p64 := 2 ^ 64
  let values : List Nat := [
    ok! (FromU8.from_ (u8 0)), ok! (FromU8.from_ (u8 1)), ok! (FromU16.from_ (u16 2)),
    ok! (FromU32.from_ (u32 7)), ok! (FromUsize.from_ (usize 12)),
    ok! (FromU32.from_ (u32 (2 ^ 32 - 1))), ok! (FromU64.from_ (u64 (2 ^ 32))),
    ok! (FromU64.from_ (u64 (2 ^ 64 - 1))), p64, ok! (FromU128.from_ (u128 (2 ^ 128 - 1))),
    ok! (MulBigUint.mul p64 (ok! (FromU8.from_ (u8 3)))),
    ok! (MulBigUint.mul (2 ^ 128 - 1) (2 ^ 128 - 1))]
  for (v, i) in values.zipIdx do
    IO.println s!"value-{i}={v}"
    IO.println s!"clone-{i}={ok! (CloneBigUint.clone v)}"
    IO.println s!"try-u64-{i}={tryStr (ok! (TryFromU64.try_from v))}"
    IO.println s!"try-usize-{i}={tryStr (ok! (TryFromUsize.try_from v))}"
  for (a, i) in values.zipIdx do
    for (b, j) in values.zipIdx do
      IO.println s!"mul-{i}-{j}={ok! (MulBigUint.mul a b)}"
      IO.println s!"mulref-{i}-{j}={ok! (MulShared.mul a b)}"
      IO.println s!"div-{i}-{j}={render (DivShared.div a b)}"
      IO.println s!"rem-{i}-{j}={render (RemShared.rem a b)}"
      IO.println s!"divval-{i}-{j}={render (DivBigUint.div a b)}"
      IO.println s!"remval-{i}-{j}={render (RemBigUint.rem a b)}"
      let eq := ok! (num_bigint.PartialEqBigUint.eq a b)
      let ne := ok! (num_bigint.PartialEqBigUint.ne a b)
      IO.println s!"eq-{i}-{j}={eq}"
      IO.println s!"ne-{i}-{j}={ne}"
      IO.println s!"cmp-{i}-{j}={ordStr (ok! (num_bigint.OrdBigUint.cmp a b))}"
      let pc := match ok! (num_bigint.PartialOrdBigUint.partial_cmp a b) with
        | some o => ordStr o | none => "none"
      IO.println s!"partial_cmp-{i}-{j}={pc}"
  for (n, i) in [0, 1, 255].zipIdx do
    IO.println s!"from-u8-{i}={ok! (FromU8.from_ (u8 n))}"
  for (n, i) in [0, 1, 2 ^ 64 - 1].zipIdx do
    IO.println s!"from-u64-{i}={ok! (FromU64.from_ (u64 n))}"
  for (n, i) in [0, 1, 2 ^ 64 - 1].zipIdx do
    IO.println s!"from-usize-{i}={ok! (FromUsize.from_ (usize n))}"
  for (n, i) in [0, 2 ^ 128 - 1].zipIdx do
    IO.println s!"from-u128-{i}={ok! (FromU128.from_ (u128 n))}"
  for (v, i) in values.zipIdx do
    let (bytes, writes) := ok! (HashBigUint.hash logHasher v ([], 0))
    IO.println s!"hash-{i}={writes} {bytes}"
  for i in [0, 3, 7, 8, 11] do
    let v := values[i]!
    let line (label : String) (k : core.fmt.Formatter → core.fmt.Outcome)
        (o : core.fmt.Options) : IO Unit :=
      IO.println s!"fmt-{i}-{label}=[{run k o}]"
    let disp := DisplayBigUint.fmt v
    let debug := DebugBigUint.fmt v
    line "display" disp {}
    line "debug" debug {}
    line "plus" disp (opts (plus := true))
    line "alt" debug (opts (alt := true))
    line "lowerhex-debug" debug (opts (lowerHex := true))
    line "upperhex-debug" debug (opts (alt := true) (upperHex := true))
    line "width" disp (opts (width := some 8))
    line "width-debug" debug (opts (width := some 8))
    line "left" disp (opts (align := 0) (width := some 8))
    line "center" debug (opts (align := 2) (width := some 9))
    line "right" disp (opts (align := 1) (width := some 8))
    line "fill" disp (opts (fill := '*') (align := 2) (width := some 9))
    line "fill-utf8" debug (opts (fill := 'é') (align := 1) (width := some 6))
    line "zero" disp (opts (zero := true) (width := some 8))
    line "zero-plus" debug (opts (plus := true) (zero := true) (width := some 8))
    line "zero-left" disp (opts (fill := '*') (align := 0) (plus := true) (zero := true) (width := some 8))
    line "wide" debug (opts (align := 1) (plus := true) (width := some 44))
    line "narrow" disp (opts (width := some 1))
    line "derived" (literal v) {}
    line "derived-width" (quotient v values[3]!) (opts (zero := true) (width := some 6))
    line "derived-pretty" (quotient v values[1]!) (opts (alt := true))

#eval main

end BigUintModelChecks
