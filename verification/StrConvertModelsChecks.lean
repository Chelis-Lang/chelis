import StrConvertModels

open Aeneas Aeneas.Std

namespace StrConvertModelsChecks

/-! Mirrors the native tests of `str_convert_models.rs`. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def nat {ty : UScalarTy} (x : UScalar ty) : Nat := x.val

def int {ty : IScalarTy} (x : IScalar ty) : Int := x.val

def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def us (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩

def vals {α β : Type} (f : α → β) (l : { l : List α // l.length ≤ Usize.max }) : List β :=
  l.val.map f

def str (s : String) : Str := match Str.ofString s with | .ok v => v | _ => ⟨[], by simp⟩

def text (s : Str) : String := match Str.toString s with | .ok v => v | _ => "<invalid>"

open str_convert_models

#eval do
  expect "bytes_of" (decode (vals nat) (bytes_of "aé")) (.ok [97, 195, 169])
  expect "bytes_of empty" (decode (vals nat) (bytes_of "")) (.ok [])
  expect "is_target" (decode id (is_target (str "map") "map")) (.ok (true, false))
  expect "is_target differ" (decode id (is_target (str "map") "mapé")) (.ok (false, true))
  expect "is_target empty" (decode id (is_target (str "") "")) (.ok (true, false))
  expect "owned_name" (decode id (owned_name "x.y")) (.ok "x.y")
  expect "boxed_context" (decode (fun ((n, b) : Usize × Str) => (n.val, text b))
    (boxed_context (str "héllo"))) (.ok (6, "héllo"))

#eval do
  let source := some (str "héllo")
  expect "spelling é" (decode id (spelling source (us 1) (us 3))) (.ok (some "é"))
  expect "spelling inside é" (decode id (spelling source (us 1) (us 2))) (.ok none)
  expect "spelling reversed" (decode id (spelling source (us 3) (us 2))) (.ok none)
  expect "spelling all" (decode id (spelling source (us 0) (us 6))) (.ok (some "héllo"))
  expect "spelling past end" (decode id (spelling source (us 0) (us 7))) (.ok none)
  expect "spelling empty" (decode id (spelling source (us 6) (us 6))) (.ok (some ""))
  expect "spelling none" (decode id (spelling none (us 0) (us 1))) (.ok none)

#eval do
  expect "bare_name" (decode text (bare_name (str "core.ops.Add"))) (.ok "Add")
  expect "bare_name plain" (decode text (bare_name (str "Add"))) (.ok "Add")
  expect "bare_name trailing" (decode text (bare_name (str "a."))) (.ok "")
  expect "segments" (decode (vals text) (segments_from_end (str "a.b..c"))) (.ok ["c", "", "b", "a"])
  expect "segments empty" (decode (vals text) (segments_from_end (str ""))) (.ok [""])

#eval do
  let r := fun (s : String) (radix : Nat) =>
    decode (fun (x : core.result.Result I64 U8) => match x with
      | .Ok v => Sum.inl (int v)
      | .Err k => Sum.inr (nat k)) (parse_radix (str s) (u32 radix))
  expect "ff" (r "ff" 16) (.ok (.inl 255))
  expect "Ff" (r "Ff" 16) (.ok (.inl 255))
  expect "+7f" (r "+7f" 16) (.ok (.inl 127))
  expect "-ff" (r "-ff" 16) (.ok (.inl (-255)))
  expect "max" (r "7fffffffffffffff" 16) (.ok (.inl (2 ^ 63 - 1)))
  expect "pos overflow" (r "8000000000000000" 16) (.ok (.inr 3))
  expect "min" (r "-8000000000000000" 16) (.ok (.inl (-(2 ^ 63))))
  expect "neg overflow" (r "-8000000000000001" 16) (.ok (.inr 4))
  expect "1g" (r "1g" 16) (.ok (.inr 2))
  expect "overflow before g" (r "fffffffffffffffffg" 16) (.ok (.inr 3))
  expect "g at the overflowing step" (r "7fffffffffffffffg" 16) (.ok (.inr 2))
  expect "101" (r "101" 2) (.ok (.inl 5))
  expect "102" (r "102" 2) (.ok (.inr 2))
  expect "zz" (r "zz" 36) (.ok (.inl 1295))
  expect "empty" (r "" 16) (.ok (.inr 1))
  expect "-" (r "-" 16) (.ok (.inr 2))
  expect "+" (r "+" 2) (.ok (.inr 2))
  expect "0x10" (r "0x10" 16) (.ok (.inr 2))
  expect "radix 37" (r "1" 37) (.error "panic")

end StrConvertModelsChecks
