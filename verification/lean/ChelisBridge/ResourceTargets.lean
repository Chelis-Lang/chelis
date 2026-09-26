import ChelisBridge.Resource

/-!
# Resource build targets: extracted `validate_build_target_handler` versus `Target.accepts`

`chelis_effects::validate_build_target_handler` checks a `Resource` handler against the build
target: it pushes one `BuildTargetMismatch` error when the device is not admitted and leaves the
error list unchanged otherwise. "Rust accepts" below means the call returns the unchanged list.

Domain restriction (stated as theorems, not assumed away): only a device given as a string
literal in the **first** kid is checked (`kids.first().and_then(string_literal)`); a missing or
non-literal first kid, a `Random` handler and an undecodable effect kind (`Err _`) are accepted
for every target.

* `hip`/`metal`: Rust accepts ⇔ the device starts with `"gpu"` ⇔ `Target.accepts` (equivalence).
* `c`: Rust accepts ⇔ the device is exactly `"cpu"`; this implies `Target.accepts .c` but not
  conversely (`"tpu"`).
* any other target: Rust accepts every device (there is no reference target for it).
-/

namespace ChelisBridge.Resource

open Aeneas.Std LaCaDiLECore
open_chelis_model

abbrev Expr := chelis_deep.ast.Expr

/-- The only device Rust reads: `kids.first().and_then(string_literal)`. -/
def literalDevice (kids : Slice Expr) : Result (Option Str) :=
  match kids.val[0]? with
  | none => .ok none
  | some e => chelis_effects.string_literal e

/-- The extracted check on a decoded handler kind. -/
abbrev validate (kind : core.result.Result chelis_vocab.EffectKind
      chelis_vocab.EffectKindDecodeError)
    (kids : Slice Expr) (target : Str) (errors : alloc.vec.Vec chelis_effects.EffectError) :=
  chelis_effects.validate_build_target_handler kind kids target errors

/-- Rust's admission rule on a literal device (Lean strings). -/
def rustAccepts (target device : String) : Bool :=
  if target = "c" then decide (device = "cpu")
  else if target = "hip" ∨ target = "metal" then device.startsWith "gpu"
  else true

theorem str_eq_lit (t lit : String) (tgt : Str) (ht : Str.ofString t = .ok tgt) (h) :
    core.str.traits.PartialEqStr.eq tgt (toStr lit h) = .ok (decide (t = lit)) := by
  simp only [core.str.traits.PartialEqStr.eq]
  rw [ofString_val ht, toStr_val]
  congr 1
  by_cases hs : t = lit
  · subst hs; simp
  · simp only [hs, decide_false]
    rw [beq_eq_false_iff_ne]
    exact fun e => hs (bytesU8_inj e)

theorem starts_with_gpu (s : String) (d : Str) (hd : Str.ofString s = .ok d) (h) :
    core.str.Str.starts_with core.str.pattern.PatternSharedStr d (toStr "gpu" h) =
      .ok (s.startsWith "gpu") := by
  simp only [core.str.Str.starts_with, core.str.pattern.PatternSharedStr.is_prefix_of]
  congr 1
  rw [ofString_val hd, toStr_val]
  apply Bool.eq_iff_iff.mpr
  rw [List.isPrefixOf_iff_prefix, startsWith_iff_prefix]

theorem is_c_eq' (s : String) (d : Str) (hd : Str.ofString s = .ok d) :
    chelis_effects.is_c_host_device_designator d = .ok (decide (s = "cpu")) := by
  unfold chelis_effects.is_c_host_device_designator
  exact str_eq_lit s "cpu" d hd _

theorem first_eq (kids : Slice Expr) :
    core.slice.Slice.first kids = .ok kids.val[0]? := by
  simp [core.slice.Slice.first]

/-- `kids.first().and_then(string_literal)` is `literalDevice`. -/
theorem first_and_then (kids : Slice Expr) :
    (do
      let o ← core.slice.Slice.first kids
      core.option.Option.and_then
        (BuiltinFnOnce Expr (Option Str)) o chelis_effects.string_literal) =
      literalDevice kids := by
  rw [first_eq, bind_tc_ok]
  unfold literalDevice core.option.Option.and_then
  cases kids.val[0]? with
  | none => rfl
  | some e =>
    simp only [BuiltinFnOnce]
    cases chelis_effects.string_literal e <;> rfl

/-- The rejection branch pushes one error: it never returns the unchanged list. -/
theorem push_ne (errors : alloc.vec.Vec chelis_effects.EffectError)
    (e : chelis_effects.EffectError) : alloc.vec.Vec.push errors e ≠ .ok errors := by
  unfold alloc.vec.Vec.push
  intro h
  dsimp only at h
  split at h
  · have := congrArg (fun v => v.val.length) (Result.ok.inj h)
    simp at this
  · cases h

theorem bind_push_ne {α β : Type} (A : Result α) (B : α → Result β)
    (f : α → β → chelis_effects.EffectError) (errors : alloc.vec.Vec chelis_effects.EffectError) :
    (do let a ← A; let b ← B a; alloc.vec.Vec.push errors (f a b)) ≠ .ok errors := by
  cases A with
  | ok a =>
    simp only [bind_tc_ok]
    cases B a with
    | ok b => simp only [bind_tc_ok]; exact push_ne errors _
    | fail e => intro h; cases h
    | div => intro h; cases h
  | fail e => intro h; cases h
  | div => intro h; cases h

theorem ite_reject_iff (errors : alloc.vec.Vec chelis_effects.EffectError) (okv : Bool)
    (R : Result (alloc.vec.Vec chelis_effects.EffectError)) (hR : R ≠ .ok errors) :
    ((if okv = true then .ok errors else R) = .ok errors) ↔ okv = true := by
  cases okv <;> simp [hR]

/-- **Rust admission on a literal device.** For a `Resource` handler whose first kid is the
string literal `s`, the extracted check leaves the error list unchanged exactly when
`rustAccepts target s`, and otherwise pushes an error (never returns the unchanged list). -/
theorem resource_accepts_iff (t s : String) (tgt d : Str) (ht : Str.ofString t = .ok tgt)
    (hd : Str.ofString s = .ok d) (kids : Slice Expr)
    (hkids : literalDevice kids = .ok (some d))
    (errors : alloc.vec.Vec chelis_effects.EffectError) :
    validate (.Ok .Resource) kids tgt errors = .ok errors ↔ rustAccepts t s = true := by
  unfold validate chelis_effects.validate_build_target_handler
  simp only [← bind_assoc_eq]
  rw [first_and_then, hkids, bind_tc_ok]
  simp only [str_eq_lit t _ tgt ht, is_c_eq' s d hd, starts_with_gpu s d hd, bind_tc_ok]
  have hX : (if decide (t = "c") = true then Result.ok (decide (s = "cpu"))
      else if decide (t = "hip") = true then Result.ok (s.startsWith "gpu")
      else if decide (t = "metal") = true then Result.ok (s.startsWith "gpu")
      else Result.ok true : Result Bool) = .ok (rustAccepts t s) := by
    unfold rustAccepts
    by_cases hc : t = "c" <;> by_cases hh : t = "hip" <;> by_cases hm : t = "metal" <;>
      simp [hc, hh, hm]
  rw [hX, bind_tc_ok]
  exact ite_reject_iff errors _ _ (bind_push_ne _ _ _ errors)

/-- **GPU targets (`hip`, `metal`): equivalence.** On a literal device `s`, the extracted check
accepts exactly when `s` starts with `"gpu"`, i.e. exactly when the reference
`Target.accepts .hip` / `.metal` does. -/
theorem gpu_targets_equivalent (t s : String) (hgpu : t = "hip" ∨ t = "metal") (tgt d : Str)
    (ht : Str.ofString t = .ok tgt) (hd : Str.ofString s = .ok d) (kids : Slice Expr)
    (hkids : literalDevice kids = .ok (some d))
    (errors : alloc.vec.Vec chelis_effects.EffectError) :
    (validate (.Ok .Resource) kids tgt errors = .ok errors ↔ s.startsWith "gpu" = true) ∧
      (validate (.Ok .Resource) kids tgt errors = .ok errors ↔ Target.accepts .hip s = true) ∧
      (validate (.Ok .Resource) kids tgt errors = .ok errors ↔ Target.accepts .metal s = true) := by
  have h := resource_accepts_iff t s tgt d ht hd kids hkids errors
  have hc : t ≠ "c" := by rcases hgpu with rfl | rfl <;> decide
  have hr : rustAccepts t s = s.startsWith "gpu" := by
    simp [rustAccepts, hc, hgpu]
  rw [hr] at h
  exact ⟨h, h, h⟩

/-- **C target.** On a literal device `s`, the extracted check accepts exactly `"cpu"`. -/
theorem c_target_iff (s : String) (tgt d : Str) (ht : Str.ofString "c" = .ok tgt)
    (hd : Str.ofString s = .ok d) (kids : Slice Expr) (hkids : literalDevice kids = .ok (some d))
    (errors : alloc.vec.Vec chelis_effects.EffectError) :
    validate (.Ok .Resource) kids tgt errors = .ok errors ↔ s = "cpu" := by
  rw [resource_accepts_iff "c" s tgt d ht hd kids hkids errors]
  simp [rustAccepts]

/-- **C target, one-way refinement.** Rust acceptance implies the reference predicate. -/
theorem c_target_implies_reference (s : String) (tgt d : Str) (ht : Str.ofString "c" = .ok tgt)
    (hd : Str.ofString s = .ok d) (kids : Slice Expr) (hkids : literalDevice kids = .ok (some d))
    (errors : alloc.vec.Vec chelis_effects.EffectError)
    (hacc : validate (.Ok .Resource) kids tgt errors = .ok errors) :
    Target.accepts .c s = true := by
  have := (c_target_iff s tgt d ht hd kids hkids errors).mp hacc
  subst this; exact c_accepts_cpu

/-- **C target, converse fails.** The reference admits `"tpu"` on `c`; the extracted check
rejects it (pushes an error). -/
theorem c_target_reference_not_implies (tgt d : Str) (ht : Str.ofString "c" = .ok tgt)
    (hd : Str.ofString "tpu" = .ok d) (kids : Slice Expr)
    (hkids : literalDevice kids = .ok (some d))
    (errors : alloc.vec.Vec chelis_effects.EffectError) :
    Target.accepts .c "tpu" = true ∧ validate (.Ok .Resource) kids tgt errors ≠ .ok errors := by
  refine ⟨by decide +kernel, fun h => ?_⟩
  have := (c_target_iff "tpu" tgt d ht hd kids hkids errors).mp h
  exact absurd this (by decide)

/-- **Unknown targets accept every literal device.** -/
theorem unknown_target_accepts_all (t s : String) (hc : t ≠ "c") (hh : t ≠ "hip")
    (hm : t ≠ "metal") (tgt d : Str) (ht : Str.ofString t = .ok tgt)
    (hd : Str.ofString s = .ok d) (kids : Slice Expr) (hkids : literalDevice kids = .ok (some d))
    (errors : alloc.vec.Vec chelis_effects.EffectError) :
    validate (.Ok .Resource) kids tgt errors = .ok errors := by
  rw [resource_accepts_iff t s tgt d ht hd kids hkids errors]
  simp [rustAccepts, hc, hh, hm]

/-- **Domain restriction: non-literal devices are not checked.** When the first kid is missing
or is not a string literal, the extracted check accepts on every target. -/
theorem non_literal_device_unchecked (kids : Slice Expr) (tgt : Str)
    (hkids : literalDevice kids = .ok none)
    (errors : alloc.vec.Vec chelis_effects.EffectError) :
    validate (.Ok .Resource) kids tgt errors = .ok errors := by
  unfold validate chelis_effects.validate_build_target_handler
  simp only [← bind_assoc_eq]
  rw [first_and_then, hkids, bind_tc_ok]

/-- **Domain restriction: `Random` handlers and undecodable kinds are not checked.** -/
theorem random_and_decode_error_unchecked (kids : Slice Expr) (tgt : Str)
    (errors : alloc.vec.Vec chelis_effects.EffectError)
    (e : chelis_vocab.EffectKindDecodeError) :
    validate (.Ok .Random) kids tgt errors = .ok errors ∧
      validate (.Err e) kids tgt errors = .ok errors := ⟨rfl, rfl⟩

end ChelisBridge.Resource
