import ChelisBridge.EffectsModel
import ChelisBridge.StartsWith
import LaCaDiLECore.Protocol

/-!
# Resource target admission: extracted Rust checks versus `Target.accepts`

Devices are Lean `String`s; the extracted code sees the model's `str` view
`Str.ofString s` (its UTF-8 bytes as `U8`s).

* C target (one-way refinement). The extracted `is_c_host_device_designator` accepts exactly
  `"cpu"` (`is_c_eq`), so Rust acceptance implies reference acceptance
  (`c_accept_implies_reference`); the converse fails: the reference C predicate accepts every
  device not starting with `"gpu"`, e.g. `"tpu"`, which Rust rejects
  (`c_reference_not_implies`).
-/

namespace ChelisBridge.Resource

open Aeneas.Std LaCaDiLECore
open_chelis_effects_model

theorem ofString_val {s : String} {d : Str} (h : Str.ofString s = .ok d) : d.val = bytesU8 s := by
  unfold Str.ofString at h
  dsimp only at h
  split at h
  · cases h; rfl
  · cases h

theorem ofString_ok (s : String) (hs : s.toByteArray.size ≤ Usize.max) :
    ∃ d, Str.ofString s = .ok d := by
  unfold Str.ofString; dsimp only; rw [dif_pos hs]; exact ⟨_, rfl⟩

theorem toStr_val (s : String) (h : s.toByteArray.size ≤ U32.max) :
    (toStr s h).val = bytesU8 s := rfl

theorem bytesU8_inj {s t : String} (h : bytesU8 s = bytesU8 t) : s = t := by
  have hl := (List.map_injective_iff.mpr u8_injective) h
  have hb : s.toByteArray = t.toByteArray := ByteArray.ext (Array.ext' hl)
  cases s; cases t; cases hb; rfl

/-- The extracted C host check is exactly string equality with `"cpu"`. -/
theorem is_c_eq (s : String) (d : Str) (h : Str.ofString s = .ok d) :
    EffectsModel.isCHostDeviceDesignator d = .ok (decide (s = "cpu")) := by
  unfold EffectsModel.isCHostDeviceDesignator chelis_effects.is_c_host_device_designator
  simp only [core.str.traits.PartialEqStr.eq]
  rw [ofString_val h, toStr_val]
  congr 1
  by_cases hs : s = "cpu"
  · subst hs; simp
  · simp only [hs, decide_false]
    rw [beq_eq_false_iff_ne]
    exact fun e => hs (bytesU8_inj e)

theorem c_accepts_cpu : Target.accepts .c "cpu" = true := by decide +kernel

/-- **C target, one-way refinement.** Whenever the extracted host-C check accepts a device,
the reference C predicate accepts it. -/
theorem c_accept_implies_reference (s : String) (d : Str) (h : Str.ofString s = .ok d)
    (hacc : EffectsModel.isCHostDeviceDesignator d = .ok true) :
    Target.accepts .c s = true := by
  rw [is_c_eq s d h] at hacc
  have : s = "cpu" := by simpa using hacc
  subst this; exact c_accepts_cpu

/-- **C target, converse fails.** The reference C predicate accepts `"tpu"` (it does not start
with `"gpu"`), while the extracted host-C check rejects it. -/
theorem c_reference_not_implies :
    Target.accepts .c "tpu" = true ∧
      ∃ d, Str.ofString "tpu" = .ok d ∧ EffectsModel.isCHostDeviceDesignator d = .ok false := by
  have hsize : "tpu".toByteArray.size = 3 := by decide +kernel
  obtain ⟨d, hd⟩ := ofString_ok "tpu" (by rw [hsize]; scalar_tac)
  refine ⟨by decide +kernel, d, hd, ?_⟩
  rw [is_c_eq "tpu" d hd]
  congr 1

end ChelisBridge.Resource
