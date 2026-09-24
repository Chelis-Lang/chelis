import Aeneas
open String String.Slice.Pattern Aeneas.Std

namespace ChelisBridge.Resource

theorem startsWith_bytes (s p : String) :
    s.startsWith p = true ↔
      p.toByteArray.size ≤ s.toByteArray.size ∧
        s.toByteArray.extract 0 p.toByteArray.size = p.toByteArray := by
  show ForwardSliceSearcher.startsWith p.toSlice s.toSlice = true ↔ _
  rw [ForwardSliceSearcher.startsWith]
  simp [Internal.memcmpSlice_eq_true_iff, -size_toByteArray]
  simp only [← size_toByteArray, ByteArray.extract_zero_size]

/-- The byte image used by the model's own `str` view of a Lean string. -/
def bytesU8 (s : String) : List U8 := s.toByteArray.data.toList.map (fun b => (⟨b.toBitVec⟩ : U8))

theorem u8_injective : Function.Injective (fun b : UInt8 => (⟨b.toBitVec⟩ : U8)) := by
  intro a b hab
  have : a.toBitVec = b.toBitVec := congrArg UScalar.bv hab
  exact UInt8.toBitVec_inj.mp this

theorem byteArray_prefix_iff (a b : ByteArray) :
    (a.size ≤ b.size ∧ b.extract 0 a.size = a) ↔ a.data.toList <+: b.data.toList := by
  rw [List.prefix_iff_eq_take]
  constructor
  · rintro ⟨hle, he⟩
    have := congrArg (fun x : ByteArray => x.data.toList) he
    simp only [ByteArray.data_extract, Array.toList_extract, List.extract_eq_take_drop,
      List.drop_zero, Nat.sub_zero] at this
    rw [Array.length_toList]
    exact this.symm
  · intro h
    have hl : a.data.toList.length ≤ b.data.toList.length := by
      rw [h]; exact List.length_take_le' _ _
    refine ⟨by simpa using hl, ?_⟩
    apply ByteArray.ext
    apply Array.ext'
    simp only [ByteArray.data_extract, Array.toList_extract, List.extract_eq_take_drop,
      List.drop_zero, Nat.sub_zero]
    show List.take a.data.size _ = _
    rw [← Array.length_toList]; exact h.symm

/-- `String.startsWith` is the byte-prefix relation on the model's `U8` images. -/
theorem startsWith_iff_prefix (s p : String) :
    s.startsWith p = true ↔ bytesU8 p <+: bytesU8 s := by
  rw [startsWith_bytes, byteArray_prefix_iff]
  unfold bytesU8
  exact (List.prefix_map_iff_of_injective u8_injective).symm

end ChelisBridge.Resource

