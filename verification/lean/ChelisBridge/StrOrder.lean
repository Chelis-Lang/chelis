import Aeneas

/-!
# `str` and `String` in the model: byte order, views and set operations

The model represents a Rust `String` as a Lean `String` (ordered by `compare`, the
lexicographic order of its characters) and a `&str` as its UTF-8 bytes (`Str`, ordered
byte-wise, `Ord for str`). A `BTreeSet<String>` looked up by `&str` (`UnordSet::remove(name)`)
therefore compares in both orders. This module proves, against the Aeneas library models:

* `compare_eq_lexCmp`: the byte-wise order of the UTF-8 encodings IS the `String` order
  (UTF-8 preserves the scalar-value order, `lexCmp_enc_ne`), so the two agree on every pair;
* `toString_ofString`, `to_string_ofString`: the `str` view of a representable `String`
  decodes, and `str::to_string` formats, back to that `String`;
* `insertList_eq`, `removeList_eq`: on a String-ordered set (`BTreeMap.Ordered`, the model's
  representation invariant), `BTreeMap::insert` is sorted insertion reporting membership and
  `BTreeMap::remove` by a `str` query removes exactly that key, provided every key it visits
  has a `str` view (at most `usize::MAX` bytes);
* `oversized_ofString`: that proviso is needed: the model's `String` is unbounded and a
  `String` with `usize::MAX + 1` bytes has no `str` view (no native `String` is that long:
  Rust allocations are at most `isize::MAX` bytes).
-/

open Aeneas Aeneas.Std

namespace ChelisBridge.StrOrder

/-- Lexicographic comparison of byte values (the comparison of `str`'s `Ord`). -/
def lexCmp : List Nat → List Nat → Ordering
  | [], [] => .eq
  | [], _ :: _ => .lt
  | _ :: _, [] => .gt
  | x :: xs, y :: ys =>
    match compare x y with
    | .eq => lexCmp xs ys
    | o => o

theorem lexCmp_cons (x y : Nat) (xs ys : List Nat) :
    lexCmp (x :: xs) (y :: ys) = if x < y then .lt else if y < x then .gt else lexCmp xs ys := by
  rcases Nat.lt_trichotomy x y with h | h | h
  · simp [lexCmp, Nat.compare_eq_lt.mpr h, h]
  · subst h; simp [lexCmp]
  · simp [lexCmp, Nat.compare_eq_gt.mpr h, h, Nat.lt_asymm h]

theorem lexCmp_append_same (p xs ys : List Nat) : lexCmp (p ++ xs) (p ++ ys) = lexCmp xs ys := by
  induction p with
  | nil => rfl
  | cons a p ih => simp [lexCmp_cons, ih]

/-- The byte values of one scalar value's UTF-8 encoding (Prelude `String.utf8EncodeChar`). -/
def encN (v : Nat) : List Nat :=
  if v ≤ 0x7f then [v]
  else if v ≤ 0x7ff then [v / 64 % 0x20 + 0xc0, v % 0x40 + 0x80]
  else if v ≤ 0xffff then [v / 4096 % 0x10 + 0xe0, v / 64 % 0x40 + 0x80, v % 0x40 + 0x80]
  else [v / 262144 % 0x08 + 0xf0, v / 4096 % 0x40 + 0x80, v / 64 % 0x40 + 0x80, v % 0x40 + 0x80]

theorem char_lt (c : Char) : c.val.toNat < 0x110000 := by
  have := c.valid
  simp only [UInt32.isValidChar, Nat.isValidChar] at this
  omega

theorem encodeChar_toNat (c : Char) :
    (String.utf8EncodeChar c).map UInt8.toNat = encN c.val.toNat := by
  have hc := char_lt c
  unfold String.utf8EncodeChar encN
  simp only
  generalize c.val.toNat = v at hc ⊢
  split_ifs <;> simp <;> omega

theorem encN_ne_nil (v : Nat) : encN v ≠ [] := by
  unfold encN; split_ifs <;> simp

/-- Distinct scalar values: their encodings differ within both, in the value order. -/
theorem lexCmp_enc_ne (v w : Nat) (hv : v < 0x110000) (hw : w < 0x110000) (hne : v ≠ w)
    (xs ys : List Nat) : lexCmp (encN v ++ xs) (encN w ++ ys) = compare v w := by
  unfold encN
  split_ifs <;> simp only [List.cons_append, List.nil_append, lexCmp_cons] <;> split_ifs <;>
    first
    | exact (Nat.compare_eq_lt.mpr (by omega)).symm
    | exact (Nat.compare_eq_gt.mpr (by omega)).symm
    | exact absurd (by omega : v = w) hne


/-- The byte values of the UTF-8 encoding of a character list. -/
def bytesOf (l : List Char) : List Nat := l.flatMap fun c => encN c.val.toNat

theorem val_toNat_inj {c d : Char} (h : c.val.toNat = d.val.toNat) : c = d := by
  apply Char.ext; exact UInt32.toNat_inj.mp h

/-- UTF-8 byte order is the lexicographic scalar-value order of the characters. -/
theorem lexCmp_bytesOf (l m : List Char) :
    lexCmp (bytesOf l) (bytesOf m) = if l < m then .lt else if l = m then .eq else .gt := by
  induction l generalizing m with
  | nil =>
    cases m with
    | nil => rfl
    | cons d m =>
      obtain ⟨b, bs, hb⟩ := List.exists_cons_of_ne_nil (encN_ne_nil d.val.toNat)
      simp [bytesOf, hb, lexCmp]
  | cons c l ih =>
    cases m with
    | nil =>
      obtain ⟨b, bs, hb⟩ := List.exists_cons_of_ne_nil (encN_ne_nil c.val.toNat)
      simp [bytesOf, hb, lexCmp]
    | cons d m =>
      have hbl : bytesOf (c :: l) = encN c.val.toNat ++ bytesOf l := by simp [bytesOf]
      have hbm : bytesOf (d :: m) = encN d.val.toNat ++ bytesOf m := by simp [bytesOf]
      rw [hbl, hbm]
      by_cases hcd : c = d
      · subst hcd
        rw [lexCmp_append_same, ih]
        simp
      · have hne : c.val.toNat ≠ d.val.toNat := fun h => hcd (val_toNat_inj h)
        rw [lexCmp_enc_ne _ _ (char_lt c) (char_lt d) hne]
        have hlt : c < d ↔ c.val.toNat < d.val.toNat := by
          rw [Char.lt_def, UInt32.lt_iff_toNat_lt]
        rcases Nat.lt_trichotomy c.val.toNat d.val.toNat with h | h | h
        · rw [Nat.compare_eq_lt.mpr h]
          simp [List.cons_lt_cons_iff, hlt.mpr h]
        · exact absurd h hne
        · rw [Nat.compare_eq_gt.mpr h]
          have : ¬ c < d := by rw [hlt]; omega
          simp [List.cons_lt_cons_iff, this, hcd]

/-- The bytes of a `String`. -/
theorem toByteArray_bytes (a : String) :
    a.toByteArray.data.toList.map UInt8.toNat = bytesOf a.toList := by
  rw [← String.utf8Encode_toList, List.utf8Encode, List.toList_data_toByteArray, List.map_flatMap]
  simp only [bytesOf, encodeChar_toNat]

/-- `compare` on `String` is the lexicographic order of the UTF-8 bytes. -/
theorem compare_eq_lexCmp (a b : String) :
    compare a b = lexCmp (a.toByteArray.data.toList.map UInt8.toNat)
      (b.toByteArray.data.toList.map UInt8.toNat) := by
  rw [toByteArray_bytes, toByteArray_bytes, lexCmp_bytesOf]
  show compareOfLessAndEq a b = _
  simp only [compareOfLessAndEq]
  have hlt : a < b ↔ a.toList < b.toList := Iff.rfl
  have heq : a = b ↔ a.toList = b.toList := ⟨fun h => h ▸ rfl, String.toList_inj.mp⟩
  by_cases h1 : a < b
  · simp [h1, hlt.mp h1]
  · simp [h1, (not_congr hlt).mp h1, heq]

open private compareStrBytes from Aeneas.Std.String in
theorem compareStrBytes_eq (xs ys : List U8) :
    compareStrBytes xs ys = lexCmp (xs.map (·.val)) (ys.map (·.val)) := by
  induction xs generalizing ys with
  | nil => cases ys <;> rfl
  | cons x xs ih =>
    cases ys with
    | nil => rfl
    | cons y ys =>
      simp only [compareStrBytes, List.map_cons, lexCmp, ih]
      generalize compare x.val y.val = o
      cases o <;> rfl

/-- `Ord for str`, on the byte values. -/
theorem ordStr_cmp (s t : Str) :
    core.str.traits.OrdStr.cmp s t = .ok (lexCmp (s.val.map (·.val)) (t.val.map (·.val))) := by
  simp only [core.str.traits.OrdStr.cmp, compareStrBytes_eq]

/-- The model's `str` view of a representable `String`. -/
theorem ofString_ok (a : String) (h : a.toByteArray.size ≤ Usize.max) :
    ∃ s : Str, Str.ofString a = .ok s ∧ s.val.map (·.val) = a.toByteArray.data.toList.map UInt8.toNat := by
  refine ⟨_, by rw [Str.ofString, dif_pos h], ?_⟩
  simp only [List.map_map]
  rfl


open alloc.collections.btree.map

/-- `compare` on `String` in terms of `<`. -/
theorem compare_string (a b : String) :
    compare a b = if a < b then .lt else if a = b then .eq else .gt := rfl

theorem compare_lt_iff (a b : String) : compare a b = .lt ↔ a < b := by
  rw [compare_string]; split_ifs <;> simp_all

/-- The keys of a set entry list. -/
def keys (l : List (String × Unit)) : List String := l.map Prod.fst

/-- Sorted insertion (a present key leaves the list unchanged). -/
def insList (m : String) : List (String × Unit) → List (String × Unit)
  | [] => [(m, ())]
  | (k, u) :: rest =>
    if m < k then (m, ()) :: (k, u) :: rest
    else if m = k then (k, ()) :: rest
    else (k, u) :: insList m rest

/-- Removal of every entry with key `m`. -/
def delList (m : String) (l : List (String × Unit)) : List (String × Unit) :=
  l.filter fun kv => kv.1 != m

/-- String-ordered entry list (the model's `BTreeMap.Ordered` for `Ord for String`). -/
theorem ordered_iff (l : List (String × Unit)) :
    BTreeMap.Ordered core.cmp.OrdString ⟨l⟩ ↔ l.Pairwise (fun a b => a.1 < b.1) := by
  simp only [BTreeMap.Ordered, alloc.string.OrdString.cmp, Result.ok.injEq,
    compare_lt_iff]

theorem not_mem_of_lt {m k : String} {rest : List (String × Unit)}
    (hs : ((k, ()) :: rest).Pairwise (fun a b => a.1 < b.1)) (h : m < k) :
    m ∉ keys ((k, ()) :: rest) := by
  simp only [keys, List.map_cons, List.mem_cons, List.mem_map, not_or]
  refine ⟨fun e => String.lt_irrefl m (e ▸ h), ?_⟩
  rintro ⟨⟨k', u⟩, hk', rfl⟩
  have := List.rel_of_pairwise_cons hs hk'
  exact String.lt_irrefl _ (String.lt_trans this h)

/-- `BTreeMap::insert` on a String-keyed set: the sorted insertion; the returned old value is
present exactly when the key was. -/
theorem insertList_eq (m : String) (l : List (String × Unit))
    (hs : l.Pairwise (fun a b => a.1 < b.1)) :
    insertList core.cmp.OrdString m () l =
      .ok (if m ∈ keys l then some () else none, insList m l) := by
  induction l with
  | nil => simp [insertList, insList, keys]
  | cons kv rest ih =>
    obtain ⟨k, ⟨⟩⟩ := kv
    have ih := ih (List.Pairwise.of_cons hs)
    simp only [insertList, core.cmp.OrdString, alloc.string.OrdString.cmp, bind_tc_ok,
      compare_string, insList]
    by_cases h1 : m < k
    · simp [h1, not_mem_of_lt hs h1]
    · by_cases h2 : m = k
      · subst h2; simp [keys]
      · simp only [h1, h2, if_false, ih, bind_tc_ok]
        simp [keys, h2]

/-- `BTreeMap::remove` with a `str` query on a String-ordered set of representable keys: removal
of the key (the `str` byte order agrees with the `String` order). -/
theorem removeList_eq (m : String) (s : Str) (hm : Str.ofString m = .ok s)
    (l : List (String × Unit)) (hs : l.Pairwise (fun a b => a.1 < b.1))
    (hr : ∀ k ∈ keys l, k.toByteArray.size ≤ Usize.max) :
    removeList core.borrow.BorrowString core.cmp.OrdStr s l =
      .ok (if m ∈ keys l then some () else none, delList m l) := by
  have hmr : m.toByteArray.size ≤ Usize.max := by
    by_contra hc
    rw [String.size_toByteArray] at hc
    simp [Str.ofString, hc] at hm
  obtain ⟨s', hs', hsv⟩ := ofString_ok m hmr
  rw [hm] at hs'; cases hs'
  induction l with
  | nil => simp [removeList, keys, delList]
  | cons kv rest ih =>
    obtain ⟨k, ⟨⟩⟩ := kv
    have ih := ih (List.Pairwise.of_cons hs) (fun k' hk' => hr k' (by simp [keys] at hk' ⊢; exact Or.inr hk'))
    obtain ⟨sk, hsk, hskv⟩ := ofString_ok k (hr k (by simp [keys]))
    have hcmp : core.str.traits.OrdStr.cmp s sk = .ok (compare m k) := by
      rw [ordStr_cmp, hsv, hskv, compare_eq_lexCmp]
    simp only [removeList, core.borrow.BorrowString, alloc.str.BorrowString.borrow, hsk,
      bind_tc_ok, core.cmp.OrdStr, hcmp, compare_string]
    by_cases h1 : m < k
    · simp only [h1, if_true, not_mem_of_lt hs h1, if_false, Result.ok.injEq, Prod.mk.injEq, true_and]
      simp only [delList, List.filter_cons]
      have : ∀ kv ∈ rest, (kv.1 != m) = true := by
        intro kv hkv
        have := String.lt_trans h1 (List.rel_of_pairwise_cons hs hkv)
        simp only [bne_iff_ne, ne_eq]
        exact fun e => String.lt_irrefl m (e ▸ this)
      simp [List.filter_eq_self.mpr this, show k ≠ m from fun e => String.lt_irrefl m (e ▸ h1)]
    · by_cases h2 : m = k
      · subst h2
        have : ∀ kv ∈ rest, (kv.1 != m) = true := by
          intro kv hkv
          have := List.rel_of_pairwise_cons hs hkv
          simp only [bne_iff_ne, ne_eq]
          exact fun e => String.lt_irrefl m (e ▸ this)
        simp [keys, delList, List.filter_eq_self.mpr this]
      · simp only [h1, h2, if_false, ih, bind_tc_ok]
        simp [keys, delList, Ne.symm h2, h2]


theorem foldl_push (l : List UInt8) (b : ByteArray) :
    l.foldl (fun bytes byte => bytes.push byte) b = ⟨b.data ++ l.toArray⟩ := by
  induction l generalizing b with
  | nil => simp
  | cons x l ih =>
    rw [List.foldl_cons, ih]
    cases b
    simp [ByteArray.push]

/-- A representable `String`'s `str` view decodes back to it. -/
theorem toString_ofString (m : String) (s : Str) (hm : Str.ofString m = .ok s) :
    Str.toString s = .ok m := by
  have hmr : m.toByteArray.size ≤ Usize.max := by
    by_contra hc
    rw [String.size_toByteArray] at hc
    simp [Str.ofString, hc] at hm
  rw [Str.ofString, dif_pos hmr] at hm
  cases hm
  have hb : (List.map (fun byte => (⟨byte.toBitVec⟩ : U8)) m.toByteArray.data.toList).foldl
      (fun bytes byte => bytes.push (UInt8.ofBitVec byte.bv)) ByteArray.empty = m.toByteArray := by
    rw [List.foldl_map]
    have : ∀ byte : UInt8, UInt8.ofBitVec (⟨byte.toBitVec⟩ : U8).bv = byte := fun _ => rfl
    simp only [this, foldl_push]
    simp
  simp only [Str.toString, hb, String.fromUTF8?, dif_pos m.isValidUTF8]
  rfl

/-- `str::to_string` (the blanket `ToString` through `Display for str`) of a representable
`String`'s view is that `String`. -/
theorem to_string_ofString (m : String) (s : Str) (hm : Str.ofString m = .ok s) :
    alloc.string.ToStringBlanket.to_string core.fmt.DisplayStr s = .ok m := by
  have hmr : m.toByteArray.size ≤ Usize.max := by
    by_contra hc
    rw [String.size_toByteArray] at hc
    simp [Str.ofString, hc] at hm
  have hlen : s.val.length ≤ Usize.max := s.property
  simp [alloc.string.ToStringBlanket.to_string, core.fmt.DisplayStr.fmt, core.fmt.decode, hlen, toString_ofString m s hm, core.fmt.displayString, core.fmt.Options.bit,
    core.fmt.Formatter.pad, core.fmt.Formatter.padWith, core.fmt.writeFill, core.fmt.writeRepeated,
    core.fmt.thenFormat, core.fmt.Formatter.append]


/-- A `String` of `n` ASCII letters has `n` UTF-8 bytes. -/
theorem size_replicate (n : Nat) :
    (String.ofList (List.replicate n 'a')).toByteArray.size = n := by
  rw [String.toByteArray_ofList, List.utf8Encode, ← ByteArray.size_data, List.data_toByteArray,
    List.size_toArray]
  have h : String.utf8EncodeChar 'a' = [97] := by decide
  induction n with
  | zero => rfl
  | succ n ih => simp [List.replicate_succ, h, ih]

/-- A `String` with more UTF-8 bytes than `usize::MAX`: no native `String` is that long (Rust
allocations are at most `isize::MAX` bytes), but the model's `String` is unbounded. -/
def oversized : String := String.ofList (List.replicate (Usize.max + 1) 'a')

theorem oversized_ofString : Str.ofString oversized = .fail .panic := by
  have : ¬ oversized.toByteArray.size ≤ Usize.max := by
    rw [oversized, size_replicate]; omega
  rw [Str.ofString, dif_neg this]

/-! ## `BTreeSet<String>` insert by `String`, remove by `&str` -/

open alloc.collections.btree.set in
/-- `BTreeSet::insert` on a String-ordered set: reports whether the key was absent and inserts
it in order. -/
theorem btreeSet_insert (m : String) (S : BTreeSet String)
    (hs : S.map.entries.Pairwise (fun a b => a.1 < b.1)) :
    BTreeSet.insert core.cmp.OrdString S m =
      .ok (decide (m ∉ keys S.map.entries), ⟨⟨insList m S.map.entries⟩⟩) := by
  obtain ⟨⟨l⟩⟩ := S
  simp only [BTreeSet.insert, BTreeMap.insert, insertList_eq m l hs, bind_tc_ok]
  by_cases h : m ∈ keys l <;> simp [h]

open alloc.collections.btree.set in
/-- `BTreeSet::remove` by a `str` query on a String-ordered set of representable keys: reports
whether the key was present and removes it. -/
theorem btreeSet_remove (m : String) (s : Str) (hm : Str.ofString m = .ok s) (S : BTreeSet String)
    (hs : S.map.entries.Pairwise (fun a b => a.1 < b.1))
    (hr : ∀ k ∈ keys S.map.entries, k.toByteArray.size ≤ Usize.max) :
    BTreeSet.remove core.borrow.BorrowString core.cmp.OrdString core.cmp.OrdStr S s =
      .ok (decide (m ∈ keys S.map.entries), ⟨⟨delList m S.map.entries⟩⟩) := by
  obtain ⟨⟨l⟩⟩ := S
  simp only [BTreeSet.remove, BTreeMap.remove, removeList_eq m s hm l hs hr, bind_tc_ok]
  by_cases h : m ∈ keys l <;> simp [h]

/-- Sorted insertion keeps the order. -/
theorem insList_pairwise (m : String) (l : List (String × Unit))
    (hs : l.Pairwise (fun a b => a.1 < b.1)) :
    (insList m l).Pairwise (fun a b => a.1 < b.1) := by
  induction l with
  | nil => simp [insList]
  | cons kv rest ih =>
    obtain ⟨k, ⟨⟩⟩ := kv
    have hk := List.pairwise_cons.mp hs
    simp only [insList]
    by_cases h1 : m < k
    · simp only [h1, if_true]
      refine List.Pairwise.cons (fun b hb => ?_) hs
      simp only [List.mem_cons] at hb
      rcases hb with rfl | hb
      · exact h1
      · exact String.lt_trans h1 (hk.1 b hb)
    · by_cases h2 : m = k
      · subst h2; simpa [h1] using hs
      · simp only [h1, h2, if_false]
        have hmem : ∀ b ∈ insList m rest, b.1 = m ∨ b ∈ rest := by
          intro b hb
          clear ih hk hs h1 h2
          induction rest with
          | nil => simp [insList] at hb; exact Or.inl (by rw [hb])
          | cons kv' rest' ih' =>
            obtain ⟨k', ⟨⟩⟩ := kv'
            simp only [insList] at hb
            split_ifs at hb with h3 h4
            · simp only [List.mem_cons] at hb
              rcases hb with rfl | rfl | hb
              · exact Or.inl rfl
              · exact Or.inr (List.mem_cons_self ..)
              · exact Or.inr (List.mem_cons_of_mem _ hb)
            · simp only [List.mem_cons] at hb
              rcases hb with rfl | hb
              · exact Or.inr (by subst h4; exact List.mem_cons_self ..)
              · exact Or.inr (List.mem_cons_of_mem _ hb)
            · simp only [List.mem_cons] at hb
              rcases hb with rfl | hb
              · exact Or.inr (List.mem_cons_self ..)
              · rcases ih' hb with h | h
                · exact Or.inl h
                · exact Or.inr (List.mem_cons_of_mem _ h)
        refine List.Pairwise.cons (fun b hb => ?_) (ih hk.2)
        rcases hmem b hb with h | h
        · rw [h]
          by_contra h3
          exact h2 (String.le_antisymm (String.not_lt.mp h3) (String.not_lt.mp h1))
        · exact hk.1 b h

/-- Removing a key that was absent after inserting it gives back the set. -/
theorem delList_insList (m : String) (l : List (String × Unit)) (hm : m ∉ keys l) :
    delList m (insList m l) = l := by
  induction l with
  | nil => simp [insList, delList]
  | cons kv rest ih =>
    obtain ⟨k, ⟨⟩⟩ := kv
    simp only [keys, List.map_cons, List.mem_cons, not_or] at hm
    have hrest : delList m rest = rest := by
      unfold delList
      refine List.filter_eq_self.mpr fun b hb => ?_
      simp only [bne_iff_ne, ne_eq]
      intro e; exact hm.2 (List.mem_map.mpr ⟨b, hb, e⟩)
    simp only [insList]
    by_cases h1 : m < k
    · simp only [h1, if_true, delList, List.filter_cons]
      simp only [delList] at hrest
      simp [hrest, Ne.symm hm.1]
    · simp only [h1, hm.1, if_false, delList, List.filter_cons]
      have := ih hm.2
      simp only [delList] at this
      simp [this, Ne.symm hm.1]

/-- `BTreeMap::insert` of an absent key, on any entry list (ordered or not): no old value. -/
theorem insertList_absent (m : String) (l : List (String × Unit)) (hm : m ∉ keys l) :
    insertList core.cmp.OrdString m () l = .ok (none, insList m l) := by
  induction l with
  | nil => simp [insertList, insList]
  | cons kv rest ih =>
    obtain ⟨k, ⟨⟩⟩ := kv
    simp only [keys, List.map_cons, List.mem_cons, not_or] at hm
    have ih := ih hm.2
    simp only [insertList, core.cmp.OrdString, alloc.string.OrdString.cmp, bind_tc_ok,
      compare_string, insList]
    by_cases h1 : m < k
    · simp [h1]
    · simp only [h1, hm.1, if_false, ih, bind_tc_ok]

/-- `BTreeMap::remove` by any `str` query fails on a set whose first key has no `str` view. -/
theorem removeList_oversized (s : Str) (rest : List (String × Unit)) :
    removeList core.borrow.BorrowString core.cmp.OrdStr s ((oversized, ()) :: rest) = .fail .panic := by
  simp only [removeList, core.borrow.BorrowString, alloc.str.BorrowString.borrow, oversized_ofString,
    bind_tc_fail]

/-- The keys after a sorted insertion: the inserted key and the old ones. -/
theorem mem_keys_insList (m k : String) (l : List (String × Unit)) (hk : k ∈ keys (insList m l)) :
    k = m ∨ k ∈ keys l := by
  induction l with
  | nil => simp [insList, keys] at hk; exact Or.inl hk
  | cons kv rest ih =>
    obtain ⟨k', ⟨⟩⟩ := kv
    simp only [insList] at hk
    split_ifs at hk with h1 h2
    · simp only [keys, List.map_cons, List.mem_cons] at hk ⊢
      rcases hk with h | h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
      · exact Or.inr (Or.inr h)
    · simp only [keys, List.map_cons, List.mem_cons] at hk ⊢
      exact Or.inr hk
    · simp only [keys, List.map_cons, List.mem_cons] at hk ⊢
      rcases hk with h | h
      · exact Or.inr (Or.inl h)
      · rcases ih h with h' | h'
        · exact Or.inl h'
        · exact Or.inr (Or.inr h')

/-- Inserting a present key into a String-ordered set changes nothing. -/
theorem insList_of_mem (m : String) (l : List (String × Unit))
    (hs : l.Pairwise (fun a b => a.1 < b.1)) (hm : m ∈ keys l) : insList m l = l := by
  induction l with
  | nil => simp [keys] at hm
  | cons kv rest ih =>
    obtain ⟨k, ⟨⟩⟩ := kv
    simp only [insList]
    by_cases h1 : m < k
    · exact absurd hm (not_mem_of_lt hs h1)
    · by_cases h2 : m = k
      · subst h2; simp
      · simp only [keys, List.map_cons, List.mem_cons] at hm
        rcases hm with hm | hm
        · exact absurd hm h2
        · simp only [h1, h2, if_false, ih (List.Pairwise.of_cons hs) hm]

end ChelisBridge.StrOrder
