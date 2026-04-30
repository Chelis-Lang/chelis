-- LaCaDiLE/StringHelpers.lean
--
-- Helper lemmas supporting fresh-name disequality proofs needed by
-- `adjoint_typed_aux`'s `add`/`mul` cases in `AdjointTyping.lean`.
--
-- Strategy: avoid `Nat.toString` injectivity (which would require
-- cracking `Nat.repr`'s digit-list representation) by redefining
-- `freshName` in `AdjointTransform.lean` to use a bijective
-- character-list encoding:
--
--     freshName base n = String.ofList (base.toList ++ '#' :: replicate n 'x')
--
-- The counter is a unary run of `'x'` characters at the tail, separated
-- from the base by a fixed marker character `'#'`. Injectivity facts
-- reduce to `List.length` and `List.append` reasoning through
-- `String.toList_ofList`.
--
-- Cross-base disequality requires that neither base contains `'#'`.
-- All concrete bases used by `adjointFrom` (`"gA"`, `"gB"`, `"a"`,
-- `"aTape"`, `"b"`, `"bTape"`, `"y"`, `"adjA"`, `"adjHb"`) are
-- alphabetic, so the `NoHash` hypothesis is discharged by `decide`
-- at each call site.

namespace LaCaDiLE

/-- Counter-based fresh name. Attaches the counter `n` to the given
    prefix so two recursion levels with different counters produce
    disjoint names. Keeping this in `StringHelpers` makes the generated
    name discipline available to `Syntax`/`Typing` without an
    `AdjointTransform` import cycle. -/
def freshName (base : String) (n : Nat) : String :=
  String.ofList (base.toList ++ '#' :: List.replicate n 'x')

/-- A `String` contains no `'#'` character. Used to guarantee the
    `freshName` separator is unambiguous. -/
def NoHash (s : String) : Prop := '#' ∉ s.toList

/-- `String.ofList` is injective on the underlying `List Char`. -/
theorem ofList_inj {xs ys : List Char}
    (h : String.ofList xs = String.ofList ys) : xs = ys := by
  have h1 : (String.ofList xs).toList = (String.ofList ys).toList := by rw [h]
  rw [String.toList_ofList, String.toList_ofList] at h1
  exact h1

/-- `List.replicate` is injective in its length argument. -/
theorem replicate_length_eq {α : Type _} (a : α) (n m : Nat)
    (h : List.replicate n a = List.replicate m a) : n = m := by
  have := congrArg List.length h
  simpa using this

/-- Left-cancellation for `List.append`. -/
theorem append_left_cancel {α : Type _} :
    ∀ (xs ys zs : List α), xs ++ ys = xs ++ zs → ys = zs
  | [], _, _, h => by simpa using h
  | _ :: xs, ys, zs, h => by
      rw [List.cons_append, List.cons_append, List.cons.injEq] at h
      exact append_left_cancel xs ys zs h.2

/-- Framing uniqueness: if `xs ++ ('#' :: ys) = zs ++ ('#' :: ws)` and
    neither `xs` nor `zs` contains `'#'`, then `xs = zs` and `ys = ws`. -/
theorem frame_inj :
    ∀ (xs zs ys ws : List Char),
      '#' ∉ xs → '#' ∉ zs →
      xs ++ ('#' :: ys) = zs ++ ('#' :: ws) →
      xs = zs ∧ ys = ws
  | [], [], ys, ws, _, _, h => by
      rw [List.nil_append, List.nil_append, List.cons.injEq] at h
      exact ⟨rfl, h.2⟩
  | [], (z :: zs), ys, ws, _, hz, h => by
      rw [List.nil_append, List.cons_append, List.cons.injEq] at h
      have hz_eq : z = '#' := h.1.symm
      exact absurd (hz_eq ▸ List.mem_cons_self (a := z) (l := zs)) hz
  | (x :: xs), [], ys, ws, hx, _, h => by
      rw [List.cons_append, List.nil_append, List.cons.injEq] at h
      exact absurd (h.1 ▸ List.mem_cons_self (a := x) (l := xs)) hx
  | (x :: xs), (z :: zs), ys, ws, hx, hz, h => by
      rw [List.cons_append, List.cons_append, List.cons.injEq] at h
      obtain ⟨hxz, htail⟩ := h
      have hx' : '#' ∉ xs := fun k => hx (List.mem_cons_of_mem _ k)
      have hz' : '#' ∉ zs := fun k => hz (List.mem_cons_of_mem _ k)
      have ih := frame_inj xs zs ys ws hx' hz' htail
      refine ⟨?_, ih.2⟩
      rw [hxz, ih.1]

/-- Replicating `'x'` never introduces a `'#'`. -/
theorem notHash_replicate (n : Nat) : '#' ∉ List.replicate n 'x' := by
  induction n with
  | zero => simp
  | succ k ih =>
      intro hmem
      rw [List.replicate_succ, List.mem_cons] at hmem
      cases hmem with
      | inl h => exact absurd h (by decide)
      | inr h => exact ih h

/-- Underlying `toList` form of `freshName`. -/
theorem freshName_toList (base : String) (n : Nat) :
    (freshName base n).toList = base.toList ++ '#' :: List.replicate n 'x' := by
  unfold freshName
  exact String.toList_ofList

/-- Every `freshName` contains the separator `'#'`, so it is never
    hash-free. -/
theorem freshName_not_noHash (base : String) (n : Nat) :
    ¬ NoHash (freshName base n) := by
  intro h
  have hmem : '#' ∈ (freshName base n).toList := by
    rw [freshName_toList]
    simp
  exact h hmem

/-- A hash-free source name can never coincide with a generated
    `freshName`. -/
theorem freshName_ne_of_noHash_name
    (base x : String) (n : Nat)
    (hx : NoHash x) :
    freshName base n ≠ x := by
  intro hEq
  have : NoHash (freshName base n) := by simpa [hEq] using hx
  exact freshName_not_noHash base n this

/-- Primary injectivity: if two `freshName`s over the same `base` are
    equal, their counters are equal. -/
theorem freshName_injective_suffix (base : String) (n m : Nat)
    (h : freshName base n = freshName base m) : n = m := by
  have hd : (freshName base n).toList = (freshName base m).toList := by
    rw [h]
  rw [freshName_toList, freshName_toList] at hd
  have hcons : ('#' :: List.replicate n 'x') = ('#' :: List.replicate m 'x') :=
    append_left_cancel base.toList _ _ hd
  have hreps : List.replicate n 'x' = List.replicate m 'x' :=
    List.tail_eq_of_cons_eq hcons
  exact replicate_length_eq 'x' n m hreps

/-- Cross-`Nat` disequality: distinct counters give distinct names. -/
theorem freshName_ne_of_nat_ne (base : String) (n m : Nat) (h : n ≠ m) :
    freshName base n ≠ freshName base m := by
  intro heq
  exact h (freshName_injective_suffix base n m heq)

/-- Cross-base disequality: distinct hash-free bases give distinct names
    regardless of counter values. -/
theorem freshName_ne_of_base_ne (b1 b2 : String) (n m : Nat)
    (hb1 : NoHash b1) (hb2 : NoHash b2) (h_bases : b1 ≠ b2) :
    freshName b1 n ≠ freshName b2 m := by
  intro heq
  have hd : (freshName b1 n).toList = (freshName b2 m).toList := by rw [heq]
  rw [freshName_toList, freshName_toList] at hd
  have hframe := frame_inj b1.toList b2.toList
                   (List.replicate n 'x') (List.replicate m 'x')
                   hb1 hb2 hd
  apply h_bases
  -- Promote `b1.toList = b2.toList` back to `b1 = b2`.
  have hs : String.ofList b1.toList = String.ofList b2.toList := by
    rw [hframe.1]
  rw [String.ofList_toList, String.ofList_toList] at hs
  exact hs

/-- Same-counter distinct-base corollary. -/
theorem freshName_distinct_bases (b1 b2 : String) (n : Nat)
    (hb1 : NoHash b1) (hb2 : NoHash b2) (h_bases : b1 ≠ b2)
    (h : freshName b1 n = freshName b2 n) : False :=
  freshName_ne_of_base_ne b1 b2 n n hb1 hb2 h_bases h

end LaCaDiLE
