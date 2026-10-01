#!/usr/bin/env python3
"""Install kernel-checked Lean proofs for the full Chelis axis kernel.

The input twin is generated from byte-identical production Rust source.
The statement hashes reject stale manual proofs after a contract change.
"""

from pathlib import Path
import re
import sys


HELPER = """\
-- vrml:user:begin
private theorem chelis_rank_signed (rank : Int)
    (hr : Vermilion.inUnsignedRange 64 rank)
    (hguard : ((rank % 340282366920938463463374607431768211456) >
      (9223372036854775807 % 340282366920938463463374607431768211456)) → False) :
    0 ≤ rank ∧ rank ≤ 9223372036854775807 ∧ Vermilion.sclip 64 rank = rank := by
  have h128 : rank % 340282366920938463463374607431768211456 = rank :=
    Int.emod_eq_of_lt hr.1 (by rcases hr with ⟨_, h⟩; omega)
  have hmax : rank ≤ 9223372036854775807 := by
    by_contra h
    apply hguard
    simp only [h128]
    norm_num
    omega
  refine ⟨hr.1, hmax, ?_⟩
  simp only [Vermilion.sclip_eq, Vermilion.iteP]
  split_ifs <;> omega
-- vrml:user:end

"""


PROOFS = {
    ('verified.is_permutation.invariant_preserve_1_6', '9b0860b7dae748a8'): """\
  vrml_norm
  have hj : 0 ≤ j + 1 ∧ j + 1 < 18446744073709551616 := by omega
  rw [Int.emod_eq_of_lt hj.1 hj.2]
  intro k hk
  by_cases h : k < j
  · exact loop_1_iteration_10 k ⟨hk.1, h⟩
  · have hkj : k = j := by omega
    subst k
    exact then_2_assume_15

""",
    ('verified.is_permutation.invariant_preserve_0_2', 'e5a357a0d59eb383'): """\
  vrml_norm
  have hi : 0 ≤ i + 1 ∧ i + 1 < 18446744073709551616 := by omega
  rw [Int.emod_eq_of_lt hi.1 hi.2]
  intro k hk
  by_cases h : k < i
  · exact loop_1_exit_5 k ⟨hk.1, h⟩
  · have hki : k = i := by omega
    subst k
    exact loop_1_exit_4

""",
    ('verified.is_permutation.invariant_preserve_0_3', 'db5710a38ecd279b'): """\
  vrml_norm
  have hi : 0 ≤ i + 1 ∧ i + 1 < 18446744073709551616 := by omega
  rw [Int.emod_eq_of_lt hi.1 hi.2]
  intro a b hab
  by_cases h : b < i
  · exact loop_1_exit_6 a b ⟨hab.1, h⟩
  · have hbi : b = i := by omega
    subst b
    have hji : j = i := by omega
    have ha : 0 ≤ a ∧ a < j := by omega
    exact loop_1_exit_7 a ha

""",
    ('verified.normalize_axis.assert_1', '03f61d9554aa0988'): """\
  obtain ⟨hr0, hrmax, hclip⟩ := chelis_rank_signed rank requires_0 then_0_assume_3
  rw [hclip]
  unfold Vermilion.inSignedRange
  omega

""",
    ('verified.normalize_axis.ensures_2_0', 'd39edad90c28de71'): """\
  obtain ⟨hr0, hrmax, hclip⟩ := chelis_rank_signed rank requires_0 then_0_assume_3
  have hnegclip : Vermilion.sclip 64 (0 - rank) = 0 - rank := by
    simp only [Vermilion.sclip_eq, Vermilion.iteP]
    split_ifs <;> omega
  have hraw : raw < -rank := by
    rw [hclip, hnegclip] at branch_2
    omega
  simp [verified.normalized_axis_model, Vermilion.iteP, hrmax, branch_1]
  omega

""",
    ('verified.normalize_axis.assert_3', '3d524b5807f48a5b'): """\
  obtain ⟨hr0, hrmax, hclip⟩ := chelis_rank_signed rank requires_0 then_0_assume_3
  have hnegclip : Vermilion.sclip 64 (0 - rank) = 0 - rank := by
    simp only [Vermilion.sclip_eq, Vermilion.iteP]
    split_ifs <;> omega
  have hlow : -rank ≤ raw := by
    by_contra h
    apply then_2_assume_7
    rw [hclip, hnegclip]
    omega
  rw [hclip]
  unfold Vermilion.inSignedRange
  omega

""",
    ('verified.normalize_axis.ensures_4_0', '243349815a97efe4'): """\
  obtain ⟨hr0, hrmax, hclip⟩ := chelis_rank_signed rank requires_0 then_0_assume_3
  have hnegclip : Vermilion.sclip 64 (0 - rank) = 0 - rank := by
    simp only [Vermilion.sclip_eq, Vermilion.iteP]
    split_ifs <;> omega
  have hlow : -rank ≤ raw := by
    by_contra h
    apply then_2_assume_7
    rw [hclip, hnegclip]
    omega
  have hsum : 0 ≤ raw + rank ∧ raw + rank < rank := by omega
  have hsumclip : Vermilion.sclip 64 (raw + Vermilion.sclip 64 rank) = raw + rank := by
    rw [hclip]
    simp only [Vermilion.sclip_eq, Vermilion.iteP]
    split_ifs <;> omega
  rw [hsumclip]
  unfold verified.normalized_axis_model
  simp [Vermilion.iteP, hrmax, branch_1, hsum]

""",
    ('verified.normalize_axis.ensures_4_1', 'f94686916f565d60'): """\
  obtain ⟨hr0, hrmax, hclip⟩ := chelis_rank_signed rank requires_0 then_0_assume_3
  have hnegclip : Vermilion.sclip 64 (0 - rank) = 0 - rank := by
    simp only [Vermilion.sclip_eq, Vermilion.iteP]
    split_ifs <;> omega
  have hlow : -rank ≤ raw := by
    by_contra h
    apply then_2_assume_7
    rw [hclip, hnegclip]
    omega
  have hsum : 0 ≤ raw + rank ∧ raw + rank < rank := by omega
  have hsumclip : Vermilion.sclip 64 (raw + Vermilion.sclip 64 rank) = raw + rank := by
    rw [hclip]
    simp only [Vermilion.sclip_eq, Vermilion.iteP]
    split_ifs <;> omega
  rw [hsumclip]
  have hmod : (raw + rank) % 18446744073709551616 = raw + rank :=
    Int.emod_eq_of_lt hsum.1 (by omega)
  simp [hmod, hsum.2]

""",
    ('verified.normalize_axis.ensures_5_0', '75d2864fa95946df'): """\
  obtain ⟨hr0, hrmax, hclip⟩ := chelis_rank_signed rank requires_0 then_0_assume_3
  have hraw : 0 ≤ raw := by
    by_contra h
    exact then_1_assume_9 (by omega)
  have hnraw : ¬ raw < 0 := by omega
  have hnrank : ¬ rank > 9223372036854775807 := by omega
  rw [hclip]
  unfold verified.normalized_axis_model
  simp [Vermilion.iteP, hnrank, hnraw, hraw]
  by_cases h : raw < rank <;> simp [h]

""",
    ('verified.normalize_axis.ensures_5_1', 'ebf0975187481270'): """\
  obtain ⟨hr0, hrmax, hclip⟩ := chelis_rank_signed rank requires_0 then_0_assume_3
  have hraw : 0 ≤ raw := by
    by_contra h
    exact then_1_assume_9 (by omega)
  rw [hclip]
  simp only [Vermilion.iteP]
  by_cases h : rank ≤ raw
  · simp [h]
  · have hlt : raw < rank := by omega
    have hmod : raw % 18446744073709551616 = raw :=
      Int.emod_eq_of_lt hraw (by omega)
    simpa [h, hmod] using hlt

""",
    ('verified.reduction_survivors.invariant_preserve_0_2', 'c6877216856ed806'): """\
  vrml_norm
  rw [Int.emod_eq_of_lt assert_3.1 assert_3.2]
  by_cases h : i = axis
  · have hp : positions_2 = positions := else_1_assume_13 (by simp [h])
    rw [hp, loop_0_iteration_9]
    simp [h]
  · have hp : positions_2 = Vermilion.Seq.push positions i := by
      exact (then_1_assume_16 h).trans (then_1_call_push_ensures_0 h)
    rw [hp, Vermilion.Seq.len_push, loop_0_iteration_9]
    split_ifs <;> omega

""",
    ('verified.reduction_survivors.invariant_preserve_0_3', 'e1a09664e344d1b9'): """\
  vrml_norm
  intro k hk
  by_cases h : i = axis
  · have hp : positions_2 = positions := else_1_assume_13 (by simp [h])
    rw [hp] at hk ⊢
    exact loop_0_iteration_10 k hk
  · have hp : positions_2 = Vermilion.Seq.push positions i := by
      exact (then_1_assume_16 h).trans (then_1_call_push_ensures_0 h)
    rw [hp, Vermilion.Seq.len_push] at hk
    rw [hp]
    by_cases hlt : k < Vermilion.Seq.len positions
    · rw [Vermilion.Seq.index_push_prefix positions i k hk.1 hlt]
      exact loop_0_iteration_10 k ⟨hk.1, hlt⟩
    · have heq : k = Vermilion.Seq.len positions := by omega
      rw [heq, Vermilion.Seq.index_push_last]
      simp only [verified.survivor_at, Vermilion.iteP]
      split_ifs <;> omega

""",
    ('verified.checked_inverse.invariant_preserve_1_2', '2e6609d707f46596'): """\
  rw [call_vec_index_mut_ensures_2]
  simpa using loop_1_iteration_14

""",
    ('verified.checked_inverse.invariant_preserve_1_3', '06d0c4d17f7a46ec'): """\
  have hnext : (i + 1) % 18446744073709551616 = i + 1 :=
    Int.emod_eq_of_lt assert_7.1 assert_7.2
  rw [hnext]
  intro k hk
  by_cases hki : k = i
  · subst k
    have heq : Vermilion.Seq.index tmp__post_2 (Vermilion.Seq.index axes i) = i := by
      rw [← call_vec_index_mut_ensures_3]
      exact assume_25
    have himod : i % 18446744073709551616 = i :=
      Int.emod_eq_of_lt (by exact loop_1_iteration_11.1) (by exact loop_1_iteration_11.2)
    simpa [himod] using heq
  · have hklt : k < i := by omega
    unfold verified.valid_permutation at loop_1_iteration_13
    obtain ⟨⟨_, _⟩, hdistinct⟩ := loop_1_iteration_13
    have hneq : Vermilion.Seq.index axes i ≠ Vermilion.Seq.index axes k := by
      exact Ne.symm (hdistinct k i ⟨⟨hk.1, hklt⟩, by omega⟩)
    rw [call_vec_index_mut_ensures_2]
    rw [Vermilion.Seq.index_update_other inverse_2 (Vermilion.Seq.index axes k) tmp__post_3 (Vermilion.Seq.index axes i) hneq]
    exact loop_1_iteration_15 k ⟨hk.1, hklt⟩

""",
}

def main(path: Path) -> None:
    source = path.read_text()
    if source.count("-- vrml:begin ") != 75:
        raise SystemExit("expected 75 full-kernel Lean obligations")
    marker = "namespace verified.is_permutation\n"
    if source.count(marker) != 1:
        raise SystemExit("cannot locate the exact full-kernel proof namespace")
    if "-- vrml:user:begin" not in source:
        source = source.replace(marker, HELPER + "\n" + marker, 1)
    else:
        existing = source.split("-- vrml:user:begin", 1)[1].split("-- vrml:user:end", 1)[0]
        expected = HELPER.split("-- vrml:user:begin", 1)[1].split("-- vrml:user:end", 1)[0]
        if existing.strip() != expected.strip():
            raise SystemExit("the preserved Lean helper differs from the checked helper")
    for (name, digest), proof in PROOFS.items():
        begin = f"-- vrml:begin {name} {digest}"
        end = f"-- vrml:end {name}"
        if source.count(begin) != 1 or source.count(end) != 1:
            raise SystemExit(f"missing or changed Lean obligation: {begin}")
        start = source.index(begin)
        stop = source.index(end, start)
        block = source[start:stop]
        signature = block.index("@[vrml_obligation] theorem")
        body_start = block.index(":= by\n", signature) + len(":= by\n")
        source = source[: start + body_start] + proof + source[stop:]
    if re.search(r"^\s*sorry\s*$", source, re.MULTILINE):
        raise SystemExit("full-kernel proof twin still contains sorry")
    path.write_text(source)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
