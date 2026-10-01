#!/usr/bin/env python3
"""Install the three hand proofs for Chelis's permutation gate.

Vermilion generates the surrounding Lean declarations from the exact Rust
source. These tactics are checked by Lean; no theorem or assumption is added.
The statement hashes prevent silently applying a proof to a changed contract.
"""

from pathlib import Path
import sys


PROOFS = {
    ("axis_perm.is_permutation.invariant_preserve_1_6", "9b0860b7dae748a8"): """\
  vrml_norm
  have hj : 0 ≤ j + 1 ∧ j + 1 < 18446744073709551616 := by omega
  rw [Int.emod_eq_of_lt hj.1 hj.2]
  intro k hk
  by_cases h : k < j
  · exact loop_1_iteration_10 k ⟨hk.1, h⟩
  · have hkj : k = j := by omega
    subst k
    exact then_2_assume_15""",
    ("axis_perm.is_permutation.invariant_preserve_0_2", "e5a357a0d59eb383"): """\
  vrml_norm
  have hi : 0 ≤ i + 1 ∧ i + 1 < 18446744073709551616 := by omega
  rw [Int.emod_eq_of_lt hi.1 hi.2]
  intro k hk
  by_cases h : k < i
  · exact loop_1_exit_5 k ⟨hk.1, h⟩
  · have hki : k = i := by omega
    subst k
    exact loop_1_exit_4""",
    ("axis_perm.is_permutation.invariant_preserve_0_3", "db5710a38ecd279b"): """\
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
    exact loop_1_exit_7 a ha""",
}


def main(path: Path) -> None:
    source = path.read_text()
    for (name, digest), proof in PROOFS.items():
        begin = f"-- vrml:begin {name} {digest}"
        end = f"-- vrml:end {name}"
        assert source.count(begin) == 1, f"missing or changed Lean obligation: {begin}"
        start = source.index(begin)
        stop = source.index(end, start)
        block = source[start:stop]
        if "  sorry" in block:
            assert block.count("  sorry") == 1, name
            block = block.replace("  sorry", proof)
            source = source[:start] + block + source[stop:]
        else:
            assert proof in block or "  vrml [" in block, f"unexpected existing proof for {name}"
    path.write_text(source)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
