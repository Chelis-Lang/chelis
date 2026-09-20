//! chelis#2205: the consuming string mutator.
//!
//! `chelis_string_concat_owned` takes ownership of its left-hand operand. A
//! uniquely owned string is extended in place and comes back as the same
//! handle; a shared one is left untouched, a fresh string is returned exactly
//! as `chelis_string_concat` builds it, and the consumed input loses the one
//! owner the caller gave up. The static half of the contract (no un-retained
//! reference survives) is the ownership verifier's; this file locks the
//! runtime half, as `list_owned_mutators.rs` and `dict_owned_mutators.rs` do
//! for the other two kinds.
//!
//! `RuntimeString` is the first kind this optimisation mutates that carries
//! derived state, so most of this file is about the two derived fields rather
//! than about ownership. `char_count` serves the character-indexed
//! `chelis_string_len` and decides `chelis_string_slice`'s strategy, and
//! `nul_terminated` serves `chelis_string_data`. A mutation that updated
//! `value` alone would leave both wrong in ways no ASCII fixture notices, so
//! the multibyte rows below are the ones that carry this file.
use chelis_runtime::{
    chelis_string, chelis_string_concat, chelis_string_concat_owned, chelis_string_data,
    chelis_string_eq, chelis_string_from_utf8, chelis_string_len, chelis_string_release,
    chelis_string_retain, chelis_string_slice,
};

unsafe fn text(value: &str) -> chelis_string {
    chelis_string_from_utf8(value.as_ptr(), value.len() as i64)
}

/// The address inside a `chelis_string`, so a test can say whether the same
/// allocation came back.
///
/// `chelis_string` is a `repr(C)` struct holding one handle, and its field is
/// public, but `RuntimeString` is crate-private so the field's type cannot be
/// used from here at all. Reading the carrier's one pointer-sized word is the
/// way to compare identity without naming that type; the assertion below
/// keeps the read honest if the carrier ever gains a second field.
const _: () = assert!(
    std::mem::size_of::<chelis_string>() == std::mem::size_of::<usize>(),
    "chelis_string is one pointer-sized word"
);

unsafe fn handle_of(value: chelis_string) -> usize {
    std::mem::transmute_copy::<chelis_string, usize>(&value)
}

/// The bytes `chelis_string_data` serves, up to its terminator, which is what
/// a C consumer of the published ABI actually reads.
unsafe fn served_bytes(value: chelis_string) -> Vec<u8> {
    let pointer = chelis_string_data(value).cast::<u8>();
    let mut out = Vec::new();
    let mut offset = 0;
    loop {
        let byte = *pointer.add(offset);
        if byte == 0 {
            break;
        }
        out.push(byte);
        offset += 1;
    }
    out
}

/// Assert every observable of `value` at once: its value equality, its
/// character-indexed length, and the bytes its interior pointer serves.
unsafe fn assert_observables(value: chelis_string, expected: &str, label: &str) {
    let want = text(expected);
    assert!(
        chelis_string_eq(value, want),
        "{label}: value differs from {expected:?}"
    );
    chelis_string_release(want);
    assert_eq!(
        chelis_string_len(value),
        expected.chars().count() as i64,
        "{label}: character-indexed length"
    );
    assert_eq!(
        served_bytes(value),
        expected.as_bytes().to_vec(),
        "{label}: bytes served through the interior pointer"
    );
}

#[test]
fn unique_concat_owned_extends_in_place_and_returns_the_same_handle() {
    unsafe {
        let lhs = text("abc");
        let rhs = text("de");
        let handle = handle_of(lhs);
        let grown = chelis_string_concat_owned(lhs, rhs);
        assert!(
            handle_of(grown) == handle,
            "a uniquely owned string is extended in place"
        );
        assert_observables(grown, "abcde", "unique in-place concat");
        chelis_string_release(rhs);
        chelis_string_release(grown);
    }
}

#[test]
fn shared_concat_owned_clones_and_leaves_the_shared_view_untouched() {
    unsafe {
        let lhs = text("abc");
        // A second owner: the observable source of the alias controls.
        chelis_string_retain(lhs);
        let rhs = text("de");
        let fresh = chelis_string_concat_owned(lhs, rhs);
        assert!(
            handle_of(fresh) != handle_of(lhs),
            "a shared string is never mutated in place"
        );
        assert_observables(lhs, "abc", "the shared view");
        assert_observables(fresh, "abcde", "the cloned result");
        // The consumed owner was released: exactly one strong owner is left,
        // so this release must free the source without a ledger complaint.
        chelis_string_release(lhs);
        chelis_string_release(rhs);
        chelis_string_release(fresh);
    }
}

/// The string counterpart of RT-2225's P0: a right-hand side that aliases the
/// consumed left-hand side is a retained second owner, so it takes the
/// cloning path rather than reading a buffer it is extending.
#[test]
fn concat_owned_with_an_aliasing_rhs_clones_instead_of_reading_what_it_writes() {
    unsafe {
        let value = text("ab");
        chelis_string_retain(value);
        let doubled = chelis_string_concat_owned(value, value);
        assert!(
            handle_of(doubled) != handle_of(value),
            "an aliasing rhs never extends in place"
        );
        assert_observables(doubled, "abab", "aliased concat");
        assert_observables(value, "ab", "the shared view is untouched");
        chelis_string_release(value);
        chelis_string_release(doubled);
    }
}

/// The aliasing check is not redundant with the uniqueness check, and this is
/// the row that shows it.
///
/// `concat_owned_with_an_aliasing_rhs_clones_instead_of_reading_what_it_writes`
/// above retains the string, so the strong count rejects the in-place arm
/// before the aliasing test is ever consulted, and deleting the aliasing test
/// leaves that row passing. The shape that reaches it is a sole owner passing
/// its own handle as both operands, which this entry point's C signature
/// admits even though the scheduler never emits it: an application whose
/// other operand names the same owner is refused the upgrade. In-place here
/// would have `push_str` read the buffer it is reallocating.
///
/// Evidentiary status: REGRESSION TEST, proven failing first by replacing the
/// aliasing test with `false`, which leaves the other eight rows green.
#[test]
fn a_sole_owner_passing_itself_as_both_operands_takes_the_cloning_path() {
    unsafe {
        let value = text("ab");
        // Exactly one strong owner, and both operands are that owner. The
        // call consumes it, so the result is the only string left here.
        let doubled = chelis_string_concat_owned(value, value);
        assert_observables(doubled, "abab", "sole owner concatenated with itself");
        chelis_string_release(doubled);
    }
}

/// The derived fields are why this file exists. An in-place append that
/// updated `value` alone leaves `char_count` short and `nul_terminated`
/// un-terminated, and no ASCII fixture notices the first because ASCII makes
/// bytes and characters agree.
///
/// Evidentiary status: REGRESSION TESTS, each proven failing first by
/// deleting the one maintenance line it covers. Dropping the `char_count`
/// update makes `chelis_string_len` report 3 where the string holds 6
/// characters, and makes `chelis_string_slice` take the ASCII branch on a
/// multibyte string. Dropping the `nul_terminated` rebuild makes
/// `chelis_string_data` serve the pre-concat bytes.
#[test]
fn in_place_concat_keeps_the_character_count_exact_across_multibyte_bytes() {
    unsafe {
        let lhs = text("héllo");
        let rhs = text(" wörld");
        let grown = chelis_string_concat_owned(lhs, rhs);
        assert_observables(grown, "héllo wörld", "multibyte in-place concat");
        assert_eq!(
            chelis_string_len(grown),
            11,
            "eleven Unicode scalar values, not thirteen bytes"
        );
        // `chelis_string_slice` reads `char_count` to decide whether byte
        // indices are character indices. A stale count sends a multibyte
        // string down the ASCII branch and slices mid-character.
        let middle = chelis_string_slice(grown, 6, 5);
        assert_observables(middle, "wörld", "slice of the grown string");
        chelis_string_release(middle);
        chelis_string_release(rhs);
        chelis_string_release(grown);
    }
}

#[test]
fn in_place_concat_is_exact_when_only_one_side_is_multibyte() {
    unsafe {
        for (left, right, expected) in [
            ("ascii", "→", "ascii→"),
            ("→", "ascii", "→ascii"),
            ("", "→←", "→←"),
            ("→←", "", "→←"),
        ] {
            let lhs = text(left);
            let rhs = text(right);
            let grown = chelis_string_concat_owned(lhs, rhs);
            assert_observables(grown, expected, &format!("{left:?} + {right:?}"));
            chelis_string_release(rhs);
            chelis_string_release(grown);
        }
    }
}

/// The accumulation pattern this row exists for: repeated in-place extension
/// of one owner. Each step must leave every observable exact, not just the
/// last one.
#[test]
fn repeated_in_place_concats_accumulate_exactly() {
    unsafe {
        let mut accumulated = text("");
        let mut expected = String::new();
        for step in 0..64 {
            let piece = if step % 3 == 0 {
                "é"
            } else if step % 3 == 1 {
                "xy"
            } else {
                "→"
            };
            let rhs = text(piece);
            accumulated = chelis_string_concat_owned(accumulated, rhs);
            chelis_string_release(rhs);
            expected.push_str(piece);
            assert_observables(accumulated, &expected, &format!("after step {step}"));
        }
        chelis_string_release(accumulated);
    }
}

#[test]
fn an_empty_right_hand_side_leaves_the_owner_unchanged() {
    unsafe {
        let lhs = text("abc");
        let handle = handle_of(lhs);
        let rhs = text("");
        let grown = chelis_string_concat_owned(lhs, rhs);
        assert_eq!(handle_of(grown), handle);
        assert_observables(grown, "abc", "empty right-hand side");
        chelis_string_release(rhs);
        chelis_string_release(grown);
    }
}

#[test]
fn owned_concat_agrees_with_the_cloning_concat() {
    unsafe {
        for (left, right) in [
            ("", ""),
            ("a", ""),
            ("", "b"),
            ("héllo", " wörld"),
            ("→←", "↑↓"),
        ] {
            let cloning_lhs = text(left);
            let rhs = text(right);
            let cloned = chelis_string_concat(cloning_lhs, rhs);
            let owned_lhs = text(left);
            let owned = chelis_string_concat_owned(owned_lhs, rhs);
            assert!(
                chelis_string_eq(cloned, owned),
                "{left:?} + {right:?}: owned and cloning results differ"
            );
            assert_eq!(
                chelis_string_len(cloned),
                chelis_string_len(owned),
                "{left:?} + {right:?}: character-indexed lengths differ"
            );
            assert_eq!(
                served_bytes(cloned),
                served_bytes(owned),
                "{left:?} + {right:?}: served bytes differ"
            );
            chelis_string_release(cloning_lhs);
            chelis_string_release(rhs);
            chelis_string_release(cloned);
            chelis_string_release(owned);
        }
    }
}
