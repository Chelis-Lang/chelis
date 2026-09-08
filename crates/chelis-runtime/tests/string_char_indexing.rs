//! `chelis_string_len` and `chelis_string_slice` are CHARACTER-indexed.
//!
//! Both were rewritten for cost: `chelis_string_len` reads a count taken
//! once at construction instead of re-decoding on every call, and
//! `chelis_string_slice` converts a character range to a byte range with
//! `char_indices()` (taking an O(1) all-ASCII shortcut) instead of
//! collecting the whole string into a `Vec<char>` on every call.
//!
//! The cheap way to write either of those is byte indexing, which is
//! indistinguishable from the correct implementation on ASCII and silently
//! wrong on everything else. These tests pin the character semantics on
//! 2-byte (Latin-1 supplement), 3-byte (CJK), and 4-byte (emoji, astral
//! plane) encodings, and on a combining sequence — `string_len` counts
//! Unicode scalar values, not grapheme clusters, so `"e\u{301}"` is two
//! characters and slicing splits it.
//!
//! Proven to fire: replacing the body of `chelis_string_slice` with
//! `String::from_utf8_lossy(&text.as_bytes()[start..end])` and
//! `chelis_string_len` with `value.len()` — the plausible byte-indexed
//! shape, chosen because it returns wrong data instead of panicking — fails
//! every multi-byte case here while leaving the ASCII cases green.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use chelis_runtime::{
    chelis_string, chelis_string_data, chelis_string_from_cstr, chelis_string_len,
    chelis_string_release, chelis_string_slice,
};

/// Build a runtime string from Rust text at the real FFI boundary.
fn make(text: &str) -> chelis_string {
    let owned = CString::new(text).expect("test inputs contain no interior NUL");
    unsafe { chelis_string_from_cstr(owned.as_ptr() as *const c_char) }
}

fn read(value: chelis_string) -> String {
    unsafe {
        CStr::from_ptr(chelis_string_data(value))
            .to_str()
            .expect("runtime strings are UTF-8")
            .to_owned()
    }
}

/// `slice(text, start, len)` through the FFI, returned as Rust text.
fn slice(text: &str, start: i64, len: i64) -> String {
    unsafe {
        let source = make(text);
        let out = chelis_string_slice(source, start, len);
        let rendered = read(out);
        chelis_string_release(out);
        chelis_string_release(source);
        rendered
    }
}

fn char_len(text: &str) -> i64 {
    unsafe {
        let source = make(text);
        let n = chelis_string_len(source);
        chelis_string_release(source);
        n
    }
}

/// The reference semantics, stated independently of the implementation.
fn expected_slice(text: &str, start: usize, len: usize) -> String {
    text.chars().skip(start).take(len).collect()
}

#[test]
fn string_len_counts_characters_not_bytes() {
    assert_eq!(char_len(""), 0);
    assert_eq!(char_len("hello"), 5);
    // 2-byte: each accented letter is one character, two bytes.
    assert_eq!(char_len("café"), 4, "'é' is 2 bytes but 1 character");
    assert_eq!(
        "café".len(),
        5,
        "byte length differs, so this case discriminates"
    );
    // 3-byte CJK.
    assert_eq!(char_len("日本語"), 3);
    assert_eq!("日本語".len(), 9);
    // 4-byte astral plane.
    assert_eq!(char_len("🙂🙃"), 2);
    assert_eq!("🙂🙃".len(), 8);
    // Mixed widths in one string.
    assert_eq!(char_len("aé日🙂"), 4);
    assert_eq!("aé日🙂".len(), 10);
    // Scalar values, not graphemes: "e" + COMBINING ACUTE ACCENT.
    assert_eq!(char_len("e\u{301}"), 2);
}

#[test]
fn slice_indexes_by_character_on_ascii() {
    assert_eq!(slice("hello", 0, 1), "h");
    assert_eq!(slice("hello", 1, 3), "ell");
    assert_eq!(slice("hello", 4, 1), "o");
    assert_eq!(slice("hello", 0, 5), "hello");
}

#[test]
fn slice_indexes_by_character_on_two_byte_encodings() {
    assert_eq!(slice("café", 3, 1), "é", "index 3 is the 4th CHARACTER");
    assert_eq!(slice("café", 2, 2), "fé");
    assert_eq!(slice("café", 0, 4), "café");
    assert_eq!(slice("café", 1, 1), "a");
    // Every accented character, one at a time.
    assert_eq!(slice("àéîõü", 2, 1), "î");
    assert_eq!(slice("àéîõü", 0, 3), "àéî");
}

#[test]
fn slice_indexes_by_character_on_three_byte_encodings() {
    assert_eq!(slice("日本語", 0, 1), "日");
    assert_eq!(slice("日本語", 1, 1), "本");
    assert_eq!(slice("日本語", 2, 1), "語");
    assert_eq!(slice("日本語", 0, 2), "日本");
    assert_eq!(slice("日本語", 1, 2), "本語");
}

#[test]
fn slice_indexes_by_character_on_four_byte_encodings() {
    assert_eq!(slice("a🙂b", 1, 1), "🙂");
    assert_eq!(
        slice("a🙂b", 2, 1),
        "b",
        "the character after a 4-byte scalar"
    );
    assert_eq!(slice("a🙂b", 0, 2), "a🙂");
    assert_eq!(slice("🙂🙃😀", 1, 2), "🙃😀");
}

#[test]
fn slice_indexes_by_character_across_mixed_widths() {
    let text = "aé日🙂z";
    let expected = ["a", "é", "日", "🙂", "z"];
    for (index, want) in expected.iter().enumerate() {
        assert_eq!(
            slice(text, index as i64, 1),
            *want,
            "character at index {index}"
        );
    }
    assert_eq!(slice(text, 1, 3), "é日🙂");
}

#[test]
fn slice_splits_a_combining_sequence_because_indexing_is_by_scalar_value() {
    // Not grapheme clusters: "é" written as e + U+0301 is two characters and
    // the two halves are individually addressable.
    assert_eq!(slice("e\u{301}", 0, 1), "e");
    assert_eq!(slice("e\u{301}", 1, 1), "\u{301}");
    assert_eq!(slice("e\u{301}", 0, 2), "e\u{301}");
}

#[test]
fn slice_boundary_and_out_of_range_behavior_is_unchanged() {
    // Zero length is empty, at any in-range start.
    assert_eq!(slice("日本語", 0, 0), "");
    assert_eq!(slice("日本語", 2, 0), "");
    // A start at or past the end is empty, not a panic.
    assert_eq!(slice("日本語", 3, 1), "");
    assert_eq!(slice("日本語", 99, 1), "");
    assert_eq!(slice("", 0, 1), "");
    // A length past the end truncates to the remaining characters.
    assert_eq!(slice("日本語", 1, 99), "本語");
    assert_eq!(slice("café", 2, 1000), "fé");
    // Negative arguments are empty.
    assert_eq!(slice("日本語", -1, 1), "");
    assert_eq!(slice("日本語", 0, -1), "");
    // Saturating, not wrapping, at the extremes.
    assert_eq!(slice("日本語", 0, i64::MAX), "日本語");
    assert_eq!(slice("日本語", i64::MAX, 1), "");
}

#[test]
fn every_character_range_matches_the_reference_semantics() {
    // Exhaustive over both endpoints for a string that mixes all four UTF-8
    // widths, so an off-by-one at any boundary shows up.
    let cases = ["", "abc", "café", "日本語", "a🙂b", "aé日🙂z", "e\u{301}x"];
    for text in cases {
        let n = text.chars().count();
        for start in 0..=n + 2 {
            for len in 0..=n + 2 {
                assert_eq!(
                    slice(text, start as i64, len as i64),
                    expected_slice(text, start, len),
                    "slice({text:?}, {start}, {len})"
                );
            }
        }
        assert_eq!(char_len(text), n as i64, "len({text:?})");
    }
}

#[test]
fn single_character_slices_reassemble_the_original() {
    // The scanner shape `parse_line_chars` uses: walk the string one
    // character at a time and concatenate. Byte indexing would produce
    // replacement characters or split scalars here.
    for text in ["hello", "café", "日本語", "a🙂b", "aé日🙂z"] {
        let n = char_len(text);
        let mut rebuilt = String::new();
        for index in 0..n {
            rebuilt.push_str(&slice(text, index, 1));
        }
        assert_eq!(rebuilt, text, "reassembled {text:?}");
    }
}
