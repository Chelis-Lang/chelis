//! Unit coverage for the chelis#729 Phase 0 domain-validity checker
//! (`common::assert_elements_in_domain` and friends): positive AND
//! negative membership per dtype family, per the repo's
//! negative-test-parity rule.
//!
//! The checker implements ONLY the value-set column of the §C1 table in
//! `spec/design/dtype_semantics.md` over printed lane output. Its API is
//! frozen at Phase 0 exit; these tests are the executable statement of
//! its membership semantics, including the documented own-width shortest
//! form. Print-truncation slack is retired: membership is exact.
//!
//! New cells the checker exposes that the audit did not enumerate get
//! their own `#[ignore]`d rows in the matrix files with a fresh issue
//! link (dtype_semantics.md §B2.4: discoveries fork, they do not
//! scope-creep). None were known at the time this file landed.

#[path = "common/mod.rs"]
mod common;

use chelis_types::observation::{ElementRef, format_element};
use chelis_types::types::Prim;
use common::{
    DOMAIN_PRINT_TRUNCATION_SLACK, assert_elements_in_domain, element_domain_violation,
    printed_value_tokens,
};

fn is_member(prim: &str, token: &str) -> bool {
    element_domain_violation(prim, token).is_none()
}

// ---------------------------------------------------------------------------
// Payload extraction
// ---------------------------------------------------------------------------

#[test]
fn extracts_tensor_data_payload() {
    assert_eq!(
        printed_value_tokens("tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])"),
        vec!["1.0", "2.0", "3.0", "4.0"]
    );
    // Binding echo plus tensor form (compiled-lane shape).
    assert_eq!(
        printed_value_tokens("out = tensor(shape=[], data=[750.0])"),
        vec!["750.0"]
    );
}

#[test]
fn extracts_list_and_scalar_payloads() {
    assert_eq!(
        printed_value_tokens("[9007199254740993]"),
        vec!["9007199254740993"]
    );
    // Nested list (pad_sequences / to_list of rank-2).
    assert_eq!(
        printed_value_tokens("[[3000000000, 1], [2, 0]]"),
        vec!["3000000000", "1", "2", "0"]
    );
    assert_eq!(printed_value_tokens("out = 5"), vec!["5"]);
    assert_eq!(printed_value_tokens("false"), vec!["false"]);
    // Empty tensor payload has nothing to check.
    assert!(printed_value_tokens("tensor(shape=[0], data=[])").is_empty());
}

// ---------------------------------------------------------------------------
// Integer widths: integral values inside the width, both renderings
// ---------------------------------------------------------------------------

#[test]
fn int_members_accept_both_renderings_at_width() {
    for (prim, ok) in [
        ("int8", vec!["127", "-128", "0", "100.0"]),
        ("int16", vec!["32767", "-32768", "100.0"]),
        ("int32", vec!["2147483647", "-2147483648", "16777217"]),
        (
            "int64",
            vec![
                "9223372036854775807",
                "-9223372036854775808",
                "9007199254740993",
                // The eval tensor lane's float-formatted integers are a
                // formatting lie (chelis#732's problem), not a domain
                // violation: the VALUE is integral and in range.
                "9007199254740992.0",
                "750.0",
            ],
        ),
    ] {
        for t in ok {
            assert!(is_member(prim, t), "{t} must be in the {prim} value set");
        }
    }
}

#[test]
fn int_rejects_out_of_width_fractional_and_specials() {
    // Width escapes (the chelis#718 compiled-scalar shapes).
    assert!(!is_member("int8", "128"));
    assert!(!is_member("int8", "200"));
    assert!(!is_member("int8", "-129"));
    assert!(!is_member("int16", "60000"));
    assert!(!is_member("int32", "4000000000"));
    // int64 via double: 2^63 reads back out of range (i64::MAX is not
    // f64-representable; its double rendering rounds UP to 2^63).
    assert!(!is_member("int64", "9223372036854775808"));
    assert!(!is_member("int64", "9.223372036854776e18"));
    // Fractional in an integer buffer (the chelis#724 integer-mean shape).
    assert!(!is_member("int64", "187.5"));
    // Specials do not exist at integer dtypes.
    assert!(!is_member("int64", "inf"));
    assert!(!is_member("int64", "nan"));
    assert!(!is_member("int64", "not-a-number"));
}

#[test]
fn int64_boundary_is_exact_in_integer_rendering() {
    assert!(is_member("int64", "9223372036854775807"));
    assert!(!is_member("int64", "9223372036854775808"));
    // Float-rendered i64::MIN is exactly representable and in range.
    assert!(is_member("int64", "-9223372036854775808.0"));
}

// ---------------------------------------------------------------------------
// bool: {0, 1} only
// ---------------------------------------------------------------------------

#[test]
fn bool_members_and_rejections() {
    for t in ["true", "false", "0", "1", "0.0", "1.0"] {
        assert!(is_member("bool", t), "{t} must be in the bool value set");
    }
    for t in ["2", "-1", "0.5", "2.0", "yes"] {
        assert!(
            !is_member("bool", t),
            "{t} must NOT be in the bool value set"
        );
    }
}

// ---------------------------------------------------------------------------
// Floats: membership at the dtype's own width
// ---------------------------------------------------------------------------

#[test]
fn f64_accepts_every_parseable_value_including_specials() {
    for t in [
        "9007199254740992",
        "0.30000000000000004",
        "1.4142135623730951",
        "inf",
        "-inf",
        "nan",
        "-0.0",
    ] {
        assert!(is_member("f64", t), "{t} must be in the f64 value set");
    }
    assert!(!is_member("f64", "garbage"));
}

#[test]
fn f32_accepts_exact_widened_and_own_width_shortest_forms() {
    // Exact widened rendering (eval's f64-shortest of an f32 value).
    assert!(is_member("f32", "0.30000001192092896"));
    assert!(is_member("f32", "16777216"));
    assert!(is_member("f32", "1.4142135381698608"));
    // Own-width shortest rendering (Rust {:?} of the f32). Note this
    // branch accepts EXACTLY the {:?} form, not every near-rendering.
    assert!(is_member("f32", "0.1"));
    assert!(is_member("f32", "0.3"));
    // Specials.
    assert!(is_member("f32", "inf"));
    assert!(is_member("f32", "nan"));
}

#[test]
fn f32_rejects_the_retired_percent_g_truncation_form() {
    assert_eq!(DOMAIN_PRINT_TRUNCATION_SLACK, 0.0);
    assert!(!is_member("f32", "0.300000011920929"));
}

#[test]
fn f32_rejects_unrounded_f64_values() {
    // The chelis#717 shapes: raw f64 results in an f32 tensor. Each sits
    // ~1e-8 relative from the nearest f32. With exact-zero slack, only the
    // exact widened value or the dtype's own-width shortest form is accepted.
    assert!(!is_member("f32", "0.30000000447034836"));
    assert!(!is_member("f32", "0.3333333333333333"));
    assert!(!is_member("f32", "0.6666666666666666"));
    assert!(!is_member("f32", "1.4142135623730951"));
    // Finite value beyond the f32 finite range.
    assert!(!is_member("f32", "1e39"));
}

#[test]
fn f16_membership_at_11_bit_mantissa() {
    for t in [
        "2048",
        "2048.0",
        "0.75",
        "65504",
        "-1.5",
        "1.4140625",
        // Own-width shortest form of the stored f16 value nearest 1/3.
        "0.3333",
        // Exact widened f16 products from the locked eval-scalar rows.
        "0.0099945068359375",
        "0.333251953125",
        "inf",
    ] {
        assert!(is_member("f16", t), "{t} must be in the f16 value set");
    }
    // First non-representable integer (the chelis#717 tensor-lane shape).
    assert!(!is_member("f16", "2049"));
    assert!(!is_member("f16", "2049.0"));
    // Finite value beyond f16 max 65504 (the chelis#714 inf-collapse shape).
    assert!(!is_member("f16", "131008"));
    // The chelis#716 misprint (an f16 buffer read as f32).
    assert!(!is_member("f16", "0.0004898309707641602"));
    // A nearby decimal that merely rounds to the same f16 is not the
    // canonical own-width shortest rendering.
    assert!(!is_member("f16", "0.3334"));
}

#[test]
fn bf16_membership_at_8_bit_mantissa() {
    for t in [
        "256",
        "256.0",
        "0.75",
        "0.010009765625",
        // Own-width shortest form of the stored bf16 value nearest 1/3.
        "0.334",
        "inf",
    ] {
        assert!(is_member("bf16", t), "{t} must be in the bf16 value set");
    }
    assert!(!is_member("bf16", "257"));
    assert!(!is_member("bf16", "257.0"));
    // A nearby decimal that merely rounds to the same bf16 is not the
    // canonical own-width shortest rendering.
    assert!(!is_member("bf16", "0.3339"));
}

#[test]
fn every_half_width_canonical_rendering_is_a_domain_member() {
    for bits in 0..=u16::MAX {
        let f16_value = half::f16::from_bits(bits);
        let f16_token = format_element(Prim::F16, ElementRef::F16(f16_value));
        assert!(
            is_member("f16", &f16_token),
            "f16 bits {bits:#06x} rendered as {f16_token:?}"
        );

        let bf16_value = half::bf16::from_bits(bits);
        let bf16_token = format_element(Prim::Bf16, ElementRef::Bf16(bf16_value));
        assert!(
            is_member("bf16", &bf16_token),
            "bf16 bits {bits:#06x} rendered as {bf16_token:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// f8e4m3 and prim-name handling
// ---------------------------------------------------------------------------

#[test]
fn f8e4m3_is_never_a_valid_runtime_dtype() {
    assert!(!is_member("f8e4m3", "1.0"));
}

#[test]
fn list_wrapper_prims_normalize_to_the_element_prim() {
    assert!(is_member("List[int64]", "9007199254740993"));
    assert!(!is_member("List[int8]", "200"));
    assert!(is_member("List[List[int64]]", "3000000000"));
}

#[test]
#[should_panic(expected = "unknown prim")]
fn unknown_prim_names_panic_loudly() {
    let _ = element_domain_violation("float32", "1.0");
}

#[test]
#[should_panic(expected = "domain violation")]
fn assert_entry_point_reports_the_offending_token() {
    assert_elements_in_domain("f16", "tensor(shape=[2], data=[2049.0, 0.75])", "self-test");
}

#[test]
fn assert_entry_point_passes_clean_payloads() {
    assert_elements_in_domain("f16", "tensor(shape=[2], data=[2048.0, 0.75])", "self-test");
    assert_elements_in_domain("int64", "[9007199254740993]", "self-test");
    assert_elements_in_domain(
        "bool",
        "tensor(shape=[3], data=[1.0, 0.0, 1.0])",
        "self-test",
    );
}
