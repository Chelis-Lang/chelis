//! chelis#730 Phase 3 / [05-UNS-5]: rejection authority is typed and
//! registry-validated rather than inferred from free-form hint prose.

use chelis_types::unsupported::{RejectionAuthorityKind, Stage, Unsupported, UnsupportedKind};

#[test]
fn registered_atom_constructs_a_deliberate_authority() {
    let authority = chelis_types::unsupported::__build_deliberate_rejection(
        "[04-TOT-3]",
        "malformed typed forms are rejected",
    )
    .expect("the numbered spec declares this atom");

    assert_eq!(authority.kind(), RejectionAuthorityKind::Deliberate);
    assert_eq!(authority.citation(), "[04-TOT-3]");
    assert_eq!(authority.hint(), "malformed typed forms are rejected");
}

#[test]
fn atom_admission_rejects_every_invalid_route() {
    for atom in ["", "05-UNS-1", "[05-uns-1]", "[05-UNS-0]", "[05-ZZZ-999]"] {
        assert!(
            chelis_types::unsupported::__build_deliberate_rejection(atom, "hint").is_err(),
            "invalid or undeclared atom {atom:?} must not become authority"
        );
    }
}

#[test]
fn response_contract_atoms_cannot_authorize_a_semantic_case() {
    for atom in [
        "[05-UNS-1]",
        "[05-UNS-2]",
        "[05-UNS-3]",
        "[05-UNS-4]",
        "[05-UNS-5]",
        "[05-UNS-6]",
    ] {
        assert!(
            chelis_types::unsupported::__build_deliberate_rejection(atom, "hint").is_err(),
            "response-contract atom {atom} must not authorize a semantic case"
        );
    }
}

#[test]
fn registered_open_issue_constructs_an_unimplemented_authority() {
    let authority = chelis_types::unsupported::__build_unimplemented_rejection(
        879,
        "general C closure ABI is pending",
    )
    .expect("chelis#879 is in the live-verified manifest");

    assert_eq!(authority.kind(), RejectionAuthorityKind::Unimplemented);
    assert_eq!(authority.citation(), "chelis#879");
    assert_eq!(authority.hint(), "general C closure ABI is pending");
}

#[test]
fn issue_admission_rejects_zero_closed_pr_and_missing_numbers() {
    for issue in [0, 944, 1, 999_999_999] {
        assert!(
            chelis_types::unsupported::__build_unimplemented_rejection(issue, "hint").is_err(),
            "non-live issue authority chelis#{issue} must be rejected"
        );
    }
}

#[test]
fn both_authority_constructors_reject_empty_hints() {
    assert!(chelis_types::unsupported::__build_deliberate_rejection("[04-TOT-3]", "").is_err());
    assert!(chelis_types::unsupported::__build_unimplemented_rejection(879, "").is_err());
}

#[test]
fn compile_time_helpers_accept_only_registered_authorities() {
    let deliberate =
        chelis_types::deliberate_rejection!("[04-TOT-3]", "malformed typed forms are rejected");
    let unimplemented = chelis_types::unimplemented_rejection!(
        879,
        "general C first-class function values are not implemented"
    );

    assert_eq!(deliberate.kind(), RejectionAuthorityKind::Deliberate);
    assert_eq!(unimplemented.kind(), RejectionAuthorityKind::Unimplemented);
}

#[test]
#[allow(clippy::inconsistent_digit_grouping, clippy::zero_prefixed_literal)]
fn unimplemented_macro_accepts_every_valid_decimal_token_shape() {
    let authorities = [
        chelis_types::unimplemented_rejection!(000879, "leading zeroes"),
        chelis_types::unimplemented_rejection!(8__79, "repeated underscores"),
        chelis_types::unimplemented_rejection!(879_, "trailing underscore"),
        chelis_types::unimplemented_rejection!(0_879, "zero-prefixed digits"),
    ];
    assert!(
        authorities
            .iter()
            .all(|authority| authority.citation() == "chelis#879")
    );
}

#[test]
fn diagnostic_surface_exposes_deliberate_host_only_authority() {
    let error = Unsupported::new(
        UnsupportedKind::Builtin("tensor_scan".to_string()),
        "`chelis build --target c` host emission",
        Stage::Codegen("c"),
        chelis_types::deliberate_rejection!(
            "[05-HOST-1]",
            "host-runtime builders are intentionally excluded from compiled targets; run under `chelis eval`"
        ),
    );

    assert_eq!(
        error.to_string(),
        "unsupported: builtin `tensor_scan` on `chelis build --target c` host emission \
         (codegen:c); deliberate [05-HOST-1]: host-runtime builders are intentionally \
         excluded from compiled targets; run under `chelis eval`"
    );
}

#[test]
fn diagnostic_surface_exposes_deliberate_kind_and_validated_atom() {
    let error = Unsupported::new(
        UnsupportedKind::Construct("a malformed Deep form".to_string()),
        "host lowering",
        Stage::Lowering,
        chelis_types::deliberate_rejection!(
            "[04-TOT-3]",
            "the hint may mention an unrelated chelis#879 without replacing the authority"
        ),
    );

    assert_eq!(
        error.to_string(),
        "unsupported: a malformed Deep form on host lowering (lowering); deliberate \
         [04-TOT-3]: the hint may mention an unrelated chelis#879 without replacing the authority"
    );
}
