//! chelis#730 Phase 3 / [05-UNS-5]: rejection authority is typed and
//! registry-validated rather than inferred from free-form hint prose.

use chelis_types::unsupported::{
    IssueRef, RejectionAuthority, RejectionAuthorityKind, SpecAtomRef, Stage, Unsupported,
    UnsupportedKind,
};

#[test]
fn registered_atom_constructs_a_deliberate_authority() {
    let atom = SpecAtomRef::new("[05-UNS-1]").expect("the numbered spec declares this atom");
    let authority = RejectionAuthority::deliberate(atom, "reject instead of substituting")
        .expect("a non-empty hint is valid");

    assert_eq!(authority.kind(), RejectionAuthorityKind::Deliberate);
    assert_eq!(authority.citation(), "[05-UNS-1]");
    assert_eq!(authority.hint(), "reject instead of substituting");
}

#[test]
fn atom_admission_rejects_every_invalid_route() {
    for atom in ["", "05-UNS-1", "[05-uns-1]", "[05-UNS-0]", "[05-ZZZ-999]"] {
        assert!(
            SpecAtomRef::new(atom).is_err(),
            "invalid or undeclared atom {atom:?} must not become authority"
        );
    }
}

#[test]
fn registered_open_issue_constructs_an_unimplemented_authority() {
    let issue = IssueRef::new(879).expect("chelis#879 is in the live-verified manifest");
    let authority = RejectionAuthority::unimplemented(issue, "general C closure ABI is pending")
        .expect("a non-empty hint is valid");

    assert_eq!(authority.kind(), RejectionAuthorityKind::Unimplemented);
    assert_eq!(authority.citation(), "chelis#879");
    assert_eq!(authority.hint(), "general C closure ABI is pending");
}

#[test]
fn issue_admission_rejects_zero_closed_pr_and_missing_numbers() {
    for issue in [0, 944, 1, 999_999_999] {
        assert!(
            IssueRef::new(issue).is_err(),
            "non-live issue authority chelis#{issue} must be rejected"
        );
    }
}

#[test]
fn both_authority_constructors_reject_empty_hints() {
    let atom = SpecAtomRef::new("[05-UNS-1]").unwrap();
    let issue = IssueRef::new(879).unwrap();
    assert!(RejectionAuthority::deliberate(atom, "").is_err());
    assert!(RejectionAuthority::unimplemented(issue, "").is_err());
}

#[test]
fn compile_time_helpers_accept_only_registered_authorities() {
    let deliberate = chelis_types::deliberate_rejection!(
        "[05-UNS-1]",
        "the checked case must reject instead of substituting"
    );
    let unimplemented = chelis_types::unimplemented_rejection!(
        879,
        "general C first-class function values are not implemented"
    );

    assert_eq!(deliberate.kind(), RejectionAuthorityKind::Deliberate);
    assert_eq!(unimplemented.kind(), RejectionAuthorityKind::Unimplemented);
}

#[test]
fn authority_typing_preserves_the_frozen_human_rendering() {
    let error = Unsupported::new(
        UnsupportedKind::Builtin("tensor_scan".to_string()),
        "`chelis build --target c` host emission",
        Stage::Codegen("c"),
        chelis_types::unimplemented_rejection!(
            705,
            "host-only builtin; run it under `chelis eval` (chelis#705)"
        ),
    );

    assert_eq!(
        error.to_string(),
        "unsupported: builtin `tensor_scan` on `chelis build --target c` host emission \
         (codegen:c); host-only builtin; run it under `chelis eval` (chelis#705)"
    );
}
