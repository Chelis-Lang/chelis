#![allow(dead_code)]

//! Shared oracle for tests that pin which uses consume.
//!
//! spec/04 section 8.3: a later use after an ordinary consume, a borrow
//! included, is consuming fan-out that an inserted copy repairs. A program
//! that reuses a binding after an ordinary consume is therefore accepted, and
//! the consume shows as the copy site of a [`CopyRepair`] rather than as a
//! `UseAfterConsume` error. A use that does not consume records no repair, so
//! the repair list is the observable that tells the two apart.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::{CopyRepair, CopyRepairUseKind, check_typed_program, copy_repairs};

/// The copy repairs of a Surf program that must pass linearity.
pub fn copy_repairs_of(source: &str) -> Vec<CopyRepair> {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&deep)
        .unwrap_or_else(|error| panic!("type check should succeed: {:?}", error.errors));
    copy_repairs(&checked)
        .unwrap_or_else(|errors| panic!("expected the fan-out to be copy-repaired; got {errors:?}"))
}

/// Assert that `source` passes linearity and that `binding` receives a copy at
/// a consume whose description contains `consumed_by`, forced by a later use
/// of kind `later`.
#[track_caller]
pub fn assert_copy_repaired(
    source: &str,
    binding: &str,
    consumed_by: &str,
    later: CopyRepairUseKind,
) {
    let repairs = copy_repairs_of(source);
    assert!(
        repairs.iter().any(|repair| {
            repair.binding == binding
                && repair.consumed_by.contains(consumed_by)
                && repair.forced_by.iter().any(|use_| use_.kind == later)
        }),
        "expected a copy of `{binding}` at a consume by {consumed_by}, forced by a later \
         {later:?}; got {repairs:#?}"
    );
}
