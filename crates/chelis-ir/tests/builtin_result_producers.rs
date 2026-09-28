//! spec/04 section 4.7: every builtin produces the tensors it returns,
//! directly or held in its aggregate result, except the contiguous-selection
//! projections `index`, `take` and `skip`. Both lanes ask
//! `produces_its_result`; this test holds the projection triple independently
//! of the implementation's constant and checks every catalogue builtin.

use chelis_ir::host::{BUILTIN_PROJECTIONS, produces_its_result};

/// spec/04 section 4.7's contiguous-selection projections, stated here
/// independently of the implementation's constant.
const SPEC_PROJECTIONS: &[&str] = &["index", "skip", "take"];

#[test]
fn every_builtin_but_the_projections_produces_its_result() {
    assert_eq!(BUILTIN_PROJECTIONS, SPEC_PROJECTIONS);
    let mut wrong = Vec::new();
    for builtin in chelis_types::BUILTINS {
        let expected = !SPEC_PROJECTIONS.contains(&builtin.name);
        if produces_its_result(builtin.name) != expected {
            wrong.push(format!("{}: expected {expected}", builtin.name));
        }
    }
    for projection in SPEC_PROJECTIONS {
        assert!(
            chelis_types::builtin_decl(projection).is_some(),
            "{projection} is not a catalogue builtin"
        );
    }
    // A name outside the catalogue (an internal host form such as
    // `tuple-get`) is not a builtin and produces nothing here.
    assert!(!produces_its_result("tuple-get"));
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
