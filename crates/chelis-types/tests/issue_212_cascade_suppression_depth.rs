//! Issue #212's cascade marker existed only to suppress repeated instances of
//! the old checker-level `conv` concrete-metadata refusal. PP9 relocates that
//! backend capability restriction to #730, so symbolic conv chains are legal
//! checker input and need no failed-derivation side channel.

use chelis_deep::Expr;
use chelis_deep::parser::parse_str;
use chelis_types::{check_ir_program, check_typed_program};

fn core_fixture(source: &str) -> Vec<Expr> {
    parse_str(source).expect("core fixture")
}

#[test]
fn symbolic_conv_chain_is_accepted_without_cascade_state() {
    let source = include_str!(
        "fixtures/core/tests/issue_212_cascade_suppression_depth/symbolic_conv_chain_is_accepted_without_cascade_state.deep"
    );
    let expressions = core_fixture(source);
    check_ir_program(&expressions, chelis_types::ProgramOrigin::Unlinked)
        .expect("IR checker accepts symbolic conv metadata");
    check_typed_program(&expressions, chelis_types::ProgramOrigin::Unlinked)
        .expect("typed checker accepts symbolic conv metadata");
}
