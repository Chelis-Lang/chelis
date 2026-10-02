//! chelis#469: a runtime negative `expand`/`insert` size traps `Domain` in the
//! host interpreter exactly as compiled C renders it.
//!
//! spec/04-type-system.md section 4.7.2: "A runtime negative size traps
//! `Domain` before allocation or access." The C runtime prints the metadata
//! line `Domain: expansion axis or extent outside domain` and then [04-NUM-9]'s
//! `numeric trap: domain in <op> at i64`. The host interpreter used to return
//! the untyped `<op> requires non-negative extent, got <n>` instead, so the
//! same program failed with different, untyped text on the two lanes. The
//! cross-lane comparison is
//! `crates/chelis-cli/tests/issue_469_runtime_scalar_extent.rs`
//! (`a_runtime_negative_size_traps_domain_on_both_lanes`).

use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;

fn eval_error(source: &str) -> String {
    let error = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings: BTreeMap::new(),
        },
        &["out".into()],
    )
    .expect_err("a runtime negative size must not evaluate");
    error
        .errors
        .iter()
        .map(|e| e.message.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn a_runtime_negative_size_traps_domain_in_the_host_interpreter() {
    for (op, operand) in [
        ("expand", "to_tensor([[1i64, 2i64]])"),
        ("insert", "to_tensor([1i64, 2i64])"),
    ] {
        let message = eval_error(&format!(
            "def f(k: i64) = {op}({operand}, 0, k)\n\
             out = f(sub(tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0)), 5i64))\n"
        ));
        assert!(
            message.contains(&format!(
                "Domain: expansion axis or extent outside domain\nnumeric trap: domain in {op} at i64"
            )),
            "{op}: the trap renders as the C runtime renders it; got {message}"
        );
        assert!(
            !message.contains("requires non-negative extent"),
            "{op}: the untyped rendering is gone; got {message}"
        );
    }
}
