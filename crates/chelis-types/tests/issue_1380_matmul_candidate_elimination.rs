//! `matmul`'s operand rank rule, after chelis#1380's candidate model was
//! superseded.
//!
//! chelis#1380 reported that `matmul` over a deferred positional `expand`
//! result published an inference variable as its checked result type. The
//! model that defect lived in no longer exists: `spec/04-type-system.md`
//! section 4.7.2 gives `expand` and `insert` exactly one result shape each, so
//! no consumer eliminates candidates and there is no deferred operand for
//! `matmul` to publish. The nine rows whose subject was elimination were
//! removed with the model (chelis#1277 S2b), and the design doc records #1380
//! as superseded and to be re-read against section 4.7.2.
//!
//! Two rows survive, and neither is about candidate elimination.
//!
//! `matmul_rejects_an_operand_no_candidate_form_satisfies` keeps its exact
//! subject, that `matmul` rejects a rank-1 operand, and spells its producer
//! `insert`, the operation that raises a rank-0 input to rank 1 under section
//! 4.7.2. It is a disposition lock: the rejection it asserts never depended on
//! any repair this file's history describes.
//!
//! `batched_matmul_without_a_deferred_operand_is_supported` never involved a
//! deferred operand, which is what its name says, so the single result shape
//! leaves it untouched.

use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

fn check(source: &str) -> Result<String, Vec<CheckError>> {
    let decls = parse_surf(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    match check_typed_program(&deep) {
        Ok(checked) => Ok(print_canonical(checked.annotated_exprs())),
        Err(report) => Err(report.errors),
    }
}

fn summary(errors: &[CheckError]) -> String {
    errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn matmul_rejects_an_operand_no_candidate_form_satisfies() {
    let errors = check(
        "def f(s: f32, xb: tensor[3, 4, f32]) = \
         matmul(insert(scalar_to_tensor(s), 0, 3i64), xb)\n",
    )
    .expect_err("`insert` raises the rank-0 input to rank 1, which matmul rejects");
    assert!(
        errors.iter().any(|error| {
            let kind = format!("{:?}", error.kind);
            kind.contains("Dimension") || kind.contains("TypeMismatch")
        }),
        "matmul rejects a rank-1 operand:\n{}",
        summary(&errors)
    );
}

/// Batched operands type cleanly on their own. This row never involved a
/// deferred operand, so the single result shape leaves it alone, and it stays
/// as the control that `matmul` supports batching at all.
#[test]
fn batched_matmul_without_a_deferred_operand_is_supported() {
    check("def f(xa: tensor[5, 3, 4, f32], xb: tensor[5, 4, 7, f32]) = matmul(xa, xb)\n")
        .expect("batched matmul is supported");
}
