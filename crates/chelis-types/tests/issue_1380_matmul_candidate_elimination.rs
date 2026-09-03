//! chelis#1380: `matmul` over a deferred positional `expand` result published
//! an inference variable as its checked result type.
//!
//! `spec/04-type-system.md` §4.7.2: "A consumer that supplies no complete
//! shape equation still eliminates candidates by its own typing rule: the
//! result keeps only the forms that consumer admits, so a consumer admitting
//! exactly one form fixes the result, a consumer admitting both leaves the
//! choice open, and a consumer admitting neither rejects the program. A rank
//! requirement alone can decide it: `matmul` admits only operands of rank at
//! least two (`spec/05-risc-primitives.md` §4.1), so a positional `expand`
//! over a rank-1 input is fixed to the insertion form even though `matmul`
//! constrains none of its extents. A consumer that fixes its operand this way
//! types its own result from the fixed form and does not publish an
//! unresolved candidate as its checked result type."
//!
//! Evidentiary status is per row, and it was established by execution: the two
//! source files were reverted to `6b60742cd` and the file rerun.
//!
//! Five rows are regression tests and fail on that base. The filed reproducer,
//! the single-pending-operand row, and the contraction row all publish
//! `(t-var ...)` into the checked program with an empty error list, and the
//! two rejecting rows accept a program `matmul` admits no operand form of.
//!
//! Two rows are disposition locks and pass on that base.
//! `a_declared_result_still_selects_without_the_rank_rule` passes because a
//! declared result already reached the settlement transaction through
//! `bind_tvar`, which is what shows the machinery was present and only the
//! early return was in the way.
//! `residual_a_rank_two_input_leaves_both_candidates_and_still_escapes` passes
//! because it asserts the escape rather than its absence. It is the one row
//! here that records a defect rather than a repair, and it is expected to keep
//! asserting that escape until `matmul` carries a relation from its result
//! back to its operands while the choice is open. When it starts failing, the
//! remaining half of chelis#1380 has been fixed and the row should be deleted
//! and the issue closed, never relaxed to keep it green.

use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;
use chelis_types::errors::CheckError;

fn check(source: &str) -> Result<String, Vec<CheckError>> {
    let decls = parse_surf(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    match check_typed_program(&deep) {
        Ok(checked) => Ok(print_canonical(checked.annotated_exprs())),
        Err(report) => Err(report.errors),
    }
}

/// Every type variable that survives into the checked program, other than the
/// wildcard slot the Surf desugarer synthesizes into a `defsig` for a part the
/// author left to inference.
///
/// A bare `contains("(t-var")` would also catch that wildcard, which is source
/// text rather than an unresolved inference variable, so it would report a
/// failure on a program the checker resolved completely.
fn escaping_type_variables(rendered: &str) -> Vec<String> {
    rendered
        .match_indices("(t-var {} ")
        .filter_map(|(at, marker)| {
            let rest = &rendered[at + marker.len()..];
            let end = rest.find(')')?;
            let name = rest[..end].trim().to_string();
            (name != "_").then_some(name)
        })
        .collect()
}

/// Whitespace-insensitive containment.
///
/// The canonical printer wraps a long type across several lines, so a
/// single-line needle can silently never match and an assertion built on one
/// passes or fails for the wrong reason. Both sides are collapsed to single
/// spaces before comparing.
fn contains_type(rendered: &str, needle: &str) -> bool {
    fn collapse(text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }
    collapse(rendered).contains(&collapse(needle))
}

fn summary(errors: &[CheckError]) -> String {
    errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn issue1380_filed_reproducer_publishes_no_inference_variable() {
    let rendered = check(
        "def f(xa: tensor[2, f32], xb: tensor[2, f32]) = \
         matmul(expand(xa, 0, 6i64), expand(xb, 1, 3i64))\n\
         def main() = f(to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32]))\n",
    )
    .unwrap_or_else(|errors| panic!("the rank rule fixes both operands:\n{}", summary(&errors)));
    assert_eq!(
        escaping_type_variables(&rendered),
        Vec::<String>::new(),
        "an unresolved inference variable must never reach a checked \
         signature:\n{rendered}"
    );
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 6) (d-lit {} 3) (t-prim {} f32))"),
        "matmul types its own result from the fixed operand forms:\n{rendered}"
    );
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 6) (d-lit {} 2) (t-prim {} f32))"),
        "the rank-1 lhs is fixed to its insertion form, the only candidate \
         matmul admits:\n{rendered}"
    );
}

#[test]
fn one_pending_operand_against_a_concrete_one_publishes_no_inference_variable() {
    let rendered = check(
        "def f(xa: tensor[2, f32], xb: tensor[2, 3, f32]) = matmul(expand(xa, 0, 6i64), xb)\n",
    )
    .unwrap_or_else(|errors| panic!("the rank rule fixes the lhs:\n{}", summary(&errors)));
    assert_eq!(
        escaping_type_variables(&rendered),
        Vec::<String>::new(),
        "a single pending operand escapes just as readily as two:\n{rendered}"
    );
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 6) (d-lit {} 2) (t-prim {} f32))"),
        "the pending lhs is fixed to its rank-2 insertion form:\n{rendered}"
    );
}

#[test]
fn a_declared_result_still_selects_without_the_rank_rule() {
    let rendered = check(
        "def f(xa: tensor[2, f32], xb: tensor[2, f32]) -> tensor[6, 3, f32] = \
         matmul(expand(xa, 0, 6i64), expand(xb, 1, 3i64))\n",
    )
    .expect("a declared result already resolved this before the rank rule existed");
    assert_eq!(
        escaping_type_variables(&rendered),
        Vec::<String>::new(),
        "{rendered}"
    );
}

#[test]
fn matmul_rejects_an_operand_no_candidate_form_satisfies() {
    let errors = check(
        "def f(s: f32, xb: tensor[3, 4, f32]) = \
         matmul(expand(scalar_to_tensor(s), 0, 3i64), xb)\n",
    )
    .expect_err("a rank-0 input has only the rank-1 insertion form, which matmul rejects");
    assert!(
        errors.iter().any(|error| {
            let kind = format!("{:?}", error.kind);
            kind.contains("Dimension") || kind.contains("TypeMismatch")
        }),
        "a consumer admitting neither candidate rejects the program:\n{}",
        summary(&errors)
    );
}

/// Residual, and deliberately recorded rather than fixed here.
///
/// §4.7.2 says a consumer admitting both forms "leaves the choice open", so
/// when the `expand` input already has rank two or more both candidates
/// survive `matmul`'s rank rule and it decides nothing. The operand then
/// settles at the program freeze point, but `matmul`'s own result variable was
/// never related to it, so an inference variable still reaches the checked
/// program. That is the same contract violation chelis#1380 reports, in a
/// shape the rank rule cannot reach.
///
/// Closing it needs `matmul` to carry a relation from its operands to its
/// result while the choice is open, which is a new mechanism and new scope.
/// This test records the current verdict so the shape is not forgotten and so
/// a later fix has to flip it deliberately. chelis#1380 stays open for it.
#[test]
fn residual_a_rank_two_input_leaves_both_candidates_and_still_escapes() {
    let rendered = check(
        "def f(xa: tensor[2, 3, f32], xb: tensor[3, 4, f32]) = matmul(expand(xa, 0, 6i64), xb)\n",
    )
    .expect("both candidate forms have rank two or more, so the rank rule decides nothing");
    assert!(
        !escaping_type_variables(&rendered).is_empty(),
        "if this now passes with no escaping variable, the residual half of \
         chelis#1380 has been fixed: delete this test and close the issue \
         instead of relaxing it:\n{rendered}"
    );
}

/// The rank requirement is an instance of the elimination rule, not its
/// limit. Here both candidate forms clear rank two, so the rank alone decides
/// nothing, and it is `matmul`'s contraction that admits exactly one of them:
/// `tensor[2, 6]` contracts against `tensor[6, 4]`, while `tensor[2, 6, 3]`
/// would have to contract a 3 against a 6.
#[test]
fn matmuls_contraction_decides_where_the_rank_requirement_cannot() {
    let rendered = check(
        "def f(xa: tensor[2, 3, f32], xb: tensor[6, 4, f32]) = matmul(expand(xa, 1, 6i64), xb)\n",
    )
    .unwrap_or_else(|errors| {
        panic!(
            "the contraction admits exactly one form:\n{}",
            summary(&errors)
        )
    });
    assert_eq!(
        escaping_type_variables(&rendered),
        Vec::<String>::new(),
        "a consumer that fixes its operand types its own result from the \
         fixed form:\n{rendered}"
    );
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 2) (d-lit {} 4) (t-prim {} f32))"),
        "the result follows from the one admitted operand form:\n{rendered}"
    );
}

/// The elimination rule is conservative on the rejecting side too: when the
/// contraction can satisfy neither form, the program is rejected rather than
/// left open.
#[test]
fn matmul_rejects_when_the_contraction_admits_neither_form() {
    let errors = check(
        "def f(xa: tensor[2, 3, f32], xb: tensor[5, 4, f32]) = matmul(expand(xa, 1, 6i64), xb)\n",
    )
    .expect_err("neither tensor[2, 6] nor tensor[2, 6, 3] contracts against a leading 5");
    assert!(
        errors.iter().any(|error| {
            let kind = format!("{:?}", error.kind);
            kind.contains("Dimension") || kind.contains("TypeMismatch")
        }),
        "a consumer admitting neither candidate rejects the program:\n{}",
        summary(&errors)
    );
}

// ---------------------------------------------------------------------
// Batched siblings.
//
// `matmul` contracts the left operand's last axis against the right operand's
// second-to-last, and those are the same index only when the right operand has
// rank exactly two. Every other row in this file uses a rank-2 sibling, so the
// two readings agree and none of them can tell a correct accessor from one
// that reads the sibling's first axis instead.
//
// Each row below therefore uses a rank-3 sibling whose batch extent differs
// from its contracted extent. If the two were equal the row would pass under
// either accessor and the defect would return invisibly.
// ---------------------------------------------------------------------

/// The candidate is the left operand, so its last axis contracts against the
/// sibling's second-to-last. Sibling `tensor[5, 4, 7]` batches at 5 and
/// contracts at 4, so reading its first axis compares against 5 and eliminates
/// both candidate forms.
#[test]
fn a_batched_right_sibling_does_not_eliminate_the_left_operand() {
    let rendered = check(
        "def f(xa: tensor[3, 4, f32], xb: tensor[5, 4, 7, f32]) = matmul(expand(xa, 0, 5i64), xb)\n",
    )
    .unwrap_or_else(|errors| {
        panic!(
            "both candidate forms contract against the sibling's second-to-last \
             axis, so elimination must not reject:\n{}",
            summary(&errors)
        )
    });
    // Both candidate forms end in 4 and the sibling contracts at 4, so both
    // are admissible and elimination correctly decides nothing. The freeze
    // default then supplies the same-rank replacement form, which is what
    // `6b60742cd` produces for this program too.
    assert!(
        contains_type(
            &rendered,
            "(t-tensor {} (d-lit {} 5) (d-lit {} 4) (t-prim {} f32))"
        ),
        "elimination must leave a choice both forms satisfy to the freeze \
         default:\n{rendered}"
    );
}

/// The candidate is the right operand, so the sibling's last axis contracts
/// against the candidate's second-to-last. Sibling `tensor[5, 3, 4]` ends in 4;
/// only the insertion form `tensor[5, 4, 7]` carries 4 in that position.
#[test]
fn a_batched_left_sibling_selects_the_right_operand_form() {
    let rendered = check(
        "def f(xa: tensor[5, 3, 4, f32], xb: tensor[4, 7, f32]) = matmul(xa, expand(xb, 0, 5i64))\n",
    )
    .unwrap_or_else(|errors| {
        panic!(
            "the insertion form contracts against the sibling's last axis:\n{}",
            summary(&errors)
        )
    });
    assert!(
        contains_type(
            &rendered,
            "(t-tensor {} (d-lit {} 5) (d-lit {} 4) (d-lit {} 7) (t-prim {} f32))"
        ),
        "elimination must select the form whose second-to-last axis is 4:\n{rendered}"
    );
}

/// The sharpest row: reading the sibling's first axis does not merely
/// over-reject here, it admits exactly one form and admits the wrong one.
/// Candidates are `tensor[5, 5]` and `tensor[5, 5, 4]`; the sibling contracts
/// at 4, so only the second is admissible, while its first axis is 5, so only
/// the first would be.
#[test]
fn a_batched_sibling_selects_the_form_matmul_actually_admits() {
    let rendered = check(
        "def f(xa: tensor[5, 4, f32], xb: tensor[5, 4, 7, f32]) = matmul(expand(xa, 1, 5i64), xb)\n",
    )
    .unwrap_or_else(|errors| {
        panic!(
            "the admissible form is the rank-3 one, not the rank-2 one:\n{}",
            summary(&errors)
        )
    });
    assert!(
        contains_type(
            &rendered,
            "(t-tensor {} (d-lit {} 5) (d-lit {} 5) (d-lit {} 4) (t-prim {} f32))"
        ),
        "elimination must not select the form whose contracted extent is 5:\n{rendered}"
    );
}

/// Control: the same batched shapes with no deferral type cleanly, so these
/// rows are testing candidate elimination rather than `matmul`'s own support
/// for batched operands.
#[test]
fn batched_matmul_without_a_deferred_operand_is_supported() {
    check("def f(xa: tensor[5, 3, 4, f32], xb: tensor[5, 4, 7, f32]) = matmul(xa, xb)\n")
        .expect("batched matmul is supported independently of any deferral");
}
