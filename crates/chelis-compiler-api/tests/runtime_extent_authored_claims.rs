//! chelis#1374/#1376: a named extent claim is minted only from an AUTHORED
//! signature.
//!
//! `spec/04-type-system.md` section 4.7 reads a binder repeated across
//! parameters, and a declared result's named extent, as equalities the
//! SIGNATURE asserts. A staged host region and a signatureless subexpression
//! program are lowered with machine-built parameter lists - captured root
//! bindings and marshalled host arguments - whose types spell a binder only
//! because the unrelated definitions that produced them did. Relating two such
//! parameters states an obligation no author wrote.
//!
//! The two rows below are the same spelling on either side of that line, so
//! the property is measured rather than argued: the authored `def` mints the
//! claim, and a program whose roots merely happen to share the spelling mints
//! none.

use chelis_compiler_api::pipeline::{
    LoweringMode, PipelineGoal, PipelineOutcome, PipelineRequest, run_source,
};
use chelis_compiler_api::schema::SourceKind;
use chelis_ir::dag::{Dag, RiscOp};

fn request(source: &str) -> PipelineRequest<'_> {
    PipelineRequest {
        source_kind: SourceKind::Surf,
        source,
        entry: None,
        goal: PipelineGoal::Lower(LoweringMode::Strict),
    }
}

fn lowered(source: &str) -> Dag {
    let outcome = run_source(request(source)).expect("source must lower");
    let PipelineOutcome::Lowered(lowered) = outcome else {
        panic!("the lower goal must return a LoweredCompilation");
    };
    lowered.into_parts().dag
}

/// Every named claim in the graph, as `(binder, parameter observing it)`.
fn named_claims(dag: &Dag) -> Vec<(String, String)> {
    let mut found = Vec::new();
    for node in dag.nodes() {
        let RiscOp::ExtentWitness {
            parameter, claims, ..
        } = &node.op
        else {
            continue;
        };
        for claim in claims {
            found.push((claim.claim.clone(), parameter.clone()));
        }
    }
    found.sort();
    found
}

/// An author wrote `extent` twice in one parameter list, so section 4.7 owes
/// the equality "regardless of data use" - here the body never reads `p` at
/// all.
///
/// EVIDENTIARY STATUS: disposition lock. This is the case chelis#1374's
/// repeated-binder claim exists for, and it passed before the authored-signature
/// restriction below; the row exists so narrowing the restriction further
/// cannot silently drop it.
#[test]
fn an_authored_repeated_binder_mints_its_claim() {
    let dag = lowered(
        "def f(x: tensor[extent, f32], p: tensor[extent, f32]) -> tensor[f32] = sum(x, 0i32)\n",
    );
    assert_eq!(
        named_claims(&dag),
        vec![("extent".to_string(), "p".to_string())],
        "the later parameter's witness carries the equality against the earlier one"
    );
}

/// The same spelling, arrived at by coincidence, at the one level this entry
/// reaches.
///
/// `reduce_seq` and `total` are unrelated definitions that each named an axis
/// `seq`. This row pins that the tensor-lane lowering relates nothing across
/// them. It does NOT reach the synthesized multi-root host kernel, which is
/// where chelis#1374's collision actually bit; that lane is measured on the
/// emitted host source by
/// `chelis-cli::runtime_extent_slice_b::a_synthesized_kernel_parameter_collision_emits_no_guard`.
///
/// EVIDENTIARY STATUS: disposition lock, NOT a regression test. Measured GREEN
/// at `01c6e33a1`, before the authored-signature restriction, so it would not
/// have caught the defect. It is here to keep this lane from acquiring the
/// behaviour later.
#[test]
fn unrelated_definitions_sharing_a_spelling_relate_nothing_on_the_tensor_lane() {
    let dag = lowered(
        "def id2(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = relu(x)\n\
         def total(x: &tensor[seq, f32]) -> f32 = tensor_to_scalar(sum(x, seq))\n\
         y = id2(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n\
         outt = total(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
    );
    assert_eq!(
        named_claims(&dag),
        Vec::<(String, String)>::new(),
        "no author related `y`'s axis 1 to `total`'s parameter"
    );
}
