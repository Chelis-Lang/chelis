use std::cell::Cell;

use chelis_checker_core::{CheckFacts, check_ir_program, check_linearity};
use chelis_ir_core::claim_pattern::{ClaimDim, ClaimNode, ClaimPattern, ClaimPatternError};
use chelis_ir_core::host::{
    ManifestEntry, ManifestView, PreparedHostFacts, host_def_kernel,
    try_lower_compiled_program_with_manifest,
};
use chelis_ir_core::host_type_state::{ConcreteHostType, decode_host_type};
use chelis_ir_core::resources::{Observer, Resources};
use chelis_numeric_core::prim::Prim;
use chelis_stack_services::StackServices;

const STACK_BYTES: usize = 768 * 1024 * 1024;
const STACK_RESERVE: usize = 64 * 1024;

struct ThreadStack {
    origin: usize,
    bytes: usize,
}

impl StackServices for ThreadStack {
    fn grow<R>(&self, bytes: usize, operation: impl FnOnce() -> R) -> R {
        assert!(self.remaining_stack().unwrap() >= bytes);
        operation()
    }

    fn maybe_grow<R>(
        &self,
        red_zone: usize,
        bytes: usize,
        operation: impl FnOnce() -> R,
    ) -> R {
        let remaining = self.remaining_stack().unwrap();
        assert!(remaining >= red_zone);
        assert!(remaining >= bytes);
        operation()
    }

    fn remaining_stack(&self) -> Option<usize> {
        let marker = 0u8;
        let current = &marker as *const u8 as usize;
        let used = self.origin.abs_diff(current);
        Some(self.bytes.saturating_sub(used).saturating_sub(STACK_RESERVE))
    }
}

#[derive(Default)]
struct Observations {
    preparations: Cell<usize>,
    sessions: Cell<usize>,
}

impl Observer for Observations {
    fn program_context_prepared(&self) {
        self.preparations.set(self.preparations.get() + 1);
    }

    fn host_lowering_session(&self) {
        self.sessions.set(self.sessions.get() + 1);
    }
}

struct BorrowedManifest<'a>(&'a [(&'a str, bool)]);

impl ManifestView for BorrowedManifest<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn entry(&self, index: usize) -> ManifestEntry<'_> {
        let (def_name, tensor_lane) = self.0[index];
        ManifestEntry { def_name, tensor_lane }
    }
}

fn parse_one(source: &str, stack: &ThreadStack) -> chelis_deep::Expr {
    let mut expressions = chelis_deep::parser::parse_str(source, stack).expect("Deep syntax");
    assert_eq!(expressions.len(), 1);
    expressions.remove(0)
}

fn checked_facts(stack: &ThreadStack) -> CheckFacts {
    let source = r#"
        (defsig {} identity (t-fn {} (t-prim {} string) (t-prim {} string)))
        (def {} identity
          (fn {} (params {} (value {type: (t-prim {} string)})) (var {} value)))
        (def {} observed
          (app {} (var {} identity) (lit {type: (t-prim {} string)} "portable")))
    "#;
    let expressions = chelis_deep::parser::parse_str(source, stack).expect("program syntax");
    let resources = chelis_checker_core::Resources::new(stack);
    let completed = check_ir_program(&expressions, &resources).expect("canonical checker facts");
    check_linearity(completed.facts(), &resources).expect("canonical linearity facts")
}

fn run() {
    let marker = 0u8;
    let stack = ThreadStack {
        origin: &marker as *const u8 as usize,
        bytes: STACK_BYTES,
    };
    let facts = checked_facts(&stack);
    let observations = Observations::default();
    let mut resources = Resources::new(&stack);
    resources.observer = Some(&observations);
    let entries = [("observed", false)];
    let manifest = BorrowedManifest(&entries);
    let prepared = PreparedHostFacts::new(facts);
    let facts = prepared.program();

    resources.run(|operation| {
        let compiled = try_lower_compiled_program_with_manifest(facts, &manifest, operation)
            .expect("portable host construction");
        let host = compiled.host.expect("string program uses the host lane");
        assert_eq!(host.globals.len(), 1);
        assert_eq!(host.functions.len(), 1);
        let identity = host.functions.iter().find(|function| function.name == "identity")
            .expect("identity host function");
        assert_eq!(identity.params.len(), 1);
        assert_eq!(identity.params[0].ty, ConcreteHostType::Scalar(Prim::String));
        assert_eq!(identity.ret_ty, ConcreteHostType::Scalar(Prim::String));
        let observed = host.globals.iter().find(|binding| binding.name == "observed")
            .expect("observable host binding");
        assert_eq!(observed.ty, ConcreteHostType::Scalar(Prim::String));
        assert!(host.summary_rejections.is_empty());

        let before = observations.preparations.get();
        {
            let session = prepared.session(operation);
            let _context = session.checked_subexpr_lowering_context().expect("prepared context");
            assert!(host_def_kernel(&session, "identity").expect("host helper analysis").is_none());
        }
        let after_first = observations.preparations.get();
        assert_eq!(after_first, before + 1);
        {
            let session = prepared.session(operation);
            let _context = session.checked_subexpr_lowering_context().expect("cached prepared context");
            assert!(host_def_kernel(&session, "identity").expect("cached helper analysis").is_none());
        }
        assert_eq!(observations.preparations.get(), after_first);
    });

    let authored = parse_one("(t-tensor {} (d-lit {} 3) (t-prim {} f32))", &stack);
    let concrete = decode_host_type(&authored).expect("logical tensor type")
        .into_concrete().expect("resolved tensor type");
    let ConcreteHostType::Tensor(tensor) = concrete else {
        panic!("tensor type expected");
    };
    assert_eq!(tensor.precision, Prim::F32);
    assert_eq!(tensor.dims, [chelis_dag_core::dag::DimInfo::Lit(3)]);

    let pattern = ClaimPattern::derive(&authored, facts.adt_registry()).expect("finite claim graph");
    let ClaimNode::Tensor(claim) = pattern.node(pattern.root().expect("tensor claim root")) else {
        panic!("tensor claim expected");
    };
    assert_eq!(claim.rank, Some(1));
    assert_eq!(claim.axes.len(), 1);
    assert_eq!(claim.axes[0].claim, ClaimDim::Literal(3));

    let unresolved = parse_one(
        "(t-adt {} Missing (t-tensor {} (d-lit {} 3) (t-prim {} f32)))",
        &stack,
    );
    assert_eq!(
        ClaimPattern::derive(&unresolved, facts.adt_registry()).unwrap_err(),
        ClaimPatternError::UnresolvedNominal { nominal: "Missing".into() },
    );

    println!(
        "IR_HOST_CONSUMER {{\"host_globals\":1,\"host_functions\":1,\"claim_axes\":1,\"cache_reused\":true,\"unresolved_nominal_rejected\":true}}"
    );
}

fn main() {
    std::thread::Builder::new()
        .name("portable-host-consumer".into())
        .stack_size(STACK_BYTES)
        .spawn(run)
        .expect("consumer stack allocation")
        .join()
        .expect("portable host consumer");
}
