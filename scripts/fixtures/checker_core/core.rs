extern crate alloc;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::Cell;
use chelis_checker_core::{Cancellation, CheckFacts, CheckObserver, Resources};
use chelis_deep::Expr;
use chelis_stack_services::StackServices;

struct CallerStack;
impl StackServices for CallerStack {
    fn grow<R>(&self, _: usize, operation: impl FnOnce() -> R) -> R { operation() }
    fn maybe_grow<R>(&self, _: usize, _: usize, operation: impl FnOnce() -> R) -> R { operation() }
    fn remaining_stack(&self) -> Option<usize> { None }
}

struct NoStack;
impl StackServices for NoStack {
    fn grow<R>(&self, _: usize, operation: impl FnOnce() -> R) -> R { operation() }
    fn maybe_grow<R>(&self, _: usize, _: usize, operation: impl FnOnce() -> R) -> R { operation() }
    fn remaining_stack(&self) -> Option<usize> { Some(0) }
}

fn parse(source: &str) -> Vec<Expr> {
    chelis_deep::parse_and_stamp_file(source, &CallerStack).unwrap()
}

struct CancelAfter { polls: Cell<usize>, limit: usize }
impl Cancellation for CancelAfter {
    fn is_cancelled(&self) -> bool {
        let polls = self.polls.get();
        self.polls.set(polls + 1);
        polls >= self.limit
    }
}

struct Nested<'a> { expressions: &'a [Expr], calls: Cell<usize> }
impl CheckObserver for Nested<'_> {
    fn phase(&self, _: &str) {
        self.calls.set(self.calls.get() + 1);
        let nested = chelis_checker_core::check_ir_program(self.expressions, &Resources::new(&CallerStack)).unwrap();
        assert!(!nested.facts().annotated_exprs().is_empty());
    }
}

fn recursive_group() -> Vec<Expr> {
    let mut source = String::new();
    for index in 0..64 {
        source.push_str(&format!("(defsig {{}} f{index} (t-fn {{}} (t-prim {{}} i32) (t-prim {{}} i32)))\n(def {{}} f{index} (fn {{}} (params {{}} x) (app {{}} (var {{}} f{}) (var {{}} x))))\n", (index + 1) % 64));
    }
    parse(&source)
}

pub fn exercise() -> CheckFacts {
    let stack = CallerStack;
    let resources = Resources::new(&stack);
    let program = parse("(def {} identity (fn {} (params {} value) (var {} value)))\n\
        (def {} integer (app {} (var {} identity) (lit {type: (t-prim {} i32)} 1)))\n\
        (def {} boolean (app {} (var {} identity) (lit {type: (t-prim {} bool)} true)))");
    let before = program.clone();
    assert!(chelis_checker_core::infer_program(&program, &resources).errors.is_empty());
    assert!(chelis_checker_core::infer_ir_program(&program, &resources).errors.is_empty());
    let completion = chelis_checker_core::check_ir_program(&program, &resources).unwrap();
    assert!(!completion.facts().annotated_exprs().is_empty());
    let facts = completion.into_facts();
    assert!(facts.infer_stats().typed_nodes > 0);
    assert_eq!(program, before);
    let typed = chelis_checker_core::check_typed_program(&program, &resources).unwrap();
    assert_eq!(facts.annotated_exprs(), typed.facts().annotated_exprs());
    let declared = parse("(defsig {} declared (t-fn {} (t-prim {} i32) (t-prim {} i32)))\n\
        (def {} declared (fn {} (params {} value) (var {} value)))");
    assert!(chelis_checker_core::resolve_declared_surface(&declared, &resources).is_ok());
    assert!(chelis_checker_core::check_linearity(&facts, &resources).is_ok());
    assert!(chelis_checker_core::copy_repairs(&facts, None, &resources).is_ok());

    let library = parse("(def {} library_id (fn {} (params {} x) (var {} x)))");
    let (context, library_completion) = chelis_checker_core::build_compiled_library_context(&library, &resources).unwrap();
    let simple_context = chelis_checker_core::build_type_env_from_library(&library, &resources).unwrap();
    let extension = parse("(def {} use_id (app {} (var {} library_id) (lit {type: (t-prim {} i32)} 1)))");
    let extension_completion = chelis_checker_core::check_ir_with_context(&context, &extension, &resources).unwrap();
    assert!(chelis_checker_core::check_ir_with_context(&simple_context, &extension, &resources).is_ok());
    assert!(chelis_checker_core::check_linearity_with_context(library_completion.facts(), extension_completion.facts(), &resources).is_ok());
    let (layered, _) = chelis_checker_core::build_compiled_library_context_with_base(&context, &extension, &resources).unwrap();
    let use_layer = parse("(def {} final_value (var {} use_id))");
    assert!(chelis_checker_core::check_ir_with_context(&layered, &use_layer, &resources).is_ok());
    let missing = parse("(def {span: \"oracle:missing\"} missing (var {} absent))");
    let error = chelis_checker_core::check_ir_program(&missing, &resources).err().expect("The missing name must reject.");
    assert!(!error.errors.is_empty());
    assert!(chelis_checker_core::check_ir_with_context(&context, &extension, &resources).is_ok());

    let nested = Nested { expressions: &program, calls: Cell::new(0) };
    let mut observed = Resources::new(&stack);
    observed.observer = Some(&nested);
    assert!(chelis_checker_core::check_ir_program(&declared, &observed).is_ok());
    assert!(nested.calls.get() > 0);

    let cancelled = CancelAfter { polls: Cell::new(0), limit: 0 };
    let mut cancelled_resources = Resources::new(&stack);
    cancelled_resources.cancellation = Some(&cancelled);
    assert!(chelis_checker_core::check_ir_program(&program, &cancelled_resources).is_err());
    assert!(cancelled.polls.get() > 0);
    let group = recursive_group();
    let interrupted = CancelAfter { polls: Cell::new(0), limit: 3 };
    cancelled_resources.cancellation = Some(&interrupted);
    assert!(chelis_checker_core::check_ir_program(&group, &cancelled_resources).is_err());
    assert!(interrupted.polls.get() > 3);
    assert!(chelis_checker_core::check_ir_program(&group, &resources).is_ok());
    assert!(chelis_checker_core::check_ir_with_context(&context, &extension, &resources).is_ok());
    assert!(chelis_checker_core::check_ir_program(&program, &Resources::new(&NoStack)).is_err());
    assert!(chelis_checker_core::check_ir_program(&program, &resources).is_ok());
    let deep_source = String::from("(def {} deep ")
        + &"(tuple {} ".repeat(64) + "(lit {} 1)" + &")".repeat(65);
    let deep = parse(&deep_source);
    assert!(chelis_checker_core::check_ir_program(&deep, &resources).is_err());
    assert!(chelis_checker_core::check_ir_program(&program, &resources).is_ok());

    let authored_linker_name = parse("(deftype {opaque: true} Pkg__foo__Secret () \
        (variant {} Pkg__foo__Secret (field {} value (t-prim {} f32)))) \
        (defsig {} pkg__foo__forge (t-fn {} (t-prim {} f32) (t-adt {} Pkg__foo__Secret))) \
        (def {} pkg__foo__forge (fn {} (params {} (x {type: (t-prim {} f32)})) \
        (record {} Pkg__foo__Secret (kv {} value (var {} x)))))");
    let rejected = chelis_checker_core::infer_ir_program(&authored_linker_name, &resources);
    assert!(rejected.errors.iter().any(|error| error.kind.diagnostic_name() == "ReservedLinkerName"));
    let mut linked_resources = Resources::new(&stack);
    linked_resources.policy.linked_program = true;
    let linked = chelis_checker_core::infer_ir_program(&authored_linker_name, &linked_resources);
    assert!(!linked.errors.iter().any(|error| error.kind.diagnostic_name() == "ReservedLinkerName"));
    let repeated = chelis_checker_core::infer_ir_program(&authored_linker_name, &resources);
    assert!(repeated.errors.iter().any(|error| error.kind.diagnostic_name() == "ReservedLinkerName"));

    let descriptor = chelis_abi::metadata::ShapeMetadata::contiguous(&[2, 3], chelis_vocab::RuntimeDType::F32).unwrap();
    assert_eq!(descriptor.elements().get(), 6);
    assert!(chelis_abi::metadata::ShapeMetadata::contiguous(&[0, -1], chelis_vocab::RuntimeDType::F32).is_err());
    assert!(chelis_abi::metadata::ShapeMetadata::contiguous(&[0, i64::MAX, 2], chelis_vocab::RuntimeDType::F32).is_err());
    facts
}
