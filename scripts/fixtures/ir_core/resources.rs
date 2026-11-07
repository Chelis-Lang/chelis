use std::cell::Cell;
use std::sync::{Arc, Barrier};
use chelis_checker_core::{check_ir_program, check_linearity};
use chelis_ir_core::host::{ManifestEntry, ManifestView, try_lower_compiled_program_with_manifest};
use chelis_ir_core::resources::Resources;
use chelis_stack_services::StackServices;

const STACK_BYTES: usize = 768 * 1024 * 1024;

struct Stack { available: Cell<usize> }
impl StackServices for Stack {
    fn grow<R>(&self, bytes: usize, operation: impl FnOnce() -> R) -> R {
        assert!(bytes <= STACK_BYTES);
        operation()
    }
    fn maybe_grow<R>(&self, red_zone: usize, bytes: usize, operation: impl FnOnce() -> R) -> R {
        assert!(red_zone <= STACK_BYTES && bytes <= STACK_BYTES);
        operation()
    }
    fn remaining_stack(&self) -> Option<usize> { Some(self.available.get()) }
}
struct Manifest;
impl ManifestView for Manifest {
    fn len(&self) -> usize { 1 }
    fn entry(&self, index: usize) -> ManifestEntry<'_> {
        assert_eq!(index, 0);
        ManifestEntry { def_name: "result", tensor_lane: false }
    }
}

fn run(barrier: Arc<Barrier>, label: &'static str) {
    let stack = Stack { available: Cell::new(STACK_BYTES) };
    let expressions = chelis_deep::parser::parse_str(
        "(def {} result (lit {type: (t-prim {} string)} \"portable\"))", &stack).unwrap();
    let mut resources = Resources::new(&stack);
    resources.checker_stack_bytes = 64 * 1024 * 1024;
    resources.run(|operation| {
        let checker = operation.checker_resources();
        let checked = check_ir_program(&expressions, &checker).unwrap();
        let facts = check_linearity(checked.facts(), &checker).unwrap();
        let binding = Some((label.to_string(), format!("symbol_{label}")));
        operation.host.with_native_provider_call(binding.clone(), || {
            barrier.wait();
            assert_eq!(operation.host.active_native_provider_call(), binding);
            let refusal: Result<(), &str> = operation.host.with_native_provider_call(None, || {
                assert_eq!(operation.host.active_native_provider_call(), None);
                Err("expected refusal")
            });
            assert_eq!(refusal, Err("expected refusal"));
            assert_eq!(operation.host.active_native_provider_call(), binding);
            resources.run(|child| assert_eq!(child.host.active_native_provider_call(), None));
            assert_eq!(operation.host.active_native_provider_call(), binding);
        });
        assert_eq!(operation.host.active_native_provider_call(), None);
        stack.available.set(1);
        let failure = try_lower_compiled_program_with_manifest(&facts, &Manifest, operation)
            .expect_err("insufficient stack must reject the host operation");
        assert!(failure.to_string().contains("host lowering stack budget exhausted"));
        stack.available.set(STACK_BYTES);
        let first = try_lower_compiled_program_with_manifest(&facts, &Manifest, operation)
            .expect("the same operation must recover its temporary state");
        let second = try_lower_compiled_program_with_manifest(&facts, &Manifest, operation).unwrap();
        assert_eq!(format!("{first:?}"), format!("{second:?}"));
        assert_eq!(first.host.unwrap().globals.len(), 1);
    });
}

fn main() {
    fn shared<T: Send + Sync>() {}
    shared::<chelis_ir_core::eval::TensorValue>();
    shared::<chelis_ir_core::host::PreparedHostFacts<chelis_checker_core::CheckFacts>>();
    let barrier = Arc::new(Barrier::new(2));
    let threads = ["left", "right"].map(|label| {
        let barrier = barrier.clone();
        std::thread::Builder::new().stack_size(STACK_BYTES)
            .spawn(move || run(barrier, label)).unwrap()
    });
    for thread in threads { thread.join().unwrap(); }
    println!("IR_RESOURCE_CONSUMER {{\"concurrent_operations\":2,\"nested_isolation\":true,\"expected_failure_restoration\":true,\"stack_refusal\":true,\"same_operation_retry\":true,\"send_sync\":true}}");
}
