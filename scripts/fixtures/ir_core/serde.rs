use chelis_checker_core::{check_ir_program, check_linearity};
use chelis_ir_core::lower::{LoweredLibrary, try_lower_program_to_library};
use chelis_ir_core::resources::Resources;
use chelis_stack_services::StackServices;

const STACK_BYTES: usize = 768 * 1024 * 1024;
struct Stack;
impl StackServices for Stack {
    fn grow<R>(&self, bytes: usize, operation: impl FnOnce() -> R) -> R {
        assert!(bytes <= STACK_BYTES);
        operation()
    }
    fn maybe_grow<R>(&self, red_zone: usize, bytes: usize, operation: impl FnOnce() -> R) -> R {
        assert!(red_zone < STACK_BYTES && bytes <= STACK_BYTES);
        operation()
    }
    fn remaining_stack(&self) -> Option<usize> { None }
}
fn run() {
    let stack = Stack;
    let expressions = chelis_deep::parser::parse_str(
        "(defsig {} value (t-prim {} i32)) (def {} value (lit {} 7))", &stack).unwrap();
    let mut resources = Resources::new(&stack);
    resources.checker_stack_bytes = 64 * 1024 * 1024;
    resources.run(|operation| {
        let checker = operation.checker_resources();
        let completed = check_ir_program(&expressions, &checker).unwrap();
        let facts = check_linearity(completed.facts(), &checker).unwrap();
        let library = try_lower_program_to_library(&facts, operation).unwrap();
        let bytes = bincode::serialize(&library).unwrap();
        let decoded: LoweredLibrary = bincode::deserialize(&bytes).unwrap();
        assert_eq!(bytes, bincode::serialize(&decoded).unwrap());
        assert!(bincode::deserialize::<LoweredLibrary>(&bytes[..bytes.len() - 1]).is_err());
        let mut incomplete = serde_json::to_value(&library).unwrap();
        assert!(incomplete.as_object_mut().unwrap().remove("local_ascriptions").is_some());
        let failure = serde_json::from_value::<LoweredLibrary>(incomplete).unwrap_err();
        assert!(failure.to_string().contains("local_ascriptions"));
        println!("IR_SERDE_CONSUMER {{\"exact_binary_roundtrip\":true,\"truncated_binary_rejected\":true,\"missing_obligations_rejected\":true}}");
    });
}
fn main() {
    std::thread::Builder::new().stack_size(STACK_BYTES).spawn(run).unwrap().join().unwrap();
}
