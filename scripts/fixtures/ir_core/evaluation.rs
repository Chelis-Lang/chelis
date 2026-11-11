#![forbid(unsafe_code)]

extern crate alloc;

use alloc::vec::Vec;
use chelis_checker_core::Cancellation;
use chelis_dag_core::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir_core::eval::{
    InheritedResultAxis, InheritedResultClaim, TensorValue,
    eval_tensor_roots_exact_with_result_claims, eval_tensor_roots_with_strict,
};
use chelis_ir_core::resources::Resources;
use chelis_numeric_core::prim::Prim;
use chelis_stack_services::StackServices;
use core::sync::atomic::{AtomicBool, Ordering};

#[global_allocator]
static ALLOCATOR: std::alloc::System = std::alloc::System;

const STACK_BYTES: usize = 64 * 1024 * 1024;
const STACK_MARGIN: usize = 1024 * 1024;

// The platform reserves this stack before any core operation starts.
struct ReservedThreadStack {
    origin: usize,
}

impl ReservedThreadStack {
    fn available(&self) -> usize {
        let marker = std::hint::black_box(0_u8);
        let position = core::ptr::from_ref(&marker).addr();
        STACK_BYTES
            .saturating_sub(STACK_MARGIN)
            .saturating_sub(self.origin.abs_diff(position))
    }
}

impl StackServices for ReservedThreadStack {
    fn grow<R>(&self, bytes: usize, operation: impl FnOnce() -> R) -> R {
        assert!(bytes <= self.available(), "the thread must contain the reservation");
        operation()
    }

    fn maybe_grow<R>(&self, red_zone: usize, bytes: usize, operation: impl FnOnce() -> R) -> R {
        let available = self.available();
        assert!(available >= red_zone, "the fixture must remain above its red zone");
        assert!(bytes <= available, "the thread must contain the reservation");
        operation()
    }

    fn remaining_stack(&self) -> Option<usize> {
        Some(self.available())
    }
}

struct CancelFlag(AtomicBool);

impl Cancellation for CancelFlag {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

fn exact_tensor(values: [i64; 2]) -> TensorValue {
    TensorValue::finalize_from_wide_int("consumer", Prim::Int64, vec![2], values.to_vec())
        .expect("exact i64 ingress must succeed")
}

fn fixture() -> (Dag, NodeId, NodeId) {
    let mut dag = Dag::new();
    let selected = dag.declare("selected");
    let ty = TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision: Prim::Int64,
    };
    let a = dag.add_node(
        selected,
        RiscOp::Load { name: "a".into() },
        vec![],
        ty.clone(),
        None,
    );
    let b = dag.add_node(
        selected,
        RiscOp::Load { name: "b".into() },
        vec![],
        ty.clone(),
        None,
    );
    let sum = dag.add_node(selected, RiscOp::Add, vec![a, b], ty.clone(), None);
    dag.add_root(sum);
    let excluded = dag.declare("excluded");
    let other = dag.add_node(
        excluded,
        RiscOp::Load { name: "unselected".into() },
        vec![],
        ty,
        None,
    );
    dag.add_root(other);
    (dag, sum, other)
}

fn claim(root: NodeId, required: usize) -> InheritedResultClaim {
    InheritedResultClaim {
        root,
        rank: 1,
        axes: vec![InheritedResultAxis {
            axis: 0,
            required,
            source: None,
        }],
    }
}

fn run() {
    let anchor = std::hint::black_box(0_u8);
    let stack = ReservedThreadStack {
        origin: core::ptr::from_ref(&anchor).addr(),
    };
    assert!(stack.available() >= 32 * 1024 * 1024);
    let cancellation = CancelFlag(AtomicBool::new(false));
    let mut resources = Resources::new(&stack);
    resources.cancellation = Some(&cancellation);
    resources.checker_stack_bytes = 32 * 1024 * 1024;
    stack.grow(resources.checker_stack_bytes, || resources.run(|operation| {
        let (dag, root, excluded) = fixture();
        let base = 1_i64 << 53;
        let left = exact_tensor([base + 1, base + 2]);
        let right = exact_tensor([1, -1]);
        let expected = vec![base + 2, base + 1];
        assert_eq!(left.prim(), Prim::Int64);
        assert_eq!(left.storage().to_i64_exact_vec(), Some(vec![base + 1, base + 2]));
        let load = |name: &str| match name {
            "a" => Some(left.clone()),
            "b" => Some(right.clone()),
            _ => None,
        };

        let mut requested = Vec::new();
        let selected = eval_tensor_roots_with_strict(
            &dag,
            &[root],
            |name| {
                requested.push(name.to_string());
                load(name)
            },
            operation,
        )
        .expect("selected roots must not demand excluded inputs");
        assert_eq!(requested, ["a", "b"]);
        assert_eq!(selected.len(), 1);
        assert!(!selected.contains_key(&excluded));
        assert_eq!(selected[&root].prim(), Prim::Int64);
        let exact = selected[&root].storage().to_i64_exact_vec().unwrap();
        assert_eq!(exact, expected);

        let excluded_failure = eval_tensor_roots_with_strict(
            &dag,
            &[excluded],
            |_| None,
            operation,
        )
        .unwrap_err();
        assert_eq!(excluded_failure, "missing required input `unselected`");

        let valid_claim = eval_tensor_roots_exact_with_result_claims(
            &dag,
            &[root],
            &[claim(root, 2)],
            load,
            operation,
        )
        .expect("a matching result extent must succeed");
        assert_eq!(valid_claim[&root].storage().to_i64_exact_vec(), Some(expected.clone()));
        let claim_failure = eval_tensor_roots_exact_with_result_claims(
            &dag,
            &[root],
            &[claim(root, 3)],
            load,
            operation,
        )
        .unwrap_err();
        assert_eq!(
            claim_failure,
            "extent `3`: claimed = 3, add axis 0 = 2\nnumeric trap: domain in add at i64",
        );

        let overflow_left = exact_tensor([i64::MAX, base + 2]);
        let overflow_failure = eval_tensor_roots_with_strict(
            &dag,
            &[root],
            |name| match name {
                "a" => Some(overflow_left.clone()),
                "b" => Some(right.clone()),
                _ => None,
            },
            operation,
        )
        .unwrap_err();
        assert_eq!(overflow_failure, "numeric trap: overflow in add at i64");

        cancellation.0.store(true, Ordering::Relaxed);
        let cancelled = eval_tensor_roots_with_strict(&dag, &[root], load, operation).unwrap_err();
        assert_eq!(cancelled, chelis_checker_core::cancel::EVAL_CANCELLED_MSG);
        cancellation.0.store(false, Ordering::Relaxed);
        let retried = eval_tensor_roots_exact_with_result_claims(
            &dag,
            &[root],
            &[claim(root, 2)],
            load,
            operation,
        )
        .expect("the same operation must succeed after cancellation clears");
        assert_eq!(retried.len(), 1);
        assert_eq!(retried[&root].storage().to_i64_exact_vec(), Some(expected));

        println!(
            "IR_EVAL_CONSUMER {{\"exact_i64\":{:?},\"selected_roots\":1,\"excluded_failure\":{:?},\"claim_failure\":{:?},\"overflow_failure\":{:?},\"cancellation\":{:?},\"retry\":\"same_operation\",\"stack_reserved_bytes\":{}}}",
            exact,
            excluded_failure,
            claim_failure,
            overflow_failure,
            cancelled,
            STACK_BYTES,
        );
    }));
}

fn main() {
    std::thread::Builder::new()
        .name("ir-evaluation-consumer".into())
        .stack_size(STACK_BYTES)
        .spawn(run)
        .expect("the platform must reserve the consumer stack")
        .join()
        .expect("the evaluation consumer must succeed");
}
