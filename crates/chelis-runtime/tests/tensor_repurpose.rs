#![cfg(feature = "ownership-ledger")]

//! [05-OP-44] exact descriptor repurposing for compiler-proved storage reuse.

use std::fs;
use std::process::{Command, Output};

use chelis_runtime::{
    chelis_alloc, chelis_scalar, chelis_tensor_begin_write, chelis_tensor_end_write,
    chelis_tensor_entry_borrow, chelis_tensor_numel, chelis_tensor_rank, chelis_tensor_read_view,
    chelis_tensor_release, chelis_tensor_repurpose, chelis_tensor_retain, chelis_tensor_shape,
    CHELIS_DTYPE_F32, CHELIS_DTYPE_I64,
};

const CHILD_CASE: &str = "CHELIS_TENSOR_REPURPOSE_CHILD_CASE";
const LEDGER_PATH: &str = "CHELIS_OWNERSHIP_LEDGER_PATH";

fn ledger_path(case: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "chelis-tensor-repurpose-{}-{case}.jsonl",
        std::process::id()
    ))
}

fn run_child(case: &str, ledger: &std::path::Path) -> Output {
    let _ = fs::remove_file(ledger);
    Command::new(std::env::current_exe().expect("current test executable"))
        .args(["--exact", "tensor_repurpose_contract_child", "--nocapture"])
        .env(CHILD_CASE, case)
        .env(LEDGER_PATH, ledger)
        .output()
        .unwrap_or_else(|error| panic!("run tensor-repurpose child `{case}`: {error}"))
}

fn i64_scalar(value: i64) -> chelis_scalar {
    chelis_scalar {
        dtype: CHELIS_DTYPE_I64,
        reserved: [0; 7],
        bits: u64::from_ne_bytes(value.to_ne_bytes()),
    }
}

fn tagged_shape(shape: &[i64]) -> Vec<chelis_scalar> {
    shape.iter().copied().map(i64_scalar).collect()
}

#[test]
fn tensor_repurpose_contract_child() {
    let Ok(case) = std::env::var(CHILD_CASE) else {
        return;
    };
    unsafe {
        let original = [2_i64, 3];
        let repurposed = [3_i64, 2];
        let repurposed_tagged = tagged_shape(&repurposed);
        let repurposed_rank = i64_scalar(repurposed.len() as i64);
        match case.as_str() {
            "balanced" => {
                let tensor = chelis_alloc(2, original.as_ptr(), CHELIS_DTYPE_F32);
                let before = chelis_tensor_read_view(tensor).data;
                chelis_tensor_repurpose(tensor, repurposed_rank, repurposed_tagged.as_ptr());
                let after = chelis_tensor_read_view(tensor).data;
                assert_eq!(
                    before, after,
                    "repurpose must preserve the storage allocation"
                );
                assert_eq!(chelis_tensor_rank(tensor), 2);
                assert_eq!(chelis_tensor_shape(tensor, 0), 3);
                assert_eq!(chelis_tensor_shape(tensor, 1), 2);
                assert_eq!(chelis_tensor_numel(tensor), 6);
                chelis_tensor_release(tensor);
            }
            "shared-descriptor" => {
                let tensor = chelis_alloc(2, original.as_ptr(), CHELIS_DTYPE_F32);
                chelis_tensor_retain(tensor);
                chelis_tensor_repurpose(tensor, repurposed_rank, repurposed_tagged.as_ptr());
                panic!("repurpose accepted a shared descriptor");
            }
            "entry-borrow" => {
                let data = [0.0_f32; 6];
                let tensor = chelis_tensor_entry_borrow(
                    2,
                    original.as_ptr(),
                    CHELIS_DTYPE_F32,
                    data.as_ptr().cast(),
                    24,
                );
                chelis_tensor_repurpose(tensor, repurposed_rank, repurposed_tagged.as_ptr());
                panic!("repurpose accepted caller-owned storage");
            }
            "active-guard" => {
                let tensor = chelis_alloc(2, original.as_ptr(), CHELIS_DTYPE_F32);
                let guard = chelis_tensor_begin_write(tensor);
                chelis_tensor_repurpose(tensor, repurposed_rank, repurposed_tagged.as_ptr());
                chelis_tensor_end_write(guard);
                panic!("repurpose accepted an active write guard");
            }
            "capacity-mismatch" => {
                let tensor = chelis_alloc(2, original.as_ptr(), CHELIS_DTYPE_F32);
                let mismatch = [7_i64];
                let mismatch = tagged_shape(&mismatch);
                chelis_tensor_repurpose(tensor, i64_scalar(1), mismatch.as_ptr());
                panic!("repurpose accepted a different byte capacity");
            }
            "rank-dtype" => {
                let tensor = chelis_alloc(2, original.as_ptr(), CHELIS_DTYPE_F32);
                let wrong_rank = chelis_scalar {
                    dtype: CHELIS_DTYPE_F32,
                    reserved: [0; 7],
                    bits: u64::from(2.0_f32.to_bits()),
                };
                chelis_tensor_repurpose(tensor, wrong_rank, repurposed_tagged.as_ptr());
                panic!("repurpose accepted a non-i64 rank");
            }
            "rank-reserved" => {
                let tensor = chelis_alloc(2, original.as_ptr(), CHELIS_DTYPE_F32);
                let mut wrong_rank = repurposed_rank;
                wrong_rank.reserved[0] = 1;
                chelis_tensor_repurpose(tensor, wrong_rank, repurposed_tagged.as_ptr());
                panic!("repurpose accepted nonzero scalar reserved bytes");
            }
            "shape-dtype" => {
                let tensor = chelis_alloc(2, original.as_ptr(), CHELIS_DTYPE_F32);
                let mut wrong_shape = repurposed_tagged;
                wrong_shape[0].dtype = CHELIS_DTYPE_F32;
                chelis_tensor_repurpose(tensor, repurposed_rank, wrong_shape.as_ptr());
                panic!("repurpose accepted a non-i64 shape extent");
            }
            "shape-null" => {
                let tensor = chelis_alloc(2, original.as_ptr(), CHELIS_DTYPE_F32);
                chelis_tensor_repurpose(tensor, repurposed_rank, std::ptr::null());
                panic!("repurpose accepted a null positive-rank shape");
            }
            other => panic!("unknown tensor-repurpose child case `{other}`"),
        }
    }
}

#[test]
fn exact_same_capacity_repurpose_preserves_storage_and_balances_ownership() {
    let path = ledger_path("balanced");
    let output = run_child("balanced", &path);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ledger = fs::read_to_string(&path).expect("tensor repurpose ledger");
    assert!(ledger.contains(r#""live_owners":0"#), "{ledger}");
    assert!(ledger.contains(r#""live_bytes":0"#), "{ledger}");
    assert!(ledger.contains(r#""invalid_operations":0"#), "{ledger}");
    let _ = fs::remove_file(path);
}

#[test]
fn repurpose_rejects_every_missing_runtime_precondition() {
    for case in [
        "shared-descriptor",
        "entry-borrow",
        "active-guard",
        "capacity-mismatch",
        "rank-dtype",
        "rank-reserved",
        "shape-dtype",
        "shape-null",
    ] {
        let path = ledger_path(case);
        let output = run_child(case, &path);
        assert!(
            !output.status.success(),
            "case `{case}` unexpectedly succeeded"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("Domain: chelis_tensor_repurpose"),
            "{case}: {stderr}"
        );
        let _ = fs::remove_file(path);
    }
}
