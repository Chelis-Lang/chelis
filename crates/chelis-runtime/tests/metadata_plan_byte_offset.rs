//! [05-OP-33]: actual tagged C entry and checked logical storage offsets.
use chelis_runtime::*;
use std::{env, process::Command, ptr};
fn tagged(value: i64) -> chelis_scalar {
    chelis_scalar_from_bits(CHELIS_DTYPE_I64, value as u64)
}

#[test]
fn c_plan_projects_exact_logical_offsets_for_all_layouts() {
    unsafe {
        for (shape, strides, capacity, expected) in [
            (vec![3, 2], vec![1, 3], 48, vec![0, 24, 8, 32, 16, 40]),
            (vec![2, 2], vec![4, 2], 56, vec![0, 16, 32, 48]),
            (vec![2, 3], vec![0, 1], 24, vec![0, 8, 16, 0, 8, 16]),
        ] {
            let shape: Vec<_> = shape.into_iter().map(tagged).collect();
            let strides: Vec<_> = strides.into_iter().map(tagged).collect();
            let plan = chelis_metadata_plan_view(
                tagged(2),
                shape.as_ptr(),
                strides.as_ptr(),
                chelis_scalar_from_bits(CHELIS_DTYPE_F64, 0),
                tagged(capacity),
            );
            let actual: Vec<_> = (0..chelis_metadata_plan_count(plan))
                .map(|index| chelis_metadata_plan_byte_offset(plan, tagged(index)))
                .collect();
            assert_eq!(actual, expected);
            chelis_metadata_plan_release(plan);
        }
        let scalar = chelis_metadata_plan_new(
            tagged(0),
            ptr::null(),
            chelis_scalar_from_bits(CHELIS_DTYPE_I8, 0),
        );
        assert_eq!(chelis_metadata_plan_byte_offset(scalar, tagged(0)), 0);
        chelis_metadata_plan_release(scalar);
    }
}

#[test]
fn c_plan_rejects_invalid_index_carriers_and_empty_access() {
    const CHILD: &str = "CHELIS_METADATA_OFFSET_CHILD";
    if let Ok(case) = env::var(CHILD) {
        unsafe {
            let shape = [tagged(if case == "empty" { 0 } else { 3 })];
            let plan = chelis_metadata_plan_new(
                tagged(1),
                shape.as_ptr(),
                chelis_scalar_from_bits(CHELIS_DTYPE_F64, 0),
            );
            let mut index = tagged(match case.as_str() {
                "negative" => -1,
                "end" => 3,
                _ => 0,
            });
            if case == "tag" {
                index = chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0);
            }
            if case == "reserved" {
                index.reserved[0] = 1;
            }
            chelis_metadata_plan_byte_offset(plan, index);
            chelis_metadata_plan_release(plan);
        }
        return;
    }
    for case in ["negative", "end", "empty", "tag", "reserved"] {
        let output = Command::new(env::current_exe().unwrap())
            .env(CHILD, case)
            .args([
                "--exact",
                "c_plan_rejects_invalid_index_carriers_and_empty_access",
                "--nocapture",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{case} returned success");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("numeric trap: domain in metadata_plan at int64"),
            "{case}: {stderr}"
        );
        assert!(!stderr.contains("panicked at"), "{case}: {stderr}");
    }
}
