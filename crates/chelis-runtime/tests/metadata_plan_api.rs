//! [05-OP-33]: metadata-only C plans own immutable projections, never payload.
use chelis_runtime::*;
use std::{env, process::Command, ptr};

fn tagged(value: i64) -> chelis_scalar {
    chelis_scalar_from_bits(CHELIS_DTYPE_I64, value as u64)
}
fn dimensions(values: &[i64]) -> Vec<chelis_scalar> {
    values.iter().map(|&value| tagged(value)).collect()
}
fn exemplar(dtype: chelis_dtype) -> chelis_scalar {
    chelis_scalar_from_bits(dtype, 0)
}

#[test]
fn plans_own_exact_immutable_metadata_for_all_representations() {
    unsafe {
        for (dtype, width) in [
            (CHELIS_DTYPE_F32, 4),
            (CHELIS_DTYPE_F64, 8),
            (CHELIS_DTYPE_I32, 4),
            (CHELIS_DTYPE_I64, 8),
            (CHELIS_DTYPE_BOOL, 1),
            (CHELIS_DTYPE_F16, 2),
            (CHELIS_DTYPE_BF16, 2),
            (CHELIS_DTYPE_I8, 1),
            (CHELIS_DTYPE_I16, 2),
        ] {
            let mut shape = dimensions(&[2, 3]);
            let plan = chelis_metadata_plan_new(tagged(2), shape.as_ptr(), exemplar(dtype));
            shape[0] = tagged(99);
            assert_eq!(chelis_metadata_plan_rank(plan), 2);
            assert_eq!(chelis_metadata_plan_dtype(plan), dtype);
            assert_eq!(chelis_metadata_plan_count(plan), 6);
            assert_eq!(chelis_metadata_plan_byte_count(plan), 6 * width);
            let extents = chelis_metadata_plan_shape(plan);
            let strides = chelis_metadata_plan_strides(plan);
            assert_eq!((extents.read(), extents.add(1).read()), (2, 3));
            assert_eq!((strides.read(), strides.add(1).read()), (3, 1));
            assert_eq!(extents, chelis_metadata_plan_shape(plan));
            chelis_metadata_plan_check_capacity(plan, tagged(6 * width));
            chelis_metadata_plan_release(plan);
        }
    }
}

#[test]
fn scalar_empty_and_dynamic_rank_plans_need_no_tensor_storage() {
    unsafe {
        let scalar = chelis_metadata_plan_new(tagged(0), ptr::null(), exemplar(CHELIS_DTYPE_F64));
        assert_eq!(chelis_metadata_plan_count(scalar), 1);
        assert_eq!(chelis_metadata_plan_byte_count(scalar), 8);
        assert!(chelis_metadata_plan_shape(scalar).is_null());
        assert!(chelis_metadata_plan_strides(scalar).is_null());
        chelis_metadata_plan_release(scalar);
        for rank in [1, 8, 9, 33] {
            let shape = dimensions(&vec![1; rank]);
            let plan = chelis_metadata_plan_new(
                tagged(rank as i64),
                shape.as_ptr(),
                exemplar(CHELIS_DTYPE_I8),
            );
            assert_eq!(chelis_metadata_plan_rank(plan), rank as i32);
            chelis_metadata_plan_release(plan);
        }
        let shape = dimensions(&[0, i64::MAX, i64::MAX]);
        let strides = dimensions(&[i64::MAX; 3]);
        let empty = chelis_metadata_plan_view(
            tagged(3),
            shape.as_ptr(),
            strides.as_ptr(),
            exemplar(CHELIS_DTYPE_F64),
            tagged(0),
        );
        assert_eq!(chelis_metadata_plan_count(empty), 0);
        assert_eq!(chelis_metadata_plan_byte_count(empty), 0);
        chelis_metadata_plan_release(empty);
        let wide = dimensions(&[i64::from(i32::MAX) + 1]);
        let plan = chelis_metadata_plan_new(tagged(1), wide.as_ptr(), exemplar(CHELIS_DTYPE_I8));
        assert_eq!(chelis_metadata_plan_count(plan), i64::from(i32::MAX) + 1);
        chelis_metadata_plan_release(plan);
    }
}

#[test]
fn view_capacity_is_reachable_span_and_caller_arrays_are_not_retained() {
    unsafe {
        let mut shape = dimensions(&[1_000, 3]);
        let mut strides = dimensions(&[0, 1]);
        let view = chelis_metadata_plan_view(
            tagged(2),
            shape.as_ptr(),
            strides.as_ptr(),
            exemplar(CHELIS_DTYPE_F64),
            tagged(24),
        );
        shape[0] = tagged(0);
        strides[1] = tagged(99);
        assert_eq!(chelis_metadata_plan_count(view), 3_000);
        assert_eq!(chelis_metadata_plan_byte_count(view), 24_000);
        assert_eq!(chelis_metadata_plan_shape(view).read(), 1_000);
        assert_eq!(chelis_metadata_plan_strides(view).add(1).read(), 1);
        chelis_metadata_plan_check_capacity(view, tagged(24));
        chelis_metadata_plan_release(view);
    }
}

#[test]
fn malformed_plan_inputs_trap_before_payload_allocation_or_projection() {
    const CHILD: &str = "CHELIS_METADATA_PLAN_CHILD";
    if let Ok(case) = env::var(CHILD) {
        unsafe {
            let mut rank = tagged(2);
            let mut shape = dimensions(&[2, 3]);
            let mut strides = dimensions(&[3, 1]);
            let mut dtype = exemplar(CHELIS_DTYPE_F64);
            let mut bytes = tagged(48);
            match case.as_str() {
                "negative-rank" => rank = tagged(-1),
                "rank-overflow" => rank = tagged(i64::from(i32::MAX) + 1),
                "rank-tag" => rank = exemplar(CHELIS_DTYPE_F64),
                "negative-extent" => shape[0] = tagged(-1),
                "extent-tag" => shape[0] = exemplar(CHELIS_DTYPE_F32),
                "negative-stride" => strides[0] = tagged(-1),
                "stride-tag" => strides[0] = exemplar(CHELIS_DTYPE_F32),
                "negative-capacity" => bytes = tagged(-1),
                "capacity-tag" => bytes = exemplar(CHELIS_DTYPE_F64),
                "short-capacity" => bytes = tagged(47),
                "exemplar-reserved" => dtype.reserved[0] = 1,
                "exemplar-tag" => dtype.dtype = 255,
                "exemplar-bool" => {
                    dtype = exemplar(CHELIS_DTYPE_BOOL);
                    dtype.bits = 2;
                }
                "product-overflow" => shape = dimensions(&[i64::MAX, 2]),
                "span-overflow" => strides[0] = tagged(i64::MAX),
                "empty-negative-stride" => {
                    shape[0] = tagged(0);
                    strides[0] = tagged(-1);
                }
                "null-plan" => {
                    chelis_metadata_plan_count(ptr::null());
                    return;
                }
                _ => {}
            }
            let shape_ptr = if case == "null-shape" {
                ptr::null()
            } else if case == "misaligned-shape" {
                shape.as_ptr().cast::<u8>().add(1).cast()
            } else {
                shape.as_ptr()
            };
            let stride_ptr = if case == "null-strides" {
                ptr::null()
            } else {
                strides.as_ptr()
            };
            let plan = chelis_metadata_plan_view(rank, shape_ptr, stride_ptr, dtype, bytes);
            if case == "capacity-recheck" {
                chelis_metadata_plan_check_capacity(plan, tagged(47));
            }
            chelis_metadata_plan_release(plan);
        }
        return;
    }
    for (case, class) in [
        ("negative-rank", "domain"),
        ("rank-overflow", "overflow"),
        ("rank-tag", "domain"),
        ("negative-extent", "domain"),
        ("extent-tag", "domain"),
        ("negative-stride", "domain"),
        ("stride-tag", "domain"),
        ("negative-capacity", "domain"),
        ("capacity-tag", "domain"),
        ("short-capacity", "domain"),
        ("exemplar-reserved", "domain"),
        ("exemplar-tag", "domain"),
        ("exemplar-bool", "domain"),
        ("product-overflow", "overflow"),
        ("span-overflow", "overflow"),
        ("empty-negative-stride", "domain"),
        ("null-plan", "domain"),
        ("null-shape", "domain"),
        ("misaligned-shape", "domain"),
        ("null-strides", "domain"),
        ("capacity-recheck", "domain"),
    ] {
        let output = Command::new(env::current_exe().unwrap())
            .env(CHILD, case)
            .args([
                "--exact",
                "malformed_plan_inputs_trap_before_payload_allocation_or_projection",
                "--nocapture",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{case} returned success");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("numeric trap: {class} in metadata_plan at int64")),
            "{case}: {stderr}"
        );
        assert!(!stderr.contains("panicked at"), "{case}: {stderr}");
    }
}
