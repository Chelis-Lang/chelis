//! [05-OP-33]: literal buffers match checked result metadata before allocation.
use chelis_runtime::*;
use std::{env, process::Command};
fn int(n: i64) -> chelis_scalar {
    chelis_scalar_from_bits(CHELIS_DTYPE_I64, n as u64)
}
#[test]
fn literal_domains_accept_exact_counts_across_all_storage_representations() {
    unsafe {
        for dtype in [
            CHELIS_DTYPE_F32,
            CHELIS_DTYPE_F64,
            CHELIS_DTYPE_F16,
            CHELIS_DTYPE_BF16,
            CHELIS_DTYPE_I8,
            CHELIS_DTYPE_I16,
            CHELIS_DTYPE_I32,
            CHELIS_DTYPE_I64,
            CHELIS_DTYPE_BOOL,
        ] {
            let exemplar = chelis_scalar_from_bits(dtype, 0);
            chelis_tensor_check_literal(int(0), std::ptr::null(), exemplar, int(1));
            chelis_tensor_check_literal(int(2), [int(2), int(3)].as_ptr(), exemplar, int(6));
            chelis_tensor_check_literal(
                int(3),
                [int(2), int(0), int(3)].as_ptr(),
                exemplar,
                int(0),
            );
        }
    }
}
#[test]
fn invalid_literal_child() {
    let Ok(case) = env::var("CHELIS_LITERAL_CHILD") else {
        return;
    };
    unsafe {
        let rank = if case == "rank" { int(-1) } else { int(2) };
        let a = match case.as_str() {
            "negative" => -1,
            "product" => i64::MAX,
            "bytes" => i64::MAX / 4,
            _ => 2,
        };
        let count = match case.as_str() {
            "count" => -1,
            "mismatch" => 5,
            "wrong-tag" => 6,
            _ => 6,
        };
        let count = if case == "wrong-tag" {
            chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0)
        } else {
            int(count)
        };
        let exemplar = chelis_scalar_from_bits(
            CHELIS_DTYPE_F64,
            if case == "exemplar" { 1_u64 << 63 } else { 0 },
        );
        let shape = [int(a), int(3)];
        let ptr = if case == "null" {
            std::ptr::null()
        } else {
            shape.as_ptr()
        };
        chelis_tensor_check_literal(rank, ptr, exemplar, count);
        panic!("invalid literal returned");
    }
}
#[test]
fn literal_metadata_failures_have_the_canonical_const_identity() {
    for case in [
        "rank",
        "negative",
        "product",
        "bytes",
        "count",
        "mismatch",
        "wrong-tag",
        "null",
        "exemplar",
    ] {
        let output = Command::new(env::current_exe().unwrap())
            .args(["--exact", "invalid_literal_child", "--nocapture"])
            .env("CHELIS_LITERAL_CHILD", case)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{case}: {output:?}");
        let class = if matches!(case, "product" | "bytes") {
            "overflow"
        } else {
            "domain"
        };
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .lines()
                .any(|s| s == format!("numeric trap: {class} in const at int64")),
            "{case}: {output:?}"
        );
    }
}

#[test]
fn literal_writes_preserve_width_and_leave_the_guard_live() {
    unsafe {
        for dtype in [
            CHELIS_DTYPE_F32,
            CHELIS_DTYPE_F64,
            CHELIS_DTYPE_F16,
            CHELIS_DTYPE_BF16,
            CHELIS_DTYPE_I8,
            CHELIS_DTYPE_I16,
            CHELIS_DTYPE_I32,
            CHELIS_DTYPE_I64,
            CHELIS_DTYPE_BOOL,
        ] {
            for count in [3, 0] {
                let tensor = chelis_alloc(1, [count].as_ptr(), dtype);
                let guard = chelis_tensor_begin_write(tensor);
                let values = [
                    chelis_scalar_from_bits(dtype, 0),
                    chelis_scalar_from_bits(dtype, 1),
                    chelis_scalar_from_bits(dtype, 1),
                ];
                chelis_tensor_write_literal(
                    guard,
                    int(count),
                    if count == 0 {
                        std::ptr::null()
                    } else {
                        values.as_ptr()
                    },
                );
                let view = chelis_tensor_write_view(guard);
                assert_eq!(view.count, count);
                for (index, value) in values.iter().take(count as usize).enumerate() {
                    let bits = match chelis_dtype_size(dtype) {
                        8 => view.data.cast::<u64>().add(index).read(),
                        4 => u64::from(view.data.cast::<u32>().add(index).read()),
                        2 => u64::from(view.data.cast::<u16>().add(index).read()),
                        1 => u64::from(view.data.cast::<u8>().add(index).read()),
                        _ => panic!("unexpected representation"),
                    };
                    assert_eq!(bits, value.bits);
                }
                chelis_tensor_end_write(guard);
                chelis_tensor_release(tensor);
            }
        }
    }
}
static LITERAL_DESTINATION: std::sync::atomic::AtomicPtr<u32> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
extern "C" fn verify_literal_failure_left_destination_unchanged() {
    let pointer = LITERAL_DESTINATION.load(std::sync::atomic::Ordering::Relaxed);
    unsafe {
        for i in 0..3 {
            if pointer.add(i).read() != 1_f32.to_bits() {
                libc::_exit(92);
            }
        }
    }
}
#[test]
fn invalid_literal_write_child() {
    let Ok(case) = env::var("CHELIS_LITERAL_WRITE_CHILD") else {
        return;
    };
    unsafe {
        let tensor = chelis_alloc(1, [3].as_ptr(), CHELIS_DTYPE_F32);
        let guard = chelis_tensor_begin_write(tensor);
        chelis_fill_scalar(
            guard,
            chelis_scalar_from_bits(CHELIS_DTYPE_F32, u64::from(1_f32.to_bits())),
        );
        LITERAL_DESTINATION.store(
            chelis_tensor_write_view(guard).data.cast(),
            std::sync::atomic::Ordering::Relaxed,
        );
        assert_eq!(
            libc::atexit(verify_literal_failure_left_destination_unchanged),
            0
        );
        let mut values = [chelis_scalar_from_bits(CHELIS_DTYPE_F32, u64::from(2_f32.to_bits())); 3];
        match case.as_str() {
            "last-dtype" => values[2] = int(1),
            "last-reserved" => values[2].reserved[0] = 1,
            "last-unused" => values[2].bits = 1_u64 << 32,
            "last-bool" => {
                values[2] = chelis_scalar {
                    dtype: CHELIS_DTYPE_BOOL,
                    reserved: [0; 7],
                    bits: 2,
                }
            }
            "count" | "null" => {}
            _ => panic!("unknown child"),
        }
        chelis_tensor_write_literal(
            guard,
            int(if case == "count" { 2 } else { 3 }),
            if case == "null" {
                std::ptr::null()
            } else {
                values.as_ptr()
            },
        );
        panic!("invalid literal write returned");
    }
}
#[test]
fn every_literal_carrier_is_validated_before_any_destination_write() {
    for case in [
        "last-dtype",
        "last-reserved",
        "last-unused",
        "last-bool",
        "count",
        "null",
    ] {
        let output = Command::new(env::current_exe().unwrap())
            .args(["--exact", "invalid_literal_write_child", "--nocapture"])
            .env("CHELIS_LITERAL_WRITE_CHILD", case)
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(1),
            "{case} must fail with unchanged destination: {output:?}"
        );
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .lines()
                .any(|s| s == "numeric trap: domain in const at int64"),
            "{case}: {output:?}"
        );
    }
}
