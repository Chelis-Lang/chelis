//! [05-OP-33]: allocate-like uses checked input shape and explicit output representation.
use chelis_runtime::*;
use chelis_vocab::RuntimeDType;
use std::{env, process::Command, ptr};
#[test]
fn alloc_like_preserves_shape_and_checks_each_result_representation() {
    unsafe {
        for source in RuntimeDType::ALL {
            for target in RuntimeDType::ALL {
                for shape in [
                    vec![],
                    vec![3],
                    vec![1; 8],
                    vec![1; 9],
                    vec![i64::from(i32::MAX) + 2, 0],
                ] {
                    let input = chelis_alloc(
                        shape.len() as i32,
                        shape.as_ptr(),
                        source.id() as chelis_dtype,
                    );
                    let guard = chelis_tensor_begin_write(input);
                    let output = chelis_tensor_alloc_like(
                        input,
                        chelis_scalar_from_bits(target.id() as chelis_dtype, 0),
                    );
                    chelis_tensor_end_write(guard);
                    chelis_tensor_release(input);
                    let view = chelis_tensor_read_view(output);
                    assert_eq!(view.dtype, target.id() as chelis_dtype);
                    assert_eq!(chelis_tensor_rank(output), shape.len() as i32);
                    for (axis, extent) in shape.iter().enumerate() {
                        assert_eq!(chelis_tensor_shape(output, axis as i32), *extent);
                    }
                    if view.count == 0 {
                        assert!(view.data.is_null());
                    } else {
                        assert!(std::slice::from_raw_parts(
                            view.data.cast::<u8>(),
                            chelis_tensor_byte_count(output) as usize
                        )
                        .iter()
                        .all(|byte| *byte == 0));
                    }
                    chelis_tensor_release(output);
                }
            }
        }
    }
}
#[test]
fn invalid_alloc_like_child() {
    let Ok(case) = env::var("CHELIS_ALLOC_LIKE_CHILD") else {
        return;
    };
    unsafe {
        let mut virtual_byte = 0_i8;
        let input = if case == "null" {
            ptr::null_mut()
        } else if case == "bytes" {
            // A metadata-only virtual range: allocate-like must reject the f64
            // projection before allocation, and never reads these input bytes.
            chelis_tensor_entry_borrow(
                1,
                [i64::MAX].as_ptr(),
                CHELIS_DTYPE_I8,
                (&mut virtual_byte as *mut i8).cast(),
                i64::MAX,
            )
        } else {
            chelis_alloc(1, [1].as_ptr(), CHELIS_DTYPE_I8)
        };
        let mut exemplar = chelis_scalar_from_bits(CHELIS_DTYPE_F64, u64::from(case == "nonzero"));
        if case == "reserved" {
            exemplar.reserved[0] = 1;
        }
        chelis_tensor_alloc_like(input, exemplar);
        panic!("invalid allocate-like returned");
    }
}
#[test]
fn alloc_like_rejects_invalid_input_or_exemplar() {
    for case in ["null", "nonzero", "reserved", "bytes"] {
        let out = Command::new(env::current_exe().unwrap())
            .args(["--exact", "invalid_alloc_like_child", "--nocapture"])
            .env("CHELIS_ALLOC_LIKE_CHILD", case)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(1), "{case}: {out:?}");
        let expected = match case {
            "null" => "Domain: chelis_tensor_alloc_like",
            "bytes" => "numeric trap: overflow in alloc_like at int64",
            _ => "numeric trap: domain in alloc_like at int64",
        };
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(expected),
            "{case}: {out:?}"
        );
    }
}
