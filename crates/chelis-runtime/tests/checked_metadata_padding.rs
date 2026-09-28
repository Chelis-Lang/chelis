//! [05-OP-9/10/33]: padding preserves exact extents, checked offsets, and bits.
use chelis_runtime::{
    chelis_dtype, chelis_list, chelis_list_from_values, chelis_list_index, chelis_list_len,
    chelis_list_release, chelis_pad_sequences, chelis_pad_sequences_to, chelis_scalar_from_bits,
    chelis_tensor, chelis_tensor_elements, chelis_tensor_numel, chelis_tensor_release,
    chelis_tensor_shape, chelis_value_box_scalar, chelis_value_release, chelis_value_take_list,
    chelis_value_unbox_scalar, CHELIS_DTYPE_F32,
};
use chelis_vocab::RuntimeDType;

unsafe fn rows(dtype: chelis_dtype, bits: u64) -> *mut chelis_list {
    let value = chelis_value_box_scalar(chelis_scalar_from_bits(dtype, bits));
    let first = chelis_list_from_values([value, value].as_ptr(), 2);
    let second = chelis_list_from_values([value].as_ptr(), 1);
    let values = [
        chelis_value_take_list(first),
        chelis_value_take_list(second),
    ];
    let out = chelis_list_from_values(values.as_ptr(), 2);
    for value in values {
        chelis_value_release(value);
    }
    out
}

unsafe fn bits(tensor: *const chelis_tensor) -> Vec<u64> {
    let elements = chelis_tensor_elements(tensor);
    let result = (0..chelis_list_len(elements))
        .map(|i| chelis_value_unbox_scalar(chelis_list_index(elements, i)).bits)
        .collect();
    chelis_list_release(elements);
    result
}

#[test]
fn padding_preserves_each_representation_and_empty_exact_extents() {
    unsafe {
        for dtype in RuntimeDType::ALL {
            let bit_pattern = match dtype {
                RuntimeDType::F64 => 0x8000_0000_0000_0000,
                RuntimeDType::F32 => 0x8000_0000,
                RuntimeDType::F16 | RuntimeDType::Bf16 => 0x8000,
                RuntimeDType::I64 => (1 << 53) + 17,
                RuntimeDType::I32 => 0x7fff_ffff,
                RuntimeDType::I16 => 0x7fff,
                RuntimeDType::I8 => 0x7f,
                RuntimeDType::Bool => 1,
                // A key has no [05-OP-31] scalar carrier, so no pad value.
                RuntimeDType::Key => continue,
            };
            let dtype = dtype.id() as chelis_dtype;
            let rows = rows(dtype, bit_pattern);
            let pad = chelis_scalar_from_bits(dtype, 0);
            let inferred = chelis_pad_sequences(rows, pad);
            assert_eq!(chelis_tensor_shape(inferred, 0), 2);
            assert_eq!(chelis_tensor_shape(inferred, 1), 2);
            assert_eq!(bits(inferred), [bit_pattern, bit_pattern, bit_pattern, 0]);
            let explicit = chelis_pad_sequences_to(rows, 3, pad);
            assert_eq!(chelis_tensor_shape(explicit, 1), 3);
            assert_eq!(
                bits(explicit),
                [bit_pattern, bit_pattern, 0, bit_pattern, 0, 0]
            );
            let empty = chelis_pad_sequences_to(rows, 0, pad);
            assert_eq!(chelis_tensor_numel(empty), 0);
            let no_rows = chelis_list_from_values(std::ptr::null(), 0);
            let wide_empty = chelis_pad_sequences_to(no_rows, i64::MAX, pad);
            assert_eq!(chelis_tensor_numel(wide_empty), 0);
            assert_eq!(chelis_tensor_shape(wide_empty, 1), i64::MAX);
            for tensor in [inferred, explicit, empty, wide_empty] {
                chelis_tensor_release(tensor);
            }
            chelis_list_release(rows);
            chelis_list_release(no_rows);
        }
    }
}

#[test]
fn padding_rejects_negative_width_and_unrepresentable_output_before_access() {
    const CASE: &str = "CHELIS_CHECKED_PADDING_CASE";
    if let Ok(case) = std::env::var(CASE) {
        unsafe {
            let input = rows(CHELIS_DTYPE_F32, 1.0_f32.to_bits().into());
            let width = match case.as_str() {
                "valid" => 1,
                "negative" => -1,
                "product" => i64::MAX,
                "bytes" => i64::MAX / 4,
                _ => panic!("unknown case"),
            };
            let out =
                chelis_pad_sequences_to(input, width, chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0));
            assert_eq!(case, "valid", "invalid padding returned a tensor");
            assert_eq!(chelis_tensor_numel(out), 2);
            chelis_tensor_release(out);
            chelis_list_release(input);
        }
        return;
    }
    for (case, reason) in [
        ("valid", None),
        ("negative", Some("Domain:")),
        ("product", Some("Overflow:")),
        ("bytes", Some("Overflow:")),
    ] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "padding_rejects_negative_width_and_unrepresentable_output_before_access",
                "--nocapture",
            ])
            .env(CASE, case)
            .output()
            .unwrap();
        if let Some(reason) = reason {
            assert_eq!(output.status.code(), Some(1), "{case}: {:?}", output.status);
            assert!(
                String::from_utf8_lossy(&output.stderr).contains(reason),
                "{case}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        } else {
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
