use chelis_vocab::{RuntimeDType, RuntimeDTypeDecodeError};

use super::{chelis_tensor, read_index_slot, tensor_elem_size};

#[test]
fn runtime_dtype_ffi_decoder_rejects_invalid_ids_before_dispatch() {
    let first_unknown = RuntimeDType::ALL
        .iter()
        .map(|dtype| dtype.id())
        .max()
        .expect("non-empty runtime dtype vocabulary")
        + 1;
    for raw in [first_unknown as u8, u8::MAX] {
        assert_eq!(
            super::decode_runtime_dtype(raw),
            Err(RuntimeDTypeDecodeError::InvalidId { id: i32::from(raw) })
        );
    }
}

#[test]
fn internal_sizing_and_reading_helpers_require_a_decoded_dtype() {
    let _: fn(RuntimeDType) -> usize = tensor_elem_size;
    let _: unsafe fn(*const chelis_tensor, usize, RuntimeDType) -> i64 = read_index_slot;
}
