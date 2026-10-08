//! [05-OP-79]: the evaluator's `mmap_tensor` storage holds each payload
//! element's exact bits, NaN payloads and signaling bits included, and
//! refuses what no element dtype stores.

use chelis_types::dtype_semantics::StorageView;
use chelis_types::tensor_from_le_payload;
use chelis_types::types::Prim;

fn le<T: Copy, const N: usize>(values: &[T], bytes: fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|value| bytes(*value)).collect()
}

#[test]
fn float_bits_are_stored_unchanged() {
    let f64_bits = [0x7ff0_0000_0000_0001u64, 0x8000_0000_0000_0000];
    let f32_bits = [0x7f80_0001u32, 0xffc0_1234];
    let f16_bits = [0x7c01u16, 0x0001];
    let bf16_bits = [0x7f81u16, 0xffc1];
    let f64s = tensor_from_le_payload(Prim::F64, &le(&f64_bits, u64::to_le_bytes)).unwrap();
    let f32s = tensor_from_le_payload(Prim::F32, &le(&f32_bits, u32::to_le_bytes)).unwrap();
    let f16s = tensor_from_le_payload(Prim::F16, &le(&f16_bits, u16::to_le_bytes)).unwrap();
    let bf16s = tensor_from_le_payload(Prim::Bf16, &le(&bf16_bits, u16::to_le_bytes)).unwrap();
    let StorageView::F64(values) = f64s.view() else { panic!("f64 storage") };
    assert_eq!(values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), f64_bits);
    let StorageView::F32(values) = f32s.view() else { panic!("f32 storage") };
    assert_eq!(values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), f32_bits);
    let StorageView::F16(values) = f16s.view() else { panic!("f16 storage") };
    assert_eq!(values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), f16_bits);
    let StorageView::Bf16(values) = bf16s.view() else { panic!("bf16 storage") };
    assert_eq!(values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), bf16_bits);
}

#[test]
fn integers_and_bools_are_stored_unchanged() {
    let i16s = tensor_from_le_payload(Prim::Int16, &le(&[i16::MIN, -2], i16::to_le_bytes)).unwrap();
    let StorageView::I16(values) = i16s.view() else { panic!("i16 storage") };
    assert_eq!(values, [i16::MIN, -2]);
    let bools = tensor_from_le_payload(Prim::Bool, &[1, 0]).unwrap();
    let StorageView::Bool(values) = bools.view() else { panic!("bool storage") };
    assert_eq!(values, [1, 0]);
}

#[test]
fn what_no_element_stores_is_refused() {
    assert!(tensor_from_le_payload(Prim::Bool, &[2]).is_err());
    assert!(tensor_from_le_payload(Prim::F32, &[0, 0, 0]).is_err());
    assert!(tensor_from_le_payload(Prim::Key, &[0; 8]).is_err());
    assert!(tensor_from_le_payload(Prim::String, &[]).is_err());
}
