//! [05-OP-79]: compiled `mmap_tensor` stores each payload element's exact
//! bits. Rendering cannot show a NaN's payload or its quiet/signaling bit, so
//! this reads the stored elements back through the runtime's read view.

use std::ffi::CString;
use std::fs;

use chelis_runtime::{
    chelis_mapped_file_release, chelis_mmap_file, chelis_mmap_tensor, chelis_string_from_cstr,
    chelis_string_release, chelis_tensor, chelis_tensor_read_view, chelis_tensor_release,
    CHELIS_DTYPE_BF16, CHELIS_DTYPE_F16, CHELIS_DTYPE_F32, CHELIS_DTYPE_F64,
};

unsafe fn stored<T: Copy>(tensor: *const chelis_tensor) -> Vec<T> {
    let view = chelis_tensor_read_view(tensor);
    std::slice::from_raw_parts(view.data.cast::<T>(), view.count as usize).to_vec()
}

#[test]
fn nan_payloads_and_signaling_bits_survive_the_read() {
    let f64_bits = [0x7ff0_0000_0000_0001u64, 0xfff8_dead_beef_0000];
    let f32_bits = [0x7f80_0001u32, 0xffc0_1234];
    let f16_bits = [0x7c01u16, 0xfe01];
    let bf16_bits = [0x7f81u16, 0xffc1];
    // One byte of prefix leaves every payload unaligned.
    let mut bytes = vec![0xaau8];
    bytes.extend(f64_bits.iter().flat_map(|bits| bits.to_le_bytes()));
    bytes.extend(f32_bits.iter().flat_map(|bits| bits.to_le_bytes()));
    bytes.extend(f16_bits.iter().flat_map(|bits| bits.to_le_bytes()));
    bytes.extend(bf16_bits.iter().flat_map(|bits| bits.to_le_bytes()));
    let path = std::env::temp_dir().join(format!(
        "chelis-mmap-tensor-bits-{}.bin",
        std::process::id()
    ));
    fs::write(&path, &bytes).expect("write payload");
    unsafe {
        let path_text = CString::new(path.to_str().unwrap()).unwrap();
        let path_value = chelis_string_from_cstr(path_text.as_ptr());
        let mapped = chelis_mmap_file(path_value);
        chelis_string_release(path_value);
        let f64s = chelis_mmap_tensor(mapped, 1, 2, CHELIS_DTYPE_F64);
        let f32s = chelis_mmap_tensor(mapped, 17, 2, CHELIS_DTYPE_F32);
        let f16s = chelis_mmap_tensor(mapped, 25, 2, CHELIS_DTYPE_F16);
        let bf16s = chelis_mmap_tensor(mapped, 29, 2, CHELIS_DTYPE_BF16);
        assert_eq!(stored::<u64>(f64s), f64_bits);
        assert_eq!(stored::<u32>(f32s), f32_bits);
        assert_eq!(stored::<u16>(f16s), f16_bits);
        assert_eq!(stored::<u16>(bf16s), bf16_bits);
        for tensor in [f64s, f32s, f16s, bf16s] {
            chelis_tensor_release(tensor);
        }
        chelis_mapped_file_release(mapped);
    }
    let _ = fs::remove_file(&path);
}
