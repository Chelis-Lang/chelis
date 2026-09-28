//! chelis#2413: the byte-equality lock between the compiled lane's key text
//! ([05-OBS-2]) and the reference renderer
//! `chelis_types::observation::format_key_bits`, through the public C ABI
//! entry `chelis_string_from_key`. Recursive observation of a key tensor
//! shares the runtime's one key formatter; the compiled roots that reach it
//! are locked end to end in `chelis-compiler-api`'s `key_root_lanes`.

use chelis_runtime::{
    chelis_key, chelis_string_data, chelis_string_from_key, chelis_string_release,
};
use chelis_types::format_key_bits;

/// Seeds, the extremes, one bit per nibble position and a 64-bit sweep.
fn key_words() -> Vec<u64> {
    let mut words = vec![
        0,
        1,
        7,
        u64::MAX,
        0x8000_0000_0000_0000,
        0x1c3b_e871_ed9d_079c,
    ];
    words.extend((0..16).map(|nibble| 0xa_u64 << (nibble * 4)));
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..4096 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        words.push(state);
    }
    words
}

fn owned_text(rendered: chelis_runtime::chelis_string) -> String {
    unsafe {
        let text = std::ffi::CStr::from_ptr(chelis_string_data(rendered))
            .to_str()
            .expect("key text is UTF-8")
            .to_string();
        chelis_string_release(rendered);
        text
    }
}

#[test]
fn the_c_key_text_is_the_reference_key_text() {
    for bits in key_words() {
        let text = owned_text(chelis_string_from_key(chelis_key { bits }));
        assert_eq!(text, format_key_bits(bits), "bits {bits:#018x}");
        assert!(
            text.len() == 21 && text.starts_with("key(") && text.ends_with(')'),
            "{text}"
        );
    }
}
