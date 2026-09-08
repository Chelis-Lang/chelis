//! Renders the checked-in C dtype header from the Rust vocabulary.
//!
//! This module owns the renderer because `chelis-runtime` owns the generated
//! header. The dependency-bottom vocabulary contains no source renderer.

use chelis_vocab::RuntimeDType;

/// Renders `include/chelis_runtime_dtype.h`.
pub fn render_runtime_dtype_c_header() -> String {
    let mut header = String::from(
        "#ifndef CHELIS_RUNTIME_DTYPE_H\n#define CHELIS_RUNTIME_DTYPE_H\n\n#include <stdint.h>\n\ntypedef uint8_t chelis_dtype;\nenum {\n",
    );
    for (index, dtype) in RuntimeDType::ALL.iter().enumerate() {
        let comma = if index + 1 == RuntimeDType::ALL.len() {
            ""
        } else {
            ","
        };
        header.push_str(&format!(
            "    {} = {}{}\n",
            dtype.c_macro(),
            dtype.id(),
            comma
        ));
    }
    header.push_str("};\n\n#endif\n");
    header
}
