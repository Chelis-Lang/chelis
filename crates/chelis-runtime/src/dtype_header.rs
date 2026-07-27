//! Renders the checked-in C ABI dtype fragment from the Rust vocabulary.
//!
//! This lives here rather than in `chelis-vocab` because emitting source text
//! for another language is shell work, and `chelis-vocab` is the
//! dependency-bottom vocabulary: `#![no_std]`, allocation-free, and holding
//! only data plus total decode functions. The generator belongs with the
//! artifact it produces, which is `include/chelis_runtime_dtype.h` in this
//! crate.
//!
//! The generated decoder terminates on invalid input, so C callers cannot
//! reinterpret an unknown ABI tag as `f32`.

use chelis_vocab::RuntimeDType;

/// Regenerate with:
/// `cargo run -p chelis-runtime --example gen_dtype_header > crates/chelis-runtime/include/chelis_runtime_dtype.h`
pub fn render_runtime_dtype_c_header() -> String {
    let mut header = String::from(
        "#ifndef CHELIS_RUNTIME_DTYPE_H\n#define CHELIS_RUNTIME_DTYPE_H\n\n#include <stddef.h>\n#include <stdio.h>\n#include <stdlib.h>\n\n",
    );
    for dtype in RuntimeDType::ALL {
        header.push_str(&format!("#define {} {}\n", dtype.c_macro(), dtype.id()));
    }
    header.push_str(
        "\nstatic inline size_t chelis_runtime_dtype_size_checked(int dtype) {\n    switch (dtype) {\n",
    );
    for dtype in RuntimeDType::ALL {
        header.push_str(&format!(
            "        case {}: return {};\n",
            dtype.c_macro(),
            dtype.byte_width()
        ));
    }
    header.push_str(
        "        default:\n            fprintf(stderr, \"invalid Chelis runtime dtype id: %d\\n\", dtype);\n            abort();\n    }\n}\n\n#endif\n",
    );
    header
}
