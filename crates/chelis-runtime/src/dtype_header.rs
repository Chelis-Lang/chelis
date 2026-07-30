//! Renders the checked-in C dtype header from the Rust vocabulary.
//!
//! This module owns the renderer because `chelis-runtime` owns the generated
//! header. The dependency-bottom vocabulary contains no source renderer.

use chelis_vocab::RuntimeDType;

/// Renders `include/chelis_runtime_dtype.h`.
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
