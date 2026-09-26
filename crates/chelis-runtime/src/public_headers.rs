//! The public runtime headers, exported from the runtime crate so a compiler
//! stages the headers of the same runtime compilation whose archive it carries
//! (`spec/08-backends.md` §2.1).

/// File name and contents of every public runtime header, in staging order.
pub const PUBLIC_HEADERS: &[(&str, &str)] = &[
    (
        "chelis_runtime.h",
        include_str!("../include/chelis_runtime.h"),
    ),
    (
        "chelis_runtime_views.h",
        include_str!("../include/chelis_runtime_views.h"),
    ),
    (
        "chelis_runtime_dtype.h",
        include_str!("../include/chelis_runtime_dtype.h"),
    ),
    ("chelis_blas.h", include_str!("../include/chelis_blas.h")),
    ("chelis_simd.h", include_str!("../include/chelis_simd.h")),
    ("chelis_math.h", include_str!("../include/chelis_math.h")),
];

#[cfg(test)]
mod tests {
    use super::PUBLIC_HEADERS;
    use std::collections::BTreeSet;
    use std::path::Path;

    #[test]
    fn every_shipped_header_is_exported_once() {
        let include = Path::new(env!("CARGO_MANIFEST_DIR")).join("include");
        let on_disk = std::fs::read_dir(&include)
            .expect("runtime include directory")
            .map(|entry| entry.expect("include entry").file_name())
            .map(|name| name.into_string().expect("UTF-8 header name"))
            .filter(|name| name.ends_with(".h"))
            .collect::<BTreeSet<_>>();
        let exported = PUBLIC_HEADERS
            .iter()
            .map(|(name, _)| (*name).to_owned())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            exported.len(),
            PUBLIC_HEADERS.len(),
            "duplicate header name"
        );
        assert_eq!(exported, on_disk);
    }
}
