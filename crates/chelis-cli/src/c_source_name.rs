//! Keep filesystem names distinct from the C identifiers derived from them.
use std::{
    ffi::{OsStr, OsString},
    fmt::Write,
    path::{Path, PathBuf},
};

const ESCAPE_PREFIX: &str = "chelis_file_";
const C_KEYWORDS: &[&str] = &[
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "_Alignas",
    "_Alignof",
    "_Atomic",
    "_Bool",
    "_Complex",
    "_Generic",
    "_Imaginary",
    "_Noreturn",
    "_Static_assert",
    "_Thread_local",
    "alignas",
    "alignof",
    "bool",
    "constexpr",
    "false",
    "nullptr",
    "static_assert",
    "thread_local",
    "true",
    "typeof",
    "typeof_unqual",
    // The CLI emits its own observation entry point.
    "main",
];

pub(crate) struct CSourceName {
    stem: OsString,
    symbol: String,
}

impl CSourceName {
    pub(crate) fn from_path(path: &Path) -> Self {
        let stem = path
            .file_stem()
            .unwrap_or_else(|| OsStr::new("chelis_main"));
        Self {
            stem: stem.to_owned(),
            symbol: encode(stem),
        }
    }

    pub(crate) fn symbol(&self) -> &str {
        &self.symbol
    }

    pub(crate) fn filename(&self) -> PathBuf {
        let mut filename = self.stem.clone();
        filename.push(".c");
        filename.into()
    }
}

fn encode(stem: &OsStr) -> String {
    let bytes = stem.as_encoded_bytes();
    if bytes.first().is_some_and(u8::is_ascii_alphabetic)
        && bytes
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || *b == b'_')
        && !bytes.starts_with(ESCAPE_PREFIX.as_bytes())
        && !C_KEYWORDS.iter().any(|word| word.as_bytes() == bytes)
    {
        // The preceding ASCII check establishes UTF-8 without a lossy conversion.
        return std::str::from_utf8(bytes)
            .expect("ASCII source stem")
            .to_owned();
    }
    let mut symbol = String::from(ESCAPE_PREFIX);
    for byte in bytes {
        write!(symbol, "{byte:02x}").expect("write to String");
    }
    symbol
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_spelling_and_symbol_have_separate_identities() {
        let name = CSourceName::from_path(Path::new("dir/simple-shape.ch"));
        assert_eq!(name.symbol(), "chelis_file_73696d706c652d7368617065");
        assert_eq!(name.filename(), PathBuf::from("simple-shape.c"));
        let safe = CSourceName::from_path(Path::new("simple_shape.dp"));
        assert_eq!(safe.symbol(), "simple_shape");
    }

    #[test]
    fn encoded_and_literal_prefix_names_do_not_collide() {
        assert_eq!(encode(OsStr::new("a-b")), "chelis_file_612d62");
        assert_eq!(encode(OsStr::new("a_b")), "a_b");
        assert_eq!(
            encode(OsStr::new("chelis_file_612d62")),
            "chelis_file_6368656c69735f66696c655f363132643632"
        );
    }

    #[test]
    fn keywords_and_nonidentifiers_are_encoded_without_substitution() {
        for word in C_KEYWORDS {
            assert!(encode(OsStr::new(word)).starts_with(ESCAPE_PREFIX));
        }
        assert_eq!(encode(OsStr::new("7shape")), "chelis_file_377368617065");
        assert_eq!(encode(OsStr::new("café")), "chelis_file_636166c3a9");
        assert_eq!(encode(OsStr::new("_shape")), "chelis_file_5f7368617065");
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_stems_retain_their_distinct_bytes_and_output_names() {
        use std::os::unix::ffi::OsStrExt;
        let first = CSourceName::from_path(Path::new(OsStr::from_bytes(b"shape\xff.ch")));
        let second = CSourceName::from_path(Path::new(OsStr::from_bytes(b"shape\xfe.ch")));
        assert_eq!(first.symbol(), "chelis_file_7368617065ff");
        assert_eq!(second.symbol(), "chelis_file_7368617065fe");
        assert_eq!(first.filename().as_os_str().as_bytes(), b"shape\xff.c");
    }
}
