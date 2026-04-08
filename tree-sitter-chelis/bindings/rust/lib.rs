use tree_sitter::{Language, ffi::TSLanguage};

unsafe extern "C" {
    fn tree_sitter_chelis_surf() -> *const TSLanguage;
    fn tree_sitter_chelis_deep() -> *const TSLanguage;
}

pub fn surf_language() -> Language {
    unsafe { Language::from_raw(tree_sitter_chelis_surf()) }
}

pub fn deep_language() -> Language {
    unsafe { Language::from_raw(tree_sitter_chelis_deep()) }
}
