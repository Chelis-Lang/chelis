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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use tree_sitter::Parser;

    fn surf_has_error(source: &str) -> bool {
        let mut parser = Parser::new();
        parser
            .set_language(&surf_language())
            .expect("Chelis Surf grammar loads");
        parser
            .parse(source, None)
            .expect("tree-sitter returns a tree")
            .root_node()
            .has_error()
    }

    #[test]
    fn surf_v019_tree_sitter_accepts_the_canonical_surface() {
        for source in [
            "module Canonical.Syntax\n",
            "type Point = | Point { x: f32, y: f32 }\n",
            "def choose(x: f32) -> f32 ! { IO } = { y = f(x)\n y }\n",
            "sequence = do { f(x); g(y) }\n",
            "parallel = par { f(x); g(y) }\n",
            "updated = point with { x: next_x, y }\n",
            "syntax = quote(f(x))\n",
            "singleton = (x,)\n",
            "unit_match = match x with { | () => 0 }\n",
            "nullary_function = f()\n",
            "nullary_constructor = None\n",
            "wide_int = 42i64\n",
            "wide_float = 1.0f64\n",
            "default_int_commitment = 42i32\n",
            "default_float_commitment = 1.0f32\n",
        ] {
            assert!(
                !surf_has_error(source),
                "canonical Surf parsed with an error: {source}"
            );
        }
    }

    #[test]
    fn surf_v019_tree_sitter_marks_legacy_aliases_as_errors() {
        for source in [
            "def f(x: f32): f32 = x\n",
            "let x = f(y)\n",
            "result = f x\n",
            "result = { x = f(y); x }\n",
            "result = vmap(f, 1)\n",
            "result = vmap(f, axis=0)\n",
            "result = 0x10\n",
            "result = 42f64\n",
            "result = f(x,)\n",
            "result = [x,]\n",
            "result = Point { x, }\n",
            "result = (x, y,)\n",
            "result = Some(x,)\n",
            "result = None()\n",
            "result = match x with { | Some(v,) => v }\n",
            "result = par { f(x); g(y); }\n",
            "result = do { f(x); g(y); }\n",
            "def f[a,](x) = x\n",
            "def f() ! { IO, } = ()\n",
            "result = grad(f,)\n",
            "result = vmap(f, axis=1,)\n",
            "result = cast(x, f64,)\n",
            "result = with seed(1,) { x }\n",
            "def resource() ! { Resource(\"gpu:0\",) } = ()\n",
            "type Boxed = | Boxed(int32,)\n",
            "import Demo (value,)\n",
            "export (value,)\n",
        ] {
            assert!(
                surf_has_error(source),
                "legacy alias parsed cleanly: {source}"
            );
        }
    }

    #[test]
    fn tracked_surf_corpus_is_tree_sitter_error_free() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("tree-sitter crate lives at the workspace root");
        let mut paths = Vec::new();
        collect_surf_files(workspace, &mut paths);
        assert!(!paths.is_empty(), "Surf corpus must not be empty");

        let mut parser = Parser::new();
        parser
            .set_language(&surf_language())
            .expect("Chelis Surf grammar loads");
        let mut failures = Vec::new();
        for path in paths {
            let source = fs::read_to_string(&path).expect("Surf corpus file is UTF-8");
            let tree = parser
                .parse(&source, None)
                .expect("tree-sitter returns a tree");
            if tree.root_node().has_error() {
                failures.push(format!("{}: {}", path.display(), tree.root_node()));
            }
        }

        assert!(
            failures.is_empty(),
            "canonical Surf corpus contains tree-sitter errors:\n{}",
            failures.join("\n")
        );
    }

    fn collect_surf_files(directory: &Path, paths: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(directory).expect("read workspace directory") {
            let entry = entry.expect("read workspace entry");
            let path = entry.path();
            if path.is_dir() {
                if matches!(
                    path.file_name().and_then(|name| name.to_str()),
                    Some(".git" | "target" | "node_modules")
                ) {
                    continue;
                }
                collect_surf_files(&path, paths);
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("ch") {
                paths.push(path);
            }
        }
    }
}
