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
    use std::process::Command;
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

    fn surf_sexp(source: &str) -> String {
        let mut parser = Parser::new();
        parser
            .set_language(&surf_language())
            .expect("Chelis Surf grammar loads");
        parser
            .parse(source, None)
            .expect("tree-sitter returns a tree")
            .root_node()
            .to_sexp()
    }

    fn rust_surf_has_error(source: &str) -> bool {
        chelis_surf::parser::parse_str(source).is_err()
    }

    fn assert_surf_parser_parity(source: &str, should_accept: bool) {
        let tree_sitter_error = surf_has_error(source);
        let rust_error = rust_surf_has_error(source);
        assert_eq!(
            tree_sitter_error,
            rust_error,
            "Rust/tree-sitter parser disagreement for {source:?}:\n{}",
            surf_sexp(source),
        );
        assert_eq!(
            tree_sitter_error,
            !should_accept,
            "unexpected shared parser verdict for {source:?}:\n{}",
            surf_sexp(source),
        );
    }

    #[test]
    fn surf_v019_tree_sitter_accepts_the_canonical_surface() {
        for source in [
            "module Canonical.Syntax\n",
            "type Point = | Point { x: f32, y: f32 }\n",
            "def choose(x: f32) -> f32 ! { IO } = { y = f(x)\n y }\n",
            "def unit_value() -> unit = ()\n",
            "sequence = do { f(x); g(y) }\n",
            "parallel = par { f(x); g(y) }\n",
            "updated = point with { x: next_x, y }\n",
            "syntax = quote(f(x))\n",
            "singleton = (x,)\n",
            "unit_match = match x with { | () => 0 }\n",
            "nullary_function = f()\n",
            "nullary_constructor = None\n",
            "nullary_constructor_call = None()\n",
            "contextual_where = where\n",
            "wide_int = 42i64\n",
            "wide_float = 1.0f64\n",
            "default_int_commitment = 42i32\n",
            "default_float_commitment = 1.0f32\n",
            "negative_int_pattern = match x with { | -42 => 0 }\n",
            "negative_float_pattern = match x with { | -0.0 => 0 }\n",
            "minimum_int_pattern = match x with { | -9223372036854775808 => 0 }\n",
            "minimum_int = -9223372036854775808\n",
            "typed_minimum_int = -9223372036854775808i64\n",
            "largest_int = 9223372036854775807\n",
            "largest_axis: tensor[9223372036854775807, f32] = value\n",
            "empty_record = Empty {}\n",
            "empty_record_pattern = match x with { | Empty {} => 0 }\n",
            "def explicitly_pure() ! {} = ()\n",
            "different_record_field = Point { x: y }\n",
            "different_record_pattern = match p with { | Point { x: y } => y }\n",
            "later_pipe_argument = x |> fn (v) -> f(y, v)\n",
            "controls = \"\\u{8}\\u{1f}\\u{7f}\\u{85}\\0\\t\\n\\r\\\"\\\\\"\n",
            "@property bounded forall(x: int32) where x <= 1: true\n",
            "@property grouped_operand forall(x: int32, y: int32) where (x + 1) <= y: true\n",
            "@property contracted forall():\n  true\n  with contract = \"std.identity\"\n",
            "result = seed\n  |> f\n  |> g\n  |> h\n",
            "result = {\n  x =\n    seed\n    |> f\n    |> g\n    |> h\n  x\n}\n",
        ] {
            assert_surf_parser_parity(source, true);
        }
    }

    #[test]
    fn surf_v019_tree_sitter_accepts_multiline_pipeline_chains() {
        for source in [
            "result = seed |> f |> g |> h\n",
            "result = seed\n  |> f\n  |> g\n  |> h\n",
        ] {
            assert_surf_parser_parity(source, true);
        }
    }

    #[test]
    fn surf_v019_tree_sitter_accepts_reviewed_cosmetic_aliases() {
        for source in [
            "result = 0x10\n",
            "result = 0b1010\n",
            "result = 1_000\n",
            "result = 0x10i64\n",
            "result = 1e3\n",
            "result = 1.0E+3\n",
            "result = 1_0.5_0e+1\n",
            "result = f(x,)\n",
            "result = [x,]\n",
            "result = Point { x, }\n",
            "result = (x, y,)\n",
            "def trailing[a,](x,) ! { IO, } = x\n",
            "result = par { f(x); g(y); }\n",
            "result = do { f(x); g(y); }\n",
            "result = Some(x,)\n",
            "result = match x with { | Some(v,) => v }\n",
            "sig trailing_type: Option[int32,]\n",
            "type Trailing[a,] = | Trailing(a,)\n",
            "type TrailingRecord = | TrailingRecord { value: int32, }\n",
            "import Demo (value,)\n",
            "export (value,)\n",
            "result = grad(f, wrt=(x, y,),)\n",
            "result = vmap(f, axis=1,)\n",
            "result = cast(x, f64,)\n",
            "result = cast_trunc(x, int32,)\n",
            "result = with seed(1,) { x }\n",
            "def resource() ! { Resource(\"gpu:0\",), } = ()\n",
            "value = \"\\u{08}\\u{0}\\u{9}\\u{a}\\u{d}\\u{22}\\u{5c}\\u{41}\\u{B}\"\n",
            "value = match x with { | -9_223_372_036_854_775_808 => 0 }\n",
            "result = f (x)\n",
            "result = Some (x)\n",
            "result = f (x, y)\n",
            // Parenthesized application is syntactically valid for every
            // expression atom. The checker, rather than either parser,
            // decides whether the callee has a function type.
            "result = 4(x)\n",
            "result = 1.5(x)\n",
            "result = \"a\"(x)\n",
            "result = 50.f\n",
        ] {
            assert_surf_parser_parity(source, true);
        }
    }

    #[test]
    fn surf_v019_tree_sitter_marks_legacy_aliases_as_errors() {
        for source in [
            "def f(x: f32): f32 = x\n",
            "let x = f(y)\n",
            "result = f x\n",
            "result = { x = f(y); x }\n",
            "result = { x = f(y) x }\n",
            "result = a == b == c\n",
            "result = a < b < c\n",
            "result = vmap(f, 1)\n",
            "result = vmap(f, axis=0)\n",
            "result = 42f64\n",
            "value: Option[] = None\n",
            "def f[](x) = x\n",
            "type Empty = | Empty()\n",
            "type Empty = | Empty {}\n",
            "value = point with {}\n",
            "import Demo ()\n",
            "export ()\n",
            "value = match x with { | 42i64 => 0 }\n",
            "value = match x with { | 1.0f64 => 0 }\n",
            "value = 1.00\n",
            "value = 42.00f32\n",
            "value = 1.00000000000000001\n",
            "value = 0.10000000000000001\n",
            "value = 0.10000000000000001f64\n",
            "value = 9223372036854775808\n",
            "value = 9223372036854775808i64\n",
            "value = 9223372036854775809\n",
            "value = 999999999999999999999999999999999999999999\n",
            "value = -9223372036854775809\n",
            "value = -9223372036854775808i32\n",
            "value = match x with { | 9223372036854775808 => 0 }\n",
            "value: tensor[9223372036854775808, f32] = x\n",
            "@property bad forall():\n  true\n  with contract = contract_name\n",
            "@property grouped forall(x: int32) where (x <= 1): true\n",
            "value = \"raw\tcontrol\"\n",
            "value = \"raw\u{8}control\"\n",
            "value = \"raw\u{7f}control\"\n",
            "def unit_value() -> () = ()\n",
            "value = 1__0\n",
            "value = 1_\n",
            "value = 0x_10\n",
            "value = 0x10_\n",
            "value = 0b_10\n",
            "value = 1e3_\n",
            "result = Point { x: x }\n",
            "result = match p with { | Point { x: x } => x }\n",
            "result = x |> fn (v) -> f(v, y)\n",
            "result = x |> fn (v) -> realize(v)\n",
            "result = x |> fn (v) -> copy(v)\n",
            "effect = 1\n",
            "handler = 1\n",
            "perform = 1\n",
            "resume = 1\n",
            "borrow = 1\n",
            "result = if c thenx else z\n",
            "result = if c then a elsewhere\n",
            "a = f(1)b = f(2)\n",
            "result = x > fn (v) -> f(v)\n",
        ] {
            assert_surf_parser_parity(source, false);
        }
    }

    #[test]
    fn tracked_surf_corpus_is_tree_sitter_error_free() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("tree-sitter crate lives at the workspace root");
        let paths = tracked_surf_files(workspace);
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
            if rust_surf_has_error(&source) {
                failures.push(format!(
                    "{}: Rust Surf parser rejected the corpus file",
                    path.display()
                ));
            }
        }

        assert!(
            failures.is_empty(),
            "canonical Surf corpus contains tree-sitter errors:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn tracked_surf_corpus_ignores_untracked_files() {
        let repository = tempfile::tempdir().expect("temporary git repository");
        let tracked = repository.path().join("tracked.ch");
        let untracked = repository.path().join("scratch.ch");
        fs::write(&tracked, "tracked = 1\n").expect("write tracked Surf");
        fs::write(&untracked, "scratch = 2\n").expect("write untracked Surf");
        for arguments in [["init", "-q"].as_slice(), ["add", "tracked.ch"].as_slice()] {
            let status = Command::new("git")
                .args(arguments)
                .current_dir(repository.path())
                .status()
                .expect("run git fixture command");
            assert!(status.success(), "git {arguments:?} failed");
        }

        assert_eq!(tracked_surf_files(repository.path()), vec![tracked]);
    }

    #[test]
    fn tree_sitter_accepts_ryu_shortest_float_corpus() {
        let mut bits = 0x1234_5678_9abc_def0_u64;
        for index in 0..65_536 {
            bits = bits
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let value = f64::from_bits(bits & i64::MAX as u64);
            if !value.is_finite() {
                continue;
            }
            let mut buffer = ryu::Buffer::new();
            let shortest = buffer.format_finite(value);
            let canonical = if shortest.contains('.') || shortest.contains('e') {
                shortest.to_string()
            } else {
                format!("{shortest}.0")
            };
            for source in [
                format!("value_{index} = {canonical}\n"),
                format!("typed_{index} = {canonical}f64\n"),
            ] {
                assert!(
                    !surf_has_error(&source),
                    "Ryū-shortest canonical float parsed with an error: {source}"
                );
            }
        }
    }

    #[test]
    fn surf_scanner_does_not_require_floating_charconv() {
        // Debian 11's GCC/libstdc++ 10 implements integer `<charconv>` but
        // not the floating overloads. Keep the external scanner usable by
        // the glibc 2.31 compatibility build instead of silently raising its
        // C++ runtime floor.
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("tree-sitter crate lives at the workspace root");
        let scanner =
            fs::read_to_string(workspace.join("grammars/tree-sitter-chelis-surf/src/scanner.cc"))
                .expect("read Surf external scanner");
        assert!(
            !scanner.contains("std::chars_format"),
            "Surf scanner must not require floating std::to_chars"
        );
        assert!(
            !scanner.contains("std::from_chars(begin, end, value)"),
            "Surf scanner must not require floating std::from_chars"
        );
    }

    fn tracked_surf_files(workspace: &Path) -> Vec<PathBuf> {
        let output = Command::new("git")
            .args(["ls-files", "-z", "--", "*.ch"])
            .current_dir(workspace)
            .output()
            .expect("run git ls-files for the Surf corpus");
        assert!(
            output.status.success(),
            "git ls-files failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let listed = std::str::from_utf8(&output.stdout).expect("git paths are UTF-8");
        listed
            .split('\0')
            .filter(|path| !path.is_empty())
            .map(|path| workspace.join(path))
            .collect()
    }
}
