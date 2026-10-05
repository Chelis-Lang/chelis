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

    /// chelis#2985: spec/02 `CallArgs` ends with an optional
    /// `accumulator=<dtype>`; `accumulator` stays an ordinary identifier
    /// everywhere else.
    #[test]
    fn surf_tree_sitter_accepts_the_explicit_accumulator_argument() {
        for source in [
            "out = sum(x, 0i32, accumulator=f64)\n",
            "out = einsum(\"i,i->\", a, b, accumulator=i64)\n",
            "out = x |> sum(0i32, accumulator=f64)\n",
            "out = f(accumulator=f64)\n",
            "out = f(accumulator)\n",
            "out = f(accumulator == y)\n",
            "out = accumulator\n",
            "accumulator = 1\n",
            "accumulator = f64\n",
            "def accumulator() -> i32 = 1\n",
        ] {
            assert_surf_parser_parity(source, true);
        }
        for source in [
            "out = sum(x, accumulator=f64, 0i32)\n",
            "out = sum(x, 0i32, accumulator=)\n",
        ] {
            assert_surf_parser_parity(source, false);
        }
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
            // spec/02 §P10a: an integer body under a float suffix is
            // [04-LIT-1]'s exact `literal_source: integer` form (chelis#2119).
            "integer_bodied_float = 42f64\n",
            "wide_integer_bodied_float = 8000000f32\n",
            "largest_f16_integer_body = 65504f16\n",
            "largest_finite_f16_integer_body = 65519f16\n",
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
            "@property bounded forall(x: i32) where x <= 1: true\n",
            "@property grouped_operand forall(x: i32, y: i32) where (x + 1) <= y: true\n",
            "@property contracted forall():\n  true\n  with contract = \"std.identity\"\n",
            "result = seed\n  |> f\n  |> g\n  |> h\n",
            "result = {\n  x =\n    seed\n    |> f\n    |> g\n    |> h\n  x\n}\n",
            // spec/02 §P4c dtype-family bounds (chelis#1417), on both
            // declaration forms and at every family.
            "sig arange[p: Int]: p -> p -> tensor[n, p]\n",
            "sig linspace[p: Float]: p -> p -> i64 -> tensor[n, p]\n",
            "sig total[p: Numeric]: p -> p -> p\n",
            "def only_ints[p: Int](x: p) -> p = x\n",
            "def scale[n, p: Float](x: tensor[n, p]) -> tensor[n, p] = x\n",
            "def unbounded[a](x: a) -> a = x\n",
        ] {
            assert_surf_parser_parity(source, true);
        }
    }

    /// Both parsers reject the same non-family bound spellings, so an ADT
    /// name in the bound position cannot become a silent bound in an editor.
    #[test]
    fn surf_dtype_family_bounds_reject_the_same_spellings_in_both_parsers() {
        for source in [
            "sig f[p: Tensor]: p -> p\n",
            "sig f[p: float]: p -> p\n",
            "sig f[p: f32]: p -> p\n",
            "def f[p: ](x: p) -> p = x\n",
        ] {
            assert_surf_parser_parity(source, false);
        }
    }

    /// chelis#849 / #1024: the newline-continuation rule is a two-parser
    /// contract, so the parity corpus owns it too.
    ///
    /// `|>` was already covered above. `then` and `else` are newly admitted
    /// as continuations, and the layout they enable reaches every
    /// `block_expr_end` consumer -- block binding values, block tails, `do`
    /// and `par` items, and property option values. Without these rows a
    /// tree-sitter regression on the new layout would be invisible to the
    /// committed suite even though the Rust side is locked.
    #[test]
    fn surf_v019_tree_sitter_accepts_newline_led_then_and_else() {
        for source in [
            // block binding value
            "def f(a: f32, b: f32) -> f32 = {\n  c = neq(a, a)\n  d = if c\n  then a\n  else b\n  d\n}\n",
            // block tail
            "def f(a: f32, b: f32) -> f32 = {\n  c = neq(a, a)\n  if c\n  then a\n  else b\n}\n",
            // `do` item
            "def f(a: f32, b: f32) -> f32 ! { IO } = {\n  c = neq(a, a)\n  g = do {\n    if c\n    then print(\"y\")\n    else print(\"n\")\n  }\n  a\n}\n",
            // `par` item
            "def f(a: f32, b: f32) -> f32 = {\n  c = neq(a, a)\n  g = par {\n    if c\n    then a\n    else b\n  }\n  g\n}\n",
            // property option value
            "@property p forall(x: f32): true\n  with tolerance = if lte(x, 1.0f32)\n  then 1e-6f32\n  else 1e-3f32\n",
            // the multiline `else if` chain that motivated admitting `then`
            "def f(a: f32, b: f32) -> f32 = {\n  c = neq(a, a)\n  if c\n  then a\n  else if lt(a, b)\n  then b\n  else mul(a, b)\n}\n",
        ] {
            assert_surf_parser_parity(source, true);
        }
    }

    /// The permissive boundaries, which the closed continuation set does NOT
    /// govern (spec/02 P12). A declaration body ends only at a declaration
    /// start, so a newline-led `with` continues it. Pinned in both parsers so
    /// #849 cannot narrow a boundary it does not own.
    #[test]
    fn surf_v019_tree_sitter_keeps_the_permissive_declaration_boundary() {
        for source in [
            "type Point = | Point { x: f32 }\ndef update(p: Point) -> Point = p\n  with { x: 1.0f32 }\n",
            "@property p forall(x: i32) where if lte(x, 1i32)\n  then true\n  else false: true\n",
        ] {
            assert_surf_parser_parity(source, true);
        }
    }

    /// Admitting a keyword as a newline continuation must not make it
    /// OPTIONAL. Both parsers must still reject a half-formed `if`.
    #[test]
    fn surf_v019_tree_sitter_still_rejects_a_half_formed_if() {
        for source in [
            "def f(a: f32, b: f32) -> f32 = {\n  c = neq(a, a)\n  if c\n  then a\n}\n",
            "def f(a: f32, b: f32) -> f32 = {\n  c = neq(a, a)\n  if c\n  else b\n}\n",
        ] {
            assert_surf_parser_parity(source, false);
        }
    }

    /// spec/02 P12's block continuation set is closed. Infix tokens other
    /// than `|>` remain separators when they lead the next physical line,
    /// even though they cannot begin a standalone expression. Cover every
    /// `block_expr_end` consumer so the editor grammar cannot silently widen
    /// a boundary the Rust parser keeps closed.
    #[test]
    fn surf_v019_tree_sitter_rejects_newline_led_non_continuations() {
        for source in [
            // block binding value
            "def f() -> i32 = {\n  x = 1i32\n  + 2i32\n  x\n}\n",
            // block tail
            "def f() -> i32 = {\n  x = 1i32\n  x\n  * 2i32\n}\n",
            // `do` item
            "def f() -> bool = {\n  x = do {\n    true\n    == false\n  }\n  x\n}\n",
            // `par` item
            "def f() -> bool = {\n  x = par {\n    true\n    && false\n  }\n  x\n}\n",
            // property option value
            "@property p forall(): true\n  with samples = 1i32\n  + 1i32\n  with seed = 1i64\n",
        ] {
            assert_surf_parser_parity(source, false);
        }
    }

    /// Representative excluded infix families at the block-tail boundary.
    /// Non-prefix status is a safety argument for a selected continuation;
    /// it does not itself admit every infix token after a separator.
    #[test]
    fn surf_v019_tree_sitter_keeps_the_exact_block_continuation_set() {
        for source in [
            "def f() -> i32 = {\n  x = 1i32\n  x\n  + 2i32\n}\n",
            "def f() -> i32 = {\n  x = 1i32\n  x\n  * 2i32\n}\n",
            "def f() -> bool = {\n  x = 1i32\n  x\n  == 2i32\n}\n",
            "def f() -> bool = {\n  x = true\n  x\n  && false\n}\n",
            "def f() -> bool = {\n  x = true\n  x\n  || false\n}\n",
        ] {
            assert_surf_parser_parity(source, false);
        }
    }

    #[test]
    fn surf_v019_tree_sitter_accepts_cast_pipe_stages() {
        for source in [
            "result = value |> cast(f32)\n",
            "result = value |> cast_trunc(f64)\n",
            "result = value |> cast(p)\n",
        ] {
            assert_surf_parser_parity(source, true);
        }
        for source in [
            "result = value |> cast()\n",
            "result = value |> cast_trunc()\n",
            "result = cast(f32)\n",
            "result = cast_trunc(f64)\n",
        ] {
            assert_surf_parser_parity(source, false);
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
            "sig trailing_type: Option[i32,]\n",
            "type Trailing[a,] = | Trailing(a,)\n",
            "type TrailingRecord = | TrailingRecord { value: i32, }\n",
            "import Demo (value,)\n",
            "export (value,)\n",
            "result = grad(f, wrt=(x, y,),)\n",
            "result = vmap(f, axis=1,)\n",
            "result = cast(x, f64,)\n",
            "result = cast_trunc(x, i32,)\n",
            "result = with device(\"gpu:0\",) { x }\n",
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
            // chelis#2119: a float body carrying digits past the shortest
            // round-trippable spelling decodes to the same value, so both
            // parsers admit it and `chelis fmt` canonicalizes it.
            "value = 1.00\n",
            "value = 42.00f32\n",
            "value = 1.00000000000000001\n",
            "value = 0.10000000000000001\n",
            "value = 0.10000000000000001f64\n",
            "value = 0.319381530f64\n",
            "value = 0.99999999999980993f64\n",
            "value = 86.50532032941677f64\n",
            "value = 007.5\n",
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
            "value: Option[] = None\n",
            "def f[](x) = x\n",
            "type Empty = | Empty()\n",
            "type Empty = | Empty {}\n",
            "value = point with {}\n",
            "import Demo ()\n",
            "export ()\n",
            "value = match x with { | 42i64 => 0 }\n",
            "value = match x with { | 1.0f64 => 0 }\n",
            "value = 9223372036854775808\n",
            "value = 9223372036854775808i64\n",
            "value = 9223372036854775809\n",
            "value = 999999999999999999999999999999999999999999\n",
            "value = -9223372036854775809\n",
            "value = -9223372036854775808i32\n",
            "value = match x with { | 9223372036854775808 => 0 }\n",
            "value: tensor[9223372036854775808, f32] = x\n",
            "@property bad forall():\n  true\n  with contract = contract_name\n",
            "@property grouped forall(x: i32) where (x <= 1): true\n",
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
            // chelis#2119: an integer body keeps the canonical decimal rule,
            // with or without a float suffix, and no radix form carries one.
            "value = 007\n",
            "value = 007f64\n",
            "value = 0b1010f32\n",
            "value = 0x10f64\n",
            // chelis#2119: an integer body binds at the suffix width, so a
            // magnitude that rounds to infinity there is not a literal of that
            // type. f16 is the only width an i64 body can overflow.
            "value = 65520f16\n",
            "value = 65536f16\n",
            "value = 9223372036854775807f16\n",
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

    /// spec/02 §P5a: randomness has no handler and no effect, so the
    /// retired `with seed(...)` handler and the `Random` effect name are
    /// parse errors in both parsers. The property-test option
    /// `with seed = ...` is a different production and stays accepted.
    #[test]
    fn surf_tree_sitter_rejects_the_retired_randomness_spellings() {
        for source in [
            "result = with seed(1i64) { x }\n",
            "result = with seed(1,) { x }\n",
            "def f() -> i32 ! { Random } = 1i32\n",
            "def f() -> i32 ! { IO, Random } = 1i32\n",
            "sig f: i32 -> i32 ! { Random }\n",
        ] {
            assert_surf_parser_parity(source, false);
        }
        for source in [
            "result = with device(\"gpu:0\") { x }\n",
            "def f() -> i32 ! { IO } = 1i32\n",
            "@property p forall(): true\n  with seed = 1i64\n",
        ] {
            assert_surf_parser_parity(source, true);
        }
    }

    #[test]
    fn surf_integer_literals_are_dimensions_not_types_in_both_parsers() {
        // chelis#1179: `IntLit` has exactly two type-grammar productions,
        // the tensor `DimExpr` and the type-application `TypeArg` (the
        // concrete dimension instantiation of a dimension-parameterized
        // ADT, chelis#940). Both parsers accept those positions and reject
        // an integer in every bare type position.
        for source in [
            "value: tensor[732, f32] = x\n",
            "def g(a: tensor[3, 4, f32]) -> tensor[3, 4, f32] = a\n",
            "def concrete(frame: Frame[2]) -> Frame[2] = frame\n",
            "value: Hamt[Column[2]] = x\n",
            "value: Option[732] = x\n",
        ] {
            assert_surf_parser_parity(source, true);
        }
        for source in [
            "def g(a: 732) -> f32 = a\n",
            "def g(a: f32) -> 732 = a\n",
            "value: 732 = x\n",
            "sig g: 732 -> f32\n",
            "value: (732, f32) = x\n",
            "def g(a: &732) -> f32 = a\n",
            "type Wrap = | Wrap(732)\n",
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
