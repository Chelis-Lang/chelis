use chelis_deep::printer::print_canonical;
use chelis_macros::{
    CallableDeclaration, ExpansionError, ExpansionOptions, expand_program,
    reject_standard_prelude_collisions,
};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;

fn expand_surf(source: &str) -> String {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let expanded = expand_program(&deep, &ExpansionOptions::default()).expect("macro expansion");
    print_canonical(expanded.exprs())
}

fn assert_contains_var_ref(text: &str, name: &str) {
    assert!(
        text.contains("(var {") && text.contains(&format!(" {name})")),
        "expected expanded Deep to contain var reference `{name}`; got:\n{text}"
    );
}

#[test]
fn simple_macro_expands_to_base_tags_with_source_metadata() {
    let text = expand_surf(
        r#"
macro relu_ref(x) = max_elem(x, 0.0)
def f(x: tensor[4, f32]) -> tensor[4, f32] = relu_ref(x)
"#,
    );

    assert!(!text.contains("defmacro"));
    assert!(!text.contains("macro-invoke"));
    assert!(text.contains("max_elem"));
    assert!(text.contains("source: (relu_ref"));
}

/// Provenance on a macro-introduced typed parameter lands on the parameter's
/// own map, as it does on a node, and not on the type syntax inside it.
#[test]
fn typed_parameter_source_stays_on_the_parameter_map() {
    let text = expand_surf(
        r#"
macro add_one_to(v) = (fn (y: f32) -> add(y, v))(1.0f32)
def run(y: f32) -> f32 = add_one_to(y)
"#,
    );
    // The canonical printer wraps long maps; compare on single spaces.
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");

    assert!(
        flat.contains("(y_macro_0 {source: (add_one_to"),
        "the renamed parameter must carry the invocation provenance:\n{text}"
    );
    assert!(
        flat.contains("type: (t-prim {} f32) }))"),
        "the parameter's own type syntax must stay unannotated:\n{text}"
    );
    assert!(
        !flat.contains("(t-prim {source:"),
        "provenance must not be stamped onto the type syntax:\n{text}"
    );
}

/// A macro invocation is an expression in its caller's binding position.
/// Replacing it must retain the caller's explicit type obligation and Surf
/// origin marker on the expansion root.
#[test]
fn macro_expansion_retains_caller_binding_ascription() {
    let text = expand_surf(
        r#"
macro twice(v) = add(v, v)
def run(x: f32) -> f32 = {
  t: i32 = twice(x)
  cast(t, f32)
}
"#,
    );
    assert!(
        text.contains("surf_binding_type: \"explicit\""),
        "the expanded binding value must retain the caller's origin marker:\n{text}"
    );
    assert!(
        text.contains("type: (t-prim {} i32)"),
        "the expanded binding value must retain its authored ascription:\n{text}"
    );
}

#[test]
fn caller_and_template_type_metadata_remain_separate() {
    let text = expand_surf(
        r#"
macro one() = 1.0f32
def run() -> f32 = {
  t: i32 = one()
  cast(t, f32)
}
"#,
    );
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(
        flat.contains("(block {source: (one),")
            && flat.contains("surf_binding_type: \"explicit\"")
            && flat.contains("type: (t-prim {} i32)")
            && flat.contains("type: (t-prim {} f32)"),
        "caller and template annotations require independent owners:\n{text}"
    );
}

#[test]
fn template_type_metadata_survives_parameter_substitution() {
    let text = expand_surf(
        r#"
macro as_i32(v) = v : i32
def run(x: f32) -> f32 = cast(as_i32(x), f32)
"#,
    );
    assert!(
        text.contains("type: (t-prim {} i32)"),
        "the template's ascription must constrain the substituted argument:\n{text}"
    );
}

/// A typed vocabulary-named parameter uses a prefix metadata wrapper. It is
/// still a binder, with the same hygiene and provenance as a typed ordinary
/// name represented by a structural two-element list.
#[test]
fn vocabulary_named_typed_parameter_is_hygienized_and_carries_source() {
    let text = expand_surf(
        r#"
macro bump(v) = (fn (record: f32) -> add(record, v))(1.0f32)
def run(record: f32) -> f32 = bump(record)
"#,
    );
    assert!(
        text.contains("record_macro_0"),
        "the macro-introduced parameter must be renamed:\n{text}"
    );
    assert!(
        text.contains("span: \"surf:97..103\"} record)"),
        "the caller's argument must retain its free reference:\n{text}"
    );
    assert!(
        !text.contains("(t-prim {source:"),
        "provenance belongs to the parameter, not its type syntax:\n{text}"
    );
}

#[test]
fn parsed_deep_internal_macro_expands_from_raw_form_boundary() {
    let deep = chelis_deep::parser::parse_str(
        r#"(defmacro {} bump (params {} x) (app {} (var {} add) (var {} x) (lit {} 1.0)))
(def {} f (app {} (var {} bump) (lit {} 2.0)))"#,
    )
    .expect("Deep internal macro fixture must stamp");
    assert!(
        matches!(
            deep.first(),
            Some(chelis_deep::Expr::BareList(elements, _))
                if matches!(
                    elements.first(),
                    Some(chelis_deep::Expr::Atom(chelis_deep::Atom::Name(head), _))
                        if head == "defmacro"
                )
        ),
        "the compiler-internal macro definition must retain its recorded raw-string boundary"
    );

    let expanded = expand_program(
        &deep,
        &ExpansionOptions {
            max_iterations: 100,
            load_std_prelude: false,
        },
    )
    .expect("parsed Deep macro must expand");
    let text = print_canonical(expanded.exprs());
    assert!(!text.contains("defmacro"), "{text}");
    assert!(!text.contains(" bump)"), "{text}");
    assert!(text.contains(" add)"), "{text}");
}

#[test]
fn lexical_binding_blocks_prelude_macro_expansion() {
    let text = expand_surf(
        r#"
def f(residual, x: tensor[4, f32]) -> tensor[4, f32] = residual(x)
"#,
    );

    assert_contains_var_ref(&text, "residual");
    assert!(!text.contains("source: (residual"));
}

#[test]
fn ordinary_defs_cannot_collide_with_standard_prelude_macros() {
    for name in ["linear_layer", "residual", "cross_entropy"] {
        let decls = parse_str(&format!("def {name}(x: f32) -> f32 = x\n"))
            .expect("surf parse should succeed");
        let deep = desugar_program(&decls).expect("Surf fixture must desugar");
        let err = expand_program(&deep, &ExpansionOptions::default())
            .expect_err("a standard-prelude macro name must reject an ordinary def");
        let message = err.to_string();
        assert!(
            message.contains(&format!("`def {name}`"))
                && message.contains("standard prelude macro")
                && message.contains("spec/02-surf-syntax.md §P5b"),
            "collision diagnostic for `{name}` must name the declaration, macro class, and owning spec; got: {message}"
        );
    }

    let decls = parse_str("sig residual: f32 -> f32\n").expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let err = expand_program(&deep, &ExpansionOptions::default())
        .expect_err("a standard-prelude macro name must reject an ordinary sig");
    assert!(
        err.to_string().contains("`sig residual`")
            && err.to_string().contains("standard prelude macro"),
        "sig collision diagnostic must name the authored declaration; got: {err}"
    );
}

/// chelis#3270: a caller that renames declarations before expansion (the reef
/// package linker) checks their authored names; it gets the verdict and the
/// diagnostic expansion gives the same declarations, including the first
/// collision in order and an annotated `def` reported as a `def`.
#[test]
fn authored_name_collision_check_matches_expansion() {
    use CallableDeclaration::{Def, Sig};
    /// Surf source, its authored declarations, and the expected diagnostic
    /// prefix (`None` when the declarations are admitted).
    type Case<'a> = (
        &'a str,
        &'a [(&'a str, CallableDeclaration)],
        Option<&'a str>,
    );
    let cases: [Case; 5] = [
        (
            "def residual(x: f32) -> f32 = x\n",
            &[("residual", Def)],
            Some("`def residual`"),
        ),
        (
            "sig cross_entropy: f32 -> f32\n",
            &[("cross_entropy", Sig)],
            Some("`sig cross_entropy`"),
        ),
        (
            "sig linear_layer: f32 -> f32\nlinear_layer = fn (x: f32) -> x\n",
            &[("linear_layer", Sig), ("linear_layer", Def)],
            Some("`def linear_layer`"),
        ),
        (
            "def keep(x: f32) -> f32 = x\nsig cross_entropy: f32 -> f32\ndef residual(x: f32) -> f32 = x\n",
            &[("keep", Def), ("cross_entropy", Sig), ("residual", Def)],
            Some("`sig cross_entropy`"),
        ),
        (
            "def residual_step(x: f32) -> f32 = x\n",
            &[("residual_step", Def)],
            None,
        ),
    ];
    for (source, declarations, diagnostic) in cases {
        let deep = desugar_program(&parse_str(source).expect("surf parse should succeed"))
            .expect("Surf fixture must desugar");
        let expansion = expand_program(&deep, &ExpansionOptions::default())
            .map(drop)
            .map_err(|error| error.to_string());
        let authored = reject_standard_prelude_collisions(declarations).map_err(|e| e.to_string());
        assert_eq!(authored, expansion, "{source}");
        match diagnostic {
            Some(declaration) => assert!(
                authored
                    .as_ref()
                    .is_err_and(|message| message.starts_with(declaration)),
                "{source}: {authored:?}"
            ),
            None => assert_eq!(authored, Ok(()), "{source}"),
        }
    }
}

#[test]
fn user_macro_may_override_standard_prelude_macro() {
    let text = expand_surf(
        r#"
macro residual(x, y) = sub(x, y)
def f(x: f32, y: f32) -> f32 = residual(x, y)
"#,
    );

    assert!(text.contains(" sub)"), "user macro body must win: {text}");
}

#[test]
fn ordinary_prelude_names_are_available_when_prelude_loading_is_disabled() {
    for name in ["linear_layer", "residual", "cross_entropy"] {
        let decls = parse_str(&format!("def {name}(x: f32) -> f32 = x\n"))
            .expect("surf parse should succeed");
        let deep = desugar_program(&decls).expect("Surf fixture must desugar");
        expand_program(
            &deep,
            &ExpansionOptions {
                max_iterations: 100,
                load_std_prelude: false,
            },
        )
        .expect("without a loaded prelude there is no compiler-provided collision");
    }
}

#[test]
fn hygiene_renames_macro_introduced_binders_only() {
    let decls = parse_str(
        r#"
macro capture(y) = {
  x = 1.0
  add(x, y)
}
def f(x: f32) -> f32 = capture(x)
"#,
    )
    .expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let expanded = expand_program(&deep, &ExpansionOptions::default()).expect("macro expansion");
    let text = print_canonical(expanded.exprs());

    assert!(text.contains("x_macro_"));
    assert_contains_var_ref(&text, "x");
}

#[test]
fn hygiene_preserves_call_argument_binders() {
    let text = expand_surf(
        r#"
macro bump(x) = add(x, 1.0)
def f(y: f32) -> f32 = bump({
  z = y
  z
})
"#,
    );

    assert_contains_var_ref(&text, "z");
    assert!(
        !text.contains("z_macro_"),
        "call-site binders must not be hygienized as macro-introduced binders:\n{text}"
    );
}

#[test]
fn free_references_survive_hygiene() {
    let text = expand_surf(
        r#"
def f(batch: i32, x: tensor[batch, hidden, f32], w: tensor[hidden, out_dim, f32], b: tensor[out_dim, f32]) -> tensor[batch, out_dim, f32] =
  linear_layer(x, w, b)
"#,
    );

    assert!(text.contains(" batch)"));
    assert!(!text.contains("batch_macro_"));
}

#[test]
fn recursive_macro_hits_expansion_limit() {
    let decls = parse_str(
        r#"
macro loop(x) = loop(x)
def f(x: f32) -> f32 = loop(x)
"#,
    )
    .expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let err = expand_program(
        &deep,
        &ExpansionOptions {
            max_iterations: 4,
            load_std_prelude: true,
        },
    )
    .expect_err("recursive macro should fail");

    assert!(matches!(err, ExpansionError::ExpansionLimitExceeded { .. }));
}
