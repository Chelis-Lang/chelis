//! chelis#1955 / chelis#1473: named `grad` selectors are resolved from the
//! callable value's immutable lexical origin. They are never guessed from
//! selector position or a same-spelled global.

use chelis_deep::printer::print_canonical;
use chelis_surf::{
    ast::Decl,
    decompile::try_decompile_program,
    desugar::{
        DesugarError, desugar_decl_only, desugar_expr_in_program, desugar_expr_in_program_scope,
        desugar_expr_only, desugar_program,
    },
    parser::parse_str,
    resugar::{resugar_expression, resugar_program},
};

#[test]
fn canonical_public_desugar_apis_are_fallible() {
    let _: fn(&[Decl]) -> Result<Vec<chelis_deep::Expr>, DesugarError> = desugar_program;
    let _: fn(&Decl) -> Result<Vec<chelis_deep::Expr>, DesugarError> = desugar_decl_only;
    let _: fn(&chelis_surf::ast::Expr) -> Result<chelis_deep::Expr, DesugarError> =
        desugar_expr_only;
}

fn deep(source: &str) -> String {
    let declarations = parse_str(source).expect("Surf fixture parses");
    print_canonical(&desugar_program(&declarations).expect("selector resolves"))
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn error(source: &str) -> DesugarError {
    let declarations = parse_str(source).expect("Surf fixture parses");
    desugar_program(&declarations).expect_err("selector must reject")
}

#[test]
fn direct_inline_and_alias_targets_preserve_formal_identity() {
    let source = r#"
def pair(x: f32, w: f32) -> f32 = mul(x, w)
top = pair
def probe() -> unit = {
  local = top
  direct = grad(pair, wrt=w)
  inline = grad(fn (x: f32, w: f32) -> mul(x, w), wrt=w)
  aliased = grad(local, wrt=w)
  drop((direct, inline, aliased))
}
"#;
    let actual = deep(source);
    assert_eq!(
        actual.matches("(lit {type: (t-prim {} i32)} 1)").count(),
        3,
        "{actual}"
    );
}

#[test]
fn alias_chains_snapshot_the_callable_before_rebinding() {
    let actual = deep(
        r#"
def pair(x: f32, w: f32) -> f32 = mul(x, w)
def swapped(w: f32, x: f32) -> f32 = mul(x, w)
def probe() -> unit = {
  current = pair
  snapshot = current
  current = swapped
  selected = grad(snapshot, wrt=w)
  drop(selected)
}
"#,
    );
    assert!(actual.contains("wrt: (var {} w)"), "{actual}");
    assert_eq!(
        actual.matches("(lit {type: (t-prim {} i32)} 1)").count(),
        1,
        "{actual}"
    );
    assert_eq!(
        actual.matches("(lit {type: (t-prim {} i32)} 0)").count(),
        0,
        "{actual}"
    );
}

#[test]
fn branch_joins_require_exact_inline_lambda_origin() {
    for (label, source) in [
        (
            "if",
            r#"
def probe(flag: bool) -> unit = {
  chosen = if flag then fn (x: f32, w: f32) -> mul(x, w) else fn (x: f32, w: f32) -> add(x, w)
  selected = grad(chosen, wrt=w)
  drop(selected)
}
"#,
        ),
        (
            "match",
            r#"
def probe(flag: bool) -> unit = {
  chosen = match flag with {
    | true => fn (x: f32, w: f32) -> mul(x, w)
    | false => fn (x: f32, w: f32) -> add(x, w)
  }
  selected = grad(chosen, wrt=w)
  drop(selected)
}
"#,
        ),
        (
            "tuple projection",
            r#"
def probe(flag: bool) -> unit = {
  chosen = (if flag then (fn (x: f32, w: f32) -> mul(x, w),) else (fn (x: f32, w: f32) -> add(x, w),)).0
  selected = grad(chosen, wrt=w)
  drop(selected)
}
"#,
        ),
        (
            "block return",
            r#"
def probe(flag: bool) -> unit = {
  chosen = if flag then {
    branch = fn (x: f32, w: f32) -> mul(x, w)
    branch
  } else {
    branch = fn (x: f32, w: f32) -> add(x, w)
    branch
  }
  selected = grad(chosen, wrt=w)
  drop(selected)
}
"#,
        ),
    ] {
        assert!(
            matches!(error(source), DesugarError::UnresolvedGradTarget { .. }),
            "{label}"
        );
    }
}

#[test]
fn branch_joins_retain_aliases_of_the_same_callable_origin() {
    let actual = deep(
        r#"
def pair(x: f32, w: f32) -> f32 = mul(x, w)
def probe(flag: bool) -> unit = {
  left = pair
  right = pair
  if_join = if flag then left else right
  match_join = match flag with {
    | true => left
    | false => right
  }
  tuple_join = (if flag then (left,) else (right,)).0
  mixed_tuple_join = (if flag then
    (fn (x: f32, w: f32) -> add(x, w), left)
  else
    (fn (x: f32, w: f32) -> sub(x, w), right)).1
  block_join = if flag then {
    branch = left
    branch
  } else {
    branch = right
    branch
  }
  drop((
    grad(if_join, wrt=w),
    grad(match_join, wrt=w),
    grad(tuple_join, wrt=w),
    grad(mixed_tuple_join, wrt=w),
    grad(block_join, wrt=w)
  ))
}
"#,
    );
    assert_eq!(actual.matches("(grad ").count(), 5, "{actual}");
    assert_eq!(actual.matches("wrt: (var {} w)").count(), 5, "{actual}");
}

#[test]
fn written_selector_order_and_duplicates_are_preserved() {
    let actual = deep(
        r#"
def pair(x: f32, w: f32) -> f32 = mul(x, w)
selected = grad(pair, wrt=(w, x, w))
"#,
    );
    assert!(
        actual.contains(
            "(tuple {} (lit {type: (t-prim {} i32)} 1) (lit {type: (t-prim {} i32)} 0) (lit {type: (t-prim {} i32)} 1))"
        ),
        "{actual}"
    );
}

#[test]
fn cloned_fragments_resolve_against_program_context_and_lexical_shadowing() {
    let declarations = parse_str(
        "def pair(x: f32, w: f32) -> f32 = mul(x, w)\n\
         alias = pair\n\
         selected = grad(alias, wrt=w)\n",
    )
    .expect("program parses");
    let Decl::LetDef { value, .. } = &declarations[2] else {
        panic!("selected value binding")
    };
    let cloned = value.clone();
    let deep = print_canonical(&[desugar_expr_in_program(&declarations, &cloned)
        .expect("cloned expression resolves in program context")]);
    assert!(deep.contains("(lit {type: (t-prim {} i32)} 1)"), "{deep}");

    assert!(matches!(
        desugar_expr_in_program_scope(&declarations, &cloned, &["alias".to_string()]),
        Err(DesugarError::UnresolvedGradTarget { .. })
    ));
}

#[test]
fn unknown_direct_inline_and_partially_valid_selectors_reject() {
    for (label, source, missing) in [
        (
            "direct",
            "def pair(x: f32, w: f32) -> f32 = mul(x, w)\nout = grad(pair, wrt=typo)\n",
            "typo",
        ),
        (
            "inline",
            "out = grad(fn (x: f32, w: f32) -> mul(x, w), wrt=typo)\n",
            "typo",
        ),
        (
            "partial",
            "def pair(x: f32, w: f32) -> f32 = mul(x, w)\nout = grad(pair, wrt=(w, typo))\n",
            "typo",
        ),
    ] {
        assert!(
            matches!(
                error(source),
                DesugarError::UnknownGradParameter { parameter, .. } if parameter == missing
            ),
            "{label}"
        );
    }
}

#[test]
fn dynamic_noncallable_and_shadowing_targets_reject() {
    let dynamic = error(
        r#"
def apply(f: f32 -> f32) -> unit = {
  g = grad(f, wrt=x)
  drop(g)
}
"#,
    );
    assert!(matches!(dynamic, DesugarError::UnresolvedGradTarget { .. }));

    let noncallable = error("value = 1.0f32\nout = grad(value, wrt=x)\n");
    assert!(matches!(
        noncallable,
        DesugarError::NonCallableGradTarget { .. }
    ));

    let shadowed = error(
        r#"
def pair(x: f32, w: f32) -> f32 = mul(x, w)
def apply(pair: f32 -> f32) -> unit = {
  g = grad(pair, wrt=w)
  drop(g)
}
"#,
    );
    assert!(matches!(
        shadowed,
        DesugarError::UnresolvedGradTarget { .. }
    ));
}

#[test]
fn matching_deep_round_trips_and_contradictory_metadata_rejects() {
    let matching = chelis_deep::parser::parse_str(
        "(def {} pair (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))))\n\
         (def {} selected (grad {wrt: (var {} w)} (var {} pair) (lit {type: (t-prim {} i32)} 1)))",
    )
    .expect("matching Deep parses");
    let surf = resugar_program(&matching).expect("matching selector resugars");
    let redesugared = desugar_program(&surf).expect("matching selector redesugars");
    let original = chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&matching)
        .expect("normalize original");
    let roundtrip = chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&redesugared)
        .expect("normalize roundtrip");
    assert_eq!(print_canonical(&roundtrip), print_canonical(&original));

    let contradictory = chelis_deep::parser::parse_str(
        "(def {} pair (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))))\n\
         (def {} selected (grad {wrt: (var {} w)} (var {} pair) (lit {type: (t-prim {} i32)} 0)))",
    )
    .expect("contradictory Deep parses structurally");
    let error = resugar_program(&contradictory).expect_err("contradiction must reject");
    assert!(
        error.to_string().contains("selector metadata")
            && error.to_string().contains("index 0")
            && error.to_string().contains("parameter `x`"),
        "{error}"
    );

    let local_alias = chelis_deep::parser::parse_str(
        "(def {} pair (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))))\n\
         (def {} selected (let {} (bind {} local (var {} pair)) \
           (grad {wrt: (var {} w)} (var {} local) (lit {type: (t-prim {} i32)} 1))))",
    )
    .expect("local-alias Deep parses");
    resugar_program(&local_alias).expect("matching local alias selector resugars");

    let local_contradiction = chelis_deep::parser::parse_str(
        "(def {} pair (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))))\n\
         (def {} selected (let {} (bind {} local (var {} pair)) \
           (grad {wrt: (var {} w)} (var {} local) (lit {type: (t-prim {} i32)} 0))))",
    )
    .expect("contradictory local-alias Deep parses");
    assert!(
        resugar_program(&local_contradiction)
            .expect_err("local alias contradiction must reject")
            .to_string()
            .contains("index 0")
    );

    let out_of_range = chelis_deep::parser::parse_str(
        "(def {} pair (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))))\n\
         (def {} selected (grad {wrt: (var {} w)} (var {} pair) \
           (lit {type: (t-prim {} i32)} 4)))",
    )
    .expect("out-of-range Deep parses structurally");
    assert!(
        resugar_program(&out_of_range)
            .expect_err("out-of-range selector must reject")
            .to_string()
            .contains("outside")
    );
}

#[test]
fn deep_resugaring_preseeds_forward_function_origins_before_aliases() {
    let forward_alias = chelis_deep::parser::parse_str(
        "(def {} alias (var {} pair))\n\
         (def {} pair (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))))\n\
         (def {} selected (grad {wrt: (var {} w)} (var {} alias) \
           (lit {type: (t-prim {} i32)} 1)))",
    )
    .expect("forward-alias Deep parses");
    let surf = resugar_program(&forward_alias).expect("forward function alias resugars");
    desugar_program(&surf).expect("forward function alias round-trips");

    let distinct = chelis_deep::parser::parse_str(
        "(def {} chosen (if {} (lit {} true) (var {} left) (var {} right)))\n\
         (def {} left (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))))\n\
         (def {} right (fn {} (params {} x w) (app {} (var {} add) (var {} x) (var {} w))))\n\
         (def {} selected (grad {wrt: (var {} w)} (var {} chosen) \
           (lit {type: (t-prim {} i32)} 1)))",
    )
    .expect("distinct forward-origin Deep parses");
    assert!(matches!(
        resugar_program(&distinct),
        Err(chelis_surf::resugar::ResugarError::UnresolvedGradSelectorTarget { .. })
    ));
}

#[test]
fn deep_resugaring_projects_match_bound_callable_origins() {
    let matching = chelis_deep::parser::parse_str(
        "(def {} pair (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))))\n\
         (def {} matched \
           (match {} (tuple {} (var {} pair)) \
             (arm {} (pat-tuple {} (pat-var {} chosen)) () (var {} chosen))))\n\
         (def {} selected \
           (tuple {} \
             (grad {wrt: (var {} w)} (var {} matched) \
               (lit {type: (t-prim {} i32)} 1)) \
             (match {} (tuple {} (var {} pair)) \
               (arm {} (pat-tuple {} (pat-var {} chosen)) () \
                 (grad {wrt: (var {} w)} (var {} chosen) \
                   (lit {type: (t-prim {} i32)} 1))))))",
    )
    .expect("match-bound Deep parses");
    resugar_program(&matching).expect("match-bound callable origins resugar");

    for (label, source) in [
        (
            "distinct tuple element origins",
            "(def {} left (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))))\n\
             (def {} right (fn {} (params {} x w) (app {} (var {} add) (var {} x) (var {} w))))\n\
             (def {} selected \
               (match {} \
                 (if {} (lit {} true) \
                   (tuple {} (var {} left)) \
                   (tuple {} (var {} right))) \
                 (arm {} (pat-tuple {} (pat-var {} chosen)) () \
                   (grad {wrt: (var {} w)} (var {} chosen) \
                     (lit {type: (t-prim {} i32)} 1)))))",
        ),
        (
            "dynamic scrutinee",
            "(def {} chosen \
               (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))))\n\
             (def {} selected \
               (match {} (var {} candidates) \
                 (arm {} (pat-tuple {} (pat-var {} chosen)) () \
                   (grad {wrt: (var {} w)} (var {} chosen) \
                     (lit {type: (t-prim {} i32)} 1)))))",
        ),
    ] {
        let deep = chelis_deep::parser::parse_str(source).expect("negative Deep parses");
        assert!(
            matches!(
                resugar_program(&deep),
                Err(chelis_surf::resugar::ResugarError::UnresolvedGradSelectorTarget { .. })
            ),
            "{label}"
        );
    }
}

#[test]
fn surf_constructor_and_record_patterns_project_callable_payload_origins() {
    let actual = deep(
        r#"
def pair(x: f32, w: f32) -> f32 = mul(x, w)
alias = pair
boxed = FnBox(alias)
nested_boxed = Outer(boxed)
recorded = FnRecord { callback: alias }
nested_recorded = OuterRecord { payload: recorded }
def probe() -> unit = {
  direct_ctor = match boxed with {
    | FnBox(chosen) => chosen
  }
  nested_ctor = match nested_boxed with {
    | Outer(FnBox(chosen)) => chosen
  }
  direct_record = match recorded with {
    | FnRecord { callback: chosen } => chosen
  }
  nested_record = match nested_recorded with {
    | OuterRecord { payload: FnRecord { callback: chosen } } => chosen
  }
  drop((
    grad(direct_ctor, wrt=w),
    grad(nested_ctor, wrt=w),
    grad(direct_record, wrt=w),
    grad(nested_record, wrt=w),
    direct_ctor(2.0f32, 3.0f32),
    direct_record(2.0f32, 3.0f32)
  ))
}
"#,
    );
    assert_eq!(actual.matches("(grad ").count(), 4, "{actual}");
    assert_eq!(actual.matches("wrt: (var {} w)").count(), 4, "{actual}");
    assert_eq!(
        actual.matches("(lit {type: (t-prim {} i32)} 1)").count(),
        4,
        "{actual}"
    );
}

#[test]
fn surf_constructor_and_record_patterns_reject_distinct_or_dynamic_payload_origins() {
    for (label, source) in [
        (
            "distinct constructor payloads",
            r#"
def left(x: f32, w: f32) -> f32 = mul(x, w)
def right(x: f32, w: f32) -> f32 = add(x, w)
def probe(flag: bool) -> unit = {
  boxed = if flag then FnBox(left) else FnBox(right)
  chosen = match boxed with {
    | FnBox(value) => value
  }
  drop(grad(chosen, wrt=w))
}
"#,
        ),
        (
            "dynamic constructor payload",
            r#"
def probe(candidates: FnBox) -> unit = {
  chosen = match candidates with {
    | FnBox(value) => value
  }
  drop(grad(chosen, wrt=w))
}
"#,
        ),
        (
            "distinct record payloads",
            r#"
def left(x: f32, w: f32) -> f32 = mul(x, w)
def right(x: f32, w: f32) -> f32 = add(x, w)
def probe(flag: bool) -> unit = {
  boxed = if flag then FnRecord { callback: left } else FnRecord { callback: right }
  chosen = match boxed with {
    | FnRecord { callback: value } => value
  }
  drop(grad(chosen, wrt=w))
}
"#,
        ),
        (
            "dynamic record payload",
            r#"
def probe(candidates: FnRecord) -> unit = {
  chosen = match candidates with {
    | FnRecord { callback: value } => value
  }
  drop(grad(chosen, wrt=w))
}
"#,
        ),
    ] {
        assert!(
            matches!(error(source), DesugarError::UnresolvedGradTarget { .. }),
            "{label}"
        );
    }
}

#[test]
fn deep_resugaring_projects_constructor_and_record_payload_origins() {
    let matching = chelis_deep::parser::parse_str(
        "(def {} pair (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))))\n\
         (def {} alias (var {} pair))\n\
         (def {} boxed (app {} (var {} FnBox) (var {} alias)))\n\
         (def {} nested_boxed (app {} (var {} Outer) (var {} boxed)))\n\
         (def {} recorded (record {} FnRecord (kv {} callback (var {} alias))))\n\
         (def {} nested_recorded (record {} OuterRecord (kv {} payload (var {} recorded))))\n\
         (def {} selected \
           (tuple {} \
             (match {} (var {} boxed) \
               (arm {} (pat-ctor {} FnBox (pat-var {} chosen)) () \
                 (grad {wrt: (var {} w)} (var {} chosen) \
                   (lit {type: (t-prim {} i32)} 1)))) \
             (match {} (var {} nested_boxed) \
               (arm {} (pat-ctor {} Outer (pat-ctor {} FnBox (pat-var {} chosen))) () \
                 (grad {wrt: (var {} w)} (var {} chosen) \
                   (lit {type: (t-prim {} i32)} 1)))) \
             (match {} (var {} recorded) \
               (arm {} (pat-record {} FnRecord (kv {} callback (pat-var {} chosen))) () \
                 (grad {wrt: (var {} w)} (var {} chosen) \
                   (lit {type: (t-prim {} i32)} 1)))) \
             (match {} (var {} nested_recorded) \
               (arm {} \
                 (pat-record {} OuterRecord \
                   (kv {} payload \
                     (pat-record {} FnRecord (kv {} callback (pat-var {} chosen))))) \
                 () \
                 (grad {wrt: (var {} w)} (var {} chosen) \
                   (lit {type: (t-prim {} i32)} 1))))))",
    )
    .expect("aggregate-payload Deep parses");
    let surf = resugar_program(&matching).expect("aggregate payload origins resugar");
    desugar_program(&surf).expect("aggregate payload origins round-trip");
}

#[test]
fn deep_resugaring_rejects_distinct_or_dynamic_constructor_and_record_payloads() {
    for (label, source) in [
        (
            "distinct constructor payloads",
            "(def {} left (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))))\n\
             (def {} right (fn {} (params {} x w) (app {} (var {} add) (var {} x) (var {} w))))\n\
             (def {} selected \
               (match {} \
                 (if {} (lit {} true) \
                   (app {} (var {} FnBox) (var {} left)) \
                   (app {} (var {} FnBox) (var {} right))) \
                 (arm {} (pat-ctor {} FnBox (pat-var {} chosen)) () \
                   (grad {wrt: (var {} w)} (var {} chosen) \
                     (lit {type: (t-prim {} i32)} 1)))))",
        ),
        (
            "dynamic constructor payload",
            "(def {} selected \
               (match {} (var {} candidates) \
                 (arm {} (pat-ctor {} FnBox (pat-var {} chosen)) () \
                   (grad {wrt: (var {} w)} (var {} chosen) \
                     (lit {type: (t-prim {} i32)} 1)))))",
        ),
        (
            "distinct record payloads",
            "(def {} left (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))))\n\
             (def {} right (fn {} (params {} x w) (app {} (var {} add) (var {} x) (var {} w))))\n\
             (def {} selected \
               (match {} \
                 (if {} (lit {} true) \
                   (record {} FnRecord (kv {} callback (var {} left))) \
                   (record {} FnRecord (kv {} callback (var {} right)))) \
                 (arm {} (pat-record {} FnRecord \
                   (kv {} callback (pat-var {} chosen))) () \
                   (grad {wrt: (var {} w)} (var {} chosen) \
                     (lit {type: (t-prim {} i32)} 1)))))",
        ),
        (
            "dynamic record payload",
            "(def {} selected \
               (match {} (var {} candidates) \
                 (arm {} (pat-record {} FnRecord \
                   (kv {} callback (pat-var {} chosen))) () \
                   (grad {wrt: (var {} w)} (var {} chosen) \
                     (lit {type: (t-prim {} i32)} 1)))))",
        ),
    ] {
        let deep = chelis_deep::parser::parse_str(source).expect("negative Deep parses");
        assert!(
            matches!(
                resugar_program(&deep),
                Err(chelis_surf::resugar::ResugarError::UnresolvedGradSelectorTarget { .. })
            ),
            "{label}"
        );
    }
}

#[test]
fn expression_resugaring_and_lone_decompilation_reject_unpreservable_selectors() {
    for (label, source) in [
        (
            "contradictory inline callable",
            "(grad {wrt: (var {} w)} \
               (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))) \
               (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "malformed non-integer selector",
            "(grad {wrt: (var {} w)} \
               (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))) \
               (var {} selector))",
        ),
        (
            "dynamic callable target",
            "(grad {wrt: (var {} w)} (var {} chosen) \
               (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "distinct inline branch origins",
            "(grad {wrt: (var {} w)} \
               (if {} (lit {} true) \
                 (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))) \
                 (fn {} (params {} x w) (app {} (var {} add) (var {} x) (var {} w)))) \
               (lit {type: (t-prim {} i32)} 1))",
        ),
    ] {
        let deep = chelis_deep::parser::parse_str(source).expect("Deep expression parses");
        assert_eq!(deep.len(), 1, "{label}");
        assert!(
            resugar_expression(&deep[0]).is_err(),
            "public expression resugaring rewrote {label}"
        );
        assert!(
            try_decompile_program(&deep).is_err(),
            "lone-expression decompilation fallback rewrote {label}"
        );
    }

    let same_origin = chelis_deep::parser::parse_str(
        "(let {} \
           (bind {} \
             base (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))) \
             left (var {} base) \
             right (var {} base)) \
           (grad {wrt: (var {} w)} \
             (if {} (lit {} true) (var {} left) (var {} right)) \
             (lit {type: (t-prim {} i32)} 1)))",
    )
    .expect("same-origin Deep expression parses");
    resugar_expression(&same_origin[0]).expect("same-origin branch selector resugars");
    try_decompile_program(&same_origin).expect("same-origin branch selector decompiles");
}
