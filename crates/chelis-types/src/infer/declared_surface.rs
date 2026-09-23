//! Declared types for source inventories, without expression-body inference.
//!
//! This shares the compiler's nominal headers, declaration registration and
//! Deep type resolver. Alias edges remain explicit: expanding them here would
//! lose legal recursive aliases before a consumer computes its fixed point.

use super::*;

#[derive(Debug, Clone)]
pub struct DeclaredSignature {
    pub ty: Type,
    pub type_names: BTreeMap<TypeVar, String>,
    pub bounds: Vec<(TypeVar, TypeVarRestriction)>,
}

/// Validated declaration graph. This is not a checked expression program or
/// a proof of body executability, effects, realizability, or numeric authority.
#[derive(Debug, Clone)]
pub struct DeclaredTypeSurface {
    registry: AdtRegistry,
    signatures: BTreeMap<String, DeclaredSignature>,
}

impl DeclaredTypeSurface {
    pub fn registry(&self) -> &AdtRegistry {
        &self.registry
    }

    pub fn signatures(&self) -> &BTreeMap<String, DeclaredSignature> {
        &self.signatures
    }

    /// These exact headers also seed the compiler's type resolver. Their
    /// arguments are payloads; registered nominal types retain their fields.
    pub fn native_nominals(&self) -> &'static [(&'static str, usize)] {
        crate::deep_type::NATIVE_NOMINAL_HEADERS
    }
}

pub fn resolve_declared_surface(exprs: &[deep::Expr]) -> Result<DeclaredTypeSurface, InferResult> {
    crate::session::resolve_declared_surface(exprs)
}

pub(crate) fn resolve_declared_surface_in_session(
    exprs: &[deep::Expr],
    sink: &mut DiagnosticSink<'_>,
) -> Result<DeclaredTypeSurface, InferStats> {
    let stack_scope = StackExhaustionScope::enter();
    fn validate_carriers(expr: &deep::Expr, sink: &mut DiagnosticSink<'_>) {
        stack_guard!("validate_carriers", expr);
        if let Some((tag, _, children)) = stamped_parts(expr) {
            // Node binder roles also admit patterns. Declaration collectors
            // require names here and otherwise skip the malformed declaration.
            if matches!(
                tag,
                DeepTag::Module
                    | DeepTag::Defsig
                    | DeepTag::Defdim
                    | DeepTag::Deftype
                    | DeepTag::Typealias
                    | DeepTag::Variant
                    | DeepTag::Field
            ) && children.first().and_then(symbol_name).is_none()
            {
                sink.push(
                    CheckError::new(
                        CheckErrorKind::MalformedForm,
                        "declaration requires a symbol name".to_string(),
                        vec![],
                    )
                    .at_offset(expr.span().offset),
                );
            }
            for child in children {
                validate_carriers(child, sink);
            }
        } else if let deep::Expr::BareList(children, _) = expr {
            for child in children {
                validate_carriers(child, sink);
            }
        }
    }
    fn items<'a>(
        expr: &'a deep::Expr,
        module: Option<String>,
        out: &mut Vec<(Option<String>, &'a deep::Expr)>,
    ) {
        stack_guard!("declared_surface_items", expr);
        if let Some((DeepTag::Module, _, children)) = stamped_parts(expr) {
            let owner = children.first().and_then(symbol_name).map(str::to_string);
            for child in &children[1..] {
                items(child, owner.clone(), out);
            }
        } else {
            out.push((module, expr));
        }
    }
    let failed = || InferStats {
        typed_nodes: 0,
        total_nodes: 0,
    };
    // The usual compiler ingress has already validated declaration shapes.
    // This separate public ingress must do so before collector helpers that
    // assume those shapes, including complete parameter and field lists.
    for warning in chelis_deep::validate::validate(exprs) {
        sink.push(
            CheckError::new(CheckErrorKind::MalformedForm, warning.message, vec![])
                .at_offset(warning.offset),
        );
    }
    for expr in exprs {
        validate_carriers(expr, sink);
    }
    stack_scope.drain_into(sink);
    if !sink.is_empty() {
        return Err(failed());
    }
    let mut declarations = Vec::new();
    for expr in exprs {
        items(expr, None, &mut declarations);
    }
    stack_scope.drain_into(sink);
    if !sink.is_empty() {
        return Err(failed());
    }
    for (_, expr) in &declarations {
        let valid = if let Some((tag, _, children)) = stamped_parts(expr) {
            match tag {
                DeepTag::Deftype | DeepTag::Typealias => {
                    let params = match &children[1] {
                        deep::Expr::BareList(elements, _) => elements.as_slice(),
                        _ => unreachable!("validated type parameter list"),
                    };
                    let names: BTreeSet<_> = params.iter().filter_map(symbol_name).collect();
                    names.len() == params.len()
                        && (tag != DeepTag::Deftype
                            || children[2..].iter().all(|child| {
                                stamped_parts(child)
                                    .is_some_and(|(tag, _, _)| tag == DeepTag::Variant)
                            }))
                }
                DeepTag::Def | DeepTag::Defsig | DeepTag::Defdim | DeepTag::Export => true,
                // Import resolution belongs to the Reef linker, before this
                // API. An unlinked import is not an empty declaration.
                _ => false,
            }
        } else {
            false
        };
        if !valid {
            sink.push(CheckError::new(CheckErrorKind::MalformedForm,
                "expected a linked declaration with distinct nominal parameters and complete variants".to_string(), vec![])
                .at_offset(expr.span().offset));
        }
    }
    if !sink.is_empty() {
        return Err(failed());
    }
    let bare = declarations
        .iter()
        .map(|(_, expr)| *expr)
        .collect::<Vec<_>>();
    report_duplicate_defs(&bare, sink);
    report_duplicate_defsigs(&bare, sink);
    let (mut env, mut vg) = builtins::builtin_env();
    let mut registry = AdtRegistry::new();
    builtins::register_prelude_adts(&mut env, &mut vg, &mut registry);
    let headers = precollect_type_resolution_env(&declarations, &registry);
    registry.install_resolution_env(headers.clone());
    let mut subst = Subst::new();

    // Register fields while the alias table is empty, so register_deftype's
    // normal alias expansion preserves every edge. The same compiler helper
    // then resolves alias bodies without expanding them. No declaration body
    // is inferred, and all self/forward names were installed above.
    for phase in [DeclPhase::Rest, DeclPhase::Aliases] {
        for (module, expr) in &declarations {
            let Some((tag, _, _)) = stamped_parts(expr) else {
                continue;
            };
            if !matches!(tag, DeepTag::Deftype | DeepTag::Typealias) {
                continue;
            }
            collect_declarations(
                expr,
                module.as_deref(),
                &mut env,
                &mut vg,
                &mut subst,
                &mut registry,
                &headers,
                sink,
                phase,
                None,
            );
        }
    }
    let mut signatures = BTreeMap::new();
    for (_, expr) in &declarations {
        let Some((DeepTag::Defsig, meta, children)) = stamped_parts(expr) else {
            continue;
        };
        let Some((name_expr, binder_list, type_expr)) = defsig_parts(children) else {
            continue;
        };
        let name = symbol_name(name_expr).expect("validated defsig name");
        let Some(binders) = defsig_binder_names(binder_list, sink) else {
            continue;
        };
        let mut resolver = DeepTypeResolver::new(
            TypeUseSite::Defsig,
            BinderMode::ExplicitGeneric(&binders),
            &headers,
            &mut vg,
            sink,
        )
        .with_dtype_bounds(declaration_dtype_bounds(meta));
        if let Ok(ty) = resolver.resolve(type_expr) {
            let type_names = resolver
                .type_var_names()
                .into_sorted()
                .into_iter()
                .collect();
            if let Ok(bounds) = resolver.finish_dtype_bounds() {
                signatures.insert(
                    name.to_string(),
                    DeclaredSignature {
                        ty: ty.into_type(),
                        type_names,
                        bounds,
                    },
                );
            }
        }
    }
    stack_scope.drain_into(sink);
    if sink.is_empty() {
        Ok(DeclaredTypeSurface {
            registry,
            signatures,
        })
    } else {
        Err(failed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_recursive_declarations_preserve_alias_edges() {
        let source = chelis_deep::parser::parse_str(
            "(typealias {} A () (t-adt {} B)) (typealias {} B () (t-tuple {} (t-prim {} f64) (t-adt {} A)))"
        ).unwrap();
        let resolved = resolve_declared_surface(&source).unwrap();
        assert!(
            matches!(&resolved.registry().aliases["A"].body, Type::Adt(name, _) if name == "B")
        );
    }

    #[test]
    fn malformed_declaration_parts_never_form_a_partial_surface() {
        // The text parser already rejects these shapes. Build them through
        // the public node constructor instead: a row the constructor refuses
        // cannot reach the API at all, and every row it admits must be
        // rejected by the API rather than form a partial surface.
        type Built = Result<deep::Expr, chelis_deep::node::NodeError>;
        let span = Span::new(0, 0);
        let name = |name: &str| deep::Expr::Atom(deep::Atom::Name(name.to_string()), span);
        let params = |names| deep::Expr::BareList(names, span);
        let form = |tag, children: Vec<deep::Expr>| -> Built {
            DeepNode::try_new(tag, deep::Metadata::default(), children)
                .map(|node| deep::Expr::Node(Box::new(node), span))
        };
        let number = || deep::Expr::Atom(deep::Atom::Int(1), span);
        let float = || form(DeepTag::TPrim, vec![name("f64")]);
        let rows: Vec<(&str, Built)> = vec![
            (
                "defsig without a type",
                form(DeepTag::Defsig, vec![name("f")]),
            ),
            (
                "defsig with a numeric name",
                (|| form(DeepTag::Defsig, vec![number(), float()?]))(),
            ),
            (
                "typealias with a numeric name",
                (|| form(DeepTag::Typealias, vec![number(), params(vec![]), float()?]))(),
            ),
            (
                "deftype with a numeric name",
                form(DeepTag::Deftype, vec![number(), params(vec![])]),
            ),
            (
                "variant with a numeric name",
                (|| {
                    form(
                        DeepTag::Deftype,
                        vec![
                            name("A"),
                            params(vec![]),
                            form(DeepTag::Variant, vec![number(), float()?])?,
                        ],
                    )
                })(),
            ),
            (
                "field with a numeric name",
                (|| {
                    form(
                        DeepTag::Deftype,
                        vec![
                            name("A"),
                            params(vec![]),
                            form(
                                DeepTag::Variant,
                                vec![name("A"), form(DeepTag::Field, vec![number(), float()?])?],
                            )?,
                        ],
                    )
                })(),
            ),
            (
                "field without a type",
                (|| {
                    form(
                        DeepTag::Deftype,
                        vec![
                            name("A"),
                            params(vec![]),
                            form(
                                DeepTag::Variant,
                                vec![name("A"), form(DeepTag::Field, vec![name("x")])?],
                            )?,
                        ],
                    )
                })(),
            ),
            (
                "numeric type parameter",
                (|| {
                    form(
                        DeepTag::Typealias,
                        vec![name("A"), params(vec![number()]), float()?],
                    )
                })(),
            ),
            (
                "absent alias target",
                (|| {
                    form(
                        DeepTag::Typealias,
                        vec![
                            name("A"),
                            params(vec![]),
                            form(DeepTag::TAdt, vec![name("Absent")])?,
                        ],
                    )
                })(),
            ),
            (
                "duplicate type parameter",
                (|| {
                    form(
                        DeepTag::Typealias,
                        vec![
                            name("A"),
                            params(vec![name("a"), name("a")]),
                            form(DeepTag::TVar, vec![name("a")])?,
                        ],
                    )
                })(),
            ),
            (
                "deftype with a non-variant member",
                (|| form(DeepTag::Deftype, vec![name("A"), params(vec![]), float()?]))(),
            ),
            (
                "import without a name list",
                form(DeepTag::Import, vec![name("Absent")]),
            ),
            ("a bare number", Ok(number())),
        ];
        let mut admitted = 0;
        for (row, built) in rows {
            match built {
                Ok(source) => {
                    admitted += 1;
                    assert!(
                        resolve_declared_surface(&[source]).is_err(),
                        "{row}: an admitted malformed declaration formed a surface"
                    );
                }
                Err(error) => assert!(
                    matches!(error, chelis_deep::node::NodeError::ArityViolation { .. }),
                    "{row}: the node constructor refused it for an unexpected reason: {error}"
                ),
            }
        }
        // Most rows are admitted by the constructor, so the API's own
        // rejection stays exercised.
        assert!(admitted >= 9, "only {admitted} rows reached the API");
    }
}
