//! Declaration collection, the checker's first pass over a program.
//!
//! One responsibility: register every top-level declaration's header before
//! any body is inferred. That covers the type-resolution pre-collection,
//! nominal kind evidence, opacity metadata, `collect_declarations` itself,
//! and the duplicate, orphan and builtin-shadowing reports it owns.

use super::*;

// ── Declaration collection (first pass) ──────────────────────────

/// Which declaration kinds a `collect_declarations` sub-pass should process.
///
/// `deftype` constructor schemes expand transparent type aliases in their
/// field types at registration (see `AdtRegistry::expand_aliases`), so every
/// `typealias` must be in the registry first. Running `Aliases` over all
/// top-level items before `Rest` guarantees that even for a forward reference
/// — an alias declared textually after the `deftype` that uses it, as in
/// `Hull.Ast` where `type EffectRow = List[Effect]` follows `type Type = ...
/// | TArrow(Type, Type, EffectRow) | ...`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeclPhase {
    /// Process only `typealias` declarations.
    Aliases,
    /// Process everything except `typealias` (`deftype`, `defsig`, ...).
    Rest,
}

pub(super) fn definition_owns_function_metadata_prebind(expr: &deep::Expr) -> bool {
    let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
        return false;
    };
    let Some(body) = kids.get(1) else {
        return false;
    };
    tagged_children(body, DeepTag::Fn).is_some()
}

/// Record the [04-INF-4] source position of every top-level eager value, and
/// the outer binding each one shadows.
///
/// This runs before any signature is collected, so `env.lookup` still sees the
/// pre-existing import or stacked-library binding rather than this unit's
/// own. The ordinal is the value's `def`, never a separated sibling `defsig`:
/// a signature is metadata about a declaration, not the declaration itself,
/// so it may not publish the value early. The first `def` of a duplicated
/// name owns the position; the duplicate itself is already an error.
fn note_eager_value_ordinals(items: &[(Option<String>, &deep::Expr)], env: &mut Env) {
    env.reset_top_level_value_scope();
    for (declaration_index, (_, expr)) in items.iter().enumerate() {
        let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        if definition_owns_function_metadata_prebind(expr) {
            continue;
        }
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        if env.top_level_value_ordinal(name).is_some() {
            continue;
        }
        let shadowed = env.lookup(name).cloned();
        env.note_top_level_value_ordinal(name.to_string(), declaration_index, shadowed);
    }
}

/// Run the two-phase declaration collection over `items` (already flattened
/// past `module` wrappers, each paired with its lexical module key):
/// register all type aliases, then everything else.
///
/// Every declared signature stays in the global header environment, including
/// an eager value's. [04-INF-4] scope is decided by
/// [`Env::top_level_value_visibility`] at each reference, not by withholding
/// or replaying bindings along the inference schedule: the schedule reorders
/// function bodies, so a binding timeline cannot express source order.
pub(super) fn collect_all_declarations(
    items: &[(Option<String>, &deep::Expr)],
    declaration_diagnostic_owners: &[Option<DeclarationDiagnosticOwner>],
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &mut AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) {
    debug_assert_eq!(items.len(), declaration_diagnostic_owners.len());
    // chelis#258 (main): duplicate-def / builtin-shadowing rejection runs
    // over the bare item list. Our `items` is paired with module keys, so
    // project to the `&deep::Expr` slice the reporters expect.
    let bare_items: Vec<&deep::Expr> = items.iter().map(|(_, expr)| *expr).collect();
    report_duplicate_defs(&bare_items, errors);
    report_duplicate_defsigs(&bare_items, errors);
    report_orphan_defsigs(items, errors);
    report_builtin_shadowing(&bare_items, errors);
    report_reserved_binders(&bare_items, errors);
    note_eager_value_ordinals(items, env);
    let resolution_env = precollect_type_resolution_env(items, adt_reg);
    // Install the provisional self/forward header scope explicitly in this
    // per-check registry clone. Declaration bodies resolve against it, while
    // only successful bodies enter the validated maps that survive serde.
    adt_reg.install_resolution_env(resolution_env.clone());
    // chelis#930: per-declaration cancellation. This runs BEFORE body
    // inference, so without it the first ~1.3 s of a 1500-declaration check
    // (measured, debug build) is uninterruptible and a cancellation arriving
    // in that window waits it out. An abandoned collection leaves later
    // declarations unbound; the check entry's `cancellation_gate` rejects the
    // unit before anything reads it.
    let cancel = crate::cancel::current_cancel_token();
    for (index, (module, expr)) in items.iter().enumerate() {
        if cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            return;
        }
        let diagnostic_owner = declaration_diagnostic_owners
            .get(index)
            .and_then(Option::as_ref);
        collect_declarations(
            expr,
            module.as_deref(),
            env,
            vg,
            subst,
            adt_reg,
            &resolution_env,
            errors,
            DeclPhase::Aliases,
            diagnostic_owner,
        );
    }
    for (index, (module, expr)) in items.iter().enumerate() {
        if cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            return;
        }
        let diagnostic_owner = declaration_diagnostic_owners
            .get(index)
            .and_then(Option::as_ref);
        collect_declarations(
            expr,
            module.as_deref(),
            env,
            vg,
            subst,
            adt_reg,
            &resolution_env,
            errors,
            DeclPhase::Rest,
            diagnostic_owner,
        );
    }
    // [04-LIN-10] / spec/04 section 8.4.1: every `type` of this check, and of
    // the library context it extends, is registered now and no body has been
    // inferred yet, so the key-carrying set is complete before any generic is
    // instantiated.
    subst.set_key_carrying_adts(adt_reg.key_carrying_adts());
}

/// Collect nominal names and arities before resolving any declaration body.
/// This permits self and forward references without registering an unchecked
/// definition in the serde-backed ADT registry.
pub(super) fn precollect_type_resolution_env(
    items: &[(Option<String>, &deep::Expr)],
    adt_reg: &AdtRegistry,
) -> TypeResolutionEnv {
    let mut headers = TypeResolutionEnv::from_registry(adt_reg);
    let mut declarations: UnordMap<String, (Vec<String>, Vec<&deep::Expr>)> = UnordMap::new();
    for (_, expr) in items {
        let Some((tag, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        if !matches!(tag, DeepTag::Deftype | DeepTag::Typealias) {
            continue;
        }
        let (Some(name), Some(params_expr)) = (kids.first().and_then(symbol_name), kids.get(1))
        else {
            continue;
        };
        if stamped_parts(params_expr).is_some_and(|(tag, _, _)| tag == DeepTag::Variant) {
            // Legacy Deep permits omitting the explicit empty parameter list.
            declarations.insert(name.to_string(), (Vec::new(), kids[1..].iter().collect()));
        } else {
            let params = match params_expr {
                deep::Expr::BareList(elements, _) => elements.as_slice(),
                _ => continue,
            };
            if params.iter().all(|param| symbol_name(param).is_some()) {
                let names = params
                    .iter()
                    .filter_map(symbol_name)
                    .map(str::to_string)
                    .collect::<Vec<_>>();
                let bodies = if tag == DeepTag::Typealias {
                    kids.get(2).into_iter().collect()
                } else {
                    kids[2..].iter().collect()
                };
                declarations.insert(name.to_string(), (names, bodies));
            }
        }
    }

    // Kind inference is a deterministic fixed point over declaration
    // headers. A parameter becomes dimension-kinded only when every observed
    // use is dimensional; type, mixed, and unused parameters remain ordinary
    // types, preserving the language's unkinded-binder default.
    let mut evidence = declarations
        .to_sorted()
        .into_iter()
        .map(|(name, (params, _))| (name.clone(), vec![(false, false); params.len()]))
        .collect::<UnordMap<_, _>>();
    loop {
        let before = evidence.clone();
        for (name, (params, bodies)) in declarations.to_sorted() {
            let indices = params
                .iter()
                .enumerate()
                .map(|(index, param)| (param.as_str(), index))
                .collect::<UnordMap<_, _>>();
            for body in bodies {
                collect_nominal_kind_evidence(
                    body,
                    Some(NominalParamKind::Type),
                    &indices,
                    &mut evidence,
                    name,
                    &headers,
                );
            }
        }
        if evidence == before {
            break;
        }
    }
    for (name, (params, _)) in declarations.into_sorted() {
        let kinds = evidence
            .remove(&name)
            .unwrap_or_else(|| vec![(false, false); params.len()])
            .into_iter()
            .map(|(type_use, dimension_use)| {
                if dimension_use && !type_use {
                    NominalParamKind::Dimension
                } else {
                    NominalParamKind::Type
                }
            })
            .collect();
        headers.insert(name, kinds);
    }
    headers
}

fn collect_nominal_kind_evidence(
    expr: &deep::Expr,
    context: Option<NominalParamKind>,
    own_params: &UnordMap<&str, usize>,
    evidence: &mut UnordMap<String, Vec<(bool, bool)>>,
    owner: &str,
    existing_headers: &TypeResolutionEnv,
) {
    stack_guard!("collect_nominal_kind_evidence", expr);
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        return;
    };
    let record = |name: &str,
                  context: Option<NominalParamKind>,
                  evidence: &mut UnordMap<String, Vec<(bool, bool)>>| {
        let Some(index) = own_params.get(name).copied() else {
            return;
        };
        let Some(entries) = evidence.get_mut(owner) else {
            return;
        };
        match context {
            Some(NominalParamKind::Type) => entries[index].0 = true,
            Some(NominalParamKind::Dimension) => entries[index].1 = true,
            None => {}
        }
    };
    match tag {
        DeepTag::TVar => {
            if let Some(name) = kids.first().and_then(symbol_name) {
                record(name, context, evidence);
            }
        }
        DeepTag::DVar | DeepTag::DRank => {
            if let Some(name) = kids.first().and_then(symbol_name) {
                record(name, Some(NominalParamKind::Dimension), evidence);
            }
        }
        DeepTag::TTensor => {
            if let Some((precision, dimensions)) = kids.split_last() {
                for dimension in dimensions {
                    collect_nominal_kind_evidence(
                        dimension,
                        Some(NominalParamKind::Dimension),
                        own_params,
                        evidence,
                        owner,
                        existing_headers,
                    );
                }
                collect_nominal_kind_evidence(
                    precision,
                    Some(NominalParamKind::Type),
                    own_params,
                    evidence,
                    owner,
                    existing_headers,
                );
            }
        }
        DeepTag::TAdt => {
            let Some(target) = kids.first().and_then(symbol_name) else {
                return;
            };
            for (index, argument) in kids.iter().skip(1).enumerate() {
                let propagated = evidence
                    .get(target)
                    .and_then(|entries| entries.get(index))
                    .and_then(|(type_use, dimension_use)| {
                        if *type_use {
                            Some(NominalParamKind::Type)
                        } else if *dimension_use {
                            Some(NominalParamKind::Dimension)
                        } else {
                            None
                        }
                    })
                    .or_else(|| {
                        existing_headers
                            .param_kinds(target)
                            .and_then(|kinds| kinds.get(index).copied())
                    });
                collect_nominal_kind_evidence(
                    argument,
                    propagated,
                    own_params,
                    evidence,
                    owner,
                    existing_headers,
                );
            }
        }
        DeepTag::TFn | DeepTag::TRef | DeepTag::TTuple => {
            for child in kids {
                collect_nominal_kind_evidence(
                    child,
                    Some(NominalParamKind::Type),
                    own_params,
                    evidence,
                    owner,
                    existing_headers,
                );
            }
        }
        _ => {
            for child in kids {
                collect_nominal_kind_evidence(
                    child,
                    context,
                    own_params,
                    evidence,
                    owner,
                    existing_headers,
                );
            }
        }
    }
}

/// Build the program-shape opacity metadata (RFC D-CHECK) from the
/// flattened `(module key, item)` list: per-module export sets, the
/// top-level binding -> module map, and the formatted "exported
/// producers with signatures" entries per opaque type used by the
/// violation error contract. Runs after declaration collection so the
/// registry already carries every `deftype`'s `opaque` flag and
/// defining module.
pub(super) fn build_opacity_meta(
    items: &[(Option<String>, &deep::Expr)],
    adt_reg: &AdtRegistry,
    env: &Env,
) -> crate::opacity::OpacityModuleMeta {
    let mut meta = crate::opacity::OpacityModuleMeta::default();
    // Declared signature types (from `defsig` nodes) for producer
    // display; keyed by binding name like `meta.bindings`.
    let mut declared_sigs: UnordMap<String, Type> = UnordMap::new();
    // Pass 1: export sets. Lexical `(export ...)` nodes attribute to
    // their module wrapper; package-linked exports arrive as
    // top-level nodes whose names carry the reef internal-name stem
    // (the reef rewrite emits them with internal names), so each
    // exported name self-attributes through its stem.
    for (module, item) in items {
        // chelis#1107: carrier-preserving read. A `List`-only destructure
        // skipped every stamped declaration, leaving both maps empty on the
        // stamped ingress -- the opacity diagnostic then reported "exported
        // producers: none" for a module that exports one.
        let Some((tag, _, kids)) = stamped_parts(item) else {
            continue;
        };
        if tag != DeepTag::Export {
            continue;
        }
        for child in kids {
            let Some(name) = symbol_name(child) else {
                continue;
            };
            let target = match module {
                Some(module) => Some(module.clone()),
                None => crate::opacity::reef_module_stem(name),
            };
            if let Some(target) = target {
                meta.exports
                    .entry(target)
                    .or_default()
                    .insert(name.to_string());
            }
        }
    }
    // Pass 2: binding -> module attribution and declared sigs.
    // Stem-attributed (package-linked) bindings are recorded only
    // when their module's export set is known: without it, the sixth
    // rejection's no-export-decl-means-sealed rule would reject
    // legitimately exported producers in pipelines that strip Export
    // decls (fail-open for unattributable names by design).
    for (module, item) in items {
        // chelis#1107: carrier-preserving read, as in pass 1 above.
        let Some((tag, _, kids)) = stamped_parts(item) else {
            continue;
        };
        if !matches!(tag, DeepTag::Def | DeepTag::Defsig) {
            continue;
        }
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let target = match module {
            Some(module) => Some(module.clone()),
            None => crate::opacity::reef_module_stem(name)
                .filter(|stem| meta.exports.contains_key(stem)),
        };
        let Some(target) = target else {
            continue;
        };
        meta.bindings.insert(name.to_string(), target);
        if tag == DeepTag::Defsig
            && let Some(scheme) = env.lookup(name)
        {
            declared_sigs.insert(name.to_string(), scheme.body.clone());
        }
    }
    // Producer enumeration per opaque type: exported bindings of the
    // defining module whose declared RESULT type mentions the type
    // (containment chased through named type definitions).
    for (adt_name, def) in &adt_reg.defs {
        if !def.opaque {
            continue;
        }
        let Some(module) = &def.defining_module else {
            continue;
        };
        let Some(export_set) = meta.exports.get(module) else {
            continue;
        };
        let mut entries: std::collections::BTreeSet<String> = Default::default();
        for name in export_set {
            if meta.bindings.get(name) != Some(module) {
                continue;
            }
            let Some(sig) = declared_sigs.get(name) else {
                continue;
            };
            let result = match sig {
                Type::Fn(_, ret) => ret.as_ref(),
                other => other,
            };
            if crate::opacity::type_mentions_adt(result, adt_name, adt_reg) {
                // RT-1 F3: store the producer entry de-mangled so the
                // reef surface renders `probability: (f32) -> Probability`
                // rather than the internal `pkg__...` names.
                entries.insert(format!(
                    "{}: {}",
                    crate::opacity::demangle_ident(name),
                    crate::opacity::demangle_type(sig)
                ));
            }
        }
        if !entries.is_empty() {
            meta.producer_entries
                .entry(adt_name.clone())
                .or_default()
                .extend(entries);
        }
    }
    meta
}

/// Reject two same-name `def` declarations in one program (chelis#258).
///
/// A def's value binding is silent last-write-wins (`env.bind` →
/// `UnordMap::insert`, like the `defsig` arm of `collect_declarations`), and
/// Chelis does not dispatch same-name `def`s by argument arity or tensor
/// rank. So two `def f`s whose sigs differ only in rank leave just one arm
/// reachable: callers of the other rank fire a confusing `DimensionMismatch`
/// at the call site instead of a clear error at the redundant definition.
/// This mirrors the duplicate-`deftype` / duplicate-`typealias` rejection
/// already in `collect_declarations`, moving the diagnostic to the
/// definition site.
///
/// Scoped to `def` (not `defsig`): a `defsig` legitimately co-occurs with a
/// synthesized signature for the same name (an inline-annotated `def`
/// desugars to both a `defsig` and a `def`), so a same-name `defsig` is not
/// on its own a duplicate definition. `items` is already flattened past
/// `module` wrappers, and the prelude lives in the builtin env rather than as
/// `def` nodes here, so only genuine in-program user redefinitions match.
pub(super) fn report_duplicate_defs(items: &[&deep::Expr], errors: &mut DiagnosticSink<'_>) {
    let mut seen: UnordSet<&str> = UnordSet::new();
    for expr in items {
        // chelis#1107: `stamped_parts` reads both carriers. A `List`-only
        // destructure skipped every stamped declaration, so this check fired
        // on `check_ir_program` (which normalizes Node to List) and never on
        // `check_typed_program`.
        let Some((tag, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        if tag != DeepTag::Def {
            continue;
        }
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        if !seen.insert(name) {
            errors.push(at_check_site(expr, CheckError::new(
                CheckErrorKind::DuplicateDefinition,
                format!("duplicate definition: `{name}` is defined more than once"),
                vec![format!(
                    "rename one of the `{name}` definitions: Chelis does not dispatch same-name `def`s by argument type or rank"
                )],
            )));
        }
    }
}

/// Reject two same-name `defsig` declarations in one program.
///
/// Chelis does not dispatch user functions by arity, type, or rank; the valid
/// same-name declaration pair is exactly one `defsig` plus one `def`. Multiple
/// `defsig`s for a name otherwise feed several last-write-wins maps
/// (`collect_declarations`, declared-param-type collection, signature metadata)
/// and make the enforced signature order-dependent.
pub(super) fn report_duplicate_defsigs(items: &[&deep::Expr], errors: &mut DiagnosticSink<'_>) {
    let mut seen: UnordSet<&str> = UnordSet::new();
    for expr in items {
        // chelis#1107: carrier-preserving read, as in `report_duplicate_defs`.
        let Some((tag, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        if tag != DeepTag::Defsig {
            continue;
        }
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        if !seen.insert(name) {
            errors.push(at_check_site(expr, CheckError::new(
                CheckErrorKind::DuplicateDefinition,
                format!("duplicate signature: `{name}` has more than one `defsig`"),
                vec![format!(
                    "keep a single `defsig` for `{name}`: Chelis does not dispatch same-name functions by argument type, arity, or rank"
                )],
            )));
        }
    }
}

/// Reject a `defsig` that has no same-name `def` in the same check unit and
/// lexical module. A signature describes a Chelis definition; it is not an
/// extern/runtime declaration. Letting it create a callable binding by itself
/// makes `check` accept a symbol that no lowering lane can define (#850).
pub(super) fn report_orphan_defsigs(
    items: &[(Option<String>, &deep::Expr)],
    errors: &mut DiagnosticSink<'_>,
) {
    // Linked dependency interfaces intentionally contain signature-only rows:
    // their bodies live in the supplying package artifact. Reef validates
    // every authored source module and synthetic entry before installing the
    // linked-program guard. The exemption therefore requires both that
    // in-process provenance and the linker's reserved mangled-name format;
    // neither fact alone can exempt an authored orphan. Raw Deep/Surf units
    // and persistent checker contexts still take the same-unit check below.
    let mut defs: UnordSet<(Option<&str>, &str)> = UnordSet::new();
    for (module, expr) in items {
        let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        if let Some(name) = kids.first().and_then(symbol_name) {
            defs.insert((module.as_deref(), name));
        }
    }

    for (module, expr) in items {
        let Some((DeepTag::Defsig, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        if crate::opacity::linked_program() && crate::opacity::is_linker_format_name(name) {
            continue;
        }
        if !defs.contains(&(module.as_deref(), name)) {
            let qualified = module
                .as_deref()
                .map(|module| format!("{module}.{name}"))
                .unwrap_or_else(|| name.to_string());
            errors.push(CheckError::new(
                CheckErrorKind::UnboundVariable {
                    identifier: qualified.clone(),
                },
                format!(
                    "defsig `{qualified}` has no matching `def` in the same check unit: \
                     signatures describe Chelis definitions and do not declare runtime symbols"
                ),
                vec![
                    format!("add `def {name}` beside the signature, or remove the orphan signature"),
                    "A future external-call surface must use an explicit typed capability; a bare `defsig` is not one"
                        .to_string(),
                ],
            ));
        }
    }
}

/// Reject a top-level `def` or `defsig` whose name appears in the closed
/// builtin vocabulary (chelis#353, spec/04-type-system.md §8.6).
///
/// Top-level builtin names identify the language's intrinsic call surface.
/// This check imports `BUILTIN_NAMES`, the same closed table evaluator and
/// lowering dispatch consume, so the reserved declaration set cannot drift
/// from that surface. Rejecting a same-name top-level declaration prevents a
/// user signature from redefining the intrinsic operation. Lexical bindings
/// are different: ordinary scope resolution selects them before builtin
/// dispatch in every execution lane (chelis#1076).
///
/// Deliberately narrow scope:
/// - Reef package modules never reach this check with bare names: reef
///   rewrites package decls to internal `pkg__...` names (and rewrites
///   their call sites with them) before the checker runs, so a package
///   `def sum` is allowed and genuinely dispatches to the user def (the
///   stdlib's `Std.Test.fail` relies on this).
/// - Function parameters and block-locals may reuse builtin names: they
///   shadow the builtin under ordinary lexical scoping in every lane.
///
/// An inline-annotated `def` desugars to a `defsig` AND a `def` with the
/// same name; report once per name, as the `def` (what the user wrote).
pub(super) fn report_builtin_shadowing(items: &[&deep::Expr], errors: &mut DiagnosticSink<'_>) {
    let decl_name = |expr: &deep::Expr, tag: DeepTag| -> Option<String> {
        // chelis#1107: carrier-preserving read. Without it the §8.6 shadowing
        // gate never fired on the stamped ingress.
        let (found_tag, _, kids) = stamped_parts(expr)?;
        if found_tag != tag {
            return None;
        }
        kids.first()
            .and_then(symbol_name)
            .filter(|name| builtins::BUILTIN_NAMES.contains(name))
            .map(str::to_string)
    };

    let def_names: UnordSet<String> = items
        .iter()
        .filter_map(|expr| decl_name(expr, DeepTag::Def))
        .collect();

    let mut reported: UnordSet<String> = UnordSet::new();
    for expr in items {
        let Some(name) = decl_name(expr, DeepTag::Def).or_else(|| decl_name(expr, DeepTag::Defsig))
        else {
            continue;
        };
        if !reported.insert(name.clone()) {
            continue;
        }
        let decl_kw = if def_names.contains(&name) {
            "def"
        } else {
            "sig"
        };
        errors.push(CheckError::new(
            CheckErrorKind::BuiltinShadowing,
            format!(
                "`{decl_kw} {name}` shadows the builtin function `{name}`: user `def`/`sig` \
                 declarations may not reuse builtin names (spec/04-type-system.md \u{00a7}8.6). \
                 Top-level builtin names identify the intrinsic call surface and cannot be \
                 rebound with a user signature."
            ),
            vec![format!(
                "rename `{name}` (e.g. `{name}2` or `my_{name}`); inside a reef package \
                 module the name is allowed because package declarations are \
                 internal-name-rewritten before checking"
            )],
        ));
    }
}

/// spec/04-type-system.md §8.6: the intrinsic names no binder may bind.
pub(super) const RESERVED_INTRINSIC_NAMES: &[&str] = &["to_tensor"];

/// spec/04-type-system.md §8.6 at Deep ingress: a binder of a reserved
/// intrinsic name in any scope (a `def` or `defsig`, an import, a function
/// parameter, a `bind`, or a pattern variable) is `ReservedName`.
pub(super) fn report_reserved_binders(items: &[&deep::Expr], errors: &mut DiagnosticSink<'_>) {
    fn reserved(name: &str, binder: &str, expr: &deep::Expr, errors: &mut DiagnosticSink<'_>) {
        if !RESERVED_INTRINSIC_NAMES.contains(&name) {
            return;
        }
        let error = CheckError::new(
            CheckErrorKind::ReservedName,
            format!(
                "`{name}` is reserved and cannot be bound (spec/04-type-system.md \u{00a7}8.6): \
                 it states literal dtypes (\u{00a7}5.6) and always names the intrinsic \
                 conversion; rename this {binder}"
            ),
            vec![],
        );
        errors.push(match TypeDiagnosticLocation::from_expr(expr) {
            Some(location) => location.attach(error),
            None => error,
        });
    }
    fn walk(expr: &deep::Expr, errors: &mut DiagnosticSink<'_>) {
        stack_guard!("report_reserved_binders", expr);
        match expr {
            deep::Expr::BareList(elements, _) => {
                for element in elements {
                    walk(element, errors);
                }
                return;
            }
            deep::Expr::MetaExpr(meta, _) => return walk(&meta.expr, errors),
            _ => {}
        }
        let Some((tag, _, kids)) = stamped_parts(expr) else {
            return;
        };
        match tag {
            DeepTag::Def | DeepTag::Defsig | DeepTag::Bind | DeepTag::PatVar => {
                if let Some(name) = kids.first().and_then(symbol_name) {
                    let binder = match tag {
                        DeepTag::Def => "definition",
                        DeepTag::Defsig => "signature",
                        DeepTag::Bind => "binding",
                        _ => "pattern binder",
                    };
                    reserved(name, binder, expr, errors);
                }
            }
            DeepTag::Params => {
                for param in kids {
                    if let Some(name) = param_name_for_refs(param) {
                        reserved(&name, "parameter", expr, errors);
                    }
                }
            }
            DeepTag::Import => {
                if let Some(deep::Expr::BareList(names, _)) = kids.get(1) {
                    for name in names.iter().filter_map(symbol_name) {
                        reserved(name, "import", expr, errors);
                    }
                }
            }
            _ => {}
        }
        for kid in kids {
            walk(kid, errors);
        }
    }
    for item in items {
        walk(item, errors);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn collect_declarations(
    expr: &deep::Expr,
    lexical_module: Option<&str>,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &mut AdtRegistry,
    headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
    phase: DeclPhase,
    declaration_diagnostic_owner: Option<&DeclarationDiagnosticOwner>,
) {
    let Some((tag, meta, kids)) = stamped_parts(expr) else {
        return;
    };

    // Aliases register first so `deftype` field-type alias expansion sees a
    // fully-populated alias table; every other decl kind runs in the second
    // sub-pass.
    let in_phase = match phase {
        DeclPhase::Aliases => tag == DeepTag::Typealias,
        DeclPhase::Rest => tag != DeepTag::Typealias,
    };
    if !in_phase {
        return;
    }

    match tag {
        DeepTag::Deftype => {
            // Reject same-namespace collisions (another `deftype`, a
            // `typealias`, or a prelude ADT registered earlier in this
            // program). Without this check `AdtRegistry::defs` is
            // silently last-write-wins, which propagates wrong
            // constructor types and (per `compute_tensor_carrying_adts`
            // in linearity.rs) order-dependent borrow semantics.
            if let Some(name) = kids.first().and_then(symbol_name)
                && let Some(prior_kind) = adt_reg.existing_kind(name)
            {
                errors.push(at_check_site(expr, CheckError::new(
                    CheckErrorKind::DuplicateDefinition,
                    format!(
                        "duplicate type definition: `{name}` was already declared as a {prior_kind}"
                    ),
                    vec![format!("rename one of the `{name}` declarations")],
                )));
                return;
            }
            // RFC D-CHECK: record opacity + module identity on the
            // registered AdtDef. The module key is the lexical
            // wrapper when present, else the reef internal-name stem
            // of the deftype's own (rewritten) name. `@opaque`
            // requires a named module (RT-0 M6): a top-level opaque
            // declaration has no module identity, which would make
            // the enforcement boundary collide across combined
            // sources.
            let opaque = deftype_opaque_meta(meta);
            let defining_module = crate::opacity::module_key_for_item(
                lexical_module,
                kids.first().and_then(symbol_name),
            );
            if opaque
                && defining_module.is_none()
                && let Some(name) = kids.first().and_then(symbol_name)
            {
                errors.push(crate::opacity::unmoduled_opaque_error(name));
            }
            let adt_name = kids.first().and_then(symbol_name).map(str::to_string);
            if let Ok(ctors) =
                adt_reg.register_deftype(kids, vg, headers, errors, opaque, defining_module)
            {
                for (name, scheme) in ctors {
                    if let Some(owner) = &adt_name {
                        env.bind_constructor(name, owner.clone(), scheme);
                    } else {
                        // `register_deftype` reports malformed declarations;
                        // retain the former defensive binding behavior if a
                        // future parser shape can return constructors without
                        // an authored owner name.
                        env.bind(name, scheme);
                    }
                }
            }
        }
        DeepTag::Defsig => {
            // (defsig {dtype_bounds?} name [(binders...)] type_expr)
            if let Some((name_expr, binder_list, type_expr)) = defsig_parts(kids)
                && let Some(name) = symbol_name(name_expr)
                && let Some(binder_names) = defsig_binder_names(binder_list, errors)
            {
                let dtype_bounds = declaration_dtype_bounds(meta);
                let signature_level = subst.enter_level(vg);
                let resolved = resolve_deep_type_with_binder_identities(
                    type_expr,
                    vg,
                    adt_reg,
                    TypeUseSite::Defsig,
                    BinderMode::ExplicitGeneric(&binder_names),
                    dtype_bounds,
                    declaration_diagnostic_owner,
                    errors,
                );
                let bounds = match &resolved {
                    Ok(resolved) => &resolved.bounds,
                    Err(rejected) => &rejected.bounds,
                };
                let installed = install_declared_bounds(bounds, subst, name, errors);
                subst.leave_level(signature_level, vg);
                match (resolved, installed) {
                    (Ok(resolved), Ok(())) => {
                        let scheme = env.generalize(&resolved.ty, subst);
                        subst.name_generic_parameters(
                            &scheme,
                            name,
                            &resolved.binder_identities.type_names(),
                        );
                        env.bind(name.to_string(), scheme);
                        env.record_declared_binder_identities(name, resolved.binder_identities);
                    }
                    (Ok(resolved), Err(witness)) => {
                        let recovery = crate::deep_type::RejectedSignatureType::from_resolved(
                            resolved.ty,
                            witness,
                        );
                        env.bind_rejected_signature(name.to_string(), recovery, subst);
                        env.record_declared_binder_identities(name, resolved.binder_identities);
                    }
                    (Err(rejected), _) => {
                        env.bind_rejected_signature(name.to_string(), rejected.recovery, subst);
                        env.record_declared_binder_identities(name, rejected.binder_identities);
                    }
                }
            }
        }
        DeepTag::Typealias => {
            // (typealias {} Name (params...) type_expr)
            if kids.len() >= 3
                && let Some(name) = symbol_name(&kids[0])
            {
                if let Some(prior_kind) = adt_reg.existing_kind(name) {
                    errors.push(at_check_site(expr, CheckError::new(
                        CheckErrorKind::DuplicateDefinition,
                        format!(
                            "duplicate type definition: `{name}` was already declared as a {prior_kind}"
                        ),
                        vec![format!("rename one of the `{name}` declarations")],
                    )));
                    return;
                }
                let params = match &kids[1] {
                    deep::Expr::BareList(elements, _) => elements
                        .iter()
                        .filter_map(symbol_name)
                        .map(str::to_string)
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                };

                let param_kinds = headers
                    .param_kinds(name)
                    .map(<[NominalParamKind]>::to_vec)
                    .unwrap_or_else(|| vec![NominalParamKind::Type; params.len()]);
                let explicit_params = params
                    .iter()
                    .cloned()
                    .zip(param_kinds.iter().copied())
                    .collect::<UnordMap<_, _>>();
                let mut resolver = DeepTypeResolver::new(
                    TypeUseSite::TypeAliasBody,
                    BinderMode::ExplicitKinds(&explicit_params),
                    headers,
                    vg,
                    errors,
                );
                if let Ok(aliased_ty) = resolver.resolve(&kids[2]) {
                    let param_args = params
                        .iter()
                        .zip(&param_kinds)
                        .map(|(param, kind)| match kind {
                            NominalParamKind::Type => NominalArg::Type(Type::Var(
                                resolver
                                    .type_var(param)
                                    .expect("type-kinded alias parameter is pre-bound"),
                            )),
                            NominalParamKind::Dimension => NominalArg::Dimension(Dim::Var(
                                resolver
                                    .dim_var(param)
                                    .expect("dimension-kinded alias parameter is pre-bound"),
                            )),
                        })
                        .collect();
                    adt_reg.register_alias(
                        name.to_string(),
                        params,
                        param_kinds,
                        param_args,
                        aliased_ty.into_type(),
                    );
                }
            }
        }
        DeepTag::Def => {
            // `spec/03-deep-syntax.md` §2.2: a declaration's signature owns
            // its binders, so a dtype-family bound on a `def` is an error
            // rather than a second, silently-preferred source of truth. Surf
            // routes a `def [..]` bound here only when a standalone `sig`
            // already declares the name; without one the desugarer emits the
            // bound on the synthesized `defsig` instead.
            if let Some(name) = kids.first().and_then(symbol_name)
                && meta.dtype_bounds().is_some()
            {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "`{name}` declares a dtype-family bound on its `def`, but a declaration's `defsig` owns its binders"
                    ),
                    vec![format!("declare the bound on `{name}`'s signature")],
                ));
            }
        }
        _ => {}
    }
}
