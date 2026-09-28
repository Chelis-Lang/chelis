//! In-memory declaration linking for a complete source package.
//!
//! Imports and symbol identities use Reef's compiler linker. This entry does
//! no fetching, package-cache writes, or expression inference. An import
//! outside the supplied package is an error; callers cannot silently omit a
//! dependency and still obtain a complete declaration surface.

use super::*;

#[derive(Debug, Clone)]
pub struct LinkedDeclarations {
    pub source_label: String,
    pub module_name: String,
    /// Original source spelling -> the compiler's package/module identity.
    pub names: BTreeMap<String, String>,
    pub exports: BTreeSet<String>,
    pub declarations: Vec<Decl>,
}

pub fn link_package_declarations(
    package: &str,
    sources: &[(String, Vec<Decl>)],
) -> Result<Vec<LinkedDeclarations>, String> {
    let mut modules = BTreeMap::new();
    for (label, source) in sources {
        let [Decl::Module { name, decls, .. }] = source.as_slice() else {
            return Err(format!(
                "{label} must contain exactly one top-level module declaration"
            ));
        };
        validate_source_signature_pairs(decls, name)?;
        let module = ModuleSource {
            package_name: package.to_string(),
            module_name: name.clone(),
            decls: decls.clone(),
            file_rel: PathBuf::from(label),
            source_root: "src".to_string(),
            exports: compute_exports(decls),
            symbols: collect_symbol_kinds(decls),
        };
        if modules.insert(name.clone(), module).is_some() {
            return Err(format!("duplicate module `{name}` in declaration surface"));
        }
    }
    // The normal compiler linker consumes a PackageGraph. This in-memory
    // graph carries only supplied source; none of its cache/installation
    // fields are consulted by build_name_resolver or build_internal_maps.
    let graph = PackageGraph {
        root_package: package.to_string(),
        packages: BTreeMap::from([(
            package.to_string(),
            LoadedPackage {
                id: PackageId {
                    name: package.to_string(),
                    version: String::new(),
                },
                manifest: ReefManifest {
                    package: ManifestPackage {
                        name: package.to_string(),
                        version: String::new(),
                        compiler: String::new(),
                        module_prefix: String::new(),
                        additional_sources: vec![],
                    },
                    dependencies: BTreeMap::new(),
                    chelis_src: None,
                    conform: None,
                    artifacts: BTreeMap::new(),
                },
                modules,
                source: LoadedSourceKind::Root {
                    root: PathBuf::new(),
                },
                archive_sha256: None,
                shell_sha256: None,
                shell: None,
                remote_origin: None,
            },
        )]),
    };
    let internal_maps = build_internal_maps(&graph);
    let package_source = &graph.packages[package];
    let type_names = package_source
        .modules
        .values()
        .flat_map(|module| {
            module
                .symbols
                .iter()
                .filter(|(_, kind)| matches!(kind, SymbolKind::Type | SymbolKind::Dim))
                .map(|(name, _)| internal_name(package, &module.module_name, name))
        })
        .collect::<BTreeSet<_>>();
    let mut result = Vec::new();
    for module in package_source.modules.values() {
        let resolver = build_name_resolver(module, &graph, &internal_maps, &BTreeMap::new())?;
        let names = resolver
            .own_names
            .to_sorted()
            .into_iter()
            .map(|(name, internal)| (name.clone(), internal.clone()))
            .collect();
        let mut declarations = Vec::new();
        for source_decl in &module.decls {
            let mut decl = source_decl.clone();
            let bound = match &mut decl {
                Decl::FunDef {
                    body,
                    type_binders,
                    span,
                    ..
                } => {
                    *body = Expr::Tuple(vec![], *span);
                    type_binders
                        .iter()
                        .map(|binder| binder.name.clone())
                        .collect::<BTreeSet<_>>()
                }
                Decl::LetDef { value, span, .. } => {
                    *value = Expr::Tuple(vec![], *span);
                    BTreeSet::new()
                }
                Decl::Sig { type_binders, .. } => type_binders
                    .iter()
                    .map(|binder| binder.name.clone())
                    .collect(),
                Decl::TypeDef {
                    params, invariant, ..
                } => {
                    // Invariants are expressions; they are outside this API's
                    // declared-type evidence, like function and value bodies.
                    *invariant = None;
                    params.iter().cloned().collect()
                }
                Decl::TypeAlias { params, .. } => params.iter().cloned().collect(),
                Decl::Dim { .. } => BTreeSet::new(),
                Decl::Import { .. }
                | Decl::Export { .. }
                | Decl::Property { .. }
                | Decl::MacroDef { .. } => continue,
                Decl::Module { .. } => {
                    return Err("nested module in package declarations".to_string());
                }
            };
            // Type binders are local to their declaration. Value symbols do
            // not occupy the type namespace; in particular a function named
            // `p` cannot capture an implicit signature variable of that name.
            let keep = |name: &String, target: &String| {
                !bound.contains(name) && type_names.contains(target)
            };
            let scoped = NameResolver {
                own_names: resolver
                    .own_names
                    .to_sorted()
                    .into_iter()
                    .filter(|(name, target)| keep(name, target))
                    .map(|(name, target)| (name.clone(), target.clone()))
                    .collect(),
                imported_names: resolver
                    .imported_names
                    .to_sorted()
                    .into_iter()
                    .filter(|(name, target)| keep(name, target))
                    .map(|(name, target)| (name.clone(), target.clone()))
                    .collect(),
                qualified_modules: resolver
                    .qualified_modules
                    .to_sorted()
                    .into_iter()
                    .map(|(module, members)| {
                        (
                            module.clone(),
                            members
                                .to_sorted()
                                .into_iter()
                                .filter(|(_, target)| type_names.contains(*target))
                                .map(|(name, target)| (name.clone(), target.clone()))
                                .collect(),
                        )
                    })
                    .collect(),
                qualified_failures: RefCell::new(Vec::new()),
            };
            declarations.push(rewrite_decl(&decl, &scoped, package, &module.module_name));
            drain_qualified_failures(&scoped)?;
        }
        result.push(LinkedDeclarations {
            source_label: module.file_rel.to_string_lossy().into_owned(),
            module_name: module.module_name.clone(),
            names,
            exports: module.exports.clone(),
            declarations,
        });
    }
    Ok(result)
}
