//! C6 stdlib discovery: compiler-linked names and resolved declared types.

use super::*;
use chelis_types::types::{NominalArg, TensorPrec, Type, TypeVar};
use chelis_types::{DeclaredSignature, DeclaredTypeSurface};
use std::collections::VecDeque;

/// A finite transfer function: concrete numeric domains plus formal payload
/// positions. Substituting summaries, rather than expanding Type trees, also
/// terminates for `Tree[a] -> Tree[List[a]]` and recursive aliases.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Capacity {
    prims: BTreeSet<String>,
    params: BTreeSet<usize>,
}

impl Capacity {
    fn extend(&mut self, other: Self) {
        self.prims.extend(other.prims);
        self.params.extend(other.params);
    }
}

struct Declaration {
    params: BTreeMap<TypeVar, usize>,
    names: BTreeMap<TypeVar, String>,
    bodies: Vec<Type>,
    nominal: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Summary {
    reachable: Capacity,
    bare: Capacity,
}

struct Closure {
    summaries: BTreeMap<String, Summary>,
}

impl Closure {
    fn new(surface: &DeclaredTypeSurface) -> Self {
        fn params(
            args: &[NominalArg],
            names: &[String],
        ) -> (BTreeMap<TypeVar, usize>, BTreeMap<TypeVar, String>) {
            let mut positions = BTreeMap::new();
            let mut binders = BTreeMap::new();
            for (index, argument) in args.iter().enumerate() {
                if let NominalArg::Type(Type::Var(var)) = argument {
                    positions.insert(*var, index);
                    binders.insert(*var, names[index].clone());
                }
            }
            (positions, binders)
        }
        let mut declarations = BTreeMap::new();
        for (name, def) in &surface.registry().defs {
            let (params, names) = params(&def.param_args, &def.type_params);
            declarations.insert(
                name.clone(),
                Declaration {
                    params,
                    names,
                    bodies: def
                        .variants
                        .iter()
                        .flat_map(|variant| variant.fields.iter().map(|(_, ty)| ty.clone()))
                        .collect(),
                    nominal: true,
                },
            );
        }
        for (name, alias) in &surface.registry().aliases {
            let (params, names) = params(&alias.param_args, &alias.params);
            declarations.insert(
                name.clone(),
                Declaration {
                    params,
                    names,
                    bodies: vec![alias.body.clone()],
                    nominal: false,
                },
            );
        }
        let mut closure = Self {
            summaries: declarations
                .keys()
                .map(|name| (name.clone(), Summary::default()))
                .collect(),
        };
        for &(name, arity) in surface.native_nominals() {
            closure
                .summaries
                .entry(name.to_string())
                .or_insert_with(|| Summary {
                    reachable: Capacity {
                        prims: BTreeSet::new(),
                        params: (0..arity).collect(),
                    },
                    bare: Capacity::default(),
                });
        }
        // Every changed transfer function schedules its consumers. The finite
        // sets only grow: no recursion depth limit and no cycle-as-empty exit.
        let mut consumers: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        fn dependencies(ty: &Type, out: &mut BTreeSet<String>) {
            match ty {
                Type::Adt(name, args) => {
                    out.insert(name.clone());
                    for arg in args {
                        dependencies(arg, out);
                    }
                }
                Type::KindedAdt(name, args) => {
                    out.insert(name.clone());
                    for arg in args {
                        if let NominalArg::Type(ty) = arg {
                            dependencies(ty, out);
                        }
                    }
                }
                Type::Fn(args, ret) => {
                    for arg in args {
                        dependencies(arg, out);
                    }
                    dependencies(ret, out);
                }
                Type::Tuple(parts) => {
                    for part in parts {
                        dependencies(part, out);
                    }
                }
                Type::Ref(inner) => dependencies(inner, out),
                Type::Error(_) => panic!("unresolved Type::Error in stdlib declaration graph"),
                Type::Prim(_) | Type::Tensor(_, _) | Type::Var(_) | Type::Unit => {}
            }
        }
        for (name, declaration) in &declarations {
            let mut referenced = BTreeSet::new();
            for body in &declaration.bodies {
                dependencies(body, &mut referenced);
            }
            for dependency in referenced {
                assert!(
                    closure.summaries.contains_key(&dependency),
                    "unknown nominal `{dependency}` in stdlib closure"
                );
                consumers
                    .entry(dependency)
                    .or_default()
                    .insert(name.clone());
            }
        }
        let mut pending: VecDeque<_> = declarations.keys().cloned().collect();
        let mut queued: BTreeSet<_> = declarations.keys().cloned().collect();
        while let Some(name) = pending.pop_front() {
            queued.remove(&name);
            let declaration = &declarations[&name];
            let mut summary = Summary::default();
            for body in &declaration.bodies {
                summary.reachable.extend(closure.capacity(
                    body,
                    false,
                    &declaration.params,
                    &declaration.names,
                ));
                if !declaration.nominal {
                    summary.bare.extend(closure.capacity(
                        body,
                        true,
                        &declaration.params,
                        &declaration.names,
                    ));
                }
            }
            if summary != closure.summaries[&name] {
                closure.summaries.insert(name.clone(), summary);
                if let Some(users) = consumers.get(&name) {
                    for user in users {
                        if queued.insert(user.clone()) {
                            pending.push_back(user.clone());
                        }
                    }
                }
            }
        }
        closure
    }

    fn capacity(
        &self,
        ty: &Type,
        bare: bool,
        params: &BTreeMap<TypeVar, usize>,
        names: &BTreeMap<TypeVar, String>,
    ) -> Capacity {
        let mut result = Capacity::default();
        match ty {
            Type::Prim(prim) | Type::Tensor(_, TensorPrec::Concrete(prim)) => {
                if !matches!(prim_census_class(*prim), PrimCensusClass::NonNumeric) {
                    result.prims.insert(prim.name().to_string());
                }
            }
            Type::Var(var) | Type::Tensor(_, TensorPrec::Var(var)) => {
                if let Some(index) = params.get(var) {
                    result.params.insert(*index);
                }
                if !bare {
                    if matches!(ty, Type::Tensor(..)) {
                        result.prims.insert("tensor-precision".to_string());
                    }
                    if let Some(name) = names.get(var)
                        && matches!(name.as_str(), "p_float" | "p_int" | "p_numeric" | "q" | "Q")
                    {
                        result.prims.insert(name.clone());
                    }
                }
            }
            Type::Fn(args, ret) => {
                for arg in args {
                    result.extend(self.capacity(arg, bare, params, names));
                }
                result.extend(self.capacity(ret, bare, params, names));
            }
            Type::Tuple(parts) => {
                for part in parts {
                    result.extend(self.capacity(part, bare, params, names));
                }
            }
            Type::Ref(inner) => result.extend(self.capacity(inner, bare, params, names)),
            Type::Adt(name, args) => {
                result.extend(self.application(
                    name,
                    &args.iter().map(Some).collect::<Vec<_>>(),
                    bare,
                    params,
                    names,
                ));
            }
            Type::KindedAdt(name, args) => {
                result.extend(self.application(
                    name,
                    &args.iter().map(NominalArg::as_type).collect::<Vec<_>>(),
                    bare,
                    params,
                    names,
                ));
            }
            Type::Error(_) => panic!("unresolved Type::Error in stdlib declaration graph"),
            Type::Unit => {}
        }
        result
    }

    fn application(
        &self,
        name: &str,
        args: &[Option<&Type>],
        bare: bool,
        params: &BTreeMap<TypeVar, usize>,
        names: &BTreeMap<TypeVar, String>,
    ) -> Capacity {
        let summary = self
            .summaries
            .get(name)
            .unwrap_or_else(|| panic!("unknown nominal `{name}` in stdlib closure"));
        let transfer = if bare {
            &summary.bare
        } else {
            &summary.reachable
        };
        let mut result = Capacity {
            prims: transfer.prims.clone(),
            params: BTreeSet::new(),
        };
        // Resolve every argument even for a phantom parameter: an unused
        // argument does not authorize an unresolved name or error sentinel.
        let actuals = args
            .iter()
            .map(|arg| arg.map(|ty| self.capacity(ty, bare, params, names)))
            .collect::<Vec<_>>();
        for &index in &transfer.params {
            result.extend(
                actuals
                    .get(index)
                    .and_then(Clone::clone)
                    .unwrap_or_else(|| {
                        panic!("invalid type argument {index} for `{name}` in stdlib closure")
                    }),
            );
        }
        result
    }

    fn signature(&self, signature: &DeclaredSignature, bare: bool) -> Capacity {
        let mut result =
            self.capacity(&signature.ty, bare, &BTreeMap::new(), &signature.type_names);
        if !bare && !signature.bounds.is_empty() {
            result.prims.insert("dtype-bound".to_string());
        }
        result
    }
}

fn symbol(expr: &Expr) -> Option<&str> {
    if let Expr::Atom(Atom::Name(name), _) = expr {
        Some(name)
    } else {
        None
    }
}

fn declarations(exprs: &[Expr]) -> Vec<List> {
    let mut result = Vec::new();
    for expr in exprs {
        let list = match expr {
            Expr::List(list, _) => list.clone(),
            Expr::Node(node, span) => node.to_list(*span),
            _ => panic!("unresolved stdlib declaration shape"),
        };
        if list.tag() == Some(DeepTag::Module) {
            result.extend(declarations(&list.elements[3..]));
        } else {
            result.push(list);
        }
    }
    result
}

fn rows_for_source(
    exprs: &[Expr],
    label: &str,
    names: Option<&BTreeMap<String, String>>,
    exports: Option<&BTreeSet<String>>,
    surface: &DeclaredTypeSurface,
    closure: &Closure,
) -> Vec<Row> {
    let declarations = declarations(exprs);
    let mut signatures = BTreeMap::new();
    let mut values = BTreeSet::new();
    let mut source_exports = BTreeSet::new();
    let mut rows = Vec::new();
    let internal = |name: &str| match names {
        Some(names) => names
            .get(name)
            .expect("linked declaration has an exact identity")
            .clone(),
        None => name.to_string(),
    };
    for declaration in &declarations {
        match declaration.tag() {
            Some(DeepTag::Export) => {
                source_exports.extend(
                    declaration.elements[2..]
                        .iter()
                        .filter_map(symbol)
                        .map(str::to_string),
                );
            }
            Some(DeepTag::Def) => {
                values.insert(symbol(&declaration.elements[2]).expect("def name"));
            }
            Some(DeepTag::Defsig) => {
                signatures.insert(
                    symbol(&declaration.elements[2]).expect("defsig name"),
                    declaration,
                );
            }
            Some(DeepTag::Deftype) => {
                let name = symbol(&declaration.elements[2]).expect("deftype name");
                let summary = closure
                    .summaries
                    .get(&internal(name))
                    .expect("resolved ADT");
                if !summary.reachable.prims.is_empty() {
                    let shape = declaration.elements[3..]
                        .iter()
                        .map(chelis_deep::printer::print_expr_flat)
                        .collect::<Vec<_>>()
                        .join(" ");
                    rows.push(Row {
                        kind: "std-adt-numeric".to_string(),
                        id: format!("{label}::{name}: {shape}"),
                        flags: numeric_carrier_flags(&summary.reachable.prims),
                        citation: String::new(),
                    });
                }
            }
            _ => {}
        }
    }
    for name in exports.unwrap_or(&source_exports) {
        let Some(declaration) = signatures.get(name.as_str()) else {
            assert!(
                !values.contains(name.as_str()),
                "EXPORTED DEFINITION WITHOUT A DECLARED SIGNATURE `{label}::{name}`: declare the signature or stop exporting the binding"
            );
            continue;
        };
        let signature = &surface.signatures()[&internal(name)];
        if closure.signature(signature, false).prims.is_empty() {
            continue;
        }
        let bounds = if let Expr::Map(meta, _) = &declaration.elements[1] {
            chelis_deep::decode_dtype_bounds(meta)
        } else {
            vec![]
        };
        let prefix = if bounds.is_empty() {
            String::new()
        } else {
            format!(
                "[{}] ",
                bounds
                    .iter()
                    .map(|(binder, family)| format!("{binder}: {}", family.surf_name()))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        rows.push(Row {
            kind: "std-def-numeric".to_string(),
            id: format!(
                "{label}::{name}: {prefix}{}",
                chelis_deep::printer::print_expr_flat(&declaration.elements[3])
            ),
            flags: numeric_carrier_flags(&closure.signature(signature, true).prims),
            citation: String::new(),
        });
    }
    rows
}

fn resolve(exprs: &[Expr]) -> DeclaredTypeSurface {
    chelis_types::resolve_declared_surface(exprs)
        .unwrap_or_else(|error| panic!("stdlib declaration resolution failed: {:?}", error.errors))
}

pub(super) fn scan_deftypes(exprs: &[Expr], label: &str, rows: &mut Vec<Row>) {
    let surface = resolve(exprs);
    let closure = Closure::new(&surface);
    rows.extend(rows_for_source(
        exprs, label, None, None, &surface, &closure,
    ));
}

pub(super) fn stdlib_rows(root: &Path) -> Vec<Row> {
    let src_dir = root.join(STD_SRC_REL);
    let mut files = Vec::new();
    walk_ch_files(&src_dir, &mut files);
    files.sort();
    let sources = files
        .iter()
        .map(|path| {
            let source = fs::read_to_string(path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            let decls = chelis_surf::parser::parse_str(&source).unwrap_or_else(|error| {
                panic!(
                    "stdlib source must parse for the capacity census: {}: {error:?}",
                    path.display()
                )
            });
            let label = path
                .strip_prefix(&src_dir)
                .expect("under src dir")
                .with_extension("")
                .to_string_lossy()
                .replace('\\', "/");
            (label, decls)
        })
        .collect::<Vec<_>>();
    let linked = chelis_reef::declared_surface::link_package_declarations("chelis-std", &sources)
        .unwrap_or_else(|error| panic!("stdlib declaration linking failed: {error}"));
    let exprs = linked
        .iter()
        .flat_map(|module| chelis_surf::desugar::desugar_program(&module.declarations))
        .collect::<Vec<_>>();
    let surface = resolve(&exprs);
    let closure = Closure::new(&surface);
    let originals = sources.into_iter().collect::<BTreeMap<_, _>>();
    linked
        .iter()
        .flat_map(|module| {
            rows_for_source(
                &chelis_surf::desugar::desugar_program(&originals[&module.source_label]),
                &module.source_label,
                Some(&module.names),
                Some(&module.exports),
                &surface,
                &closure,
            )
        })
        .collect()
}
