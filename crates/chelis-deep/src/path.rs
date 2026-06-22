//! Path addressing and by-name function resolution over the Deep AST.
//!
//! This module locates a node inside a Deep program by address and by
//! module-qualified function name, and rewrites a resolved function's
//! body in place while preserving everything else verbatim.
//!
//! ## Node shapes
//!
//! A module renders to `(module {} <dotted.lower.name> decls...)`: the
//! `module` tag is at `elements[0]`, its metadata map at `elements[1]`,
//! the flattened lowercase dotted module name at `elements[2]`, and the
//! top-level declarations follow at `elements[3..]`. A function
//! definition renders to `(def {meta} <bare-name> (fn {} (params {} ...)
//! BODY))`, so the function body slot is `def.elements[3].elements[3]`:
//! `elements[3]` is the `(fn ...)` node and the body is that node's
//! `elements[3]` (after the `fn` tag, its metadata map, and the
//! `(params {} ...)` node). The def metadata at `elements[1]` carries
//! producer annotations such as `chelis_role` on property defs and must
//! survive a body rewrite untouched.
//!
//! A top-level `name: T = expr` value binding renders to a bare
//! `(def {} name expr)` value node with no `(fn ...)` child. That is not
//! a function-with-body, and resolving it as a function is a structured
//! error rather than an out-of-bounds index.

use crate::ast::{Atom, Expr, List};

/// The element index of the function body inside a `(fn {} (params {} ...)
/// BODY)` node: the `fn` tag is at 0, its metadata map at 1, the
/// `(params {} ...)` node at 2, and the body at 3.
const FN_BODY_INDEX: usize = 3;

/// The element index of the `(fn ...)` node inside a `(def {meta} <name>
/// (fn ...))` function definition: the `def` tag is at 0, its metadata
/// map at 1, the bare function name at 2, and the `(fn ...)` node at 3.
const DEF_FN_INDEX: usize = 3;

/// The element index of the module name symbol inside a `(module {}
/// <name> decls...)` node: the `module` tag is at 0, its metadata map at
/// 1, and the flattened lowercase dotted name at 2.
const MODULE_NAME_INDEX: usize = 2;

/// The element index at which a module's top-level declarations begin,
/// after the `module` tag, its metadata map, and the module name.
const MODULE_DECLS_START: usize = 3;

/// A single step in a [`DeepPath`].
///
/// The variants name the addressing forms a Deep path can take. Today
/// only [`PathSegment::Body`] and [`PathSegment::Child`] are exercised
/// (function-body resolution and indexing within it), but the set is
/// public so an L2 error path can address `body[2].args[0]`-style
/// locations without a breaking change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSegment {
    /// Enter the body slot of a function definition. From a `(def {meta}
    /// <name> (fn {} (params {} ...) BODY))` node this addresses `BODY`.
    Body,
    /// Index into the children of the current list node, where child 0 is
    /// the first element after the tag and metadata map (i.e. the
    /// `elements[2..]` slice). For `(app {} f a b)`, `Child(0)` addresses
    /// `f`, `Child(1)` addresses `a`, and so on.
    Child(usize),
}

/// An address of a node location within a function definition.
///
/// A path is resolved relative to a `(def ...)` node. The first segment
/// is normally [`PathSegment::Body`] to step into the function body;
/// later [`PathSegment::Child`] segments index into the body subtree.
/// The empty path addresses the def node itself.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeepPath {
    segments: Vec<PathSegment>,
}

impl DeepPath {
    /// An empty path addressing the def node itself.
    pub fn new() -> Self {
        Self::default()
    }

    /// A path that addresses the body of a function definition.
    pub fn body() -> Self {
        Self {
            segments: vec![PathSegment::Body],
        }
    }

    /// Append a [`PathSegment`], returning the extended path.
    pub fn then(mut self, segment: PathSegment) -> Self {
        self.segments.push(segment);
        self
    }

    /// Append a [`PathSegment::Child`] index, returning the extended path.
    pub fn child(self, index: usize) -> Self {
        self.then(PathSegment::Child(index))
    }

    /// The ordered segments of this path.
    pub fn segments(&self) -> &[PathSegment] {
        &self.segments
    }

    /// Resolve this path to a shared reference, starting at a `(def ...)`
    /// node.
    pub fn resolve<'a>(&self, def: &'a Expr) -> Result<&'a Expr, PathError> {
        let mut current = def;
        for (depth, segment) in self.segments.iter().enumerate() {
            current = step(current, segment, depth)?;
        }
        Ok(current)
    }

    /// Resolve this path to a mutable reference, starting at a `(def ...)`
    /// node.
    pub fn resolve_mut<'a>(&self, def: &'a mut Expr) -> Result<&'a mut Expr, PathError> {
        let mut current = def;
        for (depth, segment) in self.segments.iter().enumerate() {
            current = step_mut(current, segment, depth)?;
        }
        Ok(current)
    }
}

fn step<'a>(node: &'a Expr, segment: &PathSegment, depth: usize) -> Result<&'a Expr, PathError> {
    match segment {
        PathSegment::Body => {
            // Step from a `(def {meta} <name> (fn {} (params {} ...)
            // BODY))` node directly to BODY, descending through the `(fn
            // ...)` wrapper. The def must carry a `(fn ...)` child;
            // otherwise it is a value binding, not a function.
            if as_tagged_list(node, "def").is_none() {
                return Err(PathError::NotAtDef { depth });
            }
            function_body(node).ok_or(PathError::BodyNeedsFnAddressing { depth })
        }
        PathSegment::Child(index) => {
            let list = expect_list(node, depth)?;
            let element_index = 2 + index;
            list.elements
                .get(element_index)
                .ok_or(PathError::OutOfBounds {
                    depth,
                    index: element_index,
                })
        }
    }
}

fn step_mut<'a>(
    node: &'a mut Expr,
    segment: &PathSegment,
    depth: usize,
) -> Result<&'a mut Expr, PathError> {
    match segment {
        PathSegment::Body => {
            if as_tagged_list(node, "def").is_none() {
                return Err(PathError::NotAtDef { depth });
            }
            function_body_mut(node).ok_or(PathError::BodyNeedsFnAddressing { depth })
        }
        PathSegment::Child(index) => {
            let list = expect_list_mut(node, depth)?;
            let element_index = 2 + index;
            list.elements
                .get_mut(element_index)
                .ok_or(PathError::OutOfBounds {
                    depth,
                    index: element_index,
                })
        }
    }
}

fn expect_list(node: &Expr, depth: usize) -> Result<&List, PathError> {
    match node {
        Expr::List(list, _) => Ok(list),
        _ => Err(PathError::NotAList { depth }),
    }
}

fn expect_list_mut(node: &mut Expr, depth: usize) -> Result<&mut List, PathError> {
    match node {
        Expr::List(list, _) => Ok(list),
        _ => Err(PathError::NotAList { depth }),
    }
}

/// An error resolving a [`DeepPath`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PathError {
    /// A segment expected a list node but found an atom, map, or
    /// meta-expression at the given path depth.
    #[error("path step {depth}: expected a list node")]
    NotAList { depth: usize },

    /// A `Body` segment was applied to a node that is not a `(def ...)`
    /// node at the given path depth.
    #[error("path step {depth}: body addressing requires a def node")]
    NotAtDef { depth: usize },

    /// A `Body` segment addressed a `(def ...)` node that has no `(fn
    /// ...)` body (a value binding rather than a function definition), so
    /// there is no body slot to step into.
    #[error("path step {depth}: def has no function body to address")]
    BodyNeedsFnAddressing { depth: usize },

    /// A computed element index was out of bounds at the given path depth.
    #[error("path step {depth}: child index {index} is out of bounds")]
    OutOfBounds { depth: usize, index: usize },
}

/// A located function definition within a module.
#[derive(Debug, Clone)]
pub struct ResolvedFunction {
    /// Index of the `(def ...)` node among the module's top-level
    /// declarations (`module.elements[3..]`), i.e. relative to the first
    /// declaration after the module name.
    pub decl_index: usize,
    /// The fully-qualified Deep name of the function: the lowercase
    /// dotted module prefix joined to the bare function name with a dot
    /// (e.g. `economoist.growth.gordon_pv`).
    pub qualified_name: String,
    /// A path addressing the function body relative to the `(def ...)`
    /// node (a single [`PathSegment::Body`]).
    pub body_path: DeepPath,
}

/// An error resolving a function by module-qualified name.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResolveError {
    /// No top-level `(module {} <name> ...)` node was found in the
    /// program slice.
    #[error("no module declaration found while resolving `{searched}`")]
    NoModule { searched: String },

    /// The requested module prefix did not match the module's own
    /// flattened lowercase dotted name.
    #[error("module `{requested}` not found (module is `{actual}`) while resolving `{searched}`")]
    ModuleNotFound {
        searched: String,
        requested: String,
        actual: String,
    },

    /// No function definition with the requested bare name was found in
    /// the module.
    #[error("function `{name}` not found while resolving `{searched}`")]
    FunctionNotFound { searched: String, name: String },

    /// More than one definition with the requested bare name was found in
    /// the module.
    #[error("function `{name}` is ambiguous ({count} definitions) while resolving `{searched}`")]
    Ambiguous {
        searched: String,
        name: String,
        count: usize,
    },

    /// A definition with the requested name exists but is a value binding
    /// (`(def {} name expr)`), not a function definition with a `(fn ...)`
    /// body.
    #[error("`{name}` is a value binding, not a function, while resolving `{searched}`")]
    NotAFunction { searched: String, name: String },
}

/// Find a top-level function definition by a module-qualified name and
/// return its location and body slot.
///
/// `qualified_name` may be:
/// - a Deep-style lowercase dotted name (`economoist.growth.gordon_pv`),
/// - a Surf-style mixed-case dotted name
///   (`Economoist.Growth.gordon_pv`), which is normalized to the Deep
///   lowercase convention before matching, or
/// - a bare function name (`gordon_pv`), matched within the single module.
///
/// Resolution requires the program slice to contain exactly one
/// `(module {} <name> ...)` node. A name with a module prefix must match
/// that module's flattened lowercase dotted name; otherwise a
/// [`ResolveError::ModuleNotFound`] is returned.
pub fn resolve_function(
    module_exprs: &[Expr],
    qualified_name: &str,
) -> Result<ResolvedFunction, ResolveError> {
    let module = find_module(module_exprs).ok_or_else(|| ResolveError::NoModule {
        searched: qualified_name.to_string(),
    })?;
    let module_name = module_name(module).unwrap_or_default();

    let (requested_prefix, bare_name) = split_qualified_name(qualified_name);

    if let Some(prefix) = requested_prefix {
        let normalized = normalize_module_prefix(prefix);
        if normalized != module_name {
            return Err(ResolveError::ModuleNotFound {
                searched: qualified_name.to_string(),
                requested: normalized,
                actual: module_name,
            });
        }
    }

    let mut matches: Vec<(usize, &List)> = Vec::new();
    for (decl_index, decl) in decls(module).iter().enumerate() {
        if let Some(def) = as_tagged_list(decl, "def")
            && def_name(def) == Some(bare_name)
        {
            matches.push((decl_index, def));
        }
    }

    match matches.as_slice() {
        [] => Err(ResolveError::FunctionNotFound {
            searched: qualified_name.to_string(),
            name: bare_name.to_string(),
        }),
        [(decl_index, def)] => {
            if !def_is_function(def) {
                return Err(ResolveError::NotAFunction {
                    searched: qualified_name.to_string(),
                    name: bare_name.to_string(),
                });
            }
            let qualified = if module_name.is_empty() {
                bare_name.to_string()
            } else {
                format!("{module_name}.{bare_name}")
            };
            Ok(ResolvedFunction {
                decl_index: *decl_index,
                qualified_name: qualified,
                body_path: DeepPath::body(),
            })
        }
        many => Err(ResolveError::Ambiguous {
            searched: qualified_name.to_string(),
            name: bare_name.to_string(),
            count: many.len(),
        }),
    }
}

/// Replace only the body subtree of the named function with `new_body`,
/// preserving the def metadata, the bare name, the fn metadata, and the
/// params verbatim. Returns the full rewritten module program.
///
/// Everything outside `def.elements[{DEF_FN_INDEX}].elements[{FN_BODY_INDEX}]`
/// of the resolved function is left byte-identical under canonical
/// printing; only the function body changes.
pub fn splice_function_body(
    module_exprs: &[Expr],
    qualified_name: &str,
    new_body: Expr,
) -> Result<Vec<Expr>, ResolveError> {
    let resolved = resolve_function(module_exprs, qualified_name)?;
    let mut program = module_exprs.to_vec();

    let module = program
        .iter_mut()
        .find_map(|expr| as_tagged_list_mut(expr, "module"))
        .ok_or_else(|| ResolveError::NoModule {
            searched: qualified_name.to_string(),
        })?;

    // Declarations live at module.elements[MODULE_DECLS_START..]; the
    // resolved decl_index is relative to that slice.
    let def = module
        .elements
        .get_mut(MODULE_DECLS_START + resolved.decl_index)
        .expect("resolve_function returned an in-range decl index");

    let body_slot = function_body_mut(def).expect("resolved function has a body slot");
    *body_slot = new_body;

    Ok(program)
}

/// Borrow the body subtree of a function `(def ...)` node, if present.
pub fn function_body(def: &Expr) -> Option<&Expr> {
    let Expr::List(def_list, _) = def else {
        return None;
    };
    let Some(Expr::List(fn_list, _)) = def_list.elements.get(DEF_FN_INDEX) else {
        return None;
    };
    if tag(fn_list) != Some("fn") {
        return None;
    }
    fn_list.elements.get(FN_BODY_INDEX)
}

/// Mutably borrow the body subtree of a function `(def ...)` node, if
/// present.
pub fn function_body_mut(def: &mut Expr) -> Option<&mut Expr> {
    let Expr::List(def_list, _) = def else {
        return None;
    };
    let Some(Expr::List(fn_list, _)) = def_list.elements.get_mut(DEF_FN_INDEX) else {
        return None;
    };
    if tag(fn_list) != Some("fn") {
        return None;
    }
    fn_list.elements.get_mut(FN_BODY_INDEX)
}

// ── Local AST helpers over the public chelis-deep AST ────────────────

fn tag(list: &List) -> Option<&str> {
    list.elements.first().and_then(|e| match e {
        Expr::Atom(Atom::Symbol(s), _) => Some(s.as_str()),
        _ => None,
    })
}

fn as_tagged_list<'a>(expr: &'a Expr, expected_tag: &str) -> Option<&'a List> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    (tag(list) == Some(expected_tag)).then_some(list)
}

fn as_tagged_list_mut<'a>(expr: &'a mut Expr, expected_tag: &str) -> Option<&'a mut List> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    (tag(list) == Some(expected_tag)).then_some(list)
}

fn find_module(exprs: &[Expr]) -> Option<&Expr> {
    exprs.iter().find(|e| as_tagged_list(e, "module").is_some())
}

/// The flattened lowercase dotted module name. In canonical form
/// `(module {} <name> decls...)` the module tag is at `elements[0]`, the
/// metadata map at `elements[1]`, and the name symbol at `elements[2]`.
fn module_name(module: &Expr) -> Option<String> {
    let list = as_tagged_list(module, "module")?;
    match list.elements.get(MODULE_NAME_INDEX) {
        Some(Expr::Atom(Atom::Symbol(name), _)) => Some(name.clone()),
        _ => None,
    }
}

/// The module's top-level declarations (`module.elements[3..]`), after the
/// `module` tag, its metadata map, and the module name.
fn decls(module: &Expr) -> &[Expr] {
    match module {
        Expr::List(list, _) if list.elements.len() > MODULE_DECLS_START => {
            &list.elements[MODULE_DECLS_START..]
        }
        _ => &[],
    }
}

/// The bare name of a `(def {meta} <name> ...)` node at `elements[2]`.
fn def_name(def: &List) -> Option<&str> {
    match def.elements.get(2) {
        Some(Expr::Atom(Atom::Symbol(name), _)) => Some(name.as_str()),
        _ => None,
    }
}

/// True when a `(def ...)` node is a function definition: its child at
/// `elements[3]` is a `(fn ...)` node. A value binding `(def {} name
/// expr)` has a non-`fn` child (or no child at that index) and is not a
/// function.
fn def_is_function(def: &List) -> bool {
    matches!(
        def.elements.get(DEF_FN_INDEX),
        Some(Expr::List(fn_list, _)) if tag(fn_list) == Some("fn")
    )
}

/// Split a qualified name into an optional module prefix and the bare
/// function name. `economoist.growth.gordon_pv` splits into
/// (`economoist.growth`, `gordon_pv`); a bare `gordon_pv` splits into
/// (`None`, `gordon_pv`).
fn split_qualified_name(qualified: &str) -> (Option<&str>, &str) {
    match qualified.rsplit_once('.') {
        Some((prefix, bare)) => (Some(prefix), bare),
        None => (None, qualified),
    }
}

/// Normalize a module prefix to the Deep convention: a flattened
/// lowercase dotted symbol. A Surf-style `Economoist.Growth` and a
/// Deep-style `economoist.growth` both normalize to `economoist.growth`.
fn normalize_module_prefix(prefix: &str) -> String {
    prefix.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str;

    fn parse_one(src: &str) -> Expr {
        parse_str(src)
            .expect("parse failed")
            .into_iter()
            .next()
            .expect("at least one expr")
    }

    #[test]
    fn split_qualified_name_with_and_without_prefix() {
        assert_eq!(
            split_qualified_name("economoist.growth.gordon_pv"),
            (Some("economoist.growth"), "gordon_pv")
        );
        assert_eq!(split_qualified_name("gordon_pv"), (None, "gordon_pv"));
    }

    #[test]
    fn normalize_prefix_lowercases_pascalcase() {
        assert_eq!(
            normalize_module_prefix("Economoist.Growth"),
            "economoist.growth"
        );
        // An already-lowercase prefix is unchanged.
        assert_eq!(
            normalize_module_prefix("economoist.growth"),
            "economoist.growth"
        );
    }

    #[test]
    fn child_out_of_bounds_is_structured_error() {
        // `(app {} f)` has one child at element 2; Child(5) addresses
        // element 7, which is out of bounds.
        let app = parse_one("(app {} (var {} f))");
        let path = DeepPath::new().child(5);
        let err = path.resolve(&app).unwrap_err();
        assert_eq!(err, PathError::OutOfBounds { depth: 0, index: 7 });
    }

    #[test]
    fn child_segment_on_atom_is_not_a_list() {
        let atom = parse_one("42");
        let path = DeepPath::new().child(0);
        let err = path.resolve(&atom).unwrap_err();
        assert_eq!(err, PathError::NotAList { depth: 0 });
    }

    #[test]
    fn body_segment_on_non_def_is_not_at_def() {
        let app = parse_one("(app {} (var {} f))");
        let err = DeepPath::body().resolve(&app).unwrap_err();
        assert_eq!(err, PathError::NotAtDef { depth: 0 });
    }

    #[test]
    fn function_body_returns_none_for_value_binding() {
        // A value binding `(def {} pi (lit ...))` has a non-fn child.
        let value_def = parse_one("(def {} pi (lit {type: (t-prim {} f32)} 3.14))");
        assert!(function_body(&value_def).is_none());
    }

    #[test]
    fn function_body_returns_body_for_function_def() {
        let fn_def =
            parse_one("(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))");
        let body = function_body(&fn_def).expect("function body present");
        assert!(matches!(body, Expr::List(list, _) if tag(list) == Some("var")));
    }
}
