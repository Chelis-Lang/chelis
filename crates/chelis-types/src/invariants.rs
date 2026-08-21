//! Declaration-time well-formedness for opaque-type invariants
//! (RFC `opaque_invariants_rfc.md` D-WF).
//!
//! This pass runs over raw Deep, so it covers both `.ch` (post-desugar)
//! and hand-written `.dp`. It is wired into the IR check pipeline next to
//! `validate_tensor_precisions_in_program`. It records nothing and
//! evaluates nothing: the invariant is invisible to type checking (RFC
//! feature statement). It only checks that a declared invariant is
//! WELL-FORMED, in five ways:
//!
//! 1. `invariant` present requires `opaque: true`.
//! 2. The representation is exactly one record-shaped variant, every
//!    field in the V1 value class (scalar prims, fixed-shape numeric
//!    tensors, or nested single-variant records of those).
//! 3. The predicate is in the declaration grammar (chelis-pred), its
//!    free variables are a subset of {binder} union {in-module zero-arg
//!    constant defs}, and it is boolean-shaped at the top.
//! 4. The recorded `invariant_amenability` equals a recomputation by
//!    `chelis_pred::classify_predicate` (protects hand-written `.dp`).
//!
//! The `==`-over-float advisory (D-WF / D-STARVE) is NOT here: the
//! checker has no non-blocking warning channel (deep-validate warnings
//! become errors in the check path), so it is emitted through the lint
//! advisory channel instead (see `chelis-lint`
//! `invariant-float-equality`).
//!
//! CRITICAL (D-WF + feature statement): a program whose invariant is
//! VIOLATED by an in-module constructor still type-checks. The invariant
//! is not refinement typing; the checker records it, never evaluates it.

use chelis_deep::DeepTag;
use std::collections::{HashMap, HashSet};

use chelis_deep::{Atom, Expr};
use chelis_pred::{PredAmenability, PredGrammarError};

use crate::errors::{CheckError, CheckErrorKind};
use crate::session::DiagnosticSink;
use crate::types::Prim;

trait DiagnosticOutput {
    fn push_error(&mut self, error: CheckError);
}

impl DiagnosticOutput for Vec<CheckError> {
    fn push_error(&mut self, error: CheckError) {
        self.push(error);
    }
}

impl DiagnosticOutput for DiagnosticSink<'_> {
    fn push_error(&mut self, error: CheckError) {
        self.push(error);
    }
}

/// Whether a primitive name is a SCALAR in the V1 invariant value class: the
/// numeric and boolean prims an invariant predicate can actually be verified
/// over (`f32`, `f64`, the signed integer widths, `bool`). `string`,
/// `f8e4m3`, and any other prim are NOT in the class -- the predicate grammar
/// is arithmetic/comparison over numeric values, so no verifiable invariant
/// can be expressed over them.
///
/// SHARED by the D-WF value-class check and the `chelis-prove` field model
/// (`chelis_prove::opaque::field_type_from_deep`) so the checker and the
/// prover agree on EXACTLY which representations carry a mechanically
/// verifiable invariant -- the divergence the two had let a representation
/// pass `chelis check` while `chelis prove` silently collected zero
/// obligations for it.
pub fn invariant_value_class_prim(name: &str) -> bool {
    matches!(name, "f32" | "f64" | "bool") || Prim::parse_name(name).is_some_and(|p| p.is_integer())
}

/// Validate every declared type invariant in a program (RFC D-WF).
/// Pushes one [`CheckError`] per well-formedness violation found.
pub fn validate_type_invariants_in_program(exprs: &[Expr], errors: &mut Vec<CheckError>) {
    validate_type_invariants(exprs, errors);
}

pub(crate) fn validate_type_invariants_in_program_with_sink(
    exprs: &[Expr],
    errors: &mut DiagnosticSink<'_>,
) {
    validate_type_invariants(exprs, errors);
}

fn validate_type_invariants(exprs: &[Expr], errors: &mut impl DiagnosticOutput) {
    let items = flatten_with_modules(exprs);

    // Collect, per module key, the names of zero-arg constant defs. A
    // predicate may freely reference these (RFC D-WF). A constant def is
    // a `(def {} name <value>)` whose body is a bare value or a `fn` with
    // an empty params node.
    let mut constants: HashMap<Option<String>, HashSet<String>> = HashMap::new();
    // Resolve nested `t-adt` field references against the program's own
    // `deftype` declarations: a field typed as a single-variant record of
    // value-class types is in the value class. Builtin ADTs (`List`,
    // `Option`, ...) are not in this map, so they reject (correct: they
    // are multi-variant, outside the value class).
    let mut deftypes: HashMap<String, &Expr> = HashMap::new();
    for (module, item) in &items {
        if let Some((name, body)) = as_def(item)
            && is_zero_arg_constant(body)
        {
            constants
                .entry(module.clone())
                .or_default()
                .insert(name.to_string());
        }
        if tag(item) == Some(DeepTag::Deftype)
            && let Some(name) = children(item).first().and_then(sym_str)
        {
            deftypes.insert(name.to_string(), item);
        }
    }

    for (module, item) in &items {
        if tag(item) != Some(DeepTag::Deftype) {
            continue;
        }
        let empty = HashSet::new();
        let in_module_constants = constants.get(module).unwrap_or(&empty);
        validate_one_deftype(item, in_module_constants, &deftypes, errors);
    }
}

/// Validate one `deftype` node carrying (or lacking) invariant metadata.
fn validate_one_deftype(
    deftype: &Expr,
    in_module_constants: &HashSet<String>,
    deftypes: &HashMap<String, &Expr>,
    errors: &mut impl DiagnosticOutput,
) {
    let invariant = meta_value(deftype, "invariant");
    let Some(invariant_fn) = invariant else {
        return; // No invariant declared: nothing to check.
    };
    let type_name = children(deftype).first().and_then(sym_str).unwrap_or("_");

    // Check 1: invariant requires opaque: true.
    if !has_true_meta(deftype, "opaque") {
        errors.push_error(err(format!(
            "invariant on type `{type_name}` requires `@opaque`: \
             assumption injection is unsound for a forgeable type"
        )));
        // Keep checking the remaining well-formedness anyway so the
        // author sees every problem in one pass.
    }

    // Check 2: representation value class.
    validate_representation(deftype, type_name, deftypes, errors);

    // Check 3 + 4: predicate grammar, free vars, boolean-shaped, and the
    // recorded amenability matches recomputation.
    validate_predicate(
        invariant_fn,
        deftype,
        type_name,
        in_module_constants,
        errors,
    );
}

// ===========================================================================
// Check 2: representation value class
// ===========================================================================

fn validate_representation(
    deftype: &Expr,
    type_name: &str,
    deftypes: &HashMap<String, &Expr>,
    errors: &mut impl DiagnosticOutput,
) {
    let variants: Vec<&Expr> = children(deftype)
        .iter()
        .skip(1) // skip the param-list node
        .filter(|c| tag(c) == Some(DeepTag::Variant))
        .collect();

    if variants.len() != 1 {
        errors.push_error(err(format!(
            "opaque type `{type_name}` with an invariant must have exactly one \
             record-shaped variant; found {} variants",
            variants.len()
        )));
        return;
    }
    let variant = variants[0];
    let fields: Vec<&Expr> = children(variant)
        .iter()
        .skip(1) // skip the variant name symbol
        .collect();
    if fields.is_empty() || !fields.iter().all(|f| tag(f) == Some(DeepTag::Field)) {
        errors.push_error(err(format!(
            "opaque type `{type_name}` with an invariant must have a single \
             record-shaped variant (named fields)"
        )));
        return;
    }
    for field in fields {
        let field_kids = children(field);
        let field_name = field_kids.first().and_then(sym_str).unwrap_or("_");
        let Some(field_ty) = field_kids.get(1) else {
            continue;
        };
        let mut visiting = HashSet::new();
        if !is_value_class_type(field_ty, deftypes, &mut visiting) {
            errors.push_error(err(format!(
                "field `{field_name}` of opaque type `{type_name}` is not in the V1 \
                 invariant value class (scalar primitives, fixed-shape numeric \
                 tensors, or nested single-variant records of those)"
            )));
        }
    }
}

/// Whether a Deep type expression is in the V1 invariant value class:
/// scalar primitives, fixed-shape numeric tensors (all dims literal), or
/// a `t-adt` reference to an in-program single-variant record whose
/// every field is itself in the value class. `visiting` breaks recursive
/// type cycles (a recursive ADT is not value-class anyway).
fn is_value_class_type(
    ty: &Expr,
    deftypes: &HashMap<String, &Expr>,
    visiting: &mut HashSet<String>,
) -> bool {
    match tag(ty) {
        // Only the numeric/boolean scalar prims an invariant can be verified
        // over (NOT `string`/`f8e4m3`/...) -- shared with the prover's field
        // model so the two agree on the value class.
        Some(DeepTag::TPrim) => children(ty)
            .first()
            .and_then(sym_str)
            .is_some_and(invariant_value_class_prim),
        Some(DeepTag::TTensor) => {
            // All dims must be literal (`d-lit`) and the element an `f32`/`f64`
            // prim: the prover only models `f32`/`f64` tensors, so a tensor of
            // any other element (e.g. an int tensor) is outside the verifiable
            // value class.
            let kids = children(ty);
            if kids.is_empty() {
                return false;
            }
            let (dims, element) = kids.split_at(kids.len() - 1);
            let element_is_float = tag(&element[0]) == Some(DeepTag::TPrim)
                && matches!(
                    children(&element[0]).first().and_then(sym_str),
                    Some("f32" | "f64")
                );
            dims.iter().all(|d| tag(d) == Some(DeepTag::DLit)) && element_is_float
        }
        Some(DeepTag::TAdt) => {
            // Resolve the referenced type against the program's deftypes.
            // Builtin / out-of-program ADTs (List, Option, ...) are not in
            // the map, so they reject. A single-variant record whose
            // fields are all value-class is admitted (RFC nested-record
            // allowance).
            let Some(name) = children(ty).first().and_then(sym_str) else {
                return false;
            };
            // A type argument means a generic instantiation (e.g.
            // List[f32]); even if resolvable, parameterized records are
            // out of the V1 value class.
            if children(ty).len() > 1 {
                return false;
            }
            if !visiting.insert(name.to_string()) {
                return false; // cycle: recursive ADT, not value-class
            }
            let admitted = deftypes
                .get(name)
                .map(|referenced| is_single_record_of_value_class(referenced, deftypes, visiting))
                .unwrap_or(false);
            visiting.remove(name);
            admitted
        }
        _ => false,
    }
}

/// Whether a referenced `deftype` is a single-variant record whose every
/// field is in the value class.
fn is_single_record_of_value_class(
    deftype: &Expr,
    deftypes: &HashMap<String, &Expr>,
    visiting: &mut HashSet<String>,
) -> bool {
    let variants: Vec<&Expr> = children(deftype)
        .iter()
        .skip(1)
        .filter(|c| tag(c) == Some(DeepTag::Variant))
        .collect();
    if variants.len() != 1 {
        return false;
    }
    let fields: Vec<&Expr> = children(variants[0]).iter().skip(1).collect();
    if fields.is_empty() || !fields.iter().all(|f| tag(f) == Some(DeepTag::Field)) {
        return false;
    }
    fields.iter().all(|field| {
        children(field)
            .get(1)
            .map(|ty| is_value_class_type(ty, deftypes, visiting))
            .unwrap_or(false)
    })
}

// ===========================================================================
// Check 3 + 4: predicate grammar, free vars, amenability
// ===========================================================================

fn validate_predicate(
    invariant_fn: &Expr,
    deftype: &Expr,
    type_name: &str,
    in_module_constants: &HashSet<String>,
    errors: &mut impl DiagnosticOutput,
) {
    // Grammar (D-WF).
    match chelis_pred::predicate_in_grammar(invariant_fn) {
        Ok(()) => {}
        Err(grammar_err) => {
            errors.push_error(err(format!(
                "invariant on type `{type_name}` is outside the predicate grammar: {}",
                describe_grammar_error(&grammar_err)
            )));
            // Without a valid grammar the remaining checks are not
            // meaningful; stop here for this type.
            return;
        }
    }

    // Free variables subset {binder} union {in-module constants}.
    for fv in chelis_pred::predicate_free_vars(invariant_fn) {
        if !in_module_constants.contains(&fv) {
            errors.push_error(err(format!(
                "invariant on type `{type_name}` references `{fv}`, which is not the \
                 binder or an in-module zero-argument constant"
            )));
        }
    }

    // Boolean-shaped at the top: the body's outermost operator must be a
    // comparison, a boolean connective, or an `if` whose branches are
    // boolean.
    if let Some(body) = fn_body(invariant_fn)
        && !is_boolean_shaped(body)
    {
        errors.push_error(err(format!(
            "invariant on type `{type_name}` must be a boolean predicate at the top"
        )));
    }

    // Recorded amenability equals recomputation (protects hand-written
    // `.dp`).
    let recomputed = chelis_pred::classify_predicate(invariant_fn);
    match meta_value(deftype, "invariant_amenability").and_then(str_value) {
        Some(recorded) => {
            if PredAmenability::from_str(recorded) != Some(recomputed) {
                errors.push_error(err(format!(
                    "invariant on type `{type_name}` records amenability `{recorded}`, \
                     but the predicate classifies as `{}`",
                    recomputed.as_str()
                )));
            }
        }
        None => {
            errors.push_error(err(format!(
                "invariant on type `{type_name}` is missing the \
                 `invariant_amenability` metadata key (expected `{}`)",
                recomputed.as_str()
            )));
        }
    }
}

/// Whether the top of a predicate body is boolean-shaped.
fn is_boolean_shaped(body: &Expr) -> bool {
    match tag(body) {
        Some(DeepTag::App) => {
            let kids = children(body);
            let callee = kids.first().and_then(var_name);
            matches!(
                callee,
                Some("eq" | "neq" | "cmplt" | "gt" | "lte" | "gte" | "and" | "or" | "not")
            )
        }
        Some(DeepTag::If) => {
            // The CONDITION and both branches must be boolean-shaped. Checking
            // only the branches let `if p.value then true else false` (a
            // non-boolean f32 condition) pass D-WF even though the type rule
            // requires the condition to be `bool` (spec/04 type rule for `if`).
            let kids = children(body);
            kids.len() == 3
                && is_boolean_shaped(&kids[0])
                && is_boolean_shaped(&kids[1])
                && is_boolean_shaped(&kids[2])
        }
        Some(DeepTag::Lit) => is_bool_lit(body),
        _ => false,
    }
}

fn is_bool_lit(expr: &Expr) -> bool {
    // (lit {type: (t-prim {} bool)} <bool>)
    let kids = children(expr);
    matches!(kids.first(), Some(Expr::Atom(Atom::Bool(_), _)))
}

fn describe_grammar_error(e: &PredGrammarError) -> String {
    match e {
        PredGrammarError::NotAPredicateFn(d) => {
            format!("not a well-formed predicate fn ({d})")
        }
        PredGrammarError::DisallowedCall(name) => {
            format!("call to `{name}` is not in the grammar")
        }
        PredGrammarError::DisallowedNode(d) => format!("{d} is not in the grammar"),
        PredGrammarError::BadSum => {
            "`sum` must be applied to a binder field projection".to_string()
        }
    }
}

// ===========================================================================
// Deep node helpers (local to this module to stay decoupled from infer.rs)
// ===========================================================================

fn tag(expr: &Expr) -> Option<DeepTag> {
    match expr {
        Expr::List(list, _) => list.tag(),
        _ => None,
    }
}

fn children(expr: &Expr) -> &[Expr] {
    if let Expr::List(list, _) = expr
        && list.elements.len() >= 2
    {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn sym_str(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(s), _) => Some(s.as_str()),
        _ => None,
    }
}

fn var_name(expr: &Expr) -> Option<&str> {
    if tag(expr) == Some(DeepTag::Var) {
        children(expr).first().and_then(sym_str)
    } else {
        None
    }
}

fn meta_map(expr: &Expr) -> Option<&chelis_deep::MetaMap> {
    if let Expr::List(list, _) = expr
        && let Some(Expr::Map(map, _)) = list.elements.get(1)
    {
        Some(map)
    } else {
        None
    }
}

fn meta_value<'a>(expr: &'a Expr, key: &str) -> Option<&'a Expr> {
    meta_map(expr)?
        .entries
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v)
}

fn has_true_meta(expr: &Expr, key: &str) -> bool {
    matches!(meta_value(expr, key), Some(Expr::Atom(Atom::Bool(true), _)))
}

fn str_value(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Str(s), _) => Some(s.as_str()),
        _ => None,
    }
}

/// The `(def {} name <body>)` parts, if `expr` is a def.
fn as_def(expr: &Expr) -> Option<(&str, &Expr)> {
    if tag(expr) == Some(DeepTag::Def) {
        let kids = children(expr);
        if let (Some(name), Some(body)) = (kids.first().and_then(sym_str), kids.get(1)) {
            return Some((name, body));
        }
    }
    None
}

/// Whether a def body is a zero-argument constant: a bare value, or a
/// `fn` with an empty params node.
fn is_zero_arg_constant(body: &Expr) -> bool {
    if tag(body) == Some(DeepTag::Fn) {
        let kids = children(body);
        if let Some(params) = kids.first()
            && tag(params) == Some(DeepTag::Params)
        {
            return children(params).is_empty();
        }
        return false;
    }
    // Any non-fn body is a constant binding.
    true
}

/// The body (predicate) of a `(fn {} (params {} <binder>) <body>)` node.
fn fn_body(fn_node: &Expr) -> Option<&Expr> {
    if tag(fn_node) == Some(DeepTag::Fn) {
        children(fn_node).get(1)
    } else {
        None
    }
}

/// Flatten the program, pairing each top-level decl with its enclosing
/// lexical module key. Mirrors `infer::top_level_decl_items_with_modules`
/// but is local to this module (chelis-pred + chelis-deep only).
fn flatten_with_modules(exprs: &[Expr]) -> Vec<(Option<String>, &Expr)> {
    fn push<'a>(expr: &'a Expr, prefix: Option<&str>, out: &mut Vec<(Option<String>, &'a Expr)>) {
        if tag(expr) == Some(DeepTag::Module) {
            let name = children(expr).first().and_then(sym_str);
            let key = match (prefix, name) {
                (Some(p), Some(n)) => Some(format!("{p}.{n}")),
                (None, Some(n)) => Some(n.to_string()),
                (p, None) => p.map(str::to_string),
            };
            if let Expr::List(list, _) = expr {
                // `(module {} name children...)`: skip tag, meta, name.
                for child in list.elements.iter().skip(3) {
                    push(child, key.as_deref(), out);
                }
            }
            return;
        }
        out.push((prefix.map(str::to_string), expr));
    }
    let mut out = Vec::new();
    for expr in exprs {
        push(expr, None, &mut out);
    }
    out
}

fn err(message: String) -> CheckError {
    CheckError::new(CheckErrorKind::OpaqueTypeViolation, message, Vec::new())
}
