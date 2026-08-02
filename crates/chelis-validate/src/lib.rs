use chelis_deep::DeepTag;
use chelis_deep::printer::print_canonical;
use pest::Parser;
use pest::iterators::Pair;
use thiserror::Error;

mod deep {
    use pest_derive::Parser;

    #[derive(Parser)]
    #[grammar = "deep.pest"]
    pub struct Grammar;
}

mod surf {
    use pest_derive::Parser;

    #[derive(Parser)]
    #[grammar = "surf.pest"]
    pub struct Grammar;
}

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("validation failed: {0}")]
    Failed(String),
}

pub fn validate_surf(source: &str) -> Result<(), ValidationError> {
    // Phase 1f (spec/design/phase1f_executable_grammar.md): the pest
    // grammar is an independent second implementation and is explicitly
    // "not a replacement for the parser" — the hand-written compiler
    // parser is the acceptance authority, and validator/compiler
    // *agreement* is the oracle. The exit verdict therefore follows the
    // parser in BOTH directions, not just when the grammar happens to
    // reject:
    //   * grammar accepts + parser accepts -> accept (agreement)
    //   * grammar rejects + parser accepts -> accept; the grammar is
    //     merely incomplete relative to the shipped surface (the
    //     executable examples rely on this rescue path)
    //   * grammar accepts + parser rejects -> REJECT with the compiler's
    //     reason (chelis#706: the grammar admits bare-statement
    //     juxtaposition the parser rejects). Reporting success here would
    //     disagree with the compile path — `check`/`fmt`/`build`/`eval`/
    //     `validate --desugar` all reject the same input.
    //   * grammar rejects + parser rejects -> reject, surfacing both views
    // Whenever the two disagree the diagnostic names the split — the
    // conformance tool's "valuable finding" is preserved, not papered over.
    let grammar = surf::Grammar::parse(surf::Rule::program, source);
    match (chelis_surf::parser::parse_str(source), grammar) {
        (Ok(_), _) => Ok(()),
        (Err(parse_err), Ok(_)) => Err(ValidationError::Failed(format!(
            "the Surf PEG grammar accepts this program but the compiler parser rejects it \
             (the grammar is too lenient, a chelis-validate conformance gap); \
             compiler parse failed: {parse_err}"
        ))),
        (Err(parse_err), Err(pest_err)) => Err(ValidationError::Failed(format!(
            "{pest_err}\ncompiler parse failed: {parse_err}"
        ))),
    }
}

pub fn validate_deep(source: &str) -> Result<(), ValidationError> {
    let mut parsed = deep::Grammar::parse(deep::Rule::program, source)
        .map_err(|err| ValidationError::Failed(err.to_string()))?;
    let Some(program) = parsed.next() else {
        return Err(ValidationError::Failed("empty Deep program".to_string()));
    };
    for pair in program.into_inner() {
        if pair.as_rule() == deep::Rule::node {
            validate_deep_node(pair)?;
        }
    }
    // RFC v4b (RT-1 F2) + v5 (RT-1 F2 bypass): structural module-identity
    // forgery checks. The grammar admits two same-name wrappers and the
    // reef linker's reserved internal-name format, but both forge module
    // identity (the checker rejects them too). `validate --deep` only
    // ever processes hand-authored `.dp` (the linker feeds Deep to the
    // checker in-process and never writes `.dp`), so the reserved-name
    // rejection here is unconditional. Parse through the AST parser (the
    // grammar already validated above, so this succeeds).
    if let Ok(exprs) = chelis_deep::parser::parse_str_strict(source) {
        // #1047: stamped parsing may produce Expr::Node; normalize for checks.
        let exprs = normalize_to_lists(&exprs);
        if let Some(name) = first_forged_linker_name(&exprs) {
            return Err(ValidationError::Failed(format!(
                "`{name}` uses the reef package-linker's reserved internal-name \
                 format (`Pkg__`/`pkg__`...), which only the linker may produce; \
                 rename the declaration"
            )));
        }
        if let Some(name) = first_reopened_module(&exprs) {
            return Err(ValidationError::Failed(format!(
                "module `{name}` is opened by more than one module wrapper; \
                 a named module may be opened at most once"
            )));
        }
        if let Some(name) = first_duplicate_defsig(&exprs) {
            return Err(ValidationError::Failed(format!(
                "duplicate signature: `{name}` has more than one `defsig`; \
                 Chelis does not dispatch same-name functions by argument type, arity, or rank"
            )));
        }
    }
    Ok(())
}

/// True when `name` matches the reef linker's internal-name format
/// (`Pkg__<pkg>__<Module>__<Name>` / lowercase twin): a marker prefix
/// plus a non-empty module stem before the terminal. Mirrors the
/// checker's `opacity::is_linker_format_name` (RFC v5).
fn is_linker_format_name(name: &str) -> bool {
    let Some(stem) = name
        .strip_prefix("Pkg__")
        .or_else(|| name.strip_prefix("pkg__"))
    else {
        return false;
    };
    matches!(stem.rsplit_once("__"), Some((module, _)) if !module.is_empty())
}

/// Return the first top-level declaration binding name that matches the
/// reef linker's reserved internal-name format (RFC v5), or `None`.
fn first_forged_linker_name(exprs: &[chelis_deep::ast::Expr]) -> Option<String> {
    fn binding_name(list: &chelis_deep::ast::List) -> Option<&str> {
        // `defmacro` is compiler-internal pre-expansion syntax outside the
        // vocabulary and stays symbol-headed (raw-string boundary).
        let is_binding_decl = matches!(
            list.tag(),
            Some(DeepTag::Deftype | DeepTag::Def | DeepTag::Defsig | DeepTag::Typealias)
        ) || list.unknown_tag_symbol() == Some("defmacro");
        if !is_binding_decl {
            return None;
        }
        match list.elements.get(2) {
            Some(chelis_deep::ast::Expr::Atom(chelis_deep::ast::Atom::Name(name), _)) => {
                Some(name.as_str())
            }
            _ => None,
        }
    }
    fn walk(expr: &chelis_deep::ast::Expr) -> Option<String> {
        let chelis_deep::ast::Expr::List(list, _) = expr else {
            return None;
        };
        let is_module = list.tag() == Some(DeepTag::Module);
        if is_module {
            // Descend into a module wrapper's children.
            for child in list.elements.iter().skip(3) {
                if let Some(found) = walk(child) {
                    return Some(found);
                }
            }
            return None;
        }
        if let Some(name) = binding_name(list)
            && is_linker_format_name(name)
        {
            return Some(name.to_string());
        }
        None
    }
    exprs.iter().find_map(walk)
}

/// Return the first module name opened by more than one `(module ...)`
/// wrapper (nested wrappers keyed by their full `.`-joined path), or
/// `None` if every wrapper name is unique. Mirrors the checker's
/// `detect_module_reopens` (RFC v4b, RT-1 F2).
fn first_reopened_module(exprs: &[chelis_deep::ast::Expr]) -> Option<String> {
    fn module_name(list: &chelis_deep::ast::List) -> Option<&str> {
        if list.tag() != Some(DeepTag::Module) {
            return None;
        }
        match list.elements.get(2) {
            Some(chelis_deep::ast::Expr::Atom(chelis_deep::ast::Atom::Name(name), _)) => {
                Some(name.as_str())
            }
            _ => None,
        }
    }
    fn walk(
        expr: &chelis_deep::ast::Expr,
        prefix: Option<&str>,
        seen: &mut std::collections::HashSet<String>,
    ) -> Option<String> {
        let chelis_deep::ast::Expr::List(list, _) = expr else {
            return None;
        };
        let name = module_name(list)?;
        let key = match prefix {
            Some(p) => format!("{p}.{name}"),
            None => name.to_string(),
        };
        if !seen.insert(key.clone()) {
            return Some(key);
        }
        for child in list.elements.iter().skip(3) {
            if let Some(dup) = walk(child, Some(&key), seen) {
                return Some(dup);
            }
        }
        None
    }
    let mut seen = std::collections::HashSet::new();
    for expr in exprs {
        if let Some(dup) = walk(expr, None, &mut seen) {
            return Some(dup);
        }
    }
    None
}

/// Return the first function name that appears in more than one `(defsig ...)`.
/// This mirrors the checker's language rule: Chelis does not overload user
/// functions by arity, type, or rank, so multiple signatures for one name have
/// no valid dispatch meaning.
fn first_duplicate_defsig(exprs: &[chelis_deep::ast::Expr]) -> Option<String> {
    fn tag(list: &chelis_deep::ast::List) -> Option<DeepTag> {
        list.tag()
    }

    fn symbol_child(list: &chelis_deep::ast::List, index: usize) -> Option<&str> {
        match list.elements.get(index) {
            Some(chelis_deep::ast::Expr::Atom(chelis_deep::ast::Atom::Name(name), _)) => {
                Some(name.as_str())
            }
            _ => None,
        }
    }

    fn walk(
        expr: &chelis_deep::ast::Expr,
        seen: &mut std::collections::HashSet<String>,
    ) -> Option<String> {
        let chelis_deep::ast::Expr::List(list, _) = expr else {
            return None;
        };
        match tag(list) {
            Some(DeepTag::Module) => list
                .elements
                .iter()
                .skip(3)
                .find_map(|child| walk(child, seen)),
            Some(DeepTag::Defsig) => {
                let name = symbol_child(list, 2)?;
                if seen.insert(name.to_string()) {
                    None
                } else {
                    Some(name.to_string())
                }
            }
            _ => None,
        }
    }

    let mut seen = std::collections::HashSet::new();
    exprs.iter().find_map(|expr| walk(expr, &mut seen))
}

pub fn validate_desugared(source: &str) -> Result<(), ValidationError> {
    let decls = chelis_surf::parser::parse_str(source)
        .map_err(|err| ValidationError::Failed(format!("compiler parse failed: {err}")))?;
    let deep = chelis_surf::desugar::desugar_program(&decls);
    let deep = chelis_macros::expand_program(&deep, &chelis_macros::ExpansionOptions::default())
        .map_err(|err| ValidationError::Failed(format!("macro expansion failed: {err}")))?
        .into_exprs();
    let canonical = print_canonical(&deep);
    validate_deep(&canonical)
}

/// True for inner pairs that are structurally invisible to a node's
/// shape: atomic `comment` tokens (which `spacing` can capture anywhere a
/// node's internal `spacing` appears) and the synthetic `EOI` marker.
/// Every site that reads a node's tag, meta, or children must skip these
/// so a leading `;` comment is never mistaken for the tag. See issue #167.
fn is_structural_pair(pair: &Pair<'_, deep::Rule>) -> bool {
    !matches!(pair.as_rule(), deep::Rule::comment | deep::Rule::EOI)
}

fn validate_deep_node(pair: Pair<'_, deep::Rule>) -> Result<(), ValidationError> {
    let span = pair.as_span();
    // `comment` is an atomic (visible) rule, so any comment captured by a
    // node's internal `spacing` (before the tag, around the meta block, or
    // trailing after the last child) surfaces as an inner pair here. Filter
    // those out up front, like `EOI`, so they are never mistaken for the
    // tag, the meta block, or a node child. See issue #167.
    let mut inner = pair.into_inner().filter(is_structural_pair);
    let tag = inner.next().expect("node tag").as_str().to_string();
    let meta = inner.next().expect("node meta");
    let children: Vec<_> = inner
        .filter_map(|pair| match pair.as_rule() {
            deep::Rule::child => pair.into_inner().next(),
            _ => Some(pair),
        })
        .collect();

    if meta.as_rule() != deep::Rule::meta {
        return Err(ValidationError::Failed(format!(
            "Deep node `{tag}` at byte {} is missing metadata",
            span.start()
        )));
    }

    // Membership in the closed vocabulary is the single `DeepTag`
    // declaration (chelis#731 Phase 3); the duplicated local string list
    // and its drift test are retired.
    let Some(deep_tag) = DeepTag::parse(&tag) else {
        return Err(ValidationError::Failed(format!(
            "unknown Deep tag `{tag}` at byte {}",
            span.start()
        )));
    };

    for child in &children {
        match child.as_rule() {
            deep::Rule::node => validate_deep_node(child.clone())?,
            deep::Rule::typed_helper => validate_typed_helper(child.clone())?,
            deep::Rule::unit_list | deep::Rule::literal | deep::Rule::bare_name => {}
            other => {
                return Err(ValidationError::Failed(format!(
                    "unexpected Deep child rule {:?} under `{tag}` at byte {}",
                    other,
                    child.as_span().start()
                )));
            }
        }
    }

    validate_tag_shape(deep_tag, &children, span.start())
}

fn validate_typed_helper(pair: Pair<'_, deep::Rule>) -> Result<(), ValidationError> {
    // Skip any visible `comment` pairs captured by the helper's internal
    // `spacing`, mirroring `validate_deep_node`. See issue #167.
    let mut inner = pair.clone().into_inner().filter(is_structural_pair);
    let name = inner
        .next()
        .expect("typed helper name")
        .as_str()
        .to_string();
    let meta = inner.next().expect("typed helper meta");
    if meta.as_rule() != deep::Rule::typed_meta {
        return Err(ValidationError::Failed(format!(
            "typed helper `{name}` at byte {} is missing metadata",
            pair.as_span().start()
        )));
    }
    Ok(())
}

/// Per-tag shape validation over the closed vocabulary. The match is
/// exhaustive with no `_` arm (chelis#731 Phase 3, checker_totality.md
/// §C4.2): a 63rd `DeepTag` variant fails to compile here until it gets an
/// explicit shape disposition.
fn validate_tag_shape(
    deep_tag: DeepTag,
    children: &[Pair<'_, deep::Rule>],
    offset: usize,
) -> Result<(), ValidationError> {
    let tag = deep_tag.as_str();
    let child_count = children.len();

    let wrong_arity = |expected: &str| {
        ValidationError::Failed(format!(
            "Deep tag `{tag}` has invalid arity at byte {offset}: expected {expected}, found {child_count}"
        ))
    };

    match deep_tag {
        DeepTag::If | DeepTag::Arm => {
            if child_count != 3 {
                return Err(wrong_arity("exactly 3 children"));
            }
            Ok(())
        }
        DeepTag::HandleEffect => {
            if child_count != 2 {
                return Err(wrong_arity("exactly 2 children"));
            }
            Ok(())
        }
        DeepTag::Fn => {
            if child_count != 2 {
                return Err(wrong_arity("exactly 2 children"));
            }
            expect_node_tag(&children[0], "params", offset)
        }
        DeepTag::Let => {
            if child_count != 2 {
                return Err(wrong_arity("exactly 2 children"));
            }
            expect_node_tag(&children[0], "bind", offset)
        }
        DeepTag::App => {
            if child_count < 1 {
                return Err(wrong_arity("at least 1 child"));
            }
            Ok(())
        }
        DeepTag::Params => validate_params_children(children, offset),
        DeepTag::Bind => validate_bind_children(children, offset),
        DeepTag::Effects => validate_effects_children(children, offset),
        DeepTag::Resource => {
            if child_count != 1 {
                return Err(wrong_arity("exactly 1 child"));
            }
            Ok(())
        }
        // No additional shape constraint at this validator: these tags'
        // arity/shape rules are owned by the type checker (spec/03
        // §2.5.1/§2.6, §8.2) or by their enclosing form. Listed explicitly
        // rather than wildcarded so a 63rd tag forces a decision here.
        DeepTag::Module
        | DeepTag::Import
        | DeepTag::ImportAll
        | DeepTag::Export
        | DeepTag::Def
        | DeepTag::Defsig
        | DeepTag::Deftype
        | DeepTag::Typealias
        | DeepTag::Variant
        | DeepTag::Field
        | DeepTag::Defdim
        | DeepTag::Match
        | DeepTag::Var
        | DeepTag::Lit
        | DeepTag::Record
        | DeepTag::Access
        | DeepTag::Pipe
        | DeepTag::Block
        | DeepTag::Tuple
        | DeepTag::TupleGet
        | DeepTag::RecordUpdate
        | DeepTag::Par
        | DeepTag::Borrow
        | DeepTag::PatVar
        | DeepTag::PatLit
        | DeepTag::PatCtor
        | DeepTag::PatTuple
        | DeepTag::PatRecord
        | DeepTag::PatWild
        | DeepTag::PatAs
        | DeepTag::TPrim
        | DeepTag::TFn
        | DeepTag::TTensor
        | DeepTag::TRef
        | DeepTag::TAdt
        | DeepTag::TVar
        | DeepTag::TUnit
        | DeepTag::TTuple
        | DeepTag::DName
        | DeepTag::DVar
        | DeepTag::DLit
        | DeepTag::DRank
        | DeepTag::Grad
        | DeepTag::Vmap
        | DeepTag::Jit
        | DeepTag::Realize
        | DeepTag::Cast
        | DeepTag::Copy
        | DeepTag::Quote
        | DeepTag::Unquote
        | DeepTag::Splice
        | DeepTag::Kv => Ok(()),
    }
}

fn expect_node_tag(
    child: &Pair<'_, deep::Rule>,
    expected_tag: &str,
    offset: usize,
) -> Result<(), ValidationError> {
    if child.as_rule() != deep::Rule::node {
        return Err(ValidationError::Failed(format!(
            "Deep tag `{expected_tag}` expected nested node at byte {offset}"
        )));
    }
    // A nested node's first internal `spacing` can capture a `;` comment
    // (e.g. `(; note\nparams {} ...)`), which surfaces as a visible
    // `comment` pair ahead of the tag. Skip those so the tag is read
    // correctly, mirroring `validate_deep_node`. See issue #167.
    let mut inner = child.clone().into_inner().filter(is_structural_pair);
    let found = inner.next().expect("nested tag").as_str().to_string();
    if found != expected_tag {
        return Err(ValidationError::Failed(format!(
            "expected `{expected_tag}` helper at byte {offset}, found `{found}`"
        )));
    }
    Ok(())
}

fn validate_params_children(
    children: &[Pair<'_, deep::Rule>],
    offset: usize,
) -> Result<(), ValidationError> {
    for child in children {
        match child.as_rule() {
            deep::Rule::bare_name | deep::Rule::typed_helper => {}
            _ => {
                return Err(ValidationError::Failed(format!(
                    "`params` must contain bare names or typed helpers at byte {offset}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_bind_children(
    children: &[Pair<'_, deep::Rule>],
    offset: usize,
) -> Result<(), ValidationError> {
    if !children.len().is_multiple_of(2) {
        return Err(ValidationError::Failed(format!(
            "`bind` must contain name/expression pairs at byte {offset}"
        )));
    }
    for (index, child) in children.iter().enumerate() {
        if index % 2 == 0 && child.as_rule() != deep::Rule::bare_name {
            return Err(ValidationError::Failed(format!(
                "`bind` name position {} must be a bare name at byte {offset}",
                index / 2
            )));
        }
    }
    Ok(())
}

fn validate_effects_children(
    children: &[Pair<'_, deep::Rule>],
    offset: usize,
) -> Result<(), ValidationError> {
    for child in children {
        match child.as_rule() {
            deep::Rule::bare_name => {}
            // A `(resource {} ...)` entry whose first internal `spacing`
            // captures a `;` comment surfaces that comment ahead of the
            // tag, so skip non-structural pairs before reading the tag.
            // See issue #167.
            deep::Rule::node
                if child
                    .clone()
                    .into_inner()
                    .find(is_structural_pair)
                    .is_some_and(|tag| tag.as_str() == "resource") => {}
            _ => {
                return Err(ValidationError::Failed(format!(
                    "`effects` must contain bare names or `(resource {{}} ...)` entries at byte {offset}"
                )));
            }
        }
    }
    Ok(())
}


/// Recursively normalize `Expr::Node`/`Expr::BareList` → `Expr::List` for
/// structural validation checks that only handle `Expr::List`.
fn normalize_to_lists(exprs: &[chelis_deep::ast::Expr]) -> Vec<chelis_deep::ast::Expr> {
    exprs.iter().map(normalize_expr_to_list).collect()
}

fn normalize_expr_to_list(expr: &chelis_deep::ast::Expr) -> chelis_deep::ast::Expr {
    use chelis_deep::ast::{Expr, List};
    match expr {
        Expr::Node(node, span) => {
            let list = node.to_list(*span);
            let elements = list.elements.iter().map(normalize_expr_to_list).collect();
            Expr::List(List { elements }, *span)
        }
        Expr::BareList(elems, span) => {
            let elements = elems.iter().map(normalize_expr_to_list).collect();
            Expr::List(List { elements }, *span)
        }
        Expr::List(list, span) => {
            let elements = list.elements.iter().map(normalize_expr_to_list).collect();
            Expr::List(List { elements }, *span)
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::{validate_deep, validate_desugared, validate_surf};

    #[test]
    fn valid_tags_match_canonical_deep_vocabulary() {
        // The Phase 1f executable-grammar validator must accept exactly the
        // tags the compiler's strict parser accepts. Pre-chelis#731-Phase-3
        // this asserted set equality between a local `VALID_TAGS` copy and
        // `chelis_deep::validate::VALID_TAGS`; the local copy is retired and
        // membership is now the single `chelis_deep::DeepTag` declaration on
        // both surfaces, so a list drift is unrepresentable. What remains
        // observable is rejection parity on non-members (acceptance parity
        // on members is exercised by the corpus tests in this module, e.g.
        // the `t-ref`/`d-rank` case that motivated the original test).
        for bogus in ["mystery", "defmacro", "macro-invoke", "sig", "apply"] {
            let source = format!("({bogus} {{}} x)");
            assert!(
                validate_deep(&source).is_err(),
                "`{bogus}` must be rejected by the executable-grammar validator"
            );
            assert!(
                chelis_deep::parser::parse_str_strict(&source).is_err(),
                "`{bogus}` must be rejected by the compiler's strict parser"
            );
        }
        assert_eq!(
            chelis_deep::validate::VALID_TAGS.len(),
            chelis_deep::DeepTag::COUNT,
            "the derived string list must cover the whole vocabulary"
        );
    }

    #[test]
    fn deep_accepts_rank_polymorphic_borrow_annotation() {
        // `&tensor[..r, f32]` desugars to `(t-ref {} (t-tensor {} (d-rank {} r) ...))`.
        // Both `t-ref` and `d-rank` must be in the vocabulary.
        let source = "(defsig {} f (t-fn {} (t-ref {} (t-tensor {} (d-rank {} r) (t-prim {} f32))) (t-tensor {} (d-rank {} r) (t-prim {} f32))))\n";
        validate_deep(source)
            .expect("validator should accept canonical t-ref / d-rank rank-polymorphic Deep");
    }

    fn assert_duplicate_defsig_rejected(source: &str) {
        let error = validate_deep(source).expect_err("duplicate defsig should fail validation");
        assert!(
            error.to_string().contains("duplicate signature"),
            "diagnostic should name duplicate signature, got: {error}"
        );
    }

    #[test]
    fn deep_rejects_duplicate_defsig_conflicting_order_a() {
        assert_duplicate_defsig_rejected(
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (defsig {} f (t-fn {} (t-prim {} bool) (t-prim {} f32)))",
        );
    }

    #[test]
    fn deep_rejects_duplicate_defsig_conflicting_order_b() {
        assert_duplicate_defsig_rejected(
            "(defsig {} f (t-fn {} (t-prim {} bool) (t-prim {} f32)))
             (defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))",
        );
    }

    #[test]
    fn deep_rejects_duplicate_defsig_identical_signature() {
        assert_duplicate_defsig_rejected(
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))",
        );
    }

    #[test]
    fn surf_accepts_top_level_binding_program() {
        let source = "x = (x : tensor[32, 784, f32])\ny = relu(x)\n";
        validate_surf(source).expect("validator should accept executable script-style bindings");
    }

    #[test]
    fn surf_accepts_semicolon_block_and_axis_identifier() {
        let source = "def f(axis) = { y = axis; y }\ndef g() = par { a; b }\n";
        validate_surf(source)
            .expect("validator should accept semicolon block/par and axis identifiers");
    }

    #[test]
    fn surf_accepts_arrow_return_types() {
        let source = "def f(x: tensor[n, f32]) -> tensor[n, f32] = relu(x)\n";
        validate_surf(source).expect("validator should accept arrow return types");
    }

    #[test]
    fn surf_accepts_short_block_bindings() {
        let source = "def f(x) = {\n  y = relu(x)\n  y\n}\n";
        validate_surf(source).expect("validator should accept short block bindings");
    }

    // chelis#706: `validate --surf` must agree with the compiler parser.
    // The pest grammar admits bare-statement juxtaposition inside a block,
    // but the hand-written parser rejects it (BareStatementInBlock), so the
    // validator — whose exit verdict follows the parser — must reject too,
    // and name the grammar/parser split.
    #[test]
    fn surf_rejects_bare_statement_the_parser_rejects() {
        let source = "def f(a: f32, b: f32, c: f32, d: f32) -> f32 = {\n  g(a, b)\n  h(c, d)\n}\n";
        use pest::Parser as _;
        // Premise: the compiler parser rejects it and the pest grammar accepts it,
        // i.e. this is exactly the grammar-too-lenient case.
        assert!(
            chelis_surf::parser::parse_str(source).is_err(),
            "precondition: the compiler parser rejects the bare-statement block"
        );
        assert!(
            super::surf::Grammar::parse(super::surf::Rule::program, source).is_ok(),
            "precondition: the pest grammar admits the bare-statement block"
        );
        let err = validate_surf(source)
            .expect_err("validator must reject what the compiler parser rejects");
        let msg = err.to_string();
        assert!(
            msg.contains("compiler parser rejects it") && msg.contains("too lenient"),
            "diagnostic should name the grammar/parser split; got: {msg}"
        );
        assert!(
            msg.contains("expression statement must be bound"),
            "diagnostic should carry the parser's #706 reason; got: {msg}"
        );
    }

    #[test]
    fn surf_par_bare_items_are_rejected_like_the_parser() {
        // The par{} companion: newline-separated items the parser rejects.
        let source = "def f(x, y) -> Unit = par {\n  g(x)\n  h(y)\n}\n";
        assert!(
            chelis_surf::parser::parse_str(source).is_err(),
            "precondition: the compiler parser rejects bare par items"
        );
        validate_surf(source).expect_err("validator must reject bare par items too");
    }

    /// chelis#858 rejection parity: both `.dp` validation surfaces reject
    /// an untagged top-level list. The pest grammar always did; the
    /// AST-walking validator (`chelis_deep::validate`) now warns too.
    #[test]
    fn deep_rejects_untagged_top_level_list_on_both_surfaces() {
        let source = "((var {} f) (var {} x))\n";
        validate_deep(source).expect_err("pest grammar must reject an untagged top-level list");
        let exprs = chelis_deep::parser::parse_str(source).expect("lenient parse");
        assert!(
            !chelis_deep::validate::validate(&exprs).is_empty(),
            "the AST validator must reject the same input class"
        );
    }

    #[test]
    fn deep_rejects_unknown_tag() {
        let source = "(mystery {} x)";
        let error = validate_deep(source).expect_err("unknown tag should fail");
        assert!(error.to_string().contains("unknown Deep tag"));
    }

    #[test]
    fn deep_accepts_snake_case_metadata_keys() {
        let source = "(def {c_earchin_role: \"property_witness\"} req_PRC_001 (fn {} (params {}) (lit {type: (t-prim {} bool)} true)))";
        validate_deep(source).expect("producer metadata keys may be snake_case");
    }

    #[test]
    fn deep_rejects_invalid_effects_children() {
        let source = "(effects {} 1)";
        let error = validate_deep(source).expect_err("non-symbol effects child should fail");
        assert!(
            error
                .to_string()
                .contains("`effects` must contain bare names")
        );
    }

    #[test]
    fn deep_rejects_invalid_resource_arity() {
        let source = "(resource {} x y)";
        let error = validate_deep(source).expect_err("resource arity should fail");
        assert!(error.to_string().contains("expected exactly 1 child"));
    }

    #[test]
    fn desugared_accepts_dotted_module_paths() {
        let source = "module Foo.Bar\nimport Baz.Qux(..)\ndef f(x) = x\n";
        validate_desugared(source).expect("desugared dotted module/import paths should validate");
    }

    #[test]
    fn desugared_accepts_executable_example_shape() {
        let source = "module HelloTensor\n\ndef main() -> tensor[f32] = 1\n";
        validate_desugared(source).expect("desugared Deep should validate");
    }

    // ---- issue #167: leading `;` comment lines ----------------------------
    //
    // The pest grammar must treat `;` as a line comment everywhere Deep
    // does, including before the first node and inside `{...}` metadata.
    // Before the `comment = @{...}` atomic fix, pest auto-inserted
    // implicit WHITESPACE that swallowed the newline terminating the
    // comment, so any of these leading-comment shapes failed to parse.
    //
    // Each positive case asserts that the commented form validates AND
    // that it accepts exactly what the comment-free form does, so the two
    // surfaces stay in parity with `chelis_deep::parser::parse_str_strict`.

    /// The base program these comment cases wrap, comment-free.
    const BASE: &str = "(module {} hello)\n";

    fn assert_validates(source: &str, label: &str) {
        // Both the commented and the bare form must validate, and the
        // strict hand-rolled parser must also accept the source, so the
        // two Deep surfaces agree (the core complaint of #167).
        validate_deep(source).unwrap_or_else(|err| panic!("{label} should validate: {err}"));
        validate_deep(BASE).expect("base program should validate");
        chelis_deep::parser::parse_str_strict(source)
            .unwrap_or_else(|err| panic!("{label} should parse strictly: {err}"));
    }

    #[test]
    fn deep_accepts_single_leading_comment() {
        assert_validates(
            "; a leading comment\n(module {} hello)\n",
            "single leading comment",
        );
    }

    #[test]
    fn deep_accepts_directive_leading_comment() {
        // The `; chelis-lint:` directive form is the shape that forced
        // every CLI callsite to pre-strip; it must now validate directly.
        assert_validates(
            "; chelis-lint: disable=foo\n(module {} hello)\n",
            "leading chelis-lint directive",
        );
    }

    #[test]
    fn deep_accepts_multiple_leading_comments() {
        assert_validates(
            "; first\n; second\n(module {} hello)\n",
            "two consecutive leading comments",
        );
    }

    #[test]
    fn deep_accepts_leading_comment_with_blank_lines() {
        assert_validates(
            "; a leading comment\n\n; another\n\n(module {} hello)\n",
            "leading comments separated by blank lines",
        );
    }

    #[test]
    fn deep_accepts_trailing_comment() {
        assert_validates("(module {} hello)\n; trailing\n", "trailing comment");
    }

    #[test]
    fn deep_accepts_mid_program_comment() {
        assert_validates(
            "(module {} hello)\n; mid\n(module {} world)\n",
            "mid-program comment between nodes",
        );
    }

    #[test]
    fn deep_accepts_meta_internal_comment() {
        // The surprising case from the issue table: a `;` comment inside a
        // `{...}` metadata block. `spacing` (which includes `comment`)
        // appears throughout `meta`, so the atomic fix covers it too.
        assert_validates(
            "(module {\n; inside meta\n} hello)\n",
            "comment inside metadata block",
        );
    }

    #[test]
    fn deep_accepts_comment_before_tag() {
        // A comment captured by the node's first internal `spacing`, between
        // `(` and the tag. The visible `comment` token must be skipped so the
        // tag is still read correctly.
        assert_validates("(; before tag\nmodule {} hello)\n", "comment before tag");
    }

    #[test]
    fn deep_accepts_comment_around_meta_and_children() {
        // Comments in the node's internal spacing around the meta block and
        // before a child must not be mistaken for the meta block or a child.
        assert_validates(
            "(module ; after tag\n{} ; after meta\nhello ; after child\n)\n",
            "comments around meta and children",
        );
    }

    #[test]
    fn deep_accepts_leading_comment_without_trailing_newline() {
        // No newline after the final node; the leading comment is still
        // terminated by its own newline before the node.
        assert_validates(
            "; leading\n(module {} hello)",
            "leading comment, no trailing newline",
        );
    }

    #[test]
    fn deep_leading_comment_equals_uncommented_program() {
        // The commented program validates to the same acceptance decision
        // as the program with the comments removed, which is the issue's
        // core "equals the same program without the comments" requirement.
        let commented = "; chelis-lint: disable=foo\n; plain comment\n(module {} hello)\n";
        validate_deep(commented).expect("commented program should validate");
        validate_deep(BASE).expect("uncommented program should validate");
    }

    // ---- issue #167 regression: comments leading a *nested* node's tag ----
    //
    // `validate_deep_node` filters comments before reading its own tag, but
    // several shape checks reach into a nested node and read *its* first
    // inner pair as the tag: `expect_node_tag` (the `params` child of `fn`
    // and the `bind` child of `let`) and `validate_effects_children` (the
    // `resource` entry of `effects`). Each of those nested nodes can carry
    // a `;` comment in its first internal `spacing`, which surfaces as a
    // visible `comment` pair ahead of the tag. If the introspection does
    // not skip it, the validator reads the comment text as the tag and
    // wrongly rejects source that `parse_str_strict` accepts.

    #[test]
    fn deep_accepts_comment_before_params_tag_in_fn() {
        assert_validates(
            "(fn {} (; note\nparams {}) (var {} x))\n",
            "comment before nested `params` tag",
        );
    }

    #[test]
    fn deep_accepts_comment_before_bind_tag_in_let() {
        assert_validates(
            "(let {} (; note\nbind {}) (var {} x))\n",
            "comment before nested `bind` tag",
        );
    }

    #[test]
    fn deep_accepts_comment_before_resource_tag_in_effects() {
        assert_validates(
            "(effects {} (; note\nresource {} foo))\n",
            "comment before nested `resource` tag",
        );
    }

    // ---- negative parity: the grammar was not loosened -------------------

    #[test]
    fn deep_still_rejects_unterminated_node() {
        // A genuinely malformed Deep program (missing closing paren) must
        // still fail to parse. Confirms the atomic-comment fix did not
        // relax the grammar into accepting broken structure.
        let source = "; a comment\n(module {} hello";
        validate_deep(source).expect_err("unterminated node must still fail to parse");
    }

    #[test]
    fn deep_still_rejects_missing_metadata() {
        // A node without its mandatory `{}` metadata block must still fail,
        // even when preceded by a leading comment.
        let source = "; a comment\n(module hello)\n";
        validate_deep(source).expect_err("node without metadata must still fail");
    }

    #[test]
    fn deep_still_rejects_garbage_after_comment() {
        // Non-node, non-comment garbage following a leading comment must
        // not be swallowed into a comment and silently accepted.
        let source = "; a comment\n@@@ not a node\n";
        validate_deep(source).expect_err("garbage after a comment must still fail to parse");
    }
}
