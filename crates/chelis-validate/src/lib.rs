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
    // chelis#1088: the stamped `.dp` ingress runs FIRST, so `validate --deep`
    // reaches the same program-level verdict `chelis check` does, in the same
    // words. The order is the whole point. This auxiliary Pest grammar admits
    // only `node+`, so when it ran first a headless top-level form died as
    // `expected program` before anything could name it, and a reader got a
    // caret instead of the [03-PROG-2] class the rule requires. Running the
    // stamped ingress first means [03-PROG-1], [03-PROG-2], and [03-PROG-3]
    // decide top-level acceptance, and the grammar keeps the structural
    // checks below the top level that it alone performs.
    //
    // The second thing this ordering buys: a source the stamp rejects is a
    // validation failure rather than a silent skip of the forgery checks
    // below. That skip was the original hole, because a forged name in a
    // source that happened not to stamp passed unreported.
    let exprs = chelis_deep::parse_and_stamp_file(source)
        .map_err(|err| ValidationError::Failed(err.to_string()))?;

    // The AST-side structural sweep, on the stamped tree. The Pest leg below
    // walks node children only, so a malformed shape inside a metadata value
    // -- `(effects {} 1)` under a `t-fn`'s `eff:`, say -- is structurally
    // invisible to it. `chelis check` has always caught those through this
    // validator; running it here is what stops the two surfaces disagreeing
    // about anything but the top-level rule.
    if let Some(warning) = chelis_deep::validate::validate(&exprs).into_iter().next() {
        return Err(ValidationError::Failed(warning.message));
    }

    // [03-META-3]: the sealed data parser owns extension syntax. The
    // independent program grammar must not impose its expression grammar on it.
    let structural_source = deep_program_projection(source, &exprs);
    let mut parsed = deep::Grammar::parse(deep::Rule::program, &structural_source)
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
    // rejection here is unconditional.
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

fn deep_node_parts(expr: &chelis_deep::ast::Expr) -> Option<(DeepTag, &[chelis_deep::ast::Expr])> {
    match expr {
        chelis_deep::ast::Expr::Node(node, _) => Some((node.tag(), node.children_slice())),
        _ => None,
    }
}

fn deep_symbol(expr: &chelis_deep::ast::Expr) -> Option<&str> {
    match expr {
        chelis_deep::ast::Expr::Atom(chelis_deep::ast::Atom::Name(name), _) => Some(name),
        _ => None,
    }
}

/// Return the first top-level declaration binding name that matches the
/// reef linker's reserved internal-name format (RFC v5), or `None`.
fn first_forged_linker_name(exprs: &[chelis_deep::ast::Expr]) -> Option<String> {
    fn binding_name(expr: &chelis_deep::ast::Expr) -> Option<&str> {
        let (tag, children) = deep_node_parts(expr)?;
        let is_binding_decl = matches!(
            tag,
            DeepTag::Deftype | DeepTag::Def | DeepTag::Defsig | DeepTag::Typealias
        );
        if !is_binding_decl {
            return None;
        }
        children.first().and_then(deep_symbol)
    }
    fn walk(expr: &chelis_deep::ast::Expr) -> Option<String> {
        let (tag, children) = deep_node_parts(expr)?;
        if tag == DeepTag::Module {
            // Descend into a module wrapper's children.
            for child in children.iter().skip(1) {
                if let Some(found) = walk(child) {
                    return Some(found);
                }
            }
            return None;
        }
        if let Some(name) = binding_name(expr)
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
    fn module_name(expr: &chelis_deep::ast::Expr) -> Option<&str> {
        let (tag, children) = deep_node_parts(expr)?;
        if tag != DeepTag::Module {
            return None;
        }
        children.first().and_then(deep_symbol)
    }
    fn walk(
        expr: &chelis_deep::ast::Expr,
        prefix: Option<&str>,
        seen: &mut chelis_unord::UnordSet<String>,
    ) -> Option<String> {
        let (_, children) = deep_node_parts(expr)?;
        let name = module_name(expr)?;
        let key = match prefix {
            Some(p) => format!("{p}.{name}"),
            None => name.to_string(),
        };
        if !seen.insert(key.clone()) {
            return Some(key);
        }
        for child in children.iter().skip(1) {
            if let Some(dup) = walk(child, Some(&key), seen) {
                return Some(dup);
            }
        }
        None
    }
    let mut seen = chelis_unord::UnordSet::new();
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
    fn walk(
        expr: &chelis_deep::ast::Expr,
        seen: &mut chelis_unord::UnordSet<String>,
    ) -> Option<String> {
        let (tag, children) = deep_node_parts(expr)?;
        match tag {
            DeepTag::Module => children.iter().skip(1).find_map(|child| walk(child, seen)),
            DeepTag::Defsig => {
                let name = children.first().and_then(deep_symbol)?;
                if seen.insert(name.to_string()) {
                    None
                } else {
                    Some(name.to_string())
                }
            }
            _ => None,
        }
    }

    let mut seen = chelis_unord::UnordSet::new();
    exprs.iter().find_map(|expr| walk(expr, &mut seen))
}

pub fn validate_desugared(source: &str) -> Result<(), ValidationError> {
    let decls = chelis_surf::parser::parse_str(source)
        .map_err(|err| ValidationError::Failed(format!("compiler parse failed: {err}")))?;
    let deep = chelis_surf::desugar::desugar_program(&decls)
        .map_err(|err| ValidationError::Failed(format!("desugaring failed: {err}")))?;
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

/// Private to the text ingress above: all spans come from this exact source.
/// Replace opaque values only in the auxiliary validator's view, preserving
/// every byte offset and newline. The returned AST and source remain untouched.
fn deep_program_projection(source: &str, exprs: &[chelis_deep::Expr]) -> String {
    use chelis_deep::{Expr, Metadata};
    fn metadata(meta: &Metadata, bytes: &mut [u8]) {
        for (_, data) in meta.extensions().iter() {
            let span = data.span();
            let value = &mut bytes[span.offset..span.end()];
            for byte in value.iter_mut() {
                if !matches!(*byte, b'\n' | b'\r') {
                    *byte = b' ';
                }
            }
            value[0] = b'0';
        }
        meta.visit_syntax(&mut |_, value| visit(value, bytes));
    }
    fn visit(expr: &Expr, bytes: &mut [u8]) {
        match expr {
            Expr::Node(node, _) => {
                metadata(node.meta(), bytes);
                for child in node.children_slice() {
                    visit(child, bytes);
                }
            }
            Expr::BareList(items, _) => {
                for child in items {
                    visit(child, bytes);
                }
            }
            Expr::Map(meta, _) => metadata(meta, bytes),
            Expr::MetaExpr(meta, _) => {
                metadata(&meta.metadata, bytes);
                visit(&meta.expr, bytes);
            }
            Expr::UnknownForm(data) => {
                metadata(&data.meta, bytes);
                for child in &data.children {
                    visit(child, bytes);
                }
            }
            Expr::Atom(..) => {}
        }
    }
    let mut bytes = source.as_bytes().to_vec();
    for expr in exprs {
        visit(expr, &mut bytes);
    }
    String::from_utf8(bytes).expect("whole lexical values replaced with ASCII")
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
            deep::Rule::bare_list => validate_bare_list(child.clone())?,
            deep::Rule::unit_list | deep::Rule::literal | deep::Rule::bare_name => {}
            deep::Rule::wildcard if deep_tag == DeepTag::DName => {}
            deep::Rule::wildcard => {
                return Err(ValidationError::Failed(format!(
                    "Deep wildcard `*` at byte {} is only valid as the sole child of `d-name`",
                    child.as_span().start()
                )));
            }
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

fn validate_bare_list(pair: Pair<'_, deep::Rule>) -> Result<(), ValidationError> {
    for item in pair.into_inner().filter(is_structural_pair) {
        let child = match item.as_rule() {
            deep::Rule::bare_list_item => item.into_inner().find(is_structural_pair),
            _ => Some(item),
        };
        let Some(child) = child else {
            continue;
        };
        match child.as_rule() {
            deep::Rule::node => validate_deep_node(child)?,
            deep::Rule::typed_helper => validate_typed_helper(child)?,
            deep::Rule::unit_list | deep::Rule::literal | deep::Rule::bare_name => {}
            other => {
                return Err(ValidationError::Failed(format!(
                    "unexpected Deep bare-list rule {:?} at byte {}",
                    other,
                    child.as_span().start()
                )));
            }
        }
    }
    Ok(())
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
        DeepTag::DName => {
            if child_count != 1 {
                return Err(wrong_arity("exactly 1 name or wildcard child"));
            }
            if matches!(
                children[0].as_rule(),
                deep::Rule::bare_name | deep::Rule::wildcard
            ) {
                Ok(())
            } else {
                Err(ValidationError::Failed(format!(
                    "Deep tag `d-name` at byte {offset} expects one name or wildcard child"
                )))
            }
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
                chelis_deep::parse_and_stamp_file(&source).is_err(),
                "`{bogus}` must be rejected by the compiler's stamped ingress"
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
        let source = "(defsig {} f (r) (t-fn {} (t-ref {} (t-tensor {} (d-rank {} r) (t-prim {} f32))) (t-tensor {} (d-rank {} r) (t-prim {} f32))))\n";
        validate_deep(source)
            .expect("validator should accept canonical t-ref / d-rank rank-polymorphic Deep");
    }

    #[test]
    fn deep_accepts_the_explicit_d_name_wildcard() {
        // spec/03-deep-syntax.md §2.6: `*` is the explicit wildcard
        // spelling for the sole symbol child of `d-name`.
        let source = "(defsig {} f (t-tensor {} (d-name {} *) (t-prim {} f32)))\n";
        chelis_deep::parse_and_stamp_file(source)
            .expect("the stamped compiler ingress accepts the wildcard dimension");
        validate_deep(source).expect("the independent grammar must accept it too");
    }

    #[test]
    fn deep_rejects_wildcards_outside_d_name() {
        // The wildcard exception is a child-role rule, not an identifier
        // spelling. In particular it must not become a variable, dimension
        // variable, rank variable, primitive, binder, or declaration name.
        for source in [
            "(def {} f (var {} *))\n",
            "(defsig {} f (t-tensor {} (d-var {} *) (t-prim {} f32)))\n",
            "(defsig {} f (t-tensor {} (d-rank {} *) (t-prim {} f32)))\n",
            "(defsig {} f (t-prim {} *))\n",
            "(def {} * (lit {} 1))\n",
        ] {
            let error = validate_deep(source)
                .expect_err("`*` outside the sole child of `d-name` must be rejected");
            assert!(
                error
                    .to_string()
                    .contains("only valid as the sole child of `d-name`"),
                "wrong reason for {source:?}: {error}"
            );
        }
    }

    #[test]
    fn deep_rejects_malformed_wildcard_dimensions() {
        for source in [
            "(defsig {} f (t-tensor {} (d-name {}) (t-prim {} f32)))\n",
            "(defsig {} f (t-tensor {} (d-name {} * batch) (t-prim {} f32)))\n",
            "(defsig {} f (t-tensor {} (d-name {} **) (t-prim {} f32)))\n",
            "(defsig {} f (t-tensor {} (d-name {} *foo) (t-prim {} f32)))\n",
        ] {
            assert!(
                validate_deep(source).is_err(),
                "a wildcard dimension must have exactly one standalone `*` child: {source}"
            );
        }
    }

    #[test]
    fn deep_accepts_canonical_nominal_parameter_lists() {
        let source = "(deftype {} Column (n a) (variant {} Column (field {} items (t-tensor {} (d-var {} n) (t-var {} a)))))\n";
        validate_deep(source)
            .expect("validator must accept the compiler's canonical bare nominal-parameter list");
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
    fn surf_accepts_newline_block_semicolon_par_and_axis_identifier() {
        let source = "def f(axis) = {\n  y = axis\n  y\n}\ndef g() = par { a; b }\n";
        validate_surf(source)
            .expect("validator should accept canonical block/par separators and axis identifiers");
    }

    #[test]
    fn surf_accepts_arrow_return_types() {
        let source = "def f[n](x: tensor[n, f32]) -> tensor[n, f32] = relu(x)\n";
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
        // chelis#1088: the fixture moved inside a declaration. A top-level
        // `(mystery {} x)` is now a [03-PROG-1] rejection, which would make
        // this test pass for the wrong reason; below a `def` the unknown
        // head is what the validator is left to decide.
        let source = "(def {} f (mystery {} x))";
        let error = validate_deep(source).expect_err("unknown tag should fail");
        let rendered = error.to_string();
        assert!(rendered.contains("unknown tag"), "{rendered}");
        assert!(rendered.contains("mystery"), "{rendered}");
    }

    #[test]
    fn deep_accepts_snake_case_metadata_keys() {
        let source = "(def {c_earchin_role: \"property_witness\"} req_PRC_001 (fn {} (params {}) (lit {type: (t-prim {} bool)} true)))";
        validate_deep(source).expect("producer metadata keys may be snake_case");
    }

    #[test]
    fn deep_metadata_preserves_arbitrary_macro_argument_syntax() {
        let source = "(def {source: (macro_name {surf_future: 1, span: 2} ((original_name) ^{:type f32} x))} f (lit {} 1))";
        validate_deep(source).unwrap();
        validate_deep("(def {source: 1} f (lit {} 1))").unwrap_err();
    }

    #[test]
    fn opaque_data_grammar_is_owned_by_its_parser() {
        for payload in [
            "1e-3f32",
            "{type: false type: (var {}),}",
            "^{:span 7} (missing_macro x)",
            "(a-b \"λ\" (lit {} 7i8))",
        ] {
            let source = format!("(def {{tool_data: {payload}}} f (lit {{}} 1))");
            validate_deep(&source).unwrap_or_else(|error| panic!("{source}: {error}"));
        }
        validate_deep("(def {property_quantifiers: (params {tool_data: 7f32})} f (lit {} 1))")
            .unwrap();
        for source in [
            "(def {tool_data: {broken:}} f (lit {} 1))",
            "(def {type: false} f (lit {} 1))",
            "(def {tool_data: 1f32} f (var {} x y))",
        ] {
            assert!(validate_deep(source).is_err(), "{source}");
        }
    }

    #[test]
    fn deep_rejects_invalid_effects_children() {
        // chelis#1088: `(effects ...)` is only ever a metadata value in real
        // Deep, never a top-level form, so the fixture now sits where it
        // actually occurs. That position is invisible to the Pest leg, which
        // walks node children; the AST-side sweep is what reaches it, and
        // running that sweep here is what keeps `validate --deep` agreeing
        // with `check` below the top level too.
        let source = "(defsig {} f (t-fn {eff: (effects {} 1)} (t-prim {} f32)))";
        let error = validate_deep(source).expect_err("non-symbol effects child should fail");
        let rendered = error.to_string();
        assert!(
            rendered.contains("metadata `eff`") && rendered.contains("effects node"),
            "{rendered}"
        );
    }

    #[test]
    fn deep_rejects_invalid_resource_arity() {
        // chelis#1088: likewise nested where a `resource` entry really
        // appears. The stamped ingress reaches the arity first and says so
        // in the node vocabulary's own terms; the Pest arity arm remains as
        // the second line of defence.
        let source = "(defsig {} f (t-fn {eff: (effects {} (resource {} x y))} (t-prim {} f32)))";
        let error = validate_deep(source).expect_err("resource arity should fail");
        let rendered = error.to_string();
        assert!(rendered.contains("resource"), "{rendered}");
        assert!(rendered.contains("metadata `eff`"), "{rendered}");
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
    // surfaces stay in parity with `chelis_deep::parse_and_stamp_file`
    // (chelis#1088: that is now the one Deep ingress both surfaces use).

    /// The base program these comment cases wrap, comment-free.
    const BASE: &str = "(module {} hello)\n";

    fn assert_validates(source: &str, label: &str) {
        // Both the commented and the bare form must validate, and the
        // stamped hand-rolled ingress must also accept the source, so the
        // two Deep surfaces agree (the core complaint of #167).
        validate_deep(source).unwrap_or_else(|err| panic!("{label} should validate: {err}"));
        validate_deep(BASE).expect("base program should validate");
        chelis_deep::parse_and_stamp_file(source)
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
    // wrongly rejects source that the stamped ingress accepts.

    #[test]
    fn deep_accepts_comment_before_params_tag_in_fn() {
        assert_validates(
            "(def {} f (fn {} (; note\nparams {}) (var {} x)))\n",
            "comment before nested `params` tag",
        );
    }

    #[test]
    fn deep_accepts_comment_before_bind_tag_in_let() {
        assert_validates(
            "(def {} f (let {} (; note\nbind {}) (var {} x)))\n",
            "comment before nested `bind` tag",
        );
    }

    #[test]
    fn deep_accepts_comment_before_resource_tag_in_effects() {
        assert_validates(
            "(defsig {} f (t-fn {eff: (effects {} (; note\nresource {} \"foo\"))} (t-prim {} f32)))\n",
            "comment before nested `resource` tag",
        );
    }

    // ---- ingress parity with `chelis check` (chelis#1088) ----------------

    #[test]
    fn deep_rejects_a_program_with_no_top_level_form() {
        // [03-PROG-3]. Empty, whitespace-only, and comments-only text all
        // yield zero forms, and all three are the same rejection.
        for source in ["", "   \n\t\n", "; just a comment\n", "\n; a\n; b\n\n"] {
            let error = validate_deep(source)
                .expect_err("[03-PROG-3] rejects text yielding no top-level form");
            assert!(
                error.to_string().contains("empty program"),
                "{source:?}: {error}"
            );
            assert!(
                chelis_deep::parse_and_stamp_file(source).is_err(),
                "the compiler's stamped ingress must agree about {source:?}"
            );
        }
    }

    #[test]
    fn deep_identifies_every_headless_class_the_way_check_does() {
        // chelis#1088: the Pest grammar admits only `node+`, so before the
        // stamped ingress ran first a headless top-level form died as
        // `expected program` and the reader never learned its class. These
        // are the nine [03-PROG-2] classes, through `validate --deep`.
        let cases: [(&str, &str); 9] = [
            ("some_name", "a bare identifier"),
            ("42", "a bare integer literal"),
            ("1.5", "a bare float literal"),
            ("\"text\"", "a bare string literal"),
            ("true", "a bare boolean literal"),
            ("()", "an empty list"),
            ("((var {} f) (var {} x))", "a list without a tag symbol"),
            ("{key: 1}", "a metadata map"),
            (
                "^{:surf_literal_style \"explicit\"} (var {} x)",
                "a metadata-annotated form",
            ),
        ];
        for (source, identification) in cases {
            let error = validate_deep(source)
                .expect_err("[03-PROG-1] rejects every headless top-level form");
            let rendered = error.to_string();
            assert!(
                rendered.contains(identification),
                "[03-PROG-2] requires {source:?} to be identified as \
                 {identification}, got: {rendered}"
            );
            assert!(
                !rendered.contains('<') && !rendered.contains('>'),
                "[03-PROG-2] forbids a placeholder identification: {rendered}"
            );
        }
    }

    #[test]
    fn deep_rejects_a_top_level_form_that_is_not_a_declaration() {
        // The pest grammar admits any top-level tagged node, and the AST leg
        // used to skip its checks silently whenever its own parse failed.
        // Both legs now agree with `chelis check`: a top-level form is a
        // `(module ...)` wrapper or a declaration.
        for source in [
            "(fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x))\n",
            "(var {} x)\n",
            "(app {} (var {} f) (var {} x))\n",
        ] {
            let error =
                validate_deep(source).expect_err("a top-level non-declaration must be rejected");
            assert!(
                error.to_string().contains("expected declaration"),
                "wrong reason for `{source}`: {error}"
            );
            assert!(
                chelis_deep::parse_and_stamp_file(source).is_err(),
                "the compiler's stamped ingress must agree about `{source}`"
            );
        }
    }

    #[test]
    fn deep_accepts_both_module_wrappers_and_bare_declarations() {
        // The positive control for the rule above: the two admissible
        // top-level shapes still validate, and the compiler's stamped
        // ingress agrees.
        for source in [
            "(module {} hello (def {} f (var {} x)))\n",
            "(def {} f (var {} x))\n",
            "(defsig {} f (t-fn {eff: (effects {})} (t-prim {} f32)))\n",
        ] {
            validate_deep(source).unwrap_or_else(|err| panic!("`{source}` should validate: {err}"));
            chelis_deep::parse_and_stamp_file(source)
                .unwrap_or_else(|err| panic!("`{source}` should stamp: {err}"));
        }
    }

    #[test]
    fn deep_reports_a_forged_linker_name_it_used_to_skip_silently() {
        // The forgery checks used to run only when the AST leg's own parse
        // succeeded, so a source that failed it passed unreported. Now the
        // parse failure is itself a rejection, and a forged name in a source
        // that does stamp is still named.
        let error = validate_deep("(def {} Pkg__p__M__forged (var {} x))\n")
            .expect_err("a reserved linker-format name must be rejected");
        assert!(
            error.to_string().contains("reserved internal-name"),
            "wrong reason: {error}"
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

    /// chelis#1417: `dtype_bounds` is the first map-valued metadata key, so
    /// this grammar is the second implementation that has to admit it.
    /// Reverting `meta_value`'s `meta` alternative turns this RED with
    /// `expected meta_value`, which is exactly how every migrated stdlib
    /// module failed `chelis deep <file> | chelis validate --deep`.
    #[test]
    fn deep_admits_a_map_valued_metadata_key() {
        for source in [
            "(defsig {dtype_bounds: {p: int}} arange (p) (t-fn {} (t-var {} p) (t-var {} p)))\n",
            "(defsig {dtype_bounds: {p: float, q: numeric}} f (p q) (t-fn {} (t-var {} p) (t-var {} q)))\n",
            // The empty map is a legal value, as it is a legal node meta.
            "(defsig {dtype_bounds: {}} f (p) (t-fn {} (t-var {} p) (t-var {} p)))\n",
        ] {
            validate_deep(source)
                .unwrap_or_else(|e| panic!("map-valued metadata must validate: {source}\n{e}"));
        }
    }

    /// The Surf half of the same second-implementation contract. `validate_surf`
    /// has a "grammar rejects, parser accepts" rescue, so without these the new
    /// binder syntax would pass through that hole and the grammar would cover
    /// none of it. Reverting `type_binders` in `surf.pest` does not change the
    /// exit verdict, so these assert on the GRAMMAR directly.
    #[test]
    fn surf_grammar_admits_the_dtype_family_binder_list() {
        use pest::Parser;
        for source in [
            "sig arange[n, p: Int]: p -> p -> tensor[n, p]\n",
            "sig total[p: Numeric]: p -> p -> p\n",
            "def only_floats[p: Float](x: p) -> p = x\n",
            "def scale[n, p: Float](x: tensor[n, p]) -> tensor[n, p] = x\n",
            "def unbounded[a](x: a) -> a = x\n",
        ] {
            super::surf::Grammar::parse(super::surf::Rule::program, source)
                .unwrap_or_else(|e| panic!("the Surf grammar must admit `{source}`: {e}"));
            validate_surf(source).expect("and the compiler parser agrees");
        }
    }

    /// The negative half: the bound position is a closed three-name set in the
    /// grammar too, so an ADT name there is not quietly admitted.
    #[test]
    fn surf_grammar_rejects_a_non_family_bound() {
        use pest::Parser;
        for source in ["sig f[p: Tensor]: p -> p\n", "sig f[p: f32]: p -> p\n"] {
            assert!(
                super::surf::Grammar::parse(super::surf::Rule::program, source).is_err(),
                "the Surf grammar must reject `{source}`"
            );
            validate_surf(source).expect_err("and the compiler parser rejects it too");
        }
    }

    /// §1.1 declares the metadata key charset as `[A-Za-z_][A-Za-z0-9_]*`.
    /// `dtype_bounds` needs the underscore; §7's PEG and this grammar are the
    /// two implementations that have to agree with that sentence.
    #[test]
    fn deep_admits_the_declared_metadata_key_charset() {
        for (key, source) in [
            (
                "dtype_bounds",
                "(defsig {dtype_bounds: {p: float}} f (p) (t-var {} p))",
            ),
            (
                "chelis_role",
                "(def {chelis_role: \"custom\"} f (lit {} 1))",
            ),
            ("surf_path", "(module {surf_path: \"M.Path\"} m.path)"),
            ("_leading", "(def {_leading: x} f (lit {} 1))"),
            ("Upper", "(def {Upper: x} f (lit {} 1))"),
        ] {
            validate_deep(source)
                .unwrap_or_else(|e| panic!("`{key}` is a legal metadata key: {e}"));
        }
        // The no-hyphen rule that keeps Deep symbols portable still holds.
        validate_deep("(defsig {has-hyphen: x} f (p) (t-var {} p))\n")
            .expect_err("a hyphenated metadata key must still be rejected");
    }

    /// The whole path the red team exercised: desugar a bounded declaration
    /// the way `chelis deep` does, print it, and validate the result. This is
    /// the control whose absence let the grammar regression reach a green
    /// gate — nothing else runs `validate --deep` over Deep carrying a
    /// map-valued metadata key.
    #[test]
    fn a_bounded_declaration_survives_desugar_then_deep_validation() {
        let source = "module Std.Planted\n\
                      export (planted_pick)\n\
                      sig planted_pick[p: Numeric]: p -> p -> p\n\
                      def planted_pick(a, b) = a\n";
        let decls = chelis_surf::parser::parse_str(source).expect("bounded Surf parses");
        let printed = chelis_deep::printer::print_canonical(
            &chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar"),
        );
        assert!(
            printed.contains("dtype_bounds: {p: numeric}"),
            "the fixture must actually carry a map-valued key: {printed}"
        );
        validate_deep(&printed)
            .unwrap_or_else(|e| panic!("desugared bounded Deep must validate:\n{printed}\n{e}"));
    }
}
