const PREFIX: &str = "test_unchecked_metadata_";

fn masked_metadata_source(source: &str) -> String {
    use chelis_deep::lexer::{self, TokenKind};
    let tokens = lexer::lex(source).expect("fixture lexes");
    let mut masked = source.to_string();
    for (i, token) in tokens.iter().enumerate().rev() {
        let (key, keyword) = match &token.kind {
            TokenKind::Symbol(key) if matches!(tokens.get(i + 1).map(|t| &t.kind), Some(TokenKind::Symbol(colon)) if colon == ":") => {
                (key, false)
            }
            TokenKind::Keyword(key) => (key, true),
            _ => continue,
        };
        if chelis_deep::metadata::REGISTERED_METADATA_KEYS.contains(&key.as_str())
            || key.starts_with("surf_")
        {
            let offset = token.span.offset + usize::from(keyword);
            masked.insert_str(offset, PREFIX);
        }
    }
    masked
}

// Test-only construction of ungated legacy metadata carriers. Hide annotation
// keys while using the ordinary syntax parser, then restore them ONLY on List,
// Map and MetaExpr carriers. This keeps consumer-defense tests executable after
// public parsing and Node construction reject the same malformed metadata.
// No production parser or constructor gets an unchecked escape hatch.
#[allow(dead_code)]
pub fn legacy_metadata_fixture(source: &str) -> Vec<chelis_deep::Expr> {
    use chelis_deep::{Atom, Expr, List, MetaExpr, MetaMap};
    let masked = masked_metadata_source(source);
    fn meta(entries: Vec<(String, Expr)>) -> MetaMap {
        MetaMap {
            entries: entries
                .into_iter()
                .map(|(k, v)| (k.strip_prefix(PREFIX).unwrap_or(&k).to_string(), restore(v)))
                .collect(),
        }
    }
    fn restore(expr: Expr) -> Expr {
        match expr {
            Expr::Node(node, span) => {
                let mut elements = vec![
                    Expr::Atom(Atom::Tag(node.tag()), span),
                    Expr::Map(meta(node.meta().entries.clone()), span),
                ];
                elements.extend(node.children_slice().iter().cloned().map(restore));
                Expr::List(List { elements }, span)
            }
            Expr::List(list, span) => Expr::List(
                List {
                    elements: list.elements.into_iter().map(restore).collect(),
                },
                span,
            ),
            Expr::Map(map, span) => Expr::Map(meta(map.entries), span),
            Expr::MetaExpr(m, span) => Expr::MetaExpr(
                MetaExpr {
                    entries: meta(m.entries).entries,
                    expr: Box::new(restore(*m.expr)),
                },
                span,
            ),
            Expr::BareList(v, span) => Expr::BareList(v.into_iter().map(restore).collect(), span),
            Expr::UnknownForm(mut v) => {
                v.meta = meta(v.meta.entries);
                v.children = v.children.into_iter().map(restore).collect();
                Expr::UnknownForm(v)
            }
            atom @ Expr::Atom(..) => atom,
        }
    }
    chelis_deep::parser::parse_str(&masked)
        .expect("masked metadata fixture parses")
        .into_iter()
        .map(restore)
        .collect()
}

// The arity/role attack fixtures also need raw syntax before Node stamping.
#[allow(dead_code)]
pub fn raw_metadata_fixture(source: &str) -> Vec<chelis_deep::RawExpr> {
    use chelis_deep::RawExpr;
    fn restore(v: &mut RawExpr) {
        match v {
            RawExpr::Map(entries, _) | RawExpr::MetaExpr { entries, .. } => {
                for (key, value) in entries {
                    if let Some(original) = key.strip_prefix(PREFIX) {
                        *key = original.to_string();
                    }
                    restore(value);
                }
            }
            RawExpr::List(items, _) => items.iter_mut().for_each(restore),
            RawExpr::Atom(..) => {}
        }
        if let RawExpr::MetaExpr { expr, .. } = v {
            restore(expr);
        }
    }
    let mut raw = chelis_deep::parse_raw_str(&masked_metadata_source(source))
        .expect("masked raw syntax parses");
    raw.iter_mut().for_each(restore);
    raw
}
