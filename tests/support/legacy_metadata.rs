// Raw-only attack fixtures. No helper can manufacture invalid typed metadata.
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
        let offset = token.span.offset + usize::from(keyword);
        let _ = key;
        masked.insert_str(offset, &format!("{PREFIX}{i}_"));
    }
    masked
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
                        *key = original
                            .split_once('_')
                            .expect("indexed masked key")
                            .1
                            .to_string();
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
