//! Exhaustiveness is checked from Rust syntax as well as compiled values.
use chelis_ir::dag::RiscAtomIdentity;
use std::collections::BTreeSet;
use syn::visit::Visit;

#[test]
fn semantic_identity_enumeration_matches_the_closed_enum() {
    let source = syn::parse_file(include_str!("../src/dag.rs")).expect("valid Rust");
    let names: BTreeSet<_> = source
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Enum(e) if e.ident == "RiscAtomIdentity" => {
                Some(e.variants.iter().map(|v| v.ident.to_string()).collect())
            }
            _ => None,
        })
        .expect("closed semantic enum");
    let actual: BTreeSet<_> = RiscAtomIdentity::ALL
        .iter()
        .map(|v| format!("{v:?}"))
        .collect();
    assert_eq!(
        actual.len(),
        RiscAtomIdentity::ALL.len(),
        "duplicate identity"
    );
    assert_eq!(actual, names, "enum extension missing from discovery");
    let canonical: BTreeSet<_> = RiscAtomIdentity::ALL.iter().map(|v| v.as_str()).collect();
    assert_eq!(
        canonical.len(),
        actual.len(),
        "canonical identity collision"
    );
}

#[test]
fn risc_disposition_has_no_wildcard_or_fallback() {
    struct NoWildcard {
        seen: bool,
    }
    impl<'a> Visit<'a> for NoWildcard {
        fn visit_impl_item_fn(&mut self, item: &'a syn::ImplItemFn) {
            if item.sig.ident != "atom_disposition" {
                return;
            }
            self.seen = true;
            struct Arms;
            impl<'b> Visit<'b> for Arms {
                fn visit_pat_wild(&mut self, _: &'b syn::PatWild) {
                    panic!("unclassified RISC variant fallback, including inside an or-pattern");
                }
                fn visit_arm(&mut self, arm: &'b syn::Arm) {
                    assert!(
                        !matches!(arm.pat, syn::Pat::Wild(_) | syn::Pat::Ident(_)),
                        "unclassified RISC variant fallback"
                    );
                    assert!(arm.guard.is_none(), "guarded disposition is not exhaustive");
                    syn::visit::visit_arm(self, arm);
                }
            }
            Arms.visit_block(&item.block);
        }
    }
    let source = syn::parse_file(include_str!("../src/dag.rs")).expect("valid Rust");
    let mut visitor = NoWildcard { seen: false };
    visitor.visit_file(&source);
    assert!(visitor.seen);
}
