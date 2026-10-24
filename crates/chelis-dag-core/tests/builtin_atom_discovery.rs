//! Syntax checks enforce the closed semantic inventory and explicit match arms.
use chelis_dag_core::dag::RiscAtomIdentity;
use std::collections::BTreeSet;
use syn::visit::Visit;

fn check_inventory(
    source: &str,
    actual: &[String],
    canonical: &[&str],
) -> Result<(), &'static str> {
    let source = syn::parse_file(source).expect("valid Rust");
    struct Identities {
        declarations: usize,
        names: BTreeSet<String>,
    }
    impl<'ast> Visit<'ast> for Identities {
        fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
            if item.ident == "RiscAtomIdentity" {
                self.declarations += 1;
                self.names = item.variants.iter().map(|v| v.ident.to_string()).collect();
            }
            syn::visit::visit_item_enum(self, item);
        }
    }
    let mut declared = Identities {
        declarations: 0,
        names: BTreeSet::new(),
    };
    declared.visit_file(&source);
    if declared.declarations != 1 {
        return Err("expected one closed semantic enum");
    }
    let unique: BTreeSet<_> = actual.iter().cloned().collect();
    if unique.len() != actual.len() {
        return Err("duplicate identity");
    }
    if unique != declared.names {
        return Err("enum extension missing from discovery");
    }
    if canonical.len() != actual.len()
        || canonical.iter().copied().collect::<BTreeSet<_>>().len() != actual.len()
    {
        return Err("canonical identity collision");
    }
    Ok(())
}

fn explicit_variant(pat: &syn::Pat) -> bool {
    match pat {
        syn::Pat::Path(_) | syn::Pat::Struct(_) | syn::Pat::TupleStruct(_) => true,
        syn::Pat::Or(pat) => pat.cases.iter().all(explicit_variant),
        syn::Pat::Paren(pat) => explicit_variant(&pat.pat),
        syn::Pat::Reference(pat) => explicit_variant(&pat.pat),
        _ => false,
    }
}

fn check_disposition(source: &str) -> Result<(), &'static str> {
    let source = syn::parse_file(source).expect("valid Rust");
    #[derive(Default)]
    struct Arms {
        matches: usize,
        error: Option<&'static str>,
    }
    impl<'ast> Visit<'ast> for Arms {
        fn visit_expr_match(&mut self, item: &'ast syn::ExprMatch) {
            self.matches += 1;
            syn::visit::visit_expr_match(self, item);
        }

        fn visit_arm(&mut self, arm: &'ast syn::Arm) {
            if !explicit_variant(&arm.pat) {
                self.error = Some("unclassified RISC variant fallback");
            }
            if arm.guard.is_some() {
                self.error = Some("guarded disposition is not exhaustive");
            }
            syn::visit::visit_arm(self, arm);
        }

        fn visit_pat_wild(&mut self, _: &'ast syn::PatWild) {
            self.error = Some("unclassified RISC variant wildcard");
        }
    }
    #[derive(Default)]
    struct Disposition {
        methods: usize,
        arms: Arms,
    }
    impl<'ast> Visit<'ast> for Disposition {
        fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
            if item.trait_.is_none()
                && matches!(item.self_ty.as_ref(), syn::Type::Path(ty)
                    if ty.path.is_ident("RiscOp"))
            {
                for member in &item.items {
                    if let syn::ImplItem::Fn(method) = member
                        && method.sig.ident == "atom_disposition"
                    {
                        self.methods += 1;
                        self.arms.visit_block(&method.block);
                    }
                }
            }
            syn::visit::visit_item_impl(self, item);
        }
    }
    let mut disposition = Disposition::default();
    disposition.visit_file(&source);
    if disposition.methods != 1 || disposition.arms.matches == 0 {
        return Err("expected one semantic disposition with explicit match arms");
    }
    match disposition.arms.error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

#[test]
fn semantic_identity_enumeration_matches_the_closed_enum() {
    let actual: Vec<_> = RiscAtomIdentity::ALL
        .iter()
        .map(|id| format!("{id:?}"))
        .collect();
    let canonical: Vec<_> = RiscAtomIdentity::ALL.iter().map(|id| id.as_str()).collect();
    check_inventory(include_str!("../src/dag.rs"), &actual, &canonical)
        .expect("complete closed semantic inventory");
}

#[test]
fn risc_disposition_has_no_wildcard_or_fallback() {
    check_disposition(include_str!("../src/dag.rs")).expect("explicit exhaustive disposition");
}

#[test]
fn builtin_atom_discovery_rejects_an_enum_variant_absent_from_all() {
    let source = "enum RiscAtomIdentity { Known, Added }";
    assert_eq!(
        check_inventory(source, &["Known".into()], &["known"]),
        Err("enum extension missing from discovery")
    );
}

#[test]
fn builtin_atom_discovery_rejects_duplicate_and_colliding_identities() {
    let source = "enum RiscAtomIdentity { First, Second }";
    assert_eq!(
        check_inventory(
            source,
            &["First".into(), "First".into()],
            &["first", "first"]
        ),
        Err("duplicate identity")
    );
    assert_eq!(
        check_inventory(
            source,
            &["First".into(), "Second".into()],
            &["same", "same"]
        ),
        Err("canonical identity collision")
    );
    assert_eq!(
        check_inventory(
            source,
            &["First".into(), "Second".into()],
            &["first", "second"]
        ),
        Ok(())
    );
}

fn disposition_source(arm: &str) -> String {
    format!("impl RiscOp {{ fn atom_disposition(&self) {{ match self {{ {arm} => () }} }} }}")
}

#[test]
fn builtin_atom_discovery_rejects_wildcards_catch_all_bindings_and_guards() {
    for arm in [
        "_",
        "other",
        "Self::Known | _",
        "Self::Known | other",
        "whole @ Self::Known",
        "Self::Known if true",
        "Self::Payload { value: _ }",
    ] {
        assert!(
            check_disposition(&disposition_source(arm)).is_err(),
            "accepted {arm}"
        );
    }
    for arm in [
        "Self::Known",
        "Self::Known | Self::Other",
        "Self::Payload { .. }",
    ] {
        assert_eq!(check_disposition(&disposition_source(arm)), Ok(()));
    }
}

#[test]
fn builtin_atom_discovery_requires_the_actual_disposition_method() {
    for source in [
        "impl RiscOp { fn another_method(&self) { match self { _ => () } } }",
        "impl Other { fn atom_disposition(&self) { match self { _ => () } } }",
        "impl RiscOp { fn atom_disposition(&self) {} }",
    ] {
        assert!(check_disposition(source).is_err());
    }
}
