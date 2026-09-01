use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use syn::{ImplItem, Item, Type};

fn public_methods(type_name: &str) -> BTreeSet<String> {
    let source = fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"))
        .expect("read chelis-unord source");
    let syntax = syn::parse_file(&source).expect("parse chelis-unord source");
    syntax
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Impl(item_impl) => Some(item_impl),
            _ => None,
        })
        .filter(|item_impl| item_impl.trait_.is_none())
        .filter(|item_impl| match item_impl.self_ty.as_ref() {
            Type::Path(path) => path
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == type_name),
            _ => false,
        })
        .flat_map(|item_impl| item_impl.items.iter())
        .filter_map(|item| match item {
            ImplItem::Fn(method) if matches!(method.vis, syn::Visibility::Public(_)) => {
                Some(method.sig.ident.to_string())
            }
            _ => None,
        })
        .collect()
}

fn public_traits(type_name: &str) -> BTreeSet<String> {
    let source = fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"))
        .expect("read chelis-unord source");
    let syntax = syn::parse_file(&source).expect("parse chelis-unord source");
    syntax
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Impl(item_impl) => Some(item_impl),
            _ => None,
        })
        .filter(|item_impl| match item_impl.self_ty.as_ref() {
            Type::Path(path) => path
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == type_name),
            _ => false,
        })
        .filter_map(|item_impl| {
            item_impl
                .trait_
                .as_ref()
                .and_then(|(_, path, _)| path.segments.last())
                .map(|segment| segment.ident.to_string())
        })
        .chain(derived_traits(&syntax, type_name))
        .collect()
}

/// Traits the type gets from a `#[derive(...)]`, which are as public as an
/// explicit `impl` and must be pinned the same way.
fn derived_traits(syntax: &syn::File, type_name: &str) -> BTreeSet<String> {
    let mut derived = BTreeSet::new();
    for item in &syntax.items {
        let Item::Struct(item) = item else { continue };
        if item.ident != type_name {
            continue;
        }
        for attribute in &item.attrs {
            if !attribute.path().is_ident("derive") {
                continue;
            }
            attribute
                .parse_nested_meta(|meta| {
                    if let Some(segment) = meta.path.segments.last() {
                        derived.insert(segment.ident.to_string());
                    }
                    Ok(())
                })
                .expect("parse derive list");
        }
    }
    derived
}

fn path_type<'a>(ty: &'a Type, expected: &str) -> &'a syn::TypePath {
    let Type::Path(path) = ty else {
        panic!("expected {expected} to be a path type");
    };
    assert_eq!(
        path.path.segments.last().unwrap().ident,
        expected,
        "unexpected path type"
    );
    path
}

#[test]
fn public_inherent_api_is_exact() {
    let expected = BTreeMap::from([
        (
            "UnordMap",
            BTreeSet::from(
                [
                    "clear",
                    "contains_key",
                    "entry",
                    "get",
                    "get_key_value",
                    "get_mut",
                    "insert",
                    "into_sorted",
                    "is_empty",
                    "len",
                    "merge",
                    "new",
                    "remove",
                    "to_sorted",
                ]
                .map(str::to_owned),
            ),
        ),
        (
            "UnordSet",
            BTreeSet::from(
                [
                    "clear",
                    "contains",
                    "insert",
                    "into_sorted",
                    "is_empty",
                    "len",
                    "merge",
                    "new",
                    "remove",
                    "to_sorted",
                ]
                .map(str::to_owned),
            ),
        ),
    ]);

    for (type_name, methods) in expected {
        assert_eq!(
            public_methods(type_name),
            methods,
            "{type_name} API drifted"
        );
    }
}

#[test]
fn public_trait_api_is_exact() {
    assert_eq!(
        public_traits("UnordMap"),
        BTreeSet::from(
            [
                "Clone",
                "Debug",
                "Default",
                "Deserialize",
                "Eq",
                "Extend",
                "From",
                "FromIterator",
                "Index",
                "PartialEq",
                "Serialize",
            ]
            .map(str::to_owned)
        )
    );
    assert_eq!(
        public_traits("UnordSet"),
        BTreeSet::from(
            [
                "Clone",
                "Debug",
                "Default",
                "Deserialize",
                "Eq",
                "Extend",
                "From",
                "FromIterator",
                "PartialEq",
                "Serialize",
            ]
            .map(str::to_owned)
        )
    );
}

#[test]
fn private_storage_is_an_ordered_collection() {
    let source = fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"))
        .expect("read chelis-unord source");
    let syntax = syn::parse_file(&source).expect("parse chelis-unord source");

    let expected = BTreeMap::from([("UnordMap", "BTreeMap"), ("UnordSet", "BTreeSet")]);
    for (type_name, storage) in expected {
        let item = syntax
            .items
            .iter()
            .find_map(|item| match item {
                Item::Struct(item) if item.ident == type_name => Some(item),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{type_name} must exist"));
        let fields = item
            .fields
            .iter()
            .map(|field| (field.ident.as_ref().unwrap().to_string(), &field.ty))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            fields.keys().cloned().collect::<BTreeSet<_>>(),
            BTreeSet::from(["inner".to_owned()]),
            "{type_name} must store exactly one private collection"
        );
        path_type(fields["inner"], storage);
    }

    // The determinism claim rests on there being no randomized order to leak,
    // not on a wrapper hiding one behind an exemption. This crate used to
    // carry the workspace's only production `disallowed_types` allowance, for
    // a private raw hash table; with ordered storage it needs none, so the
    // workspace ban now covers the wrapper itself.
    assert!(
        !source.contains("allow(clippy::disallowed_types)"),
        "chelis-unord must not exempt itself from the hash-collection ban"
    );
}
