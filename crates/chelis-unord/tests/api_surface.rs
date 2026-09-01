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
        .filter(|item_impl| {
            impl_self_name(&item_impl.self_ty).is_some_and(|name| name == type_name)
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
        .filter(|item_impl| {
            impl_self_name(&item_impl.self_ty).is_some_and(|name| name == type_name)
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
        // The entry types are public API too. A new order-bearing exit could
        // be added through one of them without touching either collection.
        (
            "Entry",
            BTreeSet::from(
                [
                    "and_modify",
                    "key",
                    "or_default",
                    "or_insert",
                    "or_insert_with",
                    "or_insert_with_key",
                ]
                .map(str::to_owned),
            ),
        ),
        (
            "OccupiedEntry",
            BTreeSet::from(
                [
                    "get",
                    "get_mut",
                    "insert",
                    "into_mut",
                    "key",
                    "remove",
                    "remove_entry",
                ]
                .map(str::to_owned),
            ),
        ),
        (
            "VacantEntry",
            BTreeSet::from(["insert", "into_key", "key"].map(str::to_owned)),
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

/// The five types the pins above cover.
const PINNED_TYPES: [&str; 5] = [
    "Entry",
    "OccupiedEntry",
    "UnordMap",
    "UnordSet",
    "VacantEntry",
];

/// The self type of an `impl`, with any reference layers stripped.
///
/// `impl IntoIterator for &UnordMap<..>` is an order-bearing exit whose self
/// type is a reference, so a check that only looks at `Type::Path` misses it.
fn impl_self_name(mut ty: &Type) -> Option<String> {
    loop {
        match ty {
            Type::Reference(reference) => ty = &reference.elem,
            Type::Paren(paren) => ty = &paren.elem,
            Type::Path(path) => {
                return path
                    .path
                    .segments
                    .last()
                    .map(|segment| segment.ident.to_string());
            }
            _ => return None,
        }
    }
}

#[test]
fn the_public_surface_is_closed_by_construction() {
    // Two earlier rounds repaired this test by adding the filter case the
    // previous round escaped through. This one enumerates by exclusion
    // instead: every public item must be one of the five pinned types, and
    // every `impl` must be on one of them, so there is no shape left to add.
    let source = fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"))
        .expect("read chelis-unord source");
    let syntax = syn::parse_file(&source).expect("parse chelis-unord source");
    let pinned: BTreeSet<String> = PINNED_TYPES.map(str::to_owned).into_iter().collect();

    // An impl on a private type cannot be a public exit, so those are allowed;
    // the closure comes from the public-item check below, which admits only
    // the pinned five.
    let private_types: BTreeSet<String> = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Struct(item) if !is_public(&item.vis) => Some(item.ident.to_string()),
            Item::Enum(item) if !is_public(&item.vis) => Some(item.ident.to_string()),
            _ => None,
        })
        .collect();

    let mut public_types = BTreeSet::new();
    for item in &syntax.items {
        let (kind, name, is_public) = match item {
            Item::Struct(item) => ("struct", item.ident.to_string(), is_public(&item.vis)),
            Item::Enum(item) => ("enum", item.ident.to_string(), is_public(&item.vis)),
            Item::Fn(item) => ("fn", item.sig.ident.to_string(), is_public(&item.vis)),
            Item::Mod(item) => ("mod", item.ident.to_string(), is_public(&item.vis)),
            Item::Type(item) => ("type", item.ident.to_string(), is_public(&item.vis)),
            Item::Trait(item) => ("trait", item.ident.to_string(), is_public(&item.vis)),
            Item::Const(item) => ("const", item.ident.to_string(), is_public(&item.vis)),
            Item::Static(item) => ("static", item.ident.to_string(), is_public(&item.vis)),
            Item::Use(item) => ("use", "<re-export>".to_owned(), is_public(&item.vis)),
            Item::Macro(item) => (
                "macro",
                item.ident
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "<anonymous>".to_owned()),
                item.attrs
                    .iter()
                    .any(|attribute| attribute.path().is_ident("macro_export")),
            ),
            Item::Impl(item) => {
                let name = impl_self_name(&item.self_ty)
                    .unwrap_or_else(|| panic!("an impl self type must resolve to a named type"));
                assert!(
                    pinned.contains(&name) || private_types.contains(&name),
                    "impl on `{name}`, which is neither pinned nor private to this crate: \
                     an impl reaching a public type through a reference is an exit the \
                     inherent and trait pins do not see"
                );
                continue;
            }
            _ => continue,
        };
        if !is_public {
            continue;
        }
        assert!(
            matches!(kind, "struct" | "enum"),
            "unexpected public {kind} `{name}`: the crate's public surface is the five \
             pinned types and nothing else"
        );
        public_types.insert(name);
    }
    assert_eq!(public_types, pinned);
}

fn is_public(visibility: &syn::Visibility) -> bool {
    matches!(visibility, syn::Visibility::Public(_))
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
