use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use syn::{GenericArgument, ImplItem, Item, PathArguments, Type};

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
        .collect()
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
                "Drop",
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
fn raw_hash_storage_contains_numeric_indices_only() {
    let source = fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"))
        .expect("read chelis-unord source");
    let syntax = syn::parse_file(&source).expect("parse chelis-unord source");
    let storage = syntax
        .items
        .iter()
        .find_map(|item| match item {
            Item::Struct(item) if item.ident == "Storage" => Some(item),
            _ => None,
        })
        .expect("Storage struct must exist");
    let fields = storage
        .fields
        .iter()
        .map(|field| (field.ident.as_ref().unwrap().to_string(), &field.ty))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        fields.keys().cloned().collect::<BTreeSet<_>>(),
        ["buckets", "entries", "free", "hash_builder", "len", "order"]
            .map(str::to_owned)
            .into_iter()
            .collect()
    );

    let map = path_type(fields["buckets"], "Map");
    let PathArguments::AngleBracketed(arguments) = &map.path.segments.last().unwrap().arguments
    else {
        panic!("raw Map must have type arguments");
    };
    let arguments = arguments.args.iter().collect::<Vec<_>>();
    let [GenericArgument::Type(digest), GenericArgument::Type(bucket)] = arguments.as_slice()
    else {
        panic!("raw Map must have exactly two type arguments");
    };
    path_type(digest, "u64");
    path_type(bucket, "usize");
}
