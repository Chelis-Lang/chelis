//! Structural seam scanner for the chelis#893 runtime-representation inventory.
//!
//! `spec/design/runtime_representation.md` section C6 names the seam classes
//! Phase 0 must freeze: descriptor fields, `data` access, pointer casts,
//! width/arithmetic matches, `normalized_key` arithmetic, fixed-rank arrays,
//! narrow metadata fields, backend element spellings, and load/store
//! templates. This crate derives those rows and nothing else.
//!
//! # One real parser per language
//!
//! The inventory's universe is a frozen list of repository files, not a
//! language. Forty-nine of them are Rust, six are C headers, and one
//! (`chelis_metal_runtime.h`) is Objective-C. Rust is read with `syn`, a
//! total parser for the language. The headers are read through clang's front
//! end by the [`c_ast`] module: the compiler supplies every declaration and
//! its enclosing owner, so the declaration forms a hand-written token walk
//! mis-modelled in turn (tagged aggregates, unions, macro-typed declarators,
//! multi-declarator lists, attributes, K&R definitions, keywords inside string
//! literals) are the compiler's problem rather than this crate's. Each header
//! is parsed under a closed set of preprocessing configurations whose union
//! is its row set, and an arm no configuration parses fails the scan. What
//! the crate keeps from the numeric capacity census is its closed type-word
//! lists (`tests/support/c_lexical.rs`), so a C type word has one
//! classification authority in the repository rather than two.
//!
//! A seam row is owned by its enclosing declaration, not by a source offset or
//! a hash of the exact bytes: Phases 3 and 4 delete functions and call sites,
//! so that is the granularity the ledger records. Reformatting a literal
//! inside a function does not churn the freeze; adding a new function that
//! carries a seam does.

#[path = "../../../tests/support/c_lexical.rs"]
pub mod c_lexical;

use std::collections::BTreeSet;
use std::fmt;

use quote::ToTokens;
use serde::{Deserialize, Serialize};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};

/// The exact seam classes section C6 enumerates.
pub const SEAM_KINDS: &[&str] = &[
    "backend-element-spelling",
    "descriptor-field",
    "direct-data-access",
    "dtype-contract",
    "fixed-rank-metadata",
    "load-store-template",
    "narrow-metadata",
    "normalized-key-arithmetic",
    "raw-element-pointer",
    "width-arithmetic",
];

/// Field names that make a struct a tensor descriptor rather than an ordinary
/// aggregate. A descriptor needs `data` and `dtype` plus at least four of
/// these, which is what separates `chelis_tensor` from a helper struct that
/// merely happens to have a `size`.
const DESCRIPTOR_FIELDS: &[&str] = &[
    "byte_capacity",
    "data",
    "dtype",
    "ndim",
    "rank",
    "shape",
    "size",
    "storage_size",
    "strides",
];

/// Descriptor fields whose declared width is narrower than the int64 extent
/// domain [05-DIM-2] requires.
const NARROWABLE_FIELDS: &[&str] = &["ndim", "rank", "size", "storage_size"];

/// Rust spellings of a tensor element. `u8` is deliberately absent: it is the
/// raw byte carrier, inventoried through `direct-data-access` instead.
const RUST_ELEMENT_TYPES: &[&str] = &[
    "Bf16Bits",
    "Bool8",
    "F16Bits",
    "bf16",
    "c_double",
    "c_float",
    "c_int",
    "c_long",
    "c_longlong",
    "c_short",
    "c_uint",
    "c_ulong",
    "c_ulonglong",
    "c_ushort",
    "f16",
    "f32",
    "f64",
    "i16",
    "i32",
    "i64",
    "i8",
    "u16",
    "u32",
    "u64",
];

/// C spellings of a tensor element.
const C_ELEMENT_TYPES: &[&str] = &[
    "__half",
    "double",
    "float",
    "half",
    "hip_bfloat16",
    "int16_t",
    "int32_t",
    "int64_t",
    "int8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "uint8_t",
];

/// Method and function names that select or compute a representation width.
const WIDTH_SELECTORS: &[&str] = &[
    "byte_capacity",
    "byte_width",
    "dtype_size",
    "elem_size",
    "element_size",
    "itemsize",
];

/// The saturating folds [#888] names, plus the checked forms a capacity
/// computation may use instead.
const CAPACITY_FOLDS: &[&str] = &[
    "checked_mul",
    "saturating_add",
    "saturating_mul",
    "saturating_pow",
    "saturating_sub",
];

/// Types whose variants are the dtype/representation contract.
const DTYPE_CONTRACT_TYPES: &[&str] = &["Repr", "RuntimeDType"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceClass {
    Runtime,
    Vocab,
    Ir,
    Python,
    Backend,
}

impl SourceClass {
    /// Classify a repository-relative path. An unrecognized path is not a
    /// silent pass: the caller registers every inventory source explicitly, so
    /// `None` here means the frozen source list and this function disagree.
    pub fn for_path(path: &str) -> Option<Self> {
        if path.starts_with("crates/chelis-runtime/") {
            Some(Self::Runtime)
        } else if path.starts_with("crates/chelis-vocab/") {
            Some(Self::Vocab)
        } else if path.starts_with("crates/chelis-ir/") {
            Some(Self::Ir)
        } else if path.starts_with("crates/chelis-python/") {
            Some(Self::Python)
        } else if path.starts_with("crates/chelis-backend-") {
            Some(Self::Backend)
        } else {
            None
        }
    }

    fn is_backend(self) -> bool {
        matches!(self, Self::Backend)
    }

    /// Only these carry a device or host tensor descriptor whose metadata can
    /// be narrow or fixed-rank.
    fn mirrors_a_descriptor(self) -> bool {
        matches!(self, Self::Python | Self::Backend | Self::Runtime)
    }
}

/// One seam occurrence, owned by its enclosing declaration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeamRow {
    pub kind: String,
    pub owner: String,
    /// A short legible excerpt. It is evidence for a reviewer, never part of
    /// the row's identity, so reformatting cannot move the freeze.
    pub sample: String,
}

impl SeamRow {
    fn new(kind: &str, owner: &str, sample: impl AsRef<str>) -> Self {
        debug_assert!(SEAM_KINDS.contains(&kind), "unregistered seam kind: {kind}");
        Self {
            kind: kind.to_string(),
            owner: owner.to_string(),
            sample: excerpt(sample.as_ref()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanError {
    pub message: String,
}

impl ScanError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ScanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ScanError {}

/// Collapse a source excerpt to one legible line, capped so the frozen
/// artifact stays readable.
fn excerpt(text: &str) -> String {
    let flattened = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flattened.chars().count() <= 100 {
        return flattened;
    }
    let head: String = flattened.chars().take(97).collect();
    format!("{head}...")
}

// ---------------------------------------------------------------------------
// Rust
// ---------------------------------------------------------------------------

fn is_exact_cfg_test_module(module: &syn::ItemMod) -> bool {
    module.attrs.iter().any(|attribute| {
        attribute.path().is_ident("cfg")
            && attribute
                .parse_args::<syn::Path>()
                .is_ok_and(|path| path.is_ident("test"))
    })
}

#[derive(Default)]
struct TestModuleSpans {
    spans: Vec<proc_macro2::Span>,
}

impl<'ast> Visit<'ast> for TestModuleSpans {
    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        if is_exact_cfg_test_module(module) {
            self.spans.push(module.span());
        } else {
            visit::visit_item_mod(self, module);
        }
    }
}

/// Blank every `#[cfg(test)]` module while preserving byte offsets and line
/// breaks. Only an exact `cfg(test)` is excluded: `cfg(not(test))`, production
/// items that follow a test module, and files whose names end in `_tests.rs`
/// all remain production source.
pub fn production_rust_source(source: &str) -> Result<String, ScanError> {
    let file = syn::parse_file(source).map_err(|error| {
        ScanError::new(format!(
            "cannot parse Rust source before test-region exclusion: {error}"
        ))
    })?;
    let mut visitor = TestModuleSpans::default();
    visitor.visit_file(&file);
    let mut line_starts = vec![0usize];
    for (index, byte) in source.bytes().enumerate() {
        if byte == b'\n' {
            line_starts.push(index + 1);
        }
    }
    let offset = |location: proc_macro2::LineColumn| -> Option<usize> {
        line_starts
            .get(location.line.checked_sub(1)?)
            .and_then(|start| start.checked_add(location.column))
    };
    let mut bytes = source.as_bytes().to_vec();
    for span in visitor.spans {
        let (Some(start), Some(end)) = (offset(span.start()), offset(span.end())) else {
            continue;
        };
        for byte in bytes.get_mut(start..end).into_iter().flatten() {
            if *byte != b'\n' {
                *byte = b' ';
            }
        }
    }
    String::from_utf8(bytes).map_err(|error| {
        ScanError::new(format!("test-region exclusion damaged Rust UTF-8: {error}"))
    })
}

/// Does this Rust type name a tensor element, or an array or slice of one?
fn is_rust_element_type(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(path) => path.path.segments.last().is_some_and(|segment| {
            RUST_ELEMENT_TYPES.contains(&segment.ident.to_string().as_str())
        }),
        syn::Type::Paren(inner) => is_rust_element_type(&inner.elem),
        syn::Type::Array(array) => is_rust_element_type(&array.elem),
        syn::Type::Slice(slice) => is_rust_element_type(&slice.elem),
        _ => false,
    }
}

/// A pointer whose pointee is an element type is a raw element pointer however
/// many `const`/`mut` layers sit above it; `NonNull<T>` is the same pointer
/// with a non-null proof and nothing else.
fn is_raw_element_pointer(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Ptr(pointer) => {
            is_rust_element_type(&pointer.elem) || is_raw_element_pointer(&pointer.elem)
        }
        syn::Type::Paren(inner) => is_raw_element_pointer(&inner.elem),
        syn::Type::Path(path) => non_null_pointee(path).is_some_and(is_rust_element_type),
        _ => false,
    }
}

/// Std pointer wrappers whose one type argument is the pointee.
const POINTER_WRAPPERS: &[&str] = &["AtomicPtr", "NonNull"];

/// Std pointer constructors whose turbofish names the pointee, so
/// `NonNull::<f32>::new(..)` or `slice_from_raw_parts::<f32>(..)` hands out an
/// element pointer without a type position ever spelling it.
const POINTER_CONSTRUCTORS: &[&str] = &[
    "AtomicPtr",
    "NonNull",
    "dangling",
    "from_raw_parts",
    "from_raw_parts_mut",
    "null",
    "null_mut",
    "slice_from_raw_parts",
    "slice_from_raw_parts_mut",
    "without_provenance",
    "without_provenance_mut",
];

/// The `T` of a `NonNull<T>` or `AtomicPtr<T>` spelling, by any path.
fn non_null_pointee(path: &syn::TypePath) -> Option<&syn::Type> {
    let segment = path.path.segments.last()?;
    if !POINTER_WRAPPERS.contains(&segment.ident.to_string().as_str()) {
        return None;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    arguments.args.iter().find_map(|argument| match argument {
        syn::GenericArgument::Type(ty) => Some(ty),
        _ => None,
    })
}

/// A fixed-rank array is one whose length is a compile-time rank cap: a
/// `*MAX_DIM` constant or a literal above one.
fn fixed_rank_extent(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Array(array) => {
            let rendered = array.len.to_token_stream().to_string();
            let is_cap = rendered.contains("MAX_DIM")
                || rendered.parse::<u64>().is_ok_and(|extent| extent > 1);
            is_cap.then_some(rendered)
        }
        syn::Type::Paren(inner) => fixed_rank_extent(&inner.elem),
        _ => None,
    }
}

fn is_i32(ty: &syn::Type) -> bool {
    matches!(ty, syn::Type::Path(path) if path.path.is_ident("i32"))
}

fn field_names(fields: &syn::Fields) -> BTreeSet<String> {
    fields
        .iter()
        .filter_map(|field| field.ident.as_ref().map(ToString::to_string))
        .collect()
}

/// A descriptor carries both `data` and `dtype` plus at least four descriptor
/// fields. That threshold is what keeps an ordinary helper struct with a
/// `size` field out of the inventory.
fn is_descriptor(names: &BTreeSet<String>) -> bool {
    names.contains("data")
        && names.contains("dtype")
        && names
            .iter()
            .filter(|name| DESCRIPTOR_FIELDS.contains(&name.as_str()))
            .count()
            >= 4
}

/// The C element type this literal spells, if any. The spelling is returned
/// rather than a bare boolean so a caller can name what it matched.
fn c_element_spelling(text: &str) -> Option<&'static str> {
    let bytes = text.as_bytes();
    C_ELEMENT_TYPES.iter().copied().find(|candidate| {
        text.match_indices(candidate).any(|(start, _)| {
            let before = start
                .checked_sub(1)
                .map(|index| bytes[index] as char)
                .is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_');
            let after = bytes
                .get(start + candidate.len())
                .map(|byte| *byte as char)
                .is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_');
            before && after
        })
    })
}

/// Does this callee path select a representation width?
///
/// The path is matched segment by segment with any turbofish removed, so
/// `size_of::<f32>()`, `std::mem::size_of::<f32>()`, and
/// `core::mem::size_of::<f32>()` are one rule rather than three spellings, and
/// a prefix test cannot miss the qualified forms.
fn is_width_selecting_path(rendered: &str) -> bool {
    let without_generics = rendered.split('<').next().unwrap_or(rendered);
    without_generics
        .split("::")
        .map(str::trim)
        .any(|segment| segment == "size_of" || WIDTH_SELECTORS.contains(&segment))
}

/// A load/store template dereferences or indexes; an element spelling only
/// names the type. Keeping them disjoint means a row cannot be counted twice.
fn is_load_store_template(text: &str) -> bool {
    text.contains("->") || text.contains('[') || text.contains("*(") || text.contains(")[")
}

struct RustSeamScanner {
    class: SourceClass,
    /// The capacity module owns [#888]'s unchecked products.
    defines_capacity_keys: bool,
    owners: Vec<String>,
    rows: Vec<SeamRow>,
    error: Option<ScanError>,
}

impl RustSeamScanner {
    fn owner(&self) -> String {
        if self.owners.is_empty() {
            "module".to_string()
        } else {
            self.owners.join("::")
        }
    }

    fn push(&mut self, kind: &str, sample: impl AsRef<str>) {
        if self.owners.is_empty() {
            if self.error.is_none() {
                self.error = Some(ScanError::new(format!(
                    "a {kind} seam sits outside every declaration: `{}`. A placeholder \
                     owner would absorb every later seam of that kind in this file, so \
                     the scanner refuses to invent one",
                    excerpt(sample.as_ref())
                )));
            }
            return;
        }
        let owner = self.owner();
        self.rows.push(SeamRow::new(kind, &owner, sample));
    }

    fn with_owner(&mut self, owner: String, visit: impl FnOnce(&mut Self)) {
        self.owners.push(owner);
        visit(self);
        self.owners.pop();
    }

    fn scan_literal(&mut self, value: &str) {
        if !self.class.is_backend() {
            return;
        }
        if c_element_spelling(value).is_none() {
            return;
        }
        let kind = if is_load_store_template(value) {
            "load-store-template"
        } else {
            "backend-element-spelling"
        };
        self.push(kind, value);
    }

    /// Walk every literal in a macro's token tree, including nested groups.
    /// A template one group deep is still emitted text.
    fn scan_token_stream_literals(&mut self, stream: proc_macro2::TokenStream) {
        for token in stream {
            match token {
                proc_macro2::TokenTree::Group(group) => {
                    self.scan_token_stream_literals(group.stream());
                }
                proc_macro2::TokenTree::Literal(literal) => {
                    if let Ok(syn::Lit::Str(text)) = syn::parse_str(&literal.to_string()) {
                        self.scan_literal(&text.value());
                    }
                }
                proc_macro2::TokenTree::Ident(_) | proc_macro2::TokenTree::Punct(_) => {}
            }
        }
    }

    /// Visit each field under its own owner (`Type::field`, or the tuple
    /// index), so two carrier fields of one aggregate are two identities.
    fn visit_member_fields(&mut self, fields: &syn::Fields) {
        for (index, field) in fields.iter().enumerate() {
            let member = field
                .ident
                .as_ref()
                .map_or_else(|| index.to_string(), ToString::to_string);
            self.with_owner(member, |scanner| visit::visit_field(scanner, field));
        }
    }

    fn scan_descriptor_fields(&mut self, name: &str, fields: &syn::Fields) {
        let names = field_names(fields);
        if !is_descriptor(&names) || !self.class.mirrors_a_descriptor() {
            return;
        }
        for field in fields {
            let Some(ident) = field.ident.as_ref() else {
                continue;
            };
            let field_name = ident.to_string();
            let rendered = format!("{field_name}: {}", field.ty.to_token_stream());
            let owner = format!("{name}::{field_name}");
            self.rows
                .push(SeamRow::new("descriptor-field", &owner, &rendered));
            if NARROWABLE_FIELDS.contains(&field_name.as_str()) && is_i32(&field.ty) {
                self.rows
                    .push(SeamRow::new("narrow-metadata", &owner, &rendered));
            }
            if let Some(extent) = fixed_rank_extent(&field.ty) {
                self.rows.push(SeamRow::new(
                    "fixed-rank-metadata",
                    &owner,
                    format!("{rendered} (extent {extent})"),
                ));
            }
        }
    }
}

impl<'ast> Visit<'ast> for RustSeamScanner {
    fn visit_attribute(&mut self, attribute: &'ast syn::Attribute) {
        // `#[path = "..."]` reaches a file outside the inventory roots and
        // cargo compiles it into the crate, so admitting it silently reopens
        // the closed universe the same way `include!` would.
        let names_a_path = attribute.path().is_ident("path")
            || (attribute.path().is_ident("cfg_attr")
                && attribute.to_token_stream().to_string().contains("path"));
        if names_a_path {
            let owner = self.owner();
            if self.error.is_none() {
                self.error = Some(ScanError::new(format!(
                    "`#[path]` in `{owner}` compiles a file the inventory roots do not \
                     reach: register the target in INVENTORY_SOURCES, or move it under a \
                     root"
                )));
            }
        }
    }

    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        if is_exact_cfg_test_module(module) {
            return;
        }
        self.with_owner(module.ident.to_string(), |scanner| {
            visit::visit_item_mod(scanner, module);
        });
    }

    fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
        self.with_owner(function.sig.ident.to_string(), |scanner| {
            visit::visit_item_fn(scanner, function);
        });
    }

    fn visit_impl_item_fn(&mut self, function: &'ast syn::ImplItemFn) {
        self.with_owner(function.sig.ident.to_string(), |scanner| {
            visit::visit_impl_item_fn(scanner, function);
        });
    }

    fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
        self.with_owner(item.ident.to_string(), |scanner| {
            visit::visit_item_const(scanner, item);
        });
    }

    fn visit_item_static(&mut self, item: &'ast syn::ItemStatic) {
        self.with_owner(item.ident.to_string(), |scanner| {
            visit::visit_item_static(scanner, item);
        });
    }

    fn visit_foreign_item_fn(&mut self, item: &'ast syn::ForeignItemFn) {
        // An `unsafe extern "C"` declaration carries the same carriers a
        // definition does, and Phase 3 deletes it the same way.
        self.with_owner(item.sig.ident.to_string(), |scanner| {
            visit::visit_foreign_item_fn(scanner, item);
        });
    }

    fn visit_item_trait(&mut self, item: &'ast syn::ItemTrait) {
        self.with_owner(item.ident.to_string(), |scanner| {
            visit::visit_item_trait(scanner, item);
        });
    }

    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        // A trait's PROVIDED method has a body like any other, and
        // `TensorElement::data_ptr` is one of the accessors chelis#893 exists
        // to seal. Without this it had no owner at all.
        self.with_owner(item.sig.ident.to_string(), |scanner| {
            visit::visit_trait_item_fn(scanner, item);
        });
    }

    fn visit_item_impl(&mut self, item: &'ast syn::ItemImpl) {
        let self_ty = item.self_ty.to_token_stream().to_string();
        let trait_name = item
            .trait_
            .as_ref()
            .and_then(|(_, path, _)| path.segments.last())
            .map(|segment| segment.ident.to_string());
        // `impl TensorElement for f32` is the dtype contract: it binds a Rust
        // element marker to a runtime dtype tag.
        if trait_name.as_deref() == Some("TensorElement") {
            let owner = format!("TensorElement for {self_ty}");
            self.rows
                .push(SeamRow::new("dtype-contract", &owner, &owner));
        }
        self.with_owner(self_ty, |scanner| visit::visit_item_impl(scanner, item));
    }

    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        self.scan_descriptor_fields(&item.ident.to_string(), &item.fields);
        self.with_owner(item.ident.to_string(), |scanner| {
            for attribute in &item.attrs {
                scanner.visit_attribute(attribute);
            }
            scanner.visit_generics(&item.generics);
            scanner.visit_member_fields(&item.fields);
        });
    }

    fn visit_item_union(&mut self, item: &'ast syn::ItemUnion) {
        self.with_owner(item.ident.to_string(), |scanner| {
            for attribute in &item.attrs {
                scanner.visit_attribute(attribute);
            }
            scanner.visit_generics(&item.generics);
            for (index, field) in item.fields.named.iter().enumerate() {
                let member = field
                    .ident
                    .as_ref()
                    .map_or_else(|| index.to_string(), ToString::to_string);
                scanner.with_owner(member, |scanner| visit::visit_field(scanner, field));
            }
        });
    }

    fn visit_variant(&mut self, variant: &'ast syn::Variant) {
        // Every variant owns its own row: two pointer-carrying variants of one
        // enum are two identities, not one.
        self.with_owner(variant.ident.to_string(), |scanner| {
            for attribute in &variant.attrs {
                scanner.visit_attribute(attribute);
            }
            scanner.visit_member_fields(&variant.fields);
            if let Some((_, discriminant)) = &variant.discriminant {
                scanner.visit_expr(discriminant);
            }
        });
    }

    fn visit_impl_item_const(&mut self, item: &'ast syn::ImplItemConst) {
        self.with_owner(item.ident.to_string(), |scanner| {
            visit::visit_impl_item_const(scanner, item);
        });
    }

    fn visit_trait_item_const(&mut self, item: &'ast syn::TraitItemConst) {
        self.with_owner(item.ident.to_string(), |scanner| {
            visit::visit_trait_item_const(scanner, item);
        });
    }

    fn visit_item_type(&mut self, item: &'ast syn::ItemType) {
        // `pub type ElemPtr = *mut f32;` hands out a carrier under its own
        // name, so it owns the row rather than failing as unowned.
        self.with_owner(item.ident.to_string(), |scanner| {
            visit::visit_item_type(scanner, item);
        });
    }

    fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
        let name = item.ident.to_string();
        if DTYPE_CONTRACT_TYPES.contains(&name.as_str()) {
            // One row per variant. A dtype added without its complete contract
            // must appear as a new identity, which it cannot do if every
            // variant collapses onto the enum.
            for variant in &item.variants {
                let owner = format!("{name}::{}", variant.ident);
                self.rows.push(SeamRow::new(
                    "dtype-contract",
                    &owner,
                    variant.to_token_stream().to_string(),
                ));
            }
        }
        self.with_owner(name, |scanner| visit::visit_item_enum(scanner, item));
    }

    fn visit_field(&mut self, field: &'ast syn::Field) {
        if is_raw_element_pointer(&field.ty) {
            self.push("raw-element-pointer", field.to_token_stream().to_string());
        }
        visit::visit_field(self, field);
    }

    fn visit_expr_field(&mut self, field: &'ast syn::ExprField) {
        if matches!(&field.member, syn::Member::Named(name) if name == "data") {
            self.push("direct-data-access", field.to_token_stream().to_string());
        }
        visit::visit_expr_field(self, field);
    }

    fn visit_type_path(&mut self, path: &'ast syn::TypePath) {
        if non_null_pointee(path).is_some_and(is_rust_element_type) {
            self.push("raw-element-pointer", path.to_token_stream().to_string());
        }
        visit::visit_type_path(self, path);
    }

    fn visit_impl_item_type(&mut self, item: &'ast syn::ImplItemType) {
        self.with_owner(item.ident.to_string(), |scanner| {
            visit::visit_impl_item_type(scanner, item);
        });
    }

    fn visit_foreign_item_static(&mut self, item: &'ast syn::ForeignItemStatic) {
        self.with_owner(item.ident.to_string(), |scanner| {
            visit::visit_foreign_item_static(scanner, item);
        });
    }

    fn visit_type_ptr(&mut self, pointer: &'ast syn::TypePtr) {
        // Casts are not the only place a raw element pointer appears: a
        // parameter, a return type, and a foreign declaration all hand one
        // out, and Phase 3's seal has to reach every one of them.
        if is_rust_element_type(&pointer.elem) || is_raw_element_pointer(&pointer.elem) {
            self.push("raw-element-pointer", pointer.to_token_stream().to_string());
        }
        visit::visit_type_ptr(self, pointer);
    }

    fn visit_expr_cast(&mut self, cast: &'ast syn::ExprCast) {
        if is_raw_element_pointer(&cast.ty) {
            self.push("raw-element-pointer", cast.to_token_stream().to_string());
        }
        visit::visit_expr_cast(self, cast);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        // `NonNull::<f32>::new(p)`, `slice_from_raw_parts::<f32>(p, n)`: a
        // pointer constructor instantiated at an element type in expression
        // position, where no type position spells the pointee.
        let names_a_constructor = path
            .path
            .segments
            .iter()
            .any(|segment| POINTER_CONSTRUCTORS.contains(&segment.ident.to_string().as_str()));
        if names_a_constructor && path_generics_name_an_element(&path.path) {
            self.push("raw-element-pointer", path.to_token_stream().to_string());
        }
        visit::visit_expr_path(self, path);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let method = call.method.to_string();
        // `p.cast::<f32>()` is the idiomatic pointer cast; the target type
        // lives only in the turbofish.
        if matches!(method.as_str(), "cast" | "cast_mut" | "cast_const")
            && call.turbofish.as_ref().is_some_and(|turbofish| {
                turbofish.args.iter().any(|argument| {
                    matches!(argument, syn::GenericArgument::Type(ty) if is_rust_element_type(ty))
                })
            })
        {
            self.push("raw-element-pointer", call.to_token_stream().to_string());
        }
        // The saturating folds [#888] names, plus every consumer that compares
        // the resulting key. Phase 1's exit requires each reuse consumer to
        // move to the exact key, so a consumer outside the IR is equally a
        // seam.
        let is_fold =
            CAPACITY_FOLDS.contains(&method.as_str()) && matches!(self.class, SourceClass::Ir);
        if is_fold || method == "normalized_key" {
            self.push(
                "normalized-key-arithmetic",
                call.to_token_stream().to_string(),
            );
        }
        if WIDTH_SELECTORS.contains(&method.as_str()) {
            self.push("width-arithmetic", call.to_token_stream().to_string());
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let rendered = call.func.to_token_stream().to_string();
        if is_width_selecting_path(&rendered) {
            self.push("width-arithmetic", call.to_token_stream().to_string());
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_expr_binary(&mut self, binary: &'ast syn::ExprBinary) {
        // [#888]'s mechanism is an unchecked product folded into a capacity
        // key. The two surviving bare products live in the module that defines
        // `DimExpr`; gating on that file rather than on a substring of the
        // enclosing function's name keeps the rule structural.
        if self.defines_capacity_keys && matches!(binary.op, syn::BinOp::Mul(_)) {
            self.push(
                "normalized-key-arithmetic",
                binary.to_token_stream().to_string(),
            );
        }
        visit::visit_expr_binary(self, binary);
    }

    fn visit_lit_str(&mut self, literal: &'ast syn::LitStr) {
        self.scan_literal(&literal.value());
    }

    fn visit_item_macro(&mut self, item: &'ast syn::ItemMacro) {
        // A `macro_rules!` transcriber is code the macro emits at every
        // expansion, so its seams are owned by the macro's name. Each arm is
        // read after its metavariables are replaced with plain identifiers;
        // one that still cannot be read and names a seam fails closed.
        if !item.mac.path.is_ident("macro_rules") {
            visit::visit_item_macro(self, item);
            return;
        }
        let name = item
            .ident
            .as_ref()
            .map_or_else(|| "macro_rules".to_string(), ToString::to_string);
        let class = self.class;
        self.with_owner(name.clone(), |scanner| {
            for transcriber in macro_rules_transcribers(item.mac.tokens.clone()) {
                let rewritten = demetavariable(transcriber);
                if let Ok(statements) = syn::Block::parse_within.parse2(rewritten.clone()) {
                    for statement in &statements {
                        scanner.visit_stmt(statement);
                    }
                } else if let Some(word) = seam_word_in(class, rewritten.clone())
                    && scanner.error.is_none()
                {
                    scanner.error = Some(ScanError::new(format!(
                        "an arm of `macro_rules! {name}` cannot be read as statements \
                         once its metavariables are substituted, and it mentions \
                         `{word}`; the inventory cannot see through it, so spell the \
                         seam outside the macro"
                    )));
                }
                scanner.scan_token_stream_literals(rewritten);
            }
        });
    }

    fn visit_macro(&mut self, macro_call: &'ast syn::Macro) {
        // `include!` splices a file cargo compiles into a registered source.
        // Admitting one silently would reopen the closed universe through the
        // back door, so it fails until the target is registered and scanned in
        // its own right.
        if macro_call.path.is_ident("include") {
            let owner = self.owner();
            if self.error.is_none() {
                self.error = Some(ScanError::new(format!(
                    "`include!` in `{owner}` splices an unscanned file into a registered \
                     source: register the included file in INVENTORY_SOURCES, or inline it"
                )));
            }
            return;
        }
        // A `format!`/`write!` template is emitted C text, so its literal is
        // the seam. Its arguments are ordinary expressions and must be walked
        // as such; `syn` does not descend into macro tokens on its own. The
        // body is read as a comma list, then as statements (`vec![x; n]`),
        // then as `matches!(expr, pat)`. A body none of those can read, and
        // that mentions a seam word, fails rather than hiding the seam.
        let tokens = macro_call.tokens.clone();
        // `offset_of!(T, data)` names the field by path rather than by access;
        // a layout computed from the field is a direct use of it.
        if macro_call
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "offset_of")
            && tokens.clone().into_iter().any(
                |token| matches!(token, proc_macro2::TokenTree::Ident(ident) if ident == "data"),
            )
        {
            self.push(
                "direct-data-access",
                macro_call.to_token_stream().to_string(),
            );
        }
        let comma_list = Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated;
        if let Ok(arguments) = comma_list.parse2(tokens.clone()) {
            for argument in &arguments {
                self.visit_expr(argument);
            }
        } else if let Ok(statements) = syn::Block::parse_within.parse2(tokens.clone()) {
            for statement in &statements {
                self.visit_stmt(statement);
            }
        } else if let Ok(scrutinee) = parse_matches_body.parse2(tokens.clone()) {
            self.visit_expr(&scrutinee);
        } else if let Some(word) = seam_word_in(self.class, tokens.clone()) {
            let owner = self.owner();
            if self.error.is_none() {
                self.error = Some(ScanError::new(format!(
                    "the body of `{}!` in `{owner}` cannot be read as expressions, \
                     statements, or a `matches!` form, and it mentions `{word}`; the \
                     inventory cannot see through it, so spell the seam outside the macro",
                    macro_call.path.to_token_stream()
                )));
            }
        }
        self.scan_token_stream_literals(tokens);
        visit::visit_macro(self, macro_call);
    }
}

/// Does any segment of this path carry a turbofish naming an element type?
fn path_generics_name_an_element(path: &syn::Path) -> bool {
    path.segments.iter().any(|segment| {
        match &segment.arguments {
        syn::PathArguments::AngleBracketed(arguments) => arguments.args.iter().any(|argument| {
            matches!(argument, syn::GenericArgument::Type(ty) if is_rust_element_type(ty))
        }),
        _ => false,
    }
    })
}

/// Read a `matches!(expr, pat)` or `matches!(expr, pat if guard)` body and
/// return its scrutinee, the only expression a seam can sit in.
fn parse_matches_body(input: syn::parse::ParseStream<'_>) -> syn::Result<syn::Expr> {
    let scrutinee: syn::Expr = input.parse()?;
    input.parse::<syn::Token![,]>()?;
    syn::Pat::parse_multi_with_leading_vert(input)?;
    if input.peek(syn::Token![if]) {
        input.parse::<syn::Token![if]>()?;
        input.parse::<syn::Expr>()?;
    }
    if input.peek(syn::Token![,]) {
        input.parse::<syn::Token![,]>()?;
    }
    Ok(scrutinee)
}

/// The first identifier in a token tree that could name a seam under this
/// source class's rules: a `data` access, a width selector, a capacity fold
/// in the IR, a key consumer, or an element type behind a pointer star.
fn seam_word_in(class: SourceClass, stream: proc_macro2::TokenStream) -> Option<String> {
    let tokens: Vec<proc_macro2::TokenTree> = stream.into_iter().collect();
    for (index, token) in tokens.iter().enumerate() {
        match token {
            proc_macro2::TokenTree::Group(group) => {
                if let Some(word) = seam_word_in(class, group.stream()) {
                    return Some(word);
                }
            }
            proc_macro2::TokenTree::Ident(ident) => {
                let word = ident.to_string();
                let is_seam_word = word == "data"
                    || word == "size_of"
                    || word == "normalized_key"
                    || WIDTH_SELECTORS.contains(&word.as_str())
                    || (CAPACITY_FOLDS.contains(&word.as_str())
                        && matches!(class, SourceClass::Ir));
                if is_seam_word {
                    return Some(word);
                }
                // `* mut f32` / `* const i64`: an element type governing a star.
                let behind_a_star = index >= 2
                    && RUST_ELEMENT_TYPES.contains(&word.as_str())
                    && matches!(&tokens[index - 1], proc_macro2::TokenTree::Ident(qualifier)
                        if qualifier == "mut" || qualifier == "const")
                    && matches!(&tokens[index - 2], proc_macro2::TokenTree::Punct(star)
                        if star.as_char() == '*');
                if behind_a_star {
                    return Some(format!("*{}", word));
                }
            }
            proc_macro2::TokenTree::Literal(_) | proc_macro2::TokenTree::Punct(_) => {}
        }
    }
    None
}

/// The transcriber token streams of a `macro_rules!` body: each arm is
/// `(matcher) => { transcriber }`, separated by `;`.
fn macro_rules_transcribers(body: proc_macro2::TokenStream) -> Vec<proc_macro2::TokenStream> {
    let mut transcribers = Vec::new();
    let mut tokens = body.into_iter().peekable();
    while let Some(token) = tokens.next() {
        let proc_macro2::TokenTree::Group(_) = token else {
            continue;
        };
        // `=` `>` then the transcriber group.
        let arrow = matches!(tokens.next(), Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == '=')
            && matches!(tokens.next(), Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == '>');
        if !arrow {
            continue;
        }
        if let Some(proc_macro2::TokenTree::Group(transcriber)) = tokens.next() {
            transcribers.push(transcriber.stream());
        }
        if matches!(tokens.peek(), Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == ';') {
            tokens.next();
        }
    }
    transcribers
}

/// Rewrite a transcriber so `syn` can read it: `$name` becomes the plain
/// identifier `__mv_name`, and a `$( ... ) sep op` repetition becomes its
/// body. The result is the code the macro would emit for one repetition,
/// which is exactly the code a seam lives in.
fn demetavariable(stream: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let mut output = proc_macro2::TokenStream::new();
    let mut tokens = stream.into_iter().peekable();
    while let Some(token) = tokens.next() {
        match token {
            proc_macro2::TokenTree::Punct(punct) if punct.as_char() == '$' => match tokens.next() {
                Some(proc_macro2::TokenTree::Ident(ident)) => {
                    output.extend([proc_macro2::TokenTree::Ident(proc_macro2::Ident::new(
                        &format!("__mv_{ident}"),
                        ident.span(),
                    ))]);
                }
                Some(proc_macro2::TokenTree::Group(group)) => {
                    output.extend(demetavariable(group.stream()));
                    // An optional separator, then the repetition operator.
                    let is_operator =
                        |p: &proc_macro2::Punct| matches!(p.as_char(), '*' | '+' | '?');
                    if let Some(proc_macro2::TokenTree::Punct(next)) = tokens.peek()
                        && !is_operator(next)
                    {
                        tokens.next();
                    }
                    if let Some(proc_macro2::TokenTree::Punct(next)) = tokens.peek()
                        && is_operator(next)
                    {
                        tokens.next();
                    }
                }
                Some(other) => {
                    output.extend([proc_macro2::TokenTree::Punct(punct), other]);
                }
                None => output.extend([proc_macro2::TokenTree::Punct(punct)]),
            },
            proc_macro2::TokenTree::Group(group) => {
                let mut rebuilt =
                    proc_macro2::Group::new(group.delimiter(), demetavariable(group.stream()));
                rebuilt.set_span(group.span());
                output.extend([proc_macro2::TokenTree::Group(rebuilt)]);
            }
            other => output.extend([other]),
        }
    }
    output
}

/// Derive every seam row in one Rust source file.
pub fn scan_rust_source(path: &str, source: &str) -> Result<Vec<SeamRow>, ScanError> {
    let Some(class) = SourceClass::for_path(path) else {
        return Err(ScanError::new(format!(
            "`{path}` is not a registered runtime-representation inventory source; \
             add it to INVENTORY_SOURCES in scripts/runtime_representation_oracle.py \
             or remove the seam it carries"
        )));
    };
    let production = production_rust_source(source)?;
    let file = syn::parse_file(&production)
        .map_err(|error| ScanError::new(format!("cannot parse Rust source `{path}`: {error}")))?;
    let mut scanner = RustSeamScanner {
        class,
        defines_capacity_keys: path.ends_with("/dag.rs"),
        owners: Vec::new(),
        rows: Vec::new(),
        error: None,
    };
    scanner.visit_file(&file);
    match scanner.error {
        Some(error) => Err(error),
        None => Ok(scanner.rows),
    }
}

// ---------------------------------------------------------------------------
// C and Objective-C headers
// ---------------------------------------------------------------------------

pub mod c_ast;
pub use c_ast::{Configuration, HeaderLane, lane_for, scan_c_header, scan_c_source};
