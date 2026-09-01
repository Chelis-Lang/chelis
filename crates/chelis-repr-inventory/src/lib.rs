//! Structural seam scanner for the chelis#893 runtime-representation inventory.
//!
//! `spec/design/runtime_representation.md` section C6 names the seam classes
//! Phase 0 must freeze: descriptor fields, `data` access, pointer casts,
//! width/arithmetic matches, `normalized_key` arithmetic, fixed-rank arrays,
//! narrow metadata fields, backend element spellings, and load/store
//! templates. This crate derives those rows and nothing else.
//!
//! # Why there is no C-family compiler front end here
//!
//! The inventory's universe is a frozen list of repository files, not a
//! language. Nineteen of them are Rust and four are plain C headers; the
//! repository contains no C++, Objective-C, CUDA, or Metal source at all.
//! Rust is read with `syn`, which is a total parser for the language, so a
//! completeness claim over Rust is one that can actually be discharged. The
//! four C headers are read with the same token vocabulary the numeric capacity
//! census uses (`tests/support/c_lexical.rs`), because that instrument is
//! already the reviewed authority for classifying a C type word and it keeps
//! one classifier rather than two.
//!
//! A seam row is owned by its enclosing declaration, not by a source offset or
//! a hash of the exact bytes: Phases 3 and 4 delete functions and call sites,
//! so that is the granularity the ledger records. Reformatting a literal
//! inside a function does not churn the freeze; adding a new function that
//! carries a seam does.

#[path = "../../../tests/support/c_lexical.rs"]
pub mod c_lexical;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use quote::ToTokens;
use serde::{Deserialize, Serialize};
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
    "Bf16Bits", "Bool8", "F16Bits", "bf16", "f16", "f32", "f64", "i16", "i32", "i64", "i8", "u16",
    "u32", "u64",
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

/// Does this Rust type name a tensor element?
fn is_rust_element_type(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Path(path) => path.path.segments.last().is_some_and(|segment| {
            RUST_ELEMENT_TYPES.contains(&segment.ident.to_string().as_str())
        }),
        syn::Type::Paren(inner) => is_rust_element_type(&inner.elem),
        _ => false,
    }
}

/// A pointer whose pointee is an element type is a raw element pointer however
/// many `const`/`mut` layers sit above it.
fn is_raw_element_pointer(ty: &syn::Type) -> bool {
    match ty {
        syn::Type::Ptr(pointer) => {
            is_rust_element_type(&pointer.elem) || is_raw_element_pointer(&pointer.elem)
        }
        syn::Type::Paren(inner) => is_raw_element_pointer(&inner.elem),
        _ => false,
    }
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

/// A load/store template dereferences or indexes; an element spelling only
/// names the type. Keeping them disjoint means a row cannot be counted twice.
fn is_load_store_template(text: &str) -> bool {
    text.contains("->") || text.contains('[') || text.contains("*(") || text.contains(")[")
}

struct RustSeamScanner {
    class: SourceClass,
    owners: Vec<String>,
    rows: Vec<SeamRow>,
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
    fn visit_attribute(&mut self, _attribute: &'ast syn::Attribute) {}

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
            visit::visit_item_struct(scanner, item);
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

    fn visit_expr_cast(&mut self, cast: &'ast syn::ExprCast) {
        if is_raw_element_pointer(&cast.ty) {
            self.push("raw-element-pointer", cast.to_token_stream().to_string());
        }
        visit::visit_expr_cast(self, cast);
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let method = call.method.to_string();
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
        let tail = rendered.rsplit("::").next().unwrap_or_default().trim();
        if WIDTH_SELECTORS.contains(&tail) || rendered.starts_with("size_of") {
            self.push("width-arithmetic", call.to_token_stream().to_string());
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_expr_binary(&mut self, binary: &'ast syn::ExprBinary) {
        // [#888]'s mechanism is an unchecked or saturating product folded into
        // a capacity key. Only the IR's key path is inventoried; ordinary
        // arithmetic elsewhere is not a representation seam.
        if matches!(self.class, SourceClass::Ir)
            && matches!(binary.op, syn::BinOp::Mul(_))
            && self.owner().contains("key")
        {
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

    fn visit_macro(&mut self, macro_call: &'ast syn::Macro) {
        // A `format!`/`write!` template is emitted C text; its literal is the
        // seam, and the interpolated arguments are ordinary expressions.
        for token in macro_call.tokens.clone() {
            if let proc_macro2::TokenTree::Literal(literal) = token
                && let Ok(syn::Lit::Str(text)) = syn::parse_str(&literal.to_string())
            {
                self.scan_literal(&text.value());
            }
        }
        visit::visit_macro(self, macro_call);
    }
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
        owners: Vec::new(),
        rows: Vec::new(),
    };
    scanner.visit_file(&file);
    Ok(scanner.rows)
}

// ---------------------------------------------------------------------------
// C headers
// ---------------------------------------------------------------------------

/// Is this token a C type word the census vocabulary does not know?
///
/// The classification rule is inverted for the same reason
/// `spec/design/dtype_semantics.md` section C6 inverts it: an allowlist of
/// arithmetic spellings can never be complete, so an unrecognized type word at
/// a carrier position is a failure rather than a silently unflagged row.
fn is_unknown_c_type_word(word: &str, aliases: &BTreeMap<String, Vec<String>>) -> bool {
    if !word
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
    {
        return false;
    }
    if c_lexical::NUMERIC_C_TYPES.contains(&word)
        || c_lexical::NON_NUMERIC_C_TYPE_WORDS.contains(&word)
        || C_ELEMENT_TYPES.contains(&word)
        || aliases.contains_key(word)
    {
        return false;
    }
    // Reserved arithmetic spellings that no vocabulary lists but that a
    // compiler would accept as a dtype.
    word.starts_with("_Float")
        || word.starts_with("_Decimal")
        || word.starts_with("__fp")
        || word.starts_with("__bf")
        || word.starts_with("__int")
}

/// C keywords and operators that can precede a `(` without naming anything.
/// Without this, a `for`, `while`, or `sizeof` becomes the owner of every row
/// in the block it opens.
const C_NON_DECLARATOR_WORDS: &[&str] = &[
    "case", "defined", "do", "else", "for", "if", "return", "sizeof", "switch", "while",
];

/// Derive seam rows from one plain C header.
///
/// The universe is four reviewed headers with no conditional ABI, so this
/// walks the declaration token stream directly rather than enumerating
/// preprocessor configurations. A header that grows a conditional declaration
/// region is caught by the capacity census's context-invariance guard, which
/// is the reviewed owner of that question; it is not re-litigated here.
pub fn scan_c_header(path: &str, source: &str) -> Result<Vec<SeamRow>, ScanError> {
    let aliases = BTreeMap::new();
    let stripped = c_lexical::strip_c_comments(source);
    let tokens = c_lexical::lex_c_tokens(&stripped);
    let mut rows = c_descriptor_rows(&stripped);
    let mut depth = 0usize;
    let mut enclosing = String::from("module");
    let mut statement: Vec<String> = Vec::new();
    let mut candidate: Option<String> = None;
    let mut pending_block_owner: Option<String> = None;

    for (index, token) in tokens.iter().enumerate() {
        if is_unknown_c_type_word(token, &aliases) {
            return Err(ScanError::new(format!(
                "`{path}` uses the unrecognized C type word `{token}` at a carrier position: \
                 classify it in tests/support/c_lexical.rs rather than routing around it"
            )));
        }
        match token.as_str() {
            "(" => {
                if let Some(previous) = index.checked_sub(1).and_then(|before| tokens.get(before))
                    && is_c_declarator_name(previous)
                {
                    candidate = Some(previous.clone());
                    pending_block_owner = Some(previous.clone());
                }
            }
            "{" => {
                if depth == 0
                    && let Some(owner) = pending_block_owner.take()
                {
                    enclosing = owner;
                }
                depth += 1;
                flush_c_statement(&mut rows, &mut statement, &enclosing, &mut candidate, depth);
            }
            "}" => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    enclosing = String::from("module");
                }
                flush_c_statement(&mut rows, &mut statement, &enclosing, &mut candidate, depth);
            }
            ";" => {
                flush_c_statement(&mut rows, &mut statement, &enclosing, &mut candidate, depth);
                pending_block_owner = None;
            }
            _ => statement.push(token.clone()),
        }
    }
    flush_c_statement(&mut rows, &mut statement, &enclosing, &mut candidate, depth);
    Ok(rows)
}

/// A declarator name is an ordinary identifier, never a control-flow keyword
/// and never a type word.
fn is_c_declarator_name(token: &str) -> bool {
    !C_NON_DECLARATOR_WORDS.contains(&token)
        && !c_lexical::NON_NUMERIC_C_TYPE_WORDS.contains(&token)
        && !c_lexical::NUMERIC_C_TYPES.contains(&token)
        && token
            .chars()
            .next()
            .is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
        && token
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
}

/// Emit the seam rows one C statement carries.
///
/// At file scope a prototype owns its own name; inside a function body the
/// enclosing function owns every statement, which is the granularity Phase 3
/// deletes at.
fn flush_c_statement(
    rows: &mut Vec<SeamRow>,
    statement: &mut Vec<String>,
    enclosing: &str,
    candidate: &mut Option<String>,
    depth: usize,
) {
    let words = std::mem::take(statement);
    let declared = candidate.take();
    if words.is_empty() {
        return;
    }
    let owner = if depth > 0 {
        enclosing.to_string()
    } else {
        declared.unwrap_or_else(|| enclosing.to_string())
    };
    let text = words.join(" ");
    let has_element = words
        .iter()
        .any(|word| C_ELEMENT_TYPES.contains(&word.as_str()) || word == "int" || word == "long");
    if words.iter().any(|word| word == "*") && has_element {
        rows.push(SeamRow::new("raw-element-pointer", &owner, &text));
    }
    if words
        .windows(2)
        .any(|pair| pair[1] == "data" && (pair[0] == "->" || pair[0] == "."))
    {
        rows.push(SeamRow::new("direct-data-access", &owner, &text));
    }
    if words.iter().any(|word| word == "sizeof") {
        rows.push(SeamRow::new("width-arithmetic", &owner, &text));
    }
}

/// Descriptor structs in a C header: `typedef struct { ... } name;`.
fn c_descriptor_rows(stripped: &str) -> Vec<SeamRow> {
    let mut rows = Vec::new();
    let mut rest = stripped;
    while let Some(start) = rest.find("typedef struct") {
        let after = &rest[start..];
        let Some(open) = after.find('{') else { break };
        let Some(close) = after.find('}') else { break };
        if close < open {
            rest = &after[close + 1..];
            continue;
        }
        let body = &after[open + 1..close];
        let tail = &after[close + 1..];
        let name = tail
            .split(';')
            .next()
            .unwrap_or_default()
            .trim()
            .to_string();
        let declarations: Vec<String> = body
            .split(';')
            .map(|declaration| declaration.split_whitespace().collect::<Vec<_>>().join(" "))
            .filter(|declaration| !declaration.is_empty())
            .collect();
        let names: BTreeSet<String> = declarations
            .iter()
            .filter_map(|declaration| c_field_name(declaration))
            .collect();
        if is_descriptor(&names) && !name.is_empty() {
            for declaration in &declarations {
                let Some(field) = c_field_name(declaration) else {
                    rows.push(SeamRow::new("descriptor-field", &name, declaration));
                    continue;
                };
                let name = format!("{name}::{field}");
                rows.push(SeamRow::new("descriptor-field", &name, declaration));
                let is_narrow = declaration
                    .split_whitespace()
                    .any(|word| word == "int" || word == "int32_t");
                if NARROWABLE_FIELDS.contains(&field.as_str()) && is_narrow {
                    rows.push(SeamRow::new("narrow-metadata", &name, declaration));
                }
                if let Some(extent) = c_fixed_extent(declaration) {
                    rows.push(SeamRow::new(
                        "fixed-rank-metadata",
                        &name,
                        format!("{declaration} (extent {extent})"),
                    ));
                }
            }
        }
        rest = &after[close + 1..];
    }
    rows
}

fn c_field_name(declaration: &str) -> Option<String> {
    let head = declaration.split('[').next()?;
    let name = head
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .rfind(|word| !word.is_empty())?;
    name.chars()
        .next()
        .is_some_and(|character| character.is_ascii_alphabetic() || character == '_')
        .then(|| name.to_string())
}

fn c_fixed_extent(declaration: &str) -> Option<String> {
    let open = declaration.find('[')?;
    let close = declaration[open..].find(']')? + open;
    let extent = declaration[open + 1..close].trim();
    if extent.is_empty() {
        return None;
    }
    let is_cap = extent.contains("MAX_DIM") || extent.parse::<u64>().is_ok_and(|value| value > 1);
    is_cap.then(|| extent.to_string())
}
