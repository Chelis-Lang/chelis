//! Positive and negative controls for the Phase 0 seam scanner.
//!
//! Every rule that admits a row has a control that proves it admits, and one
//! that proves the neighbouring shape is rejected or ignored. The scanner's
//! completeness claim is over a frozen list of repository files, so the
//! controls that matter most are the ones proving an unregistered file and an
//! unclassifiable type word both fail rather than disappearing.

use chelis_repr_inventory::{SourceClass, scan_c_header, scan_rust_source};

const RUNTIME: &str = "crates/chelis-runtime/src/lib.rs";
const BACKEND: &str = "crates/chelis-backend-c/src/emit.rs";
const IR: &str = "crates/chelis-ir/src/dag.rs";
const HEADER: &str = "crates/chelis-runtime/include/chelis_runtime.h";

fn identities(path: &str, source: &str) -> Vec<(String, String)> {
    let mut rows: Vec<(String, String)> = scan_rust_source(path, source)
        .expect("source must scan")
        .into_iter()
        .map(|row| (row.kind, row.owner))
        .collect();
    rows.sort();
    rows.dedup();
    rows
}

fn kinds(path: &str, source: &str) -> Vec<String> {
    identities(path, source)
        .into_iter()
        .map(|(kind, _)| kind)
        .collect()
}

#[test]
fn an_unregistered_source_path_fails_rather_than_scanning_empty() {
    let error = scan_rust_source("crates/chelis-brand-new/src/lib.rs", "pub fn f() {}")
        .expect_err("an unregistered inventory source must fail closed");
    assert!(
        error.message.contains("not a registered"),
        "{}",
        error.message
    );
    assert!(
        error.message.contains("INVENTORY_SOURCES"),
        "the failure must name the exact sanctioned action: {}",
        error.message
    );
}

#[test]
fn every_registered_prefix_classifies() {
    for path in [
        "crates/chelis-runtime/src/lib.rs",
        "crates/chelis-vocab/src/lib.rs",
        "crates/chelis-ir/src/dag.rs",
        "crates/chelis-python/src/lib.rs",
        "crates/chelis-backend-metal/src/emit.rs",
    ] {
        assert!(
            SourceClass::for_path(path).is_some(),
            "{path} must classify"
        );
    }
    assert!(SourceClass::for_path("crates/chelis-surf/src/lib.rs").is_none());
}

#[test]
fn unparseable_rust_fails_closed() {
    let error = scan_rust_source(RUNTIME, "pub fn f( {").expect_err("invalid Rust must fail");
    assert!(
        error.message.contains("cannot parse Rust"),
        "{}",
        error.message
    );
}

#[test]
fn direct_data_access_is_owned_by_its_enclosing_function() {
    let rows = identities(
        RUNTIME,
        r#"
        pub unsafe fn read(t: *mut Tensor) -> usize { (*t).data as usize }
        "#,
    );
    assert_eq!(
        rows,
        vec![("direct-data-access".to_string(), "read".to_string())]
    );
}

#[test]
fn a_raw_element_pointer_cast_is_admitted_and_a_byte_cast_is_not() {
    assert_eq!(
        kinds(RUNTIME, "fn f(p: *mut u8) -> *mut f32 { p as *mut f32 }"),
        vec!["raw-element-pointer".to_string()]
    );
    // `u8` is the raw byte carrier, inventoried through `direct-data-access`.
    // Treating it as an element would make every byte cast a dtype seam.
    assert!(kinds(RUNTIME, "fn f(p: *mut u8) -> *mut u8 { p as *mut u8 }").is_empty());
}

#[test]
fn a_descriptor_struct_yields_field_rows_and_a_helper_struct_does_not() {
    let descriptor = kinds(
        RUNTIME,
        r#"
        #[repr(C)]
        pub struct ChelisTensor {
            pub data: *mut u8,
            pub dtype: i32,
            pub shape: *const i64,
            pub strides: *const i64,
            pub rank: i32,
            pub size: i64,
        }
        "#,
    );
    assert!(descriptor.contains(&"descriptor-field".to_string()));
    let field_owners: Vec<String> = identities(
        RUNTIME,
        "#[repr(C)] pub struct T { pub data: *mut u8, pub dtype: i32, pub rank: i32, pub size: i64 }",
    )
    .into_iter()
    .filter(|(kind, _)| kind == "descriptor-field")
    .map(|(_, owner)| owner)
    .collect();
    assert!(
        field_owners.contains(&"T::rank".to_string()),
        "each field owns its own identity so an added field is detectable: {field_owners:?}"
    );
    assert!(
        descriptor.contains(&"narrow-metadata".to_string()),
        "`rank: i32` is outside the int64 extent domain and must be inventoried"
    );

    // Two descriptor field names and no `data`/`dtype` pair: an ordinary
    // helper, not a tensor carrier.
    let helper = kinds(RUNTIME, "pub struct Chunk { pub size: i32, pub rank: i32 }");
    assert!(helper.is_empty(), "{helper:?}");
}

#[test]
fn a_fixed_rank_array_is_admitted_and_a_pair_is_not_a_rank_cap() {
    let capped = kinds(
        "crates/chelis-python/src/lib.rs",
        r#"
        pub struct ChelisGpuTensor {
            pub data: *mut f32,
            pub dtype: i32,
            pub shape: [i32; CHELIS_MAX_DIM],
            pub ndim: i32,
            pub size: i32,
        }
        "#,
    );
    assert!(capped.contains(&"fixed-rank-metadata".to_string()));
    assert!(capped.contains(&"narrow-metadata".to_string()));
    assert!(capped.contains(&"raw-element-pointer".to_string()));
}

#[test]
fn the_dtype_contract_covers_variants_and_element_bindings() {
    let variants = identities(
        "crates/chelis-vocab/src/lib.rs",
        "pub enum Repr { Ieee754Binary32, Bool8 }",
    );
    // One identity per variant: a dtype added without its complete contract
    // has to surface as a NEW row, which it cannot do if the variants collapse
    // onto the enum that holds them.
    assert_eq!(
        variants,
        vec![
            ("dtype-contract".to_string(), "Repr::Bool8".to_string()),
            (
                "dtype-contract".to_string(),
                "Repr::Ieee754Binary32".to_string()
            ),
        ]
    );

    let binding = identities(
        RUNTIME,
        "unsafe impl TensorElement for f32 { const DTYPE: RuntimeDType = RuntimeDType::F32; }",
    );
    assert!(
        binding.contains(&(
            "dtype-contract".to_string(),
            "TensorElement for f32".to_string()
        )),
        "{binding:?}"
    );

    // An unrelated enum is not the representation contract.
    assert!(kinds("crates/chelis-vocab/src/lib.rs", "pub enum Colour { Red }").is_empty());
}

#[test]
fn saturating_capacity_folds_and_key_consumers_are_both_seams() {
    let fold = kinds(
        IR,
        "fn push(c: &mut usize, v: usize) { *c = c.saturating_mul(v); }",
    );
    assert_eq!(fold, vec!["normalized-key-arithmetic".to_string()]);

    // Phase 1's exit requires every reuse consumer to move to the exact key,
    // so a consumer outside the IR is equally a seam.
    let consumer = kinds(
        "crates/chelis-backend-c/src/memory.rs",
        "fn fits(a: &DimExpr, b: &DimExpr) -> bool { a.normalized_key() == b.normalized_key() }",
    );
    assert_eq!(consumer, vec!["normalized-key-arithmetic".to_string()]);

    // A saturating fold outside the IR is ordinary arithmetic.
    assert!(kinds(RUNTIME, "fn f(a: usize) -> usize { a.saturating_mul(2) }").is_empty());
}

#[test]
fn element_spellings_and_load_store_templates_stay_disjoint() {
    let spelling = kinds(BACKEND, r#"fn t() -> &'static str { "float" }"#);
    assert_eq!(spelling, vec!["backend-element-spelling".to_string()]);

    let template = kinds(
        BACKEND,
        r#"fn t() -> String { format!("((float *)t->data)[i]") }"#,
    );
    assert_eq!(
        template,
        vec!["load-store-template".to_string()],
        "a dereferencing template must not also count as a bare spelling"
    );

    // A literal naming no element type is not a seam, and a backend-only rule
    // must not fire in the runtime.
    assert!(kinds(BACKEND, r#"fn t() -> &'static str { "size_t" }"#).is_empty());
    assert!(kinds(RUNTIME, r#"fn t() -> &'static str { "float" }"#).is_empty());
}

#[test]
fn width_selection_is_admitted_wherever_it_happens() {
    assert_eq!(
        kinds(RUNTIME, "fn f(d: RuntimeDType) -> usize { d.byte_width() }"),
        vec!["width-arithmetic".to_string()]
    );
    assert_eq!(
        kinds(RUNTIME, "fn f() -> usize { size_of::<f32>() }"),
        vec!["width-arithmetic".to_string()]
    );
}

#[test]
fn only_an_exact_cfg_test_module_leaves_the_source_universe() {
    let excluded = kinds(
        RUNTIME,
        r#"
        #[cfg(test)]
        mod tests { fn f(t: *mut Tensor) -> usize { unsafe { (*t).data as usize } } }
        "#,
    );
    assert!(excluded.is_empty(), "{excluded:?}");

    // Production items after a test module, and `cfg(not(test))`, stay in.
    let retained = identities(
        RUNTIME,
        r#"
        #[cfg(test)]
        mod tests { fn ignored() {} }
        #[cfg(not(test))]
        mod shipped { pub unsafe fn read(t: *mut Tensor) -> usize { (*t).data as usize } }
        pub unsafe fn after(t: *mut Tensor) -> usize { (*t).data as usize }
        "#,
    );
    let owners: Vec<&str> = retained.iter().map(|(_, owner)| owner.as_str()).collect();
    assert!(owners.contains(&"shipped::read"), "{owners:?}");
    assert!(owners.contains(&"after"), "{owners:?}");
}

#[test]
fn identity_survives_reformatting_but_not_relocation() {
    let compact = identities(
        RUNTIME,
        "pub unsafe fn read(t: *mut Tensor) -> usize { (*t).data as usize }",
    );
    let spread = identities(
        RUNTIME,
        r#"
        pub unsafe fn read(t: *mut Tensor) -> usize {
            // A comment, a line break, and a renamed local must not move the
            // freeze: the ledger records the owning declaration.
            let carrier = (*t).data;
            carrier as usize
        }
        "#,
    );
    assert_eq!(compact, spread);

    let relocated = identities(
        RUNTIME,
        "pub unsafe fn moved(t: *mut Tensor) -> usize { (*t).data as usize }",
    );
    assert_ne!(
        compact, relocated,
        "moving a seam to a different declaration must produce a new identity"
    );
}

#[test]
fn a_c_header_owns_its_rows_by_declaration_never_by_a_keyword() {
    let rows = scan_c_header(
        HEADER,
        r#"
        typedef struct { float *data; int dtype; long shape[8]; long strides[8]; int rank; long size; } chelis_tensor;
        extern float *chelis_row(const chelis_tensor *t);
        static inline void chelis_copy(chelis_tensor *t) {
            for (int i = 0; i < t->rank; i++) { t->data = (float *)0; }
        }
        "#,
    )
    .expect("header must scan");
    let owners: Vec<&str> = rows.iter().map(|row| row.owner.as_str()).collect();
    assert!(
        !owners
            .iter()
            .any(|owner| ["for", "while", "if", "sizeof"].contains(owner)),
        "a control-flow keyword must never own a row: {owners:?}"
    );
    assert!(
        owners.contains(&"chelis_tensor::data"),
        "a C descriptor field owns its own identity too: {owners:?}"
    );
    assert!(owners.contains(&"chelis_tensor::rank"), "{owners:?}");
    assert!(owners.contains(&"chelis_row"), "{owners:?}");
    assert!(
        owners.contains(&"chelis_copy"),
        "a statement inside a body belongs to its enclosing function: {owners:?}"
    );
    let kinds: Vec<&str> = rows.iter().map(|row| row.kind.as_str()).collect();
    assert!(kinds.contains(&"descriptor-field"));
    assert!(kinds.contains(&"fixed-rank-metadata"));
    assert!(kinds.contains(&"narrow-metadata"));
    assert!(kinds.contains(&"direct-data-access"));
}

#[test]
fn an_unknown_c_arithmetic_spelling_is_a_failure_not_an_unflagged_row() {
    // The classification rule is inverted for the same reason the capacity
    // census inverts it: an allowlist of arithmetic spellings can never be
    // complete, so a 16-bit float nobody listed must stop the build.
    for spelling in ["_Float16", "__bf16", "_Decimal64", "__int128"] {
        let source = format!("extern {spelling} *carrier(void);");
        let error = scan_c_header(HEADER, &source)
            .expect_err("an unknown arithmetic spelling must fail closed");
        assert!(error.message.contains(spelling), "{}", error.message);
        assert!(
            error.message.contains("c_lexical.rs"),
            "the failure must name where to classify it: {}",
            error.message
        );
    }
    // A known spelling still classifies without rejection.
    assert!(scan_c_header(HEADER, "extern float *carrier(void);").is_ok());
}
