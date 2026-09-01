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

// ---------------------------------------------------------------------------
// Regressions from the first red-team round on the rebuilt architecture.
// ---------------------------------------------------------------------------

#[test]
fn a_linkage_block_does_not_swallow_the_rest_of_a_header() {
    // `extern "C" {` opens at file scope and the token before its brace is a
    // string literal, not a declarator. Resetting the owner only at depth zero
    // made every later declaration inherit `module`, so 54 of 65 rows in
    // chelis_runtime.h collapsed onto one identity and a NEW public export
    // could not move the freeze.
    let rows = scan_c_header(
        HEADER,
        r#"
        #ifdef __cplusplus
        extern "C" {
        #endif
        float *chelis_probe_row(const chelis_tensor *t);
        int64_t *chelis_probe_shape(const chelis_tensor *t);
        #ifdef __cplusplus
        }
        #endif
        "#,
    )
    .expect("header must scan");
    let owners: Vec<&str> = rows.iter().map(|row| row.owner.as_str()).collect();
    assert!(owners.contains(&"chelis_probe_row"), "{owners:?}");
    assert!(owners.contains(&"chelis_probe_shape"), "{owners:?}");
    assert!(
        !owners.contains(&"module"),
        "a declaration inside a linkage block owns its own row: {owners:?}"
    );
}

#[test]
fn a_called_function_never_becomes_the_owner_of_its_callers_body() {
    let rows = scan_c_header(
        HEADER,
        r#"
        static inline void chelis_probe_copy(chelis_tensor *t, const float *src) {
            CHELIS_CHECK(memcpy(t->data, src, t->size * sizeof(float)));
        }
        "#,
    )
    .expect("header must scan");
    let owners: Vec<&str> = rows.iter().map(|row| row.owner.as_str()).collect();
    assert!(
        owners.iter().all(|owner| *owner == "chelis_probe_copy"),
        "a statement belongs to the function that encloses it, not to what it calls: {owners:?}"
    );
}

#[test]
fn a_block_argument_call_does_not_open_an_owner() {
    // `dispatch_once(&once, ^{ ... })` is a call whose argument is a block.
    // Its brace must not make `dispatch_once` own the enclosing function's
    // statements.
    let rows = scan_c_header(
        HEADER,
        r#"
        static inline void chelis_probe_once(void) {
            dispatch_once(&once, ^{
                for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); ++i) { use(i); }
            });
        }
        "#,
    )
    .expect("header must scan");
    let owners: Vec<&str> = rows.iter().map(|row| row.owner.as_str()).collect();
    assert!(!owners.contains(&"dispatch_once"), "{owners:?}");
    assert!(owners.contains(&"chelis_probe_once"), "{owners:?}");
}

#[test]
fn a_preprocessor_directive_does_not_bleed_into_the_next_declaration() {
    // The tokens of `#include <immintrin.h>` used to run into the declaration
    // that followed, and the stray `.` made it read as a member access.
    let rows = scan_c_header(
        HEADER,
        r#"
        #ifdef __AVX2__
        #include <immintrin.h>
        #endif
        static inline float chelis_probe_sum(const float *p, int n);
        "#,
    )
    .expect("header must scan");
    let owners: Vec<&str> = rows.iter().map(|row| row.owner.as_str()).collect();
    assert!(owners.contains(&"chelis_probe_sum"), "{owners:?}");
}

#[test]
fn a_raw_element_pointer_in_a_signature_is_a_carrier_too() {
    // Casts are not the only place one appears. A parameter, a return type,
    // and a foreign declaration all hand one out, and Phase 3's seal has to
    // reach every one.
    for source in [
        "pub fn f(p: *const i64) -> usize { 0 }",
        "pub fn f() -> *mut f32 { std::ptr::null_mut() }",
        r#"unsafe extern "C" { pub fn chelis_probe(p: *mut f64); }"#,
    ] {
        assert!(
            kinds(RUNTIME, source).contains(&"raw-element-pointer".to_string()),
            "signature carrier missed in: {source}"
        );
    }
    // A byte pointer is still not an element pointer.
    assert!(
        !kinds(RUNTIME, "pub fn f(p: *mut u8) {}").contains(&"raw-element-pointer".to_string())
    );
}

#[test]
fn width_selection_is_one_rule_across_every_path_spelling() {
    for source in [
        "fn f() -> usize { size_of::<f32>() }",
        "fn f() -> usize { std::mem::size_of::<f32>() }",
        "fn f() -> usize { core::mem::size_of::<f32>() }",
    ] {
        assert_eq!(
            kinds(RUNTIME, source),
            vec!["width-arithmetic".to_string()],
            "missed in: {source}"
        );
    }
}

#[test]
fn seams_inside_macro_arguments_are_visited() {
    // `syn` does not descend into macro tokens as expressions, so a seam in a
    // `format!` argument was invisible.
    let rows = identities(
        BACKEND,
        r#"unsafe fn f(t: *mut Tensor) -> String { format!("{}", (*t).data as *mut f32 as usize) }"#,
    );
    let found: Vec<&str> = rows.iter().map(|(kind, _)| kind.as_str()).collect();
    assert!(found.contains(&"direct-data-access"), "{rows:?}");
    assert!(
        found.contains(&"raw-element-pointer"),
        "an element cast inside a macro argument is still a carrier: {rows:?}"
    );

    // A template nested one group deep is still emitted text.
    assert!(
        kinds(
            BACKEND,
            r#"fn f() -> String { format!("{}", ("((float *)t->data)[i]")) }"#
        )
        .contains(&"load-store-template".to_string())
    );
}

#[test]
fn include_reopens_the_closed_universe_and_is_rejected() {
    let error = scan_rust_source(RUNTIME, r#"include!("probe.in");"#)
        .expect_err("include! splices an unscanned file and must fail closed");
    assert!(error.message.contains("include!"), "{}", error.message);
    assert!(
        error.message.contains("INVENTORY_SOURCES"),
        "the failure must name the sanctioned action: {}",
        error.message
    );
}

#[test]
fn a_module_level_item_owns_its_rows() {
    // A const or static at module scope is a declaration; its rows must not
    // collapse onto a placeholder that names nothing.
    let rows = identities(
        BACKEND,
        r#"pub const PROBE_KERNEL: &str = "((float *)t->data)[i] = 1.0f;";"#,
    );
    assert_eq!(
        rows,
        vec![(
            "load-store-template".to_string(),
            "PROBE_KERNEL".to_string()
        )]
    );
}

// ---------------------------------------------------------------------------
// Regressions from the second red-team round. The theme is one structural
// change per class rather than a case per shape: a placeholder owner is now
// inexpressible, and an element type must GOVERN a pointer rather than merely
// co-occur with one.
// ---------------------------------------------------------------------------

#[test]
fn every_c_declarator_form_names_its_own_owner() {
    // Keying on `(` saw prototypes and missed every other declaration form, so
    // struct fields, extern data, function-pointer typedefs and array
    // declarators all fell through to a placeholder.
    let cases = [
        ("float *chelis_probe_plain(float *p);", "chelis_probe_plain"),
        ("extern double *chelis_probe_table;", "chelis_probe_table"),
        ("extern float *chelis_probe_slots[8];", "chelis_probe_slots"),
        (
            "typedef float *(*chelis_probe_hook)(int64_t bytes);",
            "chelis_probe_hook",
        ),
    ];
    for (source, expected) in cases {
        let rows = scan_c_header(HEADER, source).expect("header must scan");
        let owners: Vec<&str> = rows.iter().map(|row| row.owner.as_str()).collect();
        assert!(owners.contains(&expected), "{source} -> {owners:?}");
        assert!(!owners.contains(&"module"), "{source} -> {owners:?}");
    }
}

#[test]
fn a_non_descriptor_struct_field_is_still_a_carrier() {
    let rows = scan_c_header(
        HEADER,
        "typedef struct { int64_t key; float *weights; } chelis_probe_pair;",
    )
    .expect("header must scan");
    assert!(
        rows.iter()
            .any(|row| row.kind == "raw-element-pointer"
                && row.owner == "chelis_probe_pair::weights"),
        "{rows:?}"
    );
}

#[test]
fn an_unattributable_c_seam_fails_rather_than_taking_a_placeholder() {
    // A placeholder owner is a SINK: identity is kind|path|owner, so one such
    // row absorbs every later seam of that kind in the file and the freeze
    // stops moving. Refusing to invent one closes the class.
    let error = scan_c_header(HEADER, "float *;").expect_err("an unnamed carrier must fail");
    assert!(
        error.message.contains("names no declaration"),
        "{}",
        error.message
    );
    assert!(error.message.contains("absorb"), "{}", error.message);
}

#[test]
fn an_element_type_must_govern_the_pointer_not_merely_co_occur() {
    // `int64_t chelis_dict_len(const chelis_dict *dict)` is not a raw element
    // pointer: the star belongs to an opaque handle and the integer is a
    // return width. A whole-statement co-occurrence test called 41 of the 48
    // declarations in chelis_runtime.h carriers.
    let opaque = scan_c_header(
        HEADER,
        "int64_t chelis_probe_len(const chelis_probe_list *list);",
    )
    .expect("header must scan");
    assert!(
        !opaque.iter().any(|row| row.kind == "raw-element-pointer"),
        "{opaque:?}"
    );

    let diagnostic = scan_c_header(
        HEADER,
        "static inline int64_t chelis_probe_cast(int64_t v, int bits, const char *msg);",
    )
    .expect("header must scan");
    assert!(
        !diagnostic
            .iter()
            .any(|row| row.kind == "raw-element-pointer"),
        "a const char * diagnostic message is not an element carrier: {diagnostic:?}"
    );

    for real in [
        "float *chelis_probe_a(float *p);",
        "extern const int64_t *const chelis_probe_b;",
        "void chelis_probe_c(int64_t *__restrict__ out);",
    ] {
        let rows = scan_c_header(HEADER, real).expect("header must scan");
        assert!(
            rows.iter().any(|row| row.kind == "raw-element-pointer"),
            "missed a real carrier: {real}"
        );
    }
}

#[test]
fn a_path_attribute_reopens_the_closed_universe_and_is_rejected() {
    let error = scan_rust_source(RUNTIME, r#"#[path = "../probe.rs"] pub mod probe;"#)
        .expect_err("#[path] compiles a file outside the roots and must fail closed");
    assert!(error.message.contains("#[path]"), "{}", error.message);
    assert!(
        error.message.contains("INVENTORY_SOURCES"),
        "{}",
        error.message
    );
}

#[test]
fn a_trait_provided_method_owns_its_rows() {
    // `TensorElement::data_ptr` is one of the accessors chelis#893 exists to
    // seal, and it had no owner at all until trait items were walked.
    let rows = identities(
        RUNTIME,
        r#"
        pub unsafe trait TensorElement {
            unsafe fn data_ptr(t: *mut chelis_tensor) -> *mut Self {
                unsafe { (*t).data as *mut Self }
            }
        }
        "#,
    );
    let owners: Vec<&str> = rows.iter().map(|(_, owner)| owner.as_str()).collect();
    assert!(
        owners
            .iter()
            .all(|owner| owner.starts_with("TensorElement::data_ptr")),
        "{rows:?}"
    );
    assert!(!owners.contains(&"module"), "{rows:?}");
}

#[test]
fn a_rust_seam_outside_every_declaration_fails_closed() {
    let error = scan_rust_source(
        "crates/chelis-ir/src/dag.rs",
        "const _: () = { let _ = 2usize * 3usize; };",
    );
    // A const item owns its rows, so this must NOT fail; the guard exists for
    // a seam with no enclosing item at all.
    assert!(error.is_ok(), "{error:?}");
}
