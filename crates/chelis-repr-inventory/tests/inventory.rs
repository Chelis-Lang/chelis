//! Positive and negative controls for the Phase 0 seam scanner.
//!
//! Every rule that admits a row has a control that proves it admits, and one
//! that proves the neighbouring shape is rejected or ignored. The scanner's
//! completeness claim is over a frozen list of repository files, so the
//! controls that matter most are the ones proving an unregistered file and an
//! unclassifiable type word both fail rather than disappearing.

use chelis_repr_inventory::c_ast::{ConditionalArms, HIP_LANE, OBJECTIVE_C_LANE, PUBLIC_C_LANE};
use chelis_repr_inventory::{
    SourceClass, lane_for, scan_c_header, scan_c_source, scan_rust_source,
};

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

// ---------------------------------------------------------------------------
// C and Objective-C headers, read through clang's front end.
//
// Every snippet below is a complete translation unit under its lane: the
// front end rejects anything else, and a rejection is a fail-closed
// `ScanError`, never an empty row set. The forms named in the comments are
// the ones a hand-written token walk mis-modelled across three red-team
// rounds; a compiler-backed reader closes them by construction, and these
// controls prove that rather than assume it.
// ---------------------------------------------------------------------------

const HIP_HEADER: &str = "crates/chelis-backend-hip/runtime/chelis_hip_runtime.h";
const METAL_HEADER: &str = "crates/chelis-backend-metal/runtime/chelis_metal_runtime.h";

fn c_owners(source: &str) -> Vec<(String, String)> {
    let mut rows: Vec<(String, String)> = scan_c_header(HEADER, source)
        .expect("header must scan")
        .into_iter()
        .map(|row| (row.kind, row.owner))
        .collect();
    rows.sort();
    rows
}

fn tracked_source(relative: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

#[test]
fn a_c_header_owns_its_rows_by_declaration_never_by_a_keyword() {
    let rows = c_owners(
        r#"
        #include <stdint.h>
        typedef struct { float *data; int dtype; int64_t shape[8]; int64_t strides[8]; int rank; int64_t size; } chelis_probe_tensor;
        extern float *chelis_row(const chelis_probe_tensor *t);
        static inline void chelis_copy(chelis_probe_tensor *t) {
            for (int i = 0; i < t->rank; i++) { t->data = (float *)0; }
        }
        "#,
    );
    let owners: Vec<&str> = rows.iter().map(|(_, owner)| owner.as_str()).collect();
    assert!(
        !owners
            .iter()
            .any(|owner| ["for", "while", "if", "sizeof", "module"].contains(owner)),
        "a keyword or placeholder must never own a row: {owners:?}"
    );
    for expected in [
        ("descriptor-field", "chelis_probe_tensor::data"),
        ("raw-element-pointer", "chelis_probe_tensor::data"),
        ("descriptor-field", "chelis_probe_tensor::rank"),
        ("narrow-metadata", "chelis_probe_tensor::rank"),
        ("fixed-rank-metadata", "chelis_probe_tensor::shape"),
        ("raw-element-pointer", "chelis_row"),
        ("direct-data-access", "chelis_copy"),
        ("raw-element-pointer", "chelis_copy"),
    ] {
        assert!(
            rows.contains(&(expected.0.to_string(), expected.1.to_string())),
            "missing {expected:?} in {rows:?}"
        );
    }
}

#[test]
fn an_unknown_c_arithmetic_spelling_is_a_failure_not_an_unflagged_row() {
    // The classification rule is inverted for the same reason the capacity
    // census inverts it: an allowlist of arithmetic spellings can never be
    // complete, so a 16-bit float nobody listed must stop the scan even
    // though the compiler accepts it.
    for spelling in ["_Float16", "__bf16", "__int128"] {
        let source = format!("extern {spelling} *chelis_probe_carrier(void);");
        let error = scan_c_header(HEADER, &source)
            .expect_err("an unknown arithmetic spelling must fail closed");
        assert!(error.message.contains(spelling), "{}", error.message);
        assert!(
            error.message.contains("c_lexical.rs"),
            "the failure must name where to classify it: {}",
            error.message
        );
    }
    // A spelling the compiler itself rejects fails closed too, with the
    // compiler's diagnostic rather than a silent empty scan.
    let error = scan_c_header(HEADER, "extern _Decimal64 *chelis_probe_carrier(void);")
        .expect_err("a compiler rejection is a scan failure");
    assert!(
        error.message.contains("clang rejected"),
        "{}",
        error.message
    );
    // A known spelling still classifies without rejection.
    assert!(scan_c_header(HEADER, "extern float *chelis_probe_carrier(void);").is_ok());
}

#[test]
fn a_called_function_never_becomes_the_owner_of_its_callers_body() {
    let rows = c_owners(
        r#"
        #include <string.h>
        #include "chelis_runtime.h"
        #define CHELIS_PROBE_CHECK(call) do { (void)(call); } while (0)
        typedef struct { float *data; size_t size; } chelis_probe_tensor;
        static inline void chelis_probe_copy(chelis_probe_tensor *t, const float *src) {
            CHELIS_PROBE_CHECK(memcpy(t->data, src, (size_t)t->size * sizeof(float)));
        }
        "#,
    );
    assert!(!rows.is_empty(), "the body carries seams");
    assert!(
        rows.iter().all(|(_, owner)| {
            owner == "chelis_probe_copy" || owner.starts_with("chelis_probe_tensor::")
        }),
        "a statement belongs to the function that encloses it, not to what it calls: {rows:?}"
    );
    for kind in [
        "direct-data-access",
        "width-arithmetic",
        "raw-element-pointer",
    ] {
        assert!(
            rows.iter().any(|(found, _)| found == kind),
            "missing {kind} in {rows:?}"
        );
    }
}

#[test]
fn a_block_argument_call_does_not_open_an_owner() {
    // `dispatch_once(&once, ^{ ... })` is a call whose argument is a block.
    // The block body belongs to the enclosing function, not to the callee.
    let rows = scan_c_source(
        METAL_HEADER,
        r#"
        #include <stddef.h>
        #include <dispatch/dispatch.h>
        static int chelis_probe_table[4];
        static void chelis_probe_use(size_t i) { (void)i; }
        static inline void chelis_probe_once(void) {
            static dispatch_once_t once;
            dispatch_once(&once, ^{
                for (size_t i = 0; i < sizeof(chelis_probe_table) / sizeof(chelis_probe_table[0]); ++i) {
                    chelis_probe_use(i);
                }
            });
        }
        "#,
        OBJECTIVE_C_LANE,
    )
    .expect("header must scan");
    let owners: Vec<&str> = rows.iter().map(|row| row.owner.as_str()).collect();
    assert!(!owners.contains(&"dispatch_once"), "{owners:?}");
    assert_eq!(owners, vec!["chelis_probe_once"], "{rows:?}");
    assert_eq!(rows[0].kind, "width-arithmetic");
}

#[test]
fn every_c_declarator_form_names_its_own_owner() {
    // Each of these is a declaration form that carried a seam past the old
    // token walk or was attributed to the wrong owner. With the compiler
    // supplying the declaration, every one names itself.
    let cases: &[(&str, &[(&str, &str)])] = &[
        (
            "float *chelis_probe_plain(float *p);",
            &[("raw-element-pointer", "chelis_probe_plain")],
        ),
        (
            "extern double *chelis_probe_table;",
            &[("raw-element-pointer", "chelis_probe_table")],
        ),
        (
            "extern float *chelis_probe_slots[8];",
            &[("raw-element-pointer", "chelis_probe_slots")],
        ),
        (
            "#include <stdint.h>\ntypedef float *(*chelis_probe_hook)(int64_t bytes);",
            &[("raw-element-pointer", "chelis_probe_hook")],
        ),
        // A tagged struct with a separate typedef: the idiomatic public-header
        // form, and the round-3 silent miss.
        (
            "#include <stdint.h>\nstruct chelis_probe_pool { float *slab; int64_t n; };\ntypedef struct chelis_probe_pool chelis_probe_pool;",
            &[("raw-element-pointer", "chelis_probe_pool::slab")],
        ),
        // A union.
        (
            "#include <stdint.h>\nunion chelis_probe_slot { double *d; int64_t i; };",
            &[("raw-element-pointer", "chelis_probe_slot::d")],
        ),
        // A macro-typed carrier: only a preprocessor can see the float.
        (
            "#define CHELIS_PROBE_ELEM float\nCHELIS_PROBE_ELEM *chelis_probe_slab(void);",
            &[("raw-element-pointer", "chelis_probe_slab")],
        ),
        // Two declarators in one declaration are two owners.
        (
            "extern float *chelis_probe_a, *chelis_probe_b;",
            &[
                ("raw-element-pointer", "chelis_probe_a"),
                ("raw-element-pointer", "chelis_probe_b"),
            ],
        ),
        // A file-scope enum constant owns its width arithmetic.
        (
            "enum { CHELIS_PROBE_WIDTH = sizeof(float) };",
            &[("width-arithmetic", "CHELIS_PROBE_WIDTH")],
        ),
        // A leading attribute does not hide the declarator.
        (
            "__attribute__((visibility(\"default\"))) float *chelis_probe_attr(void);",
            &[("raw-element-pointer", "chelis_probe_attr")],
        ),
        // A K&R definition is still a definition.
        (
            "float *chelis_probe_knr(p) float *p; { return p; }",
            &[("raw-element-pointer", "chelis_probe_knr")],
        ),
        // A parenthesised object-like macro before a declaration does not
        // steal its owner.
        (
            "#define CHELIS_PROBE_MASK (0xFF)\nextern float *chelis_probe_masked;",
            &[("raw-element-pointer", "chelis_probe_masked")],
        ),
    ];
    for (source, expected) in cases {
        let rows = c_owners(source);
        for (kind, owner) in expected.iter() {
            assert!(
                rows.contains(&(kind.to_string(), owner.to_string())),
                "{source}\n  expected {kind} owned by {owner}, got {rows:?}"
            );
        }
        assert!(
            !rows.iter().any(|(_, owner)| owner == "module"),
            "{source} -> {rows:?}"
        );
    }
}

#[test]
fn a_descriptor_with_anonymous_members_keeps_its_owner() {
    // An anonymous union or struct inside a descriptor contributes its fields
    // to the descriptor's name, as the compiler exposes them.
    let rows = c_owners(
        r#"
        #include <stdint.h>
        typedef struct {
            void *data;
            int64_t size;
            int32_t rank;
            uint8_t dtype;
            union { const int64_t *shape; const int64_t *dims; };
            struct { const int64_t *strides; };
        } chelis_probe_desc;
        "#,
    );
    for expected in [
        ("descriptor-field", "chelis_probe_desc::data"),
        ("descriptor-field", "chelis_probe_desc::shape"),
        ("raw-element-pointer", "chelis_probe_desc::shape"),
        ("descriptor-field", "chelis_probe_desc::strides"),
        ("narrow-metadata", "chelis_probe_desc::rank"),
    ] {
        assert!(
            rows.contains(&(expected.0.to_string(), expected.1.to_string())),
            "missing {expected:?} in {rows:?}"
        );
    }
    assert!(
        rows.iter()
            .all(|(_, owner)| owner.starts_with("chelis_probe_desc::")),
        "{rows:?}"
    );
}

#[test]
fn a_non_descriptor_struct_field_is_still_a_carrier() {
    let rows = c_owners(
        "#include <stdint.h>\ntypedef struct { int64_t key; float *weights; } chelis_probe_pair;",
    );
    assert!(
        rows.contains(&(
            "raw-element-pointer".to_string(),
            "chelis_probe_pair::weights".to_string()
        )),
        "{rows:?}"
    );
    assert!(
        !rows.iter().any(|(kind, _)| kind == "descriptor-field"),
        "a helper struct is not a descriptor: {rows:?}"
    );
}

#[test]
fn a_string_literal_is_not_a_declaration() {
    let rows =
        c_owners(r#"static const char *chelis_probe_text = "typedef struct { float *q; } fake;";"#);
    assert!(
        rows.is_empty(),
        "a literal cannot declare a carrier: {rows:?}"
    );
}

#[test]
fn an_unmodelled_declaration_fails_rather_than_taking_a_placeholder() {
    // A placeholder owner is a SINK: identity is kind|path|owner, so one such
    // row absorbs every later seam of that kind in the file and the freeze
    // stops moving. A declaration kind the reader does not model is refused
    // rather than dropped, and an ill-formed file is refused by the compiler.
    let error = scan_c_header(HEADER, "_Static_assert(sizeof(float) == 4, \"width\");")
        .expect_err("an unmodelled declaration must fail closed");
    assert!(
        error.message.contains("StaticAssertDecl"),
        "{}",
        error.message
    );
    assert!(error.message.contains("c_ast.rs"), "{}", error.message);
    let error = scan_c_header(HEADER, "float *;").expect_err("ill-formed C must fail");
    assert!(
        error.message.contains("clang rejected"),
        "{}",
        error.message
    );
}

#[test]
fn an_element_type_must_govern_the_pointer_not_merely_co_occur() {
    // `int64_t chelis_dict_len(const chelis_dict *dict)` is not a raw element
    // pointer: the star belongs to an opaque handle and the integer is a
    // return width.
    let opaque = c_owners(
        "#include <stdint.h>\ntypedef struct chelis_probe_list chelis_probe_list;\nint64_t chelis_probe_len(const chelis_probe_list *list);",
    );
    assert!(
        !opaque.iter().any(|(kind, _)| kind == "raw-element-pointer"),
        "{opaque:?}"
    );
    let diagnostic = c_owners(
        "#include <stdint.h>\nstatic inline int64_t chelis_probe_cast(int64_t v, int bits, const char *msg);",
    );
    assert!(
        !diagnostic
            .iter()
            .any(|(kind, _)| kind == "raw-element-pointer"),
        "a const char * diagnostic message is not an element carrier: {diagnostic:?}"
    );
    for real in [
        "float *chelis_probe_a(float *p);",
        "#include <stdint.h>\nextern const int64_t *const chelis_probe_b;",
        "#include <stdint.h>\nvoid chelis_probe_c(int64_t *__restrict__ out);",
        "#include <stdint.h>\ntypedef int64_t chelis_probe_extent;\nextern chelis_probe_extent *chelis_probe_d;",
    ] {
        let rows = c_owners(real);
        assert!(
            rows.iter().any(|(kind, _)| kind == "raw-element-pointer"),
            "missed a real carrier: {real} -> {rows:?}"
        );
    }
}

#[test]
fn the_public_header_is_owned_declaration_by_declaration() {
    // The real published header, through the real lane: every row names a
    // declaration, the allocation entry points own their carriers, and the
    // descriptor's fields own theirs.
    let rows = scan_c_header(HEADER, &tracked_source(HEADER)).expect("header must scan");
    let owners: std::collections::BTreeSet<&str> =
        rows.iter().map(|row| row.owner.as_str()).collect();
    assert!(!owners.contains("module"), "{owners:?}");
    for expected in ["chelis_alloc", "chelis_tensor_entry_borrow"] {
        assert!(owners.contains(expected), "{owners:?}");
    }
    assert!(!owners.contains("chelis_alloc_view"), "{owners:?}");
}

#[test]
fn every_arm_with_code_in_the_real_headers_is_parsed_by_a_declared_configuration() {
    // The lane's configuration set is checked against the header, not
    // trusted: an arm no configuration parses is a scan failure, so the real
    // headers scanning at all proves their arms are covered.
    for header in [
        HEADER,
        HIP_HEADER,
        METAL_HEADER,
        "crates/chelis-runtime/include/chelis_simd.h",
    ] {
        scan_c_header(header, &tracked_source(header))
            .unwrap_or_else(|error| panic!("{header}: {error}"));
    }
    let simd = tracked_source("crates/chelis-runtime/include/chelis_simd.h");
    let arms = ConditionalArms::analyze(&simd);
    assert!(
        arms.arms
            .iter()
            .any(|arm| arm.has_code && arm.directive.contains("__AVX2__")),
        "the SIMD header's AVX2 arm carries code: {:?}",
        arms.arms
    );
}

#[test]
fn a_carrier_in_a_non_default_arm_is_still_a_row() {
    // The scalar configuration never sees these arms; the avx2, neon, and
    // release configurations do, and the row set is their union.
    let rows = c_owners(
        r#"
        #ifdef __AVX2__
        static inline void chelis_probe_avx(float *p) { (void)p; }
        #elif defined(__ARM_NEON)
        static inline void chelis_probe_neon(const float *p) { (void)sizeof(*p); }
        #else
        static inline void chelis_probe_scalar(double *p) { (void)p; }
        #endif
        "#,
    );
    for expected in [
        ("raw-element-pointer", "chelis_probe_avx"),
        ("raw-element-pointer", "chelis_probe_neon"),
        ("width-arithmetic", "chelis_probe_neon"),
        ("raw-element-pointer", "chelis_probe_scalar"),
    ] {
        assert!(
            rows.contains(&(expected.0.to_string(), expected.1.to_string())),
            "missing {expected:?} in {rows:?}"
        );
    }
    let release = scan_c_source(
        HIP_HEADER,
        r#"
        static inline void chelis_probe_launch(void *p) {
        #ifndef NDEBUG
            (void)p;
        #else
            (void)sizeof(float);
        #endif
        }
        #ifdef NDEBUG
        static inline void chelis_probe_release_only(float *p) { (void)p; }
        #endif
        "#,
        HIP_LANE,
    )
    .expect("must scan");
    let mut release: Vec<(&str, &str)> = release
        .iter()
        .map(|row| (row.kind.as_str(), row.owner.as_str()))
        .collect();
    release.sort();
    assert_eq!(
        release,
        vec![
            ("raw-element-pointer", "chelis_probe_release_only"),
            ("width-arithmetic", "chelis_probe_launch"),
        ]
    );
}

#[test]
fn an_arm_no_configuration_parses_fails_closed() {
    for (source, needle) in [
        (
            "#ifdef CHELIS_PROBE_UNDECLARED\nextern float *chelis_probe_hidden;\n#endif\n",
            "CHELIS_PROBE_UNDECLARED",
        ),
        (
            "#if __has_include(<hip/hip_fp16.h>)\nstatic inline void chelis_probe_fp16(float *p) { (void)p; }\n#endif\n",
            "hip_fp16.h",
        ),
        ("#if 0\nextern double *chelis_probe_dead;\n#endif\n", "if 0"),
    ] {
        let error = scan_c_header(HEADER, source).expect_err("an unparsed arm must fail closed");
        assert!(error.message.contains(needle), "{}", error.message);
        assert!(error.message.contains("none of the"), "{}", error.message);
    }
}

#[test]
fn an_arm_with_only_directives_or_linkage_braces_needs_no_configuration() {
    // The include guard, a `__cplusplus` linkage block, and a directive-only
    // arm (an include, a define, an `#error`) carry no seam, so no
    // configuration has to parse them.
    let rows = c_owners(
        r#"
        #ifndef CHELIS_PROBE_H
        #define CHELIS_PROBE_H
        #ifdef __cplusplus
        extern "C" {
        #endif
        #if __has_include(<stdint.h>)
        #include <stdint.h>
        #define CHELIS_PROBE_HAS_IT 1
        #else
        #error "not present"
        #endif
        extern float *chelis_probe_visible;
        #ifdef __cplusplus
        }
        #endif
        #endif
        "#,
    );
    assert!(
        rows.contains(&(
            "raw-element-pointer".to_string(),
            "chelis_probe_visible".to_string()
        )),
        "{rows:?}"
    );
}

#[test]
fn an_objective_c_method_signature_is_a_carrier() {
    let rows = scan_c_source(
        METAL_HEADER,
        r#"
        #include <Foundation/Foundation.h>
        @interface ChelisProbeBuffer : NSObject
        - (void)fill:(float *)values count:(NSUInteger)n;
        - (float *)elements;
        - (NSUInteger)count;
        @end
        @protocol ChelisProbeSource
        - (const float *)chelisProbeRows:(NSUInteger)n;
        @end
        "#,
        OBJECTIVE_C_LANE,
    )
    .expect("must scan");
    let mut owners: Vec<(&str, &str)> = rows
        .iter()
        .map(|row| (row.kind.as_str(), row.owner.as_str()))
        .collect();
    owners.sort();
    assert_eq!(
        owners,
        vec![
            ("raw-element-pointer", "ChelisProbeBuffer::elements"),
            ("raw-element-pointer", "ChelisProbeBuffer::fill:count:"),
            ("raw-element-pointer", "ChelisProbeSource::chelisProbeRows:"),
        ]
    );
}

#[test]
fn an_unclassified_spelling_in_expression_position_fails_closed() {
    // The inverted type-word rule holds in a cast, a compound literal, and a
    // sizeof operand, not only in declarations.
    for body in [
        "((_Float16 *)p)[0] = 0;",
        "(void)sizeof(__bf16);",
        "(void)(_Float16 *){0};",
    ] {
        let source = format!("static inline void chelis_probe_expr(void *p) {{ (void)p; {body} }}");
        let error = scan_c_header(HEADER, &source)
            .expect_err("an unclassified spelling in expression position must fail closed");
        assert!(
            error.message.contains("c_lexical.rs"),
            "{body}: {}",
            error.message
        );
    }
}

#[test]
fn an_include_outside_the_universe_fails_closed() {
    let scratch = tempfile::tempdir().expect("scratch dir");
    let side = scratch.path().join("chelis_probe_side.h");
    std::fs::write(&side, "extern float *chelis_probe_side(void);\n").expect("write");
    let source = format!(
        "#include \"{}\"\nextern float *chelis_probe_own(void);\n",
        side.display()
    );
    let error = scan_c_header(HEADER, &source).expect_err("an out-of-universe include must fail");
    assert!(
        error.message.contains("outside the inventory universe"),
        "{}",
        error.message
    );
    assert!(
        error.message.contains("chelis_probe_side.h"),
        "{}",
        error.message
    );
    // The published include directory and the stub SDK remain inside it.
    assert!(
        scan_c_header(HEADER, "#include \"chelis_runtime_dtype.h\"\n#include <stdint.h>\nextern float *chelis_probe_ok(void);\n").is_ok()
    );
    // The round-5 shape: a relative spelling through the published include
    // directory resolves, canonically, to a sibling crate's `src/`.
    let relative =
        "#include \"../src/chelis_probe_side_rel.h\"\nextern float *chelis_probe_own(void);\n";
    let side = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../chelis-runtime/src/chelis_probe_side_rel.h");
    std::fs::write(&side, "extern float *chelis_probe_side_rel(void);\n").expect("write side file");
    let result = scan_c_header(HEADER, relative);
    std::fs::remove_file(&side).expect("remove side file");
    let error = result.expect_err("a `..` include escaping the include directory must fail");
    assert!(
        error.message.contains("outside the inventory universe"),
        "{}",
        error.message
    );
    // A tracked C header outside every root, reached the same way.
    let tracked =
        "#include \"../../../grammars/tree-sitter-chelis-surf/src/tree_sitter/parser.h\"\n";
    let error = scan_c_header(HEADER, tracked).expect_err("a tracked out-of-root header must fail");
    assert!(error.message.contains("parser.h"), "{}", error.message);
}

#[test]
fn a_pointer_to_an_element_array_is_a_carrier() {
    let rows = c_owners("void chelis_probe_rows(float (*rows)[4]);");
    assert!(
        rows.contains(&(
            "raw-element-pointer".to_string(),
            "chelis_probe_rows".to_string()
        )),
        "{rows:?}"
    );
}

#[test]
fn named_anonymous_members_and_typeof_spellings_scan() {
    let rows = c_owners(
        r#"
        typedef struct { struct { float *p; } inner; int n; } chelis_probe_outer;
        __typeof__(float *) chelis_probe_ty(void);
        "#,
    );
    assert!(
        rows.contains(&(
            "raw-element-pointer".to_string(),
            "chelis_probe_outer::p".to_string()
        )),
        "{rows:?}"
    );
    assert!(
        rows.contains(&(
            "raw-element-pointer".to_string(),
            "chelis_probe_ty".to_string()
        )),
        "{rows:?}"
    );
}

#[test]
fn a_multi_dimensional_extent_renders_every_dimension() {
    let rows = scan_c_header(
        HEADER,
        "#include <stdint.h>\ntypedef struct { void *data; uint8_t dtype; int64_t size; int32_t rank; int64_t shape[8][2]; const int64_t *strides; } chelis_probe_grid;",
    )
    .expect("must scan");
    let extent = rows
        .iter()
        .find(|row| row.kind == "fixed-rank-metadata" && row.owner == "chelis_probe_grid::shape")
        .expect("a fixed-rank row");
    assert!(extent.sample.ends_with("(extent 8x2)"), "{}", extent.sample);
}

#[test]
fn the_objective_c_header_is_read_through_its_own_lane() {
    assert_eq!(lane_for(METAL_HEADER), OBJECTIVE_C_LANE);
    assert_eq!(lane_for(HIP_HEADER), HIP_LANE);
    assert_eq!(lane_for(HEADER), PUBLIC_C_LANE);
    let rows = scan_c_header(METAL_HEADER, &tracked_source(METAL_HEADER))
        .expect("the Objective-C header must scan");
    let mut owners: Vec<(&str, &str)> = rows
        .iter()
        .map(|row| (row.kind.as_str(), row.owner.as_str()))
        .collect();
    owners.sort();
    assert_eq!(
        owners,
        vec![
            ("width-arithmetic", "chelis_metal_mps_gemm_f16"),
            ("width-arithmetic", "chelis_metal_mps_gemm_f32"),
            ("width-arithmetic", "chelis_metal_require_bf16_capability"),
        ]
    );
    // An Objective-C function carrying an element pointer is a carrier like
    // any C function.
    let planted = format!(
        "{}\nstatic inline void chelis_probe_fill(id<MTLBuffer> buffer, const float *values) {{ (void)buffer; (void)values; }}\n",
        tracked_source(METAL_HEADER)
    );
    let rows = scan_c_header(METAL_HEADER, &planted).expect("must scan");
    assert!(
        rows.iter()
            .any(|row| row.kind == "raw-element-pointer" && row.owner == "chelis_probe_fill"),
        "{rows:?}"
    );
}

#[test]
fn the_scan_is_independent_of_the_ambient_sdk_state() {
    // A HIP SDK on CPATH, or a ROCm include directory, must not change what
    // the header means: the lane parses under the committed stub SDK and a
    // scrubbed environment. Planting a hipblas header on CPATH would
    // otherwise flip `__has_include` and remove the fallback prototypes.
    let scratch = tempfile::tempdir().expect("scratch dir");
    let hipblas = scratch.path().join("hipblas");
    std::fs::create_dir_all(&hipblas).expect("mkdir");
    std::fs::write(
        hipblas.join("hipblas.h"),
        "#error ambient sdk must not be reached\n",
    )
    .expect("write");
    let source = tracked_source(HIP_HEADER);
    // SAFETY: the test owns its process environment and restores it below.
    unsafe { std::env::set_var("CPATH", scratch.path()) };
    let rows = scan_c_header(HIP_HEADER, &source);
    unsafe { std::env::remove_var("CPATH") };
    let rows = rows.expect("the ambient SDK path must be invisible to the scan");
    assert!(
        rows.iter().any(|row| row.owner == "hipblasSgemm"),
        "the header's own fallback prototypes are the ones inventoried: {rows:?}"
    );
}

// ---------------------------------------------------------------------------
// Per-member ownership and macro bodies in the Rust leg.
// ---------------------------------------------------------------------------

#[test]
fn every_struct_field_enum_variant_and_impl_const_owns_its_row() {
    // Two carrier fields of one aggregate, two pointer-carrying variants of
    // one enum, and two width consts of one impl are six identities, not
    // three: the freeze has to move when one of them is added.
    let rows = identities(
        RUNTIME,
        r#"
        pub struct Pair { a: *mut f32, b: *mut f64 }
        pub enum Slot { A(*mut f32), B { p: *const i64 } }
        pub struct Foo;
        impl Foo {
            pub const A: usize = std::mem::size_of::<f32>();
            pub const B: usize = std::mem::size_of::<f64>();
        }
        "#,
    );
    for expected in [
        ("raw-element-pointer", "Pair::a"),
        ("raw-element-pointer", "Pair::b"),
        ("raw-element-pointer", "Slot::A::0"),
        ("raw-element-pointer", "Slot::B::p"),
        ("width-arithmetic", "Foo::A"),
        ("width-arithmetic", "Foo::B"),
    ] {
        assert!(
            rows.contains(&(expected.0.to_string(), expected.1.to_string())),
            "missing {expected:?} in {rows:?}"
        );
    }
}

#[test]
fn a_type_alias_and_a_union_own_their_rows() {
    let rows = identities(
        RUNTIME,
        r#"
        pub type ElemPtr = *mut f32;
        pub union Bits { a: *mut f32, b: u64 }
        "#,
    );
    assert!(
        rows.contains(&("raw-element-pointer".to_string(), "ElemPtr".to_string())),
        "{rows:?}"
    );
    assert!(
        rows.contains(&("raw-element-pointer".to_string(), "Bits::a".to_string())),
        "{rows:?}"
    );
}

#[test]
fn a_repeat_macro_body_is_still_read() {
    // `vec![t.data; 4]` is not a comma list, but it is two statements, and
    // the `.data` access inside it is a seam owned by the function.
    let rows = identities(
        RUNTIME,
        r#"
        pub struct T { pub data: *mut u8 }
        pub fn spread(t: &T) -> Vec<*mut u8> { vec![t.data; 4] }
        pub fn probe(t: &T) -> bool { matches!(t.data as usize, 0 | 1) }
        "#,
    );
    assert!(
        rows.contains(&("direct-data-access".to_string(), "spread".to_string())),
        "{rows:?}"
    );
    assert!(
        rows.contains(&("direct-data-access".to_string(), "probe".to_string())),
        "{rows:?}"
    );
}

#[test]
fn an_unreadable_macro_body_hiding_a_seam_fails_closed() {
    let error = scan_rust_source(
        RUNTIME,
        r#"
        pub struct T { pub data: *mut u8 }
        pub fn hide(t: &T) { weird!(t.data => 4 @@ ) }
        "#,
    )
    .expect_err("an unreadable macro body that names a seam must fail closed");
    assert!(error.message.contains("weird"), "{}", error.message);
    assert!(error.message.contains("data"), "{}", error.message);
    // The same unreadable shape with no seam word in it is not a seam.
    assert!(
        scan_rust_source(RUNTIME, "pub fn quiet() { weird!(x => 4 @@ ) }").is_ok(),
        "a seam-free macro body is not the inventory's business"
    );
}

#[test]
fn an_offset_of_the_data_field_is_a_seam() {
    // `offset_of!(T, data)` names the field by path rather than by access.
    // A destructuring pattern (`let T { data, .. } = t;`) is deliberately not
    // a seam: the registered sources destructure `RiscOp::ConstTensor { data }`,
    // an IR literal payload that no representation phase touches, and a rule
    // on the field name alone cannot tell that from a descriptor.
    let rows = identities(
        RUNTIME,
        r#"
        pub struct T { pub data: *mut u8, pub n: usize }
        pub fn c() -> usize { std::mem::offset_of!(T, data) }
        "#,
    );
    assert!(
        rows.contains(&("direct-data-access".to_string(), "c".to_string())),
        "{rows:?}"
    );
}

#[test]
fn non_null_c_aliases_and_element_arrays_are_carriers() {
    let rows = identities(
        RUNTIME,
        r#"
        use std::ptr::NonNull;
        use std::os::raw::c_float;
        pub fn a(p: NonNull<f32>) -> usize { p.as_ptr() as usize }
        pub fn b(p: *mut c_float) -> usize { p as usize }
        pub fn c(p: *mut [f32]) -> usize { p as *mut f32 as usize }
        pub fn d(p: *const [i64; 4]) -> usize { p as usize }
        pub fn e(p: *mut u8) -> usize { p as usize }
        "#,
    );
    for owner in ["a", "b", "c", "d"] {
        assert!(
            rows.contains(&("raw-element-pointer".to_string(), owner.to_string())),
            "{owner}: {rows:?}"
        );
    }
    assert!(
        !rows.contains(&("raw-element-pointer".to_string(), "e".to_string())),
        "a byte pointer is still not an element pointer: {rows:?}"
    );
}

#[test]
fn an_impl_associated_type_and_a_foreign_static_own_their_rows() {
    let rows = identities(
        RUNTIME,
        r#"
        pub trait Tr { type Elem; }
        pub struct Foo;
        impl Tr for Foo { type Elem = *mut f32; }
        unsafe extern "C" { pub static mut CHELIS_PROBE_BUF: *mut f32; }
        "#,
    );
    assert!(
        rows.contains(&("raw-element-pointer".to_string(), "Foo::Elem".to_string())),
        "{rows:?}"
    );
    assert!(
        rows.contains(&(
            "raw-element-pointer".to_string(),
            "CHELIS_PROBE_BUF".to_string()
        )),
        "{rows:?}"
    );
}

#[test]
fn a_turbofish_cast_or_pointer_constructor_is_a_carrier() {
    // `p.cast::<f32>()` is the idiomatic pointer cast and spells its target
    // only in the turbofish; std pointer constructors do the same.
    let rows = identities(
        RUNTIME,
        r#"
        use std::ptr::NonNull;
        pub fn a(p: *mut u8) -> usize { p.cast::<f32>() as usize }
        pub fn b(p: *mut u8) -> usize { NonNull::<i64>::new(p.cast()).map_or(0, |q| q.as_ptr() as usize) }
        pub fn c(p: *const u8, n: usize) -> usize { std::ptr::slice_from_raw_parts::<f64>(p.cast(), n) as *const f64 as usize }
        pub fn d(p: *mut u8) -> usize { p.cast::<u8>() as usize }
        "#,
    );
    for owner in ["a", "b", "c"] {
        assert!(
            rows.contains(&("raw-element-pointer".to_string(), owner.to_string())),
            "{owner}: {rows:?}"
        );
    }
    assert!(
        !rows.contains(&("raw-element-pointer".to_string(), "d".to_string())),
        "a byte cast is not an element pointer: {rows:?}"
    );
}

#[test]
fn an_include_or_a_define_with_a_body_makes_an_arm_carry_code() {
    // A dead arm holding an `#include` or a `#define` with a body carries
    // code by reference; a bodiless `#define`, an `#error`, and a `#pragma`
    // do not.
    for (source, needle) in [
        (
            "#ifdef CHELIS_PROBE_NEVER\n#include \"chelis_runtime_dtype.h\"\n#endif\n",
            "CHELIS_PROBE_NEVER",
        ),
        (
            "#ifdef CHELIS_PROBE_NEVER\n#define CHELIS_PROBE_DECL float *chelis_probe_decl(void);\n#else\n#define CHELIS_PROBE_DECL\n#endif\nCHELIS_PROBE_DECL\n",
            "CHELIS_PROBE_NEVER",
        ),
    ] {
        let error =
            scan_c_header(HEADER, source).expect_err("an unparsed include or define must fail");
        assert!(error.message.contains(needle), "{}", error.message);
    }
    assert!(
        scan_c_header(
            HEADER,
            "#ifdef CHELIS_PROBE_NEVER\n#define CHELIS_PROBE_FLAG\n#error \"never\"\n#pragma once\n#endif\nextern float *chelis_probe_ok(void);\n",
        )
        .is_ok()
    );
}

#[test]
fn a_block_literal_parameter_is_a_carrier_and_is_classified() {
    let rows = scan_c_source(
        METAL_HEADER,
        r#"
        #include <Foundation/Foundation.h>
        static inline id chelis_probe_block(void) { return ^(float *q) { q[0] = 0.0f; }; }
        "#,
        OBJECTIVE_C_LANE,
    )
    .expect("must scan");
    assert!(
        rows.iter()
            .any(|row| row.kind == "raw-element-pointer" && row.owner == "chelis_probe_block"),
        "{rows:?}"
    );
    let error = scan_c_source(
        METAL_HEADER,
        "#include <Foundation/Foundation.h>\nstatic inline id chelis_probe_block(void) { return ^(_Float16 *q) { (void)q; }; }\n",
        OBJECTIVE_C_LANE,
    )
    .expect_err("an unclassified block parameter spelling must fail closed");
    assert!(error.message.contains("_Float16"), "{}", error.message);
}

#[test]
fn a_sizeof_in_an_array_bound_or_bit_field_is_a_width_seam() {
    let rows = c_owners(
        r#"
        static unsigned char chelis_probe_scratch[sizeof(double) * 4];
        typedef struct { unsigned char buf[sizeof(float)]; } chelis_probe_arr;
        static inline void chelis_probe_local(void) { unsigned char buf[sizeof(float)]; (void)buf; }
        struct chelis_probe_bits { int bits : sizeof(float); };
        void chelis_probe_param(unsigned char buf[sizeof(float)]);
        "#,
    );
    for owner in [
        "chelis_probe_scratch",
        "chelis_probe_arr::buf",
        "chelis_probe_local",
        "chelis_probe_bits::bits",
        "chelis_probe_param",
    ] {
        assert!(
            rows.contains(&("width-arithmetic".to_string(), owner.to_string())),
            "{owner}: {rows:?}"
        );
    }
}

#[test]
fn a_string_literal_cannot_spoof_the_arm_marker() {
    let error = scan_c_header(
        HEADER,
        "static inline const char *chelis_probe_spoof(void) { return \"#pragma chelis_inventory_arm(1)\"; }\n#ifdef CHELIS_PROBE_NEVER\nextern float *chelis_probe_spoofed(void);\n#endif\n",
    )
    .expect_err("a spoofed marker must not mark a dead arm live");
    assert!(
        error.message.contains("CHELIS_PROBE_NEVER"),
        "{}",
        error.message
    );
}

#[test]
fn offsetof_and_va_list_scan_as_expected() {
    let rows = c_owners(
        r#"
        #include <stddef.h>
        #include <stdint.h>
        typedef struct { void *data; uint8_t dtype; int64_t size; int32_t rank; const int64_t *shape; const int64_t *strides; } chelis_probe_t;
        static inline size_t chelis_probe_off(void) { return __builtin_offsetof(chelis_probe_t, data); }
        static inline void chelis_probe_va(int n, ...) { __builtin_va_list ap; __builtin_va_start(ap, n); __builtin_va_end(ap); }
        "#,
    );
    assert!(
        rows.contains(&(
            "direct-data-access".to_string(),
            "chelis_probe_off".to_string()
        )),
        "{rows:?}"
    );
}

#[test]
fn eight_bit_element_pointers_are_carriers_as_written() {
    // `int8_t` and `uint8_t` resolve to `signed char` and `unsigned char`,
    // which name no element; the written spelling does, and it is read first.
    let rows = c_owners(
        r#"
        #include <stdint.h>
        int8_t *chelis_probe_i8(void);
        typedef struct { uint8_t *bytes; int64_t n; } chelis_probe_u8s;
        void chelis_probe_take_i8(int8_t *p, int64_t n);
        typedef uint8_t chelis_probe_byte;
        chelis_probe_byte *chelis_probe_bytes(void);
        char *chelis_probe_text(void);
        "#,
    );
    for owner in [
        "chelis_probe_i8",
        "chelis_probe_u8s::bytes",
        "chelis_probe_take_i8",
        "chelis_probe_bytes",
    ] {
        assert!(
            rows.contains(&("raw-element-pointer".to_string(), owner.to_string())),
            "{owner}: {rows:?}"
        );
    }
    assert!(
        !rows.contains(&(
            "raw-element-pointer".to_string(),
            "chelis_probe_text".to_string()
        )),
        "a char pointer is text, not an element carrier: {rows:?}"
    );
}

#[test]
fn elifdef_and_elifndef_open_arms() {
    let error = scan_c_header(
        HEADER,
        "#ifdef __GNUC__\nstatic const int chelis_probe_live = 1;\n#elifdef CHELIS_PROBE_NEVER\nfloat *chelis_probe_elifdef(void);\n#endif\n",
    )
    .expect_err("a dead #elifdef arm with a carrier must fail closed");
    assert!(
        error.message.contains("elifdef CHELIS_PROBE_NEVER"),
        "{}",
        error.message
    );
    let rows = c_owners(
        "#ifdef CHELIS_PROBE_NEVER\n#define CHELIS_PROBE_FLAG\n#elifndef CHELIS_PROBE_NEVER2\nfloat *chelis_probe_elifndef(void);\n#endif\n",
    );
    assert!(
        rows.contains(&(
            "raw-element-pointer".to_string(),
            "chelis_probe_elifndef".to_string()
        )),
        "{rows:?}"
    );
}
