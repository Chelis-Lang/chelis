use std::collections::BTreeMap;
use std::path::Path;

use chelis_c_surface::{
    CarrierUse, ScanErrorKind, canonical_c_tokens, classify, collect_c_aliases, collect_typedefs,
    scan_c_source, scan_c_source_at_path, scan_c_source_with_aliases, scan_rust_source,
    strip_c_comments,
};

fn signatures<'a>(rows: &'a [CarrierUse], kind: &str) -> Vec<&'a str> {
    rows.iter()
        .filter(|row| row.kind == kind)
        .map(|row| row.signature.as_str())
        .collect()
}

fn assert_carrier_is_detected_or_rejected(source: &str) {
    match scan_c_source(source, "fixture") {
        Ok(rows) => assert!(
            rows.iter().any(|row| row.kind == "raw-element-pointer"),
            "carrier was silently accepted without an inventory row: {source}"
        ),
        Err(error) => assert!(
            matches!(
                error.kind,
                ScanErrorKind::UnknownType
                    | ScanErrorKind::MalformedCandidate
                    | ScanErrorKind::IncompleteFragment
            ),
            "carrier failed for an unexpected reason: {error}"
        ),
    }
}

#[test]
fn qualifiers_arrays_casts_and_sizeof_are_parsed_from_structure() {
    let source = r#"
extern void const_after(float const *payload);
extern void array_parameter(float payload[]);
static float const_after_cast(void *payload) {
    return *((float const *)payload);
}
extern void multiline(
    const volatile double
    * restrict payload);
static size_t qualified_sizeof(void) {
    return sizeof(const float) + sizeof(
        volatile double
    );
}
"#;

    let rows = scan_c_source(source, "fixture").expect("valid C-family source parses");
    assert!(signatures(&rows, "raw-element-pointer").len() >= 4);
    assert_eq!(signatures(&rows, "width-arithmetic").len(), 2);
    assert!(
        signatures(&rows, "raw-element-pointer")
            .iter()
            .any(|signature| signature.contains("shape=array") && signature.contains("float"))
    );
    assert!(
        signatures(&rows, "raw-element-pointer")
            .iter()
            .any(|signature| signature.contains("ast=CStyleCastExpr"))
    );
}

#[test]
fn qualifier_order_has_one_canonical_type_identity() {
    let before = scan_c_source("extern void f(const volatile float *p);", "fixture")
        .expect("qualifiers before the type parse");
    let after = scan_c_source("extern void f(float volatile const *p);", "fixture")
        .expect("qualifiers after the type parse");
    assert_eq!(before[0].signature, after[0].signature);
}

#[test]
fn recognized_type_decorators_cannot_hide_element_pointers() {
    let source = r#"
extern void decorated(
    float __attribute__((aligned(16))) *gnu,
    double [[maybe_unused]] *cpp,
    _Atomic(float) *atomic_value,
    float _Nullable *nullable_value);
"#;
    let rows = scan_c_source(source, "fixture").expect("recognized decorators parse");
    assert_eq!(signatures(&rows, "raw-element-pointer").len(), 4);

    let error = scan_c_source(
        "extern void unknown(float __new_decoration *payload);",
        "fixture",
    )
    .expect_err("unknown decoration between type and pointer must fail");
    assert_eq!(error.kind, ScanErrorKind::UnknownType);
}

#[test]
fn typedef_and_macro_aliases_resolve_transitively() {
    let source = r#"
typedef float chelis_sample;
typedef chelis_sample chelis_sample_alias;
#define CHELIS_SAMPLE chelis_sample_alias
extern void from_typedef(chelis_sample_alias *payload);
extern void from_macro(CHELIS_SAMPLE *payload);
typedef double chelis_vec4[4];
"#;
    let rows = scan_c_source(source, "fixture").expect("aliases resolve");
    let raw = signatures(&rows, "raw-element-pointer");
    assert_eq!(raw.len(), 3);
    assert!(raw.iter().any(|signature| {
        signature.contains("authored=chelis_sample_alias") && signature.contains("resolved=float")
    }));
    assert!(
        raw.iter().any(|signature| {
            signature.contains("authored=CHELIS_SAMPLE") && signature.contains("resolved=float")
        }),
        "{raw:#?}"
    );
    assert!(raw.iter().any(|signature| {
        signature.contains("authored=chelis_vec4") && signature.contains("resolved=double")
    }));
}

#[test]
fn alias_redefinitions_and_scoped_shadows_cannot_erase_numeric_meaning() {
    for source in [
        r#"
#define RR_TYPE float
extern void before_redefinition(RR_TYPE *payload);
#undef RR_TYPE
#define RR_TYPE void
extern void after_redefinition(RR_TYPE *context);
"#,
        r#"
typedef float rr_type;
extern void before_shadow(rr_type *payload);
static void shadow(void) { typedef void rr_type; }
"#,
        r#"
#define RR_TYPE() float
extern void from_function_macro(RR_TYPE() *payload);
"#,
    ] {
        assert_carrier_is_detected_or_rejected(source);
    }
}

#[test]
fn exact_identities_preserve_base_modifiers_and_array_extents() {
    let signed = scan_c_source("extern void f(int *payload);", "fixture").unwrap();
    let unsigned = scan_c_source("extern void f(unsigned int *payload);", "fixture").unwrap();
    assert_ne!(signed[0].signature, unsigned[0].signature);

    let plain = scan_c_source("extern void f(double *payload);", "fixture").unwrap();
    let extended = scan_c_source("extern void f(long double *payload);", "fixture").unwrap();
    assert_ne!(plain[0].signature, extended[0].signature);

    let complex = scan_c_source("extern void f(_Complex double *payload);", "fixture").unwrap();
    assert_ne!(plain[0].signature, complex[0].signature);

    let four = scan_c_source("extern void f(float payload[4]);", "fixture").unwrap();
    let eight = scan_c_source("extern void f(float payload[8]);", "fixture").unwrap();
    assert_ne!(four[0].signature, eight[0].signature);
}

#[test]
fn exact_identities_include_the_enclosing_declaration() {
    let first = scan_c_source("extern void first(float *payload);", "fixture").unwrap();
    let second = scan_c_source("extern void second(float *payload);", "fixture").unwrap();
    assert_ne!(first[0].owner, second[0].owner);
    assert_ne!(first[0].signature, second[0].signature);
    assert!(first[0].owner.contains("first"));
    assert!(second[0].owner.contains("second"));
}

#[test]
fn anonymous_record_identities_do_not_depend_on_source_lines() {
    let compact = scan_c_source("typedef struct { float *data; } local_tensor;", "fixture")
        .expect("anonymous record parses");
    let shifted = scan_c_source(
        "\n\n\ntypedef struct { float *data; } local_tensor;",
        "fixture",
    )
    .expect("line-shifted anonymous record parses");

    assert_eq!(compact, shifted);
    assert!(compact.iter().all(|row| !row.owner.contains("/tmp/")));
}

#[test]
fn tracked_anonymous_record_owners_do_not_expose_compiler_locations() {
    let source = include_str!("../../chelis-runtime/include/chelis_runtime.h");
    let rows = scan_c_source_at_path(
        source,
        "crates/chelis-runtime/include/chelis_runtime.h",
        Path::new("crates/chelis-runtime/include/chelis_runtime.h"),
        &collect_c_aliases(source),
    )
    .expect("tracked runtime header parses");

    assert!(
        rows.iter().all(|row| !row.owner.contains("/tmp/")),
        "compiler location leaked into rows: {rows:#?}"
    );
}

#[test]
fn tracked_metal_sizeof_expressions_are_width_arithmetic() {
    let source = include_str!("../../chelis-backend-metal/runtime/chelis_metal_runtime.h");
    let rows = scan_c_source_at_path(
        source,
        "crates/chelis-backend-metal/runtime/chelis_metal_runtime.h",
        Path::new("crates/chelis-backend-metal/runtime/chelis_metal_runtime.h"),
        &collect_c_aliases(source),
    )
    .expect("tracked Metal header parses");

    assert!(
        rows.iter().any(|row| row.kind == "width-arithmetic"),
        "Metal sizeof expressions disappeared from the structural scan: {rows:#?}"
    );
}

#[test]
fn complete_declarator_shape_is_injective() {
    let sources = [
        (
            "extern void f(float *payload);",
            "extern void f(_Atomic(float) *payload);",
        ),
        (
            "extern void f(float *payload);",
            "extern void f(float __attribute__((address_space(1))) *payload);",
        ),
        (
            "extern void f(float &payload);",
            "extern void f(float &&payload);",
        ),
        (
            "extern void f(float payload[4][8]);",
            "extern void f(float payload[4][9]);",
        ),
        (
            "extern void f(float (*callback)(int));",
            "extern void f(float (*callback)(long));",
        ),
    ];
    for (left, right) in sources {
        let left_rows = scan_c_source(left, "fixture").expect("left declaration parses");
        let right_rows = scan_c_source(right, "fixture").expect("right declaration parses");
        assert_ne!(
            left_rows, right_rows,
            "declarators collapsed: {left} / {right}"
        );
    }
}

#[test]
fn every_declarator_and_pointer_typedef_use_gets_an_identity() {
    let rows = scan_c_source(
        r#"
typedef float *sample_ptr;
extern void consume(sample_ptr payload);
static float *first, *second;
"#,
        "fixture",
    )
    .expect("valid declarations parse");
    assert!(rows.iter().any(|row| row.owner.contains("consume")));
    assert!(rows.iter().any(|row| row.signature.contains("name=first")));
    assert!(rows.iter().any(|row| row.signature.contains("name=second")));
}

#[test]
fn numeric_pointer_return_declarators_get_an_identity() {
    let rows = scan_c_source(
        "extern float *runtime_representation_phase0_pointer_return(void);",
        "fixture",
    )
    .expect("pointer-return declaration parses");
    let raw = signatures(&rows, "raw-element-pointer");
    assert_eq!(raw.len(), 1, "pointer return disappeared: {rows:#?}");
    assert!(raw[0].contains("shape=return-pointer"), "{raw:#?}");
    assert!(raw[0].contains("runtime_representation_phase0_pointer_return"));
}

#[test]
fn cxx_method_result_identities_include_the_full_enclosing_declaration_chain() {
    let rows = scan_c_source(
        r#"
namespace first {
struct Provider {
    float *values();
    float &reference_result();
    float &&rvalue_result();
    float (*function_result())(int);
    float (*array_result())[4];
};
}
namespace second {
struct Provider {
    float *values();
};
}
"#,
        "fixture",
    )
    .expect("C++ method result declarations parse");
    let raw: Vec<_> = rows
        .iter()
        .filter(|row| row.kind == "raw-element-pointer")
        .collect();
    assert_eq!(
        raw.len(),
        6,
        "C++ method result identities collided: {rows:#?}"
    );
    assert!(
        raw.iter()
            .any(|row| row.owner.contains("first::Provider::values")),
        "first namespace and class are absent from the owner: {rows:#?}"
    );
    assert!(
        raw.iter()
            .any(|row| row.owner.contains("second::Provider::values")),
        "second namespace and class are absent from the owner: {rows:#?}"
    );
}

#[test]
fn objective_c_method_results_use_the_same_structural_result_path() {
    let rows = scan_c_source(
        r#"
@interface FirstRuntimeRepresentationProvider
- (float *)values;
+ (double *)sharedValues;
- (NSString *)label;
@end
@interface SecondRuntimeRepresentationProvider
- (float *)values;
@end
"#,
        "fixture",
    )
    .expect("Objective-C++ method declarations parse");
    let raw: Vec<_> = rows
        .iter()
        .filter(|row| row.kind == "raw-element-pointer")
        .collect();
    assert_eq!(
        raw.len(),
        3,
        "Objective-C method result identities collided: {rows:#?}"
    );
    assert!(
        raw.iter().any(|row| row
            .owner
            .contains("FirstRuntimeRepresentationProvider::values")),
        "first interface is absent from the owner: {rows:#?}"
    );
    assert!(
        raw.iter().any(|row| row
            .owner
            .contains("SecondRuntimeRepresentationProvider::values")),
        "second interface is absent from the owner: {rows:#?}"
    );
    assert!(raw.iter().any(|row| row.signature.contains("sharedValues")));
    assert!(raw.iter().all(|row| !row.signature.contains("label")));
}

#[test]
fn every_typed_declaration_category_is_admitted_structurally() {
    let source = r#"
float *global_value;
struct RuntimeRepresentationRecord {
    float *field_value;
};
typedef float *PointerTypedef;
using PointerAlias = float *;
template<float *TemplateValue>
struct PointerTemplate {};
@interface RuntimeRepresentationProvider {
@public
    float *ivar_value;
}
@property float *property_value;
@end
extern void consume(float *parameter_value);
"#;
    let rows = scan_c_source_at_path(
        source,
        "fixture.mm",
        Path::new("fixture.mm"),
        &collect_c_aliases(source),
    )
    .expect("every declaration category parses in Objective-C++");

    for name in [
        "global_value",
        "field_value",
        "PointerTypedef",
        "PointerAlias",
        "TemplateValue",
        "ivar_value",
        "property_value",
        "parameter_value",
    ] {
        assert!(
            rows.iter().any(|row| {
                row.owner.contains(name) || row.signature.contains(&format!("name={name}"))
            }),
            "typed declaration {name} disappeared from the structural inventory: {rows:#?}"
        );
    }
}

#[test]
fn numeric_carrier_expressions_are_admitted_by_ast_category_not_cast_kind() {
    let rows = scan_c_source(
        r#"
static float *identity(float *payload) {
    return payload;
}
"#,
        "fixture",
    )
    .expect("valid pointer expression parses");

    assert!(
        rows.iter()
            .any(|row| row.signature.contains("context=expression")),
        "numeric carrier expressions still depend on a positive cast-kind list: {rows:#?}"
    );
}

#[test]
fn repeated_carrier_expressions_preserve_multiplicity_without_location_identity() {
    let once = scan_c_source(
        "static float *identity(float *payload) { payload; return payload; }",
        "fixture",
    )
    .expect("single pointer use parses");
    let twice = scan_c_source(
        "static float *identity(float *payload) { payload; payload; return payload; }",
        "fixture",
    )
    .expect("repeated pointer use parses");
    assert!(
        signatures(&twice, "raw-element-pointer").len()
            > signatures(&once, "raw-element-pointer").len(),
        "an identical carrier expression was deduplicated: once={once:#?}, twice={twice:#?}"
    );

    let shifted = scan_c_source(
        "\n\nstatic float *identity(float *payload) { payload; return payload; }",
        "fixture",
    )
    .expect("line-shifted pointer use parses");
    assert_eq!(
        once, shifted,
        "source locations leaked into structural expression identity"
    );
}

#[test]
fn active_preprocessor_branch_cannot_hide_a_cxx_rvalue_reference() {
    let rows = scan_c_source(
        r#"
#ifdef __cplusplus
extern void cxx_only(float &&payload);
#endif
"#,
        "fixture",
    )
    .expect("the C++ configuration parses");
    assert!(
        rows.iter().any(|row| {
            row.owner.contains("cxx_only") && row.signature.contains("rvalue-reference")
        }),
        "{rows:#?}"
    );
}

#[test]
fn cxx_references_templates_and_unknown_types_cannot_bypass_classification() {
    for source in [
        "extern void cpp_ref(float &payload);",
        "extern void cpp_template(vector<float, 4> *payload);",
        "extern void posit(posit32 *payload);",
    ] {
        assert_carrier_is_detected_or_rejected(source);
    }
}

#[test]
fn unknown_type_words_fail_closed_at_carrier_positions() {
    for source in [
        "extern void f(_Float16 *payload);",
        "extern void f(__new_vendor_float *payload);",
        "extern void f(vendor_float *payload);",
        "extern void f(unknown_numeric_t payload[]);",
        "static size_t f(void) { return sizeof(_Decimal64); }",
        "typedef _Float16 sample; extern void f(sample *payload);",
        "#define SAMPLE _Float16\nextern void f(SAMPLE *payload);",
        "extern void f(NewExternal *payload);",
        "extern void f(__new_modifier float *payload);",
    ] {
        let error = scan_c_source(source, "fixture").expect_err("unknown type must fail");
        assert_eq!(error.kind, ScanErrorKind::UnknownType, "{source}");
    }
}

#[test]
fn explicit_opaque_and_aggregate_types_are_structurally_nonnumeric() {
    let source = r#"
typedef NSString ChelisString;
typedef struct ChelisContext ChelisContext;
extern void objective_c(NSString *direct, ChelisString *alias);
extern void framework(NSMutableDictionary *cache, MPSMatrix *matrix);
extern void objective_c_protocol(id<MTLBuffer> *buffers);
extern void aggregate(ChelisContext *context, enum ChelisMode *mode);
static MTLGPUFamily families[] = { MTLGPUFamilyApple7 };
static size_t opaque_width(void) { return sizeof(struct ChelisContext); }
"#;
    let rows = scan_c_source(source, "fixture").expect("nonnumeric types parse");
    assert!(rows.is_empty(), "{rows:#?}");
}

#[test]
fn aggregate_typedef_bodies_register_the_alias_without_hiding_numeric_fields() {
    let source = r#"
typedef struct {
    float *data;
    int ndim;
} local_tensor;
extern void consume(local_tensor *tensor);
"#;
    let rows = scan_c_source(source, "fixture").expect("aggregate typedef is structural");
    let raw = signatures(&rows, "raw-element-pointer");
    assert_eq!(raw.len(), 1, "{raw:#?}");
    assert!(raw[0].contains("resolved=float"));
}

#[test]
fn repository_alias_prelude_resolves_types_declared_in_included_headers() {
    let aliases = collect_c_aliases("typedef struct { float *data; } chelis_tensor;");
    let rows = scan_c_source_with_aliases(
        "extern void consume(chelis_tensor *tensor);",
        "fixture",
        &aliases,
    )
    .expect("a tracked-header typedef is visible to its consumers");
    assert!(rows.is_empty());
}

#[test]
fn repository_alias_prelude_preserves_complete_carrier_shape() {
    let aliases = collect_c_aliases(
        r#"
typedef float *runtime_pointer;
typedef double runtime_array[4];
typedef int (*runtime_callback)(float);
"#,
    );
    let source = r#"
extern void consume_pointer(runtime_pointer payload);
extern void consume_array(runtime_array payload);
extern void consume_callback(runtime_callback payload);
"#;
    let rows = scan_c_source_at_path(source, "fixture.cpp", Path::new("fixture.cpp"), &aliases)
        .expect("cross-file carrier aliases parse");

    assert!(
        rows.iter()
            .any(|row| row.signature.contains("name=payload")),
        "cross-file pointer, array, and callback aliases lost their carrier shape: {rows:#?}"
    );
    assert!(
        rows.iter().any(|row| row.signature.contains("float *")),
        "pointer alias lost its canonical shape: {rows:#?}"
    );
    assert!(
        rows.iter().any(|row| row.signature.contains("double[4]")),
        "array alias lost its canonical extent: {rows:#?}"
    );
    assert!(
        rows.iter()
            .any(|row| row.signature.contains("int (*)(float)")),
        "function-pointer alias lost its callable signature: {rows:#?}"
    );
}

#[test]
fn nested_sizeof_type_names_are_parsed_without_confusing_expressions() {
    let source = r#"
static size_t widths(float *payload, size_t index) {
    return sizeof(const float) + sizeof(double[4])
        + sizeof(payload[index]) + sizeof(index * index);
}
"#;
    let rows = scan_c_source(source, "fixture").expect("valid sizeof operands parse");
    let widths = signatures(&rows, "width-arithmetic");
    assert_eq!(widths.len(), 4);
    assert!(
        widths
            .iter()
            .any(|signature| signature.contains("resolved=float"))
    );
    assert!(
        widths
            .iter()
            .any(|signature| signature.contains("resolved=double"))
    );
}

#[test]
fn every_compiler_expression_category_can_expose_width_arithmetic() {
    let source = r#"
template<typename... Ts>
static size_t pack_width(void) {
    return sizeof...(Ts);
}
"#;
    let rows = scan_c_source(source, "fixture").expect("valid C++17 pack width parses");
    let widths = signatures(&rows, "width-arithmetic");
    assert_eq!(widths.len(), 1, "sizeof-pack disappeared: {rows:#?}");
    assert!(
        widths[0].contains("sizeof ... ( Ts )"),
        "sizeof-pack identity lost its authored structure: {rows:#?}"
    );
}

#[test]
fn non_width_parameter_pack_expressions_remain_nonnumeric() {
    let source = r#"
template<typename... Ts>
static void sink(Ts...);

template<typename... Ts>
static void forward(Ts... values) {
    sink(values...);
}
"#;
    let rows = scan_c_source(source, "fixture").expect("valid C++17 pack expansion parses");
    assert!(
        signatures(&rows, "width-arithmetic").is_empty(),
        "ordinary pack expansion was misclassified as width arithmetic: {rows:#?}"
    );
}

#[test]
fn the_compiler_prelude_supports_the_c_dialect() {
    let rows = scan_c_source_at_path(
        "static size_t widths(float values[4]) { return sizeof(values[0]); }",
        "fixture.c",
        Path::new("fixture.c"),
        &BTreeMap::new(),
    )
    .expect("C11 source parses with the compiler prelude");

    let raw = signatures(&rows, "raw-element-pointer");
    assert!(
        raw.iter()
            .any(|signature| signature.contains("context=declaration"))
    );
    assert!(
        raw.iter()
            .any(|signature| signature.contains("context=expression"))
    );
    assert_eq!(signatures(&rows, "width-arithmetic").len(), 1);
}

#[test]
fn expressions_and_comments_are_not_misread_as_declarators() {
    let source = r#"
// extern void ignored(float const *payload);
/* sizeof(const double) */
static int arithmetic(int a, int b) { return a * b; }
static int dereference(int *p) { return *p; }
"#;
    let rows = scan_c_source(source, "fixture").expect("ordinary expressions parse");
    let raw = signatures(&rows, "raw-element-pointer");
    assert_eq!(
        raw.iter()
            .filter(|signature| signature.contains("context=declaration"))
            .count(),
        1
    );
    assert_eq!(
        raw.iter()
            .filter(|signature| signature.contains("context=expression"))
            .count(),
        2
    );
    assert!(signatures(&rows, "width-arithmetic").is_empty());
}

#[test]
fn identifier_multiplication_is_not_an_unknown_type_declaration() {
    let rows = scan_c_source(
        r#"
static size_t area(size_t float_count, size_t stride) {
    hiprtcDestroyProgram(&program);
    if ((float_count & stride) != 0) return 0;
    size_t raw = (float_count & stride);
    return float_count * stride;
}
"#,
        "fixture",
    )
    .expect("ordinary multiplication and address-of expressions are not declarations");
    assert!(rows.is_empty());
}

#[test]
fn non_c_prose_is_outside_the_carrier_parser() {
    let rows = scan_c_source(
        "reduced-float chains need the target capability table's conversion rule",
        "fixture",
    )
    .expect("prose without carrier syntax is not C input");
    assert!(rows.is_empty());

    let rust_source = r#"
fn diagnostic() -> &'static str {
    "reduce_window_* must be rejected before HIP codegen by [05-RWIN-2]"
}
"#;
    let rows = scan_rust_source(rust_source).expect("prose inside Rust is not emitted C");
    assert!(rows.is_empty());
}

#[test]
fn cxx_raw_strings_are_single_lexical_tokens() {
    let source = "static NSString *const src = @R\"MSL(\n/* literal */ \"quoted\" device float *not_a_host_carrier\n)MSL\";";
    let rows = scan_c_source(source, "fixture").expect("C++ raw string parses");
    assert!(rows.is_empty());

    let malformed = "extern void f(float *payload); static NSString *src = @R\"MSL(unterminated";
    let error = scan_c_source(malformed, "fixture").expect_err("raw string must close");
    assert_eq!(error.kind, ScanErrorKind::MalformedCandidate);
}

#[test]
fn malformed_candidate_syntax_fails_instead_of_disappearing() {
    for source in [
        "extern void f(float const *payload;",
        "static size_t f(void) { return sizeof(const float; }",
        "extern void f(float payload[);",
        "extern void f(float *payload); /* unterminated",
        "extern void f(float *payload); const char *s = \"unterminated;",
    ] {
        let error = scan_c_source(source, "fixture").expect_err("malformed carrier must fail");
        assert_eq!(error.kind, ScanErrorKind::MalformedCandidate, "{source}");
    }
}

#[test]
fn rust_strings_and_format_type_holes_are_part_of_the_same_parser() {
    let source = r##"
fn emitted(ty: &str) -> String {
    let literal = r#"extern void literal(float const *payload);"#;
    format!("extern void dynamic({ty} const *payload); sizeof({ty}); {literal}")
}
"##;
    let rows = scan_rust_source(source).expect("Rust and embedded C both parse");
    let raw = signatures(&rows, "raw-element-pointer");
    assert_eq!(raw.len(), 2);
    assert!(
        raw.iter()
            .any(|signature| signature.contains("format-hole:ty"))
    );
    let widths = signatures(&rows, "width-arithmetic");
    assert_eq!(widths.len(), 1);
    assert!(widths[0].contains("format-hole:ty"));
}

#[test]
fn rust_emitted_widths_use_the_same_category_total_projection() {
    let source = r##"
fn emitted() -> &'static str {
    r#"template<typename... Ts>
static size_t pack_width(void) { return sizeof...(Ts); }"#
}
"##;
    let rows = scan_rust_source(source).expect("Rust-emitted C++17 pack width parses");
    let widths = signatures(&rows, "width-arithmetic");
    assert_eq!(
        widths.len(),
        1,
        "Rust emission silently bypassed the compiler projection: {rows:#?}"
    );
}

#[test]
fn positional_format_holes_and_macro_rules_literals_are_scanned() {
    let positional = r#"
fn emitted(ty: &str, name: &str) -> String {
    format!("{} *{}", ty, name)
}
"#;
    let rows = scan_rust_source(positional).expect("valid Rust parses");
    assert!(rows.iter().any(|row| row.kind == "raw-element-pointer"));

    let macro_rules_source = r#"
macro_rules! emitted_pointer {
    () => { "float *payload" };
}
"#;
    let rows = scan_rust_source(macro_rules_source).expect("valid Rust parses");
    assert!(rows.iter().any(|row| row.kind == "raw-element-pointer"));

    let address_expression = r#"
fn emitted(index: usize) -> String {
    format!("&d_t{}->data", index)
}
"#;
    let rows = scan_rust_source(address_expression).expect("address-of is an expression");
    assert!(rows.is_empty());
}

#[test]
fn only_exact_cfg_test_modules_are_excluded() {
    let source = r#"
#[cfg(not(test))]
mod production {
    pub fn emitted() -> &'static str { "float *payload" }
}
#[cfg(test)]
mod tests {
    const FIXTURE_ONLY: &str = "double *fixture";
}
"#;
    let rows = scan_rust_source(source).expect("valid Rust parses");
    assert_eq!(signatures(&rows, "raw-element-pointer").len(), 1);
    assert!(signatures(&rows, "raw-element-pointer")[0].contains("float"));
}

#[test]
fn rust_line_fragments_may_leave_only_outer_c_blocks_open() {
    let source = r#"
fn emitted() -> &'static str {
    "static inline float f(float const *payload) {"
}
"#;
    let rows = scan_rust_source(source).expect("the pointer declarator is locally complete");
    assert_eq!(signatures(&rows, "raw-element-pointer").len(), 1);
}

#[test]
fn split_carrier_fragments_fail_closed() {
    let pointer_source = r#"
fn emitted(ty: &str) -> String {
    format!("{ty}") + " *payload"
}
"#;
    let qualifier_source = r#"
fn emitted(ty: &str) -> String {
    format!("{ty}") + " const *payload"
}
"#;
    let array_source = r#"
fn emitted(ty: &str) -> String {
    format!("{ty}") + " payload[]"
}
"#;
    let dynamic_name_source = r#"
fn emitted(name: &str) -> String {
    "float ".to_string() + &format!("*{name}")
}
"#;
    for source in [
        pointer_source,
        qualifier_source,
        array_source,
        dynamic_name_source,
    ] {
        let error = scan_rust_source(source).expect_err("split declarations are ambiguous");
        assert_eq!(error.kind, ScanErrorKind::IncompleteFragment);
    }
}

#[test]
fn unconstrained_dynamic_c_source_fails_closed() {
    let source = r#"
fn emitted(declaration: &str) -> String {
    format!("{declaration}")
}
"#;
    let rows = scan_rust_source(source).expect("valid Rust parses");
    assert!(rows.iter().any(|row| {
        row.kind == "raw-element-pointer" && row.signature.contains("shape=unconstrained-emission")
    }));
}

#[test]
fn stringify_carriers_are_scanned() {
    let source = r#"
fn emitted() -> &'static str {
    stringify!(float *payload)
}
"#;
    let rows = scan_rust_source(source).expect("stringify input is structural Rust syntax");
    assert!(rows.iter().any(|row| row.kind == "raw-element-pointer"));
}

#[test]
fn production_after_a_cfg_test_module_is_still_scanned() {
    let source = r#"
#[cfg(test)]
mod tests {
    fn fixture(tensor: Tensor) { let _ = tensor.data; }
}

fn production(tensor: Tensor) {
    let _ = tensor.data;
}
"#;
    let rows = scan_rust_source(source).expect("valid Rust parses");
    let accesses: Vec<_> = rows
        .iter()
        .filter(|row| row.kind == "direct-data-access")
        .collect();
    assert_eq!(accesses.len(), 1);
    assert!(accesses[0].owner.contains("production"));
}

#[test]
fn shared_capacity_census_primitives_keep_their_contract() {
    assert_eq!(
        canonical_c_tokens("const  bool * value;"),
        "const _Bool * value ;"
    );
    assert_eq!(
        strip_c_comments("float /* x */ *p; // y\n"),
        "float   *p; \n"
    );
    let typedefs = collect_typedefs("typedef double sample; typedef sample alias;");
    assert_eq!(
        classify("alias *value", &typedefs),
        vec!["float-carrier", "numeric-op"]
    );
}
