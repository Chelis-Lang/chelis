use chelis_c_surface::{
    CarrierUse, ScanErrorKind, canonical_c_tokens, classify, collect_typedefs, scan_c_source,
    scan_rust_source, strip_c_comments,
};

fn signatures<'a>(rows: &'a [CarrierUse], kind: &str) -> Vec<&'a str> {
    rows.iter()
        .filter(|row| row.kind == kind)
        .map(|row| row.signature.as_str())
        .collect()
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
    assert_eq!(signatures(&rows, "raw-element-pointer").len(), 4);
    assert_eq!(signatures(&rows, "width-arithmetic").len(), 2);
    assert!(
        signatures(&rows, "raw-element-pointer")
            .iter()
            .any(|signature| signature.contains("shape=array") && signature.contains("float"))
    );
    assert!(
        signatures(&rows, "raw-element-pointer")
            .iter()
            .any(|signature| signature.contains("shape=cast-pointer"))
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
    assert!(raw.iter().any(|signature| {
        signature.contains("authored=CHELIS_SAMPLE") && signature.contains("resolved=float")
    }));
    assert!(raw.iter().any(|signature| {
        signature.contains("authored=chelis_vec4") && signature.contains("resolved=double")
    }));
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
extern void aggregate(ChelisContext *context, enum ChelisMode *mode);
static MTLGPUFamily families[] = { MTLGPUFamilyApple7 };
static size_t opaque_width(void) { return sizeof(struct ChelisContext); }
"#;
    let rows = scan_c_source(source, "fixture").expect("nonnumeric types parse");
    assert!(rows.is_empty());
}

#[test]
fn nested_sizeof_type_names_are_parsed_without_confusing_expressions() {
    let source = r#"
static size_t widths(float *payload, size_t index) {
    return sizeof((const float)) + sizeof(double[4])
        + sizeof(payload[index]) + sizeof(index * index);
}
"#;
    let rows = scan_c_source(source, "fixture").expect("valid sizeof operands parse");
    let widths = signatures(&rows, "width-arithmetic");
    assert_eq!(widths.len(), 2);
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
fn expressions_and_comments_are_not_misread_as_declarators() {
    let source = r#"
// extern void ignored(float const *payload);
/* sizeof(const double) */
static int arithmetic(int a, int b) { return a * b; }
static int dereference(int *p) { return *p; }
"#;
    let rows = scan_c_source(source, "fixture").expect("ordinary expressions parse");
    assert_eq!(signatures(&rows, "raw-element-pointer").len(), 1);
    assert!(signatures(&rows, "width-arithmetic").is_empty());
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
    for source in [pointer_source, qualifier_source, array_source] {
        let error = scan_rust_source(source).expect_err("split declarations are ambiguous");
        assert_eq!(error.kind, ScanErrorKind::IncompleteFragment);
    }
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
