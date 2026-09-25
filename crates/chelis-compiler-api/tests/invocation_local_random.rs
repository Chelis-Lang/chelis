//! Native keyed draws hold no invocation state: repeated, reentrant and
//! concurrent public calls each produce their own keys' bits. (The legacy C
//! Random frame these tests once guarded is gone with the counter stream.)
mod ownership_support;
use ownership_support::{GeneratedProgram, balanced, emit, run, run_with_peers};

const SOURCE: &str = r#"
def draw(k: key, x: tensor[2, f32]) -> tensor[2, f32] = uniform_like(k, x, 0.0f32, 1.0f32)
def seeded(x: tensor[2, f32]) -> tensor[2, f32] = draw(key_from_seed(42i64), x)
def other(x: tensor[2, f32]) -> tensor[2, f32] = draw(key_from_seed(4294967295i64), x)
"#;

// [05-OP-8] over [0, 1) at f32: the f32 rounding of key_ref.py's
// `unit(key_from_seed(SEED), i)`, i = 0, 1.
const SEED_42: &str = "0x3efa06feULL, 0x3e762d86ULL";
const SEED_4294967295: &str = "0x3eaff421ULL, 0x3f32529fULL";

/// key_ref.py's `unit` rounded to each dtype for the keys of
/// `(a, rest) = split_key(key_from_seed(42))` and `(b, c) = split_key(rest)`.
fn split_tree_bits(dtype: &str) -> [&'static str; 3] {
    match dtype {
        "f32" => [
            "0x3f639647ULL, 0x3e97c256ULL",
            "0x3f7304c5ULL, 0x3e9e91b1ULL",
            "0x3f2e46bcULL, 0x3f1127beULL",
        ],
        "f64" => [
            "0x3fec72c8d31fea46ULL, 0x3fd2f84ab85b9602ULL",
            "0x3fee60989e164c96ULL, 0x3fd3d2362d1750c2ULL",
            "0x3fe5c8d785138824ULL, 0x3fe224f7cdae1695ULL",
        ],
        "f16" => [
            "0x3b1dULL, 0x34beULL",
            "0x3b98ULL, 0x34f5ULL",
            "0x3972ULL, 0x3889ULL",
        ],
        "bf16" => [
            "0x3f64ULL, 0x3e98ULL",
            "0x3f73ULL, 0x3e9fULL",
            "0x3f2eULL, 0x3f11ULL",
        ],
        _ => unreachable!(),
    }
}

/// Every float dtype: three draws on the leaves of a split tree, repeated
/// in one process, give the reference bits each time; a call leaves no
/// state behind for the next one.
#[test]
fn split_tree_draws_repeat_their_reference_bits_at_every_float_dtype() {
    for (dtype, ctype, tag) in [
        ("f32", "uint32_t", "CHELIS_DTYPE_F32"),
        ("f64", "uint64_t", "CHELIS_DTYPE_F64"),
        ("f16", "uint16_t", "CHELIS_DTYPE_F16"),
        ("bf16", "uint16_t", "CHELIS_DTYPE_BF16"),
    ] {
        let source = format!(
            r#"
def sample(k: key, x: &tensor[2, {dtype}]) -> tensor[2, {dtype}] = uniform_like(k, x, 0.0f32, 1.0f32)
def nested(x: tensor[2, {dtype}]) -> (tensor[2, {dtype}], tensor[2, {dtype}], tensor[2, {dtype}]) = {{
  (a, rest) = split_key(key_from_seed(42i64))
  (b, c) = split_key(rest)
  (sample(a, &x), sample(b, &x), sample(c, &x))
}}
"#
        );
        let c = emit(&source, "nested");
        let nested = c.symbol("nested").to_string();
        let [first, second, third] = split_tree_bits(dtype);
        let driver = format!(
            r#"
int main(void) {{
    int64_t n = 2;
    chelis_tensor *x = chelis_alloc(1, &n, {tag});
    const {ctype} expected[3][2] = {{{{{first}}}, {{{second}}}, {{{third}}}}};
    for (int repeat = 0; repeat < 8; ++repeat) {{
        chelis_tuple *result = {nested}(x);
        assert(chelis_tuple_len(result) == 3);
        for (int i = 0; i < 3; ++i) {{
            chelis_value part = chelis_tuple_get(result, i);
            chelis_read_view view = chelis_tensor_read_view(chelis_tensor_borrow_value(part));
            assert(view.dtype == {tag} && view.count == 2);
            assert(memcmp(view.data, expected[i], sizeof(expected[i])) == 0);
            chelis_value_release(part);
        }}
        chelis_tuple_release(result);
    }}
    chelis_tensor_release(x);
    return 0;
}}
"#
        );
        balanced(&run(&c, &driver));
    }
}

#[test]
fn random_callback_parameters_retain_the_existing_unsupported_boundary() {
    use chelis_compiler_api::compiler::compile;
    use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};
    let source = r#"
def apply(f: key -> f32 -> f32, k: key, x: f32) -> f32 = f(k, x)
def sample_scalar(k: key, x: f32) -> f32 = tensor_to_scalar(uniform_like(k, scalar_to_tensor(x), 0.0f32, 1.0f32))
def applied(x: f32) -> f32 = apply(sample_scalar, key_from_seed(42i64), x)
"#;
    let error = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: Some("fixture".into()),
    })
    .unwrap_err();
    assert_eq!(error.stage, "compile");
    assert!(
        error.errors.iter().any(|error| error
            .message
            .contains("verified user-function call site has no direct-call authority")),
        "{error:?}"
    );
}

#[test]
fn external_tensor_helpers_keep_their_four_argument_abi() {
    let source = "def total(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(x, 0))\ndef derivative(x: tensor[2, f32]) -> tensor[2, f32] = grad(total)(x)";
    let declarations = chelis_surf::parser::parse_str(source).unwrap();
    let deep =
        chelis_surf::desugar::desugar_program(&declarations).expect("Surf fixture must desugar");
    let checked = chelis_types::check_typed_program(&deep).unwrap();
    let checked = chelis_effects::check_program(&checked).unwrap();
    let checked = chelis_types::check_linearity(&checked).unwrap();
    let realizability = chelis_effects::realizability::infer_realizability(
        &checked,
        chelis_backend_c::TENSOR_CAPABLE_PRIMS,
    );
    let manifest = chelis_effects::realizability::compute_root_manifest(&checked, &realizability);
    let lowered =
        chelis_ir::host::try_lower_compiled_program_with_manifest(&checked, &manifest).unwrap();
    let (helpers, host) =
        chelis_backend_c::host_tensor_helper_codegen(lowered.host.unwrap(), "fixture").unwrap();
    assert!(!helpers.is_empty());
    let selected = chelis_backend_c::prepare_host_program_for_codegen(host).unwrap();
    let manifested = chelis_types::manifest::ManifestedProgram::new(
        checked,
        manifest,
        chelis_types::types::Target::C,
    );
    let verified = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_host_ownership(&manifested, selected).unwrap(),
    )
    .unwrap();
    let names = helpers
        .iter()
        .map(|helper| helper.name.clone())
        .collect::<Vec<_>>();
    let host = chelis_backend_c::codegen_host_program_with_external_tensor_helpers(
        &verified, "fixture", &names,
    )
    .unwrap();
    let host = GeneratedProgram::new(host.c_source, host.h_header);
    assert!(
        !host.header().contains("rng"),
        "public header acquired private state"
    );
    assert!(host.declaration("total").contains("(chelis_tensor* x)"));
    let total = host.symbol("total").to_string();
    let peers =
        helpers
            .into_iter()
            .map(|helper| {
                let selected =
                    chelis_backend_c::prepare_dag_for_codegen(helper.dag, Default::default());
                let verified = chelis_ir::ownership::verify_ownership(
                    chelis_ir::ownership::lower_dag_ownership(selected).unwrap(),
                )
                .unwrap();
                let result = chelis_backend_c::codegen(verified, &helper.name).unwrap();
                assert!(result.h_header.contains(
                    "chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out)"
                ));
                result.c_source
            })
            .collect::<Vec<_>>();
    balanced(&run_with_peers(
        &host,
        &peers,
        &format!(
            r#"
{}
int main(void) {{
    chelis_tensor *x = input(2);
    float (*public_entry)(chelis_tensor *) = {};
    for (int i = 0; i < 8; ++i) assert(public_entry(x) == -4.0f);
    chelis_tensor_release(x);
    return 0;
}}
"#,
            host.header(),
            total
        ),
    ));
}

const CHECK: &str = r#"
static void check_bits(chelis_tensor *value, const uint32_t *expected) {
    chelis_read_view view = chelis_tensor_read_view(value);
    assert(view.dtype == CHELIS_DTYPE_F32 && view.count == 2);
    assert(memcmp(view.data, expected, 2 * sizeof(uint32_t)) == 0);
    chelis_tensor_release(value);
}
"#;

fn hooked(c: &GeneratedProgram) -> GeneratedProgram {
    c.with_source(format!(
        "#include \"chelis_runtime.h\"\nstatic chelis_tensor_write *intercept_write(chelis_tensor *);\n#define chelis_tensor_begin_write intercept_write\n{c}\n#undef chelis_tensor_begin_write\n"
    ))
}

#[test]
fn reentrant_public_entry_starts_its_own_context() {
    // A draw reentered from inside another call's write sees only its own key.
    let c = emit(SOURCE, "seeded");
    let draw = c.symbol("other").to_string();
    let seeded = c.symbol("seeded").to_string();
    let driver = format!(
        r#"
{CHECK}
static chelis_tensor *template;
static int armed;
static int observed;
static chelis_tensor_write *intercept_write(chelis_tensor *tensor) {{
    if (armed) {{
        armed = 0;
        observed++;
        const uint32_t zero[] = {{{zero}}};
        check_bits({draw}(template), zero);
    }}
    return chelis_tensor_begin_write(tensor);
}}
int main(void) {{
    template = input(2);
    const uint32_t expected[] = {{{expected}}};
    for (int i = 0; i < 8; ++i) {{
        armed = 1;
        check_bits({seeded}(template), expected);
    }}
    assert(observed == 8);
    chelis_tensor_release(template);
    return 0;
}}
"#,
        zero = SEED_4294967295,
        expected = SEED_42
    );
    balanced(&run(&hooked(&c), &driver));
}

#[test]
fn concurrent_public_entries_draw_independently() {
    let c = emit(SOURCE, "seeded");
    let seeded = c.symbol("seeded").to_string();
    let other = c.symbol("other").to_string();
    let driver = format!(
        r#"
#include <pthread.h>
{CHECK}
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t condition = PTHREAD_COND_INITIALIZER;
static int phase;
static _Thread_local int role;
static chelis_tensor *template;
static chelis_tensor *results[2];
static chelis_tensor_write *intercept_write(chelis_tensor *tensor) {{
    if (role) {{
        int mine = role;
        role = 0;
        pthread_mutex_lock(&mutex);
        if (mine == 1) {{
            phase = 1;
            pthread_cond_broadcast(&condition);
            while (phase < 2) pthread_cond_wait(&condition, &mutex);
        }} else {{
            phase = 2;
            pthread_cond_broadcast(&condition);
            while (phase < 3) pthread_cond_wait(&condition, &mutex);
        }}
        pthread_mutex_unlock(&mutex);
    }}
    return chelis_tensor_begin_write(tensor);
}}
static void *first(void *unused) {{
    (void)unused;
    role = 1;
    results[0] = {seeded}(template);
    pthread_mutex_lock(&mutex);
    phase = 3;
    pthread_cond_broadcast(&condition);
    pthread_mutex_unlock(&mutex);
    return NULL;
}}
static void *second(void *unused) {{
    (void)unused;
    pthread_mutex_lock(&mutex);
    while (phase < 1) pthread_cond_wait(&condition, &mutex);
    pthread_mutex_unlock(&mutex);
    role = 2;
    results[1] = {other}(template);
    return NULL;
}}
int main(void) {{
    template = input(2);
    pthread_t a, b;
    assert(pthread_create(&a, NULL, first, NULL) == 0);
    assert(pthread_create(&b, NULL, second, NULL) == 0);
    assert(pthread_join(a, NULL) == 0);
    assert(pthread_join(b, NULL) == 0);
    assert(phase == 3);
    const uint32_t expected_a[] = {{{a}}}, expected_b[] = {{{b}}};
    check_bits(results[0], expected_a);
    check_bits(results[1], expected_b);
    chelis_tensor_release(template);
    return 0;
}}
"#,
        a = SEED_42,
        b = SEED_4294967295
    );
    balanced(&run(&hooked(&c), &driver));
}
