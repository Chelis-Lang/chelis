//! Invocation-local legacy C Random state, including actual reentrant and
//! interleaved calls. These are transport tests, not fixed-control Dropout tests.
#[allow(dead_code)]
mod ownership_support;
use ownership_support::{authored_c_symbol, balanced, emit, run, run_with_peers};

const SOURCE: &str = r#"
def draw(x: tensor[2, f32]) -> tensor[2, f32] ! { Random } = uniform_like(x, 0.0f32, 1.0f32)
def seeded(x: tensor[2, f32]) -> tensor[2, f32] = with seed(42i64) { draw(x) }
def other(x: tensor[2, f32]) -> tensor[2, f32] = with seed(4294967295i64) { draw(x) }
"#;

// Deliberately observe the legacy sampler. Changing storage does not authorize
// changing its seed formula or substituting the fixed-control evaluator stream.
fn bits(seed: u64, ordinal: u64) -> String {
    stored_bits("f32", seed, ordinal)
}

fn stored_bits(dtype: &str, seed: u64, ordinal: u64) -> String {
    (0..2_u64)
        .map(|index| {
            let mut word = seed
                ^ ordinal.wrapping_mul(0x9E3779B97F4A7C15)
                ^ index.wrapping_mul(0x9E3779B97F4A7C15);
            word ^= word >> 30;
            word = word.wrapping_mul(0xBF58476D1CE4E5B9);
            word ^= word >> 27;
            word = word.wrapping_mul(0x94D049BB133111EB);
            word ^= word >> 31;
            let unit = (word >> 11) as f64 / (1_u64 << 53) as f64;
            let raw = match dtype {
                "f64" => unit.to_bits(),
                "f32" => (unit as f32).to_bits() as u64,
                "f16" => half::f16::from_f32(unit as f32).to_bits() as u64,
                "bf16" => half::bf16::from_f32(unit as f32).to_bits() as u64,
                _ => unreachable!(),
            };
            format!("0x{raw:x}ULL")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[test]
fn nested_seed_restore_and_next_draw_preserve_all_legacy_dtype_bits() {
    // The existing Uniform checker/lowerer admits f32 bounds for every
    // template dtype. This transport repair does not change that surface.
    for (dtype, ctype, tag) in [
        ("f32", "uint32_t", "CHELIS_DTYPE_F32"),
        ("f64", "uint64_t", "CHELIS_DTYPE_F64"),
        ("f16", "uint16_t", "CHELIS_DTYPE_F16"),
        ("bf16", "uint16_t", "CHELIS_DTYPE_BF16"),
    ] {
        let source = format!(
            r#"
def sample(x: tensor[2, {dtype}]) -> tensor[2, {dtype}] ! {{ Random }} = uniform_like(x, 0.0f32, 1.0f32)
def nested(x: tensor[2, {dtype}]) -> (tensor[2, {dtype}], tensor[2, {dtype}], tensor[2, {dtype}]) = with seed(42i64) {{
    a = sample(x)
    b = with seed(4294967295i64) {{ sample(x) }}
    c = sample(x)
    (a, b, c)
}}
"#
        );
        let c = emit(&source, "nested");
        let nested = authored_c_symbol("nested");
        let driver = format!(
            r#"
int main(void) {{
    int64_t n = 2;
    chelis_tensor *x = chelis_alloc(1, &n, {tag});
    const {ctype} expected[3][2] = {{{{{first}}}, {{{inner}}}, {{{next}}}}};
    assert(memcmp(expected[0], expected[2], sizeof(expected[0])) != 0);
    assert(memcmp(expected[0], expected[1], sizeof(expected[0])) != 0);
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
"#,
            first = stored_bits(dtype, 42, 0),
            inner = stored_bits(dtype, 4294967295, 0),
            next = stored_bits(dtype, 42, 1)
        );
        balanced(&run(&c, &driver));
    }
}

#[test]
fn named_map_callback_inherits_the_invocation_stream() {
    let c = emit(
        r#"
def sample_scalar(x: f32) -> f32 ! { Random } = tensor_to_scalar(uniform_like(scalar_to_tensor(x), 0.0f32, 1.0f32))
def mapped(x: f32) -> tensor[2, f32] = with seed(42i64) { to_tensor(map(sample_scalar, [x, x])) }
"#,
        "mapped",
    );
    let first = bits(42, 0).split(',').next().unwrap().to_string();
    let next = bits(42, 1).split(',').next().unwrap().to_string();
    let mapped = authored_c_symbol("mapped");
    let driver = format!(
        r#"
{CHECK}
int main(void) {{
    const uint32_t expected[] = {{{first}, {next}}};
    assert(expected[0] != expected[1]);
    for (int i = 0; i < 8; ++i) check_bits({mapped}(0.0f), expected);
    return 0;
}}
"#
    );
    balanced(&run(&c, &driver));
}

#[test]
fn random_callback_parameters_retain_the_existing_unsupported_boundary() {
    use chelis_compiler_api::compiler::compile;
    use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};
    let source = r#"
def apply(f: f32 -> f32, x: f32) -> f32 ! { Random } = f(x)
def sample_scalar(x: f32) -> f32 ! { Random } = tensor_to_scalar(uniform_like(scalar_to_tensor(x), 0.0f32, 1.0f32))
def applied(x: f32) -> f32 = with seed(42i64) { apply(sample_scalar, x) }
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
fn recursive_random_calls_inherit_the_same_ordinal() {
    let source = format!(
        r#"
{SOURCE}
def recur(x: tensor[2, f32], n: i64) -> tensor[2, f32] ! {{ Random }} = if eq(n, 0i64) then draw(x) else {{
    ignored = draw(x)
    recur(x, sub(n, 1i64))
}}
def recursive_entry(x: tensor[2, f32]) -> tensor[2, f32] = with seed(42i64) {{ recur(x, 3i64) }}
def ping(x: tensor[2, f32], n: i64) -> tensor[2, f32] ! {{ Random }} = if eq(n, 0i64) then draw(x) else {{
    ignored = draw(x)
    pong(x, sub(n, 1i64))
}}
def pong(x: tensor[2, f32], n: i64) -> tensor[2, f32] ! {{ Random }} = if eq(n, 0i64) then draw(x) else {{
    ignored = draw(x)
    ping(x, sub(n, 1i64))
}}
def mutual_entry(x: tensor[2, f32]) -> tensor[2, f32] = with seed(42i64) {{ ping(x, 3i64) }}
"#
    );
    let c = emit(&source, "recursive_entry");
    let expected = bits(42, 3);
    let recursive_entry = authored_c_symbol("recursive_entry");
    let mutual_entry = authored_c_symbol("mutual_entry");
    balanced(&run(
        &c,
        &format!(
            r#"
{CHECK}
int main(void) {{
    chelis_tensor *x = input(2);
    const uint32_t expected[] = {{{expected}}};
    for (int repeat = 0; repeat < 8; ++repeat) {{
        check_bits({recursive_entry}(x), expected);
        check_bits({mutual_entry}(x), expected);
    }}
    chelis_tensor_release(x);
    return 0;
}}
"#
        ),
    ));
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
    assert!(
        !host.h_header.contains("rng"),
        "public header acquired private state"
    );
    assert!(
        host.h_header
            .contains(&format!("{}(chelis_tensor* x)", authored_c_symbol("total")))
    );
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
        &host.c_source,
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
            host.h_header,
            authored_c_symbol("total")
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

fn hooked(c: &str) -> String {
    format!(
        "#include \"chelis_runtime.h\"\nstatic chelis_tensor_write *intercept_write(chelis_tensor *);\n#define chelis_tensor_begin_write intercept_write\n{c}\n#undef chelis_tensor_begin_write\n"
    )
}

#[test]
fn reentrant_public_entry_starts_its_own_context() {
    let c = emit(SOURCE, "seeded");
    let draw = authored_c_symbol("draw");
    let seeded = authored_c_symbol("seeded");
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
        zero = bits(0, 0),
        expected = bits(42, 0)
    );
    balanced(&run(&hooked(&c), &driver));
}

#[test]
fn concurrent_public_entries_keep_independent_seed_frames() {
    let c = emit(SOURCE, "seeded");
    let seeded = authored_c_symbol("seeded");
    let other = authored_c_symbol("other");
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
        a = bits(42, 0),
        b = bits(4294967295, 0)
    );
    balanced(&run(&hooked(&c), &driver));
}
