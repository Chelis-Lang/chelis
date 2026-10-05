//! Regeneration harness for the pre-v3 numeric wire cache fixtures.
//! Run this source at 010354f91d1efd210037f4090f6e862aa7a68773.
use chelis_compiler_api::{
    COMPILER_VERSION, CompiledContext, compile_reef_context,
    library_cache::{library_cache_key, load_or_build_library_context},
    stdlib_cache::{load_or_build_stdlib_context, stdlib_cache_key, typecheck_cache_dir},
};
use std::{fs, path::PathBuf};
fn main() {
    let output = PathBuf::from(std::env::args().nth(1).expect("output directory"));
    fs::create_dir_all(&output).unwrap();
    let std_source =
        format!("def fixture_base(x: f32) -> f32 = x + 0.5f32\n{NUMERIC_SOURCE}{WITNESS_SOURCE}");
    let dep_source = "def fixture_transform(x: f32) -> f32 = fixture_base(x) + 0.25f32\n";
    let std_decls = chelis_surf::parser::parse_str(&std_source).unwrap();
    let dep_decls = chelis_surf::parser::parse_str(dep_source).unwrap();
    let digest = [0x5a; 32];
    let std_key = stdlib_cache_key(&std_decls, digest);
    let std_ctx = load_or_build_stdlib_context(&std_decls, digest).unwrap();
    assert!(
        std_ctx.library_dag().is_some(),
        "fixture must contain lowered numeric values"
    );
    assert_literal_result(&serde_json::to_value(std_ctx.library_dag().unwrap().raw()).unwrap());
    let std_payloads = numeric_payloads(std_ctx.library_dag().unwrap().raw());
    assert_eq!(
        std_payloads.len(),
        4,
        "old producer must include scalar and tensor numeric payloads: {std_payloads:?}"
    );
    let lib_key = library_cache_key(&dep_decls, std_key);
    assert!(
        load_or_build_library_context(&std_ctx, std_key, &dep_decls)
            .unwrap()
            .is_some()
    );
    // Exercise the actual old consumer as well as the old writer.
    load_or_build_stdlib_context(&std_decls, digest).unwrap();
    assert!(
        load_or_build_library_context(&std_ctx, std_key, &dep_decls)
            .unwrap()
            .is_some()
    );
    let cache = typecheck_cache_dir().unwrap();
    let short = |key: &[u8; 32]| {
        key[..8]
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    let std_path = cache.join(format!(
        "chelis-std-{}-{}.tc",
        chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION,
        short(&std_key)
    ));
    let lib_path = cache.join(format!("chelis-lib-{}.tc", short(&lib_key)));
    assert!(
        std_path.is_file(),
        "actual cache writer must produce stdlib file"
    );
    assert!(
        lib_path.is_file(),
        "actual cache writer must produce library file"
    );
    let before = [
        fs::metadata(&std_path).unwrap().modified().unwrap(),
        fs::metadata(&lib_path).unwrap().modified().unwrap(),
    ];
    load_or_build_stdlib_context(&std_decls, digest).unwrap();
    assert!(
        load_or_build_library_context(&std_ctx, std_key, &dep_decls)
            .unwrap()
            .is_some()
    );
    assert_eq!(
        before,
        [
            fs::metadata(&std_path).unwrap().modified().unwrap(),
            fs::metadata(&lib_path).unwrap().modified().unwrap()
        ],
        "old consumers must hit without rewriting either file"
    );
    fs::write(
        output.join("stdlib-v15-key-input.bin"),
        chelis_compiler_api::stdlib_cache_key_input_bytes(&std_decls, digest),
    )
    .unwrap();
    fs::write(
        output.join("library-v11-key-input.bin"),
        chelis_compiler_api::library_cache_key_input_bytes(&dep_decls, std_key),
    )
    .unwrap();
    fs::copy(&std_path, output.join("stdlib-v15.tc")).unwrap();
    fs::copy(&lib_path, output.join("library-v11.tc")).unwrap();
    let package = output.join("package");
    package_fixture(&package);
    let reef_home = PathBuf::from(std::env::var_os("CHELIS_REEF_HOME").unwrap());
    let context =
        compile_reef_context(&reef_home, &package, &chelis_std_bundle::EMBEDDED_RUNTIME).unwrap();
    assert_literal_result(&serde_json::to_value(&context).unwrap()["library_dag"]);
    let context_payloads = context_numeric_payloads(&context);
    assert_eq!(
        context_payloads, std_payloads,
        "both old cache routes must contain the same numeric payloads"
    );
    let context_path = output.join("context-v17.ctx");
    context.save(&context_path).unwrap();
    assert!(
        CompiledContext::load_if_fresh(
            &context_path,
            &reef_home,
            &package,
            &chelis_std_bundle::EMBEDDED_RUNTIME
        )
        .unwrap()
        .is_some()
    );
    let worker_bytes = context.encode().expect("old worker producer");
    let worker_context = CompiledContext::decode(&worker_bytes).expect("old worker consumer");
    assert_literal_result(&serde_json::to_value(&worker_context).unwrap()["library_dag"]);
    let disk_context = CompiledContext::load_if_fresh(
        &context_path,
        &reef_home,
        &package,
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .unwrap()
    .unwrap();
    assert_literal_result(&serde_json::to_value(&disk_context).unwrap()["library_dag"]);
    assert_eq!(
        bincode::serialize(&context).unwrap(),
        bincode::serialize(&worker_context).unwrap()
    );
    fs::write(output.join("worker-unversioned.bin"), worker_bytes).unwrap();
    let manifest = serde_json::json!({
        "producer_commit":"010354f91d1efd210037f4090f6e862aa7a68773",
        "compiler_build":chelis_compiler_api::build_fingerprint(),
        "context_version":17, "stdlib_version":15, "library_version":11,
        "stdlib_source":std_source,"dependency_source":dep_source,
        "source_digest":digest.to_vec(),"stdlib_key":std_key.to_vec(),"library_key":lib_key.to_vec(),
        "old_producer_old_consumer":"pass",
        "old_worker_producer_old_worker_consumer":"pass",
        "stdlib_numeric_payloads":std_payloads,"context_numeric_payloads":context_payloads,
        "old_literal_result_declaration":"tensor[4, f32]"
    });
    fs::write(
        output.join("producer.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    println!("{}", serde_json::to_string(&manifest).unwrap());
}

pub const NUMERIC_SOURCE: &str = "def fixture_wide() -> int64 = 9007199254740993i64\ndef fixture_zero() -> f64 = 0.0f64\ndef fixture_tensor() -> tensor[2, f64] = to_tensor([-0.0f64, 1.0000000000000002f64])\ndef fixture_tensor_int() -> tensor[2, int64] = to_tensor([9007199254740993i64, -9007199254740993i64])\n";

pub fn numeric_payloads(library: &chelis_ir::lower::LoweredLibrary) -> Vec<serde_json::Value> {
    use chelis_ir::RiscOp;
    use chelis_types::{StorageView, types::Prim};
    let mut rows = Vec::new();
    for node in library.dag().nodes() {
        let (mut row, bytes) = match &node.op {
            RiscOp::Const { value } if value.prim() == Prim::Int64 => (
                serde_json::json!({"kind":"scalar","dtype":"int64","value":value.as_i64_exact().unwrap()}),
                bincode::serialize(value).unwrap(),
            ),
            RiscOp::Const { value } if value.prim() == Prim::F64 => (
                serde_json::json!({"kind":"scalar","dtype":"f64","bits":format!("{:016x}", value.as_f64_lossy().to_bits())}),
                bincode::serialize(value).unwrap(),
            ),
            RiscOp::ConstTensor { data } => {
                let row = match data.view() {
                    StorageView::F64(values) => {
                        serde_json::json!({"kind":"storage","dtype":"f64","bits":values.iter().map(|value|format!("{:016x}",value.to_bits())).collect::<Vec<_>>()})
                    }
                    StorageView::I64(values) => {
                        serde_json::json!({"kind":"storage","dtype":"int64","values":values})
                    }
                    _ => continue,
                };
                (row, bincode::serialize(data).unwrap())
            }
            _ => continue,
        };
        row["bincode"] = serde_json::json!(
            bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
        rows.push(row);
    }
    rows.sort_by_key(|row| row.to_string());
    rows.dedup();
    rows
}

pub fn context_numeric_payloads(context: &CompiledContext) -> Vec<serde_json::Value> {
    let body = serde_json::to_value(context).unwrap();
    let library = serde_json::from_value(body["library_dag"].clone()).unwrap();
    numeric_payloads(&library)
}

pub fn package_fixture(package: &std::path::Path) {
    fs::create_dir_all(package.join("src")).unwrap();
    fs::create_dir_all(package.join("mylib/src")).unwrap();
    fs::write(package.join("reef.toml"), format!("[package]\nname = \"fixture\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Fixture\"\n\n[dependencies]\nmylib = {{ path = \"./mylib\" }}\n")).unwrap();
    fs::write(
        package.join("src/main.ch"),
        "module Fixture.Main\n\ndef main() -> f32 = 1.0f32\n",
    )
    .unwrap();
    fs::write(package.join("mylib/reef.toml"), format!("[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Mylib\"\n")).unwrap();
    fs::write(package.join("mylib/src/math.ch"), format!("module Mylib.Math\nexport (scale, fixture_wide, fixture_zero, fixture_tensor, fixture_tensor_int, fixture_literal, fixture_claim)\n\ndef scale(x: f32) -> f32 = x * 0.5f32\n{NUMERIC_SOURCE}{WITNESS_SOURCE}")).unwrap();
    fs::write(package.join("reef.lock"), format!("[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n")).unwrap();
}

pub const WITNESS_SOURCE: &str = "def fixture_literal(b: tensor[f32], x: tensor[rows, f32]) -> tensor[4, f32] = insert(b, 0i32, shape(x, 0i32))\nfixture_claim = fixture_literal(scalar_to_tensor(7.0f32), to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))\n";

// Reusable library caches keep this callable as a checked definition. Its
// literal result must survive independently of later invocation lowering.
pub fn assert_literal_result(library: &serde_json::Value) {
    let definitions = library["program_defs"].as_object().unwrap();
    let selected: Vec<_> = definitions
        .iter()
        .filter(|(name, _)| name.ends_with("fixture_literal"))
        .collect();
    assert_eq!(
        selected.len(),
        1,
        "fixture must contain exactly one literal-result declaration"
    );
    let fields = selected[0].1["List"][0]["elements"].as_array().unwrap();
    let metadata = fields[1]["Map"][0]["entries"].as_array().unwrap();
    let signature = &metadata.iter().find(|row| row[0] == "type").unwrap()[1];
    let parts = signature["List"][0]["elements"].as_array().unwrap();
    assert_eq!(parts[0]["Atom"][0]["Tag"], "TFn");
    let result = parts.last().unwrap()["List"][0]["elements"]
        .as_array()
        .unwrap();
    assert_eq!(result.len(), 4);
    assert_eq!(result[0]["Atom"][0]["Tag"], "TTensor");
    let extent = result[2]["List"][0]["elements"].as_array().unwrap();
    assert_eq!(extent[0]["Atom"][0]["Tag"], "DLit");
    assert_eq!(extent[2]["Atom"][0]["Int"], 4);
    let dtype = result[3]["List"][0]["elements"].as_array().unwrap();
    assert_eq!(dtype[0]["Atom"][0]["Tag"], "TPrim");
    assert_eq!(dtype[2]["Atom"][0]["Name"], "f32");
}
