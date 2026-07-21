//! Phase 2 typed-consumer inventory for chelis#730.
//!
//! This test is supporting evidence: rustc exhaustiveness is the authority.
//! Once every row below consumes `chelis_vocab::EffectKind` directly and has
//! no wildcard arm, adding a variant to the single declaration creates the
//! required compiler-error work-list. The inventory prevents an untyped
//! string consumer from sitting outside that mutation oracle.

use std::fs;
use std::path::{Path, PathBuf};

struct Consumer {
    path: &'static str,
    role: &'static str,
    required: &'static [&'static str],
    forbidden: &'static [&'static str],
}

const EFFECT_KIND_CONSUMERS: &[Consumer] = &[
    Consumer {
        path: "crates/chelis-deep/src/effect_kind.rs",
        role: "Deep metadata decode boundary",
        required: &["EffectKindInput", "Result<EffectKind"],
        forbidden: &["Option<EffectKind"],
    },
    Consumer {
        path: "crates/chelis-types/src/infer.rs",
        role: "checker handle-effect semantics",
        required: &["use chelis_vocab::EffectKind", "decode_effect_kind("],
        forbidden: &["EffectKind::from_symbol", ".and_then(EffectKind"],
    },
    Consumer {
        path: "crates/chelis-effects/src/lib.rs",
        role: "effect inference and handler/target validation",
        required: &["use chelis_vocab::EffectKind", "decode_effect_kind("],
        forbidden: &[
            "match effect_name(list)",
            "effect_name(list) == Some(\"resource\")",
        ],
    },
    Consumer {
        path: "crates/chelis-ir/src/lower.rs",
        role: "IR lowering handle-effect semantics",
        required: &["use chelis_vocab::EffectKind", "decode_effect_kind("],
        forbidden: &["EffectKind::from_symbol", ".and_then(EffectKind"],
    },
    Consumer {
        path: "crates/chelis-ir/src/host.rs",
        role: "host lowering handle-effect semantics",
        required: &["chelis_vocab::EffectKind", "decode_effect_kind("],
        forbidden: &["EffectKind::from_symbol(effect)"],
    },
    Consumer {
        path: "crates/chelis-compiler-api/src/runtime/eval.rs",
        role: "compiler-api evaluator handle-effect semantics",
        required: &["use chelis_vocab::EffectKind", "decode_effect_kind("],
        forbidden: &["if effect == \"random\""],
    },
    Consumer {
        path: "crates/chelis-surf/src/desugar.rs",
        role: "Surf-to-Deep canonical effect serialization",
        required: &["use chelis_vocab::EffectKind", ".symbol()"],
        forbidden: &[
            "meta_with_entries(vec![(\"effect\".to_string(), sym(\"random\"))])",
            "meta_with_entries(vec![(\"effect\".to_string(), sym(\"resource\"))])",
        ],
    },
    Consumer {
        path: "crates/chelis-surf/src/decompile.rs",
        role: "both Deep-to-Surf handle-effect decompilers",
        required: &["use chelis_vocab::EffectKind", "decode_effect_kind("],
        forbidden: &[
            "Some(\"random\") => format!(\"with seed",
            "Some(\"resource\") => format!(\"with device",
        ],
    },
];

const RUNTIME_DTYPE_CONSUMERS: &[Consumer] = &[
    Consumer {
        path: "crates/chelis-types/src/types.rs",
        role: "Prim-to-runtime ABI adapter",
        required: &["runtime_dtype(self)", "Result<RuntimeDType"],
        forbidden: &[],
    },
    Consumer {
        path: "crates/chelis-runtime/src/lib.rs",
        role: "runtime ABI decode and semantic dispatch",
        required: &[
            "decode_runtime_dtype(dtype: c_int)",
            "tensor_elem_size(dtype: RuntimeDType)",
            "read_index_slot(",
            "dtype: RuntimeDType",
        ],
        forbidden: &[
            "fn tensor_elem_size(dtype: c_int)",
            "match (*t).dtype",
            "match (*tensor).dtype",
        ],
    },
    Consumer {
        path: "crates/chelis-backend-c/src/emit.rs",
        role: "C codegen dtype macro selection",
        required: &[".runtime_dtype()", ".c_macro()"],
        forbidden: &["Prim::F32 => \"CHELIS_F32\""],
    },
    Consumer {
        path: "crates/chelis-backend-c/src/host_emit.rs",
        role: "C host/sparse dtype macro selection",
        required: &["prim.runtime_dtype()", "self.runtime_dtype().c_macro()"],
        forbidden: &[],
    },
    Consumer {
        path: "crates/chelis-backend-hip/src/emit.rs",
        role: "HIP codegen dtype macro selection",
        required: &[".runtime_dtype()", ".c_macro()"],
        forbidden: &["Prim::F32 => \"CHELIS_F32\""],
    },
    Consumer {
        path: "crates/chelis-backend-hip/runtime/chelis_hip_runtime.h",
        role: "HIP allocation byte width",
        required: &["chelis_runtime_dtype_size_checked(dtype)"],
        forbidden: &["return sizeof(float);"],
    },
    Consumer {
        path: "crates/chelis-backend-metal/src/dtype.rs",
        role: "Metal runtime dtype tag selection",
        required: &["prec.runtime_dtype()", ".c_macro()"],
        forbidden: &["Prim::F32 => \"CHELIS_F32\""],
    },
    Consumer {
        path: "crates/chelis-python/src/lib.rs",
        role: "Python FFI dtype constant",
        required: &["RuntimeDType::F32.id()"],
        forbidden: &["const CHELIS_F32: i32 = 0"],
    },
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("chelis-cli is two levels below workspace root")
        .to_path_buf()
}

#[test]
fn added_effect_variant_will_force_every_semantic_consumer_to_decide() {
    let root = repo_root();
    let mut failures = Vec::new();
    for consumer in EFFECT_KIND_CONSUMERS {
        let path = root.join(consumer.path);
        let source = match fs::read_to_string(&path) {
            Ok(source) => source,
            Err(err) => {
                failures.push(format!(
                    "{} ({}): missing inventory target: {err}",
                    consumer.path, consumer.role
                ));
                continue;
            }
        };
        for needle in consumer.required {
            if !source.contains(needle) {
                failures.push(format!(
                    "{} ({}): missing typed marker `{needle}`",
                    consumer.path, consumer.role
                ));
            }
        }
        for needle in consumer.forbidden {
            if source.contains(needle) {
                failures.push(format!(
                    "{} ({}): raw semantic dispatch remains: `{needle}`",
                    consumer.path, consumer.role
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "EffectKind mutation inventory is not closed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn added_runtime_dtype_will_force_runtime_and_codegen_consumers_to_decide() {
    let root = repo_root();
    let mut failures = Vec::new();
    for consumer in RUNTIME_DTYPE_CONSUMERS {
        let path = root.join(consumer.path);
        let source = match fs::read_to_string(&path) {
            Ok(source) => source,
            Err(err) => {
                failures.push(format!(
                    "{} ({}): missing inventory target: {err}",
                    consumer.path, consumer.role
                ));
                continue;
            }
        };
        for needle in consumer.required {
            if !source.contains(needle) {
                failures.push(format!(
                    "{} ({}): missing typed marker `{needle}`",
                    consumer.path, consumer.role
                ));
            }
        }
        for needle in consumer.forbidden {
            if source.contains(needle) {
                failures.push(format!(
                    "{} ({}): duplicate/raw dtype authority remains: `{needle}`",
                    consumer.path, consumer.role
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "RuntimeDType mutation inventory is not closed:\n{}",
        failures.join("\n")
    );
}

#[test]
fn vocabulary_owner_has_no_dependencies() {
    let manifest = fs::read_to_string(repo_root().join("crates/chelis-vocab/Cargo.toml"))
        .expect("read chelis-vocab manifest");
    assert!(
        !manifest.contains("[dependencies]"),
        "chelis-vocab must remain a true dependency bottom; found a dependencies table"
    );
}
