//! Phase 2 typed-consumer inventory for chelis#730.
//!
//! This test is supporting evidence: rustc exhaustiveness is the authority.
//! Once every row below consumes `chelis_vocab::EffectKind` directly and has
//! no wildcard arm, adding a variant to the single declaration creates the
//! required compiler-error work-list. The inventory prevents an untyped
//! string consumer from sitting outside that mutation oracle.

use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
enum ConsumerSource {
    File(&'static str),
    RustTree(&'static str),
}

impl ConsumerSource {
    fn path(self) -> &'static str {
        match self {
            Self::File(path) | Self::RustTree(path) => path,
        }
    }
}

struct Consumer {
    source: ConsumerSource,
    role: &'static str,
    required: &'static [&'static str],
    forbidden: &'static [&'static str],
}

const EFFECT_KIND_CONSUMERS: &[Consumer] = &[
    Consumer {
        source: ConsumerSource::File("crates/chelis-deep/src/effect_kind.rs"),
        role: "Deep metadata decode boundary",
        required: &["EffectKindInput", "Result<EffectKind"],
        forbidden: &["Option<EffectKind"],
    },
    Consumer {
        source: ConsumerSource::RustTree("crates/chelis-types/src/infer"),
        role: "checker handle-effect semantics",
        required: &["use chelis_vocab::EffectKind", "decode_effect_kind("],
        forbidden: &["EffectKind::from_symbol", ".and_then(EffectKind"],
    },
    Consumer {
        source: ConsumerSource::File("crates/chelis-effects/src/lib.rs"),
        role: "effect inference and handler/target validation",
        required: &["use chelis_vocab::EffectKind", "decode_effect_kind("],
        forbidden: &[
            "match effect_name(list)",
            "effect_name(list) == Some(\"resource\")",
        ],
    },
    Consumer {
        source: ConsumerSource::File("crates/chelis-ir/src/lower.rs"),
        role: "IR lowering handle-effect semantics",
        required: &["use chelis_vocab::EffectKind", "decode_effect_kind("],
        forbidden: &["EffectKind::from_symbol", ".and_then(EffectKind"],
    },
    Consumer {
        source: ConsumerSource::File("crates/chelis-ir/src/host.rs"),
        role: "host lowering handle-effect semantics",
        required: &["chelis_vocab::EffectKind", "decode_effect_kind("],
        forbidden: &["EffectKind::from_symbol(effect)"],
    },
    Consumer {
        source: ConsumerSource::File("crates/chelis-compiler-api/src/runtime/eval.rs"),
        role: "compiler-api evaluator handle-effect semantics",
        required: &["use chelis_vocab::EffectKind", "decode_effect_kind("],
        forbidden: &["if effect == \"random\""],
    },
    Consumer {
        source: ConsumerSource::File("crates/chelis-surf/src/desugar.rs"),
        role: "Surf-to-Deep canonical effect serialization",
        required: &[
            "use chelis_vocab::EffectKind",
            "M::Effect",
            "EffectKind::Random",
            "EffectKind::Resource",
        ],
        forbidden: &[
            "meta_with_entries(vec![(\"effect\".to_string(), sym(\"random\"))])",
            "meta_with_entries(vec![(\"effect\".to_string(), sym(\"resource\"))])",
        ],
    },
    Consumer {
        source: ConsumerSource::File("crates/chelis-surf/src/resugar.rs"),
        role: "shared Deep-to-Surf AST handle-effect resugaring",
        required: &[
            "use chelis_vocab::EffectKind",
            ".effect()",
            "decode_effect_kind(",
        ],
        forbidden: &[
            "Some(\"random\") => format!(\"with seed",
            "Some(\"resource\") => format!(\"with device",
        ],
    },
];

const RUNTIME_DTYPE_CONSUMERS: &[Consumer] = &[
    Consumer {
        source: ConsumerSource::File("crates/chelis-types/src/types.rs"),
        role: "Prim-to-runtime ABI adapter",
        required: &["runtime_dtype(self)", "Result<RuntimeDType"],
        forbidden: &[],
    },
    Consumer {
        source: ConsumerSource::File("crates/chelis-runtime/src/lib.rs"),
        role: "runtime ABI decode and semantic dispatch",
        required: &[
            "decode_runtime_dtype(dtype: chelis_dtype)",
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
        source: ConsumerSource::File("crates/chelis-backend-c/src/emit.rs"),
        role: "C codegen dtype macro selection",
        required: &[".runtime_dtype()", ".c_macro()"],
        forbidden: &["Prim::F32 => \"CHELIS_DTYPE_F32\""],
    },
    Consumer {
        source: ConsumerSource::File("crates/chelis-backend-c/src/host_emit.rs"),
        role: "C host/sparse dtype macro selection",
        required: &["prim.runtime_dtype()", "self.runtime_dtype().c_macro()"],
        forbidden: &[],
    },
    Consumer {
        source: ConsumerSource::File("crates/chelis-backend-hip/src/emit.rs"),
        role: "HIP codegen dtype macro selection",
        required: &[".runtime_dtype()", ".c_macro()"],
        forbidden: &["Prim::F32 => \"CHELIS_DTYPE_F32\""],
    },
    // chelis#1360: this row used to REQUIRE `case CHELIS_DTYPE_F64:` and
    // `case CHELIS_DTYPE_BOOL:` here, which is to say it required the header
    // to hold its own copy of the width table. That copy is what broke: when
    // chelis#1308 narrowed bool to one byte, this file was updated and the
    // emitter's three other copies were not, so `cmplt` and `cast` went on
    // dispatching four-byte kernels over a one-byte allocation.
    //
    // The header no longer decides anything per dtype - it calls
    // `chelis_dtype_size`, whose Rust side is `tensor_elem_size`, required by
    // the `chelis-runtime` row above and implemented as `dtype.byte_width()`.
    // That is an exhaustive match on `Repr`, so a new dtype breaks the build
    // there rather than merely missing a string here. Per this file's own
    // header, rustc exhaustiveness is the authority and the inventory exists
    // to keep untyped string consumers from sitting outside it; delegating
    // removes this file from that category instead of keeping it compliant.
    //
    // The row therefore inverts: require the delegation, forbid the
    // restatement.
    Consumer {
        source: ConsumerSource::File("crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"),
        role: "HIP allocation byte width",
        required: &[
            "#include \"chelis_runtime_dtype.h\"",
            "return (size_t)chelis_dtype_size((chelis_dtype)dtype);",
        ],
        forbidden: &[
            "return sizeof(float);",
            "case CHELIS_DTYPE_F64:",
            "case CHELIS_DTYPE_BOOL:",
        ],
    },
    Consumer {
        source: ConsumerSource::File("crates/chelis-backend-metal/src/dtype.rs"),
        role: "Metal runtime dtype tag selection",
        required: &["prec.runtime_dtype()", ".c_macro()"],
        forbidden: &["Prim::F32 => \"CHELIS_DTYPE_F32\""],
    },
    Consumer {
        source: ConsumerSource::File("crates/chelis-python/src/lib.rs"),
        role: "Python FFI dtype constant",
        required: &["RuntimeDType::F32.id()"],
        forbidden: &["const CHELIS_DTYPE_F32: i32 = 0"],
    },
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("chelis-cli is two levels below workspace root")
        .to_path_buf()
}

fn read_consumer_source(root: &Path, source: ConsumerSource) -> std::io::Result<String> {
    match source {
        ConsumerSource::File(relative) => fs::read_to_string(root.join(relative)),
        ConsumerSource::RustTree(relative) => {
            let mut files = Vec::new();
            collect_rust_files(&root.join(relative), &mut files)?;
            files.sort();

            let mut combined = String::new();
            for path in files {
                combined.push_str(&fs::read_to_string(path)?);
                combined.push('\n');
            }
            Ok(combined)
        }
    }
}

fn collect_rust_files(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_rust_files(&path, out)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

#[test]
fn added_effect_variant_will_force_every_semantic_consumer_to_decide() {
    let root = repo_root();
    let mut failures = Vec::new();
    for consumer in EFFECT_KIND_CONSUMERS {
        let source_path = consumer.source.path();
        let source = match read_consumer_source(&root, consumer.source) {
            Ok(source) => source,
            Err(err) => {
                failures.push(format!(
                    "{} ({}): missing inventory target: {err}",
                    source_path, consumer.role
                ));
                continue;
            }
        };
        for needle in consumer.required {
            if !source.contains(needle) {
                failures.push(format!(
                    "{} ({}): missing typed marker `{needle}`",
                    source_path, consumer.role
                ));
            }
        }
        for needle in consumer.forbidden {
            if source.contains(needle) {
                failures.push(format!(
                    "{} ({}): raw semantic dispatch remains: `{needle}`",
                    source_path, consumer.role
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
        let source_path = consumer.source.path();
        let source = match read_consumer_source(&root, consumer.source) {
            Ok(source) => source,
            Err(err) => {
                failures.push(format!(
                    "{} ({}): missing inventory target: {err}",
                    source_path, consumer.role
                ));
                continue;
            }
        };
        for needle in consumer.required {
            if !source.contains(needle) {
                failures.push(format!(
                    "{} ({}): missing typed marker `{needle}`",
                    source_path, consumer.role
                ));
            }
        }
        for needle in consumer.forbidden {
            if source.contains(needle) {
                failures.push(format!(
                    "{} ({}): duplicate/raw dtype authority remains: `{needle}`",
                    source_path, consumer.role
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
