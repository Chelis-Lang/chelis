use chelis_runtime_identity::*;

fn main() {
    let input = "runtime/src/lib.rs".to_owned();
    let header = "runtime/include/runtime.h".to_owned();
    let recipe = RuntimeRecipe {
        schema_version: 1, runtime_unit: 0, cargo_profile: ProfileClass::Debug, public_abi: 2,
        required_inputs: vec![
            RequiredInput { logical_path: input.clone(), class: InputClass::Source },
            RequiredInput { logical_path: header.clone(), class: InputClass::Header },
        ],
        units: vec![CompileUnit {
            package: PackageIdentity { name: "captured-runtime".into(), version: "1.0.0".into(), source: "path:runtime".into(), checksum: None },
            target_name: "captured_runtime".into(), kind: UnitKind::Library,
            target: TargetObservation { triple: "x86_64-unknown-linux-gnu".into(), llvm_triple: "x86_64-unknown-linux-gnu".into(), data_layout: "captured x86_64 layout".into(), cpu: "generic".into(), features: vec![], specification: None },
            features: vec![],
            configuration: CompileConfiguration { opt_level: "0".into(), debuginfo: "2".into(), debug_assertions: true, panic: "unwind".into(), rustflags: vec![], cfg: vec![] },
            compiler: CompilerObservation { verbose_version: "captured rustc 1.98.0\nhost: x86_64-unknown-linux-gnu".into(), tools: vec![] },
            inputs: vec![input.clone()], dependencies: vec![], build_script: None, build_environment: vec![], compiler_environment: vec![],
        }],
    };
    let descriptor = derive_descriptor(&recipe, &[
        CapturedInput { logical_path: input, digest: hash_bytes(b"captured source") },
        CapturedInput { logical_path: header, digest: hash_bytes(b"captured header") },
    ]).unwrap();
    let encoded = encode_record(&descriptor, RecordKind::Runtime).unwrap();
    assert_eq!(decode_record(&encoded, RecordKind::Runtime).unwrap(), descriptor);
    println!("{descriptor:?}\n{encoded:?}");
    let error = derive_descriptor(&recipe, &[]).unwrap_err();
    assert!(matches!(error, InputError::MissingInput { .. }));
    println!("{error:?}");
}
