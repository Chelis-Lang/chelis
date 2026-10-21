use chelis_shell::SHELL_MAGIC;
use chelis_shell_core::{
    PackageId, SHELL_FORMAT_VERSION, ShellModule, ShellPackage, ShellSymbol, SymbolKind,
};

pub fn sample() -> ShellPackage {
    ShellPackage {
        format_version: SHELL_FORMAT_VERSION,
        package: PackageId {
            name: "demo".into(),
            version: "1.2.3".into(),
        },
        compiler: "=1.2.3-rc.1+build.7".into(),
        modules: vec![ShellModule {
            module: "App.Demo".into(),
            exports: vec![ShellSymbol {
                name: "forward".into(),
                kind: SymbolKind::Value,
                type_repr: Some("(t-fn {} (t-prim {} f32) (t-prim {} f32))".into()),
                type_variable_restrictions: Vec::new(),
                collection_obligations: Vec::new(),
                effects: vec!["z".into(), "a".into(), "z".into()],
                has_body: true,
            }],
        }],
        dependencies: vec![PackageId {
            name: "dep".into(),
            version: String::new(),
        }],
        archive_sha256: "d".repeat(64),
    }
}

pub fn unvalidated(shell: &ShellPackage) -> Vec<u8> {
    let mut bytes = SHELL_MAGIC.to_vec();
    bytes.extend_from_slice(&SHELL_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&bincode::serialize(shell).expect("serialize test metadata"));
    bytes
}

pub fn baseline_bytes() -> Vec<u8> {
    let hex = concat!(
        "4348454c434842000800000008000000040000000000000064656d6f0500000000000000312e322e33",
        "13000000000000003d312e322e332d72632e312b6275696c642e3701000000000000000800000000000000",
        "4170702e44656d6f01000000000000000700000000000000666f727761726400000000012900000000000000",
        "28742d666e207b7d2028742d7072696d207b7d20663332292028742d7072696d207b7d206633322929",
        "00000000000000000000000000000000030000000000000001000000000000007a010000000000000061",
        "01000000000000007a010100000000000000030000000000000064657000000000000000004000000000000000",
        "64646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464",
    );
    hex.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

pub fn unique_offset(bytes: &[u8], field: &[u8]) -> usize {
    let positions: Vec<_> = bytes
        .windows(field.len())
        .enumerate()
        .filter_map(|(index, candidate)| (candidate == field).then_some(index))
        .collect();
    assert_eq!(positions.len(), 1, "test field must occur once");
    positions[0]
}
