//! Executed compile controls for C1. Mutations use task-local copies, never tracked source.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_PROBE: AtomicUsize = AtomicUsize::new(0);

struct Probe(PathBuf);

impl Probe {
    fn new() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/dtype-contract-probes")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT_PROBE.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(root.parent().unwrap()).unwrap();
        fs::create_dir(&root).unwrap();
        Self(root)
    }

    fn compile(&self, name: &str, source: &str, external: bool) -> Output {
        let input = self.0.join(format!("{name}.rs"));
        fs::write(&input, source).unwrap();
        let mut command = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()));
        command.args(["--edition=2024", "--crate-type=rlib", "--crate-name", name]);
        command.arg(&input).arg("--out-dir").arg(&self.0);
        if external {
            command.arg("--extern").arg(format!(
                "chelis_vocab={}",
                self.0.join("libchelis_vocab.rlib").display()
            ));
        }
        command
            .output()
            .expect("execute rustc for contract control")
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("restore task-local compile probe");
    }
}

fn owner_source() -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs")).unwrap()
}

fn replace_once(source: &str, from: &str, to: &str) -> String {
    assert_eq!(
        source.matches(from).count(),
        1,
        "mutation anchor must be unique: {from}"
    );
    source.replacen(from, to, 1)
}

fn assert_success(output: Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_rejected(output: Output, code: &str, witness: &str) {
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        !output.status.success(),
        "negative control unexpectedly compiled"
    );
    assert!(stderr.contains(code), "wrong rejection: {stderr}");
    assert!(stderr.contains(witness), "missing witness: {stderr}");
}

#[test]
fn external_consumer_can_project_but_cannot_forge_or_rewrite_a_contract() {
    let probe = Probe::new();
    assert_success(probe.compile("chelis_vocab", &owner_source(), false));
    assert_success(probe.compile(
        "positive",
        r#"
        use chelis_vocab::{DTypeContract, RuntimeDType};
        pub const CONTRACT: DTypeContract = RuntimeDType::F32.contract();
        pub fn width() -> usize { CONTRACT.byte_width() }
    "#,
        true,
    ));
    assert_rejected(
        probe.compile(
            "forge",
            r#"
        use chelis_vocab::{DTypeContract, Repr, RuntimeDType};
        pub const FORGED: DTypeContract = DTypeContract {
            dtype: RuntimeDType::F32, repr: Repr::TwosComplement32, arithmetic: None,
        };
    "#,
            true,
        ),
        "E0451",
        "private",
    );
    assert_rejected(
        probe.compile(
            "rewrite",
            r#"
        use chelis_vocab::{DTypeContract, Repr};
        pub fn rewrite(mut value: DTypeContract) { value.repr = Repr::TwosComplement32; }
    "#,
            true,
        ),
        "E0616",
        "repr",
    );
}

#[test]
fn new_dtype_without_contract_registration_does_not_compile() {
    let probe = Probe::new();
    let source = owner_source();
    assert_success(probe.compile("positive", &source, false));
    // Complete the existing naming/decoding surfaces. The only missing match
    // is the new representation registration, not an unrelated old consumer.
    let source = replace_once(
        &source,
        "    I16 = 8,\n",
        "    I16 = 8,\n    UnregisteredDType = 9,\n",
    );
    let source = replace_once(
        &source,
        "Self::I16 => \"int16\",",
        "Self::I16 => \"int16\",\n            Self::UnregisteredDType => \"unregistered\",",
    );
    let source = replace_once(
        &source,
        "Self::I16 => \"CHELIS_DTYPE_I16\",",
        "Self::I16 => \"CHELIS_DTYPE_I16\",\n            Self::UnregisteredDType => \"CHELIS_DTYPE_UNREGISTERED\",",
    );
    let source = replace_once(
        &source,
        "8 => Ok(Self::I16),",
        "8 => Ok(Self::I16),\n            9 => Ok(Self::UnregisteredDType),",
    );
    assert_rejected(
        probe.compile("negative", &source, false),
        "E0004",
        "UnregisteredDType",
    );
}

#[test]
fn new_stored_or_arithmetic_representation_needs_an_exhaustive_projection() {
    let probe = Probe::new();
    let source = owner_source();
    assert_success(probe.compile("positive", &source, false));
    for owner in ["Repr", "ArithmeticRepr"] {
        let mutated = replace_once(
            &source,
            &format!("pub enum {owner} {{"),
            &format!("pub enum {owner} {{\n    UnregisteredRepresentation,"),
        );
        assert_rejected(
            probe.compile("negative", &mutated, false),
            "E0004",
            "UnregisteredRepresentation",
        );
    }
}
