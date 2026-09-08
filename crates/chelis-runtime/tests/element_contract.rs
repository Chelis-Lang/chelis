//! C1 storage registration: [04-NUM-4]/[04-NUM-8], not the Phase 3 access seal.

use chelis_runtime::{
    chelis_alloc, chelis_tensor_release, Bf16Bits, Bool8, F16Bits, RuntimeDType, TensorElement,
};
use chelis_vocab::{ArithmeticRepr, Repr};
use std::any::TypeId;

fn row<T: TensorElement + 'static, A: 'static>(dtype: RuntimeDType, repr: Repr)
where
    T::Arithmetic: 'static,
{
    assert_eq!(T::DTYPE, dtype);
    assert_eq!(T::REPR, repr);
    assert_eq!(T::REPR, dtype.contract().repr());
    assert_eq!(size_of::<T>(), dtype.contract().byte_width());
    assert_eq!(TypeId::of::<T::Arithmetic>(), TypeId::of::<A>());
}

#[test]
fn all_nine_storage_and_arithmetic_types_match_the_spec() {
    row::<f64, f64>(RuntimeDType::F64, Repr::Ieee754Binary64);
    row::<f32, f32>(RuntimeDType::F32, Repr::Ieee754Binary32);
    row::<F16Bits, f32>(RuntimeDType::F16, Repr::Ieee754Binary16);
    row::<Bf16Bits, f32>(RuntimeDType::Bf16, Repr::Bfloat16);
    row::<i64, i64>(RuntimeDType::I64, Repr::TwosComplement64);
    row::<i32, i32>(RuntimeDType::I32, Repr::TwosComplement32);
    row::<i16, i16>(RuntimeDType::I16, Repr::TwosComplement16);
    row::<i8, i8>(RuntimeDType::I8, Repr::TwosComplement8);
    row::<Bool8, ()>(RuntimeDType::Bool, Repr::Bool8);
    assert_eq!(RuntimeDType::ALL.len(), 9);
    assert_eq!(RuntimeDType::Bool.contract().arithmetic(), None);
    for dtype in [RuntimeDType::F16, RuntimeDType::Bf16] {
        assert_eq!(
            dtype.contract().arithmetic(),
            Some(ArithmeticRepr::Ieee754Binary32)
        );
    }
}

#[test]
fn narrow_markers_preserve_every_bit_pattern_without_sharing_a_type() {
    assert_ne!(TypeId::of::<F16Bits>(), TypeId::of::<Bf16Bits>());
    assert_ne!(TypeId::of::<F16Bits>(), TypeId::of::<u16>());
    assert_ne!(TypeId::of::<Bf16Bits>(), TypeId::of::<u16>());
    assert_eq!(align_of::<F16Bits>(), align_of::<u16>());
    assert_eq!(align_of::<Bf16Bits>(), align_of::<u16>());
    for bits in 0..=u16::MAX {
        assert_eq!(F16Bits::from_bits(bits).to_bits(), bits);
        assert_eq!(Bf16Bits::from_bits(bits).to_bits(), bits);
    }
}

#[test]
fn bool_storage_admits_only_canonical_construction() {
    for byte in 0..=u8::MAX {
        match Bool8::from_u8(byte) {
            Some(value) => {
                assert!(byte <= 1);
                assert_eq!(value.to_byte(), byte);
                assert_eq!(value.get(), byte == 1);
            }
            None => assert!(byte > 1),
        }
    }
    assert_ne!(TypeId::of::<Bool8>(), TypeId::of::<bool>());
    assert_ne!(TypeId::of::<Bool8>(), TypeId::of::<u8>());
    assert_ne!(TypeId::of::<Bool8>(), TypeId::of::<i8>());
}

fn checked_access<T: TensorElement + std::fmt::Debug + PartialEq, Wrong: TensorElement>(value: T) {
    assert_eq!(size_of::<T>(), size_of::<Wrong>());
    assert_ne!(T::REPR, Wrong::REPR);
    unsafe {
        let tensor = chelis_alloc(1, [1i64].as_ptr(), T::DTYPE.id() as u8);
        assert!(!tensor.is_null());
        let ptr = T::data_ptr(tensor).expect("matching element registration");
        ptr.write(value);
        assert_eq!(ptr.read(), value);
        let mismatch =
            Wrong::data_ptr(tensor).expect_err("same width is not matching representation");
        assert_eq!(mismatch.expected, Wrong::DTYPE);
        assert_eq!(mismatch.actual, T::DTYPE);
        chelis_tensor_release(tensor);
    }
}

#[test]
fn checked_access_rejects_equal_width_representation_swaps() {
    checked_access::<f64, i64>(1.25);
    checked_access::<i64, f64>(i64::MAX);
    checked_access::<f32, i32>(1.25);
    checked_access::<i32, f32>(i32::MAX);
    checked_access::<F16Bits, Bf16Bits>(F16Bits::from_bits(0x3555));
    checked_access::<Bf16Bits, F16Bits>(Bf16Bits::from_bits(0x3f81));
    checked_access::<i16, F16Bits>(i16::MAX);
    checked_access::<i8, Bool8>(i8::MAX);
    checked_access::<Bool8, i8>(Bool8::TRUE);
}

// The isolated Cargo fixture compiles the real registration and verbatim trait
// declaration. Only the unchanged raw-pointer environment is stubbed; no
// pointer operation is executed or claimed validated by this compile fixture.
struct CompileProbe(std::path::PathBuf);

impl CompileProbe {
    fn new() -> Self {
        let repo = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let root = repo
            .join("target/element-contract-probes")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(root.parent().unwrap()).unwrap();
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(root.join("src")).unwrap();
        let manifest = format!(
            "[package]\nname = \"element-contract-probe\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\
             [workspace]\n[dependencies]\nchelis-runtime = {{ path = {:?} }}\nchelis-vocab = {{ path = {:?} }}\n",
            repo.join("crates/chelis-runtime"), repo.join("crates/chelis-vocab"),
        );
        std::fs::write(root.join("Cargo.toml"), manifest).unwrap();
        std::fs::copy(repo.join("Cargo.lock"), root.join("Cargo.lock")).unwrap();
        Self(root)
    }

    fn check(&self, main: &str, library: &str, owner: &str) -> std::process::Output {
        std::fs::write(self.0.join("src/main.rs"), main).unwrap();
        std::fs::write(self.0.join("src/lib.rs"), library).unwrap();
        std::fs::write(self.0.join("src/element.rs"), owner).unwrap();
        std::process::Command::new("cargo")
            .args(["check", "--quiet", "--offline", "--manifest-path"])
            .arg(self.0.join("Cargo.toml"))
            .env("CARGO_TARGET_DIR", self.0.join("target"))
            .env("CARGO_HUSKY_DONT_INSTALL_HOOKS", "1")
            .output()
            .expect("execute isolated element compile control")
    }
}

impl Drop for CompileProbe {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).expect("remove owned compile fixture");
    }
}

fn rejected(output: std::process::Output, code: &str, reason: &str) {
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        !output.status.success(),
        "negative control compiled: {reason}"
    );
    assert!(
        error.contains(code) && error.contains(reason),
        "wrong rejection: {error}"
    );
}

fn compiled(output: std::process::Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn replace_once(source: &str, from: &str, to: &str) -> String {
    assert_eq!(
        source.matches(from).count(),
        1,
        "non-unique mutation anchor: {from}"
    );
    source.replacen(from, to, 1)
}

#[test]
fn executed_compile_controls_seal_the_owner_and_check_each_registration() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = std::fs::read_to_string(root.join("src/lib.rs")).unwrap();
    let start = source
        .find("pub trait TensorElement:")
        .expect("real public trait");
    let end = start + source[start..].find("\n}\n").expect("trait closing brace") + 3;
    let trait_source = &source[start..end];
    let owner = std::fs::read_to_string(root.join("src/element.rs")).unwrap();
    let library = format!(
        "#![allow(dead_code, non_camel_case_types)]\n\
         use chelis_vocab::RuntimeDType;\n\
         use chelis_runtime::{{Bool8, DtypeMismatch}};\n\
         mod element;\npub struct chelis_tensor;\n\
         impl chelis_tensor {{ fn count(&self) -> usize {{ unimplemented!() }} }}\n\
         unsafe fn tensor_dtype(_: *const chelis_tensor, _: &str) -> RuntimeDType {{ unimplemented!() }}\n\
         unsafe fn tensor_data(_: *const chelis_tensor) -> *mut u8 {{ unimplemented!() }}\n\
         {trait_source}"
    );
    // The dependency's actual narrow-float types, not test substitutes.
    let owner = replace_once(
        &owner,
        "pub use half::{bf16 as Bf16Bits, f16 as F16Bits};",
        "pub use chelis_runtime::{Bf16Bits, F16Bits};",
    );
    let probe = CompileProbe::new();
    compiled(probe.check(
        "use chelis_runtime::{TensorElement, F16Bits, Bf16Bits, Bool8};\n\
         fn element<T: TensorElement>() {}\n\
         fn main() { element::<f64>(); element::<f32>(); element::<F16Bits>();\n\
         element::<Bf16Bits>(); element::<i64>(); element::<i32>(); element::<i16>();\n\
         element::<i8>(); element::<Bool8>(); }",
        &library,
        &owner,
    ));

    for ty in ["bool", "u8", "u16", "u32", "()"] {
        rejected(probe.check(&format!(
            "fn element<T: chelis_runtime::TensorElement>() {{}}\nfn main() {{ element::<{ty}>(); }}"
        ), &library, &owner), "E0277", "ElementStorage");
    }
    let rogue = "#[derive(Copy, Clone)] struct Rogue(f32);\n\
        impl TensorElement for Rogue { const DTYPE: RuntimeDType = RuntimeDType::F32;\n\
        const REPR: chelis_vocab::Repr = chelis_vocab::Repr::Ieee754Binary32; type Arithmetic = f32; }";
    rejected(
        probe.check(
            &format!(
                "use chelis_runtime::{{TensorElement, RuntimeDType}};\n{rogue}\nfn main() {{}}"
            ),
            &library,
            &owner,
        ),
        "E0277",
        "ElementStorage",
    );
    rejected(
        probe.check("fn main() {}", &format!("{library}\n{rogue}"), &owner),
        "E0277",
        "ElementStorage",
    );
    let forged_storage = "#[derive(Copy, Clone)] struct Unregistered(f32);\n\
        impl ElementStorage for Unregistered {\n\
        const STORAGE_DTYPE: RuntimeDType = RuntimeDType::F32;\n\
        const STORED_REPR: Repr = Repr::Ieee754Binary32; type ArithmeticStorage = f32; }";
    rejected(
        probe.check(
            "fn main() {}",
            &library,
            &format!("{owner}\n{forged_storage}"),
        ),
        "E0277",
        "Sealed",
    );
    rejected(
        probe.check(
            "fn main() {}",
            &format!(
        "{library}\nuse element::ElementStorage;\nuse chelis_vocab::Repr;\n{forged_storage}"
    ),
            &owner,
        ),
        "E0277",
        "Sealed",
    );

    let wrong_repr = replace_once(
        &owner,
        "const STORED_REPR: Repr = Repr::Ieee754Binary32;",
        "const STORED_REPR: Repr = Repr::TwosComplement32;",
    );
    rejected(
        probe.check("fn main() {}", &library, &wrong_repr),
        "E0080",
        "stored representation must match dtype contract",
    );
    let wrong_size = replace_once(
        &owner,
        "impl private::Sealed for f32 {}",
        "impl private::Sealed for u8 {}",
    );
    let wrong_size = replace_once(
        &wrong_size,
        "impl ElementStorage for f32 {",
        "impl ElementStorage for u8 {",
    );
    let wrong_size = replace_once(
        &wrong_size,
        "RuntimeDType::F32 => assert_registration::<f32>(dtype),",
        "RuntimeDType::F32 => assert_registration::<u8>(dtype),",
    );
    rejected(
        probe.check("fn main() {}", &library, &wrong_size),
        "E0080",
        "storage size must match dtype contract",
    );
    let wrong_arithmetic = replace_once(&owner,
        "const STORAGE_DTYPE: RuntimeDType = RuntimeDType::F16;\n    const STORED_REPR: Repr = Repr::Ieee754Binary16;\n    type ArithmeticStorage = f32;",
        "const STORAGE_DTYPE: RuntimeDType = RuntimeDType::F16;\n    const STORED_REPR: Repr = Repr::Ieee754Binary16;\n    type ArithmeticStorage = f64;");
    rejected(
        probe.check("fn main() {}", &library, &wrong_arithmetic),
        "E0080",
        "arithmetic representation must match dtype contract",
    );
    let missing = replace_once(
        &owner,
        "RuntimeDType::F16 => assert_registration::<F16Bits>(dtype),",
        "",
    );
    rejected(
        probe.check("fn main() {}", &library, &missing),
        "E0004",
        "F16",
    );
    compiled(probe.check("fn main() {}", &library, &owner));
}
