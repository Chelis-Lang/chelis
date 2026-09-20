// Regression for chelis#1271: two packages in one dependency graph each
// declare a type whose constructor carries the same *unqualified* name.
// The reef linker gives each one a distinct
// `Pkg__<pkg>__<Module>__<Name>` identity (spec/04-type-system.md,
// "Module identity"), so the two are different types and both must keep
// their own field list, field order, field dtypes, and arity.
//
// Host lowering resolved a constructor reference by comparing only the
// TERMINAL segment after the last `__`, then took the first match out of
// a table sorted by ADT name. Both mangled names end in the same
// terminal, so the alphabetically-earlier package always won and the
// other package's constructor was lowered against the wrong declaration.
// `chelis check` scored 1.0 and `chelis eval` was right; only `chelis
// build` was wrong.
//
// The defect has six observable faces across the lowering paths that
// consume a resolved declaration. Each test below drives the issue's
// own three-package reproducer twice over the SAME `collidelib` and
// `collideapp` sources - once with the colliding package in the graph
// and once without it - because the invariant the issue states is
// exactly that: "an unqualified-name collision between two packages in
// one graph is legal and must not change which record a constructor
// resolves to". A face is fixed when both runs succeed and agree.
//
// Face                 | pre-fix symptom with the colliding dep present
// ---------------------|-----------------------------------------------
// record construction  | rejected: "has no field `amount`" naming the
//                      | constructor that *does* have that field
// record field order   | fields emitted in the other package's order
// match destructuring  | binds the other package's field index
// positional call      | field widened to the other package's dtype
// nullary reference    | lowered to a bare C identifier, not a
//                      | construction, so the emitted C does not compile
// field access         | rejected: "absent or ambiguous on the resolved
//                      | ADT type"
//
// A last test covers the IN-PACKAGE path instead: two modules of ONE
// package, reached through the module-qualified references chelis#316
// introduced. Their mangled names share a terminal exactly as two
// packages' do, so the same defect reached lowering by a second route.

mod common;

use assert_cmd::Command;
use chelis_compiler_api::COMPILER_VERSION;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::{TempDir, tempdir};

/// One published library package: its `reef.toml` identity plus the body
/// of its single module.
struct Package<'a> {
    name: &'a str,
    prefix: &'a str,
    exports: &'a str,
    body: &'a str,
}

/// The outcome of `chelis build --target c` on the app package: the
/// emitted C source when it succeeded, plus the output directory and the
/// command transcript so a rejection reports its own diagnostic instead
/// of a bare `None`.
struct Built {
    source: Option<String>,
    out_dir: PathBuf,
    unit: String,
    transcript: String,
}

impl Built {
    fn source(&self) -> &str {
        self.source.as_deref().unwrap_or_else(|| {
            panic!(
                "expected `chelis build` to emit C; it rejected:\n{}",
                self.transcript
            )
        })
    }

    /// Compile the emitted translation unit with the host toolchain,
    /// returning `None` when no host compiler is available.
    fn syntax_check(&self) -> Option<std::process::Output> {
        let compiler = chelis_backend_c::toolchain::c_compiler();
        if !host_compiler_available(&compiler) {
            return None;
        }
        std::process::Command::new(&compiler)
            .arg("-fsyntax-only")
            .arg("-I")
            .arg(&self.out_dir)
            .arg(self.out_dir.join(&self.unit))
            .output()
            .ok()
    }

    /// Link the emitted unit against a driver that calls `entry_symbol`
    /// and prints its `f32` result, run it, and return stdout. `None`
    /// when no host compiler is available.
    fn compile_and_run(&self, entry_symbol: &str) -> Option<String> {
        let source = self.source();
        let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
            chelis_backend_c::toolchain::CodegenRequirements {
                wants_openmp: true,
                needs_blas: source.contains("cblas_sgemm(") || source.contains("\"chelis_blas.h\""),
            },
        );
        if !host_compiler_available(&toolchain.compiler) {
            return None;
        }
        // [05-OBS-11] makes this generated unit independently executable.
        // This probe supplies its own driver to call the public entry symbol,
        // so rename only the generated observation entry point and retain all
        // authored exports exactly as published by the header.
        if source.contains("int main(void)") {
            fs::write(
                self.out_dir.join(&self.unit),
                source.replace("int main(void)", "int chelis_manifest_main(void)"),
            )
            .expect("rename generated observation driver for the ABI probe");
        }
        let unit_header = self.unit.replace(".c", ".h");
        write_file(
            &self.out_dir.join("driver.c"),
            &format!(
                "#include <stdio.h>\n\
                 #include \"chelis_runtime.h\"\n\
                 #include \"{unit_header}\"\n\
                 int main(void) {{\n\
                 \x20   printf(\"%.6f\\n\", (double){entry_symbol}());\n\
                 \x20   return 0;\n\
                 }}\n"
            ),
        );
        let mut command = std::process::Command::new(&toolchain.compiler);
        command.current_dir(&self.out_dir);
        command.arg("-O2");
        command.args(&toolchain.compile_flags);
        command.arg(&self.unit);
        command.arg("driver.c");
        command.args(["-L.", "-lchelis_runtime"]);
        command.args(&toolchain.link_flags);
        command.args(["-o", "prog"]);
        let compiled = command.output().expect("host compiler should run");
        assert!(
            compiled.status.success(),
            "the emitted translation unit must compile and link; the host \
             compiler said:\n{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let run = std::process::Command::new(self.out_dir.join("prog"))
            .output()
            .expect("the linked program should run");
        assert!(
            run.status.success(),
            "the linked program must exit 0; it said:\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        Some(String::from_utf8_lossy(&run.stdout).trim().to_string())
    }
}

fn host_compiler_available(compiler: &str) -> bool {
    std::process::Command::new(compiler)
        .arg("--version")
        .output()
        .is_ok_and(|probe| probe.status.success())
}

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

fn write_package(root: &Path, package: &Package<'_>, dependencies: &[&str]) -> PathBuf {
    let dependency_lines = dependencies
        .iter()
        .map(|name| format!("{name} = {{ version = \"0.1.0\" }}\n"))
        .collect::<String>();
    let package_root = root.join(package.name);
    write_file(
        &package_root.join("reef.toml"),
        &format!(
            "[package]\n\
             name = \"{name}\"\n\
             version = \"0.1.0\"\n\
             compiler = \"={COMPILER_VERSION}\"\n\
             module_prefix = \"{prefix}\"\n\
             \n\
             [dependencies]\n\
             {dependency_lines}",
            name = package.name,
            prefix = package.prefix,
        ),
    );
    write_file(
        &package_root.join("src/m.ch"),
        &format!(
            "module {prefix}.M\nexport {exports}\n{body}",
            prefix = package.prefix,
            exports = package.exports,
            body = package.body,
        ),
    );
    package_root
}

/// Publish `library` (optionally on top of `collider`) into a private
/// reef home, then `chelis build` an app that imports the library.
///
/// `collider` is the package whose constructor collides on its terminal
/// segment. It is never imported by the app: like the issue's
/// `collidedep`, it reaches the graph only as the library's dependency.
fn build_app(
    collider: Option<&Package<'_>>,
    library: &Package<'_>,
    app_body: &str,
) -> (TempDir, Built) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    let reef_home = root.join("reef-home");
    fs::create_dir_all(&reef_home).expect("mkdir reef home");

    let mut library_dependencies: Vec<&str> = Vec::new();
    if let Some(collider) = collider {
        let collider_root = write_package(root, collider, &[]);
        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_REEF_HOME", &reef_home)
            .args(["reef", "publish", collider_root.to_str().unwrap()])
            .assert()
            .success();
        library_dependencies.push(collider.name);
    }
    let library_root = write_package(root, library, &library_dependencies);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", library_root.to_str().unwrap()])
        .assert()
        .success();

    let app_root = write_package(
        root,
        &Package {
            name: "collideapp",
            prefix: "Collideapp",
            exports: "(main)",
            body: app_body,
        },
        &[library.name],
    );
    let entry = app_root.join("src/m.ch");
    let out = root.join("out");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_root)
        .args([
            "build",
            entry.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run chelis build");
    let transcript = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let built = Built {
        source: fs::read_to_string(out.join("m.c")).ok(),
        out_dir: out,
        unit: "m.c".to_string(),
        transcript,
    };
    (dir, built)
}

/// The body of one emitted C function, from its opening brace to the
/// matching close. Panics when the function is absent, so a renamed or
/// dropped definition fails loudly instead of vacuously passing.
fn function_body<'a>(source: &'a str, signature: &str) -> &'a str {
    // The public symbol is now the [04-LIN-7] borrowing adapter. Constructor
    // layout and projection are emitted in the consuming implementation body.
    let (prefix, _params) = signature
        .split_once('(')
        .expect("test signature contains parameter list");
    let (_return_type, name) = prefix
        .rsplit_once(' ')
        .expect("test signature contains a return type and function name");
    // chelis#1820: located by NAME, not by the full signature. chelis#1799
    // added a `chelis_rng_state` parameter to every host body, and the old
    // full-signature needle then missed the definition and failed before this
    // row read anything. The parameter list is not what the row asserts.
    let name = format!("{}__chelis_owned_body", common::authored_c_symbol(name));
    let rest = common::host_body_definition(source, &name);
    let end = rest
        .find("\n}\n")
        .unwrap_or_else(|| panic!("`{name}` definition is unterminated:\n{rest}"));
    &rest[..end]
}

/// `chelis check` must keep scoring the app 1.0 with no errors in both
/// graphs. The issue's whole shape is that the checked program is fine
/// and lowering disagrees with it, so a "fix" that made the checker
/// reject would not be a fix.
fn assert_app_checks_clean(collider: Option<&Package<'_>>, library: &Package<'_>, app_body: &str) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    let reef_home = root.join("reef-home");
    fs::create_dir_all(&reef_home).expect("mkdir reef home");
    let mut library_dependencies: Vec<&str> = Vec::new();
    if let Some(collider) = collider {
        let collider_root = write_package(root, collider, &[]);
        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_REEF_HOME", &reef_home)
            .args(["reef", "publish", collider_root.to_str().unwrap()])
            .assert()
            .success();
        library_dependencies.push(collider.name);
    }
    let library_root = write_package(root, library, &library_dependencies);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", library_root.to_str().unwrap()])
        .assert()
        .success();
    let app_root = write_package(
        root,
        &Package {
            name: "collideapp",
            prefix: "Collideapp",
            exports: "(main)",
            body: app_body,
        },
        &[library.name],
    );
    let entry = app_root.join("src/m.ch");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_root)
        .args(["check", entry.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("check output must be json");
    assert_eq!(json["score"], 1.0, "app must check clean: {json}");
    assert!(
        json["errors"].as_array().expect("errors array").is_empty(),
        "app must check with no errors: {json}"
    );
}

// ── Face 1: record construction (the issue's headline) ───────────────

const RECORD_COLLIDER: Package<'static> = Package {
    name: "collidedep",
    prefix: "Collidedep",
    exports: "(Wrapped, wrap)",
    body: "type Wrapped =\n  | Wrapped { inner: f32 }\ndef wrap(x: f32) -> Wrapped = Wrapped { inner: x }\n",
};

const RECORD_LIBRARY: Package<'static> = Package {
    name: "collidelib",
    prefix: "Collidelib",
    exports: "(Wrapped, make, value)",
    body: "type Wrapped =\n  | Wrapped { amount: f32, tag: f32 }\n\
           def make(x: f32) -> Wrapped = Wrapped { amount: x, tag: 1.0 }\n\
           def value(w: Wrapped) -> f32 =\n  match w with {\n    | Wrapped { amount, tag: _ } => amount\n  }\n",
};

const RECORD_APP: &str =
    "import Collidelib.M (make, value)\ndef main() -> f32 = value(make(5.0))\n";

#[test]
fn record_construction_resolves_the_authored_packages_constructor() {
    // The reported case, verbatim from the issue. Pre-fix this rejected
    // with "record constructor `Pkg__collidelib__Collidelib__M__Wrapped`
    // has no field `amount`" - a message that names the right
    // constructor while validating against the other package's field
    // list.
    assert_app_checks_clean(Some(&RECORD_COLLIDER), &RECORD_LIBRARY, RECORD_APP);
    let (_solo_dir, solo) = build_app(None, &RECORD_LIBRARY, RECORD_APP);
    let (_dir, built) = build_app(Some(&RECORD_COLLIDER), &RECORD_LIBRARY, RECORD_APP);
    let signature = "chelis_adt* pkg__collidelib__Collidelib__M__make(float x)";
    assert_eq!(
        function_body(built.source(), signature),
        function_body(solo.source(), signature),
        "a colliding package elsewhere in the graph must not change how \
         `Collidelib.M.Wrapped` is constructed"
    );
}

#[test]
fn record_construction_keeps_the_authored_packages_field_order() {
    // Same collision with the field *sets* equal and the declaration
    // order swapped, so the wrong resolution cannot be caught by a
    // missing-field check. Pre-fix, `make` stored `amount` at index 0
    // (the collider's order) while the value's tag said the library's
    // type - a silently mislaid-out record, not a rejection.
    const COLLIDER: Package<'static> = Package {
        name: "collidedep",
        prefix: "Collidedep",
        exports: "(Wrapped, wrap)",
        body: "type Wrapped =\n  | Wrapped { amount: f32, extra: f32 }\n\
               def wrap(x: f32) -> Wrapped = Wrapped { amount: x, extra: x }\n",
    };
    const LIBRARY: Package<'static> = Package {
        name: "collidelib",
        prefix: "Collidelib",
        exports: "(Wrapped, make, value)",
        body: "type Wrapped =\n  | Wrapped { extra: f32, amount: f32 }\n\
               def make(e: f32, a: f32) -> Wrapped = Wrapped { extra: e, amount: a }\n\
               def value(w: Wrapped) -> f32 =\n  match w with {\n    | Wrapped { extra: _, amount } => amount\n  }\n",
    };
    const APP: &str =
        "import Collidelib.M (make, value)\ndef main() -> f32 = value(make(1.0, 2.0))\n";

    let (_solo_dir, solo) = build_app(None, &LIBRARY, APP);
    let (_dir, built) = build_app(Some(&COLLIDER), &LIBRARY, APP);
    let signature = "chelis_adt* pkg__collidelib__Collidelib__M__make(float e, float a)";
    let body = function_body(built.source(), signature);
    assert_eq!(
        body,
        function_body(solo.source(), signature),
        "field order must follow the authored declaration, not the collider's"
    );
    let extra_first = body.find("= e;").expect("emitted C stores `e`");
    let amount_second = body.find("= a;").expect("emitted C stores `a`");
    assert!(
        extra_first < amount_second,
        "`extra` is field 0 and `amount` is field 1 in the authored \
         declaration; emitted body was:\n{body}"
    );
}

#[test]
fn match_destructuring_binds_the_authored_packages_field_index() {
    // The issue's own `value` function is a `match`, so a fix confined
    // to the record-construction path would leave the collision live
    // here. Pre-fix the arm read field index 0 (the collider's position
    // for `amount`) instead of index 1.
    const COLLIDER: Package<'static> = Package {
        name: "collidedep",
        prefix: "Collidedep",
        exports: "(Wrapped, wrap)",
        body: "type Wrapped =\n  | Wrapped { amount: f32, extra: f32 }\n\
               def wrap(x: f32) -> Wrapped = Wrapped { amount: x, extra: x }\n",
    };
    const LIBRARY: Package<'static> = Package {
        name: "collidelib",
        prefix: "Collidelib",
        exports: "(Wrapped, make, value)",
        body: "type Wrapped =\n  | Wrapped { extra: f32, amount: f32 }\n\
               def make(e: f32, a: f32) -> Wrapped = Wrapped { extra: e, amount: a }\n\
               def value(w: Wrapped) -> f32 =\n  match w with {\n    | Wrapped { extra: _, amount } => amount\n  }\n",
    };
    const APP: &str =
        "import Collidelib.M (make, value)\ndef main() -> f32 = value(make(1.0, 2.0))\n";

    let (_solo_dir, solo) = build_app(None, &LIBRARY, APP);
    let (_dir, built) = build_app(Some(&COLLIDER), &LIBRARY, APP);
    let signature = "float pkg__collidelib__Collidelib__M__value(chelis_adt* w)";
    let body = function_body(built.source(), signature);
    assert_eq!(
        body,
        function_body(solo.source(), signature),
        "a match arm must bind the authored declaration's field index"
    );
    assert!(
        body.contains("chelis_adt_get_field(__adt_0, 1)"),
        "`amount` is field 1 in the authored declaration; emitted body was:\n{body}"
    );
}

#[test]
fn positional_construction_keeps_the_authored_packages_field_dtype() {
    // Positional (non-record) construction takes a different lowering
    // path and forces each argument to the resolved declaration's field
    // type. Pre-fix, the library's `f32` payload was silently widened to
    // the collider's `f64` - a dtype change caused by an unrelated
    // package being in the graph.
    const COLLIDER: Package<'static> = Package {
        name: "collidedep",
        prefix: "Collidedep",
        exports: "(Boxed, dep_box)",
        body: "type Boxed =\n  | Boxed(f64)\ndef dep_box(x: f64) -> Boxed = Boxed(x)\n",
    };
    const LIBRARY: Package<'static> = Package {
        name: "collidelib",
        prefix: "Collidelib",
        exports: "(Boxed, make, value)",
        body: "type Boxed =\n  | Boxed(f32)\ndef make(x: f32) -> Boxed = Boxed(x)\n\
               def value(b: Boxed) -> f32 =\n  match b with {\n    | Boxed(v) => v\n  }\n",
    };
    const APP: &str = "import Collidelib.M (make, value)\ndef main() -> f32 = value(make(5.0))\n";

    let (_solo_dir, solo) = build_app(None, &LIBRARY, APP);
    let (_dir, built) = build_app(Some(&COLLIDER), &LIBRARY, APP);
    let signature = "chelis_adt* pkg__collidelib__Collidelib__M__make(float x)";
    let body = function_body(built.source(), signature);
    assert_eq!(
        body,
        function_body(solo.source(), signature),
        "the payload dtype must follow the authored declaration, not the collider's"
    );
    assert!(
        body.contains("chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_F32")
            && !body.contains("CHELIS_DTYPE_F64"),
        "the authored declaration stores f32; emitted body was:\n{body}"
    );
}

#[test]
fn nullary_constructor_reference_still_lowers_to_a_construction() {
    // A bare constructor reference only becomes a construction when the
    // resolved declaration is nullary. Pre-fix the reference resolved to
    // the collider's non-nullary `On`, the nullary branch was skipped,
    // and lowering emitted a bare C identifier that nothing declares -
    // the emitted translation unit does not compile.
    const COLLIDER: Package<'static> = Package {
        name: "collidedep",
        prefix: "Collidedep",
        exports: "(Flag, On, dep_on)",
        body: "type Flag =\n  | On { weight: f32 }\ndef dep_on(w: f32) -> Flag = On { weight: w }\n",
    };
    const LIBRARY: Package<'static> = Package {
        name: "collidelib",
        prefix: "Collidelib",
        exports: "(Flag, On, pick, value)",
        body: "type Flag =\n  | On\ndef pick() -> Flag = On\n\
               def value(f: Flag) -> f32 =\n  match f with {\n    | On => 1.0\n  }\n",
    };
    const APP: &str = "import Collidelib.M (pick, value)\ndef main() -> f32 = value(pick())\n";

    let (_solo_dir, solo) = build_app(None, &LIBRARY, APP);
    let (_dir, built) = build_app(Some(&COLLIDER), &LIBRARY, APP);
    let signature = "chelis_adt* pkg__collidelib__Collidelib__M__pick()";
    let body = function_body(built.source(), signature);
    assert_eq!(
        body,
        function_body(solo.source(), signature),
        "a nullary constructor reference must lower to a construction in both graphs"
    );
    assert!(
        body.contains("chelis_adt_construct("),
        "the authored `On` is nullary and must be constructed, not named; \
         emitted body was:\n{body}"
    );
    // The direct oracle for this face: the artifact `chelis build` hands
    // the user has to compile. Pre-fix it named an identifier nothing
    // declares. Skipped, not failed, when the host has no C compiler.
    if let Some(compiled) = built.syntax_check() {
        assert!(
            compiled.status.success(),
            "the emitted translation unit must compile; the host compiler said:\n{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
    }
}

#[test]
fn field_access_resolves_the_authored_packages_declaration() {
    // Field access resolves through the ADT *type* name rather than the
    // constructor name, and had the same terminal-only comparison. Pre-
    // fix, both packages' declarations were admitted as candidates, they
    // disagreed on the index of `amount`, and lowering rejected a valid
    // program with "field `amount` is absent or ambiguous on the
    // resolved ADT type".
    const COLLIDER: Package<'static> = Package {
        name: "collidedep",
        prefix: "Collidedep",
        exports: "(Wrapped, wrap)",
        body: "type Wrapped =\n  | Wrapped { amount: f32, extra: f32 }\n\
               def wrap(x: f32) -> Wrapped = Wrapped { amount: x, extra: x }\n",
    };
    const LIBRARY: Package<'static> = Package {
        name: "collidelib",
        prefix: "Collidelib",
        exports: "(Wrapped, make, value)",
        body: "type Wrapped =\n  | Wrapped { extra: f32, amount: f32 }\n\
               def make(e: f32, a: f32) -> Wrapped = Wrapped { extra: e, amount: a }\n\
               def value(w: Wrapped) -> f32 = w.amount\n",
    };
    const APP: &str =
        "import Collidelib.M (make, value)\ndef main() -> f32 = value(make(1.0, 2.0))\n";

    let (_solo_dir, solo) = build_app(None, &LIBRARY, APP);
    let (_dir, built) = build_app(Some(&COLLIDER), &LIBRARY, APP);
    let signature = "float pkg__collidelib__Collidelib__M__value(chelis_adt* w)";
    let body = function_body(built.source(), signature);
    assert_eq!(
        body,
        function_body(solo.source(), signature),
        "field access must resolve against the authored declaration"
    );
    assert!(
        body.contains("chelis_adt_get_field(__adt_base_0, 1)"),
        "`amount` is field 1 in the authored declaration; emitted body was:\n{body}"
    );
}

// ── The in-package path (the chelis#316 shape) ───────────────────────

/// Write one package whose modules share a `src/` and build `entry` to C.
/// No dependencies and no publish step: the reef linker still mangles each
/// module's declarations to `Pkg__demo__Demo__<Module>__<Name>`, but this
/// reaches lowering through the in-package path rather than the
/// cross-package one every test above uses.
fn build_single_package(modules: &[(&str, &str)], entry: &str) -> (TempDir, PathBuf, Built) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    let reef_home = root.join("reef-home");
    fs::create_dir_all(&reef_home).expect("mkdir reef home");
    let package_root = root.join("demo");
    write_file(
        &package_root.join("reef.toml"),
        &format!(
            "[package]\n\
             name = \"demo\"\n\
             version = \"0.1.0\"\n\
             compiler = \"={COMPILER_VERSION}\"\n\
             module_prefix = \"Demo\"\n\
             \n\
             [dependencies]\n"
        ),
    );
    for (file, body) in modules {
        write_file(&package_root.join("src").join(file), body);
    }
    let entry_path = package_root.join("src").join(entry);
    let out = package_root.join("out");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&package_root)
        .args([
            "build",
            entry_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run chelis build");
    let transcript = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let unit = entry.replace(".ch", ".c");
    let built = Built {
        source: fs::read_to_string(out.join(&unit)).ok(),
        out_dir: out,
        unit,
        transcript,
    };
    (dir, package_root, built)
}

const ALPHA: &str = "module Demo.Alpha\n\
     export (Wrapped, make, value)\n\
     type Wrapped =\n  | Wrapped { amount: f32, extra: f32 }\n\
     def make(a: f32, e: f32) -> Wrapped = Wrapped { amount: a, extra: e }\n\
     def value(w: Wrapped) -> f32 = w.amount\n";

const BETA: &str = "module Demo.Beta\n\
     export (Wrapped, make, value)\n\
     type Wrapped =\n  | Wrapped { extra: f32, amount: f32 }\n\
     def make(e: f32, a: f32) -> Wrapped = Wrapped { extra: e, amount: a }\n\
     def value(w: Wrapped) -> f32 =\n  match w with {\n    | Wrapped { extra: _, amount } => amount\n  }\n";

const DEMO_MAIN: &str = "module Demo.Main\n\
     import Demo.Alpha\n\
     import Demo.Beta\n\
     export (main)\n\
     def main() -> f32 = add(Demo.Alpha.value(Demo.Alpha.make(1.0, 2.0)), Demo.Beta.value(Demo.Beta.make(3.0, 4.0)))\n";

#[test]
fn module_qualified_constructors_in_one_package_keep_their_own_field_order() {
    // Two modules of ONE package each declare a record `Wrapped` with the
    // same field names in the opposite order, reached through the
    // module-qualified references chelis#316 introduced. Pre-fix this
    // rejected with "field `amount` is absent or ambiguous on the
    // resolved ADT type": the two declarations' mangled names share the
    // `Wrapped` terminal exactly as two packages' do, so the in-package
    // path carries the same defect. Every test above is cross-package, so
    // this pins the second path.
    let modules = [
        ("alpha.ch", ALPHA),
        ("beta.ch", BETA),
        ("main.ch", DEMO_MAIN),
    ];
    let (_dir, package_root, built) = build_single_package(&modules, "main.ch");
    let source = built.source();

    // `Alpha.value` is a field access and `Beta.value` is a match, so the
    // two consumers of a colliding declaration exercise different lowering
    // paths in one program. Each must read its own declaration's index:
    // `amount` is field 0 in Alpha and field 1 in Beta.
    let alpha = function_body(source, "float pkg__demo__Demo__Alpha__value(chelis_adt* w)");
    assert!(
        alpha.contains("chelis_adt_get_field(__adt_base_0, 0)"),
        "`amount` is field 0 in Demo.Alpha; emitted body was:\n{alpha}"
    );
    let beta = function_body(source, "float pkg__demo__Demo__Beta__value(chelis_adt* w)");
    assert!(
        beta.contains("chelis_adt_get_field(__adt_0, 1)"),
        "`amount` is field 1 in Demo.Beta; emitted body was:\n{beta}"
    );

    // Cross-lane agreement, which is what the issue reports as broken:
    // `check` scores 1.0, `eval` answers 5.0, and the compiled artifact
    // must answer the same. 1.0 from Alpha plus 4.0 from Beta; reading
    // either module's field through the other's layout gives 2.0 or 3.0.
    let entry = package_root.join("src/main.ch");
    let checked = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&package_root)
        .args(["check", entry.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    let report: serde_json::Value =
        serde_json::from_slice(&checked.stdout).expect("check output must be json");
    assert_eq!(
        report["score"], 1.0,
        "the package must check clean: {report}"
    );

    let evaluated = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&package_root)
        .args(["eval", "--file", entry.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    let evaluated = String::from_utf8_lossy(&evaluated.stdout).into_owned();
    assert!(
        evaluated.contains("main = 5.0"),
        "eval must answer 5.0; got:\n{evaluated}"
    );

    if let Some(printed) =
        built.compile_and_run(&common::authored_c_symbol("pkg__demo__Demo__Main__main"))
    {
        assert_eq!(
            printed, "5.000000",
            "the compiled artifact must agree with eval"
        );
    }
}

// ── Negative parity ──────────────────────────────────────────────────

/// Publish `library` (with `collider` under it) and return what
/// `chelis check` said about the LIBRARY itself. The two negative-parity
/// cases below are rejected by the checker, so the collider must not let
/// them through: the point is that resolving the collision correctly did
/// not also make wrong programs pass.
fn check_library(collider: &Package<'_>, library: &Package<'_>) -> String {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    let reef_home = root.join("reef-home");
    fs::create_dir_all(&reef_home).expect("mkdir reef home");
    let collider_root = write_package(root, collider, &[]);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", collider_root.to_str().unwrap()])
        .assert()
        .success();
    let library_root = write_package(root, library, &[collider.name]);
    let entry = library_root.join("src/m.ch");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&library_root)
        .args(["check", entry.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn a_genuinely_absent_record_field_is_still_rejected() {
    // The missing-field check is what mis-fired, so it must keep firing
    // when the field really is absent. `inner` exists only on the
    // collider's declaration, which is exactly the substitution the bug
    // performed - so if the collision were "resolved" by accepting the
    // other package's field list, this would compile.
    const LIBRARY: Package<'static> = Package {
        name: "collidelib",
        prefix: "Collidelib",
        exports: "(Wrapped, make)",
        body: "type Wrapped =\n  | Wrapped { amount: f32, tag: f32 }\n\
               def make(x: f32) -> Wrapped = Wrapped { amount: x, tag: 1.0, inner: x }\n",
    };

    let report = check_library(&RECORD_COLLIDER, &LIBRARY);
    assert!(
        report.contains("unknown record field 'inner' in construction of"),
        "a field the authored declaration lacks must stay rejected; got:\n{report}"
    );
}

#[test]
fn a_genuinely_absent_accessed_field_is_still_rejected() {
    // Negative parity for the access face. `inner` is the collider's
    // field name, so an over-permissive field lookup would resolve it.
    const LIBRARY: Package<'static> = Package {
        name: "collidelib",
        prefix: "Collidelib",
        exports: "(Wrapped, make, value)",
        body: "type Wrapped =\n  | Wrapped { extra: f32, amount: f32 }\n\
               def make(e: f32, a: f32) -> Wrapped = Wrapped { extra: e, amount: a }\n\
               def value(w: Wrapped) -> f32 = w.inner\n",
    };

    let report = check_library(&RECORD_COLLIDER, &LIBRARY);
    assert!(
        report.contains("unknown record field 'inner' on"),
        "accessing a field the authored declaration lacks must stay \
         rejected; got:\n{report}"
    );
}
