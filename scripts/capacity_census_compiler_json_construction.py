"""Compile exact production-adapter construction controls, without a fixture escape."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from capacity_census_graph import GraphError
from capacity_census_wire_adapters import canonical


def construction_sources(root):
    source = root / "crates/chelis-python/src/compiler_json.rs"
    prefix = ('#![allow(dead_code)]\n'
              'use pyo3::exceptions::PyRuntimeError as ChelisError;\n'
              'use chelis_compiler_api::schema::*;\n'
              f'#[path = {json.dumps(str(source))}] mod compiler_json;\n'
              'fn main() {}\n')
    cases = []
    for adapter, payload in (("CheckJson", "CheckResult"), ("CompileJson", "CompileResult"),
                             ("DesugarJson", "DesugarResult"), ("EvalJson", "EvalResult")):
        wrong = "CompileResult" if payload != "CompileResult" else "CheckResult"
        for label, parameter, expression, error in (
            ("exact", payload, f"compiler_json::{adapter}::new(value)", None),
            ("string", "String", f"compiler_json::{adapter}::new(value)", "E0308"),
            ("dynamic", "serde_json::Value", f"compiler_json::{adapter}::new(value)", "E0308"),
            ("wrong_root", wrong, f"compiler_json::{adapter}::new(value)", "E0308"),
            ("private_field", payload, f"compiler_json::{adapter} {{ value }}", "E0451"),
        ):
            cases.append((adapter + "/" + label, prefix + f"fn attempt(value: {parameter}) {{ let _ = {expression}; }}\n", error))
    cases += [
        ("EvalBindingsJson/exact", prefix + "fn attempt() { let _: std::collections::BTreeMap<String, TensorValue> = compiler_json::EvalBindingsJson::empty().into_bindings(); }\n", None),
        ("EvalBindingsJson/private_field", prefix + "fn attempt(value: std::collections::BTreeMap<String, TensorValue>) { let _ = compiler_json::EvalBindingsJson { value }; }\n", "E0451"),
    ]
    return tuple(cases)


def check_construction_outcome(expected_error, returncode, diagnostics, output_exists):
    errors = [item for item in diagnostics if item.get("level") == "error" and item.get("code")]
    if expected_error is None:
        valid = returncode == 0 and not errors and output_exists
    else:
        valid = returncode != 0 and not output_exists and len(errors) == 1 and errors[0]["code"]["code"] == expected_error
    if not valid:
        raise GraphError("construction control did not produce its exact compiler outcome")


def compile_construction_controls(root, target, collected):
    from capacity_census_wire_calls import record_process

    directory = target / "compiler-json-construction"
    directory.mkdir(parents=True, exist_ok=True)
    externs = collected["fixture_externs"]
    if {item["name"] for item in externs} != {"chelis_compiler_api", "pyo3", "serde_json"} or len(externs) != 3:
        raise GraphError("missing construction dependency ownership")
    def current():
        if any(hashlib.sha256(Path(item["artifact"]).read_bytes()).hexdigest() != item["sha256"] for item in externs):
            raise GraphError("construction dependency changed")
    current()
    compiler = collected["driver_build"]["compiler"]["path"]
    receipts = []
    for index, (name, source, expected) in enumerate(construction_sources(root)):
        path, output = directory / f"control_{index}.rs", directory / f"control_{index}.rmeta"
        path.write_text(source)
        output.unlink(missing_ok=True)
        command = [compiler, "--edition=2024", "--crate-name=compiler_json_construction",
                   "--crate-type=lib", "--emit=metadata", "--error-format=json", str(path), "-o", str(output)]
        for item in externs:
            command += ["--extern", item["name"] + "=" + item["artifact"]]
        for parent in sorted({str(Path(item["artifact"]).parent) for item in externs}):
            command += ["-L", "dependency=" + parent]
        result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                env={**os.environ, "RUSTC_BOOTSTRAP": "1", "PYO3_PYTHON": sys.executable, "VIRTUAL_ENV": sys.prefix})
        process = record_process(directory / f"control_{index}", command, result)
        try:
            diagnostics = [json.loads(line) for line in result.stderr.splitlines()]
            check_construction_outcome(expected, result.returncode, diagnostics, output.is_file())
        except (ValueError, GraphError) as error:
            raise GraphError(f"{name}: {error}\n{result.stderr[-4000:]}") from error
        receipts.append({"name": name, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                         "command": command, "expected_error": expected, "returncode": result.returncode,
                         "diagnostics_sha256": hashlib.sha256(canonical(diagnostics).encode()).hexdigest(), "process": process})
    current()
    return tuple(receipts)
