"""Actual MIR controls for the five-owner Python codec discovery scope."""
import hashlib
import json
from pathlib import Path

from capacity_census_graph import GraphError
from capacity_census_wire_adapters import canonical


def mir_control_sources(root):
    original = (root / "crates/chelis-python/src/compiler_json.rs").read_text()
    encode = "serde_json::to_string(&self.value)"
    decode = "serde_json::from_str::<BTreeMap<String, TensorValue>>(&text)"
    if original.count(encode) != 4 or original.count(decode) != 1:
        raise GraphError("changed production adapter source needs explicit MIR controls")
    header = "use pyo3::exceptions::PyRuntimeError as ChelisError;\n"
    def source(body, helper=""):
        return header + helper + "\nmod compiler_json {\n" + body + "\n}\n"
    return (
        ("actual_conversion_owners", source(original), None),
        ("output_helper_is_not_conversion_owner", source(original.replace(encode, "super::encode_check(&self.value)", 1),
             "fn encode_check(value: &chelis_compiler_api::schema::CheckResult) -> Result<String, serde_json::Error> { serde_json::to_string(value) }"), "missing concrete conversion owner"),
        ("output_wrong_payload_is_not_result_codec", source(original.replace(encode, 'serde_json::to_string(&"untyped")', 1)), "compiled codec payload differs"),
        ("input_helper_is_not_extraction_owner", source(original.replace(decode, "super::decode_bindings(&text)"),
             "fn decode_bindings(text: &str) -> Result<std::collections::BTreeMap<String, chelis_compiler_api::schema::TensorValue>, serde_json::Error> { serde_json::from_str(text) }"), "missing concrete conversion owner"),
        ("input_string_is_not_tensor_map", source(original.replace(decode, "serde_json::from_str::<String>(&text).map(|_| BTreeMap::new())")), "compiled codec payload differs"),
    )


def verify_mir_controls(root, target, driver, collected):
    from capacity_census_compiler_json import validate_conversion_calls
    from capacity_census_wire_calls import analyze_fixture

    directory = target / "compiler-json-mir-controls"
    directory.mkdir(parents=True, exist_ok=True)
    externs = {row["name"]: row["artifact"] for row in collected["fixture_externs"]}
    records = []
    for name, source, expected in mir_control_sources(root):
        evidence = analyze_fixture(driver, directory, source, externs, scope="compiler-json", log_prefix=directory / name)
        observed = None
        try:
            validate_conversion_calls(evidence.raw)
        except GraphError as error:
            observed = str(error)
        if (expected is None and observed is not None) or (expected is not None and (observed is None or expected not in observed)):
            raise GraphError(f"actual MIR control {name} failed: {observed!r}")
        records.append({"name": name, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "compiler_evidence_sha256": hashlib.sha256(canonical(evidence.raw).encode()).hexdigest(),
                        "expected_error": expected, "observed_error": observed,
                        "process": json.loads((directory / (name + ".json")).read_text())})
    return tuple(records)
