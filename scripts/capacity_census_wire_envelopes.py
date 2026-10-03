"""Independent JSON-envelope observations; no binary-envelope compatibility claim."""

from __future__ import annotations

from capacity_census_wire_adapters import CodecCase, canonical


def envelope_cases():
    cases = []
    manifest = {"target": "Eval", "entries": [], "requires_main": False}
    empty = {"schema_version": 4, "roots": [], "manifest": manifest}

    def add(carrier, name, text, expected, error=None, codec="json"):
        cases.append(
            CodecCase(
                f"{carrier}/{codec}/{name}",
                "envelope",
                carrier,
                codec,
                text,
                expected,
                error,
            )
        )

    scalar = '{"dtype":"f64","bits":"8000000000000000"}'
    root = '{"node_id":0,"value":{"type":"scalar","value":' + scalar + "}}"
    inputs = [
        ("empty", '{"schema_version":4,"roots":[]}', empty, None),
        ("reordered", '{"roots":[],"schema_version":4}', empty, None),
        (
            "scalar",
            '{"schema_version":4,"roots":[' + root + "]}",
            {
                **empty,
                "roots": [
                    {
                        "node_id": 0,
                        "value": {
                            "type": "scalar",
                            "value": {"dtype": "f64", "bits": "8000000000000000"},
                        },
                    }
                ],
            },
            None,
        ),
        (
            "duplicate-version",
            '{"schema_version":4,"roots":[],"schema_version":4}',
            None,
            "duplicate field",
        ),
        (
            "duplicate-bits",
            '{"schema_version":4,"roots":['
            + root.replace('"bits":', '"bits":"8000000000000000","bits":')
            + "]}",
            None,
            "duplicate field",
        ),
        (
            "duplicate-dtype",
            '{"schema_version":4,"roots":['
            + root.replace('"dtype":', '"dtype":"f64","dtype":')
            + "]}",
            None,
            "duplicate field",
        ),
    ]
    for version in (None, 3, 5):
        prefix = "" if version is None else f'"schema_version":{version},'
        suffix = "" if version is None else f',"schema_version":{version}'
        inputs += [
            (
                f"version-first-{version}",
                "{" + prefix + '"roots":[{"value":{"type":"unknown"}}]}',
                None,
                "schema_version",
            ),
            (
                f"version-last-{version}",
                '{"roots":[{"value":{"type":"unknown"}}]' + suffix + "}",
                None,
                "schema_version",
            ),
        ]
    for name, text, expected, error in inputs:
        add("EvalResult", name, text, expected, error)
        for carrier in ("WireApiEnvelope<EvalResult>", "WireBatchResult"):
            wrapped = '{"ok":true,"result":' + text + "}"
            observed = (
                {"variant": "success", "ok": True, "value": expected}
                if expected is not None
                else None
            )
            if carrier == "WireBatchResult":
                wrapped = '{"kind":"eval",' + wrapped[1:]
                observed = {"kind": "eval", "envelope": observed} if observed else None
            add(carrier, name, wrapped, observed, error)
    for version in (3, 4, 5):
        add(
            "EvalResult",
            "producer-version-" + str(version),
            canonical({"schema_version": version, "roots": []}),
            empty if version == 4 else None,
            None if version == 4 else "schema_version",
            "construct",
        )
    failure = '{"ok":false,"stage":"check","errors":[]}'
    expected_failure = {
        "variant": "failure",
        "ok": False,
        "stage": "check",
        "error_count": 0,
    }
    add("WireApiEnvelope<EvalResult>", "failure", failure, expected_failure)
    for kind in (
        "parse",
        "desugar",
        "check",
        "lower",
        "compile",
        "eval",
        "grad",
        "validate",
        "decompile",
    ):
        add(
            "WireBatchResult",
            "dispatch-" + kind,
            '{"kind":"' + kind + '",' + failure[1:],
            {"kind": kind, "envelope": expected_failure},
        )
    for carrier in ("WireApiEnvelope<EvalResult>", "WireBatchResult"):
        for name, text in (
            ("missing-ok", '{"stage":"check","errors":[]}'),
            ("duplicate-ok", '{"ok":false,' + failure[1:]),
            ("wrong-ok", '{"ok":0,"stage":"check","errors":[]}'),
        ):
            if carrier == "WireBatchResult":
                text = '{"kind":"eval",' + text[1:]
            add(carrier, name, text, None)
    for kind in ("unknown", "Eval"):
        add(
            "WireBatchResult",
            "unknown-kind-" + kind,
            '{"kind":"' + kind + '",' + failure[1:],
            None,
            "unknown variant",
        )
    add(
        "WireBatchResult",
        "duplicate-kind",
        '{"kind":"eval","kind":"eval",' + failure[1:],
        None,
        "duplicate field",
    )
    return cases


def dag_cases():
    import copy

    cases = []

    def add(name, value, valid, error=None):
        text = value if isinstance(value, str) else canonical(value)
        expected = copy.deepcopy(value) if valid else None
        for codec in ("json", "construct", "admit"):
            # Raw duplicate-key checks must use the actual byte decoder. A
            # constructor observes already parsed fields and cannot recreate keys.
            if isinstance(value, str) and codec == "construct":
                continue
            cases.append(
                CodecCase(
                    f"WireDag/{codec}/{name}",
                    "dag",
                    "WireDag",
                    codec,
                    text,
                    expected,
                    error,
                )
            )

    empty = {"schema_version": 25, "declarations": [], "nodes": [], "roots": []}
    add("empty", empty, True)
    for version in (None, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26):
        value = {**empty, "schema_version": version}
        if version is None:
            del value["schema_version"]
        add("version-" + str(version), value, False)
    scalar = {"dtype": "f64", "bits": "8000000000000000"}
    const = {
        "schema_version": 25,
        "declarations": ["entry"],
        "nodes": [
            {
                "shape_deps": [],
                "span_id": None,
                "merged_spans": [],
                "declaration": 0,
                "activation": None,
                "id": 0,
                "op": {"kind": "const", "value": scalar},
                "inputs": [],
                "output_type": {"dims": [], "precision": "f64"},
            }
        ],
        "roots": [0],
    }
    add("exact-scalar", const, True)
    raw = canonical(const)
    for field, value in (("bits", "8000000000000000"), ("dtype", "f64")):
        key = canonical(field) + ":"
        add(
            "duplicate-" + field,
            raw.replace(key, key + canonical(value) + "," + key),
            False,
            "duplicate field",
        )
    load = {
        "shape_deps": [],
        "span_id": None,
        "merged_spans": [],
        "declaration": 0,
        "activation": None,
        "id": 0,
        "op": {"kind": "load", "name": "x"},
        "inputs": [],
        "output_type": {"dims": [{"kind": "lit", "size": 2}], "precision": "f32"},
    }

    def graph(op):
        return {
            "schema_version": 25,
            "declarations": ["entry"],
            "nodes": [
                copy.deepcopy(load),
                {
                    "shape_deps": [],
                    "span_id": None,
                    "merged_spans": [],
                    "declaration": 0,
                    "activation": None,
                    "id": 1,
                    "op": op,
                    "inputs": [0],
                    "output_type": copy.deepcopy(load["output_type"]),
                },
            ],
            "roots": [1],
        }

    softmax = graph({"kind": "softmax", "axis": 0})
    add("softmax", softmax, True)
    for label, target, value in (
        ("axis", "op", {"kind": "softmax", "axis": 1}),
        ("negative-axis", "op", {"kind": "softmax", "axis": -1}),
        ("arity", "inputs", []),
        ("dtype", "output_type", {"dims": [{"kind": "lit", "size": 2}], "precision": "i32"}),
        ("shape", "output_type", {"dims": [{"kind": "lit", "size": 3}], "precision": "f32"}),
    ):
        bad = copy.deepcopy(softmax)
        bad["nodes"][1][target] = value
        add("softmax-invalid-" + label, bad, False)

    good = graph({"kind": "copy"})
    add("owned-reference", good, True)
    for name, field, value in (
        ("id-position", "id", 7),
        ("self-input", "inputs", [1]),
        ("large-input", "inputs", [18446744073709551615]),
    ):
        bad = copy.deepcopy(good)
        bad["nodes"][1][field] = value
        add(name, bad, False)
    bad = copy.deepcopy(good)
    bad["roots"] = [2]
    add("root-owner", bad, False)
    # spec/10 section 3.2: a node's activation is an earlier bool node, or
    # explicitly null.
    activated = copy.deepcopy(good)
    activated["nodes"].insert(
        1,
        {
            "shape_deps": [],
            "span_id": None,
            "merged_spans": [],
            "declaration": 0,
            "activation": None,
            "id": 1,
            "op": {"kind": "load", "name": "c"},
            "inputs": [],
            "output_type": {"dims": [], "precision": "bool"},
        },
    )
    activated["nodes"][2]["id"] = 2
    activated["nodes"][2]["activation"] = 1
    activated["roots"] = [2]
    add("activated-reference", activated, True)
    for name, target in (
        ("self", 2),
        ("large", 18446744073709551615),
        ("negative", -1),
        ("float", 1.0),
        ("not-bool", 0),
    ):
        bad = copy.deepcopy(activated)
        bad["nodes"][2]["activation"] = target
        add("activation-" + name, bad, False)
    for field in ("shape_deps", "span_id", "merged_spans", "declaration", "activation"):
        bad = copy.deepcopy(good)
        del bad["nodes"][1][field]
        add("missing-node-" + field, bad, False, "missing field")
    bad = copy.deepcopy(good)
    del bad["declarations"]
    add("missing-declarations", bad, False, "missing field")
    # A node names its declaration by row; two rows may share a name, and
    # every row is some node's declaration.
    shared = copy.deepcopy(good)
    shared["declarations"] = ["entry", "entry"]
    shared["nodes"][1]["declaration"] = 1
    add("shared-declaration-name", shared, True)
    for name, row in (
        ("outside", 1),
        ("large", 18446744073709551615),
        ("negative", -1),
        ("float", 0.0),
    ):
        bad = copy.deepcopy(good)
        bad["nodes"][1]["declaration"] = row
        add("declaration-row-" + name, bad, False)
    bad = copy.deepcopy(good)
    bad["declarations"] = ["entry", "other"]
    add("declaration-row-unused", bad, False)
    for size in (0, -1, 9223372036854775808, 2.0):
        changed = copy.deepcopy(good)
        for node in changed["nodes"]:
            node["output_type"]["dims"][0]["size"] = size
        add("extent-" + str(size), changed, size == 0)
    for axis in (0, -1, 1, 2147483648, 0.0):
        value = graph({"kind": "shape", "axis": axis})
        value["nodes"][1]["output_type"] = {"dims": [], "precision": "int64"}
        add("axis-" + str(axis), value, type(axis) is int and axis == 0)
    fused = graph(
        {
            "kind": "fused_elem",
            "ops": [
                {"op": "neg", "input_indices": [{"kind": "external", "index": 0}]},
                {"op": "neg", "input_indices": [{"kind": "previous_step", "index": 0}]},
            ],
        }
    )
    add("fused-owned", fused, True)
    for position, kind, index in (
        (0, "external", 1),
        (0, "previous_step", 0),
        (1, "previous_step", 1),
    ):
        bad = copy.deepcopy(fused)
        bad["nodes"][1]["op"]["ops"][position]["input_indices"] = [
            {"kind": kind, "index": index}
        ]
        add(f"fused-{position}-{kind}-{index}", bad, False)
    value_node = copy.deepcopy(load)
    value_node["output_type"]["dims"] = [{"kind": "lit", "size": 1}]
    witness_node = copy.deepcopy(load)
    witness_node["id"] = 1
    witness_node["op"]["name"] = "witness"
    witness_node["output_type"]["dims"] = [{"kind": "lit", "size": 4}]
    reference_graph = {
        "schema_version": 25,
        "declarations": ["entry"],
        "nodes": [
            value_node,
            witness_node,
            {
                "shape_deps": [],
                "span_id": None,
                "merged_spans": [],
                "declaration": 0,
                "activation": None,
                "id": 2,
                "op": {
                    "kind": "expand",
                    "axis": 0,
                    "size": {
                        "bound": "input_axis",
                        "tensor": 1,
                        "axis": {"axis": "lit", "value": 0},
                    },
                },
                "inputs": [0, 1],
                "output_type": {
                    "dims": [{"kind": "lit", "size": 4}],
                    "precision": "f32",
                },
            },
        ],
        "roots": [2],
    }
    add("input-axis-owned", reference_graph, True)
    for slot in (0, 2, 18446744073709551615):
        changed = copy.deepcopy(reference_graph)
        changed["nodes"][2]["op"]["size"]["tensor"] = slot
        add("input-axis-slot-" + str(slot), changed, False)
    for axis in (-1, 1):
        changed = copy.deepcopy(reference_graph)
        changed["nodes"][2]["op"]["size"]["axis"]["value"] = axis
        add("input-axis-axis-" + str(axis), changed, False)
    node_graph = copy.deepcopy(reference_graph)
    node_graph["nodes"][1]["output_type"] = {"dims": [], "precision": "int64"}
    node_graph["nodes"][2]["op"]["size"] = {"bound": "node", "input": 1}
    add("extent-node-owned", node_graph, True)
    for dtype, dims in (("int32", []), ("int64", [{"kind": "lit", "size": 1}])):
        changed = copy.deepcopy(node_graph)
        changed["nodes"][1]["output_type"] = {"dims": dims, "precision": dtype}
        add(
            "extent-node-source-" + dtype + str(len(dims)),
            changed,
            False,
            "rank-0 int64",
        )

    # spec/05 §2.4.1's owner matrix. A tagged reference's meaning is the
    # owning operation's slot, never permission to use that carrier anywhere.
    def movement(owner, value):
        zero = {"bound": "lit", "value": 0}
        if owner == "expand":
            return {"kind": owner, "axis": 0, "size": value}
        if owner == "reshape":
            return {"kind": owner, "new_shape": [value]}
        if owner == "pad":
            return {
                "kind": owner,
                "padding": [[zero, value]],
                "fill": {"dtype": "f32", "bits": "00000000"},
            }
        if owner == "shrink":
            return {"kind": owner, "bounds": [[zero, value]]}
        assert owner == "stride"
        return {"kind": owner, "strides": [value]}

    for owner in ("expand", "reshape", "pad", "shrink", "stride"):
        for form in ("node", "input-axis"):
            changed = copy.deepcopy(node_graph if form == "node" else reference_graph)
            value = copy.deepcopy(changed["nodes"][2]["op"]["size"])
            changed["nodes"][2]["op"] = movement(owner, value)
            if owner != "expand":
                changed["nodes"][0]["output_type"]["dims"][0]["size"] = 4
            if owner == "pad":
                changed["nodes"][2]["output_type"]["dims"][0]["size"] = 8
            admitted = form == "node" or owner in {"expand", "reshape"}
            add(
                f"owner-{owner}-{form}",
                changed,
                admitted,
                None if admitted else "forbids the input_axis carrier",
            )
            if form == "node":
                bad = copy.deepcopy(changed)
                bad["nodes"][2]["op"] = movement(owner, {"bound": "node", "input": 0})
                add(f"owner-{owner}-node-zero-slot", bad, False, "invalid input slot 0")
            if admitted:
                extra = copy.deepcopy(changed)
                extra["nodes"][2]["inputs"].append(0)
                add(f"owner-{owner}-{form}-extra-input", extra, False)
    for owner in ("expand", "reshape", "pad", "shrink", "stride"):
        changed = graph(movement(owner, {"bound": "to_end"}))
        if owner == "shrink":
            add("owner-shrink-to-end", changed, True)
            changed["nodes"][1]["op"]["bounds"][0][0] = {"bound": "lit", "value": 1}
            add("owner-shrink-to-end-start-one", changed, False, "not literal 0")
        else:
            add(f"owner-{owner}-to-end", changed, False, "forbids the to_end carrier")

    witness = {
        "schema_version": 25,
        "declarations": ["entry"],
        "nodes": [
            copy.deepcopy(load),
            {
                "shape_deps": [],
                "span_id": "call-f",
                "merged_spans": ["inlined-g"],
                "declaration": 0,
                "activation": None,
                "id": 1,
                "op": {
                    "kind": "extent_witness",
                    "site": "caller",
                    "parameter": "x",
                    "axis": {"axis": "lit", "value": 0},
                    "requirements": [4, 4, 9],
                    "claims": [],
                },
                "inputs": [0],
                "output_type": {"dims": [], "precision": "int64"},
            },
            {
                "shape_deps": [1],
                "span_id": None,
                "merged_spans": [],
                "declaration": 0,
                "activation": None,
                "id": 2,
                "op": {
                    "kind": "const",
                    "value": {"dtype": "int64", "value": 9},
                },
                "inputs": [],
                "output_type": {"dims": [], "precision": "int64"},
            },
        ],
        "roots": [2],
    }
    add("extent-witness-owned", witness, True)
    for name, node, field, value, error in (
        ("shape-dep-self", 2, "shape_deps", [2], "earlier node"),
        ("shape-dep-large", 2, "shape_deps", [18446744073709551615], "earlier node"),
        ("shape-dep-negative", 2, "shape_deps", [-1], None),
        ("shape-dep-float", 2, "shape_deps", [1.0], None),
        ("witness-negative-requirement", 1, "requirements", [-1], "nonnegative int64"),
        ("witness-float-requirement", 1, "requirements", [4.0], None),
    ):
        bad = copy.deepcopy(witness)
        if field == "requirements":
            bad["nodes"][node]["op"][field] = value
        else:
            bad["nodes"][node][field] = value
        add(name, bad, False, error)
    for field in ("parameter", "axis", "requirements", "claims"):
        bad = copy.deepcopy(witness)
        del bad["nodes"][1]["op"][field]
        add("missing-witness-" + field, bad, False, "missing field")
    # wire v11 (chelis#1374): a named claim owes one earlier witness edge, and
    # its binder identifies the obligation, so neither may be waived.
    for name, claims, inputs, error in (
        (
            "witness-claim-without-edge",
            [{"claim": "rows", "requirement_declares": True}],
            [0],
            "one earlier witness input per named claim",
        ),
        (
            "witness-claim-edge-is-not-a-witness",
            [{"claim": "rows", "requirement_declares": True}],
            [0, 0],
            "earlier rank-0 int64 extent witness",
        ),
        (
            "witness-claim-empty-binder",
            [{"claim": "", "requirement_declares": True}],
            [0, 0],
            "nonempty dimension binder",
        ),
        (
            "witness-claim-missing-role",
            [{"claim": "rows"}],
            [0, 0],
            "missing field",
        ),
    ):
        bad = copy.deepcopy(witness)
        bad["nodes"][1]["op"]["claims"] = claims
        bad["nodes"][1]["inputs"] = inputs
        add(name, bad, False, error)
    for name, field, value, error in (
        ("witness-axis-out-of-range", "axis", {"axis": "lit", "value": 1}, "wire axis"),
        ("witness-no-input", "inputs", [], "one earlier tensor input"),
        ("witness-extra-input", "inputs", [0, 0], "one earlier tensor input"),
        ("witness-wrong-output", "output_type", {"dims": [], "precision": "f32"}, "rank-0 int64"),
        (
            "witness-ranked-output",
            "output_type",
            {"dims": [{"kind": "lit", "size": 1}], "precision": "int64"},
            "rank-0 int64",
        ),
    ):
        bad = copy.deepcopy(witness)
        if field == "axis":
            bad["nodes"][1]["op"][field] = value
        else:
            bad["nodes"][1][field] = value
        add(name, bad, False, error)
    # Wire v12: a result token owns an exact earlier declaring observation,
    # and the introducing operation owns the token. Printed labels are not
    # references; the normalized axis travels in the existing fixed carrier.
    result_claim = copy.deepcopy(witness)
    result_claim["nodes"] = result_claim["nodes"][:2]
    token = copy.deepcopy(result_claim["nodes"][1])
    token["id"] = 2
    token["shape_deps"] = [1]
    token["op"]["requirements"] = []
    token["op"]["site"] = {"result_claim": {"claim": "rows", "axis": {"axis": "lit", "value": 0}}}
    result_claim["nodes"].append(token)
    value = copy.deepcopy(witness["nodes"][2])
    value["id"] = 3
    value["shape_deps"] = []
    result_claim["nodes"].append(value)
    producer = copy.deepcopy(value)
    producer["id"] = 4
    producer["op"] = {"kind": "expand", "axis": 0, "size": {"bound": "node", "input": 1}}
    producer["inputs"] = [3, 1]
    producer["shape_deps"] = [2]
    producer["output_type"]["dims"] = [{"kind": "named", "name": "*", "size": None}]
    result_claim["nodes"].append(producer)
    result_claim["roots"] = [4]
    add("result-claim-owned", result_claim, True)
    for label, axis, dependency in [("", 0, [1]), ("rows", -1, [1]), ("rows", 1, [1]), ("rows", 0, []), ("rows", 0, [0])]:
        bad = copy.deepcopy(result_claim)
        role = bad["nodes"][2]["op"]["site"]["result_claim"]
        role["claim"] = label
        role["axis"]["value"] = axis
        bad["nodes"][2]["shape_deps"] = dependency
        add(f"result-claim-invalid-{label}-{axis}-{dependency}", bad, False)
    for field in ("claim", "axis"):
        bad = copy.deepcopy(result_claim)
        del bad["nodes"][2]["op"]["site"]["result_claim"][field]
        add("result-claim-missing-" + field, bad, False, "missing field")
    for op in ({"kind": "copy"}, {"kind": "expand", "axis": 0, "size": {"bound": "lit", "value": 9}}, {"kind": "reshape", "new_shape": [{"bound": "lit", "value": 9}]}, {"kind": "stride", "strides": [{"bound": "node", "input": 1}]}):
        bad = copy.deepcopy(result_claim)
        bad["nodes"][4]["op"] = op
        add("result-claim-invalid-producer-" + op["kind"], bad, False, "supported producing axis")

    # Wire v15: the checker-assigned local-ascription id is an opaque
    # artifact-local identity. It admits the complete u64 domain but never a
    # signed or fractional numeric representation.
    local_ascription = copy.deepcopy(witness)
    local_ascription["nodes"] = local_ascription["nodes"][:3]
    local_ascription["nodes"][1]["op"] = {
        "kind": "extent_witness",
        "site": {
            "local_ascription_claim": {
                "ascription_id": 18446744073709551615,
                "binding": "y",
                "claim": "2",
                "axis": {"axis": "lit", "value": 0},
            }
        },
        "parameter": "",
        "axis": {"axis": "lit", "value": 0},
        "requirements": [2],
        "claims": [],
    }
    local_ascription["nodes"][1]["inputs"] = []
    local_ascription["nodes"][1]["shape_deps"] = []
    local_ascription["nodes"][2]["op"] = {
        "kind": "pad",
        "padding": [
            [
                {"bound": "lit", "value": 0},
                {"bound": "lit", "value": 0},
            ]
        ],
        "fill": {"dtype": "f32", "bits": "00000000"},
    }
    local_ascription["nodes"][2]["inputs"] = [0]
    local_ascription["nodes"][2]["shape_deps"] = [1]
    local_ascription["nodes"][2]["output_type"] = copy.deepcopy(
        local_ascription["nodes"][0]["output_type"]
    )
    local_ascription["roots"] = [2]
    add("local-ascription-owned", local_ascription, True)
    for name, value in (
        ("local-ascription-id-negative", -1),
        ("local-ascription-id-float", 7.0),
    ):
        bad = copy.deepcopy(local_ascription)
        bad["nodes"][1]["op"]["site"]["local_ascription_claim"][
            "ascription_id"
        ] = value
        add(name, bad, False)

    # The named form observes the same tensor axis as an earlier Caller
    # witness and retains that declaring witness as its sole shape dependency.
    # It is a distinct representation from the literal form above: neither
    # requirements nor a guessed parameter-less fallback are admitted.
    named_local_ascription = copy.deepcopy(witness)
    named_local_ascription["nodes"] = named_local_ascription["nodes"][:2]
    named_local_ascription["nodes"][1]["op"]["requirements"] = []
    named_local_ascription["nodes"][1]["op"]["claims"] = []
    named_token = copy.deepcopy(named_local_ascription["nodes"][1])
    named_token["id"] = 2
    named_token["op"]["site"] = {
        "local_ascription_claim": {
            "ascription_id": 17,
            "binding": "y",
            "claim": "rows",
            "axis": {"axis": "lit", "value": 0},
        }
    }
    named_token["shape_deps"] = [1]
    named_owner = copy.deepcopy(local_ascription["nodes"][2])
    named_owner["id"] = 3
    named_owner["shape_deps"] = [2]
    named_local_ascription["nodes"].extend([named_token, named_owner])
    named_local_ascription["roots"] = [3]
    add("named-local-ascription-owned", named_local_ascription, True)
    for name, mutate in (
        (
            "named-local-ascription-missing-declaration",
            lambda value: value["nodes"][2].update(shape_deps=[]),
        ),
        (
            "named-local-ascription-wrong-declaration",
            lambda value: value["nodes"][2].update(shape_deps=[0]),
        ),
        (
            "named-local-ascription-empty-parameter",
            lambda value: value["nodes"][2]["op"].update(parameter=""),
        ),
        (
            "named-local-ascription-literal-hybrid",
            lambda value: value["nodes"][2]["op"].update(requirements=[2]),
        ),
        (
            "named-local-ascription-missing-owner",
            lambda value: value["nodes"][3].update(shape_deps=[]),
        ),
    ):
        bad = copy.deepcopy(named_local_ascription)
        mutate(bad)
        add(name, bad, False)
    return cases


def result_reference_cases():
    import copy

    cases = []
    dag = {
        "schema_version": 25,
        "declarations": ["entry"],
        "nodes": [
            {
                "shape_deps": [],
                "span_id": None,
                "merged_spans": [],
                "declaration": 0,
                "activation": None,
                "id": 0,
                "inputs": [],
                "op": {"kind": "load", "name": "x"},
                "output_type": {"dims": [], "precision": "f32"},
            }
        ],
        "roots": [0],
    }
    for carrier, good in (
        ("LowerResult", {"dag": dag, "named_roots": {"x": 0}}),
        (
            "GradResult",
            {
                "dag": dag,
                "output_node": 0,
                "grad_nodes_by_name": {"dx": 0},
                "forward_nodes_by_name": {"x": 0},
            },
        ),
    ):
        for codec in ("json", "construct"):
            cases.append(
                CodecCase(
                    f"{carrier}/{codec}/owned",
                    "result-reference",
                    carrier,
                    codec,
                    canonical(good),
                    good,
                )
            )
            for field in good:
                if field == "dag":
                    continue
                for index in (1, 18446744073709551615):
                    bad = copy.deepcopy(good)
                    bad[field] = index if field == "output_node" else {"renamed": index}
                    cases.append(
                        CodecCase(
                            f"{carrier}/{codec}/{field}-{index}",
                            "result-reference",
                            carrier,
                            codec,
                            canonical(bad),
                            None,
                            "outside the owning DAG",
                        )
                    )
        for version in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26):
            bad = copy.deepcopy(good)
            bad["dag"]["schema_version"] = version
            cases.append(
                CodecCase(
                    f"{carrier}/json/version-{version}",
                    "result-reference",
                    carrier,
                    "json",
                    canonical(bad),
                    None,
                )
            )
    return cases


def metadata_reference_cases():
    cases = []

    def add(carrier, name, value, expected):
        cases.append(
            CodecCase(
                f"{carrier}/json/{name}",
                "metadata-reference",
                carrier,
                "json",
                canonical(value),
                expected,
            )
        )

    for name, value, expected in (
        ("absent", {"name": None}, {"name": None}),
        ("unresolved", {"name": "n", "size": None}, {"name": "n"}),
    ):
        add("ExecutionDim", name, value, expected)
    for carrier in ("ExecutionDim", "WireInferredDim"):
        for value in (
            0,
            9007199254740993,
            9223372036854775807,
            -1,
            9223372036854775808,
            1.0,
            True,
        ):
            obj = (
                {"name": None, "size": value}
                if carrier == "ExecutionDim"
                else {"kind": "lit", "size": value}
            )
            valid = type(value) is int and 0 <= value <= 9223372036854775807
            add(carrier, "extent-" + str(value), obj, obj if valid else None)
    for carrier, tag in (
        ("WireInferredType", "var"),
        ("WireInferredPrecision", "var"),
        ("WireInferredDim", "var"),
        ("WireInferredDim", "rank"),
    ):
        for value in (0, 4294967295, -1, 4294967296, 0.0):
            obj = {"kind": tag, "id": value}
            valid = type(value) is int and 0 <= value <= 4294967295
            add(carrier, f"{tag}-{value}", obj, obj if valid else None)
    for value in (0, 18446744073709551615, -1, 18446744073709551616, 0.0):
        obj = {"node_id": value, "value": {"type": "unit"}}
        valid = type(value) is int and 0 <= value <= 18446744073709551615
        add("EvaluatedRoot", "opaque-" + str(value), obj, obj if valid else None)
    return cases
