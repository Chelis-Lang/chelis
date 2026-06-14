use chelis_tide::mcp::handle_message;
use serde_json::json;

const HELLO_TENSOR: &str = include_str!("../../../examples/hello_tensor.ch");
const MATMUL_PROGRAM: &str = r#"a = (a : tensor[2, 3, f32])
b = (b : tensor[3, 4, f32])
out = (matmul(a, b) : tensor[2, 4, f32])
"#;
const LOSS_PROGRAM: &str = r#"x = (x : tensor[4, f32])
loss = (mean(x, 0) : tensor[f32])
"#;
const NON_SCALAR_PROGRAM: &str = r#"x = (x : tensor[4, f32])
out = (add(copy(x), x) : tensor[4, f32])
"#;
const SIMPLE_DEEP: &str = r#"(def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
"#;

#[test]
fn initialize_and_tool_discovery_work() {
    let init = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":1,
        "method":"initialize",
        "params":{}
    }))
    .expect("initialize response");
    assert_eq!(init["result"]["serverInfo"]["name"], "chelis-tide");

    let tools = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":2,
        "method":"tools/list",
        "params":{}
    }))
    .expect("tools/list response");
    let names = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "chelis_check",
            "chelis_compile",
            "chelis_desugar",
            "chelis_decompile",
            "chelis_eval",
            "chelis_grad",
            "chelis_validate",
            "chelis_prove",
        ]
    );

    let check_tool = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == "chelis_check")
        .expect("check tool");
    assert_eq!(
        check_tool["inputSchema"]["properties"]["source_kind"]["$ref"],
        "#/definitions/SourceKind"
    );
    assert_eq!(
        check_tool["inputSchema"]["definitions"]["SourceKind"]["enum"],
        json!(["surf", "deep"])
    );
    assert_eq!(
        check_tool["inputSchema"]["properties"]["source"]["type"],
        "string"
    );
    assert!(
        check_tool["inputSchema"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "source_kind")
    );
    assert!(
        check_tool["inputSchema"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "source")
    );

    let grad_tool = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == "chelis_grad")
        .expect("grad tool");
    assert_eq!(
        grad_tool["inputSchema"]["properties"]["source_kind"]["$ref"],
        "#/definitions/SourceKind"
    );
    assert_eq!(
        grad_tool["inputSchema"]["properties"]["output_name"]["type"],
        "string"
    );
    assert_eq!(
        grad_tool["inputSchema"]["properties"]["wrt_names"]["type"],
        "array"
    );
    assert!(
        grad_tool["inputSchema"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "output_name")
    );
    assert!(
        grad_tool["inputSchema"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "wrt_names")
    );
}

#[test]
fn each_tool_dispatches_successfully() {
    let cases = vec![
        (
            "chelis_check",
            json!({"source_kind":"surf","source":HELLO_TENSOR}),
        ),
        (
            "chelis_compile",
            json!({"source_kind":"surf","source":MATMUL_PROGRAM,"target":"c"}),
        ),
        ("chelis_desugar", json!({"source":HELLO_TENSOR})),
        ("chelis_decompile", json!({"source":SIMPLE_DEEP})),
        (
            "chelis_eval",
            json!({
                "source_kind":"surf",
                "source":LOSS_PROGRAM,
                "bindings":{"x":{"shape":[4],"data":[1.0,2.0,3.0,4.0]}}
            }),
        ),
        (
            "chelis_grad",
            json!({
                "source_kind":"surf",
                "source":LOSS_PROGRAM,
                "output_name":"loss",
                "wrt_names":["x"]
            }),
        ),
        (
            "chelis_validate",
            json!({"mode":"surf","source":HELLO_TENSOR}),
        ),
    ];

    for (idx, (name, arguments)) in cases.into_iter().enumerate() {
        let response = handle_message(&json!({
            "jsonrpc":"2.0",
            "id":idx,
            "method":"tools/call",
            "params":{"name":name,"arguments":arguments}
        }))
        .expect("tool response");
        assert_eq!(response["result"]["isError"], false, "tool {name}");
    }
}

#[test]
fn tool_failures_preserve_structured_errors() {
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":7,
        "method":"tools/call",
        "params":{
            "name":"chelis_grad",
            "arguments":{
                "source_kind":"surf",
                "source":NON_SCALAR_PROGRAM,
                "output_name":"out",
                "wrt_names":["x"]
            }
        }
    }))
    .expect("tool response");
    assert_eq!(response["result"]["isError"], true);
    let structured = &response["result"]["structuredContent"];
    assert_eq!(structured["ok"], false);
    assert_eq!(structured["stage"], "grad");
    assert!(!structured["errors"].as_array().unwrap().is_empty());
}

#[test]
fn invalid_tool_arguments_return_mcp_error_payload() {
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":8,
        "method":"tools/call",
        "params":{"name":"chelis_eval","arguments":{"source_kind":"surf"}}
    }))
    .expect("tool response");
    assert_eq!(response["result"]["isError"], true);
    assert_eq!(response["result"]["structuredContent"]["stage"], "mcp");
}

const OPAQUE_MODULE: &str = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
";

/// RFC D-PARITY: a prove invoked through the tide MCP tool runs the SAME
/// derived obligations as the CLI on the same module. We assert tide's
/// obligation records match the shared chelis-prove engine the CLI also
/// drives (same obligation set AND same proof_tier per obligation).
#[test]
fn tide_runs_obligations_via_shared_engine() {
    // Tide path.
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":9,
        "method":"tools/call",
        "params":{"name":"chelis_prove","arguments":{
            "source_kind":"surf","source": OPAQUE_MODULE, "seed": 0
        }}
    }))
    .expect("prove response");
    let tide_obs = response["result"]["structuredContent"]["obligations"]
        .as_array()
        .expect("obligations array")
        .clone();

    // Shared-engine path (what the CLI also calls).
    let engine_out = match chelis_prove::obligation_engine::run_surf_source_obligations(
        OPAQUE_MODULE,
        &chelis_prove::obligation_engine::ObligationRunOptions::default(),
    )
    .expect("engine run")
    {
        chelis_prove::obligation_engine::ObligationRunResult::Ran(o) => o,
        other => panic!("expected a clean module to run, got {other:?}"),
    };

    // Same obligation set: same names.
    let tide_names: Vec<&str> = tide_obs
        .iter()
        .map(|o| o["name"].as_str().unwrap())
        .collect();
    let engine_names: Vec<&str> = engine_out.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(
        tide_names, engine_names,
        "same obligation set across surfaces"
    );

    // Same proof_tier per obligation.
    for (tide_ob, engine_ob) in tide_obs.iter().zip(&engine_out) {
        assert_eq!(
            tide_ob["proof_tier"].as_str().unwrap(),
            engine_ob.proof_tier.as_str(),
            "same proof_tier per obligation across surfaces"
        );
        assert_eq!(
            tide_ob["status"].as_str().unwrap(),
            match engine_ob.status {
                chelis_prove::obligation_engine::ObligationStatus::Passed => "passed",
                chelis_prove::obligation_engine::ObligationStatus::Failed => "failed",
                chelis_prove::obligation_engine::ObligationStatus::Unsupported => "unsupported",
                chelis_prove::obligation_engine::ObligationStatus::Error => "error",
            }
        );
    }

    assert_eq!(tide_obs.len(), 1, "the flagship has exactly one obligation");
    assert_eq!(tide_obs[0]["name"], "invariant:Probability:probability");

    // CR-12 / D-PARITY: the response status must reflect the obligation
    // outcomes, not just the user property. The clean flagship module has
    // no failed obligation, so the response is ok.
    let all_ok = engine_out
        .iter()
        .all(|o| o.status == chelis_prove::obligation_engine::ObligationStatus::Passed);
    assert_eq!(
        response["result"]["structuredContent"]["ok"], all_ok,
        "tide ok reflects the obligation outcomes"
    );
}

/// Under the `smt` feature, the flagship obligation proves at proof_tier
/// "smt" through tide (the same tier the CLI gets).
#[cfg(feature = "smt")]
#[test]
fn tide_flagship_obligation_proves_at_smt_tier() {
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":10,
        "method":"tools/call",
        "params":{"name":"chelis_prove","arguments":{
            "source_kind":"surf","source": OPAQUE_MODULE
        }}
    }))
    .expect("prove response");
    let obs = response["result"]["structuredContent"]["obligations"]
        .as_array()
        .unwrap();
    assert_eq!(obs[0]["status"], "passed");
    assert_eq!(obs[0]["proof_tier"], "smt");
    assert_eq!(obs[0]["arith_model"], "real");
}

const VIOLATING_OBLIGATION_MODULE: &str = "module Stats.Prob
export (bad_prob)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def bad_prob(x: f32) -> Probability = Probability { value: x }
";

/// CR-12 (D-PARITY): a module with a FAILING producer obligation must
/// lower the tide response status (ok:false, summary.failed reflects it),
/// matching the CLI. The tide handler previously hardcoded ok:true and
/// derived the summary only from the single user property.
#[test]
fn cr12_failing_obligation_lowers_tide_status() {
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":11,
        "method":"tools/call",
        "params":{"name":"chelis_prove","arguments":{
            "source_kind":"surf","source": VIOLATING_OBLIGATION_MODULE, "seed": 0
        }}
    }))
    .expect("prove response");
    let structured = &response["result"]["structuredContent"];
    assert_eq!(
        structured["ok"], false,
        "a failing producer obligation makes the tide response not-ok: {structured}"
    );
    assert!(
        structured["summary"]["failed"].as_u64().unwrap_or(0) >= 1,
        "the failed obligation is counted in the summary: {structured}"
    );
    // The obligation record itself is failed.
    let obs = structured["obligations"].as_array().unwrap();
    assert_eq!(obs[0]["status"], "failed");
}

/// U4: an unsupported `@property` (a binder type outside the samplable set)
/// must make the response not-ok and be bucketed as unsupported. The fold
/// must not report a non-pass as ok.
#[test]
fn u4_unsupported_property_is_not_ok_through_tide() {
    // A `string`-band tensor element type is outside the L2 v1 samplable
    // set, so the property is unsupported.
    let source = "module M
@property tensor_prop forall(t: tensor[3, int32]):
  (t == t)
";
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":23,
        "method":"tools/call",
        "params":{"name":"chelis_prove","arguments":{
            "source_kind":"surf","source": source, "tier":"fuzz-only"
        }}
    }))
    .expect("prove response");
    let structured = &response["result"]["structuredContent"];
    assert_eq!(
        structured["ok"], false,
        "an unsupported property must not report ok:true: {structured}"
    );
    assert_eq!(
        structured["summary"]["unsupported"], 1,
        "unsupported is bucketed as unsupported: {structured}"
    );
}

/// U4: a `@property` that genuinely disproves is not-ok, and is discovered
/// by its declared name (NOT a hardcoded `property`).
#[test]
fn u4_disproved_property_is_not_ok_through_tide() {
    let source = "module M
@property false_claim forall(x: f32):
  (x > x)
";
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":25,
        "method":"tools/call",
        "params":{"name":"chelis_prove","arguments":{
            "source_kind":"surf","source": source, "tier":"fuzz-only"
        }}
    }))
    .expect("prove response");
    let structured = &response["result"]["structuredContent"];
    assert_eq!(
        structured["ok"], false,
        "a disproved property is not-ok: {structured}"
    );
    assert_eq!(
        structured["summary"]["total"], 1,
        "the differently-named property was discovered: {structured}"
    );
    let props = structured["properties"].as_array().unwrap();
    assert_eq!(props[0]["name"], "false_claim");
    assert_eq!(props[0]["status"], "failed");
}

/// U4 negative parity: a genuinely passing `@property` reports ok:true. The
/// stricter fold rejects only non-pass statuses.
#[test]
fn u4_passing_property_is_ok_through_tide() {
    let source = "module M
@property always_true forall(x: f32):
  (x == x)
";
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":26,
        "method":"tools/call",
        "params":{"name":"chelis_prove","arguments":{
            "source_kind":"surf","source": source, "tier":"fuzz-only"
        }}
    }))
    .expect("prove response");
    let structured = &response["result"]["structuredContent"];
    assert_eq!(
        structured["ok"], true,
        "a passing property reports ok:true: {structured}"
    );
    assert_eq!(structured["summary"]["total"], 1);
}

/// CR2-3 (clean obligation-only module): a module that defines NO
/// `property` binding must not be dragged to ok:false by the phantom
/// user-property probe. A clean opaque module with all obligations passing
/// stays ok:true (matches the brief's `clean module => ok:true`).
#[test]
fn cr2_3_obligation_only_clean_module_is_ok_through_tide() {
    let source = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
";
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":27,
        "method":"tools/call",
        "params":{"name":"chelis_prove","arguments":{
            "source_kind":"surf","source": source, "seed": 0
        }}
    }))
    .expect("prove response");
    let structured = &response["result"]["structuredContent"];
    assert_eq!(
        structured["ok"], true,
        "a clean obligation-only module (no `property`) is ok: {structured}"
    );
    // No phantom property record is emitted.
    assert!(
        structured["properties"].as_array().unwrap().is_empty(),
        "no user property => empty properties array: {structured}"
    );
}

/// CR-12: a type-broken module is not-ok through tide too (parity with the
/// CLI exit-3 / RT3-F2 contract).
#[test]
fn cr12_type_broken_module_is_not_ok_through_tide() {
    let source = "module Stats.Prob
export (bad_prob)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def bad_prob(x: f32) -> Probability = Probability { value: x }
def broken(x: f32) -> f32 = to_tensor([x])
";
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":12,
        "method":"tools/call",
        "params":{"name":"chelis_prove","arguments":{
            "source_kind":"surf","source": source
        }}
    }))
    .expect("prove response");
    let structured = &response["result"]["structuredContent"];
    assert_eq!(
        structured["ok"], false,
        "a type-broken module is not-ok through tide: {structured}"
    );
}

/// U4 parity: for a `.ch` module, tide's per-property verdict (name +
/// status) matches the shared `chelis_prove::property_runner` that the CLI
/// also drives, so a CLI prove and a tide prove agree on the same module.
#[test]
fn u4_tide_property_verdicts_match_shared_runner() {
    let source = "module M
def square(x: f32) -> f32 = x * x
@property square_non_negative forall(x: f32):
  (square(x) >= 0.0)
@property false_claim forall(y: f32):
  (y > y)
";
    // Tide path.
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":30,
        "method":"tools/call",
        "params":{"name":"chelis_prove","arguments":{
            "source_kind":"surf","source": source, "seed": 0
        }}
    }))
    .expect("prove response");
    let tide_props = response["result"]["structuredContent"]["properties"]
        .as_array()
        .expect("properties array")
        .clone();

    // Shared-runner path (the one the CLI also drives).
    let chelis_prove::property_runner::PropertyRunResult::Ran(engine) =
        chelis_prove::property_runner::run_surf_source_properties(
            source,
            &chelis_prove::property_runner::PropertyRunOptions::default(),
        )
        .expect("run");

    let tide_names: Vec<&str> = tide_props
        .iter()
        .map(|p| p["name"].as_str().unwrap())
        .collect();
    let engine_names: Vec<&str> = engine.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(
        tide_names, engine_names,
        "same property set across surfaces"
    );

    use chelis_prove::property_runner::PropertyStatus;
    for (tide, eng) in tide_props.iter().zip(&engine) {
        let expected = match eng.status {
            PropertyStatus::Passed => "passed",
            PropertyStatus::Failed => "failed",
            PropertyStatus::Unsupported => "unsupported",
            PropertyStatus::Error => "error",
        };
        assert_eq!(
            tide["status"].as_str().unwrap(),
            expected,
            "same status per property across surfaces"
        );
    }
}

/// U4: the tide tool handles a Deep (`.dp`) module -- discovering user
/// `@property` declarations from the Deep metadata and running them. The
/// fixture is the canonical Deep `chelis deep` emits for a Surf `@property`.
const DEEP_PROPERTY_MODULE: &str = r#"(module {}
  m
  (defsig {} always_true (t-fn {} (t-prim {} f32) (t-prim {} bool)))
  (def {chelis_role: "property",
         property_preconditions: (tuple {}),
         property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
         property_source_kind: "user"
       }
    always_true
    (fn {}
      (params {} (x {type: (t-prim {} f32)}))
      (app {}
        (var {} gte)
        (var {} x)
        (var {} x)))))
"#;

#[test]
fn u4_tide_handles_deep_module() {
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":31,
        "method":"tools/call",
        "params":{"name":"chelis_prove","arguments":{
            "source_kind":"deep","source": DEEP_PROPERTY_MODULE, "seed": 0
        }}
    }))
    .expect("prove response");
    let structured = &response["result"]["structuredContent"];
    let props = structured["properties"].as_array().expect("properties");
    assert_eq!(
        props.len(),
        1,
        "the deep user property is discovered: {structured}"
    );
    assert_eq!(props[0]["name"], "always_true");
    // x >= x is always true, so the property passes and the module is ok.
    assert_eq!(props[0]["status"], "passed");
    assert_eq!(
        structured["ok"], true,
        "clean deep module is ok: {structured}"
    );
}
