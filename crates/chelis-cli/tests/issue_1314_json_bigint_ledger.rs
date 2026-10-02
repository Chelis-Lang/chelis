//! chelis#1314's JSON ordering scratch under the ownership ledger.
//!
//! The `ownership-ledger` feature instruments the runtime this `chelis` build
//! carries, so the runtime `chelis build` stages records the ledger. The target
//! requires the feature.

use assert_cmd::Command;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

/// Output equality cannot prove scratch cleanup: returning on the last iteration
/// preserves the serialized value but leaks the guard and its owner. Link the
/// runtime `chelis build` staged, which records the ledger in this build, and
/// run the emitted helper under a fully released caller and output-preserving
/// cleanup mutations.
#[test]
fn json_scratch_execution_detects_skipped_cleanup() {
    use std::fs;
    use std::process::Command as Process;
    let (_dir, reef_home, app) = make_app("json-scratch-ownership");
    write_file(
        &app.join("src/main.ch"),
        r#"module Demo.Main
import Std.Io.Json (parse_json, to_json)
empty_object = to_json(parse_json("{}"))
ordered = to_json(parse_json("{\"b\":2,\"a\":1}"))
"#,
    );
    let out = app.join("out");
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app)
        .args([
            "build",
            "--emit-c",
            "src/main.ch",
            "--target",
            "c",
            "--output",
        ])
        .arg(&out)
        .assert()
        .success();
    let generated = fs::read_to_string(out.join("main.c")).unwrap();
    assert_eq!(generated.matches("int main(void) {").count(), 1);
    // Keep the actual emitted helper, and give it a caller whose ownership is
    // explicit. General host lowering is tested separately; this receipt owns
    // only JSON ordering scratch and the values allocated by this driver.
    let driver = r#"
int main(void) {
    for (int count = 0; count <= 2; count += 2) {
        chelis_list *pairs = chelis_list_with_capacity(count);
        for (int index = 0; index < count; ++index) {
            chelis_value fields[2] = {
                chelis_value_take_string(chelis_string_from_cstr(index == 0 ? "b" : "a")),
                chelis_value_take_string(chelis_string_from_cstr("value"))
            };
            chelis_value pair = chelis_value_take_tuple(chelis_tuple_from_values(fields, 2));
            chelis_list_push_moved(pairs, pair);
            chelis_value_release(fields[0]);
            chelis_value_release(fields[1]);
        }
        chelis_dict *dict = chelis_dict_from_pairs(pairs);
        chelis_list_release(pairs);
        chelis_list *ordered = chelis_json_canonical_object_entries(dict);
        if (chelis_list_len(ordered) != count) return 9;
        printf("%d:", count);
        for (int index = 0; index < count; ++index) {
            chelis_value item = chelis_list_index(ordered, index);
            chelis_value key = chelis_tuple_get(chelis_tuple_borrow_value(item), 0);
            printf("%s", chelis_string_data(chelis_string_borrow_value(key)));
            chelis_value_release(key);
            chelis_value_release(item);
        }
        printf("\n");
        chelis_list_release(ordered);
        chelis_dict_release(dict);
    }
    return 0;
}
"#;
    let emitted = generated.replace("int main(void) {", "int json_fixture_main(void) {") + driver;
    let anchor = "        chelis_list_push_moved(result, entry);";
    assert_eq!(
        emitted.matches(anchor).count(),
        1,
        "exact output loop anchor"
    );
    let variants = [
        ("balanced", emitted.clone(), true),
        (
            "last-return",
            emitted.replace(
                anchor,
                &format!("{anchor} if (index == len - 1) return result;"),
            ),
            false,
        ),
        (
            "missing-end",
            emitted.replace("    chelis_tensor_end_write(order_guard);", ""),
            false,
        ),
        (
            "missing-release",
            emitted.replace("    chelis_tensor_release(order_storage);", ""),
            false,
        ),
    ];
    let mut expected_stdout = None;
    for (name, source, balanced) in variants {
        if !balanced {
            assert_ne!(source, emitted, "mutation must apply: {name}");
        }
        fs::write(out.join("main.c"), source).unwrap();
        assert!(common::link_generated(&out, "main.c", name).success());
        let ledger = out.join(format!("{name}.jsonl"));
        let result = Process::new(out.join(name))
            .env("CHELIS_OWNERSHIP_LEDGER_PATH", &ledger)
            .output()
            .unwrap();
        if name == "missing-end" {
            assert_eq!(result.status.code(), Some(1));
            assert_eq!(
                String::from_utf8(result.stderr).unwrap(),
                "Domain: chelis_tensor_release: tensor has an active write guard\n"
            );
            continue;
        }
        assert!(
            result.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let stdout = String::from_utf8(result.stdout).unwrap();
        assert_eq!(stdout, "0:\n2:ab\n", "{name}");
        if let Some(expected) = &expected_stdout {
            assert_eq!(&stdout, expected, "{name}");
        } else {
            expected_stdout = Some(stdout);
        }
        // Require the real ledger header, actual scratch allocation, and exactly
        // one terminal summary. An uninstrumented archive cannot pass silently.
        let rows: Vec<serde_json::Value> = fs::read_to_string(&ledger)
            .expect("runtime ledger required")
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(rows[0]["schema"], "compiled-value-ownership-ledger-v1");
        assert!(
            rows.iter().any(|r| r["event"] == "allocate"
                && r["kind"] == "TensorStorage"
                && r["bytes"] == 16)
        );
        let summaries: Vec<_> = rows.iter().filter(|r| r["event"] == "summary").collect();
        assert_eq!(summaries.len(), 1);
        let summary = summaries[0];
        assert_eq!(Some(summary), rows.last());
        assert_eq!(summary["invalid_operations"], 0, "{name}: {summary}");
        let owners = summary["live_owners"].as_u64().unwrap();
        let bytes = summary["live_bytes"].as_u64().unwrap();
        if balanced {
            assert_eq!((owners, bytes), (0, 0), "{name}: {summary}");
        } else {
            assert!(
                owners > 0 && bytes >= 16,
                "missed skipped cleanup in {name}: {summary}"
            );
        }
    }
}
