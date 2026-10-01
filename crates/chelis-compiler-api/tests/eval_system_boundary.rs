//! Exercise the real evaluator adapter through the public compiler API. The
//! policy/refusal controls live in runtime::system_tests and invariant_decode;
//! this covers the consumer-visible values after permitted host operations.

use std::collections::BTreeMap;
use std::fs;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, EvalResult, ExecutionValue, SourceKind};

fn root<'a>(result: &'a EvalResult, name: &str) -> &'a ExecutionValue {
    &result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing {name} in {:?}", result.roots))
        .value
}

fn strings(value: &ExecutionValue) -> Vec<&str> {
    let ExecutionValue::List { value } = value else {
        panic!("expected a list, got {value:?}");
    };
    value
        .iter()
        .map(|item| match item {
            ExecutionValue::String { value } => value.as_str(),
            other => panic!("expected string item, got {other:?}"),
        })
        .collect()
}

fn integers(value: &ExecutionValue) -> Vec<i64> {
    let ExecutionValue::List { value } = value else {
        panic!("expected a list, got {value:?}");
    };
    value
        .iter()
        .map(|item| match item {
            ExecutionValue::Scalar { value } => value.get().as_i64_exact().expect("integer"),
            other => panic!("expected integer item, got {other:?}"),
        })
        .collect()
}

#[test]
fn permitted_program_preserves_file_directory_mapping_and_process_semantics() {
    let dir = tempfile::tempdir().expect("tempdir");
    let listed = dir.path().join("listed");
    fs::create_dir(&listed).expect("directory");
    // Reverse creation order proves the listing is ordered on host-name bytes.
    fs::write(listed.join("zeta"), b"z").expect("fixture");
    fs::write(listed.join("alpha"), b"a").expect("fixture");
    let file = listed.join("payload.txt");
    let missing = listed.join("absent.txt");
    let content = "alpha\r\nbeta\n";
    let source = format!(
        "written = write_file({file:?}, \"alpha\\r\\nbeta\\n\")\n\
         text = read_file({file:?})\n\
         lines = read_lines({file:?})\n\
         bytes = read_bytes({file:?})\n\
         exists = file_exists({file:?})\n\
         missing = file_exists({missing:?})\n\
         names = list_dir({listed:?})\n\
         mapped_len = mmap_len(mmap_file({file:?}))\n\
         mapped_first = mmap_read(mmap_file({file:?}), 0i64, 5i64)\n\
         process = process_run(\"echo\", [\"safe\", \"$HOME\"])\n"
    );
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source,
        bindings: BTreeMap::new(),
    })
    .expect("all eight covered operations execute under normal evaluation");

    assert_eq!(
        fs::read_to_string(&file).expect("file was written"),
        content
    );
    assert!(matches!(root(&result, "written"), ExecutionValue::Unit));
    assert!(matches!(root(&result, "text"), ExecutionValue::String { value } if value == content));
    assert_eq!(strings(root(&result, "lines")), ["alpha", "beta"]);
    assert_eq!(
        integers(root(&result, "bytes")),
        content.bytes().map(i64::from).collect::<Vec<_>>()
    );
    assert!(matches!(
        root(&result, "exists"),
        ExecutionValue::Bool { value: true }
    ));
    assert!(matches!(
        root(&result, "missing"),
        ExecutionValue::Bool { value: false }
    ));
    assert_eq!(
        strings(root(&result, "names")),
        ["alpha", "payload.txt", "zeta"]
    );
    assert!(
        matches!(root(&result, "mapped_len"), ExecutionValue::Scalar { value } if value.get().as_i64_exact() == Some(content.len() as i64))
    );
    assert_eq!(
        integers(root(&result, "mapped_first")),
        b"alpha"
            .iter()
            .map(|byte| i64::from(*byte))
            .collect::<Vec<_>>()
    );
    assert!(
        matches!(root(&result, "process.0"), ExecutionValue::Scalar { value } if value.get().as_i64_exact() == Some(0))
    );
    assert!(
        matches!(root(&result, "process.1"), ExecutionValue::String { value } if value == "safe $HOME\n")
    );
    assert!(
        matches!(root(&result, "process.2"), ExecutionValue::String { value } if value.is_empty())
    );
}

fn integer(value: &ExecutionValue) -> i64 {
    match value {
        ExecutionValue::Scalar { value } => value.get().as_i64_exact().expect("integer"),
        other => panic!("expected an integer, got {other:?}"),
    }
}

fn host_now() -> (i64, i64) {
    let since = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("the host clock is after 1970");
    (
        i64::try_from(since.as_secs()).expect("fits"),
        i64::from(since.subsec_nanos()),
    )
}

/// [05-OP-75] through the public API: each clock read is `(seconds,
/// nanoseconds)` with nanoseconds in `[0, 10^9)`, the wall reading lies
/// between host reads taken around the evaluation, and successive monotonic
/// reads never decrease.
#[test]
fn permitted_program_reads_both_host_clocks() {
    let lower = host_now();
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: "wall = clock_wall_read()\n\
                 first = clock_monotonic_read()\n\
                 second = clock_monotonic_read()\n"
            .to_owned(),
        bindings: BTreeMap::new(),
    })
    .expect("both clock reads execute under normal evaluation");
    let upper = host_now();
    let reading = |name: &str| {
        (
            integer(root(&result, &format!("{name}.0"))),
            integer(root(&result, &format!("{name}.1"))),
        )
    };
    let wall = reading("wall");
    assert!(
        lower <= wall && wall <= upper,
        "{lower:?} <= {wall:?} <= {upper:?}"
    );
    let (first, second) = (reading("first"), reading("second"));
    assert!(first <= second, "{first:?} then {second:?}");
    for (_, nanoseconds) in [wall, first, second] {
        assert!((0..1_000_000_000).contains(&nanoseconds));
    }
}
