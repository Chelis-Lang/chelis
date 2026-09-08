//! Public compiler-API device rejection for first-class Count (chelis#1287).

use chelis_compiler_api::compiler::compile;
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};

const COUNT_SOURCE: &str = include_str!("../../../examples/count_bool_axes.ch");

#[test]
fn hip_compile_api_rejects_host_helper_count_with_issue_1291_receipt() {
    let err = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: COUNT_SOURCE.to_string(),
        target: CompileTarget::Hip,
        entry_name: None,
    })
    .expect_err("HIP must not compile Count through the C-host fallback");
    let rendered = format!("{err:?}");
    assert!(
        rendered.contains("unimplemented chelis#1291:"),
        "{rendered}"
    );
    assert!(
        rendered.contains("count") && rendered.contains("--target c"),
        "{rendered}"
    );
}
