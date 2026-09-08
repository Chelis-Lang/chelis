// Raw-only attack fixtures. No helper can manufacture invalid typed metadata.
// The source-data role preserves the fixture before any annotation admission.
#[allow(dead_code)]
pub fn raw_metadata_fixture(source: &str) -> Vec<chelis_deep::RawExpr> {
    use chelis_deep::RawExpr;
    let wrapped = format!("(var {{source: (fixture {source}\n)}} fixture_owner)");
    let mut program = chelis_deep::parse_raw_str(&wrapped).expect("raw source fixture parses");
    let RawExpr::List(mut owner, _) = program.remove(0) else {
        panic!("fixture owner")
    };
    let RawExpr::Map(mut entries, _) = owner.remove(1) else {
        panic!("fixture metadata")
    };
    let RawExpr::List(mut invocation, _) = entries.remove(0).1 else {
        panic!("fixture source")
    };
    invocation.remove(0);
    invocation
}
