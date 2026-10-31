//! Intern exact private kernel implementations, retaining site adapters.
use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use crate::emit::SHARED_HELPER_TEMPLATE_NAME as TEMPLATE_NAME;

#[derive(Default)]
pub(super) struct HelperPool {
    buckets: BTreeMap<[u8; 32], Vec<usize>>,
    bodies: Vec<Body>,
}

struct Body {
    code: Vec<String>,
    spans: BTreeMap<usize, Annotations>,
}

struct Annotations {
    canonical: String,
    merged: BTreeSet<String>,
}

fn declaration(name: &str) -> String {
    format!(
        "static __attribute__((noinline)) void {name}({}, const char *__chelis_helper_site) {{",
        super::private_host_params(
            "chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out"
        )
    )
}

impl Body {
    /// Consume only the CEmitter's private template format. Span lines are
    /// emitted by emit_span_comments; producer strings cannot introduce lines.
    /// No literals, identifiers, claims, or arbitrary comments are normalized.
    fn new(lines: Vec<String>) -> Self {
        let mut body = Self {
            code: Vec::new(),
            spans: BTreeMap::new(),
        };
        for line in lines {
            if line.trim_start().starts_with("// span: ") {
                match body.spans.entry(body.code.len()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(Annotations {
                            canonical: line,
                            merged: BTreeSet::new(),
                        });
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        if entry.get().canonical != line {
                            entry.get_mut().merged.insert(line);
                        }
                    }
                }
            } else {
                body.code.push(line);
            }
        }
        let template = declaration(TEMPLATE_NAME);
        assert_eq!(
            body.code.iter().filter(|line| **line == template).count(),
            1,
            "a shared helper template has exactly one owned declaration"
        );
        body
    }

    fn merge_spans(&mut self, other: BTreeMap<usize, Annotations>) {
        for (position, spans) in other {
            if let Some(existing) = self.spans.get_mut(&position) {
                existing.merged.insert(spans.canonical);
                existing.merged.extend(spans.merged);
                existing.merged.remove(&existing.canonical);
            } else {
                self.spans.insert(position, spans);
            }
        }
    }
}

impl HelperPool {
    pub(super) fn append_adapter(&mut self, out: &mut Vec<String>, lines: Vec<String>, site: &str) {
        let body = Body::new(lines);
        let mut digest = Sha256::new();
        for line in &body.code {
            digest.update(line.as_bytes());
            digest.update(b"\n");
        }
        let digest: [u8; 32] = digest.finalize().into();
        let index = self.intern(body, digest);
        out.push(format!(
            "static void {site}({}) {{",
            super::private_host_params(
                "chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out"
            )
        ));
        out.push(format!("    __chelis_tensor_impl_{index}(inputs, n_in, outputs, n_out, __chelis_caller_result_claims, {});", super::c_string_literal(site)));
        out.push("}".to_string());
        out.push(String::new());
    }

    fn intern(&mut self, body: Body, digest: [u8; 32]) -> usize {
        let candidates = self.buckets.entry(digest).or_default();
        if let Some(index) = candidates
            .iter()
            .copied()
            .find(|index| self.bodies[*index].code == body.code)
        {
            self.bodies[index].merge_spans(body.spans);
            index
        } else {
            let index = self.bodies.len();
            self.bodies.push(body);
            candidates.push(index);
            index
        }
    }

    pub(super) fn render(self) -> Vec<String> {
        let template = declaration(TEMPLATE_NAME);
        let mut out = Vec::new();
        for (index, body) in self.bodies.into_iter().enumerate() {
            let mut spans = body.spans;
            for (position, line) in body.code.into_iter().enumerate() {
                if let Some(annotations) = spans.remove(&position) {
                    out.push(annotations.canonical);
                    out.extend(annotations.merged);
                }
                out.push(if line == template {
                    declaration(&format!("__chelis_tensor_impl_{index}"))
                } else {
                    line
                });
            }
            assert!(
                spans.is_empty(),
                "source annotations precede a generated statement"
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_dag_core::axis_sources::EntryExtentGuard;
    use chelis_dag_core::{Dag, DimInfo, NodeId, RiscOp, TensorType};
    use chelis_numeric_core::prim::Prim;

    #[test]
    fn hash_collisions_do_not_merge_code_and_annotations_follow_their_statement() {
        let make = |span: &str, code: &str| {
            Body::new(vec![
                declaration(TEMPLATE_NAME),
                format!("    // span: {span}"),
                code.into(),
                "}".into(),
            ])
        };
        let mut pool = HelperPool::default();
        assert_eq!(pool.intern(make("canonical", "    first();"), [0; 32]), 0);
        assert_eq!(pool.intern(make("other", "    second();"), [0; 32]), 1);
        assert_eq!(pool.intern(make("additional", "    first();"), [0; 32]), 0);
        let source = pool.render().join("\n");
        assert!(source.contains("// span: canonical\n    // span: additional\n    first();"));
        assert!(source.contains("// span: other\n    second();"));
    }

    #[test]
    fn discharged_and_retained_extent_checks_are_distinct() {
        let mut dag = Dag::new();
        let owner = dag.declare("checked");
        let ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            owner,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty.clone(),
            None,
        );
        let result = dag.add_node(owner, RiscOp::Neg, vec![x], ty, None);
        dag.add_root(result);
        let options = crate::CodegenOptions {
            static_entry: true,
            use_blas: false,
        };
        let verified = crate::testing::verified_dag(&dag, options).unwrap();
        let guard = EntryExtentGuard::Literal {
            required: 4,
            observed: (NodeId(0), 0),
        };
        let mut pool = HelperPool::default();
        let mut adapters = Vec::new();
        for (name, guards) in [("retained", vec![]), ("covered", vec![guard])] {
            let source = crate::emit::CEmitter::emit_verified_dag_with_options(
                verified.emission(),
                name,
                options,
                &guards,
                true,
            )
            .unwrap();
            pool.append_adapter(
                &mut adapters,
                source.lines().map(str::to_string).collect(),
                name,
            );
        }
        assert_eq!(pool.bodies.len(), 2);
    }
}
