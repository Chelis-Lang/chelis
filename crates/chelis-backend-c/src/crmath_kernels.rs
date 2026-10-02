//! The correctly rounded kernels a generated translation unit carries
//! (spec/design/correctly_rounded_math.md section 4.2).
//!
//! Emission names a transcendental only through its `chelis_cr_*` entry
//! ([05-OP-46]); this pass then defines exactly the entries the unit calls by
//! inserting their kernel text from `chelis-crmath`, the same bytes the
//! evaluator compiles. Every kernel definition is `static`, so a built static
//! library exports no new symbol and no published header gains a declaration.

use chelis_crmath::c_source::{Kernel, kernel_text};

/// The kernels whose entries `source` names, in amalgamation order.
fn called_kernels(source: &str) -> Vec<Kernel> {
    let mut called = Vec::new();
    for word in source.split(|c: char| !(c == '_' || c.is_ascii_alphanumeric())) {
        if let Some(kernel) = Kernel::from_entry(word)
            && !called.contains(&kernel)
        {
            called.push(kernel);
        }
    }
    called.sort();
    called
}

/// Insert the definitions of every kernel `source` calls after the unit's
/// runtime include, which every generated unit opens with. A unit that calls
/// no kernel is returned unchanged.
pub(crate) fn link_called_kernels(source: String) -> String {
    let called = called_kernels(&source);
    if called.is_empty() {
        return source;
    }
    let text = kernel_text(&called);
    const RUNTIME_INCLUDE: &str = "#include \"chelis_runtime.h\"\n";
    let at = source
        .find(RUNTIME_INCLUDE)
        .map_or(0, |at| at + RUNTIME_INCLUDE.len());
    let mut linked = String::with_capacity(source.len() + text.len() + 1);
    linked.push_str(&source[..at]);
    linked.push_str(&text);
    linked.push('\n');
    linked.push_str(&source[at..]);
    linked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn links_only_called_kernels_once() {
        let source = "#include \"chelis_runtime.h\"\nfloat f(float x) { return chelis_cr_expf(chelis_cr_expf(x)) + (float)chelis_cr_tanh((double)x); }\n".to_string();
        let linked = link_called_kernels(source);
        assert_eq!(
            linked
                .matches("static float chelis_cr_expf(float x)")
                .count(),
            1
        );
        assert_eq!(
            linked
                .matches("static double chelis_cr_tanh(double x)")
                .count(),
            1
        );
        assert!(!linked.contains("chelis_cr_exp(double"));
        assert!(!linked.contains("chelis_cr_tanhf(float"));
        assert!(linked.starts_with("#include \"chelis_runtime.h\"\n"));
        assert!(linked.find("static float chelis_cr_expf(") < linked.find("float f(float x)"));
    }

    #[test]
    fn unit_without_kernel_calls_is_unchanged() {
        let source = "#include \"chelis_runtime.h\"\nfloat f(float x) { return sqrtf(x); }\n";
        assert_eq!(link_called_kernels(source.to_string()), source);
    }

    #[test]
    fn identifiers_that_only_contain_an_entry_name_do_not_link() {
        let source =
            "#include \"chelis_runtime.h\"\nint my_chelis_cr_expf_count; int chelis_cr_expf2;\n";
        assert_eq!(link_called_kernels(source.to_string()), source);
    }
}
