//! chelis#3418: the host evaluator allocates an output-sized buffer only
//! through `chelis_ir::eval::admit_tensor`'s `AdmittedResult`, which admits
//! the result's metadata and reserves fallibly, so an unrepresentable result
//! traps and an ungranted one fails as the C runtime's allocation failure.
//!
//! A type cannot stop a `Vec` from being sized by a bare count, so this scan
//! is the detector: in the host runtime's production source, every
//! `Vec::with_capacity(..)` and `vec![value; count]` is sized by an existing
//! value's `.len()`, or carries a `// bounded:` line directly above it naming
//! the input that bounds it. Anything else is a result-sized allocation that
//! bypassed admission.

use std::path::Path;

const SCANNED: [&str; 2] = ["src/runtime/host_ops.rs", "src/runtime/eval.rs"];

/// Every unjustified sized allocation in `source`, as `line: text`.
/// `#[cfg(test)]` items are skipped: test fixtures size by literal counts.
fn unadmitted_allocations(source: &str) -> Vec<String> {
    let lines = source.lines().collect::<Vec<_>>();
    let mut found = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index].trim();
        if line == "#[cfg(test)]" {
            index = skip_item(&lines, index + 1);
            continue;
        }
        if let Some(argument) = sized_allocation_argument(line) {
            let justified = index > 0 && lines[index - 1].trim().starts_with("// bounded: ");
            if !argument.contains(".len()") && !justified {
                found.push(format!("{}: {line}", index + 1));
            }
        }
        index += 1;
    }
    found
}

/// The size argument of a `Vec::with_capacity(..)` or `vec![value; size]`
/// on `line`, if any.
fn sized_allocation_argument(line: &str) -> Option<&str> {
    if let Some(start) = line.find("Vec::with_capacity(") {
        let rest = &line[start + "Vec::with_capacity(".len()..];
        return Some(rest);
    }
    let start = line.find("vec![")?;
    let rest = &line[start + "vec![".len()..];
    let end = rest.find(']')?;
    let body = &rest[..end];
    body.rfind(';').map(|semicolon| &body[semicolon + 1..])
}

/// Skip the item that starts at `start` (its attributes, header, and
/// balanced braces or terminating semicolon), returning the next line index.
fn skip_item(lines: &[&str], start: usize) -> usize {
    let mut depth = 0_i64;
    let mut opened = false;
    for (offset, line) in lines[start..].iter().enumerate() {
        for character in line.chars() {
            match character {
                '{' => {
                    depth += 1;
                    opened = true;
                }
                '}' => depth -= 1,
                _ => {}
            }
        }
        if (opened && depth == 0) || (!opened && line.trim_end().ends_with(';')) {
            return start + offset + 1;
        }
    }
    lines.len()
}

#[test]
fn host_runtime_sizes_results_only_through_admission() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut found = Vec::new();
    for file in SCANNED {
        let source = std::fs::read_to_string(root.join(file)).expect("read host runtime source");
        found.extend(
            unadmitted_allocations(&source)
                .into_iter()
                .map(|site| format!("{file}:{site}")),
        );
    }
    assert!(
        found.is_empty(),
        "result-sized allocations that bypass `admit_tensor` (size them from an \
         `AdmittedResult`, or justify an input-bounded size with `// bounded: <input>` \
         on the line above):\n{}",
        found.join("\n")
    );
}

/// Negative control: the scan reports an unadmitted count, in both spellings,
/// and accepts the admitted, `.len()`-sized, justified, and test-only forms.
#[test]
fn the_scan_reports_a_bypass_and_accepts_the_sanctioned_forms() {
    let source = "\
fn op() {
    let mut picks = Vec::with_capacity(out_numel);
    let mut map = vec![None; out_numel];
    let mut ok = admitted.buffer()?;
    let mut by_input = Vec::with_capacity(tensor.value.len());
    // bounded: the input rank
    let mut axes = vec![0usize; rank];
    let literal = vec![1, 2, 3];
}
#[cfg(test)]
mod tests {
    fn fixture() {
        let data = vec![0.0; n];
    }
}
";
    assert_eq!(
        unadmitted_allocations(source),
        [
            "2: let mut picks = Vec::with_capacity(out_numel);",
            "3: let mut map = vec![None; out_numel];",
        ]
    );
}
