#[derive(Clone, Copy)]
pub struct NodeId(pub usize);

// Production verify/grad code observes the fat-pointer length of a shared
// slice of NodeId. Reading metadata must not consume the pointer or its loan.
pub fn length_and_first(nodes: &[NodeId]) -> (usize, Option<usize>) {
    let count = nodes.len();
    let first = if count == 0 { None } else { Some(nodes[0].0) };
    (count, first)
}

pub fn subslice_lengths(nodes: &[NodeId], split: usize) -> (usize, usize, usize) {
    let prefix = &nodes[..split];
    let suffix = &nodes[split..];
    (prefix.len(), suffix.len(), nodes.len())
}

pub fn node_value(node: &NodeId) -> usize {
    node.0
}

// A higher-ranked function item has no captured borrow of its own. Each call
// borrows the current node without making that argument part of its storage.
pub fn function_item_map(nodes: &[NodeId]) -> usize {
    let mut signature = 0;
    for value in nodes.iter().map(node_value) {
        signature = signature * 10 + value;
    }
    signature
}

// Literal storage has static duration even on a branch which returns early.
// It must not become a stack-local loan which frame cleanup ends.
pub fn reduction_name(kind: u8) -> &'static str {
    match kind {
        0 => "reduce_window_max",
        1 => "reduce_window_min",
        _ => "reduce_window_sum",
    }
}

#[inline(never)]
pub fn selected_node(nodes: &[NodeId], index: usize) -> Option<&NodeId> {
    nodes.get(index)
}

// The function-call abstraction contains a shared input, an ignored scalar
// input, and a structured optional output. All slots must survive projection
// when the loop fixed point is matched back against a concrete iteration.
pub fn repeat_selected(nodes: &[NodeId], index: usize, mut repeats: usize) -> usize {
    match selected_node(nodes, index) {
        None => 0,
        Some(node) => {
            let mut total = 0;
            while repeats > 0 {
                total += node.0;
                repeats -= 1;
            }
            total
        }
    }
}

#[inline(never)]
pub fn read_nested(value: &&usize) -> usize {
    **value
}

pub fn nested_shared_branches(value: &usize, mut repeats: usize) -> usize {
    let alias = value;
    let mut total = 0;
    while repeats > 0 {
        if repeats % 2 == 0 {
            // Only this branch creates the outer loan. Its end must leave
            // the inner shared borrower alive for the next iteration.
            total += read_nested(&alias);
        } else {
            total += *alias;
        }
        repeats -= 1;
    }
    total + *value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_reads_preserve_the_borrow_and_observe_the_view() {
        let nodes = [NodeId(7), NodeId(11), NodeId(19)];
        assert_eq!(length_and_first(&nodes), (3, Some(7)));
        assert_eq!(length_and_first(&nodes[1..]), (2, Some(11)));
        assert_eq!(length_and_first(&nodes[3..]), (0, None));
        assert_eq!(subslice_lengths(&nodes, 1), (1, 2, 3));
        assert_eq!(subslice_lengths(&nodes, 0), (0, 3, 3));
        assert_eq!(subslice_lengths(&nodes, 3), (3, 0, 3));
        assert_eq!(function_item_map(&nodes), 829);
        assert_eq!(function_item_map(&[]), 0);
    }

    #[test]
    fn literals_outlive_all_return_branches() {
        let names = [reduction_name(0), reduction_name(1), reduction_name(2)];
        assert_eq!(names, ["reduce_window_max", "reduce_window_min", "reduce_window_sum"]);
    }

    #[test]
    fn call_input_slots_survive_loop_projection() {
        let nodes = [NodeId(7), NodeId(11)];
        assert_eq!(repeat_selected(&nodes, 1, 3), 33);
        assert_eq!(repeat_selected(&nodes, 0, 0), 0);
        assert_eq!(repeat_selected(&nodes, 2, 3), 0);
    }

    #[test]
    fn ending_outer_shared_loans_preserves_inner_aliases() {
        assert_eq!(nested_shared_branches(&7, 0), 7);
        assert_eq!(nested_shared_branches(&7, 4), 35);
    }
}
