pub struct Node {
    pub values: [u32; 2],
}

pub fn retained_iterator_payload(nodes: &[Node]) -> u32 {
    let mut total = 0;
    for node in nodes {
        let selected = &node.values;
        for value in selected {
            total += *value;
        }
        total += selected[0];
    }
    total
}

pub fn repeated_optional(shared: &Option<&u32>, rounds: usize) -> u32 {
    let mut total = 0;
    let mut i = 0;
    while i < rounds {
        match shared {
            Some(value) => total += **value,
            None => total += 1,
        }
        i += 1;
    }
    total
}

pub fn selected_result(nodes: &[Node], index: usize, rounds: usize) -> Result<u32, u32> {
    let node = nodes.get(index).ok_or(9u32)?;
    let values = &node.values;
    let mut total = 0;
    let mut i = 0;
    while i < rounds {
        total += values[i % 2];
        i += 1;
    }
    Ok(total)
}

pub fn nested_shared_choice(left: &Option<&u32>, right: &Option<&u32>, choose_left: bool) -> u32 {
    let selected = if choose_left { left } else { right };
    let nested = &selected;
    let result = match **nested {
        Some(value) => *value,
        None => 1,
    };
    // Both original payloads remain usable after joining the shared copies.
    result + *left.unwrap_or(&2) + *right.unwrap_or(&3)
}

pub fn shared_choice_after_copies(
    left: &Option<&u32>,
    right: &Option<&u32>,
    choose_left: bool,
) -> u32 {
    let selected = if choose_left { left } else { right };
    let left_value = *left.unwrap_or(&2);
    let right_value = *right.unwrap_or(&3);
    // Copying the original payloads must not end the selected shared view.
    let selected_value = match *selected {
        Some(value) => *value,
        None => 1,
    };
    left_value + right_value + selected_value
}

pub fn simultaneous_shared_views(values: Vec<&u32>) -> u32 {
    let mut total = values.len() as u32;
    for value in values.iter() {
        total += **value;
    }
    total
}

pub fn selected_iterator_sum(left: &[u32], right: &[u32], choose_left: bool) -> u32 {
    let mut left = left.iter();
    let mut right = right.iter();
    let selected = if choose_left { &mut left } else { &mut right };
    let mut total = 0;
    while let Some(value) = selected.next() {
        total += *value;
    }
    total
}

#[cfg(test)]
#[test]
fn selected_mutable_iterator_keeps_its_shared_slice() {
    assert_eq!(selected_iterator_sum(&[7, 11], &[3, 5], true), 18);
    assert_eq!(selected_iterator_sum(&[7, 11], &[3, 5], false), 8);
    assert_eq!(selected_iterator_sum(&[], &[3, 5], true), 0);
}

static REGISTERED_VALUES: &[u32] = &[3, 7, 11];

pub fn static_shared_member(number: u32) -> u32 {
    let mut index = 0;
    while index < REGISTERED_VALUES.len() {
        if number == REGISTERED_VALUES[index] {
            return number;
        }
        index += 1;
    }
    0
}

pub struct Counter {
    pub value: u32,
}

impl Counter {
    pub fn add(&mut self, amount: u32) {
        self.value += amount;
    }
}

pub fn reserved_receiver_across_branch(choose_left: bool) -> u32 {
    let mut counter = Counter { value: 10 };
    counter.add(if choose_left { 2 } else { 5 });
    counter.value
}

pub fn borrowed_owner_join(values: &[u32]) -> (u32, u32) {
    let owners = values.iter().collect::<Vec<_>>();
    let mut failures = 0;
    if let [owner] = owners.as_slice()
        && **owner != 0
    {
        failures += 1;
    }
    let mut sum = 0;
    for owner in owners {
        sum += *owner;
    }
    (failures, sum)
}

pub enum ObservationSite {
    Caller { axis: i32 },
    Claim { axis: i32 },
    Other,
}

pub struct ObservationNode {
    pub id: u32,
    pub site: ObservationSite,
    pub inputs: Vec<u32>,
    pub shape_deps: Vec<u32>,
}

pub fn count_missing_observations(nodes: &[ObservationNode]) -> u32 {
    let mut missing = 0;
    for node in nodes {
        if let ObservationSite::Claim { axis } = &node.site {
            let declared = node
                .shape_deps
                .as_slice()
                .first()
                .and_then(|required| nodes.get(*required as usize));
            let same = node.shape_deps.len() == 1 && declared.is_some_and(|declared| {
                matches!(declared.site, ObservationSite::Caller { axis: observed } if observed == *axis)
                    && declared.inputs.first() == node.inputs.first()
                    && declared.id < node.id
            });
            if !same {
                missing += 1;
            }
        }
    }
    missing
}

#[cfg(test)]
#[test]
fn short_circuit_call_and_drop_keep_shared_observations() {
    let caller = ObservationNode {
        id: 0,
        site: ObservationSite::Caller { axis: 2 },
        inputs: vec![7],
        shape_deps: vec![],
    };
    let claim = |id, axis, inputs, shape_deps| ObservationNode {
        id,
        site: ObservationSite::Claim { axis },
        inputs,
        shape_deps,
    };
    let nodes = [
        caller,
        claim(1, 2, vec![7], vec![0]),
        claim(2, 9, vec![7], vec![0]),
        claim(3, 2, vec![7], vec![0, 0]),
        claim(4, 2, vec![7], vec![]),
        claim(5, 2, vec![7], vec![9]),
        claim(6, 2, vec![8], vec![0]),
    ];
    assert_eq!(count_missing_observations(&nodes), 5);
}

#[cfg(test)]
#[test]
fn retained_shared_payloads_survive_loop_joins() {
    let nodes = [Node { values: [7, 11] }, Node { values: [3, 5] }];
    assert_eq!(retained_iterator_payload(&nodes), 36);
    assert_eq!(retained_iterator_payload(&[]), 0);
    assert_eq!(repeated_optional(&Some(&7), 4), 28);
    assert_eq!(repeated_optional(&None, 4), 4);
    assert_eq!(repeated_optional(&Some(&7), 0), 0);
    assert_eq!(selected_result(&nodes, 0, 3), Ok(25));
    assert_eq!(selected_result(&nodes, 1, 0), Ok(0));
    assert_eq!(selected_result(&nodes, 2, 3), Err(9));
    assert_eq!(nested_shared_choice(&Some(&7), &Some(&11), true), 25);
    assert_eq!(nested_shared_choice(&Some(&7), &Some(&11), false), 29);
    assert_eq!(nested_shared_choice(&Some(&7), &None, false), 11);
    assert_eq!(static_shared_member(11), 11);
    assert_eq!(static_shared_member(5), 0);
    assert_eq!(reserved_receiver_across_branch(true), 12);
    assert_eq!(reserved_receiver_across_branch(false), 15);
    assert_eq!(shared_choice_after_copies(&Some(&7), &Some(&11), true), 25);
    assert_eq!(shared_choice_after_copies(&Some(&7), &None, false), 11);
    assert_eq!(simultaneous_shared_views(vec![&3, &8]), 13);
    assert_eq!(simultaneous_shared_views(Vec::new()), 0);
}

#[cfg(test)]
#[test]
fn borrowed_vector_payloads_survive_conditional_lending() {
    assert_eq!(borrowed_owner_join(&[]), (0, 0));
    assert_eq!(borrowed_owner_join(&[0]), (0, 0));
    assert_eq!(borrowed_owner_join(&[7]), (1, 7));
    assert_eq!(borrowed_owner_join(&[7, 3]), (0, 10));
}

pub enum SharedTree {
    Ignore,
    Leaf(u32),
    Add(Box<SharedTree>, Box<SharedTree>),
    Multiply(Box<SharedTree>, Box<SharedTree>),
}

pub fn collect_shared_tree_values(tree: &SharedTree, values: &mut Vec<u32>) {
    match tree {
        SharedTree::Ignore => {}
        SharedTree::Leaf(value) => values.push(*value),
        SharedTree::Add(left, right) | SharedTree::Multiply(left, right) => {
            collect_shared_tree_values(left, values);
            collect_shared_tree_values(right, values);
        }
    }
}

#[cfg(test)]
#[test]
fn recursive_shared_arms_keep_both_borrowed_children() {
    let tree = SharedTree::Add(
        Box::new(SharedTree::Leaf(3)),
        Box::new(SharedTree::Multiply(
            Box::new(SharedTree::Leaf(5)),
            Box::new(SharedTree::Ignore),
        )),
    );
    let mut values = vec![99];
    collect_shared_tree_values(&tree, &mut values);
    assert_eq!(values, vec![99, 3, 5]);
}
