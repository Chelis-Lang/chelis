//! `Vec::retain` with closures capturing mutable borrows (Aeneas.Std Vec.lean
//! `retain_with`, ModelStateABI closure_state_models): the closure's final
//! state gives the captured borrows back to the caller. The first shapes are
//! chelis-types `Subst::take_operand_gates_on` (unify.rs) and
//! `InferenceProduct::cancel_shape_checks_for_failed_family_call` (infer/checked.rs,
//! whose inner `Iter::any` closure reborrows the captured substitution).

#[derive(Clone, Debug, PartialEq)]
pub struct DeferredOperandGate {
    pub operand: String,
    pub dims: Vec<u32>,
}

pub struct Subst {
    pub deferred_tensor_operands: Vec<(u32, DeferredOperandGate)>,
    pub generation: u32,
}

impl Subst {
    /// As `Subst::take_operand_gates_on`: take every constraint suspended on
    /// `v`, leaving the rest of the ledger in place.
    pub fn take_operand_gates_on(&mut self, v: u32) -> Vec<DeferredOperandGate> {
        let ledger = &mut self.deferred_tensor_operands;
        let mut taken = Vec::new();
        ledger.retain(|(tv, gate)| {
            if *tv == v {
                taken.push(gate.clone());
                false
            } else {
                true
            }
        });
        self.generation += 1;
        taken
    }
}

/// A deferred shape check over argument types (as chelis-types
/// `DeferredShapeCheck`).
pub struct ShapeCheck {
    pub arg_tys: Vec<u32>,
}

impl Subst {
    /// The free variables of a resolved type (`&mut self`, as `Subst::apply`
    /// then `free_tvars`), counting the resolutions in `generation`.
    pub fn apply(&mut self, v: u32) -> Vec<u32> {
        self.generation += 1;
        if v >= 10 {
            vec![v / 10, v % 10]
        } else {
            vec![v]
        }
    }
}

/// As chelis-types `InferenceProduct::cancel_shape_checks_for_failed_family_call`:
/// the `retain` closure captures the substitution mutably and the failed
/// parameters by shared borrow; its inner `any` closure (over the slice
/// iterator) reborrows the substitution, the innermost one reads `failed`.
pub fn cancel_shape_checks(checks: &mut Vec<ShapeCheck>, subst: &mut Subst, failed: &[u32]) {
    checks.retain(|check| {
        !check
            .arg_tys
            .iter()
            .any(|argument| subst.apply(*argument).iter().any(|variable| failed.contains(variable)))
    });
}

/// Two captured counters, read after `retain`.
pub fn drop_odds(values: &mut Vec<u32>) -> (u32, u32) {
    let mut calls = 0u32;
    let mut dropped = 0u32;
    values.retain(|x| {
        calls += 1;
        if *x % 2 == 0 {
            true
        } else {
            dropped += 1;
            false
        }
    });
    (calls, dropped)
}

/// A captured vector read and extended by the closure: keeps the first
/// occurrence of every value and returns the seen values.
pub fn first_occurrences(values: &mut Vec<u32>) -> Vec<u32> {
    let mut seen: Vec<u32> = Vec::new();
    values.retain(|x| {
        if seen.contains(x) {
            false
        } else {
            seen.push(*x);
            true
        }
    });
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(operand: &str, dims: &[u32]) -> DeferredOperandGate {
        DeferredOperandGate { operand: operand.to_string(), dims: dims.to_vec() }
    }

    #[test]
    fn take_operand_gates_on_takes_the_suspended_gates() {
        let mut subst = Subst {
            deferred_tensor_operands: vec![
                (1, gate("x", &[2])),
                (2, gate("y", &[])),
                (1, gate("z", &[3, 4])),
                (3, gate("w", &[5])),
            ],
            generation: 7,
        };
        assert_eq!(subst.take_operand_gates_on(1), vec![gate("x", &[2]), gate("z", &[3, 4])]);
        assert_eq!(subst.deferred_tensor_operands, vec![(2, gate("y", &[])), (3, gate("w", &[5]))]);
        assert_eq!(subst.generation, 8);
        assert_eq!(subst.take_operand_gates_on(9), vec![]);
        assert_eq!(subst.deferred_tensor_operands, vec![(2, gate("y", &[])), (3, gate("w", &[5]))]);
        assert_eq!(subst.generation, 9);
    }

    #[test]
    fn cancel_shape_checks_drops_the_failed_and_counts_the_resolutions() {
        let mut checks: Vec<ShapeCheck> = [vec![1, 2], vec![35], vec![4], vec![], vec![12, 7]]
            .into_iter()
            .map(|arg_tys| ShapeCheck { arg_tys })
            .collect();
        let mut subst = Subst { deferred_tensor_operands: vec![(1, gate("x", &[2]))], generation: 7 };
        cancel_shape_checks(&mut checks, &mut subst, &[3, 7]);
        let kept: Vec<Vec<u32>> = checks.iter().map(|c| c.arg_tys.clone()).collect();
        assert_eq!(kept, vec![vec![1, 2], vec![4], vec![]]);
        assert_eq!(subst.generation, 13);
        assert_eq!(subst.deferred_tensor_operands, vec![(1, gate("x", &[2]))]);
        cancel_shape_checks(&mut checks, &mut subst, &[]);
        assert_eq!(checks.len(), 3);
        assert_eq!(subst.generation, 16);
    }

    #[test]
    fn drop_odds_counts_the_calls_and_the_dropped() {
        let mut values = vec![1, 2, 3, 4, 6];
        assert_eq!(drop_odds(&mut values), (5, 2));
        assert_eq!(values, vec![2, 4, 6]);
        let mut empty = vec![];
        assert_eq!(drop_odds(&mut empty), (0, 0));
        assert_eq!(empty, Vec::<u32>::new());
    }

    #[test]
    fn first_occurrences_keeps_the_first_of_each_value() {
        let mut values = vec![3, 1, 3, 2, 1, 4];
        assert_eq!(first_occurrences(&mut values), vec![3, 1, 2, 4]);
        assert_eq!(values, vec![3, 1, 2, 4]);
    }
}
