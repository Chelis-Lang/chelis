use std::fmt::Write as _;

use super::ir::{Edge, Op, Operand, OwnerId, OwnershipProgram, Terminator, Unit};

pub(crate) fn render(program: &OwnershipProgram) -> String {
    let mut out = String::new();
    for unit in &program.units {
        let _ = writeln!(out, "unit {} {:?}", unit.name, unit.kind);
        for block in &unit.blocks {
            let params = block
                .params
                .iter()
                .map(|p| format!("{:?} %{}", p.mode, p.owner.0))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "  b{} ({params}):", block.id.0);
            for op in &block.ops {
                let _ = writeln!(out, "    {}", render_op(unit, op));
            }
            let _ = writeln!(out, "    {}", render_terminator(&block.terminator));
        }
    }
    out
}

fn operand(value: &Operand) -> String {
    format!("{} %{}", value.use_.name(), value.owner.0)
}

fn edge(value: &Edge) -> String {
    format!(
        "b{} [{}]",
        value.target.0,
        value
            .args
            .iter()
            .map(operand)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn owner(unit: &Unit, id: OwnerId) -> String {
    let names = unit.owners[&id].names.join(",");
    if names.is_empty() {
        format!("%{}", id.0)
    } else {
        format!("%{}[{names}]", id.0)
    }
}

fn render_op(unit: &Unit, op: &Op) -> String {
    match op {
        Op::Define { dest, label } => format!("{} = {label}", owner(unit, *dest)),
        Op::Apply { dest, label, args } => {
            let dest = dest.map_or_else(String::new, |id| format!("%{} = ", id.0));
            format!(
                "{dest}{label}({})",
                args.iter().map(operand).collect::<Vec<_>>().join(", ")
            )
        }
        Op::Copy { dest, source } => format!("%{} = copy {}", dest.0, operand(source)),
        Op::Drop { owner } => format!("drop {}", operand(owner)),
        Op::RootConsume { root, owner } => format!("root {root} {}", operand(owner)),
    }
}

fn render_terminator(terminator: &Terminator) -> String {
    match terminator {
        Terminator::Return { result } => format!("return {}", operand(result)),
        Terminator::Jump(to) => format!("jump {}", edge(to)),
        Terminator::Branch {
            condition,
            then_edge,
            else_edge,
        } => format!(
            "branch {} then {} else {}",
            operand(condition),
            edge(then_edge),
            edge(else_edge)
        ),
        Terminator::Exit => "exit".to_string(),
    }
}
