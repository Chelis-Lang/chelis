use std::fmt::Write as _;

use super::ir::{Edge, Op, Operand, OwnerId, OwnershipProgram, Terminal, Terminator, Unit};

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
                let _ = writeln!(out, "    {}", render_op(unit, &op.kind));
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
    let terminals = value
        .terminals
        .iter()
        .map(|terminal| match terminal.kind {
            Terminal::Drop(owner) => format!("drop %{}", owner.0),
            Terminal::Discard(owner) => format!("discard %{}", owner.0),
        })
        .collect::<Vec<_>>()
        .join(", ");
    let edge = format!(
        "b{} [{}]",
        value.target.0,
        value
            .args
            .iter()
            .map(operand)
            .collect::<Vec<_>>()
            .join(", ")
    );
    if terminals.is_empty() {
        edge
    } else {
        format!("{edge} terminals [{terminals}]")
    }
}

/// `%3` for an owner with no source names, `%3[total,acc]` when lowering kept
/// the binding names it came from. Shared with the verifier so a diagnostic
/// and a dump spell the same owner the same way.
pub(crate) fn owner_label(unit: &Unit, id: OwnerId) -> String {
    let names = unit.owners[&id].names.join(",");
    if names.is_empty() {
        format!("%{}", id.0)
    } else {
        format!("%{}[{names}]", id.0)
    }
}

fn owner(unit: &Unit, id: OwnerId) -> String {
    owner_label(unit, id)
}

fn render_op(unit: &Unit, op: &Op) -> String {
    match op {
        Op::Define { dest, label } => format!("{} = {label}", owner(unit, *dest)),
        Op::Apply {
            dest, label, args, ..
        } => {
            let dest = dest.map_or_else(String::new, |id| format!("%{} = ", id.0));
            format!(
                "{dest}{label}({})",
                args.iter().map(operand).collect::<Vec<_>>().join(", ")
            )
        }
        Op::Copy { dest, source } => format!("%{} = copy {}", dest.0, operand(source)),
        Op::Project { source } => format!("project {}", operand(source)),
        Op::LoopItem { dest, list } => {
            format!("%{} = loop-item {}", dest.0, operand(list))
        }
        Op::Drop { owner } => format!("drop {}", operand(owner)),
        Op::Discard { owner } => format!("discard %{}", owner.0),
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
        Terminator::Match { scrutinee, arms } => format!(
            "match {} [{}]",
            operand(scrutinee),
            arms.iter().map(edge).collect::<Vec<_>>().join(", ")
        ),
        Terminator::Loop {
            list,
            body_edge,
            exit_edge,
        } => format!(
            "loop {} body {} exit {}",
            operand(list),
            edge(body_edge),
            edge(exit_edge)
        ),
        Terminator::Exit => "exit".to_string(),
    }
}
